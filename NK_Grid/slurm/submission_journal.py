"""Durable, standard-library journal around the existing dynamic sbatch chain."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid


def _now():
    return datetime.now(timezone.utc).isoformat()


def _read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"submission receipt is not a JSON object: {path}")
    return value


def _write(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _lock(path):
    # Persistent inode; Windows uses its real byte-range lock for portable
    # launcher tests, while the production POSIX path uses flock.
    with Path(path).open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            if handle.seek(0, os.SEEK_END) == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _describe(path, receipt):
    accepted = ", ".join(f"{job['label']}={job['slurm_job_id']}" for job in receipt.get("jobs", [])) or "none"
    pending = receipt.get("pending_submission")
    detail = "none" if pending is None else json.dumps(pending, ensure_ascii=False)
    return (f"Submission receipt: {path}\nAccepted jobs: {accepted}\n"
            f"Unconfirmed submission: {detail}\n"
            "Inspect the scheduler and this receipt before any retry. An unconfirmed "
            "sbatch response does not prove that no job was accepted.")


def _identity(plan, digest):
    return str(plan.get("execution_plan_id") or digest)


def _journal_root(plan_path, plan):
    snapshot = Path(plan["snapshot"])
    if not snapshot.is_absolute():
        snapshot = plan_path.parent / snapshot
    # Old dry-run test fixtures may refer to an unmounted snapshot. Real plans
    # bind every alias/copy of an execution to the same durable output root.
    if not snapshot.is_file():
        return plan_path.parent
    output = Path(_read(snapshot)["output_dir"])
    if not output.is_absolute():
        output = snapshot.parent / output
    if not output.is_dir():
        raise ValueError(f"dynamic output directory does not exist: {output}")
    return output.resolve()


def guard(plan_path, command):
    plan_path = Path(plan_path).resolve()
    plan = _read(plan_path)
    digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    identity = _identity(plan, digest)
    lock_name = hashlib.sha256(identity.encode()).hexdigest()[:24]
    journal_root = _journal_root(plan_path, plan)
    index = journal_root / f".submission-{lock_name}.json"
    with _lock(journal_root / f".submission-{lock_name}.lock"):
        archive = journal_root / "checkpoint-archive.json"
        if archive.exists() or archive.is_symlink():
            raise ValueError("run is terminally archived; final CSV is retained; do not submit another training chain")
        previous_paths = set(plan_path.parent.glob("*.submission-receipt-*.json"))
        if index.exists():
            previous_paths.add(Path(_read(index)["receipt"]))
        for previous in previous_paths:
            receipt = _read(previous)
            safely_unsubmitted = (
                receipt.get("status") == "failed_before_submission"
                and receipt.get("jobs") == []
                and "pending_submission" in receipt and receipt["pending_submission"] is None
            )
            if (receipt.get("submission_identity") == identity
                    and receipt.get("status") != "complete"
                    and not safely_unsubmitted):
                raise ValueError("Refusing another chain while a prior submission is incomplete or unknown.\n"
                                 + _describe(previous, receipt))
        submission = plan["submission"]
        receipt = {
            "journal_format_version": 1, "status": "submitting", "created_at_utc": _now(),
            "plan": str(plan_path), "plan_sha256": digest, "submission_identity": identity,
            "snapshot": plan["snapshot"], "execution_plan_id": plan.get("execution_plan_id"),
            "sbatch_account": submission["account"], "sbatch_constraint": submission["constraint"],
            "resources": {phase: {"sbatch_args": plan[phase]["sbatch_args"], "tmp_dir": plan[phase].get("tmp_dir")}
                          for phase in ("preparation", "verification", "finalization")},
            "expected_jobs": 3 * int(plan["rounds"]) + 2,
            "jobs": [], "pending_submission": None,
        }
        path = plan_path.with_name(plan_path.stem + ".submission-receipt-" + uuid.uuid4().hex + ".json")
        _write(path, receipt)
        _write(index, {"submission_identity": identity, "receipt": str(path)})
        environment = {**os.environ, "NKGRID_SUBMISSION_RECEIPT": str(path)}
        try:
            result = subprocess.run(command, env=environment, check=False)
            receipt = _read(path)
            complete = (result.returncode == 0 and receipt.get("pending_submission") is None
                        and len(receipt["jobs"]) == receipt["expected_jobs"])
            receipt["status"] = ("complete" if complete else "unknown" if (receipt.get("pending_submission") or result.returncode < 0)
                                 else "partial" if receipt["jobs"] else "failed_before_submission")
            receipt["submitter_exit_code"] = result.returncode
            receipt["updated_at_utc"] = _now()
            _write(path, receipt)
            if not complete:
                print(_describe(path, receipt), file=sys.stderr)
            return result.returncode if result.returncode != 0 else (0 if complete else 2)
        except BaseException:
            # The pre-submission intent remains durable even if the child or
            # the journal update fails. Never clear it based on an exception.
            print(_describe(path, _read(path)), file=sys.stderr)
            raise


def submit(path, plan_path, label, generation, arguments):
    path = Path(path)
    receipt = _read(path)
    plan_path = Path(plan_path).resolve()
    plan = _read(plan_path)
    digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    if (receipt.get("plan") != str(plan_path) or receipt.get("plan_sha256") != digest
            or receipt.get("submission_identity") != _identity(plan, digest)):
        raise ValueError("inherited submission receipt does not match this frozen plan")
    if receipt.get("status") != "submitting" or receipt.get("pending_submission") is not None:
        raise ValueError("submission receipt is not ready for a new sbatch request")
    dependency = next((value.split("=", 1)[1] for value in arguments if value.startswith("--dependency=")), "none")
    pending = {"label": label, "dependency": dependency, "generation": generation,
               "command": ["sbatch", "--parsable", *arguments], "started_at_utc": _now()}
    receipt["pending_submission"] = pending
    _write(path, receipt)
    try:
        result = subprocess.run(pending["command"], capture_output=True, text=True, check=False)
    except OSError as exc:
        receipt["pending_submission"]["error"] = f"{type(exc).__name__}: {exc}"
        receipt["status"] = "unknown"
        _write(path, receipt)
        raise
    raw_id = result.stdout.strip()
    if result.returncode != 0 or re.fullmatch(r"[1-9][0-9]*(?:;[A-Za-z0-9_.-]+)?", raw_id) is None:
        pending.update(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)
        receipt["status"] = "unknown"
        _write(path, receipt)
        raise ValueError(f"sbatch did not return an unambiguous accepted job ID for {label}; " + _describe(path, receipt))
    job_id = raw_id.split(";", 1)[0]
    receipt["jobs"].append({**pending, "slurm_job_id": job_id, "sbatch_response": raw_id,
                            "cluster": raw_id.split(";", 1)[1] if ";" in raw_id else None,
                            "accepted_at_utc": _now()})
    receipt["pending_submission"] = None
    try:
        _write(path, receipt)
    except OSError:
        print(f"Scheduler returned job {raw_id} for {label}, but persisting acceptance failed; "
              f"the durable pending intent remains at {path}.", file=sys.stderr)
        raise
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    print(job_id)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    guarding = subparsers.add_parser("guard")
    guarding.add_argument("--plan", required=True)
    guarding.add_argument("command", nargs=argparse.REMAINDER)
    submitting = subparsers.add_parser("submit")
    submitting.add_argument("--receipt", required=True)
    submitting.add_argument("--plan", required=True)
    submitting.add_argument("--label", required=True)
    submitting.add_argument("--generation", required=True)
    submitting.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required")
    if args.action == "guard":
        return guard(args.plan, command)
    return submit(args.receipt, args.plan, args.label, args.generation, command)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f"NKGRID submission: {exc}", file=sys.stderr)
        raise SystemExit(2)
