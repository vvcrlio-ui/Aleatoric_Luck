"""Cluster-resident, bounded, exactly identified Discoverer continuation.

One controller closes/verifies the previous array, prepares the next immutable
generation, and submits only that array. Before any worker submission it arms a
small recovery controller; a normal successor depends on afterany of the whole
array. Every sbatch intent precedes the call and uncertain responses are matched
against squeue AND accounting by an immutable unique job name, never resubmitted.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

import experiment as common
import discoverer_resources as resources

sys.path.insert(0, str(common.ROOT / "NK_Grid/slurm"))
from submission_journal import _lock, _write

TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}
DONE = {"complete", "round_budget_exhausted", "no_progress", "unrecoverable", "no_executable_tasks", "control_budget_exhausted", "quota_insufficient"}


def now():
    return datetime.now(timezone.utc).isoformat()


def submit_bootstrap(request, *, slurm=None):
    """Initial accepted-but-response-lost recovery, without creating another run."""
    from discoverer import bootstrap_command
    request = Path(request).resolve()
    spec = read(request)
    common.validate_source(spec)
    path = request.parent / "bootstrap-journal.json"
    with _lock(request.parent / ".bootstrap.lock"):
        digest = common.sha256(request)
        if path.exists():
            state = read(path)
            if state["request_sha256"] != digest:
                raise ValueError("Bootstrap request identity changed")
        else:
            state = {"run_id": uuid.uuid4().hex, "jobs": {}, "request_sha256": digest,
                     "request": str(request), "created_at_utc": now()}
        journal = Journal(path, state, slurm or Slurm(spec["cluster"]["account"], spec["cluster"]["qos"]))
        arguments = [a for a in bootstrap_command(spec, request)[2:] if not a.startswith("--job-name=")]
        return journal.submit("B0", arguments)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@contextmanager
def controller_lock(path, *, timeout=240):
    """A normal successor must not die while a guard briefly owns the lock."""
    deadline = time.monotonic() + timeout
    while True:
        manager = _lock(path)
        try:
            manager.__enter__()
            break
        except OSError as exc:
            import errno
            if exc.errno not in (errno.EACCES, errno.EAGAIN) or time.monotonic() >= deadline:
                raise
            time.sleep(1)
    try:
        yield
    finally:
        manager.__exit__(None, None, None)


class AmbiguousSubmission(RuntimeError):
    pass


class Slurm:
    def __init__(self, account, qos, user=None):
        import getpass
        self.account, self.qos, self.user = account, qos, user or getpass.getuser()

    def submit(self, command):
        from discoverer import batch_environment
        return subprocess.run(command, capture_output=True, text=True, check=False,
                              timeout=120, env=batch_environment())

    def find(self, name, since):
        """Return unique root IDs even when an array has thousands of rows."""
        fields = "id,name,state,account,user,qos"
        queued = resources.records(resources.query(["squeue", "-h", "-r", "-u", self.user,
                                  "--name=" + name, "-o", "%i|%j|%T|%a|%u|%q"]), fields)
        historic = resources.records(resources.query(["sacct", "-nP", "-X", "-u", self.user,
                                     "--starttime=" + since[:19], "--name=" + name,
                                     "--format=JobID%80,JobName%80,State,Account,User,QOS"]), fields)
        result = {}
        for row in historic + queued:
            if (row["name"] != name or row["account"] != self.account
                    or row["user"] != self.user or row["qos"] != self.qos):
                continue
            root = re.split(r"[_.]", row["id"])[0]
            if root.isdigit():
                result[root] = row["state"].split()[0].rstrip("+")
        return result

    def states(self, job_id):
        try:
            queued = resources.query(["squeue", "-h", "-r", "-j", str(job_id), "-o", "%T"])
        except subprocess.CalledProcessError as exc:
            # Slurm versions may exit 1 instead of returning an empty queue
            # once an old job leaves slurmctld. Other query errors stay fatal.
            if "invalid job id" not in str(exc.stderr).lower():
                raise
            queued = ""
        if queued.strip():
            return [s.strip() for s in queued.splitlines()]
        historic = resources.query(["sacct", "-nP", "-X", "-j", str(job_id), "--format=State"])
        states = [s.split("|")[0].split()[0].rstrip("+") for s in historic.splitlines() if s.strip()]
        if not states:
            raise AmbiguousSubmission(f"Slurm has no queue/accounting evidence for job {job_id}")
        return states


class Journal:
    """Called only while the per-run persistent OS lock is held."""
    def __init__(self, path, state, slurm):
        self.path, self.state, self.slurm = Path(path), state, slurm

    def save(self):
        self.state["updated_at_utc"] = now()
        _write(self.path, self.state)

    def submit(self, label, arguments):
        jobs = self.state["jobs"]
        entry = jobs.get(label)
        if entry and entry.get("job_id"):
            return entry["job_id"]
        if entry is None:
            if label.startswith(("C", "G")) and "limits" in self.state:
                controls = sum(key.startswith(("C", "G")) for key in jobs)
                if controls >= self.state["limits"]["max_control_jobs"]:
                    raise ValueError("Maximum control submission budget exhausted")
            name = "al-" + self.state["run_id"][:16] + "-" + label
            if len(name) > 64:
                raise ValueError("Internal job label is too long")
            command = ["sbatch", "--parsable", "--job-name=" + name,
                       "--comment=nkgrid:" + self.state["run_id"], *arguments]
            entry = {"name": name, "command": command, "intent_at_utc": now(), "status": "intent"}
            jobs[label] = entry
            self.save()
            try:
                result = self.slurm.submit(command)
                raw = result.stdout.strip()
                entry.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
                if result.returncode == 0 and re.fullmatch(r"[1-9][0-9]*(?:;[A-Za-z0-9_.-]+)?", raw):
                    entry.update(job_id=raw.split(";", 1)[0], status="accepted", accepted_at_utc=now())
                    self.save()
                    return entry["job_id"]
            except (OSError, subprocess.TimeoutExpired) as exc:
                entry["error"] = repr(exc)
            entry["status"] = "uncertain"
            self.save()
        # A previously persisted intent is never sent a second time, even if
        # queue/accounting visibility is delayed or sbatch exited nonzero.
        matches = self.slurm.find(entry["name"], entry["intent_at_utc"])
        if len(matches) != 1:
            raise AmbiguousSubmission(f"{label}: uncertain submission has {len(matches)} matching jobs; inspect {self.path}")
        entry.update(job_id=next(iter(matches)), status="recovered", recovered_at_utc=now())
        self.save()
        return entry["job_id"]


def control_args(spec, state_path, *, dependency=None, delayed=False):
    cluster, output = spec["cluster"], Path(spec["output"])
    args = ["--account=" + cluster["account"], "--qos=" + cluster["qos"], "--partition=" + cluster["partition"],
            "--nodes=1", "--ntasks-per-node=1", "--ntasks-per-core=1", "--cpus-per-task=1",
            "--mem=" + spec["plan_memory"], "--time=" + spec["plan_time"],
            "--chdir=" + str(output), "--output=" + str(output / "logs/control-%j.out"),
            "--error=" + str(output / "logs/control-%j.err"), "--export=ALL"]
    if dependency:
        args.append("--dependency=afterany:" + str(dependency))
    if delayed:
        args.append("--begin=now+10minutes")
    return args + [str(common.ROOT / "launch/discoverer_control.sbatch"), str(state_path)]


def start(spec, plan_path, *, slurm=None):
    """Bootstrap/resume hands ownership to one durable cluster controller."""
    output, plan_path = Path(spec["output"]), Path(plan_path).resolve()
    state_path = output / "continuation.json"
    with _lock(output / ".continuation.lock"):
        if state_path.exists():
            state = read(state_path)
            if state["plan_sha256"] != common.sha256(plan_path):
                raise ValueError("Continuation state does not match the immutable plan")
            if state["status"] in DONE:
                raise ValueError("Continuation is terminal: " + state["status"])
        else:
            plan = read(plan_path)
            if (Path(read(plan["snapshot"])["output_dir"]) / "active-generation.json").exists():
                raise ValueError("Cannot adopt a legacy or unjournaled active generation")
            state = {"format_version": 1, "run_id": uuid.uuid4().hex, "created_at_utc": now(),
                     "plan": str(plan_path), "plan_sha256": common.sha256(plan_path), "spec": spec,
                     "limits": spec["continuation"], "status": "ready", "jobs": {}, "rounds": [],
                     "control_attempts": 0, "control_failures": 0, "recovery_attempts": 0, "no_progress_rounds": 0,
                     "completed_model_keys": 0, "history": []}
        scheduler = slurm or Slurm(spec["cluster"]["account"], spec["cluster"]["qos"])
        journal = Journal(state_path, state, scheduler)
        journal.save()
        # Resume is idempotent while any previously submitted controller exists.
        for label, job in state["jobs"].items():
            if not label.startswith(("C", "G")):
                continue
            job_id = journal.submit(label, [])
            if any(s not in TERMINAL for s in scheduler.states(job_id)):
                print(f"Continuation already armed: {job_id}; state {state_path}", flush=True)
                return job_id
        label = "C" + str(len([k for k in state["jobs"] if k.startswith("C")]))
        job_id = journal.submit(label, control_args(spec, state_path, dependency=os.environ.get("SLURM_JOB_ID")))
        print(f"Continuation controller: {job_id}; state {state_path}", flush=True)
        return job_id


def frozen_round(plan, directory, *, workers, time_limit):
    """Resource-only execution contract; analysis, table and config stay exact."""
    from aleatoric_nk_grid.execution_contract import AnalysisContract, DynamicExecutionContract, immutable_json_bytes
    snapshot = read(plan["snapshot"])
    analysis = AnalysisContract.from_payload(read(snapshot["analysis_contract"]))
    previous = DynamicExecutionContract.from_payload(read(snapshot["execution_contract"]))
    payload = deepcopy(dict(previous.payload))
    payload["worker_count"] = int(workers)
    for key, args in payload["resources"].items():
        if key == "worker":
            payload["resources"][key] = ["--time=" + time_limit if a.startswith("--time=") else a for a in args]
    contract = DynamicExecutionContract(payload)
    if contract.payload["analysis_id"] != analysis.analysis_id:
        raise ValueError("Resource adjustment changed analysis identity")
    contract_path = Path(snapshot["output_dir"]) / "execution-contracts" / (contract.execution_plan_id + ".json")
    immutable_json_bytes(contract_path, contract.to_payload())
    snapshot.update(workers=workers, execution_plan_id=contract.execution_plan_id,
                    execution_contract_sha256=contract.sha256, execution_contract=str(contract_path))
    directory.mkdir(parents=True, exist_ok=True)
    snapshot_path = directory / "snapshot.json"
    immutable_json_bytes(snapshot_path, snapshot)
    return snapshot_path, contract.execution_plan_id, payload["resources"]["worker"]


def target(round_state):
    return {"round_index": round_state["round"], "submission_generation": round_state["generation"],
            "expected_prep_token": round_state["prep_job_id"], "prep_job_id": round_state["prep_job_id"],
            "expected_pointer_version": round_state["pointer_version"],
            "expected_previous_generation": round_state["previous_generation"],
            "expected_previous_execution_plan_id": round_state["previous_execution_plan_id"],
            "expected_previous_round_index": round_state["previous_round"]}


class Engine:
    def prepare(self, round_state):
        from aleatoric_nk_grid.flat_task_table import prepare_round, _exact_target, _load_snapshot, _recover_generation_activation_locked
        from aleatoric_nk_grid.generation_control import schedule_transaction, generation_dir, intent_path, outcome_path
        kwargs = target(round_state)
        snapshot = Path(round_state["snapshot"])
        # The underlying engine has exact-target activation recovery. It cannot
        # replay a partial immutable preparation by calling prepare blindly.
        if round_state.get("prepare_started"):
            payload = _load_snapshot(snapshot)
            root = Path(payload["output_dir"])
            with schedule_transaction(root):
                exact_args = {**kwargs, "prep_token": kwargs["expected_prep_token"]}
                exact_args.pop("expected_prep_token")
                _, _, exact = _exact_target(payload, **exact_args)
                directory = generation_dir(root, exact)
                staging = directory.parent / ("." + directory.name + ".staging")
                if not any(p.exists() for p in (directory, staging, intent_path(root, exact), outcome_path(root, exact))):
                    # Death before the engine's first intent is safe to retry;
                    # prepare's exact predecessor gate still owns admission.
                    fresh = {**kwargs, "prep_token": kwargs["expected_prep_token"]}
                    fresh.pop("expected_prep_token")
                    return prepare_round(snapshot, **fresh, _schedule_locked=True)
                _recover_generation_activation_locked(snapshot, root=root, target=exact, **kwargs)
                return {"recovered": True}
        kwargs["prep_token"] = kwargs.pop("expected_prep_token")
        return prepare_round(snapshot, **kwargs)

    def close_verify(self, round_state):
        from aleatoric_nk_grid.flat_task_table import close_generation, verify_rounds
        close_generation(Path(round_state["snapshot"]), **target(round_state))
        return verify_rounds(Path(round_state["snapshot"]), **target(round_state))

    def finalize(self, round_state):
        from aleatoric_nk_grid.flat_task_table import finalize_snapshot
        return finalize_snapshot(Path(round_state["snapshot"]), **target(round_state))


def worker_args(round_state, current_job):
    values = [round_state["snapshot"], str(round_state["round"]), round_state["prep_job_id"],
              round_state["generation"], round_state["previous_generation"] or "", str(round_state["pointer_version"]),
              round_state["previous_execution_plan_id"] or "", str(round_state["previous_round"] or "")]
    workers = round_state["resources"]["workers"]
    return [*round_state["worker_sbatch_args"], "--dependency=afterany:" + current_job,
            f"--array=0-{workers - 1}%{workers}", "--export=ALL", "--chdir=" + str(Path(round_state["snapshot"]).parents[2]),
            str(common.ROOT / "NK_Grid/slurm/run_flat_task_table.sbatch"), *values]


def drive(journal, *, current_job, engine=None, live_query=None, freeze=frozen_round):
    """One bounded controller activation. Injectable boundaries support fault tests."""
    state, slurm = journal.state, journal.slurm
    spec, limits = state["spec"], state["limits"]
    engine = engine or Engine()
    if state["status"] in DONE:
        return
    # A delayed recovery guard exits when a live normal successor is wired.
    for label, job in list(state["jobs"].items()):
        if job.get("job_id") == current_job or not label.startswith("C"):
            continue
        other_id = journal.submit(label, [])
        if any(s not in TERMINAL for s in slurm.states(other_id)):
            return
    state["control_attempts"] += 1
    if state["control_attempts"] > limits["max_control_jobs"]:
        state["status"] = "control_budget_exhausted"; journal.save(); return
    # Before numerical/control work, arm failure recovery after this controller
    # ends. The guard is delayed to avoid racing a normal afterany successor.
    guard_label = "G" + str(state["control_attempts"])
    journal.submit(guard_label, control_args(spec, journal.path, dependency=current_job, delayed=True))
    plan = read(state["plan"])
    remaining = int(plan["row_count"])
    latest = state["rounds"][-1] if state["rounds"] else None
    if latest and latest.get("worker_job_id"):
        statuses = slurm.states(latest["worker_job_id"])
        if any(s not in TERMINAL for s in statuses):
            label = "C" + str(len([k for k in state["jobs"] if k.startswith("C")]))
            journal.submit(label, control_args(spec, journal.path, dependency=latest["worker_job_id"]))
            state["control_failures"] = 0
            state["status"] = "workers_active"; journal.save(); return
        latest["worker_terminal_states"] = {status: statuses.count(status) for status in sorted(set(statuses))}
        journal.save()
        if not latest.get("verification"):
            verification = engine.close_verify(latest)
            latest["verification"] = verification
            previous_count = state["completed_model_keys"]
            state["completed_model_keys"] = int(verification["completed_model_keys"])
            state["no_progress_rounds"] = state["no_progress_rounds"] + 1 if state["completed_model_keys"] <= previous_count else 0
            latest["sealed_at_utc"] = now()
            journal.save()
        verification = latest["verification"]
        if any(s in {"OUT_OF_MEMORY", "BOOT_FAIL", "FAILED"} for s in statuses):
            # Missing WAL cannot certify a healthy model run: dependency/import
            # failures and OOM can occur before TASK_STARTED is durable.
            state["status"] = "unrecoverable"; journal.save(); return
        if verification["complete"]:
            # finalize's own immutable receipt/archival protocol handles repeat
            # triggers. Save publication phase before entering it.
            state["status"] = "finalizing"; journal.save()
            state["finalization"] = engine.finalize(latest)
            state["status"] = "complete"; journal.save(); return
        if verification["aborted_tasks"] or verification["failed_attempt_row_ids"]:
            state["status"] = "unrecoverable"; journal.save(); return
        remaining = int(verification["executable_task_rows"])
        if remaining == 0:
            state["status"] = "no_executable_tasks"; journal.save(); return
        if state["no_progress_rounds"] >= limits["max_no_progress_rounds"]:
            state["status"] = "no_progress"; journal.save(); return
        if len(state["rounds"]) >= limits["max_rounds"]:
            state["status"] = "round_budget_exhausted"; journal.save(); return
        latest = None
    if latest is None:
        cluster = spec["cluster"]
        live = (live_query or resources.snapshot)(cluster["account"], cluster["qos"], cluster["partition"])
        # The recovery guard is already included in squeue. Only the normal
        # successor remains to submit; one auxiliary can overlap workers.
        decision = resources.capacity(live, remaining=remaining, memory=cluster["memory_override"],
                                      requested_time=cluster["time_limit"], worker_cap=limits["worker_cap"],
                                      extra_submit=1, extra_running=0)
        number = len(state["rounds"]) + 1
        snapshot_path, execution_id, args = freeze(plan, Path(spec["output"]) / "rounds" / str(number),
                                                  workers=decision["workers"], time_limit=decision["time_limit"])
        previous = state["rounds"][-1] if state["rounds"] else None
        latest = {"round": number, "snapshot": str(snapshot_path), "execution_plan_id": execution_id,
                  "generation": uuid.uuid4().hex, "prep_job_id": current_job, "resources": decision,
                  "worker_sbatch_args": args, "previous_generation": previous["generation"] if previous else None,
                  "previous_execution_plan_id": previous["execution_plan_id"] if previous else None,
                  "previous_round": previous["round"] if previous else None,
                  "pointer_version": previous["pointer_version"] + 1 if previous else 0}
        state["rounds"].append(latest); state["status"] = "preparing"; journal.save()
    if not latest.get("prepared"):
        # Persist intent but pass the preexisting value to distinguish first
        # preparation from exact-target recovery after process termination.
        recovery = bool(latest.get("prepare_started"))
        latest["prepare_started"] = True; journal.save()
        call_state = {**latest, "prepare_started": recovery}
        latest["preparation"] = engine.prepare(call_state)
        latest["prepared"] = True; journal.save()
        if latest["preparation"].get("no_generation"):
            raise ValueError("No executable tasks appeared during preparation; inspect verification")
    worker_label = "W" + str(latest["round"])
    if worker_label not in state["jobs"]:
        # Preparation may take minutes. Recheck admission immediately before
        # sbatch; never mutate a prepared assignment to squeeze a changed cap.
        cluster = spec["cluster"]
        refreshed = (live_query or resources.snapshot)(cluster["account"], cluster["qos"], cluster["partition"])
        available = resources.capacity(refreshed, remaining=latest["resources"]["workers"],
                                       memory=cluster["memory_override"], requested_time=latest["resources"]["time_limit"],
                                       extra_submit=1, extra_running=0)
        if (available["workers"] < latest["resources"]["workers"]
                or resources.duration(available["time_limit"]) < resources.duration(latest["resources"]["time_limit"])):
            state["status"] = "quota_insufficient"
            state["history"].append({"at_utc": now(), "resource_change": available})
            journal.save(); return
        latest["submission_resource_check"] = available; journal.save()
    job_id = journal.submit(worker_label, worker_args(latest, current_job))
    latest["worker_job_id"] = job_id; state["status"] = "workers_active"; journal.save()
    label = "C" + str(len([k for k in state["jobs"] if k.startswith("C")]))
    journal.submit(label, control_args(spec, journal.path, dependency=job_id))
    state["control_failures"] = 0
    journal.save()


def control(path):
    path = Path(path).resolve()
    current_job = os.environ.get("SLURM_JOB_ID")
    if not current_job:
        raise ValueError("Continuation requires a Slurm compute allocation")
    with controller_lock(path.parent / ".continuation.lock"):
        state = read(path)
        spec = state["spec"]
        common.validate_source(spec)
        if common.sha256(state["plan"]) != state["plan_sha256"]:
            raise ValueError("Frozen continuation plan changed")
        journal = Journal(path, state, Slurm(spec["cluster"]["account"], spec["cluster"]["qos"]))
        try:
            # A guard itself may have been accepted with a lost response. Bind
            # that intent before interpreting current-controller ownership.
            for label, entry in list(state["jobs"].items()):
                if not entry.get("job_id"):
                    journal.submit(label, [])
            drive(journal, current_job=current_job)
        except Exception as exc:
            state["control_failures"] += 1
            state["history"].append({"at_utc": now(), "job_id": current_job, "error": repr(exc)})
            state["status"] = ("unrecoverable" if isinstance(exc, ValueError)
                               or state["control_failures"] >= state["limits"]["max_control_failures"]
                               else "recovery_required")
            if state["status"] == "recovery_required":
                try:
                    # A lost response may still be invisible when the existing
                    # guard wakes. Arm one bounded later reconciliation attempt.
                    # Consecutive failures reset after success; submission
                    # identities must never reset. Seed from older receipts too.
                    recorded = [int(label.removeprefix("G-retry-")) for label in state["jobs"]
                                if label.startswith("G-retry-") and label.removeprefix("G-retry-").isdigit()]
                    state["recovery_attempts"] = max([state.get("recovery_attempts", 0), *recorded]) + 1
                    journal.save()
                    journal.submit("G-retry-" + str(state["recovery_attempts"]),
                                   control_args(spec, path, dependency=current_job, delayed=True))
                except Exception as retry_error:
                    state["status"] = "unrecoverable"
                    state["history"].append({"at_utc": now(), "recovery_submission_error": repr(retry_error)})
            journal.save()
            raise
        print(json.dumps({"run_id": state["run_id"], "status": state["status"], "rounds": len(state["rounds"]),
                          "completed_model_keys": state["completed_model_keys"], "state": str(path)}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "control", "recover-bootstrap"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.action == "control":
        control(args.path)
    elif args.action == "recover-bootstrap":
        print("Bootstrap job: " + submit_bootstrap(args.path))
    else:
        spec = read(args.path.parent / "prepared-launch.json")
        start(spec, args.path)


if __name__ == "__main__":
    main()
