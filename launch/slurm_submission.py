"""Journaled Slurm submission shared by every cluster launch path.

Every sbatch intent is persisted before the call. An uncertain response is
matched against squeue AND accounting by an immutable unique job name and is
never sent a second time. A job's terminal state is recorded once and never
queried again. All squeue/sacct calls of one run keep the profile's minimum
spacing, recorded in the run directory across its controller jobs.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

import experiment as common
import slurm_resources as resources

TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


def batch_environment():
    # Slurm CLI flags override only named options; inherited SBATCH_* can inject
    # dependencies, arrays, export modes or a GPU constraint into another site.
    env = {key: value for key, value in os.environ.items() if not key.startswith("SBATCH_")}
    env.update({key: "1" for key in common.THREADS})
    env['SLURM_EXPORT_ENV'] = 'ALL'
    return env


class AmbiguousSubmission(RuntimeError):
    pass


def root_job(job_id):
    return re.split(r"[_.]", job_id)[0]


class Slurm:
    def __init__(self, spec, user=None):
        import getpass
        cluster = spec["cluster"]
        self.account, self.qos, self.user = cluster["account"], cluster.get("qos"), user or getpass.getuser()
        self.interval, self.stamp = cluster["query_interval"], Path(spec["output"]) / "slurm-query.json"

    def query(self, command):
        if not self.interval or command[0] not in ("squeue", "sacct"):
            return resources.query(command)
        if self.stamp.exists():
            time.sleep(max(0, read(self.stamp)["finished_unix"] + self.interval - time.time()))
        try:
            return resources.query(command)
        finally:
            _write(self.stamp, {"finished_unix": time.time()})

    def submit(self, command):
        return subprocess.run(command, capture_output=True, text=True, check=False,
                              timeout=120, env=batch_environment())

    def find(self, name, since):
        """Return unique root IDs even when an array has thousands of rows."""
        fields = "id,name,state,account,user,qos"
        queued = resources.records(self.query(["squeue", "-h", "-r", "-u", self.user,
                                  "--name=" + name, "-o", "%i|%j|%T|%a|%u|%q"]), fields)
        historic = resources.records(self.query(["sacct", "-nP", "-X", "-u", self.user,
                                     "--starttime=" + since[:19], "--name=" + name,
                                     "--format=JobID%80,JobName%80,State,Account,User,QOS"]), fields)
        result = {}
        for row in historic + queued:
            if (row["name"] != name or row["account"] != self.account
                    or row["user"] != self.user or (self.qos is not None and row["qos"] != self.qos)):
                continue
            root = root_job(row["id"])
            if root.isdigit():
                result[root] = row["state"].split()[0].rstrip("+")
        return result

    def states(self, jobs):
        """One squeue for every job, then one sacct for those no longer queued."""
        result = {job: [] for job in jobs}
        try:
            queued = self.query(["squeue", "-h", "-r", "-j", ",".join(result), "-o", "%i|%T"])
        except subprocess.CalledProcessError as exc:
            # Slurm versions may exit 1 instead of returning an empty queue
            # once an old job leaves slurmctld. Other query errors stay fatal.
            if "invalid job id" not in str(exc.stderr).lower():
                raise
            queued = ""
        for row in resources.records(queued, "id,state"):
            result[root_job(row["id"])].append(row["state"])
        historic = [job for job, states in result.items() if not states]
        if historic:
            text = self.query(["sacct", "-nP", "-X", "-j", ",".join(historic), "--format=JobID,State"])
            for row in resources.records(text, "id,state"):
                result[root_job(row["id"])].append(row["state"].split()[0].rstrip("+"))
        missing = [job for job, states in result.items() if not states]
        if missing:
            raise AmbiguousSubmission(f"Slurm has no queue/accounting evidence for jobs {', '.join(missing)}")
        return result


class Journal:
    """Called only while the per-run persistent OS lock is held."""
    def __init__(self, path, state, slurm):
        self.path, self.state, self.slurm = Path(path), state, slurm

    def save(self):
        self.state["updated_at_utc"] = now()
        _write(self.path, self.state)

    def active(self, labels):
        """(label, job) pairs not yet terminal; newly terminal states are recorded."""
        jobs = {label: self.submit(label, []) for label in labels}
        unknown = {label: job for label, job in jobs.items() if "terminal_states" not in self.state["jobs"][label]}
        if not unknown:
            return []
        states = self.slurm.states(unknown.values())
        active = []
        for label, job in unknown.items():
            if all(s in TERMINAL for s in states[job]):
                self.state["jobs"][label]["terminal_states"] = states[job]
            else:
                active.append((label, job))
        self.save()
        return active

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
