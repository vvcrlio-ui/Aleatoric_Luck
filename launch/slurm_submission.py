"""Journaled Slurm submission shared by every cluster launch path.

Every sbatch intent is persisted before the call. An uncertain response is
matched against squeue AND accounting by an immutable unique job name and is
never sent a second time.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import experiment as common
import discoverer_resources as resources

sys.path.insert(0, str(common.ROOT / "NK_Grid/slurm"))
from submission_journal import _lock, _write

TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}


def now():
    return datetime.now(timezone.utc).isoformat()


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


def batch_environment():
    # Slurm CLI flags override only named options; inherited SBATCH_* can inject
    # dependencies, arrays, export modes or a GPU constraint into another site.
    env = {key: value for key, value in os.environ.items() if not key.startswith("SBATCH_")}
    env.update({key: "1" for key in common.THREADS})
    env['SLURM_EXPORT_ENV'] = 'ALL'
    return env


class AmbiguousSubmission(RuntimeError):
    pass


class Slurm:
    def __init__(self, account, qos, user=None):
        import getpass
        self.account, self.qos, self.user = account, qos, user or getpass.getuser()

    def submit(self, command):
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
                    or row["user"] != self.user or (self.qos is not None and row["qos"] != self.qos)):
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
