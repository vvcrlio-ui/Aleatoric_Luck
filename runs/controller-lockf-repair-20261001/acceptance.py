"""Two-node acceptance of the real controller lock and submission journal.

Only synthetic job receipts are generated; this script never calls sbatch.
Run with srun --nodes=2 --ntasks=2 --ntasks-per-node=1.
"""
from pathlib import Path
import errno
import hashlib
import inspect
import json
import os
import socket
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "launch"))
import slurm_submission as submission


def publish(root, name, value):
    submission._write(root / name, value)


def wait_for(root, name):
    deadline = time.monotonic() + 45
    while not (root / name).exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for " + name)
        time.sleep(0.05)
    return submission.read(root / name)


class InjectedFailure(Exception):
    pass


class ReceiptOnlySlurm:
    """Explicit synthetic receipts exercise Journal without submitting jobs."""
    def __init__(self, rank):
        self.rank = rank
        self.calls = 0

    def submit(self, command):
        self.calls += 1
        return subprocess.CompletedProcess(command, 0, str(1000 + self.rank * 100 + self.calls), "")


def require_blocked(lock):
    try:
        with submission.controller_lock(lock, timeout=0):
            raise AssertionError("Entered while another node held the lock")
    except OSError as exc:
        assert exc.errno in (errno.EACCES, errno.EAGAIN), exc


def main(root):
    rank = int(os.environ["SLURM_PROCID"])
    assert rank in (0, 1)
    host = socket.gethostname()
    lock = root / ".cluster-state.lock"
    state_path = root / "cluster-state.json"
    source = inspect.getsource(submission._lock)
    assert "fcntl.lockf(" in source and "fcntl.flock(" not in source
    source_hash = hashlib.sha256(Path(submission.__file__).read_bytes()).hexdigest()
    publish(root, "node-" + str(rank) + ".json", {"host": host, "source_sha256": source_hash})

    if rank == 0:
        peer = wait_for(root, "node-1.json")
        assert peer["host"] != host, "Expected two different nodes"
        assert peer["source_sha256"] == source_hash, "Source differs across nodes"
        with submission.controller_lock(lock):
            publish(root, "cluster-state.json", {
                "jobs": {}, "run_id": "synthetic-lock-acceptance",
                "limits": {"max_control_jobs": 64}})
        try:
            with submission.controller_lock(lock):
                subprocess.run([sys.executable, "-c", "pass"], check=True, timeout=10)
                publish(root, "exception-holder.json", {"inode": lock.stat().st_ino})
                assert wait_for(root, "exception-blocked.json")["blocked"]
                raise InjectedFailure()
        except InjectedFailure:
            publish(root, "exception-released.json", {"released": True})
        assert wait_for(root, "exception-acquired.json")["acquired"]
    else:
        holder = wait_for(root, "exception-holder.json")
        assert lock.stat().st_ino == holder["inode"]
        require_blocked(lock)
        publish(root, "exception-blocked.json", {"blocked": True})
        with submission.controller_lock(lock, timeout=30):
            assert wait_for(root, "exception-released.json")["released"]
            publish(root, "exception-acquired.json", {"acquired": True})

    slurm = ReceiptOnlySlurm(rank)
    for index in range(16):
        with submission.controller_lock(lock, timeout=30):
            state = submission.read(state_path)
            time.sleep(0.01)  # Widen the read/save window that lost registrations.
            journal = submission.Journal(state_path, state, slurm)
            label = "C" + str(rank) + "-" + str(index)
            job = journal.submit(label, [])
            assert journal.submit(label, []) == job
    assert slurm.calls == 16, "Replay submitted a job twice"
    publish(root, "journal-done-" + str(rank) + ".json", {"calls": slurm.calls})

    wait_for(root, "journal-done-" + str(1 - rank) + ".json")
    if rank == 0:
        with submission.controller_lock(lock):
            jobs = submission.read(state_path)["jobs"]
        expected = {"C" + str(node) + "-" + str(i) for node in (0, 1) for i in range(16)}
        assert set(jobs) == expected, "Concurrent update lost a registration"
        assert len({entry["job_id"] for entry in jobs.values()}) == 32
        assert all(entry["status"] == "accepted" for entry in jobs.values())
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--hold", str(root)],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        try:
            assert child.stdout.readline().strip() == "held"
            publish(root, "exit-holder.json", {"inode": lock.stat().st_ino})
            assert wait_for(root, "exit-blocked.json")["blocked"]
            child.terminate()
            child.wait(timeout=10)
            assert child.returncode != 0
            publish(root, "exit-released.json", {"released": True})
            assert wait_for(root, "exit-acquired.json")["acquired"]
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)
            child.stdin.close()
            child.stdout.close()
        receipt = {"result": "PASS", "nodes": [host, peer["host"]],
                   "submission_module_sha256": source_hash,
                   "synthetic_registered_jobs": len(jobs),
                   "real_jobs_submitted_by_test": 0,
                   "checks": ["cross-node exclusion", "contention timeout",
                              "wait and exception release", "lock retained across subprocess",
                              "concurrent journal preservation", "idempotent submission replay",
                              "release on terminated owner"]}
        publish(root, "acceptance.json", receipt)
        print(json.dumps(receipt, sort_keys=True), flush=True)
    else:
        holder = wait_for(root, "exit-holder.json")
        assert holder["inode"] == lock.stat().st_ino
        require_blocked(lock)
        publish(root, "exit-blocked.json", {"blocked": True})
        with submission.controller_lock(lock, timeout=30):
            assert wait_for(root, "exit-released.json")["released"]
            publish(root, "exit-acquired.json", {"acquired": True})


if __name__ == "__main__":
    if sys.argv[1] == "--hold":
        with submission.controller_lock(Path(sys.argv[2]) / ".cluster-state.lock"):
            print("held", flush=True)
            sys.stdin.readline()
    else:
        main(Path(sys.argv[1]))
