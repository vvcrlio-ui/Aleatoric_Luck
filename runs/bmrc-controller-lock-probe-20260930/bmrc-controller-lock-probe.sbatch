#!/usr/bin/env bash
#SBATCH --job-name=controller-lock-probe
#SBATCH --account=mills.prj
#SBATCH --partition=short
#SBATCH --constraint=skl-compat
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --time=00:03:00
#SBATCH --output=controller-lock-probe-%j.out
#SBATCH --error=controller-lock-probe-%j.err

set -euo pipefail
module purge
module load Python/3.11.3-GCCcore-12.3.0
run_root=${1:?Pass the existing experiment output directory as the first argument}
test -d "$run_root"
probe_dir=$(mktemp -d "$run_root/controller-lock-probe-${SLURM_JOB_ID}.XXXXXX")
printf 'Probe directory: %s\n' "$probe_dir"

srun --nodes=2 --ntasks=2 --ntasks-per-node=1 python -u -c '
import errno
import fcntl
import json
import os
from pathlib import Path
import socket
import sys
import time

root = Path(sys.argv[1])
rank = int(os.environ["SLURM_PROCID"])
host = socket.gethostname()

def publish(name, value):
    temp = root / (name + ".tmp")
    temp.write_text(json.dumps(value))
    os.replace(temp, root / name)

def wait_for(name):
    deadline = time.monotonic() + 45
    path = root / name
    while not path.exists():
        if time.monotonic() >= deadline:
            raise RuntimeError("INCONCLUSIVE: timeout waiting for " + name)
        time.sleep(0.1)
    return json.loads(path.read_text())

# Same open mode and flock call as launch/slurm_submission.py.
# This NEW probe file is separate from the experiment controller lock.
with (root / "probe.lock").open("a+b") as handle:
    inode = os.fstat(handle.fileno()).st_ino
    if rank == 0:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            publish("holder.json", {"host": host, "inode": inode})
            result = wait_for("contender.json")
            if result["host"] == host or result["inode"] != inode:
                raise RuntimeError("INCONCLUSIVE: distinct nodes / shared inode not confirmed")
            print(json.dumps({"holder": host, "contender": result["host"],
                              "inode": inode, "contender_result": result["result"]}), flush=True)
            if result["result"] == "ACQUIRED":
                print("FAIL: both nodes acquired an exclusive flock on the same inode", flush=True)
                raise SystemExit(2)
            print("PASS: the second node was blocked while the first held the lock", flush=True)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    else:
        wait_for("holder.json")
        result = "ACQUIRED"
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in (errno.EAGAIN, errno.EACCES):
                raise
            result = "BLOCKED"
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        publish("contender.json", {"host": host, "inode": inode, "result": result})
' "$probe_dir"
