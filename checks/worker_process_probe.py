"""Bounded real-process crash probe, synthetic results, standard library only.

Runs on Windows or Linux. This does not validate Slurm or native model execution.
Only terminates child processes created by this probe.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / "NK_Grid/src"))
from aleatoric_nk_grid.queue_service import Client, execute_worker, make_server, worker_slot
from aleatoric_nk_grid.shared_queue import Dispatcher, ModelTask, QueueError, atomic_json

TOKEN = "synthetic-worker-crash-probe-token"


def child(args):
    root = args.root
    def barrier():
        atomic_json(root / "barrier.json", {"pid": os.getpid(), "mode": args.mode})
        while True:
            time.sleep(.05)

    class FaultClient(Client):
        def call(self, operation, **arguments):
            if operation == "submit" and args.mode == "before_submit":
                barrier()
            result = super().call(operation, **arguments)
            if operation == "submit" and args.mode == "after_submit":
                barrier()
            return result

    client = FaultClient(args.url, TOKEN, args.queue_id)
    try:
        with worker_slot(root / "spool", args.queue_id) as worker:
            if args.mode == "lock_only":
                return 9  # Another live owner should have rejected entry.
            def execute(task):
                with (root / "executions.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"pid": os.getpid(), "task": task}) + "\n")
                    handle.flush(); os.fsync(handle.fileno())
                if args.mode == "in_execute":
                    barrier()
                return {**task, "status": "ok", "mse": .25}
            report = execute_worker(client, worker, execute, spool=root / "spool",
                heartbeat_seconds=.1, idle_seconds=.05, deadline_seconds=20)
            atomic_json(root / "worker-report.json", report)
    except QueueError as exc:
        if args.mode == "lock_only" and "Owner already holds" in str(exc):
            return 0
        raise
    return 0


@contextmanager
def service(root, clock):
    with Dispatcher(root / "queue", scratch=root / "scratch", clock=clock) as dispatcher:
        server = make_server(dispatcher, token=TOKEN)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            yield dispatcher, "http://127.0.0.1:" + str(server.server_port)
        finally:
            server.shutdown(); server.server_close(); thread.join()


def command(root, url, queue_id, mode):
    return [sys.executable, str(Path(__file__).resolve()), "child", str(root),
            "--url", url, "--queue-id", queue_id, "--mode", mode]


def stop_child(process):
    if process.poll() is None:
        process.kill()
    process.wait(timeout=10)


def wait_barrier(root, process):
    deadline = time.monotonic() + 10
    while not (root / "barrier.json").exists():
        if process.poll() is not None:
            raise RuntimeError("Child exited before crash barrier; inspect child logs")
        if time.monotonic() >= deadline:
            raise TimeoutError("Child did not reach crash barrier")
        time.sleep(.02)
    assert json.loads((root / "barrier.json").read_bytes())["pid"] == process.pid


def probe_case(base, name, mode, *, expire=False, competing_result=False, restart_service=False):
    root = base / name; root.mkdir()
    task = ModelTask(1, 0, 10, 1, "ols")
    queue_id = Dispatcher.create(root / "queue", [(task, 1)], identity={"test": name}, lease_seconds=120)
    offset = [0.]
    clock = lambda: time.time() + offset[0]
    with service(root, clock) as (dispatcher, url):
        with (root / "crashed-child.log").open("w") as log:
            process = subprocess.Popen(command(root, url, queue_id, mode), stdout=log, stderr=subprocess.STDOUT)
            try:
                wait_barrier(root, process)
                subprocess.run(command(root, url, queue_id, "lock_only"), check=True, timeout=10,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                identity = json.loads((root / "spool/.worker-identity").read_bytes())
                before = dispatcher.stats()
                stop_child(process)
            finally:
                stop_child(process)
        if expire:
            offset[0] += 1000
        if competing_result:
            replacement = dispatcher.claim("replacement")
            dispatcher.submit(replacement["id"], replacement["token"], "replacement",
                              {**replacement["task"], "status": "ok", "mse": .8})

        def recover(q, endpoint):
            with (root / "restarted-child.log").open("w") as log:
                subprocess.run(command(root, endpoint, queue_id, "normal"), check=True,
                               timeout=25, stdout=log, stderr=subprocess.STDOUT)
            assert json.loads((root / "spool/.worker-identity").read_bytes()) == identity
            assert q.stats().get("done") == 1 and q.stats().get("leased", 0) == 0
            q.export_results(root / "results.jsonl")
            results = [json.loads(line) for line in (root / "results.jsonl").read_text().splitlines()]
            assert len(results) == 1
            assert results[0]["result"]["mse"] == (.8 if competing_result else .25)
            calls = (root / "executions.jsonl").read_text().splitlines()
            assert len(calls) == (2 if (mode == "in_execute" or expire and not competing_result) else 1)
            receipts = list((root / "spool").glob("*.json"))
            assert len(receipts) == int(competing_result)
            report = json.loads((root / "worker-report.json").read_bytes())
            assert report["state"] == "complete"
            return {"case": name, "passed": True, "before_crash": before,
                    "worker_report": report, "execution_starts": len(calls),
                    "stale_receipts_retained": len(receipts), "unique_results": len(results),
                    "owner_conflict_rejected": True, "identity_preserved": True}

        if not restart_service:
            result = recover(dispatcher, url)
    if restart_service:
        with service(root, clock) as (dispatcher, url):
            result = recover(dispatcher, url)
    try:
        with worker_slot(root / "spool", "wrong-queue"):
            raise AssertionError("Wrong queue unexpectedly acquired the slot")
    except QueueError:
        result["wrong_queue_rejected"] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["probe", "child"])
    parser.add_argument("root", type=Path)
    parser.add_argument("--url")
    parser.add_argument("--queue-id")
    parser.add_argument("--mode", default="normal")
    args = parser.parse_args()
    if args.command == "child":
        return child(args)
    args.root.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    cases = [probe_case(args.root, "crash_in_execution", "in_execute"),
             probe_case(args.root, "crash_before_delivery", "before_submit"),
             probe_case(args.root, "accepted_reply_lost", "after_submit"),
             probe_case(args.root, "accepted_then_service_restart", "after_submit", restart_service=True),
             probe_case(args.root, "expired_pending_recomputed", "before_submit", expire=True),
             probe_case(args.root, "stale_cannot_overwrite_winner", "before_submit", expire=True, competing_result=True)]
    report = {"passed": True, "platform": sys.platform, "python": sys.version,
              "scope": "Local processes, loopback HTTP, synthetic task results; not Slurm/native model validation",
              "elapsed_seconds": time.monotonic() - began, "cases": cases}
    atomic_json(args.root / "report.json", report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
