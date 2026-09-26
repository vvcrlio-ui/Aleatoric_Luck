"""Run frozen suite cells with the existing numerical engine and TLS transport."""
import argparse
from collections import OrderedDict
import gc
import json
import os
from pathlib import Path
import random
import signal
import socket
import ssl
import subprocess
import sys
import threading
import time
import uuid

from .shared_queue import QueueError, atomic_json, file_lock
from .suite_queue import SuiteDispatcher, read, check_files


class PanelExecutor:
    def __init__(self, plan):
        self.panels = {p["name"]: p for p in plan["panels"]}
        self.current = None; self.session = None; self.baselines = OrderedDict()

    def close(self):
        if self.session is not None:
            self.session.close(); self.session = None
        self.baselines.clear(); self.current = None

    def __call__(self, task):
        from .config import config_from_json
        from .nk_grid import NKGridExecutionSession, resolved_model_params, model_run_settings
        from .single_model_worker import json_result
        import numpy as np
        p = self.panels[task["panel_id"]]
        if self.current != p["name"]:
            self.close(); gc.collect(); check_files(p["files"])
            session = NKGridExecutionSession.open_from_config(config_from_json(p["config"]))
            try:
                if (session.algorithm_version != p["algorithm_version"] or session.task != p["task"]
                        or session.dataset != p["dataset"] or list(map(int, session.n_grid)) != p["N"]
                        or list(map(int, session.k_grid)) != p["K"]
                        or list(map(list, session.repeat_pairs)) != p["repeats"]
                        or model_run_settings(session.config.models) != p["environment_overrides"]
                        or resolved_model_params(session.selected_model_params) != p["model_params"]):
                    raise QueueError("Panel numerical configuration changed")
            except BaseException:
                session.close(); raise
            self.session = session; self.current = p["name"]
        session = self.session
        row = session.run_cell_group(seed=task["seed"], draw=task["draw"],
                                    n_samples=task["N"], k_features=task["K"], models=(task["model"],))[0]
        row["panel_id"] = p["name"]
        row.update(r2_holdout=None, null_mse_train_N=None, r2_holdout_reason=row["status"])
        if row["status"] == "ok":
            key = task["seed"], task["draw"], task["N"]
            if key not in self.baselines:
                indexes = session.split_manager.for_seed(task["seed"])
                orders = session._orders(task["seed"], task["draw"], indexes.train_index)
                y_train = session.frame.loc[orders.row_index[:task["N"]], session.config.outcome]
                frame = session.external_frame if indexes.external_test else session.frame
                y_test = frame.loc[indexes.test_index, session.config.outcome].to_numpy(dtype=float)
                null = float(np.mean((y_test - float(y_train.mean())) ** 2))
                self.baselines[key] = null
                if len(self.baselines) > len(p["repeats"]) * len(p["N"]):
                    self.baselines.popitem(last=False)
            null = self.baselines[key]
            row["null_mse_train_N"] = null
            row["r2_holdout_reason"] = "" if null > 0 else "zero_baseline_error"
            if null > 0:
                row["r2_holdout"] = 1 - row["mse" if p["task"] == "regression" else "brier"] / null
        return json_result(row)


def worker(launch_path):
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[key] = "1"
    from threadpoolctl import threadpool_limits
    from .queue_service import Client, execute_worker
    from .queue_readiness import wait_ready
    launch = read(launch_path)
    check_files([read(Path(launch["plan"]).with_name("plan-file.json"))])
    plan = read(launch["plan"])
    from .execution_contract import runtime_environment
    if runtime_environment() != plan["runtime_environment"]:
        raise QueueError("Worker environment differs from prepared numerical environment")
    rank = int(os.environ["SLURM_PROCID"]) - 1; control = Path(launch_path).parent
    if not 0 <= rank < launch["workers"]:
        raise QueueError("Worker rank outside frozen allocation")
    spool = control / "spools" / str(rank); spool.mkdir(parents=True, exist_ok=True)
    worker_id = str(rank) + ":" + uuid.uuid4().hex
    atomic_json(control / ("worker-start-" + str(rank) + ".json"), dict(rank=rank, host=socket.gethostname(),
                pid=os.getpid(), job_id=os.environ["SLURM_JOB_ID"], started_at=time.time(),
                cpu_affinity=sorted(os.sched_getaffinity(0)), module_cpu_type=os.environ.get("MODULE_CPU_TYPE")))
    time.sleep(random.uniform(0, min(30., launch["workers"] / 100)))
    ready = wait_ready(control / "ready.json", queue_id=launch["queue_id"], generation=launch["generation"],
                       token=launch["token"], ca_file=launch["cert"])
    client = Client(ready["url"], launch["token"], launch["queue_id"], ca_file=launch["cert"])
    with file_lock(spool / "owner.lock"), threadpool_limits(1):
        executor = PanelExecutor(plan)
        try:
            report = execute_worker(client, worker_id, executor, spool=spool,
                heartbeat_seconds=20, recover_stale_leases=True, deadline_seconds=max(1, launch["max_seconds"] - 120))
            atomic_json(control / ("worker-end-" + str(rank) + ".json"), report)
            return report
        finally:
            executor.close()


def run(plan_path, round_root):
    plan = read(plan_path); allocation = plan["allocation"]
    workers = allocation["workers"]
    if int(os.environ.get("SLURM_NTASKS", 0)) != workers + 1 or int(os.environ.get("SLURM_JOB_NUM_NODES", 0)) != allocation["nodes"]:
        raise QueueError("Actual Slurm allocation differs from frozen pressure-test resources")
    max_seconds = float(plan["wall_seconds"])
    root = Path(round_root); control = root / "control"; control.mkdir(mode=0o700, exist_ok=True)
    manifest = read(root / "manifest.json")
    launch = dict(plan=str(Path(plan_path).resolve()), workers=workers, max_seconds=max_seconds,
                  queue_id=manifest["queue_id"], generation=uuid.uuid4().hex,
                  token=uuid.uuid4().hex + uuid.uuid4().hex, cert=str(control / "ca.crt"))
    atomic_json(control / "launch.json", launch); (control / "launch.json").chmod(0o600)
    actual = subprocess.run(["scontrol", "show", "job", os.environ["SLURM_JOB_ID"], "-o"], capture_output=True, text=True, check=True)
    atomic_json(control / "allocation.json", dict(requested=allocation, slurm=actual.stdout,
        tasks=int(os.environ["SLURM_NTASKS"]), nodes=int(os.environ["SLURM_JOB_NUM_NODES"])))
    def interrupted(*args):
        raise InterruptedError("Worker allocation interrupted; resume saved cells")
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    # Rank zero is the dispatcher: Slurm gives it a core distinct from workers.
    child = subprocess.Popen(["srun", "--ntasks=" + str(workers + 1), "--cpus-per-task=1", "--ntasks-per-core=1",
        "--cpu-bind=cores", "--distribution=cyclic", "--kill-on-bad-exit=1", "--output=" + str(control / "rank-%t.out"),
        "--error=" + str(control / "rank-%t.err"), sys.executable, "-m", "aleatoric_nk_grid.suite_runtime",
        "rank", str(control / "launch.json")])
    try:
        code = child.wait()
        stats = read(control / "progress.json") if (control / "progress.json").exists() else {}
        started = len(list(control.glob("worker-start-*.json")))
        atomic_json(control / "completed.json", dict(stats=stats, worker_exit=code,
            started_workers=started, expected_workers=workers))
        if code or stats.get("done") != manifest["count"] or started != workers:
            raise QueueError("Incomplete allocation; next round uses saved cell difference")
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()


def dispatcher_task(launch_path):
    from .queue_service import make_server
    from .queue_readiness import publish_ready
    launch = read(launch_path); control = Path(launch_path).parent
    with SuiteDispatcher(control.parent) as dispatcher:
        cert, key = control / "ca.crt", control / "server.key"
        host = socket.gethostname()
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days",
            str(max(2, math_ceil_days(launch["max_seconds"]) + 1)), "-keyout", str(key), "-out", str(cert),
            "-subj", "/CN=" + host, "-addext", "subjectAltName=DNS:" + host + ",DNS:" + socket.getfqdn()],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        key.chmod(0o600)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER); context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(cert, key)
        server = make_server(dispatcher, token=launch["token"], host="0.0.0.0", tls_context=context)
        serving = threading.Thread(target=server.serve_forever); serving.start()
        try:
            leader = dict(host=host, pid=os.getpid(), cpu_affinity=sorted(os.sched_getaffinity(0)))
            atomic_json(control / "dispatcher-start.json", leader)
            seen = {}
            def progress(stage):
                for path in control.glob("worker-start-*.json"):
                    if path.name not in seen:
                        seen[path.name] = read(path)
                value = dict(dispatcher.stats(), stage=stage, started_workers=len(seen),
                             expected_workers=launch["workers"],
                             started_hosts=sorted({v["host"] for v in seen.values()}),
                             startup_complete=len(seen) == launch["workers"])
                atomic_json(control / "progress.json", value)
                return value
            publish_ready(control / "ready.json", dispatcher, host=host, port=server.server_port, tls=True, generation=launch["generation"])
            while True:
                progress("running")
                if len(list(control.glob("worker-end-*.json"))) == launch["workers"]:
                    break
                time.sleep(15)
        finally:
            server.shutdown(); serving.join(); server.server_close()


def math_ceil_days(seconds):
    return int((seconds + 86399) // 86400)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["rank"]); p.add_argument("launch", type=Path)
    args = p.parse_args()
    (dispatcher_task if int(os.environ["SLURM_PROCID"]) == 0 else worker)(args.launch)
