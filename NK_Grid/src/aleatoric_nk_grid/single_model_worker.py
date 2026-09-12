"""Opt-in numerical worker for queue_service; never submits Slurm jobs."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path

from .shared_queue import Dispatcher, ModelTask, QueueError, digest
from .scheduler_cost import CostEstimator, DEFAULT_COST_WEIGHTS

def iter_model_tasks(spec, weights=None, *, profile=None):
    """Stream full design, one model per row, no Python list of 18M tasks."""
    value = spec.payload if hasattr(spec, "payload") else spec
    estimator = CostEstimator(weights=weights, profile=profile)
    for k in value["resolved_k_grid"]:
        for n in value["resolved_n_grid"]:
            for seed, draw in value["resolved_repeat_plan"]:
                for model in value["models"]:
                    yield ModelTask(int(seed), int(draw), int(n), int(k), model), estimator.estimate(model, n, k)


def json_result(value):
    """Preserve finite numbers; JSON null represents unavailable diagnostics."""
    if isinstance(value, dict):
        return {str(k): json_result(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_result(v) for v in value]
    if hasattr(value, "item"):
        return json_result(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run(args):
    # Set numerical thread caps before importing NumPy/sklearn/native libraries.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = "1"
    from threadpoolctl import threadpool_limits
    from .execution_contract import CellExecutionSpec
    from .nk_grid import NKGridExecutionSession
    from .cell_cache import CachedSession, NodeInputStore, session_namespace
    from .queue_service import Client, execute_worker, worker_slot
    manifest = json.loads((args.root / "manifest.json").read_bytes())
    queue_id = digest(manifest)
    if json.loads((args.root / "queue-id.json").read_bytes())["queue_id"] != queue_id:
        raise QueueError("Queue identity changed")
    spec = CellExecutionSpec.from_payload(manifest["identity"]["cell_spec"])
    if spec.payload["model_n_jobs"] != 1:
        raise QueueError("First scheduler version requires one numerical thread")
    client = Client(args.url, args.token_file.read_text().strip(), queue_id, ca_file=args.ca_file)
    with worker_slot(args.spool, queue_id) as worker, threadpool_limits(1), \
            NKGridExecutionSession.open(spec, repo_root=args.repo_root) as session:
        store = None
        if args.node_cache:
            store = NodeInputStore(args.node_cache, namespace=session_namespace(session), max_bytes=args.disk_cache_mib * 1024**2)
        cached = CachedSession(session, max_bytes=args.memory_cache_mib * 1024**2, store=store)
        try:
            def execute(value):
                if args.memory_cache_mib or store is not None:
                    row = cached.run(ModelTask(**value))
                else:
                    row = session.run_cell_group(seed=value["seed"], draw=value["draw"],
                        n_samples=value["N"], k_features=value["K"], models=(value["model"],))[0]
                return json_result(row)
            report = execute_worker(client, worker, execute,
                spool=args.spool, cached_cells=lambda: cached.cached_cells,
                heartbeat_seconds=min(20., manifest["lease_seconds"] / 3),
                recover_stale_leases=getattr(args, 'recover_stale_leases', False),
                stop=lambda: bool(args.stop_file and args.stop_file.exists()),
                deadline_seconds=args.max_seconds)
            report["cache"] = cached.stats; report["worker"] = worker
            return report
        finally:
            cached.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("root", type=Path)
    plan.add_argument("--identity", type=Path, required=True, help="JSON containing cell_spec and optional migration certificate digest")
    plan.add_argument("--cost-weights", type=Path)
    plan.add_argument("--cost-profile", type=Path)
    worker = sub.add_parser("run")
    worker.add_argument("root", type=Path)
    worker.add_argument("--url", required=True)
    worker.add_argument("--repo-root", type=Path, required=True)
    worker.add_argument("--token-file", type=Path, required=True)
    worker.add_argument("--ca-file", type=Path, help="Trust the private dispatcher CA; hostname verification remains enabled")
    worker.add_argument("--spool", type=Path, required=True,
        help="Durable directory unique to this logical slot; reuse it after a process/node restart")
    worker.add_argument("--node-cache", type=Path)
    worker.add_argument("--memory-cache-mib", type=int, default=0, help="Opt in after workload-specific benchmarks")
    worker.add_argument("--disk-cache-mib", type=int, default=1024)
    worker.add_argument("--stop-file", type=Path)
    worker.add_argument("--max-seconds", type=float, default=3600)
    worker.add_argument("--recover-stale-leases", action="store_true",
        help="Quarantine fenced results and keep claiming; requires typed lease-loss RPC")
    args = parser.parse_args()
    if args.command == "plan":
        identity = json.loads(args.identity.read_bytes())
        weights = json.loads(args.cost_weights.read_bytes()) if args.cost_weights else None
        profile = json.loads(args.cost_profile.read_bytes()) if args.cost_profile else None
        identity = {**identity, "cost_profile": profile, "cost_weights": weights}
        result = {"queue_id": Dispatcher.create(args.root, iter_model_tasks(identity["cell_spec"], weights, profile=profile), identity=identity)}
    else:
        result = run(args)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
