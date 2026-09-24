"""Opt-in numerical worker for queue_service; never submits Slurm jobs."""
from __future__ import annotations
import argparse
import json
import math
import os
import time
from pathlib import Path

from .shared_queue import (Dispatcher, MAX_BATCH_TASKS, MAX_LEASE_TASKS, ModelTask, QueueError, digest,
                           heartbeat_interval)
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


def register_worker_binding(client, worker, binding, *, timeout_seconds=300.):
    """Only these idempotent startup operations may be retried after a lost reply."""
    from .queue_service import retry_delay
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise TimeoutError('No time remains for worker binding registration')
    until = time.monotonic() + timeout_seconds
    failures = 0
    while True:
        try:
            client.call('protocol', worker=worker, version=2, hostname=binding['hostname'])
            return client.call('worker_binding', worker=worker, binding=binding)
        except OSError as exc:
            remaining = until - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Worker binding registration did not recover') from exc
            client.metrics.add('binding_registration_retry')
            time.sleep(min(remaining, retry_delay(failures, base=.2, cap=5.)))
            failures += 1
            if time.monotonic() >= until:
                raise TimeoutError('Worker binding registration did not recover') from exc


def run(args):
    entered = time.monotonic()
    protocol_version = getattr(args, 'protocol_version', 1)
    max_batch = getattr(args, 'max_batch', None)
    if max_batch is None:
        max_batch = MAX_LEASE_TASKS if protocol_version == 2 else MAX_BATCH_TASKS
    # Set numerical thread caps before importing NumPy/sklearn/native libraries.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = "1"
    from threadpoolctl import threadpool_limits
    from .execution_contract import CellExecutionSpec
    from .nk_grid import NKGridExecutionSession
    from .cell_cache import CachedSession, NodeInputStore, session_namespace
    from .queue_service import Client, execute_batch_worker, worker_slot
    manifest = json.loads((args.root / "manifest.json").read_bytes())
    heartbeat_seconds = heartbeat_interval(manifest)
    queue_id = digest(manifest)
    if json.loads((args.root / "queue-id.json").read_bytes())["queue_id"] != queue_id:
        raise QueueError("Queue identity changed")
    spec = CellExecutionSpec.from_payload(manifest["identity"]["cell_spec"])
    if ((spec.payload.get("prediction_cache") or {}).get("mode", "off") != "off"
            and not manifest["identity"].get("prediction_workflow")):
        raise QueueError("Required prediction cache is unsupported by this legacy queue identity")
    if spec.payload["model_n_jobs"] != 1:
        raise QueueError("First scheduler version requires one numerical thread")
    from .protocol_metrics import ProtocolMetrics
    sample_modulo = getattr(args, 'protocol_metrics_sample_modulo', 1)
    if type(sample_modulo) is not int or not 1 <= sample_modulo <= 1024:
        raise QueueError('Invalid metrics sampling modulo')
    sampled = int(os.environ.get('SLURM_LOCALID', '0')) % sample_modulo == 0
    metrics = ProtocolMetrics(getattr(args, 'protocol_metrics', False) and sampled)
    client = Client(args.url, args.token_file.read_text().strip(), queue_id, ca_file=args.ca_file,
                    keepalive=getattr(args, 'rpc_keepalive', False), metrics=metrics)
    if getattr(args, 'node_relay', False):
        from .node_relay import attach_node_relay, relay_stats_path
        leader = os.environ.get('SLURM_LOCALID') == '0'
        # One small file per node, next to the protocol metrics, so a running relay can be watched.
        relay_stats = relay_stats_path(args.spool) if leader and getattr(args, 'protocol_metrics', False) else None
        attach_node_relay(client, ca_file=args.ca_file, idle_seconds=getattr(args, 'node_relay_idle_seconds', .5),
                          leader=leader, stats_file=relay_stats)
    aggregate = getattr(args, 'heartbeat_aggregate_seconds', 0.)
    if not math.isfinite(aggregate) or not 0 <= aggregate <= 5:
        raise QueueError('Heartbeat aggregate window must be within 0..5 seconds')
    if aggregate:
        client.relay_options = dict(ca_file=args.ca_file, window=min(aggregate, heartbeat_seconds / 4),
                                    leader=os.environ.get('SLURM_LOCALID') == '0')
    def register_binding(worker):
        if getattr(args, 'require_cpu_binding', False):
            if protocol_version != 2:
                raise QueueError('Explicit CPU binding requires protocol 2')
            from .service_binding import process_binding
            binding = process_binding()
            remaining = min(300., args.max_seconds - (time.monotonic() - entered))
            if getattr(args, 'deadline_epoch', None) is not None:
                remaining = min(remaining, args.deadline_epoch - time.time())
            register_worker_binding(client, worker, binding, timeout_seconds=remaining)
    try:
        if manifest["identity"].get("prediction_workflow"):
            if protocol_version != 2:
                raise QueueError("Prediction workflow requires incremental protocol 2")
            from .prediction_worker import PredictionTaskExecutor
            from .shared_queue import atomic_json
            with worker_slot(args.spool, queue_id) as worker, threadpool_limits(1):
                register_binding(worker)
                def storage_fault(error):
                    fault = dict(queue_id=queue_id, worker=worker, code="prediction_storage_failed",
                                 message=str(error))
                    # Either durable marker or RPC can stop the controller. Preserve
                    # the original I/O exception even if the same disk is still full.
                    for directory, filename in ((args.spool, "worker-fault"),
                        (getattr(args, 'fault_dir', None), digest(worker) + '.json')):
                        if directory is not None:
                            try:
                                Path(directory).mkdir(parents=True, exist_ok=True)
                                atomic_json(Path(directory) / filename, fault)
                            except OSError:
                                pass
                    try:
                        client.call("worker_fault", worker=worker, code=fault["code"], message=fault["message"])
                    except Exception:
                        pass
                with PredictionTaskExecutor(manifest["identity"], worker=worker,
                                            repo_root=args.repo_root, on_storage_fault=storage_fault,
                                            recovery_reference=manifest.get("prediction_recovery")) as executor:
                    executor.writer.protocol_metrics = metrics
                    from .protocol_metrics import write_import_proof
                    write_import_proof('numerical-worker')
                    if executor.fold_writer is not None:
                        executor.fold_writer.protocol_metrics = metrics
                        executor.fold_writer.metric_prefix = 'fold'
                    report = execute_batch_worker(client, worker, executor, spool=args.spool,
                        heartbeat_seconds=heartbeat_seconds, max_batch=max_batch, protocol_version=2,
                        deadline_epoch=getattr(args, 'deadline_epoch', None),
                        fault_dir=getattr(args, 'fault_dir', None),
                        recover_stale_leases=getattr(args, 'recover_stale_leases', False),
                        stop=lambda: executor.writer.failed or bool(args.stop_file and args.stop_file.exists()),
                        deadline_seconds=max(0., args.max_seconds - (time.monotonic() - entered)))
                    report.update(cache=dict(executor.stats), worker=worker,
                                  phase=executor.phase)
                    if executor.writer.failed:
                        storage_fault("Prediction persistence failed; stopped claiming work")
                        report["state"] = "prediction_storage_failed"
                    return report
        with worker_slot(args.spool, queue_id) as worker, threadpool_limits(1), \
                NKGridExecutionSession.open(spec, repo_root=args.repo_root) as session:
            register_binding(worker)
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
                report = execute_batch_worker(client, worker, execute,
                    spool=args.spool, cached_cells=lambda: cached.cached_cells,
                    heartbeat_seconds=heartbeat_seconds,
                    max_batch=max_batch,
                    protocol_version=protocol_version,
                    deadline_epoch=getattr(args, 'deadline_epoch', None),
                    fault_dir=getattr(args, 'fault_dir', None),
                    recover_stale_leases=getattr(args, 'recover_stale_leases', False),
                    stop=lambda: bool(args.stop_file and args.stop_file.exists()),
                    deadline_seconds=max(0., args.max_seconds - (time.monotonic() - entered)))
                report["cache"] = cached.stats; report["worker"] = worker
                return report
            finally:
                cached.close()

    finally:
        client.close()


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
    worker.add_argument('--protocol-metrics', action='store_true')
    worker.add_argument('--protocol-metrics-sample-modulo', type=int, default=1)
    worker.add_argument('--require-cpu-binding', action='store_true')
    worker.add_argument('--rpc-keepalive', action='store_true')
    worker.add_argument('--heartbeat-aggregate-seconds', type=float, default=0.)
    worker.add_argument('--node-relay', action='store_true',
        help="Forward this worker's requests through the node-local relay (opt-in)")
    worker.add_argument('--node-relay-idle-seconds', type=float, default=.5,
        help="Relay upstream idle limit; keep it below the server keep-alive wait")
    worker.add_argument("--node-cache", type=Path)
    worker.add_argument("--memory-cache-mib", type=int, default=0, help="Opt in after workload-specific benchmarks")
    worker.add_argument("--disk-cache-mib", type=int, default=1024)
    worker.add_argument("--stop-file", type=Path)
    worker.add_argument("--max-seconds", type=float, default=3600)
    worker.add_argument("--deadline-epoch", type=float,
        help="Shared allocation drain deadline as a Unix timestamp, including initialization time")
    worker.add_argument("--protocol-version", type=int, choices=(1, 2), default=1,
        help="Direct queue production selects 2; legacy dispatchers retain protocol 1")
    worker.add_argument("--fault-dir", type=Path,
        help="Round-scoped directory for controller-visible terminal worker fault markers")
    worker.add_argument("--recover-stale-leases", action="store_true",
        help="Quarantine fenced results and keep claiming; requires typed lease-loss RPC")
    worker.add_argument("--max-batch", type=int,
        help="Upper bound on cells leased per request; 1 restores one cell per round trip")
    args = parser.parse_args()
    if args.command == "plan":
        identity = json.loads(args.identity.read_bytes())
        if (identity.get("prediction_workflow") or
                (identity.get("cell_spec", {}).get("prediction_cache") or {}).get("mode", "off") != "off"):
            raise QueueError("Use cluster_queue.prepare for the required protocol-2 prediction workflow")
        weights = json.loads(args.cost_weights.read_bytes()) if args.cost_weights else None
        profile = json.loads(args.cost_profile.read_bytes()) if args.cost_profile else None
        identity = {**identity, "cost_profile": profile, "cost_weights": weights}
        result = {"queue_id": Dispatcher.create(args.root, iter_model_tasks(identity["cell_spec"], weights, profile=profile), identity=identity)}
    else:
        result = run(args)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
