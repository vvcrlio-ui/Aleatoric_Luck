"""Native remaining-key scale/RPC probe; all submitted results are synthetic.

Read only a completely audited first-round prefix and immutable assignments.
Never seal, export, resume, or modify the production experiment. The generated
queue is explicitly validation-only and cannot be used as a production resume.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import secrets
import shutil
import sys
import threading
import time
import urllib.error

from aleatoric_nk_grid.shared_queue import Dispatcher, ModelTask, QueueError, atomic_json, file_digest
from aleatoric_nk_grid.queue_service import Client, make_server
from aleatoric_nk_grid.task_table import read_row_group
from aleatoric_nk_grid.execution_contract import task_row_digest
from aleatoric_nk_grid.scheduler_cost import CostEstimator


def read(path):
    return json.loads(Path(path).read_bytes())


def now():
    return datetime.now(timezone.utc).isoformat()


def main(args):
    if sys.platform != 'linux':
        raise RuntimeError('Native Linux required')
    args.output.mkdir(parents=True, exist_ok=False)
    args.scratch.mkdir(parents=True, exist_ok=True)
    assert not args.scratch.resolve().is_relative_to(Path('/valhalla'))
    available_scratch = shutil.disk_usage(args.scratch).free
    if available_scratch < 12 * 1024**3:
        raise RuntimeError(f'Local scratch requires 12 GiB free; observed {available_scratch} bytes')
    assert args.output.resolve().is_relative_to(Path('/valhalla'))
    report = dict(validation_only=True, synthetic_results=True, started_at_utc=now(),
        clients=args.clients, results_per_client=args.results_per_client,
        durable_root=str(args.output), sqlite_scratch=str(args.scratch),
        scratch_free_bytes_before=available_scratch,
        scratch_backend='node-local tmpfs' if args.scratch.parent == Path('/dev/shm') else 'node-local filesystem',
        slurm_memory_mib=os.environ.get('SLURM_MEM_PER_NODE'),
        probe_sha256=file_digest(Path(__file__)),
        source_sha256={name:file_digest(Path(__import__('aleatoric_nk_grid').__file__).parent/name)
            for name in ('shared_queue.py','queue_service.py','task_table.py','scheduler_cost.py')},
        limitations='Captured first-round pending keys; loopback clients on one compute node; synthetic results. No production recovery, cross-node TLS, complete-lifetime journal volume or numerical throughput claim.')

    def mark(phase, **values):
        report.update(phase=phase, updated_at_utc=now(), **values)
        atomic_json(args.output/'progress.json', report)
        print(json.dumps({'phase':phase, **values}), flush=True)

    summary = read(args.audit/'summary.json')
    raw_report = read(args.audit/'report.json')
    assert summary['all_prefixes_checked'] and summary['workers_checked'] == summary['workers_expected'] == 698
    assert not summary['totals']['aborted'] and not summary['totals']['interrupted_starts']
    # A complete audit deliberately retains numerical failures as pending work.
    # Only the exact detailed model-failure catalog is admissible; protocol,
    # checksum, aborted-task and exception records must still fail closed.
    def error_keys(records):
        keys = []
        for worker in records:
            for error in worker['errors']:
                assert set(error) == {'row_id','model','problems'}
                keys.append((worker['worker'],error['row_id'],error['model'],tuple(error['problems'])))
        assert len(keys) == len(set(keys))
        return set(keys)
    expected_errors = {(r['worker'],r['row_id'],r['model'],tuple(r['problems'])) for r in summary['failed_cells']}
    assert error_keys(raw_report['errors']) == expected_errors
    control = read(args.audit/'control_snapshot.json')
    assert len(control['rounds']) == 1 and control['rounds'][0]['round'] == 1
    assert control['run_id'] == summary['run_id'] == '17820c4142ba439b82911477a3b4b958'
    plan = read(args.old_run/'plan.json')
    assert file_digest(args.old_run/'plan.json') == control['plan_sha256']
    snapshot = read(plan['snapshot']); rnd = control['rounds'][0]
    analysis = read(snapshot['analysis_contract'])
    assert analysis['analysis_id'] == summary['analysis_id']
    spec = analysis['cell_execution_spec']
    generation = Path(snapshot['output_dir'])/'executions'/rnd['execution_plan_id']/('round-'+str(rnd['round']))/('generation-'+rnd['generation'])
    activation = read(generation/'generation.activation.json')
    assignment = Path(activation['assignment_path'])
    assert file_digest(assignment) == activation['assignment_sha256']
    assert file_digest(activation['assignment_index_path']) == activation['assignment_index_sha256']
    index = {r['worker']:r for r in read(activation['assignment_index_path'])['row_groups']}
    workers = {r['worker']:r for r in map(json.loads,(args.audit/'workers.jsonl').read_bytes().splitlines())}
    assert set(workers) == set(index) == set(range(698))
    assert error_keys(workers.values()) == expected_errors
    expected = analysis['expected_model_rows'] - summary['totals']['valid_results']
    assert analysis['expected_model_rows'] == 18_000_000 and expected > args.clients * args.results_per_client
    failures = {(r['worker'],r['row_id'],r['model']) for r in summary['failed_cells']}
    assert len(failures) == summary['totals']['failed_results']
    costs = CostEstimator(); by_model = Counter(); generated = [0]

    def pending_tasks():
        found_failures = set()
        for worker in range(698):
            observation = workers[worker]
            assert observation['full_captured_prefix_checked']
            rows = read_row_group(assignment, worker)
            assert len(rows) == index[worker]['row_count'] == observation['assigned_tasks']
            assert task_row_digest(rows) == index[worker]['canonical_task_rows_sha256']
            completed = observation['completed_tasks']
            assert 0 <= completed <= len(rows)
            if completed:
                last = rows[completed-1]; saved = observation['last_completed_cell']
                assert (last.seed,last.draw,last.n_samples,last.k_features,last.group) == (
                    saved['seed'],saved['draw'],saved['N'],saved['K'],saved['group'])
            for position,row in enumerate(rows):
                for model in row.models:
                    failure = (worker,row.row_id,model)
                    if position < completed and failure not in failures:
                        continue
                    if failure in failures:
                        assert position < completed
                        found_failures.add(failure)
                    task = ModelTask(row.seed,row.draw,row.n_samples,row.k_features,model)
                    generated[0] += 1; by_model[model] += 1
                    yield task, costs.estimate(model,row.n_samples,row.k_features)
            if worker % 100 == 0:
                print(json.dumps({'planned_workers':worker+1,'pending_keys':generated[0]}),flush=True)
        assert found_failures == failures and generated[0] == expected
        assert dict(by_model) == {model:2_000_000-summary['by_model'].get(model,0) for model in spec['models']}

    mark('streaming_pending_plan', expected_pending=expected, audit_capture_utc=summary['frontier_captured_to_utc'],
        audit_summary_sha256=file_digest(args.audit/'summary.json'), assignment_sha256=activation['assignment_sha256'])
    root = args.output/'queue'; began = time.monotonic()
    Dispatcher.create(root, pending_tasks(), identity={'validation_only':True,'synthetic_results':True,
        'audit_summary_sha256':report['audit_summary_sha256'],'assignment_sha256':report['assignment_sha256']}, lease_seconds=1800)
    mark('building_local_index', plan_seconds=time.monotonic()-began, pending_by_model=dict(by_model),
        tasks_manifest_bytes=(root/'tasks.jsonl').stat().st_size)
    original_stack = threading.stack_size(256*1024)
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < 8192:
        resource.setrlimit(resource.RLIMIT_NOFILE,(min(8192,hard),hard))
    report['file_descriptor_soft_limit'] = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    token = secrets.token_hex(32)
    retries = Counter(); timings = []; mutex = threading.Lock(); peak_leases = [0]
    deadline = time.monotonic()+1800
    started_barrier = threading.Barrier(args.clients)
    leased_barrier = threading.Barrier(args.clients)

    try:
        began = time.monotonic()
        with Dispatcher(root,scratch=args.scratch) as queue:
            mark('rpc_pressure', index_build_seconds=time.monotonic()-began,
                index_bytes=queue.db_path.stat().st_size)
            server = make_server(queue,token=token)
            service = threading.Thread(target=server.serve_forever);service.start()
            client = Client('http://127.0.0.1:'+str(server.server_port),token,queue.queue_id)

            def call(operation, **kwargs):
                began = time.monotonic()
                while True:
                    try:
                        value = client.call(operation,**kwargs)
                        with mutex:
                            timings.append(time.monotonic()-began)
                        return value
                    except (OSError,urllib.error.URLError) as exc:
                        with mutex:
                            retries[type(exc).__name__] += 1
                        if time.monotonic() > deadline:
                            raise
                        time.sleep(.05)

            def worker(i):
                name = 'scale-client-'+str(i)
                started_barrier.wait(timeout=180)
                for iteration in range(args.results_per_client):
                    lease = call('claim',worker=name)
                    assert lease['state'] == 'task'
                    if iteration == 0:
                        with mutex:
                            peak_leases[0] = max(peak_leases[0], queue.stats().get('leased',0))
                        leased_barrier.wait(timeout=600)
                    call('heartbeat',worker=name,task_id=lease['id'],token=lease['token'])
                    result = {**lease['task'],'status':'ok','mse':.25,'rmse':.5,'mae':.4,
                        'validation_only':True,'synthetic_payload':'x'*1024}
                    call('submit',worker=name,task_id=lease['id'],token=lease['token'],result=result)
                    call('submit',worker=name,task_id=lease['id'],token=lease['token'],result=result)
                return args.results_per_client

            began = time.monotonic()
            try:
                with ThreadPoolExecutor(max_workers=args.clients) as pool:
                    accepted = sum(pool.map(worker,range(args.clients)))
                elapsed = time.monotonic()-began
                stats = queue.stats()
                assert peak_leases[0] == args.clients
                assert stats.get('done') == accepted == args.clients*args.results_per_client
                assert stats.get('pending') == expected-accepted and not stats.get('leased',0)
                queue.export_results(args.output/'synthetic-results.jsonl')
                orphan = queue.claim('orphan-before-restart')
                mark('rebuilding_and_replaying', rpc_seconds=elapsed, accepted=accepted,
                    accepted_per_second=accepted/elapsed, peak_simultaneous_leases=peak_leases[0],
                    transport_retries=dict(retries), journal_bytes=(root/'events.jsonl').stat().st_size,
                    index_bytes_after_results=queue.db_path.stat().st_size)
            finally:
                server.shutdown();server.server_close();service.join()
        began = time.monotonic()
        with Dispatcher(root,scratch=args.scratch) as queue:
            replay = time.monotonic()-began
            stats = queue.stats()
            assert stats.get('done') == accepted and stats.get('pending') == expected-accepted
            assert not stats.get('leased',0)
            try:
                queue.heartbeat(orphan['id'],orphan['token'],'orphan-before-restart')
            except QueueError:
                pass
            else:
                raise AssertionError('Pre-restart lease survived fencing')
            queue.export_results(args.output/'replayed-synthetic-results.jsonl')
        assert file_digest(args.output/'synthetic-results.jsonl') == file_digest(args.output/'replayed-synthetic-results.jsonl')
        ordered = sorted(timings)
        mark('complete', passed=True, restart_rebuild_replay_seconds=replay,
            restart_results_equal=True, stale_lease_rejected=True,
            request_latency_p50_seconds=ordered[len(ordered)//2],
            request_latency_p95_seconds=ordered[int(.95*(len(ordered)-1))],
            request_latency_max_seconds=ordered[-1],peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        atomic_json(args.output/'report.json',report)
    except BaseException as exc:
        mark('failed',passed=False,error=repr(exc))
        raise
    finally:
        threading.stack_size(original_stack)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('audit','old-run','output','scratch'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--clients',type=int,default=698)
    parser.add_argument('--results-per-client',type=int,default=2)
    main(parser.parse_args())
