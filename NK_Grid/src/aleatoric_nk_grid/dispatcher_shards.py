"""Dispatcher shards: several independent dispatcher processes serving one round.

One dispatcher is a single Python process and its GIL caps it at a few hundred records per second. A sharded
round splits the round's task list into disjoint slices. Each slice is served by its own dispatcher process (own
core, validators, journal, server) to its own group of worker nodes, all on the controller node. A shard runs under
the round's queue id, so its journal lines are exactly what the unsharded round would have written; at the end the
shard journals are concatenated into the round's ``results.jsonl`` and every consumer of a round sees an ordinary
round. Nothing crosses shards while running, so throughput scales with the number of shards.
"""
from __future__ import annotations
import math
import multiprocessing
import os
import signal
import socket
import ssl
import subprocess
import sys
import threading
import time
import uuid
from array import array
from pathlib import Path

from .shared_queue import QueueError, atomic_json, digest, file_digest, file_lock, sync_directory
from .scheduler_policy import TailMonitor, validate_policy

SUM_KEYS = ('done', 'failed', 'leased', 'pending', 'exhausted', 'total', 'expired_leases', 'fsync_count', 'fsync_seconds',
            'heartbeats_ok', 'commits', 'leased_batches', 'registered_bindings', 'fleet', 'registered_workers',
            'computing_workers', 'submitting_workers', 'idle_workers', 'unknown_workers', 'straggler_workers',
            'unacknowledged_chunks', 'active_heartbeat_older_than_900s', 'active_heartbeat_older_than_1800s')


def _read(path):
    import json
    return json.loads(Path(path).read_bytes())


# ------------------------------------------------------------------ partitioning
def split_integer(total, weights):
    """Split ``total`` into integer parts proportional to ``weights`` (largest remainder, ties to the lower index)."""
    weights = [int(w) for w in weights]
    if not weights or min(weights) < 1:
        raise QueueError('Shard weights must be positive integers')
    whole = sum(weights)
    parts = [total * w // whole for w in weights]
    order = sorted(range(len(weights)), key=lambda i: (-(total * weights[i] % whole), i))
    for i in order[:total - sum(parts)]:
        parts[i] += 1
    return parts


def shard_assignment(count, weights):
    """Shard index for each of ``count`` consecutive positions: an even interleave with exact proportional quotas.

    Every shard keeps the round's cost order (each one gets a proportional share of the expensive head as well as
    of the cheap tail) and the quotas differ from the proportional split by less than one task.
    """
    import numpy as np
    quotas = split_integer(count, weights)
    if min(quotas) < 1:
        raise QueueError('Round has too few tasks for %d dispatcher shards' % len(quotas))
    keys = np.concatenate([(np.arange(n) + .5) / n for n in quotas])
    owner = np.concatenate([np.full(n, i, dtype=np.int16) for i, n in enumerate(quotas)])
    return owner[np.lexsort((owner, keys))]


def assign_hosts(counts, shards):
    """Whole worker nodes to shards, balancing worker counts (largest first, ties to the lower shard)."""
    hosts = list(counts)
    if len(hosts) < shards:
        raise QueueError('Fewer worker nodes (%d) than dispatcher shards (%d)' % (len(hosts), shards))
    load = [0] * shards; where = {}
    for host in sorted(hosts, key=lambda h: (-counts[h], hosts.index(h))):
        target = min(range(shards), key=lambda i: (load[i], i))
        where[host] = target; load[target] += counts[host]
    return where


def materialize(root, weights):
    """Write one queue root per shard under ``root/shards``. The round root itself is not touched."""
    import numpy as np
    root = Path(root)
    manifest = _read(root / 'manifest.json'); parent_qid = digest(manifest)
    if _read(root / 'queue-id.json')['queue_id'] != parent_qid:
        raise QueueError('Round manifest changed')
    order = array('I'); order.frombytes((root / 'remaining.u32').read_bytes())
    if sys.byteorder != 'little': order.byteswap()
    if len(order) != manifest['count']:
        raise QueueError('Remaining count mismatch')
    order = np.frombuffer(order, dtype='<u4') if len(order) else np.zeros(0, dtype='<u4')
    owner = shard_assignment(len(order), weights)
    (root / 'shards').mkdir(exist_ok=False)
    result = []
    for index in range(len(weights)):
        directory = root / 'shards' / ('s%d' % index); directory.mkdir()
        part = np.ascontiguousarray(order[owner == index], dtype='<u4')
        part.tofile(directory / 'remaining.u32')
        shard = {**manifest, 'count': int(len(part)), 'remaining_sha256': file_digest(directory / 'remaining.u32'),
                 'shard': {'index': index, 'of': len(weights), 'weights': [int(w) for w in weights],
                           'parent_queue_id': parent_qid}}
        atomic_json(directory / 'parent-manifest.json', manifest)
        atomic_json(directory / 'manifest.json', shard)
        atomic_json(directory / 'queue-id.json', {'queue_id': digest(shard)})
        result.append({'index': index, 'root': str(directory), 'count': int(len(part))})
    return result


def resolve_shard(launch, hostname):
    """A worker's launch description with the readiness file and fault directory of the shard serving its node."""
    try:
        shard = launch['shards'][launch['shard_of_host'][hostname]]
    except (KeyError, IndexError, TypeError) as exc:
        raise QueueError('Worker node %s has no dispatcher shard' % hostname) from exc
    return {**launch, **shard}


# ------------------------------------------------------------------ journals
def merge_journals(root, *, block=8 * 1024 * 1024):
    """Concatenate the shard journals into ``root/results.jsonl`` (once; a torn last line of a shard is dropped).

    The result is written to a temporary file, flushed, and linked into place, so a crash leaves either nothing or
    the complete file; running it again is harmless. Returns the report, or None if there was nothing to do.
    """
    root = Path(root); target = root / 'results.jsonl'
    directories = sorted((root / 'shards').glob('s[0-9]*'), key=lambda p: int(p.name[1:]))
    if target.exists() or not directories:
        return None
    temporary = root / ('.results-merge-' + uuid.uuid4().hex + '.tmp')
    report = {'shards': []}
    try:
        with temporary.open('xb') as out:
            for directory in directories:
                source = directory / 'results.jsonl'
                lines = size = 0; carry = b''
                if source.exists():
                    with source.open('rb') as handle:
                        while True:
                            data = handle.read(block)
                            if not data:
                                break
                            data = carry + data
                            cut = data.rfind(b'\n') + 1
                            out.write(data[:cut]); lines += data.count(b'\n', 0, cut); size += cut
                            carry = data[cut:]
                report['shards'].append({'shard': directory.name, 'lines': lines, 'bytes': size, 'dropped_tail_bytes': len(carry)})
            out.flush(); os.fsync(out.fileno())
        os.link(temporary, target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    sync_directory(root)
    report['lines'] = sum(item['lines'] for item in report['shards'])
    atomic_json(root / 'shards-merged.json', report)
    return report


def heal_round(root):
    """Merge the journals of a sharded round whose controller ended before it could (idempotent; refuses live shards)."""
    root = Path(root)
    directories = sorted((root / 'shards').glob('s[0-9]*'), key=lambda p: int(p.name[1:]))
    if not directories or (root / 'results.jsonl').exists():
        return None
    from contextlib import ExitStack
    with ExitStack() as locks:
        locks.enter_context(file_lock(root / 'dispatcher.lock'))
        for directory in directories:
            locks.enter_context(file_lock(directory / 'dispatcher.lock'))
        return merge_journals(root)


# ------------------------------------------------------------------ statistics
def aggregate_stats(parts):
    """One fleet-wide stats dict from the per-shard ones (same keys the round result and the tail monitor use)."""
    stats = {key: sum(part[key] for part in parts) for key in SUM_KEYS}
    fleet = max(1, stats['fleet'])
    batches = stats['leased_batches']
    known = lambda key: None if any(part[key] is None for part in parts) else [part[key] for part in parts]
    work, tails = known('remaining_work_seconds'), known('predicted_tail_seconds')
    reasons = [part['drain_reason'] for part in parts if part['drain_reason']]
    stats.update(
        phase=parts[0]['phase'], paused=all(part['paused'] for part in parts), queue_id=parts[0]['queue_id'],
        cells_per_batch=sum(part['cells_per_batch'] * part['leased_batches'] for part in parts) / batches if batches else 0.,
        results_per_fsync=stats['commits'] / stats['fsync_count'] if stats['fsync_count'] else 0.,
        validation_pool_failed=any(part['validation_pool_failed'] for part in parts),
        binding_ready=all(part['binding_ready'] for part in parts),
        compute_busy_fraction=stats['computing_workers'] / fleet,
        effective_busy_fraction=sum(part['effective_busy_fraction'] * part['fleet'] for part in parts) / fleet,
        remaining_work_seconds=sum(work) if work is not None else None,
        predicted_tail_seconds=max(tails) if tails is not None else None,
        draining=any(part['draining'] for part in parts), drain_reason=reasons[0] if reasons else None,
        oldest_active_heartbeat_age_seconds=max(part['oldest_active_heartbeat_age_seconds'] for part in parts),
        shards=len(parts))
    return stats


# ------------------------------------------------------------------ one shard, in its own process
def _handle_faults(directory, dispatcher):
    for path in directory.glob('*.json'):
        fault = _read(path)
        if fault.get('queue_id') != dispatcher.queue_id:
            raise QueueError('Worker fault queue changed')
        dispatcher.worker_fault(fault['worker'], fault['code'], fault.get('message', ''))


def _sampled(stats):
    """The shard's own CPU seconds and clock at the moment of the sample: exact per-shard rates need no guesswork."""
    return {**stats, 'process_cpu_seconds': time.process_time(), 'sampled_at': time.time()}


def shard_main(spec, conn):
    """Child process entry: report failures to the parent instead of dying silently."""
    try:
        _serve(spec, conn)
    except BaseException as exc:                                # noqa: BLE001 - the parent decides what a failure means
        try:
            conn.send(('error', '%s: %s' % (type(exc).__name__, exc)))
        except (OSError, ValueError):
            pass
        raise


def _serve(spec, conn):
    from .direct_success_queue import FlatDispatcher, write_progress
    from .queue_readiness import publish_ready
    from .queue_service import make_server, raise_nofile_limit
    policy = validate_policy(spec['policy']); binding = spec['service_binding']
    if binding is not None:                                     # before any thread or child process exists
        os.sched_setaffinity(0, set(binding['dispatcher_cpus']) if policy['dispatcher_smt'] else {binding['dispatcher_cpu']})
    shard_dir = Path(spec['attempt']) / ('shard-%d' % spec['index']); faults = shard_dir / 'faults'
    faults.mkdir(parents=True, exist_ok=True)
    with FlatDispatcher(Path(spec['root']), fleet=spec['fleet'], policy=policy, cost_profile=spec['cost_profile'],
                        work_seconds=spec['work_seconds'], service_binding=binding) as dispatcher:
        if binding is not None:
            atomic_json(shard_dir / 'service-binding.json', {**binding, 'dispatcher_affinity': sorted(os.sched_getaffinity(0)),
                                                             'validators': dispatcher.validation_pool.binding_proof})
        context = None
        if spec['tls']:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER); context.minimum_version = ssl.TLSVersion.TLSv1_2
            context.load_cert_chain(spec['cert'], spec['key'])
        if policy['max_connections'] > 128:
            raise_nofile_limit(policy['max_connections'])
        server = make_server(dispatcher, token=Path(spec['token_file']).read_text(), host=spec['bind_host'], port=0,
                             tls_context=context, threaded_tls_handshake=bool(context),
                             max_connections=policy['max_connections'], keepalive_idle_seconds=policy['keepalive_idle_seconds'],
                             max_submissions=policy['max_submissions'], keepalive=policy['rpc_keepalive'],
                             fast_http=policy['dispatcher_fast_path'])
        serving = threading.Thread(target=server.serve_forever); serving.start()
        try:
            publish_ready(shard_dir / 'ready.json', dispatcher, host=spec['host'], port=server.server_port,
                          tls=bool(context), generation=spec['generation'])
            conn.send(('ready', {'queue_id': dispatcher.queue_id, 'epoch': dispatcher.epoch, 'port': server.server_port}))
            progress_errors = 0; stop = False
            while not stop:
                _handle_faults(faults, dispatcher)
                progress_errors = write_progress(shard_dir / 'progress.json', dispatcher, server,
                                                 previous_errors=progress_errors, monitor=None)
                conn.send(('stats', _sampled(dispatcher.stats()), server.connection_stats()))
                if time.time() >= spec['work_deadline'] and not dispatcher.draining:
                    atomic_json(shard_dir / 'drain-intent.json', {'reason': 'allocation_deadline', 'observed_at': time.time()})
                    dispatcher.drain('allocation_deadline')
                try:
                    if conn.poll(spec['poll_seconds']):
                        command = conn.recv()
                        if command[0] == 'drain':
                            dispatcher.drain(command[1])
                        elif command[0] == 'stop':
                            stop = True
                except (EOFError, OSError):
                    stop = True                                 # the parent is gone: nobody will collect this shard
        finally:
            server.shutdown(); serving.join(); server.server_close()
        _handle_faults(faults, dispatcher)                      # a last worker may have written its fault after the final poll
        conn.send(('final', _sampled(dispatcher.stats()), server.connection_stats()))
    try:
        import resource                          # Linux: peak memory of this shard and of its largest validator
        usage = {'max_rss_kb_dispatcher': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 'max_rss_kb_largest_validator': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}
    except ImportError:
        usage = None
    conn.send(('closed', usage))


class ShardSet:
    """The shard processes of one round as seen by the controller: start, observe, drain, stop."""

    def __init__(self, specs, *, context=None):
        self.context = context or multiprocessing.get_context('spawn')
        self.specs = specs; self.count = len(specs)
        self.processes = []; self.conns = []
        self.stats = [None] * self.count; self.rpc = [None] * self.count
        self.final = [None] * self.count; self.closed = [False] * self.count
        self.ready = [None] * self.count; self.usage = [None] * self.count
        for spec in specs:
            parent, child = self.context.Pipe()
            process = self.context.Process(target=shard_main, args=(spec, child), name='dispatcher-shard-%d' % spec['index'])
            process.start(); child.close()
            self.processes.append(process); self.conns.append(parent)

    def pump(self, wait=0.):
        """Read everything the shards have sent; raise if one failed or died without finishing."""
        for i, conn in enumerate(self.conns):
            try:
                while conn.poll(wait if not self.closed[i] else 0):
                    kind, *rest = conn.recv()
                    if kind == 'ready': self.ready[i] = rest[0]
                    elif kind == 'stats': self.stats[i], self.rpc[i] = rest
                    elif kind == 'final': self.final[i] = rest[0]; self.stats[i], self.rpc[i] = rest
                    elif kind == 'closed': self.closed[i] = True; self.usage[i] = rest[0]
                    elif kind == 'error': raise QueueError('Dispatcher shard %d failed: %s' % (i, rest[0]))
                    wait = 0.
            except (EOFError, OSError):
                pass
            if not self.closed[i] and not self.processes[i].is_alive():
                raise QueueError('Dispatcher shard %d exited unexpectedly (exit code %s)' % (i, self.processes[i].exitcode))

    def wait_ready(self, timeout):
        until = time.monotonic() + timeout
        while any(item is None for item in self.ready):
            self.pump(.2)
            if time.monotonic() > until:
                raise TimeoutError('Dispatcher shards did not become ready; do not admit workers')
        return self.ready

    def send_all(self, message):
        for conn in self.conns:
            try:
                conn.send(message)
            except (OSError, ValueError):
                pass

    def stop(self, timeout=180.):
        """Ask every shard to shut down (its dispatcher fsyncs and closes the journal), then reap them."""
        self.send_all(('stop',))
        until = time.monotonic() + timeout
        while not all(self.closed) and time.monotonic() < until:
            try:
                self.pump(.2)
            except QueueError:
                break
        for process in self.processes:
            process.join(max(0., min(60., until - time.monotonic())))
            if process.is_alive():
                process.terminate(); process.join(15)
            if process.is_alive():
                process.kill(); process.join(15)

    def snapshot(self):
        return [part for part in (self.final if all(item is not None for item in self.final) else self.stats)]


# ------------------------------------------------------------------ the sharded round
def run_sharded(root, repo, old, workers, validate_only=False, *, max_seconds=172800, policy=None, cost_profile=None,
                allocation=None, continuation_allowed=False, layout=None, shard_tls=True, poll_seconds=15.):
    from .direct_success_queue import allocation_start_time, merge as final_merge
    from .service_binding import service_layout, worker_step
    policy = validate_policy(policy)
    shards = int((allocation or {}).get('dispatcher_shards', 1)); validators = policy['validation_processes']
    if shards < 2 or not validators:
        raise QueueError('A sharded round needs at least two shards and the reserved service step')
    entered = time.time()
    job_started, elapsed_source, elapsed_evidence = allocation_start_time(entered, max_seconds)
    job_end = min(job_started + max_seconds, float(os.environ.get('SLURM_JOB_END_TIME', job_started + max_seconds)))
    if not math.isfinite(job_end) or job_end <= entered:
        raise QueueError('Allocation has no time remaining')
    work_deadline = max(entered, job_end - policy['drain_grace_seconds'])
    service_slots = shards * (1 + validators)
    if allocation.get('controller_task_slots') != service_slots:
        raise QueueError('Dispatcher shard CPU reservation missing from allocation')
    layout = layout or service_layout(validators, shards)
    if layout['job_task_slots'] < workers + service_slots:
        raise QueueError('Missing worker/controller/validator task reservations')
    root = Path(root)
    manifest = _read(root / 'manifest.json')
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != manifest['identity']['cell_spec']['git_commit']:
        raise QueueError('Frozen scientific code changed')
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip():
        raise QueueError('Frozen checkout dirty')
    control = root / 'control'; control.mkdir(exist_ok=True, mode=0o700)
    with file_lock(control / 'round.lock'), file_lock(root / 'dispatcher.lock'):
        generation = uuid.uuid4().hex; attempt = control / generation; attempt.mkdir(mode=0o700)
        host = socket.gethostname(); token = attempt / 'token'; token.write_text(uuid.uuid4().hex + uuid.uuid4().hex)
        token.chmod(0o600); cert = attempt / 'ca.crt'; key = attempt / 'server.key'
        if shard_tls:
            cert_days = str(max(7, (int(max_seconds) + 86399) // 86400 + 1))
            subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', cert_days,
                            '-keyout', str(key), '-out', str(cert), '-subj', '/CN=' + host,
                            '-addext', 'subjectAltName=DNS:' + host + ',DNS:' + socket.getfqdn()],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            key.chmod(0o600)
        # Which node serves which shard follows from where the workers will actually run.
        step, step_env = worker_step(allocation, layout, workers, attempt / 'worker-hosts.txt')
        counts = {}
        for name in (attempt / 'worker-hosts.txt').read_text().split():
            counts[name] = counts.get(name, 0) + 1
        host_shard = assign_hosts(counts, shards)
        fleets = [sum(n for name, n in counts.items() if host_shard[name] == i) for i in range(shards)]
        slices = materialize(root, fleets)
        total_tasks = sum(item['count'] for item in slices)
        work = (allocation or {}).get('work_seconds')
        claim_limit = policy['max_claimed_tasks']
        limits = split_integer(claim_limit, [item['count'] for item in slices]) if claim_limit is not None else [None] * shards
        specs = []
        for item in slices:
            i = item['index']
            binding = {**layout, **layout['shards'][i]} if layout.get('shards') else layout
            binding = None if layout.get('unbound') else {k: v for k, v in binding.items() if k != 'shards'}
            specs.append(dict(index=i, root=item['root'], fleet=fleets[i], cost_profile=cost_profile,
                              work_seconds=None if work is None else work * item['count'] / total_tasks,
                              policy={**policy, 'max_claimed_tasks': limits[i]}, service_binding=binding,
                              attempt=str(attempt), token_file=str(token), cert=str(cert), key=str(key), tls=shard_tls,
                              host=host if shard_tls else '127.0.0.1', bind_host='0.0.0.0' if shard_tls else '127.0.0.1', generation=generation, work_deadline=work_deadline,
                              poll_seconds=poll_seconds))
        shard_set = ShardSet(specs)
        child = None; stats = None; report = None; drain_reason = None; drained = False
        try:
            shard_set.wait_ready(900.)
            launch = dict(queue=str(root.resolve()), queue_id=shard_set.ready[0]['queue_id'], repo=str(Path(repo).resolve()),
                control=str(control.resolve()), generation=generation, token_file=str(token.resolve()),
                ca_file=str(cert.resolve()), ready_file=str((attempt / 'shard-0' / 'ready.json').resolve()),
                shards=[{'ready_file': str((attempt / ('shard-%d' % i) / 'ready.json').resolve()),
                         'fault_dir': str((attempt / ('shard-%d' % i) / 'faults').resolve())} for i in range(shards)],
                shard_of_host=host_shard, shard_fleets=fleets,
                job_id=os.environ['SLURM_JOB_ID'], source_commit=commit, workers=workers,
                recover_stale_leases=True, max_seconds=max_seconds, deadline_epoch=work_deadline,
                protocol_version=2, fault_dir=str((attempt / 'shard-0' / 'faults').resolve()),
                rpc_keepalive=policy['rpc_keepalive'], protocol_metrics=policy['protocol_metrics'],
                node_relay=policy['node_relay'], node_relay_idle_seconds=round(.8 * policy['keepalive_idle_seconds'], 3),
                protocol_metrics_sample_modulo=policy['protocol_metrics_sample_modulo'],
                require_cpu_binding=True, heartbeat_aggregate_seconds=policy['heartbeat_aggregate_seconds'],
                startup_jitter_seconds=min(300., workers / 100.))
            atomic_json(attempt / 'launch.json', launch); atomic_json(control / 'latest.json', launch)
            shard_set.pump(); first = [s for s in shard_set.stats]
            atomic_json(attempt / 'admission.json', {'shards': slices, 'fleets': fleets, 'sqlite': False,
                        'stats': aggregate_stats(first) if all(first) else None})

            def interrupted(signum, frame): raise InterruptedError('Interrupted ' + str(signum))
            signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
            child = subprocess.Popen(step + ['--kill-on-bad-exit=1',
                '--output=' + str(attempt / 'worker-%t.out'), '--error=' + str(attempt / 'worker-%t.err'),
                sys.executable, '-m', 'aleatoric_nk_grid.slurm_queue_round', 'worker', '--launch', str(attempt / 'launch.json')],
                env=step_env)
            monitor = TailMonitor(policy, started=job_started); drain_started = None
            while child.poll() is None:
                shard_set.pump()
                if all(shard_set.stats):
                    aggregate = aggregate_stats(shard_set.stats); now = time.time()
                    observation = monitor.observe(aggregate, now=now, allocation=allocation,
                                                  continuation_allowed=continuation_allowed)
                    if observation.get('idle_alert') and monitor.samples == policy['idle_samples']:
                        import json
                        print('Scheduler idle fleet: ' + json.dumps(observation), file=sys.stderr, flush=True)
                    if observation.get('drain_recommended') and not drained:
                        atomic_json(attempt / 'drain-intent.json', {'reason': 'costed_tail', 'observed_at': now, **observation})
                        shard_set.send_all(('drain', 'costed_tail')); drained = True; drain_reason = 'costed_tail'
                    if aggregate['draining'] and not drained:
                        # Draining is fleet-wide: a fault, the deadline or an economic drain in one shard ends all of them.
                        drain_reason = aggregate['drain_reason']
                        shard_set.send_all(('drain', drain_reason)); drained = True
                    atomic_json(attempt / 'progress.json', {'stats': aggregate, 'observed_at': now, 'efficiency': observation,
                                                            'shards': [{'stats': s, 'rpc': r} for s, r in zip(shard_set.stats, shard_set.rpc)]})
                    if now >= work_deadline and not drained:
                        atomic_json(attempt / 'drain-intent.json', {'reason': 'allocation_deadline', 'observed_at': now})
                        shard_set.send_all(('drain', 'allocation_deadline')); drained = True; drain_reason = 'allocation_deadline'
                    if aggregate['draining'] or drained:
                        if drain_started is None: drain_started = now
                        grace_expired = (drain_reason != 'costed_tail' and now - drain_started >= policy['drain_grace_seconds'])
                        if grace_expired or now >= job_end - 5:
                            break
                time.sleep(min(poll_seconds, max(.1, job_end - time.time() - 5)))
        finally:
            try:
                if child is not None and child.poll() is None:
                    child.terminate()
                    try: child.wait(timeout=90)
                    except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=30)
            finally:
                shard_set.stop()
            report = merge_journals(root)
            parts = [part for part in shard_set.snapshot()]
            if all(parts):
                stats = aggregate_stats(parts)
                bounded_complete = (policy['max_claimed_tasks'] is not None and stats['paused'] and not stats['leased']
                                    and not stats['failed'] and child is not None and child.returncode == 0)
                atomic_json(control / 'round-result.json', {'stats': stats, 'worker_exit': child.returncode if child else None,
                    'complete': stats['done'] == stats['total'], 'generation': generation,
                    'state': 'complete' if stats['done'] == stats['total'] else
                        ('bounded_complete' if bounded_complete else ('drained' if stats['draining'] else 'incomplete')),
                    'job_id': os.environ['SLURM_JOB_ID'], 'elapsed_seconds': max(0., time.time() - job_started),
                    'elapsed_source': elapsed_source, 'elapsed_evidence': elapsed_evidence,
                    'job_started_epoch': job_started, 'queue_id': stats['queue_id'],
                    'allocation_cpu': (allocation or {}).get('allocated_cpu_bound'),
                    'drain_reason': stats['drain_reason'], 'shards': parts, 'shard_fleets': fleets, 'shard_memory': shard_set.usage,
                    'journal_merge': report})
    if stats is None:
        raise QueueError('Dispatcher shards left no final statistics; results retained in the shard journals')
    if stats['done'] != stats['total'] and not (validate_only and (stats['draining'] or bounded_complete)):
        raise QueueError('Incomplete round; retain result receipts for next direct success scan')
    if not validate_only:
        final_merge(root, old)
