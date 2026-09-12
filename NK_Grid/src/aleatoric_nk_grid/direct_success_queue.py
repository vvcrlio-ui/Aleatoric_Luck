"""One success scan and compact remaining ordinals; no SQLite or lease replay.

Operational launcher outside the frozen scientific checkout. Numerical workers
continue to execute the original CellExecutionSpec and original worker module.
Only successful/failed result receipts are durable; leases are generation-local.
"""
import argparse
from array import array
from collections import deque
from dataclasses import asdict
import hashlib
import heapq
import json
import os
from pathlib import Path
import signal
import socket
import ssl
import subprocess
import sys
import threading
import time
import uuid

from aleatoric_nk_grid.shared_queue import (
    Dispatcher, LeaseLostError, ModelTask, QueueError, atomic_json, canonical, digest, file_digest, file_lock)
from aleatoric_nk_grid.pending_resume import Design
from aleatoric_nk_grid.result_migration import validate_scientific_result


def read(path):
    return json.loads(Path(path).read_bytes())


def task_at(design, ordinal):
    ordinal, m = divmod(int(ordinal), len(design.models))
    ordinal, r = divmod(ordinal, len(design.repeats))
    k, n = divmod(ordinal, len(design.ns))
    seed, draw = design.repeats[r]
    return ModelTask(seed, draw, design.ns[n], design.ks[k], design.models[m])


def prepare(base, output):
    """Reuse hash-bound prior scan; scan only subsequent result journal once."""
    if (base / 'manifest.json').exists():
        return prepare_flat(base, output)
    import numpy as np
    from aleatoric_nk_grid.scheduler_cost import CostEstimator
    started = time.monotonic()
    ready = read(base / 'ready.json')
    bm = read(base / 'base-manifest.json')
    if file_digest(base / 'base-manifest.json') != ready['base_manifest_sha256']:
        raise QueueError('Base manifest changed')
    bits = (base / 'completed.bits').read_bytes()
    if hashlib.sha256(bits).hexdigest() != bm['bits_sha256']:
        raise QueueError('Prior successful-key scan changed')
    design = Design(bm['cell_spec']); design.bits[:] = bits
    if sum(int(b).bit_count() for b in bits) != ready['old_valid_unique']:
        raise QueueError('Prior success count mismatch')
    for item in bm['workers']:
        if Path(item['wal_path']).stat().st_size != item['captured_bytes']:
            raise QueueError('Stopped base WAL size changed')
    output.mkdir(exist_ok=False, parents=True)
    accepted = {}; failed = 0; sequence = 0; previous = '0' * 64
    source = base / 'queue/events.jsonl'
    source_size = source.stat().st_size
    sha = hashlib.sha256(); tail_bytes = 0
    with source.open('rb') as handle, (output / 'prior-success.jsonl').open('xb') as out:
        while True:
            line = handle.readline(2 * 1024 * 1024 + 1)
            if not line:
                break
            sha.update(line)
            if not line.endswith(b'\n'):
                if len(line) > 2 * 1024 * 1024:
                    raise QueueError('Oversized event')
                tail_bytes = len(line)
                break
            frame = json.loads(line); body = frame['body']
            if (frame['sha256'] != digest(body) or body['sequence'] != sequence
                    or body['previous'] != previous or body['queue_id'] != ready['queue_id']):
                raise QueueError('Source journal integrity failure')
            sequence += 1; previous = frame['sha256']
            event = body['event']
            if event['kind'] not in ('result', 'import'):
                continue  # Do not replay lease/heartbeat/restart/pause events.
            row = event['result']; ordinal = design.ordinal(row)
            if task_at(design, ordinal).id != event['id']:
                raise QueueError('Source result key mismatch')
            if not validate_scientific_result(row, task_kind='regression'):
                failed += 1; continue
            if row['algorithm_version'] != bm['cell_spec']['algorithm_version']:
                raise QueueError('Source scientific identity mismatch')
            fingerprint = digest(row)
            if design.contains(ordinal):
                if accepted.get(ordinal) != fingerprint:
                    raise QueueError('Conflicting/overlapping successful key')
                continue
            design.mark(ordinal); accepted[ordinal] = fingerprint
            out.write(canonical({'result': row, 'origin': {'queue_id': ready['queue_id'],
                'sequence': body['sequence'], 'event_sha256': frame['sha256']}}) + b'\n')
        out.flush(); os.fsync(out.fileno())
        if handle.tell() != source_size or source.stat().st_size != source_size:
            raise QueueError('Source journal changed during scan')
    (output / 'completed.bits').write_bytes(design.bits)
    success = np.unpackbits(np.frombuffer(design.bits, dtype=np.uint8), bitorder='little')[:design.count]
    remaining = np.flatnonzero(success == 0).astype('<u4')
    estimator = CostEstimator(profile=bm['cost_profile'])
    # Cost lookup has only K*N*model entries. Repeats share the same estimate.
    costs = np.array([estimator.estimate(m, n, k) for k in design.ks
        for n in design.ns for m in design.models], dtype=np.float64)
    models = len(design.models); repeats = len(design.repeats)
    lookup = (remaining // (models * repeats)) * models + remaining % models
    remaining = remaining[np.argsort(-costs[lookup], kind='stable')]
    remaining.tofile(output / 'remaining.u32')
    manifest = {'format': 'direct-success-bitmap-v1',
        'identity': {'cell_spec': bm['cell_spec'], 'base_manifest_sha256': ready['base_manifest_sha256'],
            'prior_success_sha256': file_digest(output / 'prior-success.jsonl')},
        'count': len(remaining), 'lease_seconds': 300., 'max_attempts': 5,
        'remaining_sha256': file_digest(output / 'remaining.u32'),
        'completed_sha256': file_digest(output / 'completed.bits'),
        'base': str(base.resolve()), 'old_valid_unique': ready['old_valid_unique'],
        'prior_new_success': len(accepted), 'source_failed': failed,
        'source_events': sequence, 'source_bytes': source_size,
        'source_sha256': sha.hexdigest(), 'ignored_incomplete_tail_bytes': tail_bytes,
        'expected_total': design.count}
    if ready['old_valid_unique'] + len(accepted) + len(remaining) != design.count:
        raise QueueError('Complement count mismatch')
    atomic_json(output / 'manifest.json', manifest)
    atomic_json(output / 'queue-id.json', {'queue_id': digest(manifest)})
    atomic_json(output / 'prepared.json', {'seconds': time.monotonic() - started,
        'old_success': ready['old_valid_unique'], 'new_success': len(accepted),
        'remaining': len(remaining), 'queue_id': digest(manifest), 'sqlite': False})


def prepare_flat(base, output):
    """Carry prior success bits forward and scan this stopped round once."""
    import numpy as np
    started = time.monotonic()
    parent = read(base / 'manifest.json'); parent_qid = digest(parent)
    if read(base / 'queue-id.json')['queue_id'] != parent_qid:
        raise QueueError('Parent identity changed')
    if parent['format'] != 'direct-success-bitmap-v1':
        raise QueueError('Unsupported parent format')
    bits = (base / 'completed.bits').read_bytes()
    if hashlib.sha256(bits).hexdigest() != parent['completed_sha256']:
        raise QueueError('Parent successful bits changed')
    design = Design(parent['identity']['cell_spec']); design.bits[:] = bits
    if sum(int(x).bit_count() for x in bits) != parent['old_valid_unique'] + parent['prior_new_success']:
        raise QueueError('Parent success count mismatch')
    remaining_bytes = (base / 'remaining.u32').read_bytes()
    if hashlib.sha256(remaining_bytes).hexdigest() != parent['remaining_sha256']:
        raise QueueError('Parent remaining list changed')
    ordinals = np.frombuffer(remaining_bytes, dtype='<u4')
    if len(ordinals) != parent['count']: raise QueueError('Parent remaining count mismatch')
    membership = bytearray(len(bits))
    for ordinal in ordinals:
        ordinal = int(ordinal)
        if ordinal >= design.count or design.contains(ordinal):
            raise QueueError('Parent task complement changed')
        byte, shift = divmod(ordinal,8)
        if membership[byte] & (1 << shift): raise QueueError('Duplicate parent task')
        membership[byte] |= 1 << shift
    output.mkdir(exist_ok=False, parents=True)
    prior = base / 'prior-success.jsonl'; source = base / 'results.jsonl'
    source_size = source.stat().st_size
    fingerprints = {}; count = failed = tail_bytes = 0; sha = hashlib.sha256()
    with (output / 'prior-success.jsonl').open('xb') as out:
        prior_sha = hashlib.sha256()
        with prior.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                prior_sha.update(block); out.write(block)
        if prior_sha.hexdigest() != parent['identity']['prior_success_sha256']:
            raise QueueError('Parent prior-success export changed')
        with source.open('rb') as handle:
            while True:
                line = handle.readline(2 * 1024 * 1024 + 1)
                if not line: break
                sha.update(line)
                if not line.endswith(b'\n'):
                    if len(line) > 2 * 1024 * 1024: raise QueueError('Oversized result')
                    tail_bytes = len(line); break
                entry = json.loads(line); row = entry['result']; ordinal = design.ordinal(row)
                if entry['origin']['queue_id'] != parent_qid or entry['task_id'] != task_at(design,ordinal).id:
                    raise QueueError('Parent result provenance/key mismatch')
                if not membership[ordinal//8] & (1 << (ordinal%8)):
                    raise QueueError('Result outside parent pending design')
                if not validate_scientific_result(row, task_kind='regression'):
                    failed += 1; continue
                if row['algorithm_version'] != parent['identity']['cell_spec']['algorithm_version']:
                    raise QueueError('Parent scientific identity mismatch')
                fingerprint = digest(row)
                if ordinal in fingerprints:
                    if fingerprints[ordinal] != fingerprint: raise QueueError('Conflicting duplicate result')
                    continue
                fingerprints[ordinal] = fingerprint; design.mark(ordinal); count += 1
                out.write(line)
            if handle.tell() != source_size or source.stat().st_size != source_size:
                raise QueueError('Parent results changed during scan')
        out.flush(); os.fsync(out.fileno())
    (output / 'completed.bits').write_bytes(design.bits)
    success = np.unpackbits(np.frombuffer(design.bits,dtype=np.uint8),bitorder='little')[:design.count]
    remaining = ordinals[success[ordinals] == 0]
    remaining.tofile(output / 'remaining.u32')
    original_qid = read(Path(parent['base']) / 'ready.json')['queue_id']
    manifest = {**parent, 'count': len(remaining),
        'identity': {**parent['identity'], 'parent_manifest_sha256': parent_qid,
            'prior_success_sha256': file_digest(output/'prior-success.jsonl')},
        'remaining_sha256': file_digest(output/'remaining.u32'),
        'completed_sha256': file_digest(output/'completed.bits'),
        'prior_new_success': parent['prior_new_success'] + count,
        'prior_source_queue_ids': list(dict.fromkeys(parent.get('prior_source_queue_ids',[original_qid]) + [parent_qid])),
        'parent_root': str(base.resolve()), 'parent_result_sha256': sha.hexdigest(),
        'parent_result_bytes': source_size, 'parent_failed': failed,
        'ignored_incomplete_tail_bytes': tail_bytes}
    if manifest['old_valid_unique'] + manifest['prior_new_success'] + manifest['count'] != design.count:
        raise QueueError('Resumed complement count mismatch')
    atomic_json(output/'parent-manifest.json',parent)
    atomic_json(output/'manifest.json',manifest)
    atomic_json(output/'queue-id.json',{'queue_id':digest(manifest)})
    atomic_json(output/'prepared.json',{'seconds':time.monotonic()-started,'sqlite':False,
        'previous_success': parent['old_valid_unique']+parent['prior_new_success'],
        'new_success':count,'all_success':manifest['old_valid_unique']+manifest['prior_new_success'],
        'remaining':len(remaining),'queue_id':digest(manifest)})


class FlatDispatcher(Dispatcher):
    """Compact pending list, bounded active leases, append-only result receipts."""
    def __init__(self, root, *, clock=time.time):
        self.root = Path(root); self.clock = clock
        self.mutex = threading.RLock(); self.closed = self.poisoned = False
        self.submit_mutex = threading.RLock()
        self._owner = file_lock(self.root / 'dispatcher.lock'); self._owner.__enter__()
        self.journal = None
        try:
            self.manifest = read(self.root / 'manifest.json'); self.queue_id = digest(self.manifest)
            if read(self.root / 'queue-id.json')['queue_id'] != self.queue_id:
                raise QueueError('Manifest changed')
            if file_digest(self.root / 'remaining.u32') != self.manifest['remaining_sha256']:
                raise QueueError('Remaining list changed')
            self.design = Design(self.manifest['identity']['cell_spec'])
            self.order = array('I'); self.order.frombytes((self.root / 'remaining.u32').read_bytes())
            if sys.byteorder != 'little': self.order.byteswap()
            if len(self.order) != self.manifest['count']:
                raise QueueError('Remaining count mismatch')
            # A generation is started once. Resume creates a new success scan,
            # never replays old lease events or silently discards prior results.
            self.journal = (self.root / 'results.jsonl').open('xb', buffering=0)
            self.cursor = 0; self.retry = deque(); self.active = {}; self.by_worker = {}
            self.completed = {}; self.attempts = {}; self.exhausted = 0
            self.expiries = []; self.expired_leases = 0
            self.fsync_seconds = 0.; self.fsync_count = 0
            self.done = self.failed = 0; self.paused = False; self.epoch = uuid.uuid4().hex
        except BaseException:
            self.close(); raise

    def _reap(self):
        now = self.clock()
        while self.expiries and self.expiries[0][0] <= now:
            _, task_id, token = heapq.heappop(self.expiries)
            row = self.active.get(task_id)
            if row is None or row['token'] != token or row.get('committing'):
                continue
            if row['expiry'] > now:
                # Heartbeats update the row, not the heap: at most one queued
                # deadline per lease, without an O(active) scan on every claim.
                heapq.heappush(self.expiries, (row['expiry'], task_id, token))
                continue
            del self.active[task_id]; del self.by_worker[row['worker']]
            self.expired_leases += 1
            if row['attempt'] >= self.manifest['max_attempts']:
                self.exhausted += 1
            else:
                self.retry.append(row['ordinal'])

    def _check_lease(self, task_id, token, worker):
        row = self.active.get(task_id)
        if (row is None or row['token'] != token or row['worker'] != worker
                or (row['expiry'] <= self.clock() and not row.get('committing'))):
            raise LeaseLostError('Lease is stale, expired or owned by another worker')
        return row

    def _get(self, task_id):
        row = self.active.get(task_id) or self.completed.get(task_id)
        if row is None: raise QueueError('Unknown/nonactive task')
        return row

    def claim(self, worker, *, cached_cells=()):
        if not isinstance(worker, str) or not 0 < len(worker) <= 256:
            raise QueueError('Invalid worker')
        with self.mutex:
            if self.closed or self.poisoned: raise QueueError('Dispatcher stopped')
            self._reap()
            if worker in self.by_worker:
                return self._lease_payload(self.active[self.by_worker[worker]])
            if self.paused: return {'state': 'paused'}
            if self.retry: ordinal = self.retry.popleft()
            elif self.cursor < len(self.order):
                ordinal = self.order[self.cursor]; self.cursor += 1
            else:
                return {'state': 'wait' if self.active else ('blocked' if self.failed or self.exhausted else 'complete')}
            task = task_at(self.design, ordinal); task_id = task.id
            attempt = self.attempts.get(ordinal, 0) + 1; self.attempts[ordinal] = attempt
            row = {'id': task_id, 'ordinal': ordinal, 'task': canonical(asdict(task)).decode(),
                'cell': task.cell, 'state': 'leased', 'worker': worker, 'attempt': attempt,
                'token': self.epoch + ':' + uuid.uuid4().hex,
                'expiry': self.clock() + self.manifest['lease_seconds']}
            self.active[task_id] = row; self.by_worker[worker] = task_id
            heapq.heappush(self.expiries, (row['expiry'], task_id, row['token']))
            return self._lease_payload(row)

    def heartbeat(self, task_id, token, worker):
        with self.mutex:
            if self.closed or self.poisoned: raise QueueError('Dispatcher stopped')
            row = self._check_lease(task_id, token, worker)
            row['expiry'] = self.clock() + self.manifest['lease_seconds']
            return {'expiry': row['expiry']}

    def submit(self, task_id, token, worker, result):
        # Serialize journal writes, but never hold the lease-state mutex across
        # shared-storage I/O. A validated submission pins its lease until durable
        # commit (or poison), so reaping cannot reassign a half-written result.
        with self.submit_mutex:
            with self.mutex:
                if self.closed or self.poisoned: raise QueueError('Dispatcher stopped')
                if task_id not in self.completed:
                    self._check_lease(task_id, token, worker)
                row = self.validate_result(task_id, result)
                payload = canonical(result).decode()
                if row['state'] in ('done', 'failed'):
                    if row['accepted_token'] == token and row['result'] == payload:
                        return {'accepted': True, 'duplicate': True}
                    if row['accepted_token'] != token:
                        raise LeaseLostError('Completed task belongs to another lease')
                    raise QueueError('Conflicting completed result')
                if result['status'] != 'failed':
                    validate_scientific_result(result, task_kind='regression')
                    if result.get('algorithm_version') != self.manifest['identity']['cell_spec']['algorithm_version']:
                        raise QueueError('Scientific identity changed')
                envelope = {'result': result, 'origin': {'queue_id': self.queue_id},
                    'token': token, 'task_id': task_id}
                raw = canonical(envelope) + b'\n'
                row['committing'] = True
            try:
                pending = memoryview(raw)
                while pending:
                    count = self.journal.write(pending)
                    if not count: raise OSError('Short result write')
                    pending = pending[count:]
                started = time.monotonic()
                os.fsync(self.journal.fileno())
                elapsed = time.monotonic() - started
            except BaseException:
                with self.mutex: self.poisoned = True
                raise
            with self.mutex:
                self.fsync_count += 1; self.fsync_seconds += elapsed
                row.pop('committing', None)
                row.update(state='failed' if result['status'] == 'failed' else 'done',
                    accepted_token=token, result=payload)
                del self.active[task_id]; del self.by_worker[worker]
                self.completed[task_id] = row
                if row['state'] == 'done': self.done += 1
                else: self.failed += 1
                return {'accepted': True, 'duplicate': False}

    def pause(self, paused=True):
        with self.mutex:
            self.paused = bool(paused); return self.stats()

    def stats(self):
        with self.mutex:
            return {'done': self.done, 'failed': self.failed, 'leased': len(self.active),
                'pending': len(self.order) - self.cursor + len(self.retry), 'exhausted': self.exhausted,
                'total': len(self.order), 'paused': self.paused, 'queue_id': self.queue_id,
                'expired_leases': self.expired_leases, 'fsync_count': self.fsync_count,
                'fsync_seconds': self.fsync_seconds}

    def close(self):
        with self.submit_mutex:
            with self.mutex:
                if self.closed: return
                self.closed = True
                if self.journal: self.journal.close()
                self._owner.__exit__(None, None, None)


def merge(root, old):
    """Use the existing strict final merger with an explicit queue bridge receipt."""
    from types import SimpleNamespace
    from aleatoric_nk_grid.direct_key_resume import merge as merge_original
    import shutil
    manifest = read(root / 'manifest.json'); base = Path(manifest['base'])
    original_qid = read(base / 'ready.json')['queue_id']
    target = root / 'final'; target.mkdir(exist_ok=False)
    for name in ('base-manifest.json', 'ready.json', 'completed.bits'):
        shutil.copyfile(base / name, target / name)
    # Bind actual source queues/hashes separately; original strict merger checks
    # key uniqueness, scientific rows, original WAL hashes and full 18M coverage.
    sources = [(root / 'prior-success.jsonl', manifest.get('prior_source_queue_ids',[original_qid])),
               (root / 'results.jsonl', [digest(manifest)])]
    if file_digest(sources[0][0]) != manifest['identity']['prior_success_sha256']:
        raise QueueError('Prior success export changed')
    bridge = []
    with (target / 'combined.jsonl').open('xb') as out:
        for path, allowed_qids in sources:
            sha = hashlib.sha256(); count = 0
            with path.open('rb') as source:
                for line in source:
                    sha.update(line); entry = json.loads(line)
                    expected_qid = entry['origin']['queue_id']
                    if expected_qid not in allowed_qids:
                        raise QueueError('Unexpected source queue')
                    if not validate_scientific_result(entry['result'], task_kind='regression'):
                        raise QueueError('Cannot finalize failed result')
                    out.write(canonical({'result': entry['result'], 'origin': {'queue_id': original_qid,
                        'source_queue_id': expected_qid, 'mapping': 'frozen-spec-success-key-bridge-v1'}}) + b'\n')
                    count += 1
            bridge.append({'path': str(path), 'queue_ids': allowed_qids, 'sha256': sha.hexdigest(), 'rows': count})
        out.flush(); os.fsync(out.fileno())
    atomic_json(target / 'queue-bridge.json', {'sources': bridge, 'target_queue': original_qid,
        'frozen_spec_sha256': digest(manifest['identity']['cell_spec']),
        'combined_sha256': file_digest(target / 'combined.jsonl')})
    merge_original(SimpleNamespace(output=target, old=old, new_results=target / 'combined.jsonl'))


def run(root, repo, old, workers, validate_only=False):
    from aleatoric_nk_grid.queue_service import make_server
    from aleatoric_nk_grid.queue_readiness import publish_ready
    if int(os.environ['SLURM_NTASKS']) < workers + 1:
        raise QueueError('Missing controller task')
    manifest = read(root / 'manifest.json')
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != manifest['identity']['cell_spec']['git_commit']:
        raise QueueError('Frozen scientific code changed')
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip():
        raise QueueError('Frozen checkout dirty')
    control = root / 'control'; control.mkdir(exist_ok=True, mode=0o700)
    with file_lock(control / 'round.lock'), FlatDispatcher(root) as dispatcher:
        generation = uuid.uuid4().hex; attempt = control / generation; attempt.mkdir(mode=0o700)
        host = socket.gethostname(); token = attempt / 'token'; token.write_text(uuid.uuid4().hex + uuid.uuid4().hex)
        token.chmod(0o600); cert = attempt / 'ca.crt'; key = attempt / 'server.key'
        subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '7',
            '-keyout', str(key), '-out', str(cert), '-subj', '/CN=' + host,
            '-addext', 'subjectAltName=DNS:' + host + ',DNS:' + socket.getfqdn()],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        key.chmod(0o600)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER); context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(cert, key)
        server = make_server(dispatcher, token=token.read_text(), host='0.0.0.0', port=0,
                             tls_context=context, threaded_tls_handshake=True)
        serving = threading.Thread(target=server.serve_forever, daemon=True); serving.start()
        ready_path = attempt / 'ready.json'
        publish_ready(ready_path, dispatcher, host=host, port=server.server_port, tls=True, generation=generation)
        launch = dict(queue=str(root.resolve()), queue_id=dispatcher.queue_id, repo=str(repo.resolve()),
            control=str(control.resolve()), generation=generation, token_file=str(token.resolve()),
            ca_file=str(cert.resolve()), ready_file=str(ready_path.resolve()),
            job_id=os.environ['SLURM_JOB_ID'], source_commit=commit, workers=workers,
            recover_stale_leases=True)
        atomic_json(attempt / 'launch.json', launch); atomic_json(control / 'latest.json', launch)
        atomic_json(attempt / 'admission.json', {'stats': dispatcher.stats(), 'sqlite': False})
        def interrupted(signum, frame): raise InterruptedError('Interrupted ' + str(signum))
        signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
        child = None
        try:
            child = subprocess.Popen(['srun', '--ntasks=' + str(workers), '--cpus-per-task=1',
                '--ntasks-per-core=1', '--distribution=cyclic', '--kill-on-bad-exit=1',
                '--output=' + str(attempt / 'worker-%t.out'), '--error=' + str(attempt / 'worker-%t.err'),
                sys.executable, '-m', 'aleatoric_nk_grid.slurm_queue_round', 'worker',
                '--launch', str(attempt / 'launch.json')])
            while child.poll() is None:
                atomic_json(attempt / 'progress.json', {'stats': dispatcher.stats(), 'observed_at': time.time()})
                time.sleep(15)
            stats = dispatcher.stats()
            atomic_json(control / 'round-result.json', {'stats': stats, 'worker_exit': child.returncode,
                'complete': stats['done'] == stats['total'], 'generation': generation})
            if child.returncode or stats['done'] != stats['total']:
                raise QueueError('Incomplete round; retain result receipts for next direct success scan')
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                try: child.wait(timeout=90)
                except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=30)
            server.shutdown(); server.server_close(); serving.join(timeout=10)
    if not validate_only:
        merge(root, old)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--base', type=Path)
    parser.add_argument('--repo', type=Path)
    parser.add_argument('--old', type=Path)
    parser.add_argument('--workers', type=int, default=21)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        owner_root = args.base if (args.base/'manifest.json').exists() else args.base/'queue'
        with file_lock(args.base / 'control/round.lock'), file_lock(owner_root / 'dispatcher.lock'):
            prepare(args.base, args.root)
    else: run(args.root, args.repo, args.old, args.workers, args.validate_only)
