"""One success scan and compact remaining ordinals; no SQLite or lease replay.

Operational launcher outside the frozen scientific checkout. Numerical workers
continue to execute the original CellExecutionSpec and original worker module.
Only successful/failed result receipts are durable; leases are generation-local.
"""
import argparse
from array import array
from collections import deque
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime
import errno
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import ssl
import struct
import subprocess
import sys
import threading
import time
import uuid

from aleatoric_nk_grid.shared_queue import (
    Dispatcher, LeaseLostError, MAX_BATCH_TASKS, MAX_SUBMISSIONS, ModelTask, QueueError,
    atomic_json, canonical, digest, file_digest, file_lock, transport_manifest)
from aleatoric_nk_grid.pending_resume import Design
from aleatoric_nk_grid.result_migration import validate_scientific_result
from aleatoric_nk_grid.scheduler_cost import CostEstimator
from aleatoric_nk_grid.scheduler_policy import TailMonitor, validate_policy


def read(path):
    return json.loads(Path(path).read_bytes())


class CompletedJournalIndex(Mapping):
    """Small durable-result locators; full task/result bodies live only in the journal.

    Publication happens after fsync. A lost-ACK replay reads and verifies precisely
    the originally committed bytes, without retaining them for the whole round.
    This index is generation-local; restart still uses the ordinary success scan.
    """
    record = struct.Struct('<QI32s')   # offset, byte length, original journal-line checksum

    def __init__(self, path, design, queue_id):
        self.entries = {}
        self.workers = {}  # RPC decoding creates fresh equal strings; keep one owner string per worker.
        self.design, self.queue_id = design, queue_id
        self.reader = Path(path).open('rb', buffering=0)
        self.read_lock = threading.Lock()

    @staticmethod
    def _key(task_id):
        if not isinstance(task_id, str) or len(task_id) != 64:
            return None
        try:
            key = bytes.fromhex(task_id)
        except ValueError:
            return None
        return key if key.hex() == task_id else None

    def __len__(self):
        return len(self.entries)

    def __iter__(self):
        return (key.hex() for key in self.entries)

    def __contains__(self, task_id):
        return self._key(task_id) in self.entries

    def owner(self, task_id):
        entry = self.entries.get(self._key(task_id))
        return entry[1] if entry is not None else None

    def add(self, task_id, offset, length, worker, checksum):
        key = self._key(task_id)
        if key is None or key in self.entries:
            raise QueueError('Invalid or repeated completed-result locator')
        owner = self.workers.setdefault(worker, worker)
        self.entries[key] = (self.record.pack(offset, length, checksum), owner)

    def __getitem__(self, task_id):
        entry = self.entries.get(self._key(task_id))
        if entry is None:
            raise KeyError(task_id)
        locator, worker = entry
        offset, length, checksum = self.record.unpack(locator)
        # Separate descriptor: duplicate reads must never move the append cursor.
        # Serialize seek/read for Windows as well as POSIX; normal commits do no reads.
        with self.read_lock:
            self.reader.seek(offset)
            raw = self.reader.read(length)
        if len(raw) != length or hashlib.sha256(raw).digest() != checksum:
            raise QueueError('Completed journal record changed or truncated')
        entry = json.loads(raw)
        result = entry['result']
        task = task_at(self.design, self.design.ordinal(result))
        if (entry['task_id'] != task_id or task.id != task_id
                or entry['origin']['queue_id'] != self.queue_id):
            raise QueueError('Completed journal record identity changed')
        return {'id': task_id, 'task': canonical(asdict(task)).decode(),
                'state': 'failed' if result['status'] == 'failed' else 'done',
                'worker': worker, 'accepted_token': entry['token'],
                'result': canonical(result).decode()}

    def close(self):
        self.reader.close()


def task_at(design, ordinal):
    if hasattr(design, 'task_at'):
        return design.task_at(int(ordinal))
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
    from .prediction_workflow import reject_unphased_cache
    reject_unphased_cache(bm['cell_spec'])
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
        'count': len(remaining), **transport_manifest(), 'max_attempts': 5,
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
    if parent.get('identity', {}).get('prediction_workflow'):
        raise QueueError('Prediction workflows must resume through cluster_scheduler and its all-plan phase barrier')
    from .prediction_workflow import reject_unphased_cache
    reject_unphased_cache(parent['identity']['cell_spec'])
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
    manifest = {**parent, 'count': len(remaining), **transport_manifest(),
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
    def __init__(self, root, *, clock=time.time, fleet=1, policy=None,
                 cost_profile=None, work_seconds=None, service_binding=None):
        if type(fleet) is not int or fleet < 1: raise QueueError('Invalid fleet size')
        self.fleet = fleet; self.policy = validate_policy(policy)
        from .protocol_metrics import ProtocolMetrics
        self.metrics = ProtocolMetrics(self.policy["protocol_metrics"])
        self.validation_pool = None
        self.service_binding = service_binding
        self.worker_bindings = {}
        self.worker_core_owners = {}
        self.binding_ready = service_binding is None
        self.estimator = CostEstimator(profile=cost_profile)
        if work_seconds is not None and (not math.isfinite(work_seconds) or work_seconds < 0):
            raise QueueError('Invalid remaining work estimate')
        self.remaining_work_seconds = work_seconds
        self.root = Path(root); self.clock = clock
        self.mutex = threading.RLock(); self.closed = self.poisoned = False
        self.submit_mutex = threading.RLock(); self.fsync_mutex = threading.RLock()
        self.submission_locks = {}  # Only currently submitting/waiting lease tokens.
        self.written_seq = self.synced_seq = self.commits = 0
        self._owner = file_lock(self.root / 'dispatcher.lock'); self._owner.__enter__()
        self.journal = None
        try:
            self.manifest = read(self.root / 'manifest.json'); self.queue_id = digest(self.manifest)
            if read(self.root / 'queue-id.json')['queue_id'] != self.queue_id:
                raise QueueError('Manifest changed')
            self.shard = self.manifest.get('shard')
            if self.shard is not None:
                # A shard serves a slice of its round's tasks. On the wire and in its journal it carries the round's
                # queue id, so the merged journals are exactly what an unsharded round would have written.
                parent = read(self.root / 'parent-manifest.json')
                if (digest(parent) != self.shard['parent_queue_id']
                        or {k: v for k, v in self.manifest.items() if k not in ('shard', 'count', 'remaining_sha256')}
                        != {k: v for k, v in parent.items() if k not in ('count', 'remaining_sha256')}):
                    raise QueueError('Shard does not belong to its round')
                self.queue_id = self.shard['parent_queue_id']
            workflow = self.manifest['identity'].get('prediction_workflow')
            if not workflow and file_digest(self.root / 'remaining.u32') != self.manifest['remaining_sha256']:
                raise QueueError('Remaining list changed')
            from .prediction_workflow import design_for, ResultCacheValidator
            self.design = design_for(self.manifest['identity'])
            self.result_cache_validator = ResultCacheValidator(self.manifest['identity'],
                                                               fast_reads=self.policy['validation_fast_reads'])
            if workflow:
                from .prediction_workflow import _read_remaining
                self.order, _ = _read_remaining(self.root / 'remaining.u32', self.design, self.manifest['count'])
            else:
                self.order = array('I'); self.order.frombytes((self.root / 'remaining.u32').read_bytes())
                if sys.byteorder != 'little': self.order.byteswap()
            if len(self.order) != self.manifest['count']:
                raise QueueError('Remaining count mismatch')
            # A generation is started once. Resume creates a new success scan,
            # never replays old lease events or silently discards prior results.
            self.journal = (self.root / 'results.jsonl').open('xb', buffering=0)
            self.cursor = 0; self.retry = deque(); self.active = {}; self.by_worker = {}
            self.completed = CompletedJournalIndex(self.root / 'results.jsonl', self.design, self.queue_id)
            self.attempts = {}; self.exhausted = 0
            self.expiries = []; self.expired_leases = 0
            self.fsync_seconds = 0.; self.fsync_count = 0
            self.heartbeats_ok = 0
            self.leased_batches = self.leased_cells = 0
            self.done = self.failed = 0; self.paused = False; self.epoch = uuid.uuid4().hex
            self.worker_status = {}; self.retired_leases = {}
            self.draining = False; self.drain_reason = None
            if self.policy['validation_processes']:
                from .validation_pool import ValidationPool
                self.validation_pool = ValidationPool(self.manifest['identity'], self.policy['validation_processes'],
                    self.policy['validation_timeout_seconds'],
                    cpu_ids=service_binding['validator_cpus'] if service_binding else None,
                    fast_reads=self.policy['validation_fast_reads'])
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
            self.retired_leases[row['worker']] = (token, 'lost')
            del self.active[task_id]; self._release(row['worker'], task_id)
            self.expired_leases += 1
            if row['attempt'] >= self.manifest['max_attempts']:
                self.exhausted += 1
            else:
                self.retry.append(row['ordinal'])

    def _release(self, worker, task_id):
        """Drop one cell from its worker's batch, forgetting an emptied batch."""
        held = tuple(other for other in self.by_worker.get(worker, ()) if other != task_id)
        if held: self.by_worker[worker] = held
        else: self.by_worker.pop(worker, None)

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

    def protocol(self, worker, version=2, hostname=''):
        if version != 2: raise QueueError('Unsupported worker protocol')
        if not isinstance(worker, str) or not 0 < len(worker) <= 256:
            raise QueueError('Invalid worker')
        if not isinstance(hostname, str) or len(hostname) > 256: raise QueueError('Invalid hostname')
        with self.mutex:
            if self.closed or self.poisoned: raise QueueError('Dispatcher stopped')
            self.worker_status.setdefault(worker, {'state': 'idle', 'observed_at': self.clock(),
                'hostname': hostname, 'version': version, 'task_id': None,
                'unacknowledged_chunks': 0})
            self.worker_status[worker]['hostname'] = hostname
            p = self.policy
            return {'version': 2, 'max_batch': p['max_batch_tasks'],
                    'submit_bytes': p['submit_bytes'], 'max_request_bytes': 1024 * 1024,
                    'flush_seconds': p['flush_seconds'], 'status_seconds': p['status_seconds'],
                    'drain': self.draining, 'heartbeat_many': True}

    def _status(self, worker, status):
        if worker not in self.worker_status: return
        current = self.worker_status[worker]
        if status is not None:
            if not isinstance(status, dict): raise QueueError('Invalid worker status')
            state = status.get('state', current['state'])
            if state not in ('computing', 'submitting', 'idle', 'draining', 'blocked'):
                raise QueueError('Invalid worker state')
            task_id = status.get('task_id')
            elapsed = status.get('cell_elapsed_seconds', 0.)
            if elapsed is None and state != 'computing': elapsed = 0.
            if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
                raise QueueError('Invalid cell elapsed time')
            chunks = status.get('unacknowledged_chunks', 0)
            if type(chunks) is not int or chunks < 0: raise QueueError('Invalid unacknowledged chunks')
            if state == 'computing' and (task_id not in self.active or self.active[task_id]['worker'] != worker):
                # A concurrent submit may have just committed this status's cell.
                if self.completed.owner(task_id) != worker: raise LeaseLostError('Status refers to another lease')
                state = 'submitting'; task_id = None
            current.update(state=state, task_id=task_id, cell_elapsed_seconds=elapsed,
                           unacknowledged_chunks=chunks)
        current['observed_at'] = self.clock()

    def heartbeat_batch(self, worker, token, status=None):
        with self.mutex:
            if self.closed or self.poisoned: raise QueueError('Dispatcher stopped')
            if worker not in self.worker_status: raise QueueError('Protocol handshake required')
            retired = self.retired_leases.get(worker)
            if retired == (token, 'lost'): raise LeaseLostError('Batch lease lost')
            held = self.by_worker.get(worker, ())
            if not held:
                if retired != (token, 'complete'): raise LeaseLostError('Unknown batch lease')
                self._status(worker, {'state': 'idle'})
                return {'state': 'complete', 'drain': self.draining}
            rows = [self._check_lease(t, token, worker) for t in held]
            self._status(worker, status)
            now = self.clock(); expiry = now + self.manifest['lease_seconds']
            for row in rows:
                row['expiry'] = expiry; row['last_heartbeat'] = now
            self.heartbeats_ok += 1
            return {'state': 'active', 'expiry': expiry, 'drain': self.draining}

    def heartbeat_many(self, hostname, items):
        if not isinstance(hostname, str) or not isinstance(items, list) or not 0 < len(items) <= 128:
            raise QueueError('Invalid aggregated heartbeat')
        replies = []
        for item in items:
            try:
                if not isinstance(item, dict) or set(item) - {'worker', 'token', 'status'}:
                    raise QueueError('Invalid heartbeat item')
                with self.mutex:
                    owner = self.worker_status.get(item.get('worker'), {})
                    if not owner or owner.get('hostname') != hostname:
                        raise QueueError('Heartbeat node differs from registered worker')
                replies.append(self.heartbeat_batch(**item))
            except LeaseLostError as exc:
                replies.append({'error': str(exc), 'code': 'lease_lost'})
            except (QueueError, KeyError, TypeError, ValueError) as exc:
                replies.append({'error': str(exc), 'code': 'invalid_heartbeat'})
        self.metrics.add('heartbeat_aggregate_calls')
        self.metrics.add('heartbeat_aggregate_items', count=len(items))
        return {'items': replies}

    def worker_binding(self, worker, binding):
        with self.mutex:
            if self.service_binding is None:
                raise QueueError('CPU binding registration not enabled')
            if worker not in self.worker_status or not isinstance(binding, dict):
                raise QueueError('Worker protocol registration required')
            host = binding.get('hostname'); cores = binding.get('cores')
            if (host != self.worker_status[worker]['hostname'] or not isinstance(cores, list)
                    or len(cores) != 1 or not isinstance(cores[0], str)):
                raise QueueError('Worker must have one explicitly bound physical core')
            if host == self.service_binding['hostname'] and cores[0] in self.service_binding['cores']:
                raise QueueError('Worker overlaps a reserved service core')
            if worker in self.worker_bindings:
                if self.worker_bindings[worker] != binding:
                    raise QueueError('Worker binding changed during this generation')
                return {'ready': self.binding_ready}
            if (host, cores[0]) in self.worker_core_owners:
                raise QueueError('Two workers overlap a physical core')
            if len(self.worker_bindings) >= self.fleet:
                raise QueueError('Unexpected worker beyond admitted fleet')
            self.worker_bindings[worker] = binding
            self.worker_core_owners[(host, cores[0])] = worker
            if len(self.worker_bindings) == self.fleet:
                atomic_json(self.root / 'worker-bindings.json', self.worker_bindings)
                self.binding_ready = True
            return {'ready': self.binding_ready}

    def drain(self, reason):
        with self.mutex:
            self.draining = True; self.drain_reason = str(reason)
            return {'state': 'draining', 'reason': self.drain_reason}

    def worker_fault(self, worker, code, message=''):
        if code not in ('payload_too_large', 'spool_limit', 'submission_rejected'):
            raise QueueError('Invalid worker fault code')
        if not isinstance(message, str) or len(message) > 1000: raise QueueError('Invalid worker fault message')
        with self.mutex:
            if worker not in self.worker_status: raise QueueError('Protocol handshake required')
            self.worker_status[worker].update(state='blocked', fault=code)
            return self.drain('worker_fault:' + code)

    def _batch_payload(self, held):
        rows = [self.active[task_id] for task_id in held]
        return {'state': 'task', 'token': rows[0]['token'], 'expiry': rows[0]['expiry'],
                'queue_id': self.queue_id,
                'tasks': [{'id': row['id'], 'task': json.loads(row['task']), 'cell': row['cell'],
                           'attempt': row['attempt']} for row in rows]}

    def claim(self, worker, *, cached_cells=()):
        """One cell, for callers written before a lease covered a batch."""
        reply = self.claim_batch(worker, 1, cached_cells=cached_cells)
        if reply['state'] != 'task': return reply
        first = reply['tasks'][0]
        return {'state': 'task', 'id': first['id'], 'task': first['task'], 'cell': first['cell'],
                'attempt': first['attempt'], 'token': reply['token'], 'expiry': reply['expiry'],
                'queue_id': reply['queue_id']}

    def claim_batch(self, worker, count=1, *, cached_cells=(), remaining_seconds=None, status=None):
        """V2 prices the next cells before consuming their queue ordinals."""
        if not isinstance(worker, str) or not 0 < len(worker) <= 256:
            raise QueueError('Invalid worker')
        if type(count) is not int or count < 1:
            raise QueueError('Batch size must be a positive integer')
        with self.mutex:
            if self.closed or self.poisoned: raise QueueError('Dispatcher stopped')
            self._reap()
            self._status(worker, status)
            if self.draining: return {'state': 'draining'}
            if worker in self.by_worker:
                return self._batch_payload(self.by_worker[worker])  # lost reply, same lease
            if not self.binding_ready:
                return {'state': 'wait', 'retry_after_seconds': min(60., max(1., self.fleet / 100.)),
                        'reason': 'binding_barrier'}
            if self.paused: return {'state': 'paused'}
            claim_limit = self.policy['max_claimed_tasks']
            if claim_limit is not None:
                if self.leased_cells >= claim_limit:
                    self.paused = True
                    return {'state': 'paused', 'reason': 'allocation_task_limit'}
                count = min(count, claim_limit - self.leased_cells)
            pending = len(self.order) - self.cursor + len(self.retry)
            if not pending:
                reply = {'state': 'wait' if self.active else ('blocked' if self.failed or self.exhausted else 'complete')}
                if worker in self.worker_status:
                    w = self.worker_status[worker]
                    w['state'] = 'idle'; w['waits'] = w.get('waits', 0) + 1
                    reply['retry_after_seconds'] = min(30., 5. * 2 ** min(w['waits'] - 1, 3))
                return reply
            # Never hand one worker so much of what is left that the rest of the
            # fleet idles behind it. The tail is where batches grow largest and
            # where a stranded fleet costs the most.
            v2 = worker in self.worker_status
            share = max(1, pending // max(self.fleet, len(self.by_worker) + 1))
            count = max(1, min(count, self.policy['max_batch_tasks'] if v2 else MAX_BATCH_TASKS, pending, share))
            budget = self.policy['target_batch_seconds']
            if self.remaining_work_seconds is not None:
                budget = min(budget, max(1., self.remaining_work_seconds / (2 * self.fleet)))
            if remaining_seconds is not None:
                if isinstance(remaining_seconds, bool) or not isinstance(remaining_seconds, (int, float)) or not math.isfinite(remaining_seconds) or remaining_seconds < 0:
                    raise QueueError('Invalid worker remaining time')
                if remaining_seconds <= 0: return {'state': 'draining'}
                budget = min(budget, remaining_seconds)
            token = self.epoch + ':' + uuid.uuid4().hex
            now = self.clock(); expiry = now + self.manifest['lease_seconds']
            chosen = []; used = 0.; retry_count = len(self.retry)
            payload_size = len(canonical({'state': 'task', 'token': token, 'expiry': expiry,
                                         'queue_id': self.queue_id, 'tasks': []}))
            for offset in range(count):
                if offset < retry_count: ordinal = self.retry[offset]
                elif self.cursor + offset - retry_count < len(self.order):
                    ordinal = self.order[self.cursor + offset - retry_count]
                else: break
                task = task_at(self.design, ordinal); task_id = task.id
                from .prediction_workflow import cost_identity
                ci = (self.design.cost_identity(task) if hasattr(self.design, 'cost_identity')
                      else cost_identity(task, getattr(self.design, 'contract', None)))
                pricing = {'identity': ci} if ci is not None else {}
                price = self.estimator.batch_seconds(task.model, task.N, task.K, **pricing) if v2 else None
                if v2:
                    if chosen and (price is None or used + price > budget): break
                    if not chosen and price is not None and remaining_seconds is not None and price > remaining_seconds:
                        return {'state': 'draining'}
                attempt = self.attempts.get(ordinal, 0) + 1
                size = len(canonical({'id': task_id, 'task': asdict(task), 'cell': task.cell, 'attempt': attempt}))
                if v2 and payload_size + size + bool(chosen) > self.policy['claim_bytes']:
                    if not chosen: raise QueueError('Single task exceeds claim response limit')
                    break
                mean = self.estimator.estimate(task.model, task.N, task.K, **pricing) if self.remaining_work_seconds is not None else None
                chosen.append((ordinal, task, attempt, price, mean))
                payload_size += size + (len(chosen) > 1)
                used += price or 0.
                if v2 and (price is None or used >= budget): break
            # Commit the selected prefix only after all pricing/encoding succeeds.
            held = []
            for ordinal, task, attempt, price, mean in chosen:
                if self.retry: self.retry.popleft()
                else: self.cursor += 1
                task_id = task.id; self.attempts[ordinal] = attempt
                self.active[task_id] = {'id': task_id, 'ordinal': ordinal,
                    'task': canonical(asdict(task)).decode(), 'cell': task.cell,
                    'state': 'leased', 'worker': worker, 'attempt': attempt, 'token': token,
                    'last_heartbeat': now, 'expiry': expiry, 'price': price, 'mean_seconds': mean}
                heapq.heappush(self.expiries, (expiry, task_id, token))
                held.append(task_id)
            self.by_worker[worker] = tuple(held)
            self.retired_leases.pop(worker, None)
            if v2: self.worker_status[worker].update(waits=0, state='submitting')
            self.leased_batches += 1; self.leased_cells += len(held)
            return self._batch_payload(self.by_worker[worker])

    def heartbeat(self, task_id, token, worker):
        """Renew every cell of the batch this one belongs to: one shared deadline."""
        with self.mutex:
            if self.closed or self.poisoned: raise QueueError('Dispatcher stopped')
            self._check_lease(task_id, token, worker)
            now = self.clock(); expiry = now + self.manifest['lease_seconds']
            for held in self.by_worker.get(worker, ()):
                row = self.active.get(held)
                if row is not None and row['token'] == token:
                    row['expiry'] = expiry; row['last_heartbeat'] = now
            self.heartbeats_ok += 1
            return {'expiry': expiry}

    def submit(self, task_id, token, worker, result):
        """One result, for callers written before a lease covered a batch."""
        reply = self.submit_batch([task_id], token, worker, [result])
        return {'accepted': True, 'duplicate': reply['duplicate'][0]}

    def submit_batch(self, task_ids, token, worker, results, chunk_id=None):
        # ACK timeouts can replay a token while its original fsync is running.
        # Serialize that token through durability and final state publication;
        # distinct tokens still append concurrently and share the same fsync.
        # References include blocked callers, preventing removal/recreation of a
        # lock while a duplicate is waiting. Completed batches retain no lock.
        if not isinstance(token, str) or not token:
            raise QueueError('Invalid lease token')
        with self.mutex:
            guard = self.submission_locks.get(token)
            if guard is None:
                guard = [threading.Lock(), 0]
                self.submission_locks[token] = guard
            guard[1] += 1

        def commit_submission():
            # Serialize journal writes, but never hold the lease-state mutex across
            # shared-storage I/O. A validated submission pins its lease until durable
            # commit (or poison), so reaping cannot reassign a half-written result.
            #
            # Canonicalizing result rows and validating them scientifically read no
            # shared state and are the larger half of a commit, so they run before
            # the lock; only ownership, duplicate detection and the journal write
            # stay serialized. A failure there is deferred rather than raised, so a
            # stale worker still learns its lease is lost first, exactly as when
            # both checks ran inside the lock.
            if (not isinstance(task_ids, (list, tuple)) or not isinstance(results, (list, tuple))
                    or len(task_ids) != len(results) or not task_ids):
                raise QueueError('Batch submission needs one result per task id')
            if len(set(task_ids)) != len(task_ids):
                raise QueueError('Repeated task id in one batch submission')
            deferred = None; prepared = []
            try:
                self.metrics.add('received_records', count=len(results))
                encoded = None
                if self.policy['dispatcher_fast_path']:
                    try:                     # the journal needs these exact bytes; validators can decode them instead of unpickling dicts
                        encoded = [canonical(result) for result in results]
                    except (TypeError, ValueError):
                        encoded = None       # take the normal path so it raises its own error
                with self.metrics.span('validation'):
                    if self.validation_pool is not None:
                        try:
                            if encoded is not None:
                                self.validation_pool.validate_encoded(encoded)
                            else:
                                self.validation_pool.validate(results)
                        except OSError:
                            if self.validation_pool.failed:
                                self.drain('validation_pool_failed')
                            raise
                    else:
                        from .validation_pool import validate_rows
                        validate_rows(self.manifest['identity'], self.result_cache_validator, self.design, results)
                for index, (task_id, result) in enumerate(zip(task_ids, results)):
                    if encoded is not None:
                        # Byte-for-byte what canonical() gives for this dict (keys sorted: origin, result, task_id, token),
                        # without serializing the result a second time.
                        payload = encoded[index]
                        line = (b'{"origin":{"queue_id":' + canonical(self.queue_id) + b'},"result":' + payload +
                                b',"task_id":' + canonical(task_id) + b',"token":' + canonical(token) + b'}\n')
                    else:
                        payload = canonical(result)
                        line = canonical({'result': result, 'origin': {'queue_id': self.queue_id},
                                          'token': token, 'task_id': task_id}) + b'\n'
                    prepared.append((task_id, result, payload, line, hashlib.sha256(line).digest()))
            except (QueueError, ValueError, TypeError, KeyError) as exc:
                deferred = exc
            duplicate = [False] * len(task_ids); writing = []
            lock_started = time.monotonic()
            with self.submit_mutex:
                self.metrics.add("submit_lock_wait", time.monotonic() - lock_started)
                with self.mutex:
                    if self.closed or self.poisoned: raise QueueError('Dispatcher stopped')
                    for task_id in task_ids:
                        if task_id not in self.completed:
                            self._check_lease(task_id, token, worker)
                    if deferred is not None: raise deferred
                    try:
                        for index, (task_id, result, payload_bytes, raw, checksum) in enumerate(prepared):
                            payload = payload_bytes.decode()
                            row = self.validate_result(task_id, result, payload=payload_bytes)
                            if row['state'] in ('done', 'failed'):
                                if row['accepted_token'] == token and row['result'] == payload:
                                    duplicate[index] = True; continue
                                if row['accepted_token'] != token:
                                    raise LeaseLostError('Completed task belongs to another lease')
                                raise QueueError('Conflicting completed result')
                            row['committing'] = True
                            writing.append((row, result, payload, raw, checksum))
                    except BaseException:
                        # A rejected batch must not leave its earlier cells pinned,
                        # or reaping would never reclaim them.
                        for pinned in writing: pinned[0].pop('committing', None)
                        raise
                sequence = self.written_seq
                if writing:
                    try:
                        offset = self.journal.tell()
                        for row, _, _, raw, _ in writing:
                            row['_journal_offset'] = offset
                            offset += len(raw)
                        pending = memoryview(b''.join(entry[3] for entry in writing))
                        while pending:
                            written = self.journal.write(pending)
                            if not written: raise OSError('Short result write')
                            pending = pending[written:]
                        self.written_seq += len(writing)
                        sequence = self.written_seq
                    except BaseException:
                        with self.mutex: self.poisoned = True
                        raise
            # Durability is shared. One fsync flushes every record whose write had
            # already returned, so concurrent submissions pay for a single flush
            # instead of one each. A result is still acknowledged only after a flush
            # that started after its own bytes were written, never before. A batch
            # that proved entirely duplicate wrote nothing and needs no flush.
            elapsed = 0.; flushed = False
            if writing:
                try:
                    with self.fsync_mutex:
                        if self.synced_seq < sequence:
                            # Read the watermark under this lock: records written while
                            # the flush runs are not claimed by it.
                            target = self.written_seq
                            started = time.monotonic()
                            os.fsync(self.journal.fileno())
                            elapsed = time.monotonic() - started
                            self.synced_seq = target; flushed = True
                except BaseException:
                    with self.mutex: self.poisoned = True
                    raise
            with self.mutex:
                self.commits += len(writing)
                if flushed:
                    self.fsync_count += 1; self.fsync_seconds += elapsed
                for row, result, payload, raw, checksum in writing:
                    row.pop('committing', None)
                    row['state'] = 'failed' if result['status'] == 'failed' else 'done'
                    self.completed.add(row['id'], row.pop('_journal_offset'), len(raw), row['worker'], checksum)
                    del self.active[row['id']]; self._release(worker, row['id'])
                    self.attempts.pop(row['ordinal'], None)
                    if row['state'] == 'done':
                        self.done += 1
                        if self.remaining_work_seconds is not None:
                            self.remaining_work_seconds = max(0., self.remaining_work_seconds - row['mean_seconds'])
                    else: self.failed += 1
                if writing and worker not in self.by_worker and self.retired_leases.get(worker) != (token, 'lost'):
                    self.retired_leases[worker] = (token, 'complete')
                self.metrics.add('committed_records', count=len(writing))
                return {'accepted': True, 'duplicate': duplicate}

        try:
            with guard[0]:
                return commit_submission()
        finally:
            with self.mutex:
                guard[1] -= 1
                if not guard[1]:
                    del self.submission_locks[token]

    def pause(self, paused=True):
        with self.mutex:
            self.paused = bool(paused); return self.stats()

    def stats(self):
        with self.mutex:
            now = self.clock()
            ages = [now - row['last_heartbeat'] for row in self.active.values()]
            computing = submitting = idle = stragglers = 0
            unknown = max(0, self.fleet - len(self.worker_status))
            chunks = 0; tails = []; tail_known = True
            for worker, status in self.worker_status.items():
                age = now - status['observed_at']
                chunks += status.get('unacknowledged_chunks', 0)
                if age > self.policy['stale_status_seconds']:
                    unknown += 1; tail_known = False; continue
                task_id = status.get('task_id')
                row = self.active.get(task_id)
                if status['state'] == 'computing' and row is None:
                    # Its last reported cell committed; the worker may already
                    # compute the next. Do not mislabel it idle or drain on it.
                    unknown += 1; tail_known = False; continue
                if status['state'] == 'computing' and row is not None:
                    computing += 1
                    elapsed = status.get('cell_elapsed_seconds', 0.) + age
                    price = row['price']
                    if price is not None and elapsed > max(120., 4. * price): stragglers += 1
                    assigned = [self.active[t]['price'] for t in self.by_worker.get(worker, ())]
                    if any(p is None for p in assigned): tail_known = False
                    # An over-p99 cell is not zero remaining work. Keep one
                    # full current-cell price as an explicit conservative floor.
                    else: tails.append(max(price, sum(assigned) - elapsed))
                elif status['state'] == 'submitting': submitting += 1
                else: idle += 1
            return {'phase': getattr(self.design, 'phase', 'legacy'),
                'done': self.done, 'failed': self.failed, 'leased': len(self.active),
                'pending': len(self.order) - self.cursor + len(self.retry), 'exhausted': self.exhausted,
                'total': len(self.order), 'paused': self.paused, 'queue_id': self.queue_id,
                'shard': self.shard['index'] if self.shard else None,
                'expired_leases': self.expired_leases, 'fsync_count': self.fsync_count,
                'fsync_seconds': self.fsync_seconds, 'heartbeats_ok': self.heartbeats_ok,
                'commits': self.commits, 'leased_batches': self.leased_batches,
                'cells_per_batch': self.leased_cells / self.leased_batches if self.leased_batches else 0.,
                'results_per_fsync': self.commits / self.fsync_count if self.fsync_count else 0.,
                'protocol_metrics': self.metrics.snapshot(),
                'validation_pool_failed': bool(self.validation_pool and self.validation_pool.failed),
                'validation_pool_metrics': self.validation_pool.snapshot() if self.validation_pool else None,
                'binding_ready': self.binding_ready, 'registered_bindings': len(self.worker_bindings),
                'fleet': self.fleet, 'registered_workers': len(self.worker_status),
                'computing_workers': computing, 'submitting_workers': submitting,
                'idle_workers': idle, 'unknown_workers': unknown, 'straggler_workers': stragglers,
                'compute_busy_fraction': computing / self.fleet,
                'effective_busy_fraction': max(0, computing - stragglers) / self.fleet,
                'unacknowledged_chunks': chunks, 'remaining_work_seconds': self.remaining_work_seconds,
                'predicted_tail_seconds': max(tails, default=0.) if tail_known else None,
                'draining': self.draining, 'drain_reason': self.drain_reason,
                'oldest_active_heartbeat_age_seconds': max(ages, default=0.),
                'active_heartbeat_older_than_900s': sum(age > 900 for age in ages),
                'active_heartbeat_older_than_1800s': sum(age > 1800 for age in ages)}

    def close(self):
        if self.validation_pool is not None:
            self.validation_pool.close(); self.validation_pool = None
        with self.submit_mutex, self.fsync_mutex:
            with self.mutex:
                if self.closed: return
                self.closed = True
            journal = self.journal
            if journal is not None and not journal.closed and self.synced_seq < self.written_seq:
                try:
                    os.fsync(journal.fileno()); self.synced_seq = self.written_seq
                except OSError:
                    pass  # Unflushed results stay unacknowledged, never falsely accepted.
            if journal is not None: journal.close()
            completed = getattr(self, 'completed', None)
            if completed is not None: completed.close()
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


def write_progress(path, dispatcher, server, *, previous_errors=0, monitor=None,
                   allocation=None, continuation_allowed=False):
    """Progress is advisory; durable result writes must still fail closed."""
    try:
        stats = dispatcher.stats(); now = time.time()
        observation = (monitor.observe(stats, now=now, allocation=allocation,
                         continuation_allowed=continuation_allowed) if monitor else {})
        if observation.get('idle_alert') and monitor.samples == monitor.policy['idle_samples']:
            print('Scheduler idle fleet: ' + json.dumps(observation), file=sys.stderr, flush=True)
        if observation.get('drain_recommended') and not dispatcher.draining:
            atomic_json(Path(path).parent / 'drain-intent.json',
                        {'reason': 'costed_tail', 'observed_at': now, **observation})
            dispatcher.drain('costed_tail')
        atomic_json(path, {'stats': stats, 'observed_at': now, 'efficiency': observation,
                          'rpc': server.connection_stats(),
                          'progress_write_errors': previous_errors})
        if stats.get('protocol_metrics'):
            try:
                # Overlapping 120-second windows preserve bursts between operator
                # checks. Telemetry failures never poison durable acceptance.
                with Path(path).with_name('protocol-history.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps({'observed_at': now, 'done': stats['done'],
                        'protocol_metrics': stats['protocol_metrics'],
                        'validation_pool_metrics': stats['validation_pool_metrics'],
                        'rpc': server.connection_stats()}, separators=(',', ':')) + '\n')
            except OSError:
                dispatcher.metrics.add('telemetry_write_error')
    except OSError as exc:
        if exc.errno not in (errno.EMFILE, errno.ENFILE):
            raise
        if previous_errors == 0:
            print('Dispatcher FD pressure: progress snapshot deferred; results remain durable',
                  file=sys.stderr, flush=True)
        return previous_errors + 1
    return previous_errors


def allocation_start_time(entered, max_seconds, *, environ=None, query=None):
    """Use trusted allocation time, with a scoped live Slurm fallback.

    Some sites do not export SLURM_JOB_START_TIME. Runtime entry time cannot
    discount reserved CPU hours because module/bootstrap time precedes it.
    Missing/malformed/mismatched Slurm evidence stays conservatively untrusted.
    """
    environ = os.environ if environ is None else environ
    def valid(value):
        return math.isfinite(value) and value > 0 and 0 <= entered - value <= max_seconds
    try:
        started = float(environ['SLURM_JOB_START_TIME'])
        if valid(started): return started, 'slurm_job_start', {'source': 'SLURM_JOB_START_TIME'}
    except (KeyError, ValueError, TypeError, OverflowError):
        pass
    job_id = environ.get('SLURM_JOB_ID', '')
    if isinstance(job_id, str) and re.fullmatch(r'[1-9][0-9]*', job_id):
        try:
            if query is None:
                raw = subprocess.check_output(['scontrol', 'show', 'job', '-o', job_id],
                                              text=True, timeout=12)
            else:
                raw = query(['scontrol', 'show', 'job', '-o', job_id])
            lines = [line for line in raw.splitlines() if line.strip()]
            if len(lines) == 1:
                fields = dict(token.split('=', 1) for token in lines[0].split() if '=' in token)
                if fields.get('JobId') == job_id:
                    started = datetime.fromisoformat(fields['StartTime']).timestamp()
                    if valid(started):
                        return started, 'slurm_job_start', {'source': 'scontrol', 'job_id': job_id,
                                                           'StartTime': fields['StartTime']}
        except (OSError, subprocess.SubprocessError, ValueError, TypeError, KeyError, OverflowError):
            pass
    return entered, 'runtime_only', {'source': 'unavailable'}


def run(root, repo, old, workers, validate_only=False, *, max_seconds=172800,
        policy=None, cost_profile=None, allocation=None, continuation_allowed=False):
    from aleatoric_nk_grid.queue_service import make_server
    from aleatoric_nk_grid.queue_readiness import publish_ready
    policy = validate_policy(policy)
    if int((allocation or {}).get('dispatcher_shards', 1)) > 1:
        from .dispatcher_shards import run_sharded
        return run_sharded(root, repo, old, workers, validate_only, max_seconds=max_seconds, policy=policy,
                           cost_profile=cost_profile, allocation=allocation, continuation_allowed=continuation_allowed)
    entered = time.time()
    job_started, elapsed_source, elapsed_evidence = allocation_start_time(entered, max_seconds)
    job_end = min(job_started + max_seconds, float(os.environ.get('SLURM_JOB_END_TIME', job_started + max_seconds)))
    if not math.isfinite(job_end) or job_end <= entered: raise QueueError('Allocation has no time remaining')
    work_deadline = max(entered, job_end - policy['drain_grace_seconds'])
    service = None
    service_slots = 1 + policy['validation_processes']
    if policy['validation_processes']:
        from .service_binding import service_layout
        if (allocation or {}).get('controller_task_slots') != service_slots:
            raise QueueError('Validation CPU reservation missing from allocation')
        service = service_layout(policy['validation_processes'])
        os.sched_setaffinity(0, set(service['dispatcher_cpus']) if policy['dispatcher_smt'] else {service['dispatcher_cpu']})
    admitted_slots = (service['job_task_slots'] if service else int(os.environ['SLURM_NTASKS']))
    if admitted_slots < workers + service_slots:
        raise QueueError('Missing worker/controller/validator task reservations')
    manifest = read(root / 'manifest.json')
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != manifest['identity']['cell_spec']['git_commit']:
        raise QueueError('Frozen scientific code changed')
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip():
        raise QueueError('Frozen checkout dirty')
    control = root / 'control'; control.mkdir(exist_ok=True, mode=0o700)
    with file_lock(control / 'round.lock'), FlatDispatcher(root, fleet=workers, policy=policy,
            cost_profile=cost_profile, work_seconds=(allocation or {}).get('work_seconds'),
            service_binding=service) as dispatcher:
        generation = uuid.uuid4().hex; attempt = control / generation; attempt.mkdir(mode=0o700)
        fault_dir = attempt / 'faults'; fault_dir.mkdir()
        if service is not None:
            atomic_json(attempt / 'service-binding.json', {**service, 'dispatcher_affinity': sorted(os.sched_getaffinity(0)),
                'validators': dispatcher.validation_pool.binding_proof})
        host = socket.gethostname(); token = attempt / 'token'; token.write_text(uuid.uuid4().hex + uuid.uuid4().hex)
        token.chmod(0o600); cert = attempt / 'ca.crt'; key = attempt / 'server.key'
        cert_days = str(max(7, (int(max_seconds) + 86399) // 86400 + 1))
        subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', cert_days,
            '-keyout', str(key), '-out', str(cert), '-subj', '/CN=' + host,
            '-addext', 'subjectAltName=DNS:' + host + ',DNS:' + socket.getfqdn()],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        key.chmod(0o600)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER); context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(cert, key)
        # A larger connection budget needs descriptors to match; the default 128 needs no change.
        if policy['max_connections'] > 128:
            from aleatoric_nk_grid.queue_service import raise_nofile_limit
            raise_nofile_limit(policy['max_connections'])
        server = make_server(dispatcher, token=token.read_text(), host='0.0.0.0', port=0,
                             tls_context=context, threaded_tls_handshake=True,
                             max_connections=policy['max_connections'],
                             keepalive_idle_seconds=policy['keepalive_idle_seconds'],
                             max_submissions=policy['max_submissions'], keepalive=policy["rpc_keepalive"],
                             fast_http=policy['dispatcher_fast_path'])
        serving = threading.Thread(target=server.serve_forever); serving.start()
        child = None
        try:
            ready_path = attempt / 'ready.json'
            publish_ready(ready_path, dispatcher, host=host, port=server.server_port, tls=True, generation=generation)
            launch = dict(queue=str(root.resolve()), queue_id=dispatcher.queue_id, repo=str(repo.resolve()),
                control=str(control.resolve()), generation=generation, token_file=str(token.resolve()),
                ca_file=str(cert.resolve()), ready_file=str(ready_path.resolve()),
                job_id=os.environ['SLURM_JOB_ID'], source_commit=commit, workers=workers,
                recover_stale_leases=True, max_seconds=max_seconds, deadline_epoch=work_deadline,
                protocol_version=2, fault_dir=str(fault_dir.resolve()),
                rpc_keepalive=policy["rpc_keepalive"], protocol_metrics=policy["protocol_metrics"],
                node_relay=policy["node_relay"],
                # The relay must drop an idle upstream connection before the server does.
                node_relay_idle_seconds=round(.8 * policy["keepalive_idle_seconds"], 3),
                protocol_metrics_sample_modulo=policy['protocol_metrics_sample_modulo'],
                require_cpu_binding=service is not None,
                heartbeat_aggregate_seconds=policy["heartbeat_aggregate_seconds"],
                startup_jitter_seconds=min(300. if service is not None else 30., workers / 100.))
            atomic_json(attempt / 'launch.json', launch); atomic_json(control / 'latest.json', launch)
            atomic_json(attempt / 'admission.json', {'stats': dispatcher.stats(), 'sqlite': False,
                                                   'rpc': server.connection_stats()})
            def interrupted(signum, frame): raise InterruptedError('Interrupted ' + str(signum))
            signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
            step = ['srun', '--ntasks=' + str(workers), '--cpus-per-task=1', '--ntasks-per-core=1']
            step_env = os.environ.copy()
            if service is not None:
                from .service_binding import worker_step
                step, step_env = worker_step(allocation, service, workers, attempt / 'worker-hosts.txt')
            else:
                step += ['--distribution=cyclic']
            child = subprocess.Popen(step + ['--kill-on-bad-exit=1',
                '--output=' + str(attempt / 'worker-%t.out'), '--error=' + str(attempt / 'worker-%t.err'),
                sys.executable, '-m', 'aleatoric_nk_grid.slurm_queue_round', 'worker',
                '--launch', str(attempt / 'launch.json')], env=step_env)
            progress_errors = 0; drain_started = None
            monitor = TailMonitor(policy, started=job_started)
            while child.poll() is None:
                for path in fault_dir.glob('*.json'):
                    fault = read(path)
                    if fault.get('queue_id') != dispatcher.queue_id: raise QueueError('Worker fault queue changed')
                    dispatcher.worker_fault(fault['worker'], fault['code'], fault.get('message', ''))
                progress_errors = write_progress(attempt / 'progress.json', dispatcher, server,
                    previous_errors=progress_errors, monitor=monitor, allocation=allocation,
                    continuation_allowed=continuation_allowed)
                now = time.time()
                if now >= work_deadline and not dispatcher.draining:
                    atomic_json(attempt / 'drain-intent.json', {'reason': 'allocation_deadline', 'observed_at': now})
                    dispatcher.drain('allocation_deadline')
                if dispatcher.draining:
                    if drain_started is None: drain_started = now
                    # Economic drain waits for a persisted fold/cell boundary.
                    # Only faults and the hard allocation deadline may interrupt
                    # an unfinished fit; idle workers alone never justify it.
                    grace_expired = (dispatcher.drain_reason != 'costed_tail' and
                                     now - drain_started >= policy['drain_grace_seconds'])
                    if grace_expired or now >= job_end - 5:
                        break
                time.sleep(min(15., max(.1, job_end - now - 5)))
        finally:
            try:
                if child is not None and child.poll() is None:
                    child.terminate()
                    try: child.wait(timeout=90)
                    except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=30)
            finally:
                server.shutdown(); serving.join(); server.server_close()
            # A final worker may exit between loop polls after writing its fault.
            for path in fault_dir.glob('*.json'):
                fault = read(path)
                if fault.get('queue_id') != dispatcher.queue_id: raise QueueError('Worker fault queue changed')
                dispatcher.worker_fault(fault['worker'], fault['code'], fault.get('message', ''))
            stats = dispatcher.stats()
            bounded_complete = (policy['max_claimed_tasks'] is not None and stats['paused']
                and not stats['leased'] and not stats['failed'] and child is not None and child.returncode == 0)
            atomic_json(control / 'round-result.json', {'stats': stats, 'worker_exit': child.returncode if child else None,
                'complete': stats['done'] == stats['total'], 'generation': generation,
                'state': 'complete' if stats['done'] == stats['total'] else
                    ('bounded_complete' if bounded_complete else ('drained' if dispatcher.draining else 'incomplete')),
                'job_id': os.environ['SLURM_JOB_ID'], 'elapsed_seconds': max(0., time.time() - job_started),
                'elapsed_source': elapsed_source, 'elapsed_evidence': elapsed_evidence,
                'job_started_epoch': job_started, 'queue_id': dispatcher.queue_id,
                'allocation_cpu': (allocation or {}).get('allocated_cpu_bound'),
                'drain_reason': dispatcher.drain_reason})
        if stats['done'] != stats['total'] and not (validate_only and (dispatcher.draining or bounded_complete)):
            raise QueueError('Incomplete round; retain result receipts for next direct success scan')
    if not validate_only:
        merge(root, old)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run'])
    parser.add_argument('--root', type=Path, help='Prepared run directory; prepare defaults to <repo>/FFCWS/outputs/ffc_median_mode_gpa-<unique ID>')
    parser.add_argument('--base', type=Path)
    parser.add_argument('--repo', type=Path)
    parser.add_argument('--old', type=Path)
    parser.add_argument('--workers', type=int, default=21)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        if args.base is None:
            parser.error('prepare requires --base pointing to a stopped GPA run')
        if args.root is None:
            repo = (args.repo or Path(__file__).resolve().parents[3]).expanduser().resolve()
            args.root = repo / 'FFCWS' / 'outputs' / ('ffc_median_mode_gpa-' + uuid.uuid4().hex[:12])
        args.root = args.root.expanduser().resolve()
        args.base = args.base.expanduser().resolve()
        owner_root = args.base if (args.base/'manifest.json').exists() else args.base/'queue'
        with file_lock(args.base / 'control/round.lock'), file_lock(owner_root / 'dispatcher.lock'):
            prepare(args.base, args.root)
        print(json.dumps({'root': str(args.root),
                          'final_csv': str(args.root / 'final' / 'ffc_median_mode_gpa.csv')}), flush=True)
    else:
        if args.root is None or args.repo is None or args.old is None:
            parser.error('run requires --root from prepare, --repo and --old')
        run(args.root.expanduser().resolve(), args.repo.expanduser().resolve(),
            args.old.expanduser().resolve(), args.workers, args.validate_only)


if __name__ == '__main__':
    main()
