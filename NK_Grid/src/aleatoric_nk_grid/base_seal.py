"""Resumable base verification barrier (base-seal-v3).

Every accepted base row was fully validated by the dispatcher before its ACK
(frame checksum, task identity, provenance, sample maps, OOF folds), and every
SL read re-checks sealed-index membership and the frame checksum. Decoding all
base frames a third time here adds no information about bytes that are still
checksum-locked. This barrier therefore proves only what is not yet proven:

* exactly one accepted, provenance-checked result for every frozen base task;
* each accepted reference resolves to a sealed shard whose size equals its
  sealed index and whose index lists that exact frame (loss/truncation);
* a seeded, stratified sample passes the full content validation again.

Work is committed in batches together with its journal position, so a
controller timeout resumes instead of starting over. The published
base-records.sqlite has the same schema and content as before.
"""
from array import array
from collections import Counter
from contextlib import ExitStack, closing
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import time

from .shared_queue import QueueError, atomic_json, canonical, digest, file_lock

FORMAT = 'base-seal-v3'
SAMPLE_RULE = 'stratified-random-v1'
DEFAULT_SAMPLE_RATE = 0.001
BATCH_ROWS = 100_000
BATCH_SECONDS = 120.
SHARD_BATCH = 500


def audit_sample(design, *, rate, seed):
    """ceil(rate * size), at least one, drawn per (panel, pipeline, N, K) stratum.

    Each stratum draws from its own string-seeded generator, so the selection
    depends only on the frozen design, the rate and the seed. The chosen ordinals
    are also written beside the receipt, so no reproduction relies on this code.
    """
    if not 0 < rate <= 1: raise ValueError('Audit sample rate must be in (0, 1]')
    chosen, strata = set(), 0
    for task, ordinals in design.groups():
        strata += 1
        size = len(ordinals); count = min(size, max(1, math.ceil(rate * size)))
        key = canonical([seed, task.panel_id, task.pipeline_id, task.N, task.K]).decode()
        chosen.update(random.Random(key).sample(ordinals, count))
    return chosen, strata


class Locations:
    """In-memory copy of the immutable location catalogs.

    Walks generations exactly like prediction_layout.resolve_reference, without
    opening a SQLite connection for every moved record.
    """
    def __init__(self, root):
        from .prediction_cache import CacheIntegrityError, safe_cache_path
        self.error = CacheIntegrityError
        self.routes, self.legacy, self.tables = {}, [], []
        pointer = Path(root) / 'locations.json'
        if not pointer.exists(): return
        data = json.loads(pointer.read_bytes())
        if data.get('format') != 'prediction-locations-v1':
            raise CacheIntegrityError('Unknown location generation')
        for position, item in enumerate(data['generations']):
            catalog = safe_cache_path(root, item['path'])
            with closing(sqlite3.connect(catalog.as_uri() + '?mode=ro&immutable=1', uri=True)) as db:
                self.tables.append({(path, offset, sha256): reference for path, offset, sha256, reference
                                    in db.execute('SELECT path, offset, sha256, reference FROM locations')})
            if 'sources' not in item: self.legacy.append(position)
            else:
                for source in item['sources']: self.routes.setdefault(source, []).append(position)

    def resolve(self, reference):
        current = reference; position = -1
        while True:
            for index in sorted(self.routes.get(current['path'], []) + self.legacy):
                if index <= position: continue
                position = index
                raw = self.tables[index].get((current['path'], current['offset'], current['sha256']))
                if raw is not None:
                    replacement = json.loads(raw)
                    if any(current.get(k) != replacement.get(k) for k in ('sha256', 'identity', 'length', 'status')):
                        raise self.error('Location generation changed frame identity')
                    current = replacement
                    break  # Follow a later move of this new address.
            else:
                return current


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _fsync(path):
    with Path(path).open('r+b') as handle: os.fsync(handle.fileno())


def _reference_fields(reference):
    path, offset, length, sha256 = (reference.get(k) for k in ('path', 'offset', 'length', 'sha256'))
    if (not isinstance(path, str) or type(offset) is not int or type(length) is not int
            or offset < 0 or length < 1 or not isinstance(sha256, str) or len(sha256) != 64):
        raise QueueError('Accepted base row has a malformed prediction cache reference')
    return path, offset, length, sha256


def seal(plan, rounds, *, sample_rate=DEFAULT_SAMPLE_RATE, sample_seed=None,
         batch_rows=BATCH_ROWS, batch_seconds=BATCH_SECONDS):
    from . import prediction_workflow as flow
    from .prediction_cache import (CacheIntegrityError, _index_path, _load_index, _sync_directory,
                                   encode_index_json, safe_cache_path)
    contract = plan['prediction_workflow']
    root = Path(plan['launch']['output']); cache = Path(contract['cache_root'])
    final = root / 'base-records.sqlite'; receipt_path = root / 'base-verified.json'
    work_path = root / 'base-seal-v3.work.sqlite'; partial = root / 'base-records.v3.partial.sqlite'
    progress_path = root / 'base-seal-progress.json'
    rounds = [Path(r) for r in rounds]
    identity = flow.phase_identity(plan, 'base')
    design = flow.PredictionDesign(contract, 'base')
    shared = contract.get('storage', {}).get('layout') == 'shared-v1'
    seed = sample_seed if sample_seed is not None else 'base-audit:' + digest(plan)
    header = {'format': FORMAT, 'plan_sha256': digest(plan), 'workflow_sha256': digest(contract),
              'rounds': [str(r) for r in rounds], 'expected_rows': design.count,
              'sample_rule': SAMPLE_RULE, 'sample_rate': sample_rate, 'sample_seed': seed}

    if not work_path.exists():
        if final.exists():
            raise QueueError('Unreceipted base-records.sqlite without seal state; explicit repair required')
        partial.unlink(missing_ok=True)  # Never committed: work state is created first.

    attempt_started = time.monotonic(); attempt_rows = 0
    with ExitStack() as stack:
        for directory in rounds:
            stack.enter_context(file_lock(directory / 'dispatcher.lock'))
        connection = stack.enter_context(closing(sqlite3.connect(work_path)))
        connection.execute('ATTACH DATABASE ? AS out', (str(partial),))
        connection.execute('PRAGMA main.synchronous=FULL'); connection.execute('PRAGMA out.synchronous=FULL')
        # Both tables are keyed by values in random order; a page cache far above
        # SQLite's 2 MiB default avoids rereading B-tree pages from Lustre.
        # Control jobs request 48 GiB.
        connection.execute('PRAGMA main.cache_size=-2097152'); connection.execute('PRAGMA out.cache_size=-4194304')
        connection.execute('CREATE TABLE IF NOT EXISTS main.meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        # Grouped by resolved shard for the one-pass sealed-index check.
        connection.execute('CREATE TABLE IF NOT EXISTS main.refs(path TEXT NOT NULL, offset INTEGER NOT NULL, '
                           'ordinal INTEGER NOT NULL, round INTEGER NOT NULL, length INTEGER NOT NULL, '
                           'sha256 TEXT NOT NULL, PRIMARY KEY(path, offset, ordinal)) WITHOUT ROWID')
        # Same schema as every earlier base index; consumers are unchanged.
        connection.execute('CREATE TABLE IF NOT EXISTS out.records(task_id TEXT PRIMARY KEY, '
                           'reference TEXT NOT NULL, row BLOB NOT NULL)')
        connection.commit()
        meta = {key: json.loads(value) for key, value in connection.execute('SELECT key, value FROM main.meta')}
        if not meta:
            meta = {'header': header, 'state': 'scanning', 'position': [0, 0], 'rows': 0, 'audited': 0,
                    'statuses': {}, 'journal_bytes': {}, 'shards_checked': 0, 'shard_bytes': 0,
                    'shard_last_path': None, 'first_started_utc': _utc(), 'attempts': 0}
        elif meta.get('header') != header:
            raise QueueError('Base seal work state belongs to another plan, round list or audit rule; '
                             'explicit repair required')
        meta['attempts'] = meta.get('attempts', 0) + 1

        def publish_progress(**extra):
            elapsed = time.monotonic() - attempt_started
            atomic_json(progress_path, {'format': FORMAT, 'state': meta['state'], 'rows': meta['rows'],
                'expected_rows': design.count, 'audited': meta['audited'], 'sample_size': meta.get('sample_size'),
                'position': meta['position'], 'shards_checked': meta['shards_checked'],
                'attempt': meta['attempts'], 'attempt_rows': attempt_rows,
                'attempt_seconds': round(elapsed, 3),
                'attempt_rows_per_second': round(attempt_rows / elapsed, 3) if elapsed > 0 else None,
                'updated_at_utc': _utc(), **extra})

        def save(**changes):
            meta.update(changes)
            connection.executemany('INSERT OR REPLACE INTO main.meta VALUES (?, ?)',
                                   [(key, json.dumps(value, sort_keys=True)) for key, value in meta.items()])
            connection.commit()

        sample, strata = audit_sample(design, rate=sample_rate, seed=seed)
        meta.update(sample_size=len(sample), sample_strata=strata)
        save()

        if meta['state'] == 'scanning':
            # Committed rows, per round. They are marked round by round below so
            # each round's remaining-task check sees only earlier rounds, as in scan.
            committed = {}
            for round_index, ordinal in connection.execute('SELECT round, ordinal FROM main.refs'):
                committed.setdefault(round_index, array('I')).append(ordinal)
            if sum(map(len, committed.values())) != meta['rows']:
                raise QueueError('Base seal checkpoint row count differs from its committed references')
            statuses = Counter(meta['statuses'])
            validator = flow.ResultCacheValidator(identity, fast_reads=True)
            stack.callback(validator.context.close)
            locations = Locations(cache)
            records, refs = [], []; last_commit = time.monotonic()
            sources = []

            def flush(position):
                nonlocal records, refs, attempt_rows, last_commit
                connection.executemany('INSERT INTO out.records VALUES (?, ?, ?)', records)
                connection.executemany('INSERT INTO main.refs VALUES (?, ?, ?, ?, ?, ?)', refs)
                attempt_rows += len(records)
                save(position=position, rows=meta['rows'] + len(records),
                     statuses=dict(statuses), audited=meta['audited'])
                records, refs = [], []; last_commit = time.monotonic()
                publish_progress()

            for index, directory in enumerate(rounds):
                qid, members = flow.round_members(directory, design, identity, sources)
                for ordinal in committed.pop(index, ()):
                    if design.contains(ordinal) or not members[ordinal // 8] & (1 << (ordinal % 8)):
                        raise QueueError('Base seal checkpoint row outside its round or duplicated')
                    design.mark(ordinal)
                path = directory / 'results.jsonl'
                if not path.exists():
                    sources.append({'root': str(directory), 'queue_id': qid}); continue
                size = path.stat().st_size
                recorded = meta['journal_bytes'].get(str(index))
                if recorded is None: meta['journal_bytes'][str(index)] = size
                elif recorded != size: raise QueueError('Stopped prediction journal changed')
                if index >= meta['position'][0]:
                    start = meta['position'][1] if index == meta['position'][0] else 0
                    for row, ordinal, end in flow.journal_rows(path, design, identity, qid, members, start=start):
                        if ordinal in sample:
                            validator(row, sealed=True); meta['audited'] += 1
                        if design.contains(ordinal): raise QueueError('Duplicate prediction success')
                        design.mark(ordinal); statuses[row['status']] += 1
                        reference = row['prediction_cache_ref']
                        _reference_fields(reference)
                        path_, offset, length, sha256 = _reference_fields(locations.resolve(reference))
                        refs.append((path_, offset, ordinal, index, length, sha256))
                        records.append((design.task_at(ordinal).id, canonical(reference).decode(),
                            encode_index_json({k: v for k, v in row.items() if k != 'prediction_cache_ref'})
                            if shared else canonical(row).decode()))
                        if len(records) >= batch_rows or time.monotonic() - last_commit >= batch_seconds:
                            flush([index, end])
                if path.stat().st_size != size: raise QueueError('Stopped prediction journal changed')
                sources.append({'root': str(directory), 'queue_id': qid})
                if index >= meta['position'][0]:
                    flush([index + 1, 0])
            if committed:
                raise QueueError('Base seal checkpoint names a round outside the frozen round list')
            count = sum(b.bit_count() for b in design.bits)
            if count != design.count or meta['rows'] != count:
                raise QueueError('All-panel base cache coverage incomplete')
            if meta['audited'] != len(sample):
                raise QueueError('Audit sample was not fully validated')
            save(state='shards', sources=sources, scan_completed_utc=_utc())
            publish_progress()

        if meta['state'] == 'shards':
            def check_shard(relative, expected=None):
                index_path = _index_path(cache, relative)
                if not index_path.is_file():
                    raise CacheIntegrityError('Accepted base frame lies in an unsealed shard: ' + relative)
                index, entries = _load_index(index_path)
                if not index.get('sealed') or index.get('path') != relative:
                    raise CacheIntegrityError('Accepted base frame lies in an unsealed shard: ' + relative)
                try:
                    size = safe_cache_path(cache, relative).stat().st_size
                except FileNotFoundError as exc:
                    raise CacheIntegrityError('Sealed prediction shard is missing: ' + relative) from exc
                if size != index['bytes']:
                    raise CacheIntegrityError('Sealed prediction shard size differs from its index: ' + relative)
                for item in expected or ():
                    if item not in entries:
                        raise CacheIntegrityError('Prediction reference is absent from its sealed index: ' + relative)
                return size
            while True:
                last = meta['shard_last_path']
                paths = [p for (p,) in connection.execute(
                    'SELECT DISTINCT path FROM main.refs WHERE path > ? ORDER BY path LIMIT ?',
                    (last if last is not None else '', SHARD_BATCH))]
                if not paths: break
                checked = 0; bytes_ = 0
                for relative in paths:
                    frames = {(offset, length, sha256) for offset, length, sha256 in connection.execute(
                        'SELECT offset, length, sha256 FROM main.refs WHERE path = ?', (relative,))}
                    bytes_ += check_shard(relative, frames); checked += 1
                save(shard_last_path=paths[-1], shards_checked=meta['shards_checked'] + checked,
                     shard_bytes=meta['shard_bytes'] + bytes_)
                publish_progress()
            maps = 0
            for shard in sorted((cache / 'sample-maps').glob('*.pcshard')):
                check_shard(shard.relative_to(cache).as_posix()); maps += 1
            receipt = {'format': flow.FORMAT, 'complete': True, 'plan_sha256': digest(plan),
                'workflow_sha256': digest(contract), 'rows': meta['rows'], 'expected_rows': design.count,
                'panels': [p['panel_id'] for p in contract['panels']],
                'statuses': meta['statuses'], 'sources': meta['sources'],
                'integrity': 'record-sha256-v1',
                'score_complete': True, 'prediction_cache_complete': True, 'oof_complete': True,
                'unfinished_leases': 0, 'unsubmitted_required_results': 0, 'unresolved_failures': 0,
                'verification': {'format': FORMAT,
                    'row_content': 'full validate_result_cache by the dispatcher before each ACK',
                    'consumer_checks': 'sealed-index membership and frame sha256 on every SL read',
                    'barrier_checks': ['one accepted provenance-checked row per frozen base task',
                                       'every reference in a sealed shard index',
                                       'every referenced shard size equals its sealed index',
                                       'every sample-map shard sealed with matching size',
                                       'seeded stratified sample fully revalidated'],
                    'referenced_shards': meta['shards_checked'], 'referenced_shard_bytes': meta['shard_bytes'],
                    'sample_map_shards': maps,
                    'audit_sample': {'rule': SAMPLE_RULE, 'rate': sample_rate, 'seed': seed,
                                     'strata': meta['sample_strata'], 'rows': meta['audited'],
                                     'ordinals_file': 'base-audit-sample.json'}}}
            atomic_json(root / 'base-audit-sample.json', {'format': FORMAT, 'rule': SAMPLE_RULE,
                'rate': sample_rate, 'seed': seed, 'ordinals': sorted(sample)})
            save(state='publishing', receipt=receipt, shards_completed_utc=_utc())
            publish_progress()

        receipt = meta['receipt']
        connection.execute('DETACH DATABASE out')

    if partial.exists():
        _fsync(partial)
        os.replace(partial, final)
        _sync_directory(root)
    elif not final.is_file():
        raise QueueError('Base seal publication lost its records index; explicit repair required')
    receipt = {**receipt, 'records_index_bytes': final.stat().st_size}
    atomic_json(receipt_path, receipt)
    work_path.unlink()
    atomic_json(progress_path, {'format': FORMAT, 'state': 'complete', 'rows': receipt['rows'],
        'expected_rows': receipt['expected_rows'], 'records_index_bytes': receipt['records_index_bytes'],
        'attempt_rows': attempt_rows, 'attempt_seconds': round(time.monotonic() - attempt_started, 3),
        'updated_at_utc': _utc()})
    return receipt
