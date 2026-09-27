"""Restartable verification of stopped journals across Slurm worker processes.

Workers own disjoint byte ranges and publish checksummed intermediate artifacts.
Only the reducer checks global coverage and publishes the original phase receipt.
No numerical fitting, live journal reads, or concurrent writes to one SQLite DB.
"""
from collections import Counter, OrderedDict
from contextlib import ExitStack, closing
from dataclasses import asdict
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
import uuid

import numpy as np

from .shared_queue import QueueError, atomic_json, canonical, digest, file_digest, file_lock, sync_directory

FORMAT = 'parallel-verification-v1'
INDEX_FORMAT = 'partitioned-base-index-v1'
REFERENCE_INDEX_FORMAT = 'partitioned-base-reference-index-v1'
MAX_LINE = 2 * 1024**2


def _read(path):
    return json.loads(Path(path).read_bytes())


def _relative(root, value):
    path = (Path(root) / value).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise QueueError('Verification artifact escapes its root')
    return path


def _sync(path):
    with Path(path).open('r+b') as handle:
        os.fsync(handle.fileno())


def _members(directory, count):
    values = np.fromfile(Path(directory) / 'remaining.u32', dtype='<u4')
    if len(values) and int(values.max()) >= count:
        raise QueueError('Prediction round ordinal is outside its design')
    unique = np.unique(values)
    if len(unique) != len(values):
        raise QueueError('Duplicate prediction round task')
    bitmap = np.zeros(count, dtype=np.uint8)
    bitmap[values] = 1
    return values, np.packbits(bitmap, bitorder='little')


def prepare(plan, mode, base_rounds, sl_rounds, directory, *, block_bytes=64 * 1024**2,
            sl_block_bytes=None, base_part=None):
    """Build only range/membership metadata; decoding is done by workers.

    ``final-base`` content-checks the stopped base rounds while SL still runs and
    publishes nothing. A ``final`` given its completed ``base_part`` then checks
    only SL chunks and joins both parts at reduction.
    """
    from . import prediction_workflow as flow
    from .base_seal import audit_sample, DEFAULT_SAMPLE_RATE
    if (mode not in ('base', 'index', 'final', 'final-base') or type(block_bytes) is not int or block_bytes < 1
            or (sl_block_bytes is not None and (mode != 'final' or type(sl_block_bytes) is not int or sl_block_bytes < 1))
            or (base_part is not None and mode != 'final')):
        raise QueueError('Invalid verification mode or range size')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / 'manifest.json'
    expected = {'format': FORMAT, 'mode': mode, 'plan_sha256': digest(plan),
        'base_rounds': [str(Path(p).resolve()) for p in base_rounds],
        'sl_rounds': [str(Path(p).resolve()) for p in sl_rounds], 'block_bytes': block_bytes}
    # Absent keys keep every earlier manifest's identity unchanged.
    if sl_block_bytes is not None: expected['sl_block_bytes'] = sl_block_bytes
    if base_part is not None:
        part = _completed_base_part(plan, base_part)
        if part['identity']['base_rounds'] != expected['base_rounds']:
            raise QueueError('Completed base audit covers other base rounds')
        expected['base_part'] = {'directory': str(Path(base_part).resolve()), 'manifest_sha256': digest(part)}
    if manifest_path.exists():
        existing = _read(manifest_path)
        if existing['identity'] != expected:
            raise QueueError('Verification snapshot changed; use an explicit new generation')
        check_sources(existing)
        return existing
    columns = list(dict.fromkeys(['panel_id', 'phase', 'pipeline_id', 'variant_id', 'base_library_id'] + plan['public_columns']))
    sources, chunks = [], []
    phase_sizes = {}
    sample, strata = set(), 0
    (directory / 'chunks').mkdir(exist_ok=True)
    if base_part is not None:
        phase_sizes['base'] = flow.PredictionDesign(plan['prediction_workflow'], 'base').count
    with ExitStack() as locks:
        phases = ([] if base_part is not None else [('base', base_rounds)]) + ([('sl', sl_rounds)] if mode == 'final' else [])
        for phase, rounds in phases:
            phase_identity = flow.phase_identity(plan, phase)
            design = flow.PredictionDesign(plan['prediction_workflow'], phase)
            phase_sizes[phase] = design.count
            phase_block = sl_block_bytes if phase == 'sl' and sl_block_bytes is not None else block_bytes
            if mode == 'base':
                sample, strata = audit_sample(design, rate=DEFAULT_SAMPLE_RATE, seed='base-audit:' + digest(plan))
            previous = []
            for source_index, raw in enumerate(rounds):
                root = Path(raw).resolve()
                locks.enter_context(file_lock(root / 'dispatcher.lock'))
                manifest = _read(root / 'manifest.json'); qid = digest(manifest)
                if (manifest['identity'] != phase_identity or flow._source_ids(manifest['sources']) != flow._source_ids(previous)
                        or _read(root / 'queue-id.json')['queue_id'] != qid):
                    raise QueueError('Prediction round identity or source chain changed')
                order, members = _members(root, design.count)
                if len(order) != manifest['count']:
                    raise QueueError('Prediction round task count changed')
                number = len(sources); member_path = directory / ('members-%d.bin' % number)
                member_path.write_bytes(members.tobytes()); _sync(member_path)
                journal = root / 'results.jsonl'
                size = journal.stat().st_size if journal.exists() else 0
                source = {'root': str(root), 'phase': phase, 'queue_id': qid, 'journal_bytes': size,
                          'members': member_path.name, 'members_sha256': file_digest(member_path),
                          'round_count': len(order), 'round_index': source_index}
                sources.append(source)
                for start in range(0, size, phase_block):
                    chunks.append({'index': len(chunks), 'kind': 'rows', 'source': number,
                                   'start': start, 'end': min(size, start + phase_block)})
                previous.append({'root': str(root), 'queue_id': qid})
    if mode == 'base':
        cache = Path(plan['prediction_workflow']['cache_root'])
        maps = sorted(p.relative_to(cache).as_posix() for p in (cache / 'sample-maps').glob('*.pcshard'))
        for at in range(0, len(maps), 512):
            chunks.append({'index': len(chunks), 'kind': 'maps', 'paths': maps[at:at + 512]})
    manifest = {'identity': expected, 'plan': plan, 'columns': columns, 'sources': sources,
                'phase_sizes': phase_sizes, 'chunks': chunks, 'sample_ordinals': sorted(sample),
                'sample_strata': strata, 'sample_rate': DEFAULT_SAMPLE_RATE,
                'cache_locations': _layout_identity(plan)}
    atomic_json(manifest_path, manifest)
    return manifest


def _completed_base_part(plan, directory):
    """The manifest of a finished ``final-base`` audit of this plan, or refusal."""
    directory = Path(directory); manifest = _read(directory / 'manifest.json')
    done = _read(directory / 'complete.json')
    if (manifest['identity']['mode'] != 'final-base' or manifest['identity']['plan_sha256'] != digest(plan)
            or done.get('manifest_sha256') != digest(manifest) or any(directory.glob('failure-*.json'))):
        raise QueueError('Base content audit is incomplete or belongs to another plan')
    return manifest


def _parts(directory, manifest):
    """Every audit this reduction joins, in publication order, each source-checked."""
    parts = []
    joined = manifest['identity'].get('base_part')
    if joined is not None:
        base = Path(joined['directory'])
        base_manifest = _completed_base_part(manifest['plan'], base)
        if digest(base_manifest) != joined['manifest_sha256']:
            raise QueueError('Joined base audit changed after the final audit was prepared')
        parts.append((base, base_manifest))
    parts.append((Path(directory), manifest))
    for _, part_manifest in parts:
        check_sources(part_manifest)
    return parts


def _layout_identity(plan):
    pointer = Path(plan['prediction_workflow']['cache_root']) / 'locations.json'
    return digest(_read(pointer)) if pointer.exists() else None


def check_sources(manifest):
    """Preserve the existing stopped-source identity/size boundary."""
    from . import prediction_workflow as flow
    plan = manifest['plan']
    if digest(plan) != manifest['identity']['plan_sha256'] or _layout_identity(plan) != manifest['cache_locations']:
        raise QueueError('Verification plan or immutable location generation changed')
    identities = {phase: flow.phase_identity(plan, phase) for phase in manifest['phase_sizes']}
    for source in manifest['sources']:
        root = Path(source['root']); journal = root / 'results.jsonl'; meta = _read(root / 'manifest.json')
        if (digest(meta) != source['queue_id'] or meta['identity'] != identities[source['phase']]
                or _read(root / 'queue-id.json')['queue_id'] != source['queue_id']
                or (journal.stat().st_size if journal.exists() else 0) != source['journal_bytes']):
            raise QueueError('Stopped verification source changed')


def _sealed_reference(cache, reference, context):
    from .prediction_cache import _index_path, _load_index
    reference = context.resolve(cache, reference)
    path = context.safe_path(cache, reference['path'])
    index, members = _load_index(_index_path(cache, reference['path']))
    if (not index.get('sealed') or index.get('path') != reference['path'] or path.stat().st_size != index['bytes']
            or (reference['offset'], reference['length'], reference['sha256']) not in members):
        raise QueueError('Prediction reference is missing from a sealed shard or the shard size changed')


def _receipt(directory, chunk, manifest, *, verify_artifacts=True):
    path = Path(directory) / 'chunks' / ('c%06d.json' % chunk['index'])
    if not path.exists():
        return None
    value = _read(path)
    if value.get('identity') != digest([manifest['identity'], chunk]):
        raise QueueError('Verification chunk belongs to another source range')
    if verify_artifacts:
        for item in value['artifacts'].values():
            artifact = _relative(directory, item['path'])
            if artifact.stat().st_size != item['bytes'] or file_digest(artifact) != item['sha256']:
                raise QueueError('Published verification artifact changed')
    return value


def verify_chunk(directory, chunk, manifest=None):
    """Atomic independent unit. Retrying a completed unit never decodes it again."""
    from . import prediction_workflow as flow
    from .prediction_cache import ReadContext, encode_index_json, _index_path, _load_index, safe_cache_path
    directory = Path(directory); manifest = manifest or _read(directory / 'manifest.json')
    old = _receipt(directory, chunk, manifest)
    if old is not None:
        return old
    started, cpu_started, io_started = time.time(), time.thread_time(), _io_counters()
    plan = manifest['plan']; cache = Path(plan['prediction_workflow']['cache_root'])
    prefix = 'c%06d' % chunk['index']; attempt = directory / 'chunks' / (prefix + '.' + uuid.uuid4().hex)
    ordinals, statuses = [], Counter(); audited = 0; artifacts = {}
    if chunk['kind'] == 'maps':
        for relative in chunk['paths']:
            index, _ = _load_index(_index_path(cache, relative))
            if not index.get('sealed') or index.get('path') != relative or safe_cache_path(cache, relative).stat().st_size != index['bytes']:
                raise QueueError('Sample-map shard is unsealed or resized')
        phase = 'base'
    else:
        source = manifest['sources'][chunk['source']]; phase = source['phase']
        design = flow.PredictionDesign(plan['prediction_workflow'], phase)
        identity = flow.phase_identity(plan, phase)
        members = _relative(directory, source['members']).read_bytes()
        if hashlib.sha256(members).hexdigest() != source['members_sha256']:
            raise QueueError('Verification membership bitmap changed')
        index_only = manifest['identity']['mode'] == 'index'
        validator = None if index_only else flow.ResultCacheValidator(identity, fast_reads=True)
        sample = set(manifest['sample_ordinals'])
        csv_path = attempt.with_suffix(attempt.suffix + '.csv')
        db_path = attempt.with_suffix(attempt.suffix + '.sqlite')
        with ExitStack() as stack:
            if validator is not None:
                stack.callback(validator.context.close)
            db = None; writer = None
            if manifest['identity']['mode'] in ('base', 'index'):
                db = stack.enter_context(closing(sqlite3.connect(db_path)))
                db.execute('PRAGMA synchronous=FULL')
                schema = 'task_id TEXT PRIMARY KEY, reference TEXT NOT NULL'
                if not index_only: schema += ', row BLOB NOT NULL'
                db.execute('CREATE TABLE records(' + schema + ')')
            else:
                csv_handle = stack.enter_context(csv_path.open('x', newline='', encoding='utf-8'))
                writer = csv.DictWriter(csv_handle, fieldnames=manifest['columns'], lineterminator='\n')
            path = Path(source['root']) / 'results.jsonl'
            if path.stat().st_size != source['journal_bytes']:
                raise QueueError('Stopped journal size changed before range read')
            handle = stack.enter_context(path.open('rb', buffering=1024 * 1024))
            if chunk['start']:
                handle.seek(chunk['start'] - 1)
                if handle.read(1) != b'\n':
                    discarded = handle.readline(MAX_LINE + 1)
                    if len(discarded) > MAX_LINE:
                        raise QueueError('Oversized journal line at chunk boundary')
            while handle.tell() < chunk['end']:
                line = handle.readline(MAX_LINE + 1)
                if len(line) > MAX_LINE:
                    raise QueueError('Oversized prediction result')
                if not line.endswith(b'\n'):
                    break
                entry = json.loads(line); row = entry['result']; ordinal = design.ordinal(row)
                if (entry['origin']['queue_id'] != source['queue_id'] or entry['task_id'] != design.task_at(ordinal).id
                        or not members[ordinal // 8] & (1 << (ordinal % 8))):
                    raise QueueError('Prediction result provenance mismatch')
                if not flow.validate_scientific_result(row, task_kind=flow.task_kind(identity, row)):
                    continue
                if row.get('algorithm_version') != design.panels[row['panel_id']][2]['cell_spec']['algorithm_version']:
                    raise QueueError('Prediction algorithm changed')
                if index_only:
                    from .base_seal import _reference_fields
                    _reference_fields(row['prediction_cache_ref'])
                elif manifest['identity']['mode'] in ('final', 'final-base') or ordinal in sample:
                    validator(row, sealed=True)
                    audited += 1
                else:
                    _sealed_reference(cache, row['prediction_cache_ref'], validator.context)
                ordinals.append(ordinal); statuses[row['status']] += 1
                if db is not None:
                    reference = row['prediction_cache_ref']
                    if index_only:
                        db.execute('INSERT INTO records VALUES (?,?)', (entry['task_id'], canonical(reference).decode()))
                    else:
                        encoded = (encode_index_json({k: v for k, v in row.items() if k != 'prediction_cache_ref'})
                            if plan['prediction_workflow'].get('storage', {}).get('layout') == 'shared-v1' else canonical(row).decode())
                        db.execute('INSERT INTO records VALUES (?,?,?)', (entry['task_id'], canonical(reference).decode(), encoded))
                else:
                    writer.writerow({k: row.get(k, '') for k in manifest['columns']})
            if path.stat().st_size != source['journal_bytes']:
                raise QueueError('Stopped journal changed during range read')
            if db is not None:
                db.commit()
            else:
                csv_handle.flush(); os.fsync(csv_handle.fileno())
        values = np.asarray(ordinals, dtype='<u4')
        if len(np.unique(values)) != len(values):
            raise QueueError('Duplicate prediction success within a chunk')
        ordinal_path = attempt.with_suffix(attempt.suffix + '.u32'); values.tofile(ordinal_path)
        outputs = {'ordinals': ordinal_path, 'records' if db is not None else 'csv': db_path if db is not None else csv_path}
        for name, temporary in outputs.items():
            _sync(temporary)
            final = directory / 'chunks' / (prefix + temporary.suffix)
            os.replace(temporary, final)
            artifacts[name] = {'path': str(final.relative_to(directory)), 'bytes': final.stat().st_size,
                               'sha256': file_digest(final)}
    value = {'identity': digest([manifest['identity'], chunk]), 'phase': phase, 'rows': len(ordinals),
             'statuses': dict(statuses), 'audited': audited, 'artifacts': artifacts,
             'map_shards': len(chunk.get('paths', [])), 'hostname': os.uname().nodename if hasattr(os, 'uname') else os.environ.get('COMPUTERNAME')}
    # Operational only: which of disk and decoding bounds a chunk, never part of any receipt identity.
    finished, io_finished = time.time(), _io_counters()
    value['timing'] = {'started_epoch': started, 'finished_epoch': finished,
        'wall_seconds': finished - started, 'thread_cpu_seconds': time.thread_time() - cpu_started,
        'journal_range_bytes': chunk['end'] - chunk['start'] if chunk['kind'] == 'rows' else 0,
        'process_io_delta': ({k: io_finished[k] - io_started[k] for k in io_finished}
                             if io_started and io_finished else None)}
    atomic_json(directory / 'chunks' / (prefix + '.json'), value)
    return value


def _io_counters():
    """Linux per-process read counters; the process verifies one chunk at a time."""
    try:
        fields = dict(line.split(':', 1) for line in Path('/proc/self/io').read_text().splitlines())
        return {k: int(fields[k]) for k in ('rchar', 'read_bytes')}
    except (OSError, ValueError, KeyError):
        return None


def reduce(directory):
    """Global uniqueness/coverage, then a single atomic publication."""
    from . import prediction_workflow as flow
    directory = Path(directory); manifest = _read(directory / 'manifest.json'); plan = manifest['plan']
    parts = _parts(directory, manifest)
    output = Path(plan['launch']['output']); mode = manifest['identity']['mode']
    seen = {phase: np.zeros(count, dtype=np.bool_) for phase, count in manifest['phase_sizes'].items()}
    routes = np.full(manifest['phase_sizes']['base'], np.iinfo(np.uint32).max, dtype='<u4') if mode in ('base', 'index') else None
    received = []; statuses = Counter(); audited = maps = 0
    for part, part_manifest in parts:
        got = {}
        for chunk in part_manifest['chunks']:
            receipt = _receipt(part, chunk, part_manifest)
            if receipt is None:
                raise QueueError('Missing verification chunk; publication is blocked')
            got[chunk['index']] = receipt
            audited += receipt['audited']; maps += receipt['map_shards']
            statuses.update(receipt['statuses'])
        received.append((part, part_manifest, got))
    receipts = received[-1][2]
    # Publication order is the joined base audit's chunks, then this audit's.
    everything = {(n, i): r for n, (_, _, got) in enumerate(received) for i, r in got.items()}
    for part, part_manifest, got in received:
        for index, source in enumerate(part_manifest['sources']):
            phase = source['phase']; order, members = _members(source['root'], len(seen[phase]))
            if hashlib.sha256(members.tobytes()).hexdigest() != source['members_sha256']:
                raise QueueError('Stopped round membership changed')
            if np.any(seen[phase][order]):
                raise QueueError('Later round reclaimed a previously accepted task')
            for chunk in (c for c in part_manifest['chunks'] if c.get('source') == index):
                receipt = got[chunk['index']]
                values = np.fromfile(_relative(part, receipt['artifacts']['ordinals']['path']), dtype='<u4')
                if (len(values) != receipt['rows'] or len(np.unique(values)) != len(values)
                        or (len(values) and int(values.max()) >= len(seen[phase])) or np.any(seen[phase][values])):
                    raise QueueError('Duplicate or invalid verification coverage')
                seen[phase][values] = True
                if routes is not None:
                    routes[values] = chunk['index']
    counts = {phase: int(bits.sum()) for phase, bits in seen.items()}
    if counts != manifest['phase_sizes']:
        raise QueueError('Prediction workflow coverage incomplete')
    if mode == 'final-base':
        # Evidence for a later final audit only; the science is certified there.
        receipt = {'format': FORMAT, 'stage': 'final-base', 'rows': counts['base'], 'statuses': dict(statuses),
                   'chunks': len(receipts), 'sources': [{'root': s['root'], 'queue_id': s['queue_id']} for s in manifest['sources']]}
        atomic_json(directory / 'complete.json', {'format': FORMAT, 'manifest_sha256': digest(manifest), 'receipt': receipt})
        return receipt
    if mode in ('base', 'index'):
        if mode == 'base' and audited != len(manifest['sample_ordinals']):
            raise QueueError('Stratified base sample coverage differs')
        routing = directory / 'base-routing.u32'; routes.tofile(routing); _sync(routing)
        parts = {str(i): {**r['artifacts']['records'],
            'path': str(_relative(directory, r['artifacts']['records']['path']).relative_to(output))}
            for i, r in receipts.items() if 'records' in r['artifacts']}
        index_format = REFERENCE_INDEX_FORMAT if mode == 'index' else INDEX_FORMAT
        index = {'format': index_format, 'plan_sha256': digest(plan), 'workflow_sha256': digest(plan['prediction_workflow']),
                 'rows': counts['base'], 'route': {'path': str(routing.relative_to(output)), 'bytes': routing.stat().st_size},
                 'parts': parts}
        index_path = output / 'base-records.manifest.json'; atomic_json(index_path, index)
        if mode == 'index':
            receipt = {'format': flow.BASE_READY_FORMAT, 'index_ready': True, 'audit_complete': False,
                'plan_sha256': digest(plan), 'workflow_sha256': digest(plan['prediction_workflow']),
                'rows': counts['base'], 'expected_rows': counts['base'],
                'sources': [{'root': s['root'], 'queue_id': s['queue_id']} for s in manifest['sources']],
                'records_index_format': index_format, 'records_index_manifest_sha256': digest(index),
                'checks': 'accepted-journal provenance and unambiguous complete lookup index only; no prediction arrays read or full data audit',
                'indexed_prediction_array_reads': 0}
            atomic_json(output / 'base-input-ready.json', receipt)
            atomic_json(directory / 'complete.json', {'format': FORMAT, 'manifest_sha256': digest(manifest), 'receipt': receipt})
            return receipt
        receipt = {'format': flow.FORMAT, 'complete': True, 'plan_sha256': digest(plan),
            'workflow_sha256': digest(plan['prediction_workflow']), 'rows': counts['base'], 'expected_rows': counts['base'],
            'panels': [p['panel_id'] for p in plan['prediction_workflow']['panels']], 'statuses': dict(statuses),
            'sources': [{'root': s['root'], 'queue_id': s['queue_id']} for s in manifest['sources']],
            'integrity': 'record-sha256-v1', 'score_complete': True, 'prediction_cache_complete': True, 'oof_complete': True,
            'unfinished_leases': 0, 'unsubmitted_required_results': 0, 'unresolved_failures': 0,
            'records_index_format': INDEX_FORMAT, 'records_index_manifest_sha256': digest(index),
            'verification': {'format': FORMAT, 'chunks': len(receipts), 'map_shards': maps,
                'audit_sample': {'rate': manifest['sample_rate'], 'rows': audited, 'strata': manifest['sample_strata']},
                'checks': 'global unique keys/provenance/coverage; every reference sealed; fixed stratified full-content sample; every SL input checked on read'}}
        atomic_json(output / 'base-audit-sample.json', {'format': FORMAT, 'ordinals': manifest['sample_ordinals']})
        atomic_json(output / 'base-verified.json', receipt)
    else:
        base = publish_final_base_receipt(manifest, everything, counts)
        final = output / ('final-' + uuid.uuid4().hex + '.tmp')
        from io import StringIO
        text = StringIO(newline=''); csv.writer(text, lineterminator='\n').writerow(manifest['columns'])
        header = text.getvalue().encode('utf-8'); hasher = hashlib.sha256()
        with final.open('xb') as handle:
            handle.write(header); hasher.update(header)
            for part, part_manifest, got in received:
                for chunk in part_manifest['chunks']:
                    fragment = _relative(part, got[chunk['index']]['artifacts']['csv']['path'])
                    with fragment.open('rb') as source:
                        while data := source.read(8 * 1024**2):
                            handle.write(data); hasher.update(data)
            handle.flush(); os.fsync(handle.fileno())
        for _, part_manifest in parts:
            check_sources(part_manifest)
        os.replace(final, output / 'final.csv'); sync_directory(output)
        receipt = {'format': flow.FORMAT, 'complete': True, 'phase_rows': counts, 'rows': sum(counts.values()),
            'plan_sha256': digest(plan), 'base_verified_sha256': digest(base), 'final_csv_sha256': hasher.hexdigest(),
            'final_csv_bytes': (output / 'final.csv').stat().st_size, 'score_complete': True,
            'prediction_cache_complete': True, 'oof_complete': True,
            'verification': {'format': FORMAT, 'chunks': len(everything), 'content_checks': 'every accepted record and required sample map'}}
        atomic_json(output / 'verified.json', receipt)
    atomic_json(directory / 'complete.json', {'format': FORMAT, 'manifest_sha256': digest(manifest), 'receipt': receipt})
    return receipt


def publish_final_base_receipt(manifest, receipts, counts):
    """Deferred runs certify base data only after final content/coverage checks."""
    from . import prediction_workflow as flow
    plan = manifest['plan']; contract = plan['prediction_workflow']
    if not flow.deferred_base_audit(contract):
        return flow.verify_base_receipt(plan)
    ready = flow.verify_base_input_receipt(plan)
    statuses = Counter()
    for receipt in receipts.values():
        if receipt['phase'] == 'base': statuses.update(receipt['statuses'])
    value = {'format': flow.FORMAT, 'complete': True, 'audit_complete': True,
        'plan_sha256': digest(plan), 'workflow_sha256': digest(contract),
        'rows': counts['base'], 'expected_rows': manifest['phase_sizes']['base'],
        'panels': [p['panel_id'] for p in contract['panels']], 'statuses': dict(statuses),
        'sources': ready['sources'], 'integrity': 'record-sha256-v1',
        'score_complete': True, 'prediction_cache_complete': True, 'oof_complete': True,
        'unfinished_leases': 0, 'unsubmitted_required_results': 0, 'unresolved_failures': 0,
        'records_index_format': ready['records_index_format'],
        'records_index_manifest_sha256': ready['records_index_manifest_sha256'],
        'verification': {'format': FORMAT, 'stage': 'final', 'base_input_receipt_sha256': digest(ready),
            'checks': 'all base records and sample maps content-verified; global unique keys and complete coverage'}}
    path = Path(plan['launch']['output']) / 'base-verified.json'
    if path.exists() and _read(path) != value:
        raise QueueError('Deferred base audit receipt changed during final publication')
    atomic_json(path, value)
    return value


class BaseRecordIndex:
    """Read-only routing to worker-built indexes; no serial 16M-row DB rebuild."""
    def __init__(self, contract, receipt, *, verify_files=False):
        from .prediction_workflow import PredictionDesign
        self.root = Path(contract['output_root']); self.index = _read(self.root / 'base-records.manifest.json')
        if (self.index.get('format') not in (INDEX_FORMAT, REFERENCE_INDEX_FORMAT)
                or self.index['format'] != receipt.get('records_index_format')
                or digest(self.index) != receipt['records_index_manifest_sha256']
                or self.index['workflow_sha256'] != digest(contract)
                or self.index['plan_sha256'] != receipt['plan_sha256'] or self.index['rows'] != receipt['rows']):
            raise QueueError('Partitioned base index identity changed')
        self.reference_only = self.index['format'] == REFERENCE_INDEX_FORMAT
        route = self.index['route']; path = _relative(self.root, route['path'])
        if path.stat().st_size != route['bytes'] or route['bytes'] != self.index['rows'] * 4:
            raise QueueError('Base routing index is missing or resized')
        if verify_files:
            for entry in self.index['parts'].values():
                if _relative(self.root, entry['path']).stat().st_size != entry['bytes']:
                    raise QueueError('Base index partition is missing or resized')
        self.routes = np.memmap(path, mode='r', dtype='<u4')
        self.design = PredictionDesign(contract, 'base'); self.connections = OrderedDict()

    def lookup(self, task):
        ordinal = self.design.ordinal(asdict(task)); part = str(int(self.routes[ordinal]))
        entry = self.index['parts'].get(part)
        if entry is None:
            raise QueueError('Base routing index has no required task')
        db = self.connections.pop(part, None)
        if db is None:
            path = _relative(self.root, entry['path'])
            if path.stat().st_size != entry['bytes']:
                raise QueueError('Base index partition is missing or resized')
            db = sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1', uri=True, check_same_thread=False)
        self.connections[part] = db
        while len(self.connections) > 16:
            self.connections.popitem(last=False)[1].close()
        if self.reference_only:
            found = db.execute('SELECT reference FROM records WHERE task_id=?', (task.id,)).fetchone()
            return (found[0], None) if found is not None else None
        return db.execute('SELECT reference,row FROM records WHERE task_id=?', (task.id,)).fetchone()

    def close(self):
        for db in self.connections.values(): db.close()
        self.connections.clear()
        self.routes._mmap.close()


def run(directory, workers, *, launch=None):
    """The allocation owner holds the read boundary across srun and reduction."""
    from .prediction_cache import seal_stopped_writers
    directory = Path(directory); manifest = _read(directory / 'manifest.json')
    with ExitStack() as stack:
        stack.enter_context(file_lock(directory / 'verification.lock'))
        cache = Path(manifest['plan']['prediction_workflow']['cache_root'])
        stack.enter_context(file_lock(cache / '.layout.lock'))
        joined = manifest['identity'].get('base_part')
        sources = manifest['sources'] + (_read(Path(joined['directory']) / 'manifest.json')['sources'] if joined else [])
        for source in sources:
            stack.enter_context(file_lock(Path(source['root']) / 'dispatcher.lock'))
        check_sources(manifest)
        # A base audit overlaps live SL writers; the index run already sealed every
        # stopped base writer, and each record here must be read from a sealed shard.
        if manifest['identity']['mode'] != 'final-base':
            seal_stopped_writers(cache, writer_revoked=True)
        if launch is None:
            command = ['srun', '--ntasks=' + str(workers), '--cpus-per-task=1', '--ntasks-per-core=1',
                       '--distribution=cyclic', '--kill-on-bad-exit=1',
                       sys.executable, '-m', __name__, 'worker', str(directory)]
            subprocess.run(command, check=True)
        else:
            launch(directory, workers)
        try:
            return reduce(directory)
        except QueueError as exc:
            # Integrity refusals are deterministic: record one, as a failed chunk
            # does, so the controller stops instead of resubmitting the check.
            atomic_json(directory / 'failure-reduce.json', {'stage': 'reduce', 'error': repr(exc)})
            raise


def worker(directory, rank, workers):
    directory = Path(directory); manifest = _read(directory / 'manifest.json')
    if not 0 <= rank < workers or workers < 1:
        raise QueueError('Invalid verification rank')
    for chunk in manifest['chunks'][rank::workers]:
        try:
            verify_chunk(directory, chunk, manifest)
        except Exception as exc:
            atomic_json(directory / ('failure-%d.json' % rank), {'chunk': chunk['index'], 'error': repr(exc)})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['worker'])
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    worker(args.directory, int(os.environ['SLURM_PROCID']), int(os.environ['SLURM_NTASKS']))


if __name__ == '__main__': main()
