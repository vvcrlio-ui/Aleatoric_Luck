"""Bounded full-queue synthetic journal recovery capacity, never training.

The fixture writer emits the documented hash-chained journal format, flushing
in batches; it deliberately does not measure per-result RPC/fsync throughput.
The unmodified real Dispatcher rebuilds and validates the entire fixture.
"""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time

from aleatoric_nk_grid.shared_queue import (
    Dispatcher, ModelTask, atomic_json, canonical, digest, file_digest,
)


def now():
    return datetime.now(timezone.utc).isoformat()


def synthetic_result(task, payload_bytes):
    return {**asdict(task), 'status': 'ok', 'mse': .25, 'rmse': .5, 'mae': .4,
            'validation_only': True, 'task_fingerprint': task.id,
            'synthetic_payload': 'x' * payload_bytes}


def prefix_digest(path, size):
    hashed = hashlib.sha256()
    with path.open('rb') as f:
        remaining = size
        while remaining:
            block = f.read(min(1024 * 1024, remaining))
            if not block:
                raise AssertionError('Journal prefix shortened')
            hashed.update(block)
            remaining -= len(block)
    return hashed.hexdigest()


def main(args):
    assert sys.platform == 'linux'
    assert args.output.resolve().is_relative_to(Path('/valhalla'))
    assert args.source_queue.resolve() != (args.output / 'queue').resolve()
    assert args.scratch.resolve().parent == Path('/dev/shm')
    assert args.heartbeats_per_key >= 0 and 1024 <= args.payload_bytes <= 4096
    args.output.mkdir(parents=True, exist_ok=False)
    args.scratch.mkdir(parents=True, exist_ok=False)
    report = dict(started_at_utc=now(), validation_only=True, synthetic_results=True,
        source_queue=str(args.source_queue), heartbeats_per_key=args.heartbeats_per_key,
        payload_bytes=args.payload_bytes, probe_sha256=file_digest(Path(__file__)),
        engine_sha256=file_digest(Path(__import__('aleatoric_nk_grid').__file__).parent / 'shared_queue.py'),
        scratch_backend='node-local tmpfs within Slurm memory allocation',
        slurm_memory_mib=os.environ.get('SLURM_MEM_PER_NODE'),
        limitations='Synthetic complete-key lifetime with the stated heartbeat/payload bound. '
                    'Buffered fixture generation is not a per-result durability or RPC benchmark. '
                    'No production recovery, numerical results, all-round worst-case retry volume, '
                    'or Slurm continuation integration is established.')

    def mark(phase, **values):
        report.update(phase=phase, updated_at_utc=now(), **values)
        atomic_json(args.output / 'progress.json', report)
        print(json.dumps({'phase': phase, **values}), flush=True)

    try:
        assert shutil.disk_usage(args.scratch).free >= 32 * 1024**3
        assert shutil.disk_usage(args.output).free >= 40 * 1024**3
        source_manifest = json.loads((args.source_queue / 'manifest.json').read_bytes())
        assert source_manifest['identity'].get('validation_only') is True
        source_tasks_hash = file_digest(args.source_queue / 'tasks.jsonl')
        assert source_tasks_hash == source_manifest['tasks_sha256']
        count = source_manifest['count']
        assert count == args.expected_keys
        model_counts = Counter()

        def tasks():
            with (args.source_queue / 'tasks.jsonl').open('rb') as f:
                for line in f:
                    item = json.loads(line)
                    task = ModelTask(**item['task'])
                    model_counts[task.model] += 1
                    yield task, item['cost']

        began = time.monotonic()
        queue_root = args.output / 'queue'
        queue_id = Dispatcher.create(queue_root, tasks(), identity={
            'validation_only': True, 'synthetic_results': True,
            'source_tasks_sha256': source_tasks_hash, 'probe': 'full-queue-journal-recovery',
            'job': os.environ['SLURM_JOB_ID']}, lease_seconds=300)
        assert sum(model_counts.values()) == count
        mark('writing_synthetic_journal', expected_keys=count,
             source_tasks_sha256=source_tasks_hash,
             source_manifest_sha256=file_digest(args.source_queue / 'manifest.json'),
             copied_plan_seconds=time.monotonic() - began, model_counts=dict(model_counts),
             manifest_bytes=(queue_root / 'tasks.jsonl').stat().st_size)
        journal = queue_root / 'events.jsonl'
        sequence = 0
        previous = '0' * 64
        journal_hash = hashlib.sha256()
        began = time.monotonic()
        with journal.open('xb', buffering=1024 * 1024) as out:
            def emit(event):
                nonlocal sequence, previous
                body = {'sequence': sequence, 'previous': previous, 'queue_id': queue_id, 'event': event}
                previous = digest(body)
                data = canonical({'body': body, 'sha256': previous}) + b'\n'
                out.write(data)
                journal_hash.update(data)
                sequence += 1

            emit({'kind': 'restart', 'epoch': 'synthetic-initial-epoch'})
            with (queue_root / 'tasks.jsonl').open('rb') as f:
                for index, line in enumerate(f):
                    task = ModelTask(**json.loads(line)['task'])
                    token = 'synthetic-initial-epoch:' + str(index)
                    emit({'kind': 'lease', 'id': task.id, 'worker': 'fixture-' + str(index % 698),
                          'token': token, 'expiry': 1000.0})
                    for beat in range(args.heartbeats_per_key):
                        emit({'kind': 'heartbeat', 'id': task.id, 'expiry': 1020.0 + 20 * beat})
                    emit({'kind': 'result', 'id': task.id, 'token': token,
                          'result': synthetic_result(task, args.payload_bytes)})
                    if (index + 1) % 100000 == 0:
                        out.flush()
                        os.fsync(out.fileno())
                        mark('writing_synthetic_journal', generated_keys=index + 1,
                             generated_events=sequence, generated_bytes=out.tell())
            out.flush()
            os.fsync(out.fileno())
        assert index + 1 == count
        original_size = journal.stat().st_size
        assert sequence == 1 + count * (2 + args.heartbeats_per_key)
        with journal.open('ab') as out:
            out.write(b'{"body":')  # A deliberately uncommitted final frame.
            out.flush()
            os.fsync(out.fileno())
        mark('rebuilding_and_replaying', fixture_generation_seconds=time.monotonic() - began,
             generated_keys=count, journal_events=sequence, journal_bytes=original_size,
             complete_prefix_sha256=journal_hash.hexdigest(), truncated_tail_bytes=8)
        began = time.monotonic()
        with Dispatcher(queue_root, scratch=args.scratch) as queue:
            replay_seconds = time.monotonic() - began
            stats = queue.stats()
            assert stats['total'] == stats.get('done') == count
            assert not any(stats.get(k, 0) for k in ('pending', 'leased', 'failed', 'exhausted'))
            assert queue.sequence == sequence + 1
            with journal.open('rb') as f:
                f.seek(original_size)
                restart = json.loads(f.readline())
                assert restart['body']['sequence'] == sequence
                assert restart['body']['event']['kind'] == 'restart'
                assert not f.read(1)
            mark('verifying_all_replayed_results', rebuild_and_replay_seconds=replay_seconds,
                 sqlite_bytes=queue.db_path.stat().st_size, truncated_tail_recovered=True)
            verified = 0
            result_hash = hashlib.sha256()
            began = time.monotonic()
            for row in queue.db.execute('SELECT id,task,result,state FROM tasks ORDER BY id'):
                task = ModelTask(**json.loads(row['task']))
                expected = synthetic_result(task, args.payload_bytes)
                assert row['id'] == task.id and row['state'] == 'done'
                assert json.loads(row['result']) == expected
                result_hash.update(canonical(expected) + b'\n')
                verified += 1
            assert verified == count
            verification_seconds = time.monotonic() - began
            assert prefix_digest(journal, original_size) == journal_hash.hexdigest()
            mark('complete', passed=True, all_results_verified=verified,
                 ordered_synthetic_result_sha256=result_hash.hexdigest(),
                 result_verification_seconds=verification_seconds,
                 complete_journal_prefix_unchanged=True,
                 process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                 memory_caveat='Process RSS excludes tmpfs SQLite pages; include sqlite_bytes separately.')
        atomic_json(args.output / 'report.json', report)
    except BaseException as exc:
        mark('failed', passed=False, error=repr(exc))
        raise
    finally:
        assert args.scratch.parent == Path('/dev/shm') and args.scratch.name.startswith('ffc-journal-volume-')
        shutil.rmtree(args.scratch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-queue', 'output', 'scratch'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-keys', type=int, required=True)
    parser.add_argument('--heartbeats-per-key', type=int, default=2)
    parser.add_argument('--payload-bytes', type=int, default=1536)
    main(parser.parse_args())
