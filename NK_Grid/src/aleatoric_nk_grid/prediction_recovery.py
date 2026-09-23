"""Controller-owned immutable recovery deltas; no full index copy each round."""
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import time
import uuid

from .prediction_cache import seal_stopped_writers, safe_cache_path, _atomic_json, _sync_directory
from .shared_queue import QueueError, digest


# A delta is committed and published every CHUNK_SOURCES shard indexes, so a
# controller killed at its time limit keeps what it indexed instead of making
# its successor start over. INDEX_SECONDS then bounds what one round spends
# here: recovery is an optimisation, and a source left for a later round costs
# a recomputation, never a wrong result.
CHUNK_SOURCES = 256
INDEX_SECONDS = 600.


def _write_delta(directory, batch):
    """One immutable delta over a batch of not yet indexed shard indexes."""
    path = directory / (uuid.uuid4().hex + '.sqlite')
    sources = 0; count = 0
    with closing(sqlite3.connect(path)) as db:
        db.execute('PRAGMA synchronous=FULL')
        db.execute('CREATE TABLE sources(root_kind TEXT, path TEXT, PRIMARY KEY(root_kind,path)) WITHOUT ROWID')
        db.execute('CREATE TABLE records(root_kind TEXT, identity TEXT, content_sha256 TEXT, reference TEXT, reference_sha256 TEXT, PRIMARY KEY(root_kind,identity,reference_sha256)) WITHOUT ROWID')
        for kind, root, index_path in batch:
            index = json.loads(index_path.read_bytes())
            relative = index.get('path', '')
            if not relative.startswith(('shards/', 'meta-results/')): continue
            if not index.get('sealed') or safe_cache_path(root, relative).stat().st_size != index['bytes']:
                raise QueueError('Recovery source is unsealed or changed')
            db.execute('INSERT INTO sources VALUES (?,?)', (kind, index_path.name)); sources += 1
            rows = []
            for ref in index['records']:
                if ref['path'] != relative: raise QueueError('Recovery reference path differs from index')
                rows.append((kind, ref['identity'], ref.get('content_sha256'), json.dumps(ref), ref['sha256']))
            db.executemany('INSERT OR IGNORE INTO records VALUES (?,?,?,?,?)', rows)
            count += len(rows)
        db.commit()
    if not sources:
        path.unlink(); return None, 0, 0
    with path.open('r+b') as handle: os.fsync(handle.fileno())
    _sync_directory(directory)
    return path, sources, count


def build_incremental(plan, *, seconds=INDEX_SECONDS, chunk=CHUNK_SOURCES, clock=time.monotonic):
    contract = plan['prediction_workflow']; output = Path(contract['output_root'])
    directory = output / 'recovery-indexes'; directory.mkdir(exist_ok=True)
    head = directory / 'head.json'
    old = json.loads(head.read_bytes()) if head.exists() else None
    if old and (old['plan_sha256'] != digest(plan) or old['workflow_sha256'] != digest(contract)):
        raise QueueError('Recovery generation belongs to another submission')
    catalogs = list(old['catalogs']) if old else []
    known = set()
    for item in catalogs:
        path = Path(item['path'])
        if not path.resolve().is_relative_to(directory.resolve()) or not path.is_file():
            raise QueueError('Immutable recovery delta changed')
        with closing(sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1', uri=True)) as db:
            known.update(db.execute('SELECT root_kind,path FROM sources'))
    pending = []
    for kind, root in {'main': Path(contract['cache_root']),
                       'fold': output / 'prediction-training-checkpoints'}.items():
        if not root.exists(): continue
        seal_stopped_writers(root, writer_revoked=True)
        for index_path in (root / 'indexes').glob('*.pcshard.json'):
            if (kind, index_path.name) in known: continue
            pending.append((index_path.stat().st_mtime, kind, root, index_path))
    # Newest first: an orphan this allocation can still reuse was written by the
    # allocation that just stopped, not by a phase that ended long ago.
    pending.sort(key=lambda item: -item[0])
    deadline = clock() + seconds
    count = 0

    def publish():
        result = {'format': 'prediction-recovery-v2', 'catalogs': catalogs,
                  'plan_sha256': digest(plan), 'workflow_sha256': digest(contract),
                  'records': sum(c['records'] for c in catalogs), 'new_records': count,
                  'pending_sources': len(pending), 'contains_legacy_fold_identities': False}
        _atomic_json(head, result)
        return result

    while pending:
        batch = [(kind, root, index_path) for _, kind, root, index_path in pending[:chunk]]
        remaining = pending[chunk:]
        path, sources, records = _write_delta(directory, batch)
        pending = remaining
        if path is not None:
            catalogs.append({'path': str(path.resolve()), 'integrity': 'record-sha256-v1',
                             'records': records})
            count += records
        publish()
        if clock() >= deadline: break
    return publish()
