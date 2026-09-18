"""Controller-owned immutable recovery deltas; no full index copy each round."""
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import uuid

from .prediction_cache import seal_stopped_writers, safe_cache_path, _atomic_json, _sync_directory
from .shared_queue import QueueError, digest, file_digest


def build_incremental(plan):
    contract = plan['prediction_workflow']; output = Path(contract['output_root'])
    directory = output / 'recovery-indexes'; directory.mkdir(exist_ok=True)
    head = directory / 'head.json'
    old = json.loads(head.read_bytes()) if head.exists() else None
    if old and (old['plan_sha256'] != digest(plan) or old['workflow_sha256'] != digest(contract)):
        raise QueueError('Recovery generation belongs to another submission')
    catalogs = old['catalogs'] if old else []
    known = set()
    for item in catalogs:
        path = Path(item['path'])
        if not path.resolve().is_relative_to(directory.resolve()) or file_digest(path) != item['sha256']:
            raise QueueError('Immutable recovery delta changed')
        with closing(sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1', uri=True)) as db:
            known.update(db.execute('SELECT root_kind,path FROM sources'))
    path = directory / (uuid.uuid4().hex + '.sqlite')
    count = 0; source_count = 0
    with closing(sqlite3.connect(path)) as db:
        db.execute('PRAGMA synchronous=FULL')
        db.execute('CREATE TABLE sources(root_kind TEXT, path TEXT, PRIMARY KEY(root_kind,path)) WITHOUT ROWID')
        db.execute('CREATE TABLE records(root_kind TEXT, identity TEXT, content_sha256 TEXT, reference TEXT, reference_sha256 TEXT, PRIMARY KEY(root_kind,identity,reference_sha256)) WITHOUT ROWID')
        for kind, root in {'main': Path(contract['cache_root']),
                           'fold': output / 'prediction-training-checkpoints'}.items():
            if not root.exists(): continue
            seal_stopped_writers(root, writer_revoked=True)
            for index_path in (root / 'indexes').glob('*.pcshard.json'):
                if (kind, index_path.name) in known: continue
                index = json.loads(index_path.read_bytes())
                relative = index.get('path', '')
                if not relative.startswith(('shards/', 'meta-results/')): continue
                if not index.get('sealed') or safe_cache_path(root, relative).stat().st_size != index['bytes']:
                    raise QueueError('Recovery source is unsealed or changed')
                db.execute('INSERT INTO sources VALUES (?,?)', (kind, index_path.name)); source_count += 1
                for ref in index['records']:
                    if ref['path'] != relative: raise QueueError('Recovery reference path differs from index')
                    db.execute('INSERT OR IGNORE INTO records VALUES (?,?,?,?,?)',
                        (kind, ref['identity'], ref.get('content_sha256'), json.dumps(ref), ref['sha256']))
                    count += 1
        db.commit()
    if source_count:
        with path.open('r+b') as handle: os.fsync(handle.fileno())
        _sync_directory(directory)
        catalogs = [*catalogs, {'path': str(path.resolve()), 'sha256': file_digest(path), 'records': count}]
    else:
        path.unlink()
    result = {'format': 'prediction-recovery-v2', 'catalogs': catalogs,
              'plan_sha256': digest(plan), 'workflow_sha256': digest(contract),
              'records': sum(c['records'] for c in catalogs), 'new_records': count,
              'contains_legacy_fold_identities': False}
    _atomic_json(head, result)
    return result
