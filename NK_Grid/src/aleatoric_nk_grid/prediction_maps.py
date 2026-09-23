"""Plan-owned immutable maps. Only the preparation controller writes the catalog.

Selectors locate maps; the identity is always the complete typed array content.
Workers neither hash large map arrays nor write a shared database. Catalog and
shards are published before the first queue can be consumed.
"""
from collections import OrderedDict
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import uuid

import numpy as np

from .prediction_cache import (PredictionCacheWriter, cache_identity, read_record,
                               _atomic_json, _sync_directory)
from .shared_queue import QueueError, digest, file_lock

GROUPS = {'training': ('train_ids', 'train_positions', 'y_train', 'oof_fold'),
          'evaluation': ('holdout_ids', 'holdout_positions', 'y_holdout'),
          'features': ('feature_names', 'source_names')}


def selector(panel, seed, draw, n, k, folds, kind):
    # The selector is scoped by the full frozen contract in the receipt. It is
    # not a content identity and never licenses reuse in another submission.
    return json.dumps([panel, int(seed), int(draw), kind,
                       int(n) if kind == 'training' else int(k) if kind == 'features' else 0,
                       int(folds) if kind == 'training' else 0], separators=(',', ':'))


def prepare_maps(plan, *, repo_root, session_factory=None):
    from .execution_contract import CellExecutionSpec
    if session_factory is None:
        from .nk_grid import NKGridExecutionSession
    from .prediction_worker import array_identity
    contract = plan['prediction_workflow']
    if contract.get('storage', {}).get('layout') != 'shared-v1':
        return None
    root = Path(contract['cache_root'])
    receipt_path = root / 'maps-ready.json'
    catalog = root / 'maps.sqlite'
    with file_lock(root / '.maps.lock'):
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_bytes())
            if receipt['workflow_sha256'] != digest(contract) or not catalog.is_file():
                raise QueueError('Shared map catalog differs from its sealed contract')
            return receipt
        temporary = root / ('maps-' + uuid.uuid4().hex + '.sqlite.tmp')
        with closing(sqlite3.connect(temporary)) as db:
            db.execute('PRAGMA synchronous=FULL')
            db.execute('CREATE TABLE maps(identity TEXT PRIMARY KEY, reference TEXT NOT NULL, descriptors TEXT NOT NULL) WITHOUT ROWID')
            db.execute('CREATE TABLE selectors(selector TEXT PRIMARY KEY, identity TEXT NOT NULL) WITHOUT ROWID')
            with PredictionCacheWriter(root, writer_id='plan-maps',
                    shard_target_mib=contract['storage'].get('shard_target_mib', 128)) as writer:
                def save(panel, seed, draw, n, k, folds, kind, arrays):
                    selected = {key: arrays[key] for key in GROUPS[kind] if key in arrays}
                    descriptors = array_identity(selected)
                    identity = {'sample_map_content': cache_identity({'metadata': {'kind': kind}, 'arrays': descriptors})}
                    key = cache_identity(identity)
                    if db.execute('SELECT 1 FROM maps WHERE identity=?', (key,)).fetchone() is None:
                        reference = writer.append(identity, selected, {'kind': kind}, kind='sample_map')
                        db.execute('INSERT INTO maps VALUES (?,?,?)', (key, json.dumps(reference), json.dumps(descriptors)))
                    db.execute('INSERT INTO selectors VALUES (?,?)',
                               (selector(panel, seed, draw, n, k, folds, kind), key))
                for panel in contract['panels']:
                    factory = session_factory or (lambda p: NKGridExecutionSession.open(
                        CellExecutionSpec.from_payload(p['cell_spec']), repo_root=repo_root))
                    with factory(panel) as session:
                        spec = panel['cell_spec']; ns = spec['resolved_n_grid']; ks = spec['resolved_k_grid']
                        recipes = {}
                        for recipe in panel['pipelines']:
                            recipes.setdefault(recipe['oof_folds'], recipe)
                        for seed, draw in spec['resolved_repeat_plan']:
                            for folds, recipe in recipes.items():
                                def inputs(n, k):
                                    return session.prediction_cell_inputs(seed=seed, draw=draw, n_samples=n,
                                        k_features=k, model=recipe['model'], oof_folds=folds,
                                        pipeline_id=recipe['pipeline_id'])['sample_arrays']
                                for n in ns:
                                    arrays = inputs(n, ks[0])
                                    save(panel['panel_id'], seed, draw, n, ks[0], folds, 'training', arrays)
                                if folds == next(iter(recipes)):
                                    save(panel['panel_id'], seed, draw, ns[-1], ks[0], folds, 'evaluation', arrays)
                                    for k in ks:
                                        save(panel['panel_id'], seed, draw, ns[0], k, folds, 'features', inputs(ns[0], k))
                            db.commit()  # bounded transaction per repeat, never a worker write
                db.commit()
            count = db.execute('SELECT count(*) FROM maps').fetchone()[0]
            selectors = db.execute('SELECT count(*) FROM selectors').fetchone()[0]
        with temporary.open('r+b') as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, catalog); _sync_directory(root)
        receipt = {'format': 'plan-sample-maps-v1', 'workflow_sha256': digest(contract),
                   'integrity': 'record-sha256-v1', 'maps': count, 'selectors': selectors}
        _atomic_json(receipt_path, receipt)
        return receipt


class SharedMaps:
    def __init__(self, contract, *, max_bytes=32 * 1024**2):
        self.root = Path(contract['cache_root'])
        receipt = json.loads((self.root / 'maps-ready.json').read_bytes())
        if receipt['workflow_sha256'] != digest(contract):
            raise QueueError('Shared maps belong to another submission')
        # The catalog locates records; each used map is checked against the
        # actual sample arrays and its content identity below. No whole-DB hash.
        self.db = sqlite3.connect((self.root / 'maps.sqlite').resolve().as_uri() + '?mode=ro&immutable=1',
                                 uri=True, check_same_thread=False)
        self.entries = OrderedDict(); self.bytes = 0; self.max_bytes = max_bytes

    def get(self, task, folds, samples):
        refs, descriptors = {}, {}
        for kind, names in GROUPS.items():
            key = selector(task.panel_id, task.seed, task.draw, task.N, task.K, folds, kind)
            value = self.entries.pop(key, None)
            if value is None:
                row = self.db.execute('SELECT reference,descriptors FROM selectors JOIN maps USING(identity) WHERE selector=?', (key,)).fetchone()
                if row is None:
                    raise QueueError('Required exact shared sample map missing')
                ref, desc = map(json.loads, row)
                record = read_record(self.root, ref, require_sealed=True)
                size = sum(a.nbytes for a in record.arrays.values()) + len(row[0]) + len(row[1]) + 2048
                value = (ref, desc, record.arrays, size)
                while self.entries and self.bytes + size > self.max_bytes:
                    _, old = self.entries.popitem(last=False); self.bytes -= old[3]
                if size <= self.max_bytes:
                    self.bytes += size; self.entries[key] = value
            else:
                self.entries[key] = value
            ref, desc, arrays, _ = value
            if set(arrays) != {name for name in names if name in samples}:
                raise QueueError('Shared map fields differ from actual cell')
            for name, array in arrays.items():
                actual = np.asarray(samples[name])
                if array.dtype != actual.dtype or not np.array_equal(array, actual):
                    raise QueueError('Shared map order, labels, type or fold assignment differs: ' + name)
            refs[kind] = ref; descriptors.update(desc)
        return refs, descriptors

    def close(self):
        self.db.close(); self.entries.clear(); self.bytes = 0
