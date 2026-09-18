"""Bounded stopped-writer packing with immutable address generations.

Journal, sample-map, base-index and recovery references are stable addresses.
One atomic generation pointer redirects those addresses to byte-identical frames.
Old readers keep their files until the controller proves all readers exited.
No accepted historical cache is migrated: this requires a shared-v1 manifest.
"""
from contextlib import closing
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import uuid

from .prediction_cache import (CacheBusyError, CacheIntegrityError, CacheStorageError,
    PredictionCacheWriter, safe_cache_path, _atomic_json, _sync_directory,
    _index_path, _load_index, _sealed_file_hash, _file_signature, read_record)
from .shared_queue import file_lock


@lru_cache(maxsize=32)
def _catalogs(root, signature):
    pointer = json.loads((Path(root) / 'locations.json').read_bytes())
    if pointer.get('format') != 'prediction-locations-v1':
        raise CacheIntegrityError('Unknown location generation')
    return tuple(pointer['generations'])


@lru_cache(maxsize=16384)
def _lookup(root, generation, path, offset, sha256):
    catalog = safe_cache_path(root, generation['path'])
    if _sealed_file_hash(catalog) != generation['sha256']:
        raise CacheIntegrityError('Location catalog changed')
    with closing(sqlite3.connect(catalog.as_uri() + '?mode=ro&immutable=1', uri=True)) as db:
        row = db.execute('SELECT reference FROM locations WHERE path=? AND offset=? AND sha256=?',
                         (path, offset, sha256)).fetchone()
    return json.loads(row[0]) if row else None


def resolve_reference(root, reference):
    root = Path(root).resolve(); pointer = root / 'locations.json'
    if not pointer.exists():
        return reference
    current = dict(reference)
    # Generations are chronological, so an address can move more than once.
    for item in _catalogs(str(root), _file_signature(pointer)):
        # Cache keys use an immutable serialization, not a mutable dictionary.
        replacement = _lookup_cached(str(root), json.dumps(item, sort_keys=True),
                                     current['path'], current['offset'], current['sha256'])
        if replacement is not None:
            if any(current.get(k) != replacement.get(k) for k in ('sha256', 'identity', 'length', 'status')):
                raise CacheIntegrityError('Location generation changed frame identity')
            current = replacement
    return current


@lru_cache(maxsize=32768)
def _lookup_cached(root, generation, path, offset, sha256):
    # Open only on cache miss; databases are read-only and never jointly written.
    return _lookup.__wrapped__(root, json.loads(generation), path, offset, sha256)


def compact_stopped(root, *, temporary_byte_limit, readers_drained=False,
                    target_bytes=128 * 1024**2, directories=('shards', 'meta-results')):
    root = Path(root).resolve()
    manifest = json.loads((root / 'manifest.json').read_bytes())
    if manifest.get('contract', {}).get('storage', {}).get('layout') != 'shared-v1':
        raise CacheIntegrityError('Automatic compaction is only for new shared-v1 outputs')
    if not readers_drained:
        raise CacheBusyError('Compaction publication/retirement needs stopped allocations and drained readers')
    if temporary_byte_limit < 1:
        raise CacheStorageError('Compaction requires an explicit temporary byte budget')
    reports = []
    with file_lock(root / '.layout.lock'):
        # Recover a crash after publication but before retirement. No new reader
        # may start until this controller operation returns.
        pointer_path = root / 'locations.json'
        pointer = json.loads(pointer_path.read_bytes()) if pointer_path.exists() else {
            'format': 'prediction-locations-v1', 'generations': []}
        for generation in pointer['generations']:
            _retire(root, generation)
        batch, size = [], 0
        def flush():
            nonlocal batch, size, pointer
            if len(batch) < 2:
                batch, size = [], 0
                return
            generation = uuid.uuid4().hex
            location_dir = root / 'locations'; location_dir.mkdir(exist_ok=True)
            catalog = location_dir / (generation + '.sqlite')
            with closing(sqlite3.connect(catalog)) as db:
                db.execute('PRAGMA synchronous=FULL')
                db.execute('CREATE TABLE locations(path TEXT, offset INTEGER, sha256 TEXT, reference TEXT NOT NULL, PRIMARY KEY(path,offset,sha256)) WITHOUT ROWID')
                with PredictionCacheWriter(root, writer_id='packed', incarnation=generation,
                        shard_target_bytes=target_bytes) as writer:
                    for index in batch:
                        source = safe_cache_path(root, index['path'])
                        if source.stat().st_size != index['bytes'] or _sealed_file_hash(source) != index['sha256']:
                            raise CacheIntegrityError('Compaction source changed after sealing')
                        with source.open('rb') as handle:
                            for old in index['records']:
                                handle.seek(old['offset']); frame = handle.read(old['length'])
                                new = writer.append_frame(frame, old, directory=index['path'].split('/')[0])
                                db.execute('INSERT INTO locations VALUES (?,?,?,?)',
                                    (old['path'], old['offset'], old['sha256'], json.dumps(new)))
                db.commit()
            with catalog.open('r+b') as handle:
                os.fsync(handle.fileno())
            _sync_directory(location_dir)
            item = {'path': catalog.relative_to(root).as_posix(), 'sha256': _sealed_file_hash(catalog),
                    'sources': [i['path'] for i in batch], 'source_bytes': size,
                    'new_shards': writer.sealed_shards, 'generation': generation}
            # New shards, indexes and catalog have all been fsynced. This is the
            # sole commit point. Stable old refs now resolve through this catalog.
            pointer = {**pointer, 'generations': [*pointer['generations'], item]}
            _atomic_json(pointer_path, pointer)
            _retire(root, item)
            reports.append(item); batch, size = [], 0
        # Snapshot names only. Do not feed newly packed shards back into this pass.
        paths = sorted(path for directory in directories for path in (root / 'indexes').glob(directory + '-*.pcshard.json'))
        for path in paths:
            if not path.exists():
                continue
            index, _ = _load_index(path)
            if not index.get('sealed') or index['bytes'] >= target_bytes * .8:
                continue
            # Budget includes the new data, SQLite/index space and one bounded frame.
            if index['bytes'] * 3 + 65536 > temporary_byte_limit:
                continue
            if (size + index['bytes']) * 3 + 65536 > temporary_byte_limit:
                flush()
            batch.append(index); size += index['bytes']
        flush()
    return reports


def _retire(root, generation):
    catalog = safe_cache_path(root, generation['path'])
    if _sealed_file_hash(catalog) != generation['sha256']:
        raise CacheIntegrityError('Cannot retire sources without the durable original location catalog')
    if not any(safe_cache_path(root, name).exists() for name in generation['sources']):
        return
    for relative in generation['new_shards']:
        path = safe_cache_path(root, relative)
        index, _ = _load_index(_index_path(root, relative))
        if path.stat().st_size != index['bytes'] or _sealed_file_hash(path) != index['sha256']:
            raise CacheIntegrityError('Packed shard is not durable and intact')
    for relative in generation['sources']:
        path = safe_cache_path(root, relative)
        if path.parent.name not in ('shards', 'meta-results') or path.suffix != '.pcshard':
            raise CacheIntegrityError('Retirement source outside permitted shard directories')
        path.unlink(missing_ok=True); _index_path(root, relative).unlink(missing_ok=True)
    _sync_directory(root / 'indexes')
    for directory in ('shards', 'meta-results'):
        _sync_directory(root / directory)
