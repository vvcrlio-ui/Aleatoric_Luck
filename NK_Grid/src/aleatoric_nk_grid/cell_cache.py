"""Opt-in bounded input reuse around the unchanged numerical execution method.

No CV-fitted state is cached. Shared storage contains only the session's raw
cell and outer diagnostic transforms. It is trusted, private node-local scratch,
not a source for loading downloaded/untrusted joblib files.
"""
from __future__ import annotations
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from pathlib import Path
import sys
import time
import uuid
import json
import re

from .shared_queue import QueueError, atomic_json, digest, file_digest, file_lock, sync_directory


def session_namespace(session):
    existing = getattr(session, "_input_cache_namespace", None)
    if existing:
        return existing
    if session.spec is not None:
        value = digest(dict(session.spec.payload))
    else:
        import hashlib
        import pandas as pd
        frames = []
        for frame in (session.frame, session.external_frame):
            frames.append(None if frame is None else {
                "hash": hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest(),
                "columns": [str(c) for c in frame.columns], "dtypes": [str(t) for t in frame.dtypes]})
        value = digest({"semantic": session.semantic_contract, "frames": frames})
    session._input_cache_namespace = value
    return value


def resident_size(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if hasattr(value, "memory_usage"):
        usage = value.memory_usage(deep=True)
        return int(usage.sum() if hasattr(usage, "sum") else usage)
    if hasattr(value, "nbytes"):
        return int(value.nbytes)
    if isinstance(value, dict):
        return sys.getsizeof(value) + sum(resident_size(k, seen) + resident_size(v, seen) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return sys.getsizeof(value) + sum(resident_size(v, seen) for v in value)
    if is_dataclass(value):
        return sys.getsizeof(value) + sum(resident_size(getattr(value, f.name), seen) for f in fields(value))
    return sys.getsizeof(value)


@contextmanager
def wait_lock(path, timeout=180):
    end = time.monotonic() + timeout
    while True:
        manager = file_lock(path)
        try:
            manager.__enter__(); break
        except QueueError:
            if time.monotonic() >= end:
                raise
            time.sleep(.025)
    try:
        yield
    finally:
        manager.__exit__(None, None, None)


class NodeInputStore:
    """Bounded shared mmap store; existing entries never removed under readers.

    When full, admission stops and callers prepare privately. Reclamation is a
    quiescent operation after workers exit, avoiding unsafe mmap eviction races.
    Worker RAM has independent LRU eviction. This conservative first version is
    intended for bounded batches; production rotation needs node lifecycle wiring.
    """
    def __init__(self, root, *, namespace, max_bytes):
        if max_bytes < 0 or not namespace:
            raise ValueError("Invalid store budget/namespace")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.namespace = namespace; self.max_bytes = int(max_bytes)
        self.index_path = self.root / "index.json"
        self.verified = set()

    def _recover_unpublished(self, index):
        # Called with the exclusive build lock. An unpublished file cannot have
        # been returned to a reader; only those files can be reclaimed safely.
        if index["bytes"] != sum(e["bytes"] for e in index["entries"].values()):
            raise QueueError("Cache size accounting mismatch")
        for path in self.root.iterdir():
            blob = re.fullmatch(r"([0-9a-f]{64})\.joblib", path.name)
            temporary = re.fullmatch(r"(?:[0-9a-f]{64}|index\.json)\.[0-9a-f]{32}\.tmp", path.name)
            if not blob and not temporary:
                continue
            if path.is_symlink() or path.resolve().parent != self.root.resolve():
                raise QueueError("Unexpected cache link")
            if temporary or blob.group(1) not in index["entries"]:
                path.unlink()
        sync_directory(self.root)

    def load_or_build(self, key, build):
        import joblib
        full_key = digest([self.namespace, key])
        with wait_lock(self.root / "cache.lock"):
            if self.index_path.exists():
                index = json.loads(self.index_path.read_bytes())
                if index["namespace"] != self.namespace or index["max_bytes"] != self.max_bytes:
                    raise QueueError("Cache namespace or budget mismatch")
            else:
                index = {"namespace": self.namespace, "max_bytes": self.max_bytes, "bytes": 0, "entries": {}}
            entry = index["entries"].get(full_key)
            if entry:
                path = self.root / (full_key + ".joblib")
                if full_key not in self.verified:
                    if file_digest(path) != entry["sha256"]:
                        raise QueueError("Cache content checksum mismatch")
                    self.verified.add(full_key)
                return joblib.load(path, mmap_mode="r"), True
            self._recover_unpublished(index)
            value = build()
            estimated = resident_size(value)
            if index["bytes"] + estimated > self.max_bytes:
                return value, False
            target = self.root / (full_key + ".joblib")
            # Recovery under the same lock removes a dead builder's unpublished
            # blob/temp before admitting another cell.
            temp = self.root / (full_key + "." + uuid.uuid4().hex + ".tmp")
            try:
                joblib.dump(value, temp, compress=0)
                size = temp.stat().st_size
                if index["bytes"] + size > self.max_bytes:
                    return value, False
                import os
                with temp.open("r+b") as f:
                    os.fsync(f.fileno())
                temp.replace(target); sync_directory(self.root)
                index["entries"][full_key] = {"sha256": file_digest(target), "bytes": size}
                index["bytes"] += size
                atomic_json(self.index_path, index)
                self.verified.add(full_key)
                return joblib.load(target, mmap_mode="r"), False
            finally:
                temp.unlink(missing_ok=True)


class CachedSession:
    """A single numerical worker; reuse data without changing model task identity."""
    def __init__(self, session, *, max_bytes=256 * 1024**2, store=None):
        if max_bytes < 0:
            raise ValueError("Negative cache limit")
        self.session = session; self.max_bytes = max_bytes; self.store = store
        if store is not None and store.namespace != session_namespace(session):
            raise QueueError("Shared cache does not belong to this validated session")
        self.entries = OrderedDict(); self.bytes = 0
        self.stats = {"builds": 0, "memory_hits": 0, "shared_hits": 0, "evictions": 0, "prepare_seconds": 0.}

    @property
    def cached_cells(self):
        return list(self.entries)

    def _build(self, task):
        from .preprocessing import count_unobserved_sources, preprocess_cell
        session = self.session
        indexes = session.split_manager.for_seed(task.seed)
        orders = session._orders(task.seed, task.draw, indexes.train_index)
        selected_rows = orders.row_index[:task.N]
        units = [str(u) for u in orders.feature_names[:task.K]]
        columns = [f for u in units for f in session.feature_groups[u]]
        groups = [g for u in units for g in session.groups_by_unit[u]]
        test = session.external_frame if indexes.external_test else session.frame
        if len(selected_rows) != task.N or len(units) != task.K or test is None:
            raise ValueError("Invalid frozen cell")
        X = session.frame.loc[selected_rows, columns]
        Xt = test.loc[indexes.test_index, columns]
        unobserved = count_unobserved_sources(X, groups)
        prepared = {}
        if unobserved != task.K:
            for model in session.config.models:
                mode = "passthrough" if session.schema.imputation["model_overrides"].get(model) == "passthrough" else "imputed"
                if mode not in prepared:
                    prepared[mode] = preprocess_cell(X, Xt, groups, session.schema.imputation, model_name=model)
                    if prepared[mode].K_unobserved != unobserved:
                        raise ValueError("Prepared missingness changed")
        self.stats["builds"] += 1
        return dict(X_sub_raw=X, X_test_raw=Xt,
                    y_sub=session.frame.loc[selected_rows, session.config.outcome],
                    y_test=test.loc[indexes.test_index, session.config.outcome],
                    test_ids=indexes.test_ids, selected_groups=groups, unobserved=unobserved,
                    prepared=prepared, n_train_total=len(indexes.train_index), n_test_total=len(indexes.test_index))

    def run(self, task):
        session = self.session
        if session._closed or task.model not in session.config.models or (task.seed, task.draw) not in session.repeat_pairs or task.N not in session.n_grid or task.K not in session.k_grid:
            raise ValueError("Task outside the session contract")
        started = time.perf_counter()
        if task.cell in self.entries:
            value, size = self.entries.pop(task.cell); self.entries[task.cell] = (value, size)
            self.stats["memory_hits"] += 1
        else:
            try:
                if self.store:
                    value, shared = self.store.load_or_build(task.cell, lambda: self._build(task))
                    self.stats["shared_hits"] += int(shared)
                else:
                    value = self._build(task)
            except QueueError:
                raise
            except Exception:
                # Preserve original per-model failure/skip semantics if eager
                # preparation of another preprocessing mode is unavailable.
                return session.run_cell_group(seed=task.seed, draw=task.draw, n_samples=task.N, k_features=task.K, models=(task.model,))[0]
            size = resident_size(value)
            while self.entries and self.bytes + size > self.max_bytes:
                _, (_, removed) = self.entries.popitem(last=False)
                self.bytes -= removed; self.stats["evictions"] += 1
            if size <= self.max_bytes:
                self.entries[task.cell] = (value, size); self.bytes += size
        elapsed = time.perf_counter() - started
        self.stats["prepare_seconds"] += elapsed
        # _run_model deep-copies RAW inputs and constructs its original
        # FoldPreprocessor. Outer transformed inputs remain diagnostics only.
        return session._run_model(model_name=task.model, position=0, seed=task.seed,
                                  draw=task.draw, n_samples=task.N, k_features=task.K,
                                  slice_seconds=elapsed, preparation_errors={}, **value)

    def close(self):
        self.entries.clear(); self.bytes = 0
