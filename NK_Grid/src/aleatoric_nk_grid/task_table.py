"""Immutable task-row design and Parquet storage, independent of queue execution.

The v2 ordering, row identifiers, logical digests, and row-group boundaries are
shared by the planner and worker. Durable publication retains the existing file
and directory fsync sequence; this module imports no process or lease backend.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from .config import NKGridConfig, execution_groups_for_models, resolve_repeat_pairs
from .execution_contract import (
    canonical_json_bytes, canonical_task_row_payload, sha256_file, task_row_digest,
)
from .phase_timing import timed_phase


TABLE_FORMAT_VERSION = 2
TABLE_COLUMNS = ("row_id", "seed", "draw", "N", "K", "group", "models")
TASK_TABLE_ROWS_PER_GROUP_MAX = 100_000


@dataclass(frozen=True)
class TaskRow:
    row_id: str
    seed: int
    draw: int
    n_samples: int
    k_features: int
    group: str
    models: tuple[str, ...]

    @property
    def key(self) -> tuple[int, int, int, int, str]:
        return (self.seed, self.draw, self.n_samples, self.k_features, self.group)


@dataclass(frozen=True)
class TaskTableSummary:
    """One-pass logical and physical task-table facts for immutable contracts."""

    path: Path
    task_design_digest: str
    task_table_file_sha256: str
    expected_task_rows: int
    expected_model_rows: int
    max_n: int
    max_k: int
    row_group_digests: tuple[dict[str, object], ...]
    logical_schema_fingerprint: str


def pairs_for(
    n_samples: int, k_features: int, repeat_pairs: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    """Return repeat pairs for one grid point without changing their identity."""
    del n_samples, k_features
    return tuple((int(seed), int(draw)) for seed, draw in repeat_pairs)


def execution_groups(models: Sequence[str], *, k_features: int) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return indivisible preprocessing groups; Super Learner stays imputed."""
    del k_features
    return execution_groups_for_models(models)


GROUP_PREPROCESS_MODES: Mapping[str, str] = {
    "imputed_core": "imputed", "passthrough": "passthrough",
}


def _row_id(seed: int, draw: int, n_samples: int, k_features: int, group: str, models: tuple[str, ...]) -> str:
    raw = json.dumps([seed, draw, n_samples, k_features, group, list(models)], separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def build_rows(
    config: NKGridConfig,
    *,
    n_grid: Sequence[int],
    k_grid: Sequence[int],
    pairs_provider: Callable[[int, int], Sequence[tuple[int, int]]] | None = None,
) -> tuple[TaskRow, ...]:
    """Build the immutable deterministic cell-group design."""
    repeat_pairs = resolve_repeat_pairs(config)
    provider = pairs_provider or (lambda n, k: pairs_for(n, k, repeat_pairs))
    rows: list[TaskRow] = []
    for k_features in sorted({int(value) for value in k_grid}):
        for n_samples in sorted({int(value) for value in n_grid}):
            point_pairs = tuple((int(seed), int(draw)) for seed, draw in provider(n_samples, k_features))
            if not point_pairs:
                raise ValueError("pairs_for must return at least one pair")
            for seed, draw in point_pairs:
                for group, group_models in execution_groups(config.models, k_features=k_features):
                    rows.append(TaskRow(
                        row_id=_row_id(seed, draw, n_samples, k_features, group, group_models),
                        seed=seed, draw=draw, n_samples=n_samples, k_features=k_features,
                        group=group, models=group_models,
                    ))
    if len({row.row_id for row in rows}) != len(rows):
        raise ValueError("task table would contain duplicate row IDs")
    return tuple(rows)


def iter_task_rows_canonical(
    config: NKGridConfig,
    *,
    n_grid: Sequence[int],
    k_grid: Sequence[int],
    repeat_pairs: Sequence[tuple[int, int]] | None = None,
) -> Iterable[TaskRow]:
    """Yield the canonical table order without materialising the full design."""

    pairs = tuple(sorted((int(seed), int(draw)) for seed, draw in (repeat_pairs or resolve_repeat_pairs(config))))
    if not pairs:
        raise ValueError("repeat plan must not be empty")
    for k_features in sorted({int(value) for value in k_grid}):
        for n_samples in sorted({int(value) for value in n_grid}):
            for seed, draw in pairs:
                for group, group_models in execution_groups(config.models, k_features=k_features):
                    yield TaskRow(
                        row_id=_row_id(seed, draw, n_samples, k_features, group, group_models),
                        seed=seed,
                        draw=draw,
                        n_samples=n_samples,
                        k_features=k_features,
                        group=group,
                        models=group_models,
                    )


def _task_table_schema_fingerprint() -> str:
    return hashlib.sha256(canonical_json_bytes({
        "columns": list(TABLE_COLUMNS),
        "format": "task-table-v2",
        "models_encoding": "ordered-string-list",
    })).hexdigest()


def _durable_replace(source: Path, target: Path) -> None:
    descriptor = os.open(source, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(source, target)
    directory = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


@timed_phase("plan.enumerate_uniqueness_parquet")
def write_task_table_streaming(
    rows: Iterable[TaskRow],
    path: Path,
    *,
    rows_per_group: int = TASK_TABLE_ROWS_PER_GROUP_MAX,
    tmp_dir: Path | None = None,
) -> TaskTableSummary:
    """Write Task Table v2 in bounded Arrow/SQLite buffers.

    Both uniqueness constraints are owned by local scratch SQLite tables.  In
    particular, this does not create a Python ``set`` proportional to the
    design and does not reread Parquet as TaskRows to compute its file hash.
    """

    if not 1 <= rows_per_group <= TASK_TABLE_ROWS_PER_GROUP_MAX:
        raise ValueError(f"rows_per_group must be between 1 and {TASK_TABLE_ROWS_PER_GROUP_MAX}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    local_base = Path(tmp_dir) if tmp_dir is not None else Path(tempfile.gettempdir())
    local_base.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="nk-grid-plan-", dir=local_base))
    temporary = target.with_suffix(target.suffix + f".tmp.{uuid.uuid4().hex}")
    writer: pq.ParquetWriter | None = None
    connection: sqlite3.Connection | None = None
    digest = hashlib.sha256()
    row_group_digests: list[dict[str, object]] = []
    task_rows = 0; model_rows = 0; max_n = 0; max_k = 0
    buffer: list[TaskRow] = []
    sqlite_seconds = 0.0
    parquet_seconds = 0.0
    table_started = time.perf_counter()

    def flush() -> None:
        nonlocal buffer, parquet_seconds
        if not buffer:
            return
        assert writer is not None
        write_started = time.perf_counter()
        writer.write_table(_arrow_table(buffer))
        parquet_seconds += time.perf_counter() - write_started
        row_group_digests.append({
            "row_group": len(row_group_digests),
            "row_count": len(buffer),
            "canonical_task_rows_sha256": task_row_digest(buffer),
        })
        buffer = []

    try:
        connection = sqlite3.connect(scratch / "planning-uniqueness.sqlite")
        connection.execute("CREATE TABLE rows (row_id TEXT PRIMARY KEY) WITHOUT ROWID")
        connection.execute("CREATE TABLE model_keys (model TEXT, seed INTEGER, draw INTEGER, N INTEGER, K INTEGER, PRIMARY KEY (model, seed, draw, N, K)) WITHOUT ROWID")
        writer = pq.ParquetWriter(temporary, _empty_arrow_table().schema, compression="zstd")
        for row in rows:
            unique_started = time.perf_counter()
            try:
                connection.execute("INSERT INTO rows VALUES (?)", (row.row_id,))
                connection.executemany(
                    "INSERT INTO model_keys VALUES (?, ?, ?, ?, ?)",
                    [(model, row.seed, row.draw, row.n_samples, row.k_features) for model in row.models],
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("streaming task planning found duplicate row ID or public model key") from exc
            finally:
                sqlite_seconds += time.perf_counter() - unique_started
            digest.update(canonical_json_bytes(canonical_task_row_payload(row))); digest.update(b"\n")
            task_rows += 1; model_rows += len(row.models)
            max_n = max(max_n, row.n_samples); max_k = max(max_k, row.k_features)
            buffer.append(row)
            if len(buffer) == rows_per_group:
                flush(); connection.commit()
        flush(); connection.commit()
        if task_rows == 0:
            raise ValueError("task table must not be empty")
        writer.close(); writer = None
        _durable_replace(temporary, target)
        os.chmod(target, 0o444)
        return TaskTableSummary(
            path=target,
            task_design_digest=digest.hexdigest(),
            task_table_file_sha256=sha256_file(target),
            expected_task_rows=task_rows,
            expected_model_rows=model_rows,
            max_n=max_n,
            max_k=max_k,
            row_group_digests=tuple(row_group_digests),
            logical_schema_fingerprint=_task_table_schema_fingerprint(),
        )
    finally:
        print(json.dumps({"nkgrid_phase": "plan.detail", "task_rows": task_rows,
                          "model_rows": model_rows, "sqlite_seconds": sqlite_seconds,
                          "arrow_parquet_seconds": parquet_seconds,
                          "total_seconds": time.perf_counter() - table_started}), file=sys.stderr, flush=True)
        if writer is not None:
            writer.close()
        if connection is not None:
            connection.close()
        temporary.unlink(missing_ok=True)
        shutil.rmtree(scratch, ignore_errors=True)


def _arrow_table(rows: Sequence[TaskRow]) -> pa.Table:
    return pa.table({
        "row_id": [row.row_id for row in rows],
        "seed": [row.seed for row in rows], "draw": [row.draw for row in rows],
        "N": [row.n_samples for row in rows], "K": [row.k_features for row in rows],
        "group": [row.group for row in rows], "models": [list(row.models) for row in rows],
    })


def _empty_arrow_table() -> pa.Table:
    return pa.table({
        "row_id": pa.array([], type=pa.string()), "seed": pa.array([], type=pa.int64()),
        "draw": pa.array([], type=pa.int64()), "N": pa.array([], type=pa.int64()),
        "K": pa.array([], type=pa.int64()), "group": pa.array([], type=pa.string()),
        "models": pa.array([], type=pa.list_(pa.string())),
    })


def _write_table_groups(path: Path, groups: Sequence[Sequence[TaskRow]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    schema = _arrow_table(groups[0]).schema if groups and groups[0] else _empty_arrow_table().schema
    writer = pq.ParquetWriter(temporary, schema, compression="zstd")
    try:
        for group in groups:
            writer.write_table(_arrow_table(group) if group else _empty_arrow_table())
    finally:
        writer.close()
    os.replace(temporary, path)
    os.chmod(path, 0o444)
    return path


def write_task_table(path: Path, rows: Sequence[TaskRow], *, rows_per_group: int = 100_000) -> Path:
    """Write an immutable v2 main table in fixed-size streaming row groups."""
    if not rows:
        raise ValueError("task table must not be empty")
    if rows_per_group < 1:
        raise ValueError("rows_per_group must be positive")
    ordered = tuple(sorted(rows, key=lambda row: (row.k_features, row.n_samples, row.seed, row.draw, row.group, row.row_id)))
    return _write_table_groups(path, [ordered[start:start + rows_per_group] for start in range(0, len(ordered), rows_per_group)])


def _validate_task_table_columns(columns: Iterable[str]) -> None:
    required = set(TABLE_COLUMNS)
    actual = {str(column) for column in columns}
    if required != actual:
        if {"est_cost", "chunk_id"}.issubset(actual):
            raise ValueError("unsupported v1 flat-task table; rebuild a v2 dynamic-work-queue table")
        raise ValueError("unsupported flat-task table schema")


def _task_rows(table: pa.Table) -> tuple[TaskRow, ...]:
    payload = table.to_pydict()
    _validate_task_table_columns(payload)
    return tuple(TaskRow(
        row_id=str(payload["row_id"][index]), seed=int(payload["seed"][index]),
        draw=int(payload["draw"][index]), n_samples=int(payload["N"][index]),
        k_features=int(payload["K"][index]), group=str(payload["group"][index]),
        models=tuple(str(value) for value in payload["models"][index]),
    ) for index in range(table.num_rows))


def read_row_group(path: Path, row_group: int) -> tuple[TaskRow, ...]:
    source = pq.ParquetFile(Path(path), memory_map=True)
    if not 0 <= row_group < source.num_row_groups:
        raise IndexError(f"row_group must be between 0 and {source.num_row_groups - 1}")
    return _task_rows(source.read_row_group(row_group))


def read_task_table(path: Path) -> tuple[TaskRow, ...]:
    source = pq.ParquetFile(Path(path), memory_map=True)
    return tuple(row for group in range(source.num_row_groups) for row in read_row_group(path, group))


def expected_model_keys(rows: Iterable[TaskRow]) -> set[tuple[str, int, int, int, int]]:
    return {(model, row.seed, row.draw, row.n_samples, row.k_features) for row in rows for model in row.models}


def pending_rows(rows: Iterable[TaskRow], completed: Iterable[tuple[str, int, int, int, int]]) -> tuple[TaskRow, ...]:
    done = set(completed)
    return tuple(row for row in rows if any((model, row.seed, row.draw, row.n_samples, row.k_features) not in done for model in row.models))


def assign_rows_modulo(rows: Sequence[TaskRow], workers: int) -> tuple[tuple[TaskRow, ...], ...]:
    """Assign ordered todo rows by index modulo worker count, never by ranges."""
    if workers < 1:
        raise ValueError("workers must be positive")
    return tuple(tuple(rows[index] for index in range(worker, len(rows), workers)) for worker in range(workers))
