"""Immutable task tables and restartable dynamic-work-queue workers.

The execution rows deliberately have no cost estimate and no precomputed
chunk.  A worker receives one modulo-stratified Parquet row group for one
round, materialises every completed cell group immediately, and a later round
derives its work solely from those durable artefacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
import os
import shutil
import sqlite3
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .experiment import manifest_path, utc_now, write_json_atomic
from .nk_grid import NKGridConfig, _process_peak_rss_bytes, resolve_repeat_pairs, run_nk_grid


TABLE_FORMAT_VERSION = 2
TABLE_COLUMNS = ("row_id", "seed", "draw", "N", "K", "group", "models")
VERIFY_INCOMPLETE_EXIT_CODE = 3
FINALIZATION_FORMAT_VERSION = 1
FINALIZATION_BATCH_ROWS = 4_096
FINALIZATION_MIN_TEMP_BYTES = 64 * 1024**2
QUEUE_INDEX_BATCH_ROWS = 4_096
QUEUE_INDEX_QUERY_ROWS = 512
TERMINAL_STATUSES = frozenset({"ok", "skipped"})
VALID_RESULT_STATUSES = frozenset({"ok", "skipped", "failed"})


class FinalizationError(ValueError):
    """A fail-closed dynamic-result validation or publication error."""


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


def pairs_for(
    n_samples: int, k_features: int, repeat_pairs: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    """Return repeat pairs for one grid point without changing their identity."""
    del n_samples, k_features
    return tuple((int(seed), int(draw)) for seed, draw in repeat_pairs)


def execution_groups(models: Sequence[str], *, k_features: int) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return indivisible preprocessing groups; Super Learner stays imputed."""
    del k_features
    selected = tuple(str(model) for model in models)
    if len(selected) != len(set(selected)):
        raise ValueError("models must not contain duplicates")
    passthrough = tuple(model for model in selected if model in {"lightgbm", "xgboost"})
    imputed = tuple(model for model in selected if model not in passthrough)
    groups: list[tuple[str, tuple[str, ...]]] = []
    if imputed:
        groups.append(("imputed_core", imputed))
    if passthrough:
        groups.append(("passthrough", passthrough))
    if not groups:
        raise ValueError("models must not be empty")
    return tuple(groups)


GROUP_PREPROCESS_MODES: Mapping[str, str] = {
    "imputed_core": "imputed", "passthrough": "passthrough",
}


def preprocess_mode_for_group(group: str) -> str:
    try:
        return GROUP_PREPROCESS_MODES[str(group)]
    except KeyError:
        raise ValueError(f"unknown execution group {group!r}") from None


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


def _task_rows(table: pa.Table) -> tuple[TaskRow, ...]:
    payload = table.to_pydict()
    required = set(TABLE_COLUMNS)
    actual = set(payload)
    if required != actual:
        if {"est_cost", "chunk_id"}.issubset(actual):
            raise ValueError("unsupported v1 flat-task table; rebuild a v2 dynamic-work-queue table")
        raise ValueError("unsupported flat-task table schema")
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


def _synthetic_arrow_table(count: int) -> pa.Table:
    zeros = np.zeros(count, dtype=np.int32)
    return pa.table({
        "row_id": pa.array(["synthetic"]).take(pa.array(zeros)), "seed": zeros, "draw": zeros,
        "N": np.full(count, 100, dtype=np.int32), "K": np.full(count, 10, dtype=np.int32),
        "group": pa.array(["imputed_core"]).take(pa.array(zeros)),
        "models": pa.array([["ols", "ridge"]]).take(pa.array(zeros)),
    })


def write_synthetic_task_table(path: Path, *, row_count: int, rows_per_group: int = 100_000) -> Path:
    if row_count < 1 or rows_per_group < 1:
        raise ValueError("row_count and rows_per_group must be positive")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    writer = pq.ParquetWriter(temporary, _synthetic_arrow_table(1).schema, compression="zstd")
    try:
        for start in range(0, row_count, rows_per_group):
            writer.write_table(_synthetic_arrow_table(min(rows_per_group, row_count - start)))
    finally:
        writer.close()
    os.replace(temporary, path)
    os.chmod(path, 0o444)
    return path


def _read_group_rss_worker(path: str, row_group: int, connection) -> None:
    try:
        before = _process_peak_rss_bytes()
        rows = read_row_group(Path(path), row_group)
        after = _process_peak_rss_bytes()
        connection.send({"rows": len(rows), "rss_delta_bytes": max(0, after - before)})
    finally:
        connection.close()


def measure_group_read_rss(path: Path, *, row_group: int = 0) -> dict[str, int]:
    parent, child = multiprocessing.get_context("spawn").Pipe(duplex=False)
    process = multiprocessing.get_context("spawn").Process(
        target=_read_group_rss_worker, args=(str(Path(path)), int(row_group), child),
    )
    process.start(); child.close()
    payload = parent.recv(); process.join(); parent.close()
    if process.exitcode != 0:
        raise RuntimeError(f"RSS worker failed with exit code {process.exitcode}")
    return {"rows": int(payload["rows"]), "rss_delta_bytes": int(payload["rss_delta_bytes"])}


def expected_model_keys(rows: Iterable[TaskRow]) -> set[tuple[str, int, int, int, int]]:
    return {(model, row.seed, row.draw, row.n_samples, row.k_features) for row in rows for model in row.models}


def pending_rows(rows: Iterable[TaskRow], completed: Iterable[tuple[str, int, int, int, int]]) -> tuple[TaskRow, ...]:
    done = set(completed)
    return tuple(row for row in rows if any((model, row.seed, row.draw, row.n_samples, row.k_features) not in done for model in row.models))


def _read_csv_keys(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or ()), list(reader)


def _csv_key(row: Mapping[str, str]) -> tuple[str, int, int, int, int]:
    return (str(row["model"]), int(row["seed"]), int(row["draw"]), int(row["N"]), int(row["K"]))


def _write_materialized_rows(
    output: Path, header: Sequence[str], by_key: Mapping[tuple[str, int, int, int, int], Mapping[str, str]],
    *, execution: Mapping[str, object], expected_rows: int,
) -> None:
    """Atomically publish a whole worker shard after every successful cell group."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header))
        writer.writeheader()
        writer.writerows(by_key[key] for key in sorted(by_key))
    os.replace(temporary, output)
    write_json_atomic(manifest_path(output), {
        "format_version": TABLE_FORMAT_VERSION, "execution": dict(execution),
        "completion": {"expected_rows": expected_rows, "materialized_rows": len(by_key)},
    })


def _append_attempt(path: Path, *, round_index: int, worker_index: int, sequence: int, row_id: str) -> None:
    """Append one bounded JSON record before starting a row.

    The record is below PIPE_BUF and therefore an O_APPEND write cannot be
    interleaved with another worker's record (workers have distinct files in
    normal operation).  A trailing malformed record after SIGKILL is ignored
    by the reader below.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"round": round_index, "worker_index": worker_index, "sequence": sequence, "row_id": row_id}, sort_keys=True) + "\n"
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, payload.encode("utf-8"))
    finally:
        os.close(descriptor)


def _sweep_slice_temporaries(output: Path, *, round_index: int, worker_index: int) -> None:
    prefix = f".{output.stem}.round-{round_index}.worker-{worker_index}."
    for temporary in output.parent.glob(prefix + "*.csv"):
        temporary.unlink(missing_ok=True)
        manifest_path(temporary).unlink(missing_ok=True)


def run_slice(snapshot_path: Path, *, round_index: int, worker_index: int) -> Path:
    """Run one worker's fixed modulo slice, publishing after every cell group."""
    payload = _load_snapshot(snapshot_path)
    workers = int(payload["workers"])
    if not 0 <= worker_index < workers:
        raise IndexError("worker_index is outside the frozen worker count")
    assignment = _require_ready_assignment(payload, round_index=round_index, workers=workers)
    rows = read_row_group(assignment, worker_index)
    output = _round_directory(payload, round_index) / f"worker-{worker_index}.csv"
    _sweep_slice_temporaries(output, round_index=round_index, worker_index=worker_index)
    header: list[str] | None = None
    materialized: list[dict[str, str]] = []
    if output.exists():
        header, materialized = _read_csv_keys(output)
    by_key = {_csv_key(row): row for row in materialized}
    if len(by_key) != len(materialized):
        raise ValueError("worker output contains duplicate keys")
    completed = {key for key, row in by_key.items() if row.get("status") in {"ok", "skipped"}}
    expected = expected_model_keys(rows)
    config = _config_from_json(payload["config"])
    for sequence, row in enumerate(pending_rows(rows, completed)):
        _append_attempt(_round_directory(payload, round_index) / "attempts" / f"worker-{worker_index}.jsonl", round_index=round_index, worker_index=worker_index, sequence=sequence, row_id=row.row_id)
        row_out = output.parent / f".{output.stem}.round-{round_index}.worker-{worker_index}.{row.row_id}.csv"
        # A worker is one CPU.  Keeping this at one also preserves deterministic
        # estimator behaviour; Super Learner remains in imputed_core and its
        # internal model_n_jobs is consequently one as well.
        row_config = replace(
            config, out=row_out, models=row.models, n_grid=(row.n_samples,), k_grid=(row.k_features,),
            n_seeds=1, n_draws=1, repeat_plan=((row.seed, row.draw),), n_jobs=1,
        )
        run_nk_grid(row_config, execution_pairs=((row.seed, row.draw),), exact_output_path=True, defer_failure_policy=True)
        current_header, current_rows = _read_csv_keys(row_out)
        if header is None:
            header = current_header
        elif current_header != header:
            raise ValueError("worker rows produced inconsistent CSV headers")
        for current in current_rows:
            by_key[_csv_key(current)] = current
        _write_materialized_rows(output, header, by_key, execution={"mode": "slice", "round": round_index, "worker_index": worker_index}, expected_rows=len(expected))
        row_out.unlink(missing_ok=True)
        manifest_path(row_out).unlink(missing_ok=True)
    if header is None and rows:
        raise RuntimeError("slice produced no rows")
    if header is None:
        # Empty assignments deliberately have no output file; their manifest is
        # still useful to distinguish an idle worker from an unstarted one.
        write_json_atomic(manifest_path(output), {"format_version": TABLE_FORMAT_VERSION, "execution": {"mode": "slice", "round": round_index, "worker_index": worker_index}, "completion": {"expected_rows": 0, "materialized_rows": 0}})
    return output


def _round_directory(snapshot: Mapping[str, object], round_index: int) -> Path:
    if round_index < 1:
        raise ValueError("round_index must be >= 1")
    return Path(str(snapshot["output_dir"])) / f"round-{round_index}"


def _load_snapshot(path: Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format_version") != TABLE_FORMAT_VERSION:
        if payload.get("format_version") == 1:
            raise ValueError("unsupported v1 flat-task snapshot; rebuild a v2 dynamic-work-queue plan")
        raise ValueError("unsupported flat-task snapshot")
    return payload


def _completed_keys(output_dir: Path) -> set[tuple[str, int, int, int, int]]:
    completed: set[tuple[str, int, int, int, int]] = set()
    for path in sorted(output_dir.glob("round-*/worker-*.csv")):
        _, rows = _read_csv_keys(path)
        completed.update(_csv_key(row) for row in rows if row.get("status") in {"ok", "skipped"})
    return completed


def _attempt_records(output_dir: Path) -> list[dict[str, int | str]]:
    records: list[dict[str, int | str]] = []
    for path in sorted(output_dir.glob("round-*/attempts/worker-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
                records.append({"round": int(item["round"]), "worker_index": int(item["worker_index"]), "sequence": int(item["sequence"]), "row_id": str(item["row_id"])})
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                # A Slurm SIGKILL can leave the diagnostic tail incomplete.
                continue
    return records


def classify_attempts(
    records: Iterable[Mapping[str, int | str]], *, completed_row_ids: Iterable[str] = (),
) -> tuple[set[str], set[str]]:
    """Return (crashed, too_long) using the two intentionally distinct rules."""
    attempts = sorted(
        (dict(record) for record in records),
        key=lambda record: (int(record["round"]), int(record["worker_index"]), int(record["sequence"])),
    )
    crashed: set[str] = set()
    finals: dict[tuple[int, int], tuple[int, str]] = {}
    for record in attempts:
        round_index = int(record["round"]); worker = int(record["worker_index"]); sequence = int(record["sequence"]); row_id = str(record["row_id"])
        prior = finals.get((round_index, worker))
        if prior is not None:
            crashed.add(prior[1])
        if prior is None or sequence > prior[0]:
            finals[(round_index, worker)] = (sequence, row_id)
    final_rounds: dict[str, set[int]] = {}
    for (round_index, _), (_, row_id) in finals.items():
        final_rounds.setdefault(row_id, set()).add(round_index)
    too_long = {
        row_id for row_id, rounds in final_rounds.items()
        if any({round_index, round_index + 1, round_index + 2}.issubset(rounds) for round_index in rounds)
    }
    completed = set(completed_row_ids)
    return crashed - completed, too_long - completed


def assign_rows_modulo(rows: Sequence[TaskRow], workers: int) -> tuple[tuple[TaskRow, ...], ...]:
    """Assign ordered todo rows by index modulo worker count, never by ranges."""
    if workers < 1:
        raise ValueError("workers must be positive")
    return tuple(tuple(rows[index] for index in range(worker, len(rows), workers)) for worker in range(workers))


_QUEUE_KEY_COLUMNS = ("model", "seed", "draw", "N", "K")
_QUEUE_KEY_SQL = ', '.join(f'"{column}"' for column in _QUEUE_KEY_COLUMNS)


def _queue_key_join(left: str, right: str) -> str:
    return " AND ".join(
        f'{left}."{column}"={right}."{column}"' for column in _QUEUE_KEY_COLUMNS
    )


def _result_shards(output_dir: Path) -> tuple[Path, ...]:
    return tuple(sorted(Path(output_dir).glob("round-*/worker-*.csv")))


def _attempt_logs(output_dir: Path) -> tuple[Path, ...]:
    return tuple(sorted(Path(output_dir).glob("round-*/attempts/worker-*.jsonl")))


def _resolve_queue_index_tmp_base(
    snapshot: Mapping[str, object], explicit: Path | None, *, phase: str,
) -> Path:
    configured: object | None = None
    phase_config = snapshot.get(phase)
    if isinstance(phase_config, Mapping):
        configured = phase_config.get("tmp_dir")
    candidate = (
        explicit
        if explicit is not None
        else configured
        or os.environ.get("NK_GRID_TMPDIR")
        or os.environ.get("TMPDIR")
        or tempfile.gettempdir()
    )
    base = Path(str(candidate)).expanduser().resolve()
    try:
        base.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"cannot create {phase} temporary directory {base}: {exc}") from exc
    if not base.is_dir():
        raise RuntimeError(f"{phase} temporary path is not a directory: {base}")
    return base


def _queue_index_temp_estimate(table_path: Path, inputs: Sequence[Path]) -> int:
    try:
        task_bytes = Path(table_path).stat().st_size
        input_bytes = sum(path.stat().st_size for path in inputs)
    except OSError as exc:
        raise RuntimeError(f"cannot size queue-index inputs: {exc}") from exc
    return max(
        FINALIZATION_MIN_TEMP_BYTES,
        FINALIZATION_MIN_TEMP_BYTES + 12 * task_bytes + 3 * input_bytes,
    )


def _preflight_queue_index_space(
    tmp_base: Path, table_path: Path, inputs: Sequence[Path], *, phase: str,
) -> tuple[int, int]:
    estimated = _queue_index_temp_estimate(table_path, inputs)
    try:
        available = int(shutil.disk_usage(tmp_base).free)
    except OSError as exc:
        raise RuntimeError(f"cannot measure {phase} temporary directory {tmp_base}: {exc}") from exc
    if available < estimated:
        raise RuntimeError(
            f"insufficient {phase} temporary space: temporary_directory={tmp_base} "
            f"available_bytes={available} estimated_required_bytes={estimated}"
        )
    return available, estimated


def _create_queue_index_tables(connection: sqlite3.Connection) -> None:
    connection.executescript(f"""
        CREATE TABLE expected (
            row_id TEXT NOT NULL,
            {_QUEUE_KEY_SQL},
            PRIMARY KEY ({_QUEUE_KEY_SQL})
        ) WITHOUT ROWID;
        CREATE INDEX expected_row_id ON expected (row_id);
        CREATE TABLE completed (
            {_QUEUE_KEY_SQL},
            PRIMARY KEY ({_QUEUE_KEY_SQL})
        ) WITHOUT ROWID;
        CREATE TABLE attempts (
            ingest_order INTEGER PRIMARY KEY,
            round_index INTEGER NOT NULL,
            worker_index INTEGER NOT NULL,
            sequence INTEGER NOT NULL,
            row_id TEXT NOT NULL
        );
        CREATE INDEX attempts_order ON attempts (round_index, worker_index, sequence, ingest_order);
        CREATE TABLE completed_rows (row_id TEXT PRIMARY KEY) WITHOUT ROWID;
        CREATE TABLE crashed_rows (row_id TEXT PRIMARY KEY) WITHOUT ROWID;
        CREATE TABLE too_long_rows (row_id TEXT PRIMARY KEY) WITHOUT ROWID;
    """)


@contextmanager
def _queue_index(
    snapshot: Mapping[str, object], *, explicit_tmp_dir: Path | None, phase: str,
):
    """Yield a one-run SQLite index under configured local scratch.

    The index owns every collection whose cardinality grows with the design or
    with accumulated worker output.  Its run directory is deliberately unique
    so a preempted invocation cannot contaminate a subsequent round.
    """

    output_dir = Path(str(snapshot["output_dir"]))
    table_path = Path(str(snapshot["task_table"]))
    inputs = (*_result_shards(output_dir), *_attempt_logs(output_dir))
    tmp_base = _resolve_queue_index_tmp_base(snapshot, explicit_tmp_dir, phase=phase)
    _preflight_queue_index_space(tmp_base, table_path, inputs, phase=phase)
    run_dir = Path(tempfile.mkdtemp(prefix=f"nk-grid-{phase}-", dir=tmp_base))
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(run_dir / "index.sqlite")
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA cache_size=-16384")
        _create_queue_index_tables(connection)
        yield connection
    except sqlite3.Error as exc:
        raise RuntimeError(f"SQLite {phase} index failed in {run_dir}: {exc}") from exc
    finally:
        if connection is not None:
            connection.close()
        shutil.rmtree(run_dir, ignore_errors=True)


def _index_queue_expected_design(connection: sqlite3.Connection, table_path: Path) -> int:
    statement = f"INSERT INTO expected (row_id, {_QUEUE_KEY_SQL}) VALUES (?, ?, ?, ?, ?, ?)"
    expected_count = 0
    try:
        for rows in _iter_task_row_batches(table_path, batch_rows=QUEUE_INDEX_BATCH_ROWS):
            values = tuple(
                (row.row_id, model, row.seed, row.draw, row.n_samples, row.k_features)
                for row in rows for model in row.models
            )
            connection.executemany(statement, values)
            expected_count += len(values)
            connection.commit()
    except sqlite3.IntegrityError as exc:
        raise ValueError("task table contains duplicate expected model keys") from exc
    except (OSError, ValueError, pa.ArrowInvalid) as exc:
        raise ValueError(f"cannot stream task table {table_path}: {exc}") from exc
    return expected_count


def _index_queue_completed_shards(connection: sqlite3.Connection, output_dir: Path) -> int:
    statement = f"INSERT OR IGNORE INTO completed ({_QUEUE_KEY_SQL}) VALUES (?, ?, ?, ?, ?)"
    for path in _result_shards(output_dir):
        try:
            with path.open(newline="", encoding="utf-8") as source:
                reader = csv.DictReader(source)
                for row in reader:
                    if row.get("status") not in TERMINAL_STATUSES:
                        continue
                    connection.execute(statement, _csv_key(row))
        except (OSError, UnicodeError, csv.Error, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"cannot read completed worker CSV {path}: {exc}") from exc
        connection.commit()
    return int(connection.execute("SELECT COUNT(*) FROM completed").fetchone()[0])


def _index_queue_attempts(connection: sqlite3.Connection, output_dir: Path) -> None:
    statement = (
        "INSERT INTO attempts (round_index, worker_index, sequence, row_id) "
        "VALUES (?, ?, ?, ?)"
    )
    for path in _attempt_logs(output_dir):
        try:
            with path.open(encoding="utf-8") as source:
                for line in source:
                    try:
                        item = json.loads(line)
                        connection.execute(statement, (
                            int(item["round"]), int(item["worker_index"]),
                            int(item["sequence"]), str(item["row_id"]),
                        ))
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                        # A Slurm SIGKILL can leave only the final JSON line malformed.
                        continue
        except (OSError, UnicodeError) as exc:
            raise ValueError(f"cannot read attempt log {path}: {exc}") from exc
        connection.commit()


def _index_completed_rows(connection: sqlite3.Connection) -> None:
    connection.execute(f"""
        INSERT INTO completed_rows (row_id)
        SELECT DISTINCT e.row_id FROM expected e
        WHERE NOT EXISTS (
            SELECT 1 FROM expected p
            WHERE p.row_id=e.row_id
              AND NOT EXISTS (
                SELECT 1 FROM completed c WHERE {_queue_key_join('p', 'c')}
              )
        )
    """)
    connection.commit()


def _classify_queue_attempts(connection: sqlite3.Connection) -> None:
    """Classify attempts using SQL order but the exact existing two-rule logic."""

    connection.execute("CREATE TABLE final_attempts (round_index INTEGER, worker_index INTEGER, row_id TEXT, PRIMARY KEY (round_index, worker_index)) WITHOUT ROWID")
    prior_group: tuple[int, int] | None = None
    final_sequence: int | None = None
    final_row_id: str | None = None
    cursor = connection.execute(
        "SELECT round_index, worker_index, sequence, row_id FROM attempts "
        "ORDER BY round_index, worker_index, sequence, ingest_order"
    )
    for round_index, worker_index, sequence, row_id in cursor:
        group = (int(round_index), int(worker_index))
        if group != prior_group:
            if prior_group is not None and final_row_id is not None:
                connection.execute(
                    "INSERT INTO final_attempts VALUES (?, ?, ?)",
                    (*prior_group, final_row_id),
                )
            prior_group = group
            final_sequence = None
            final_row_id = None
        if final_row_id is not None:
            connection.execute("INSERT OR IGNORE INTO crashed_rows VALUES (?)", (final_row_id,))
        if final_sequence is None or int(sequence) > final_sequence:
            final_sequence = int(sequence)
            final_row_id = str(row_id)
    if prior_group is not None and final_row_id is not None:
        connection.execute("INSERT INTO final_attempts VALUES (?, ?, ?)", (*prior_group, final_row_id))
    connection.execute("CREATE TABLE final_rounds (row_id TEXT, round_index INTEGER, PRIMARY KEY (row_id, round_index)) WITHOUT ROWID")
    connection.execute("INSERT INTO final_rounds SELECT DISTINCT row_id, round_index FROM final_attempts")
    connection.execute("""
        INSERT INTO too_long_rows (row_id)
        SELECT DISTINCT first.row_id FROM final_rounds first
        JOIN final_rounds second
          ON second.row_id=first.row_id AND second.round_index=first.round_index + 1
        JOIN final_rounds third
          ON third.row_id=first.row_id AND third.round_index=first.round_index + 2
    """)
    connection.execute("DELETE FROM crashed_rows WHERE row_id IN (SELECT row_id FROM completed_rows)")
    connection.execute("DELETE FROM too_long_rows WHERE row_id IN (SELECT row_id FROM completed_rows)")
    connection.commit()


def _queue_row_ids(connection: sqlite3.Connection, table: str) -> list[str]:
    return [str(row_id) for (row_id,) in connection.execute(f"SELECT row_id FROM {table} ORDER BY row_id")]


def _queue_missing_model_keys(connection: sqlite3.Connection) -> int:
    return int(connection.execute(f"""
        SELECT COUNT(*) FROM expected e
        WHERE NOT EXISTS (SELECT 1 FROM completed c WHERE {_queue_key_join('e', 'c')})
    """).fetchone()[0])


def _queue_incomplete_rows(connection: sqlite3.Connection) -> None:
    connection.execute("CREATE TABLE incomplete_rows (row_id TEXT PRIMARY KEY) WITHOUT ROWID")
    connection.execute(f"""
        INSERT INTO incomplete_rows (row_id)
        SELECT DISTINCT e.row_id FROM expected e
        WHERE NOT EXISTS (SELECT 1 FROM completed c WHERE {_queue_key_join('e', 'c')})
    """)
    connection.commit()


def _stage_queue_todo(
    connection: sqlite3.Connection, table_path: Path, *, workers: int,
) -> int:
    connection.executescript("""
        CREATE TABLE todo (
            todo_index INTEGER PRIMARY KEY,
            worker_index INTEGER NOT NULL,
            row_id TEXT NOT NULL,
            seed INTEGER NOT NULL,
            draw INTEGER NOT NULL,
            N INTEGER NOT NULL,
            K INTEGER NOT NULL,
            group_name TEXT NOT NULL,
            models_json TEXT NOT NULL
        );
        CREATE INDEX todo_worker_order ON todo (worker_index, todo_index);
    """)
    todo_index = 0
    eligible_statement = (
        "SELECT row_id FROM incomplete_rows WHERE row_id IN ({}) "
        "AND row_id NOT IN (SELECT row_id FROM too_long_rows)"
    )
    insert = (
        "INSERT INTO todo (todo_index, worker_index, row_id, seed, draw, N, K, group_name, models_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )
    try:
        for rows in _iter_task_row_batches(table_path, batch_rows=QUEUE_INDEX_BATCH_ROWS):
            for start in range(0, len(rows), QUEUE_INDEX_QUERY_ROWS):
                batch = rows[start:start + QUEUE_INDEX_QUERY_ROWS]
                placeholders = ", ".join("?" for _ in batch)
                eligible = {
                    str(row_id) for (row_id,) in connection.execute(
                        eligible_statement.format(placeholders),
                        tuple(row.row_id for row in batch),
                    )
                }
                staged: list[tuple[object, ...]] = []
                for row in batch:
                    if row.row_id not in eligible:
                        continue
                    staged.append((
                        todo_index, todo_index % workers, row.row_id, row.seed, row.draw,
                        row.n_samples, row.k_features, row.group,
                        json.dumps(list(row.models), separators=(",", ":")),
                    ))
                    todo_index += 1
                connection.executemany(insert, staged)
            connection.commit()
    except (OSError, ValueError, pa.ArrowInvalid) as exc:
        raise ValueError(f"cannot stream task table {table_path}: {exc}") from exc
    return todo_index


def _write_assignment_from_queue_index(
    path: Path, connection: sqlite3.Connection, *, workers: int,
) -> tuple[Path, list[int]]:
    """Publish exactly one modulo-ordered Parquet row group for every worker."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    writer = pq.ParquetWriter(temporary, _empty_arrow_table().schema, compression="zstd")
    assigned_rows: list[int] = []
    try:
        for worker_index in range(workers):
            records = connection.execute(
                "SELECT row_id, seed, draw, N, K, group_name, models_json FROM todo "
                "WHERE worker_index=? ORDER BY todo_index",
                (worker_index,),
            ).fetchall()
            rows = tuple(TaskRow(
                row_id=str(row_id), seed=int(seed), draw=int(draw),
                n_samples=int(n_samples), k_features=int(k_features), group=str(group),
                models=tuple(str(model) for model in json.loads(models_json)),
            ) for row_id, seed, draw, n_samples, k_features, group, models_json in records)
            assigned_rows.append(len(rows))
            writer.write_table(_arrow_table(rows) if rows else _empty_arrow_table())
    finally:
        writer.close()
    os.replace(temporary, path)
    os.chmod(path, 0o444)
    return path, assigned_rows


def _assignment_ready_path(round_dir: Path) -> Path:
    return Path(round_dir) / "assignment.ready.json"


def _require_ready_assignment(
    snapshot: Mapping[str, object], *, round_index: int, workers: int,
) -> Path:
    round_dir = _round_directory(snapshot, round_index)
    assignment = round_dir / "assignment.parquet"
    ready = _assignment_ready_path(round_dir)
    try:
        payload = json.loads(ready.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"assignment is not ready for round {round_index}; prep may have failed: {ready}"
        ) from exc
    if (
        payload.get("format_version") != TABLE_FORMAT_VERSION
        or int(payload.get("round", -1)) != round_index
        or int(payload.get("workers", -1)) != workers
        or payload.get("assignment") != str(assignment.resolve())
    ):
        raise RuntimeError(
            f"assignment readiness record is invalid for round {round_index}; prep may have failed: {ready}"
        )
    if not assignment.is_file():
        raise RuntimeError(
            f"assignment is not ready for round {round_index}; prep may have failed: {assignment}"
        )
    return assignment


def prepare_round(
    snapshot_path: Path, *, round_index: int, tmp_dir: Path | None = None,
) -> dict[str, object]:
    """Stream durable state into one modulo row group per worker.

    The task design and all prior durable artefacts are indexed on disk.  The
    only per-worker materialisation is the one row group that must be written
    as that worker's assignment, preserving the frozen W-row-group contract.
    """

    payload = _load_snapshot(snapshot_path)
    output_dir = Path(str(payload["output_dir"])); workers = int(payload["workers"])
    table_path = Path(str(payload["task_table"]))
    round_dir = _round_directory(payload, round_index); round_dir.mkdir(parents=True, exist_ok=True)
    ready = _assignment_ready_path(round_dir)
    ready.unlink(missing_ok=True)
    with _queue_index(payload, explicit_tmp_dir=tmp_dir, phase="preparation") as connection:
        _index_queue_expected_design(connection, table_path)
        completed_model_keys = _index_queue_completed_shards(connection, output_dir)
        _index_queue_attempts(connection, output_dir)
        _index_completed_rows(connection)
        _classify_queue_attempts(connection)
        _queue_incomplete_rows(connection)
        todo_rows = _stage_queue_todo(connection, table_path, workers=workers)
        assignment, assigned_rows = _write_assignment_from_queue_index(
            round_dir / "assignment.parquet", connection, workers=workers,
        )
        crashed = _queue_row_ids(connection, "crashed_rows")
        too_long = _queue_row_ids(connection, "too_long_rows")
    # These diagnostics and prep receipt are durable before the worker-facing
    # readiness marker.  A failed prep therefore cannot look like an idle run.
    write_json_atomic(round_dir / "crashed.json", {"round": round_index, "row_ids": crashed})
    write_json_atomic(round_dir / "too-long.json", {"round": round_index, "row_ids": too_long})
    stats: dict[str, object] = {
        "round": round_index, "workers": workers, "todo_rows": todo_rows,
        "assigned_rows": assigned_rows, "completed_model_keys": completed_model_keys,
        "crashed_row_ids": crashed, "too_long_row_ids": too_long,
        "assignment": str(assignment),
    }
    write_json_atomic(round_dir / "prep.json", stats)
    write_json_atomic(ready, {
        "format_version": TABLE_FORMAT_VERSION, "round": round_index,
        "workers": workers, "assignment": str(assignment.resolve()),
    })
    return stats


def verify_rounds(snapshot_path: Path, *, tmp_dir: Path | None = None) -> dict[str, object]:
    """Stream all durable state into a complete-design verification receipt."""

    payload = _load_snapshot(snapshot_path)
    output_dir = Path(str(payload["output_dir"]))
    table_path = Path(str(payload["task_table"]))
    with _queue_index(payload, explicit_tmp_dir=tmp_dir, phase="verification") as connection:
        expected_model_keys = _index_queue_expected_design(connection, table_path)
        completed_model_keys = _index_queue_completed_shards(connection, output_dir)
        _index_queue_attempts(connection, output_dir)
        _index_completed_rows(connection)
        _classify_queue_attempts(connection)
        result: dict[str, object] = {
            "expected_model_keys": expected_model_keys,
            "completed_model_keys": completed_model_keys,
            "missing_model_keys": _queue_missing_model_keys(connection),
            "crashed_row_ids": _queue_row_ids(connection, "crashed_rows"),
            "too_long_row_ids": _queue_row_ids(connection, "too_long_rows"),
        }
    # Completeness failures are data results, not index construction errors:
    # publish them atomically before the CLI converts them to exit code 3.
    write_json_atomic(output_dir / "verification.json", result)
    return result


def _verification_failure_counts(result: Mapping[str, object]) -> dict[str, int]:
    return {
        "missing_model_keys": int(result["missing_model_keys"]),
        "crashed_row_ids": len(result["crashed_row_ids"]),
        "too_long_row_ids": len(result["too_long_row_ids"]),
    }


def finalize_slice_shards(table_path: Path, worker_outputs: Iterable[Path], output: Path) -> Path:
    """Merge dynamic worker shards after exact complete-design validation."""
    expected = expected_model_keys(read_task_table(table_path))
    seen: dict[tuple[str, int, int, int, int], dict[str, str]] = {}
    header: list[str] | None = None
    for path in worker_outputs:
        if not Path(path).exists():
            continue
        current_header, rows = _read_csv_keys(Path(path))
        if header is None:
            header = current_header
        elif current_header != header:
            raise ValueError("worker CSV headers differ")
        for row in rows:
            key = _csv_key(row)
            if key not in expected:
                raise ValueError(f"merged output contains an out-of-design key: {key}")
            if key in seen:
                raise ValueError(f"merged output contains a duplicate key: {key}")
            seen[key] = row
    missing = expected - set(seen)
    if missing:
        raise ValueError(f"merged output is missing {len(missing)} expected keys")
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header or [])
        writer.writeheader(); writer.writerows(seen[key] for key in sorted(seen))
    return output


def finalization_manifest_path(output: Path) -> Path:
    """Return the receipt path for one atomically materialized dynamic result."""

    output = Path(output)
    return output.with_name(f"{output.name}.finalization.json")


def _iter_task_row_batches(
    table_path: Path, *, batch_rows: int = FINALIZATION_BATCH_ROWS,
) -> Iterable[tuple[TaskRow, ...]]:
    """Stream bounded task-table batches without materializing the design."""

    if batch_rows < 1:
        raise ValueError("batch_rows must be positive")
    source = pq.ParquetFile(Path(table_path), memory_map=True)
    for batch in source.iter_batches(batch_size=batch_rows):
        yield _task_rows(pa.Table.from_batches([batch]))


def _resolve_finalization_tmp_base(
    snapshot: Mapping[str, object], explicit: Path | None,
) -> Path:
    configured: object | None = None
    finalization = snapshot.get("finalization")
    if isinstance(finalization, Mapping):
        configured = finalization.get("tmp_dir")
    candidate = (
        explicit
        if explicit is not None
        else configured
        or os.environ.get("NK_GRID_TMPDIR")
        or os.environ.get("TMPDIR")
        or tempfile.gettempdir()
    )
    base = Path(str(candidate)).expanduser().resolve()
    try:
        base.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise FinalizationError(f"cannot create finalization temporary directory {base}: {exc}") from exc
    if not base.is_dir():
        raise FinalizationError(f"finalization temporary path is not a directory: {base}")
    return base


def _estimated_finalization_temp_bytes(table_path: Path, shards: Sequence[Path]) -> int:
    """Conservatively budget the expanded SQLite keys and result payloads."""

    try:
        task_bytes = Path(table_path).stat().st_size
        shard_bytes = sum(path.stat().st_size for path in shards)
    except OSError as exc:
        raise FinalizationError(f"cannot size finalization inputs: {exc}") from exc
    return max(
        FINALIZATION_MIN_TEMP_BYTES,
        # Parquet keys are compressed; 12x reserves their expanded B-tree.
        # Three shard copies cover payload/index pages plus rollback journal.
        FINALIZATION_MIN_TEMP_BYTES + 12 * task_bytes + 3 * shard_bytes,
    )


def _preflight_finalization_space(
    tmp_base: Path, table_path: Path, shards: Sequence[Path],
) -> tuple[int, int]:
    estimated = _estimated_finalization_temp_bytes(table_path, shards)
    try:
        available = int(shutil.disk_usage(tmp_base).free)
    except OSError as exc:
        raise FinalizationError(
            f"cannot measure finalization temporary directory {tmp_base}: {exc}"
        ) from exc
    if available < estimated:
        raise FinalizationError(
            "insufficient finalization temporary space: "
            f"temporary_directory={tmp_base} available_bytes={available} "
            f"estimated_required_bytes={estimated}"
        )
    return available, estimated


def _temporary_directory_bytes(directory: Path) -> int:
    try:
        return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
    except OSError as exc:
        raise FinalizationError(f"cannot measure finalization temporary usage in {directory}: {exc}") from exc


_SQL_KEY_COLUMNS = '"model", "seed", "draw", "N", "K"'
_SQL_KEY_JOIN = " AND ".join(
    f'e.{column}=r.{column}' for column in _SQL_KEY_COLUMNS.split(", ")
)


def _create_finalization_tables(connection: sqlite3.Connection) -> None:
    connection.executescript(f"""
        CREATE TABLE expected (
            {_SQL_KEY_COLUMNS}, PRIMARY KEY ({_SQL_KEY_COLUMNS})
        ) WITHOUT ROWID;
        CREATE TABLE observed (
            {_SQL_KEY_COLUMNS}, PRIMARY KEY ({_SQL_KEY_COLUMNS})
        ) WITHOUT ROWID;
        CREATE TABLE terminal (
            {_SQL_KEY_COLUMNS}, status TEXT NOT NULL, payload TEXT NOT NULL,
            PRIMARY KEY ({_SQL_KEY_COLUMNS})
        ) WITHOUT ROWID;
        CREATE TABLE terminal_duplicate_keys (
            {_SQL_KEY_COLUMNS}, PRIMARY KEY ({_SQL_KEY_COLUMNS})
        ) WITHOUT ROWID;
        CREATE TABLE failed_counts (
            {_SQL_KEY_COLUMNS}, row_count INTEGER NOT NULL,
            PRIMARY KEY ({_SQL_KEY_COLUMNS})
        ) WITHOUT ROWID;
    """)


def _insert_expected_design(
    connection: sqlite3.Connection, table_path: Path,
) -> int:
    expected_count = 0
    statement = f"INSERT INTO expected ({_SQL_KEY_COLUMNS}) VALUES (?, ?, ?, ?, ?)"
    try:
        for rows in _iter_task_row_batches(table_path):
            connection.executemany(
                statement,
                (
                    (model, row.seed, row.draw, row.n_samples, row.k_features)
                    for row in rows
                    for model in row.models
                ),
            )
            expected_count += sum(len(row.models) for row in rows)
            connection.commit()
    except sqlite3.IntegrityError as exc:
        raise FinalizationError("task table contains duplicate expected model keys") from exc
    except (OSError, ValueError, pa.ArrowInvalid) as exc:
        raise FinalizationError(f"cannot stream task table {table_path}: {exc}") from exc
    return expected_count


def _validated_result_header(path: Path, reader: csv.DictReader) -> list[str]:
    header = list(reader.fieldnames or ())
    required = {"model", "seed", "draw", "N", "K", "status"}
    if not header or len(set(header)) != len(header) or not required.issubset(header):
        raise FinalizationError(
            f"malformed worker CSV header in {path}: required={sorted(required)} actual={header}"
        )
    return header


def _ingest_result_shards(
    connection: sqlite3.Connection, shards: Sequence[Path],
) -> tuple[list[str], int, int]:
    header: list[str] | None = None
    rows_read = 0
    duplicate_terminal_rows = 0
    observed_insert = f"INSERT OR IGNORE INTO observed ({_SQL_KEY_COLUMNS}) VALUES (?, ?, ?, ?, ?)"
    terminal_insert = (
        f"INSERT OR IGNORE INTO terminal ({_SQL_KEY_COLUMNS}, status, payload) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)"
    )
    terminal_select = (
        f"SELECT payload FROM terminal WHERE "
        + " AND ".join(f'"{column}"=?' for column in ("model", "seed", "draw", "N", "K"))
    )
    duplicate_insert = (
        f"INSERT OR IGNORE INTO terminal_duplicate_keys ({_SQL_KEY_COLUMNS}) "
        "VALUES (?, ?, ?, ?, ?)"
    )
    failed_upsert = (
        f"INSERT INTO failed_counts ({_SQL_KEY_COLUMNS}, row_count) "
        "VALUES (?, ?, ?, ?, ?, 1) ON CONFLICT (model, seed, draw, N, K) "
        "DO UPDATE SET row_count=row_count+1"
    )
    for path in shards:
        try:
            with path.open(newline="", encoding="utf-8") as source:
                reader = csv.DictReader(source)
                current_header = _validated_result_header(path, reader)
                if header is None:
                    header = current_header
                elif current_header != header:
                    raise FinalizationError(f"worker CSV headers differ: {path}")
                for row in reader:
                    rows_read += 1
                    if None in row or any(row.get(column) is None for column in current_header):
                        raise FinalizationError(f"malformed worker CSV row in {path} at row {rows_read}")
                    try:
                        key = _csv_key(row)
                    except (KeyError, TypeError, ValueError) as exc:
                        raise FinalizationError(
                            f"invalid terminal key in {path} at row {rows_read}: {exc}"
                        ) from exc
                    status = str(row.get("status"))
                    if status not in VALID_RESULT_STATUSES:
                        raise FinalizationError(
                            f"invalid result status {status!r} in {path} for key {key}"
                        )
                    connection.execute(observed_insert, key)
                    if status in TERMINAL_STATUSES:
                        payload = json.dumps(
                            [row[column] for column in current_header],
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        inserted = connection.execute(
                            terminal_insert, (*key, status, payload),
                        ).rowcount
                        if not inserted:
                            previous = connection.execute(terminal_select, key).fetchone()
                            if previous is None or previous[0] != payload:
                                raise FinalizationError(
                                    f"conflicting terminal rows for key {key}"
                                )
                            connection.execute(duplicate_insert, key)
                            duplicate_terminal_rows += 1
                    else:
                        connection.execute(failed_upsert, key)
                    if rows_read % FINALIZATION_BATCH_ROWS == 0:
                        connection.commit()
        except FinalizationError:
            raise
        except (OSError, UnicodeError, csv.Error, sqlite3.Error) as exc:
            raise FinalizationError(f"cannot ingest worker CSV {path}: {exc}") from exc
    connection.commit()
    if header is None:
        raise FinalizationError("no worker CSV shards were found")
    return header, rows_read, duplicate_terminal_rows


def _validate_finalization_database(
    connection: sqlite3.Connection, expected_count: int,
) -> tuple[int, int, int]:
    extra = connection.execute(
        f"SELECT {_SQL_KEY_COLUMNS} FROM observed r WHERE NOT EXISTS "
        f"(SELECT 1 FROM expected e WHERE {_SQL_KEY_JOIN}) LIMIT 3"
    ).fetchall()
    if extra:
        raise FinalizationError(f"worker shards contain out-of-design keys: {extra}")
    missing_count = int(connection.execute(
        f"SELECT COUNT(*) FROM expected e WHERE NOT EXISTS "
        f"(SELECT 1 FROM terminal r WHERE {_SQL_KEY_JOIN})"
    ).fetchone()[0])
    if missing_count:
        missing = connection.execute(
            f"SELECT {_SQL_KEY_COLUMNS} FROM expected e WHERE NOT EXISTS "
            f"(SELECT 1 FROM terminal r WHERE {_SQL_KEY_JOIN}) LIMIT 3"
        ).fetchall()
        raise FinalizationError(
            f"finalization is missing {missing_count} expected terminal keys; examples={missing}"
        )
    final_rows = int(connection.execute("SELECT COUNT(*) FROM terminal").fetchone()[0])
    if final_rows != expected_count:
        raise FinalizationError(
            f"terminal row count differs from expected design: final={final_rows} expected={expected_count}"
        )
    failed_overridden = int(connection.execute(
        "SELECT COALESCE(SUM(f.row_count), 0) FROM failed_counts f "
        "WHERE EXISTS (SELECT 1 FROM terminal r WHERE "
        + " AND ".join(f'f."{column}"=r."{column}"' for column in ("model", "seed", "draw", "N", "K"))
        + ")"
    ).fetchone()[0])
    duplicate_keys = int(connection.execute(
        "SELECT COUNT(*) FROM terminal_duplicate_keys"
    ).fetchone()[0])
    return final_rows, failed_overridden, duplicate_keys


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_final_csv(temporary: Path, output: Path) -> None:
    os.replace(temporary, output)
    _fsync_directory(output.parent)


def _write_final_csv(
    connection: sqlite3.Connection, header: Sequence[str], output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as destination:
            writer = csv.writer(destination)
            writer.writerow(header)
            for (payload,) in connection.execute(
                f"SELECT payload FROM terminal ORDER BY {_SQL_KEY_COLUMNS}"
            ):
                writer.writerow(json.loads(payload))
            destination.flush()
            os.fsync(destination.fileno())
        _publish_final_csv(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def finalize_snapshot(
    snapshot_path: Path, *, tmp_dir: Path | None = None,
) -> dict[str, object]:
    """Stream, validate and atomically publish all dynamic worker shards.

    The immutable Parquet design is read by bounded Arrow batches.  Expected
    keys, terminal precedence, duplicate detection and final ordering live in
    a uniquely named SQLite database under local scratch, never in Python
    collections proportional to the experiment size.
    """

    started = time.perf_counter()
    snapshot = _load_snapshot(snapshot_path)
    table_path = Path(str(snapshot["task_table"]))
    output_dir = Path(str(snapshot["output_dir"]))
    config = snapshot.get("config")
    if not isinstance(config, Mapping) or not isinstance(config.get("out"), str):
        raise FinalizationError("snapshot config.out is required for finalization")
    output = Path(str(config["out"])).expanduser().resolve()
    shards = tuple(sorted(output_dir.glob("round-*/worker-*.csv")))
    tmp_base = _resolve_finalization_tmp_base(snapshot, tmp_dir)
    available_bytes, estimated_bytes = _preflight_finalization_space(
        tmp_base, table_path, shards,
    )
    run_dir = Path(tempfile.mkdtemp(prefix="nk-grid-finalize-", dir=tmp_base))
    database = run_dir / "index.sqlite"
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(database)
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA cache_size=-32768")
        _create_finalization_tables(connection)
        expected_count = _insert_expected_design(connection, table_path)
        header, rows_read, duplicate_terminal_rows = _ingest_result_shards(
            connection, shards,
        )
        final_rows, failed_overridden, duplicate_keys = _validate_finalization_database(
            connection, expected_count,
        )
        _write_final_csv(connection, header, output)
        temporary_bytes_used = _temporary_directory_bytes(run_dir)
        receipt: dict[str, object] = {
            "format_version": FINALIZATION_FORMAT_VERSION,
            "status": "complete",
            "created_at_utc": utc_now(),
            "backend": "sqlite_streaming",
            "input_shards": len(shards),
            "rows_read": rows_read,
            "final_rows": final_rows,
            "expected_model_keys": expected_count,
            "historical_failed_rows_overridden": failed_overridden,
            "duplicate_terminal_keys": duplicate_keys,
            "duplicate_terminal_rows": duplicate_terminal_rows,
            "temporary_directory": str(run_dir),
            "temporary_available_bytes": available_bytes,
            "estimated_temporary_bytes": estimated_bytes,
            "temporary_bytes_used": temporary_bytes_used,
            "wall_time_seconds": time.perf_counter() - started,
            "task_table": str(table_path.resolve()),
            "final_output": str(output),
        }
        write_json_atomic(finalization_manifest_path(output), receipt)
        return receipt
    except sqlite3.Error as exc:
        raise FinalizationError(f"SQLite finalization failed in {run_dir}: {exc}") from exc
    finally:
        if connection is not None:
            connection.close()
        shutil.rmtree(run_dir, ignore_errors=True)


@dataclass(frozen=True)
class ResourceRequest:
    cpus_per_task: int
    partition: str
    memory: str
    time_limit: str
    account: str
    constraint: str


def sbatch_resource_args(request: ResourceRequest) -> tuple[str, ...]:
    if request.cpus_per_task != 1:
        raise ValueError("dynamic workers must request exactly one CPU")
    if not all((request.partition, request.memory, request.time_limit, request.account)):
        raise ValueError("partition, memory, time_limit, and account are required")
    args = (f"--partition={request.partition}", "--cpus-per-task=1", f"--mem={request.memory}", f"--time={request.time_limit}", f"--account={request.account}")
    return args if request.constraint == "none" else (*args, f"--constraint={request.constraint}")


def _config_to_json(config: NKGridConfig) -> dict[str, object]:
    payload: dict[str, object] = {}
    for field in fields(config):
        value = getattr(config, field.name)
        if isinstance(value, Path):
            payload[field.name] = str(value)
        elif isinstance(value, tuple):
            payload[field.name] = [list(item) if isinstance(item, tuple) else item for item in value]
        else:
            payload[field.name] = value
    return payload


def _config_from_json(payload: Mapping[str, object]) -> NKGridConfig:
    values = dict(payload)
    for field in ("schema", "out", "model_params"):
        values[field] = Path(str(values[field]))
    values["models"] = tuple(str(value) for value in values["models"])
    for field in ("n_grid", "k_grid"):
        if values.get(field) is not None:
            values[field] = tuple(int(value) for value in values[field])
    if values.get("repeat_plan") is not None:
        values["repeat_plan"] = tuple((int(pair[0]), int(pair[1])) for pair in values["repeat_plan"])
    return NKGridConfig(**values)


def write_work_snapshot(
    path: Path,
    *,
    table_path: Path,
    panel: str,
    config: NKGridConfig,
    output_dir: Path,
    workers: int,
    preparation_tmp_dir: Path | str | None = None,
    verification_tmp_dir: Path | str | None = None,
    finalization_tmp_dir: Path | str | None = None,
) -> Path:
    if workers < 1:
        raise ValueError("workers must be positive")
    table = Path(table_path).resolve()
    try:
        if pq.ParquetFile(table, memory_map=True).metadata.num_rows < 1:
            raise ValueError("task table must contain at least one row")
    except ValueError:
        raise
    except (OSError, pa.ArrowInvalid) as exc:
        raise ValueError(f"cannot read task table metadata {table}: {exc}") from exc
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "format_version": TABLE_FORMAT_VERSION,
        "panel": str(panel),
        "task_table": str(table),
        "config": _config_to_json(config),
        "output_dir": str(Path(output_dir).resolve()),
        "workers": int(workers),
    }
    for phase, temporary_directory in (
        ("preparation", preparation_tmp_dir),
        ("verification", verification_tmp_dir),
        ("finalization", finalization_tmp_dir),
    ):
        if temporary_directory is not None:
            payload[phase] = {
                "tmp_dir": str(Path(temporary_directory).expanduser().resolve()),
            }
    write_json_atomic(path, payload)
    os.chmod(path, 0o444)
    return path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run dynamic NK-grid work slices")
    parser.add_argument("command", choices=("prep", "run", "verify", "finalize"))
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--round", type=int)
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--tmp-dir", type=Path)
    args = parser.parse_args(argv)
    if args.command == "prep":
        if args.round is None: parser.error("prep requires --round")
        print(json.dumps(prepare_round(args.snapshot, round_index=args.round, tmp_dir=args.tmp_dir), sort_keys=True))
    elif args.command == "run":
        if args.round is None or args.worker_index is None: parser.error("run requires --round and --worker-index")
        print(run_slice(args.snapshot, round_index=args.round, worker_index=args.worker_index))
    elif args.command == "verify":
        result = verify_rounds(args.snapshot, tmp_dir=args.tmp_dir)
        print(json.dumps(result, sort_keys=True), flush=True)
        failures = _verification_failure_counts(result)
        if any(failures.values()):
            print(
                "verification incomplete: "
                + "; ".join(f"{category}={count}" for category, count in failures.items()),
                file=sys.stderr,
                flush=True,
            )
            raise SystemExit(VERIFY_INCOMPLETE_EXIT_CODE)
    else:
        print(json.dumps(finalize_snapshot(args.snapshot, tmp_dir=args.tmp_dir), sort_keys=True))


if __name__ == "__main__":
    main()
