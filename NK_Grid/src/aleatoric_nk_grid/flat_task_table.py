"""Immutable task tables and restartable dynamic-work-queue workers.

The execution rows deliberately have no cost estimate and no precomputed
chunk.  A worker receives one modulo-stratified Parquet row group for one
round, materialises every completed cell group immediately, and a later round
derives its work solely from those durable artefacts.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
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
from .execution_contract import (
    AnalysisContract,
    CellExecutionSpec,
    ContractError,
    DynamicExecutionContract,
    canonical_json_bytes,
    canonical_task_row_payload,
    sha256_file,
    task_row_digest,
    immutable_json_bytes,
)
from .generation_control import (
    ActivationTarget,
    ControlBusyError,
    ControlProtocolError,
    ControlSupersededError,
    PROTOCOL_EXIT_CODE,
    RETRYABLE_EXIT_CODE,
    SUCCESS_EXIT_CODE,
    SUPERSEDED_EXIT_CODE,
    activate_generation,
    classify_exact_afterany_target_read_only,
    closed_path,
    frozen_sealed_history,
    frozen_history_from_closed,
    generation_dir,
    outcome_path,
    predecessor_gate,
    publish_activation_intent,
    publish_no_generation_outcome,
    publish_verification_receipt,
    seal_generation,
    schedule_transaction,
    validate_exact_verification_receipt,
    verification_path,
)
from .nk_grid import (
    NKGridConfig,
    NKGridExecutionSession,
    _process_peak_rss_bytes,
    execution_groups_for_models,
    project_public_result,
    resolve_repeat_pairs,
    run_nk_grid,
)
from .worker_event_wal import (
    TASK_ABORTED,
    TASK_RESULT,
    TASK_STARTED,
    WAL_FORMAT,
    WALFrameTooLarge,
    WALBusyError,
    WALProtocolError,
    WorkerEventLog,
    bounded_abort_payload,
    decode_public_rows,
    scan_wal,
)


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
TASK_TABLE_ROWS_PER_GROUP_MAX = 100_000


class FinalizationError(ValueError):
    """A fail-closed dynamic-result validation or publication error."""


@contextmanager
def _generation_shared_lease(path: Path):
    """Hold a generation shared lease for the complete worker invocation."""

    descriptor = os.open(Path(path), os.O_RDWR)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ControlBusyError(f"generation lease is busy: {path}") from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


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

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        assert writer is not None
        writer.write_table(_arrow_table(buffer))
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
            try:
                connection.execute("INSERT INTO rows VALUES (?)", (row.row_id,))
                connection.executemany(
                    "INSERT INTO model_keys VALUES (?, ?, ?, ?, ?)",
                    [(model, row.seed, row.draw, row.n_samples, row.k_features) for model in row.models],
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("streaming task planning found duplicate row ID or public model key") from exc
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


def run_slice(
    snapshot_path: Path, *, round_index: int, worker_index: int,
    expected_prep_token: str,
    submission_generation: str | None = None,
    expected_previous_generation: str | None = None,
    expected_pointer_version: int | None = None,
    prep_job_id: str | None = None,
    expected_previous_execution_plan_id: str | None = None,
    expected_previous_round_index: int | None = None,
) -> Path:
    """Run one WAL-owned worker slice; no output CSV/checkpoint path is opened."""
    payload = _load_snapshot(snapshot_path)
    if "analysis_contract" not in payload:
        return _legacy_run_slice(snapshot_path, round_index=round_index, worker_index=worker_index, expected_prep_token=expected_prep_token)
    analysis, execution = _load_contract_chain(payload, validate_task_table=False)
    workers = int(payload["workers"])
    if not 0 <= worker_index < workers:
        raise IndexError("worker_index is outside the frozen worker count")
    if submission_generation is None:
        raise ValueError("worker requires an exact submission generation")
    target = ActivationTarget(
        analysis_id=analysis.analysis_id,
        execution_plan_id=execution.execution_plan_id,
        execution_contract_sha256=execution.sha256,
        round_index=int(round_index),
        submission_generation=str(submission_generation),
        expected_previous_generation=expected_previous_generation,
        expected_pointer_version=(int(round_index) - 1 if expected_pointer_version is None else int(expected_pointer_version)),
        prep_job_id=str(prep_job_id or expected_prep_token),
        prep_token=_validated_prep_token(expected_prep_token),
        expected_previous_execution_plan_id=expected_previous_execution_plan_id,
        expected_previous_round_index=expected_previous_round_index,
    )
    dispatch = classify_exact_afterany_target_read_only(Path(str(payload["output_dir"])), target)
    if dispatch.kind in {"no-generation", "sealed-generation"}:
        return Path(dispatch.outcome_path or dispatch.closed_path)
    if dispatch.kind != "active-generation":
        if dispatch.exit_code == SUPERSEDED_EXIT_CODE:
            raise ControlSupersededError("worker target was superseded")
        if dispatch.exit_code == PROTOCOL_EXIT_CODE:
            raise ControlProtocolError("worker target has a protocol conflict")
        raise ControlBusyError("worker target needs activation recovery")
    activation = json.loads(Path(dispatch.activation_path).read_text(encoding="utf-8"))
    assignment = Path(str(activation["assignment_path"])); index_path = Path(str(activation["assignment_index_path"]))
    if sha256_file(index_path) != activation["assignment_index_sha256"]:
        raise ControlProtocolError("assignment index checksum mismatch")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    own = next((item for item in index.get("row_groups", []) if int(item.get("worker", -1)) == worker_index), None)
    if not isinstance(own, Mapping):
        raise ControlProtocolError("assignment index lacks worker row group")
    rows = read_row_group(assignment, worker_index)
    if len(rows) != int(own.get("row_count", -1)) or task_row_digest(rows) != own.get("canonical_task_rows_sha256"):
        raise ControlProtocolError("worker assignment row group digest mismatch")
    identity = {
        "wal_format": WAL_FORMAT,
        "analysis_id": analysis.analysis_id,
        "execution_plan_id": execution.execution_plan_id,
        "execution_contract_sha256": execution.sha256,
        "round": round_index,
        "submission_generation": target.submission_generation,
        "worker": worker_index,
        "workers": workers,
        "assignment_path": str(assignment.resolve()),
        "assignment_sha256": activation["assignment_sha256"],
        "assignment_index_path": str(index_path.resolve()),
        "assignment_index_sha256": activation["assignment_index_sha256"],
        "assignment_row_group": worker_index,
        "assignment_row_count": len(rows),
        "assignment_row_group_digest": own["canonical_task_rows_sha256"],
    }
    generation = generation_dir(Path(str(payload["output_dir"])), target)
    wal_path = generation / f"worker-{worker_index}.events.wal"
    with _generation_shared_lease(generation / "generation.lease"):
        # A closer can seal between the first read-only classifier and this
        # lease acquisition.  Reclassify before even opening the WAL inode.
        rechecked = classify_exact_afterany_target_read_only(Path(str(payload["output_dir"])), target)
        if rechecked.kind in {"no-generation", "sealed-generation"}:
            return Path(rechecked.outcome_path or rechecked.closed_path)
        if rechecked.kind != "active-generation":
            if rechecked.exit_code == SUPERSEDED_EXIT_CODE:
                raise ControlSupersededError("worker target was superseded after lease acquisition")
            if rechecked.exit_code == PROTOCOL_EXIT_CODE:
                raise ControlProtocolError("worker target became protocol-invalid after lease acquisition")
            raise ControlBusyError("worker target needs activation recovery after lease acquisition")
        with WorkerEventLog.open_exclusive_and_repair(wal_path, identity=identity) as wal:
            completed = wal.committed_terminal_row_ids()
            spec = CellExecutionSpec.from_payload(analysis.payload["cell_execution_spec"])
            if int(spec.payload["model_n_jobs"]) != 1:
                raise ControlProtocolError("dynamic worker CellExecutionSpec must fix model_n_jobs=1")
            public_schema = tuple(str(column) for column in analysis.payload["public_result_schema"]["columns"])
            with NKGridExecutionSession.open(spec, repo_root=Path(str(payload["cell_spec_repo_root"]))) as session:
                for row in rows:
                    if row.row_id in completed:
                        continue
                    sequence = wal.commit_started(row_id=row.row_id, metadata={"execution_plan_id": execution.execution_plan_id, "round": round_index, "generation": target.submission_generation, "worker": worker_index})
                    computed_rows = session.run_cell_group(seed=row.seed, draw=row.draw, n_samples=row.n_samples, k_features=row.k_features, models=row.models)
                    try:
                        if not isinstance(computed_rows, list) or not computed_rows:
                            raise WALProtocolError("RESULT_PROJECTION_FAILED")
                        if {str(item.get("model")) for item in computed_rows} != set(row.models):
                            raise WALProtocolError("RESULT_KEY_SET_MISMATCH")
                        if any((int(item.get("seed", -1)), int(item.get("draw", -1)), int(item.get("N", -1)), int(item.get("K", -1))) != (row.seed, row.draw, row.n_samples, row.k_features) for item in computed_rows):
                            raise WALProtocolError("RESULT_KEY_SET_MISMATCH")
                        raw_rows = [
                            project_public_result(item, header=public_schema)
                            for item in computed_rows
                            if isinstance(item, Mapping)
                        ]
                        if len(raw_rows) != len(computed_rows):
                            raise WALProtocolError("PUBLIC_SCHEMA_MISMATCH")
                        wal.commit_result(sequence=sequence, row_id=row.row_id, public_rows=raw_rows, header=public_schema)
                    except (WALFrameTooLarge, WALProtocolError, UnicodeError, csv.Error, ValueError, TypeError) as exc:
                        if isinstance(exc, WALFrameTooLarge):
                            reason = "RESULT_FRAME_TOO_LARGE"
                        elif "PUBLIC_SCHEMA_MISMATCH" in str(exc):
                            reason = "PUBLIC_SCHEMA_MISMATCH"
                        elif isinstance(exc, (UnicodeError, csv.Error, TypeError)):
                            reason = "RESULT_ENCODING_FAILED"
                        else:
                            reason = "RESULT_PROTOCOL_VIOLATION"
                        wal.commit_aborted(sequence=sequence, row_id=row.row_id, payload=bounded_abort_payload(reason_code=reason, exception=exc, diagnostic=str(exc)))
                        raise ControlProtocolError(f"durable TASK_ABORTED: {reason}") from exc
    return wal_path


def close_generation(
    snapshot_path: Path, *, round_index: int, submission_generation: str,
    expected_prep_token: str, expected_previous_generation: str | None = None,
    expected_pointer_version: int | None = None, prep_job_id: str | None = None,
    expected_previous_execution_plan_id: str | None = None,
    expected_previous_round_index: int | None = None,
) -> Path:
    """Seal exactly one generation into the immutable cross-WAL inventory."""

    snapshot = _load_snapshot(snapshot_path)
    analysis, execution, target = _exact_target(
        snapshot, round_index=round_index, submission_generation=submission_generation,
        prep_token=expected_prep_token, expected_previous_generation=expected_previous_generation,
        expected_pointer_version=expected_pointer_version, prep_job_id=prep_job_id,
        expected_previous_execution_plan_id=expected_previous_execution_plan_id,
        expected_previous_round_index=expected_previous_round_index,
        validate_task_table=False,
    )
    root = Path(str(snapshot["output_dir"]))
    dispatch = classify_exact_afterany_target_read_only(root, target)
    if dispatch.kind == "no-generation":
        return Path(dispatch.outcome_path)
    if dispatch.kind == "sealed-generation":
        return Path(dispatch.closed_path)
    if dispatch.kind != "active-generation":
        if dispatch.exit_code == SUPERSEDED_EXIT_CODE:
            raise ControlSupersededError("closer target was superseded")
        if dispatch.exit_code == PROTOCOL_EXIT_CODE:
            raise ControlProtocolError("closer target has protocol conflict")
        raise ControlBusyError("closer target needs activation recovery")
    def build_inventory() -> Mapping[str, object]:
        """Run only while ``seal_generation`` holds the exclusive lease."""

        activation = json.loads(Path(dispatch.activation_path).read_text(encoding="utf-8"))
        assignment = Path(str(activation["assignment_path"])); index = Path(str(activation["assignment_index_path"]))
        assignment_sha = sha256_file(assignment)
        index_sha = sha256_file(index)
        if assignment_sha != activation["assignment_sha256"] or index_sha != activation["assignment_index_sha256"]:
            raise ControlProtocolError("closer assignment/index checksum mismatch")
        index_payload = json.loads(index.read_text(encoding="utf-8"))
        groups = index_payload.get("row_groups")
        if not isinstance(groups, list) or len(groups) != int(snapshot["workers"]):
            raise ControlProtocolError("closer assignment index worker groups are invalid")
        wal_inventory: list[dict[str, object]] = []
        for group in sorted(groups, key=lambda value: int(value["worker"])):
            worker = int(group["worker"])
            wal = generation_dir(root, target) / f"worker-{worker}.events.wal"
            if not wal.exists():
                wal_inventory.append({"worker": worker, "wal_state": "absent"})
                continue
            expected_identity = {
                "wal_format": WAL_FORMAT, "analysis_id": analysis.analysis_id,
                "execution_plan_id": execution.execution_plan_id,
                "execution_contract_sha256": execution.sha256, "round": round_index,
                "submission_generation": target.submission_generation, "worker": worker,
                "workers": int(snapshot["workers"]), "assignment_path": str(assignment.resolve()),
                "assignment_sha256": activation["assignment_sha256"], "assignment_index_path": str(index.resolve()),
                "assignment_index_sha256": activation["assignment_index_sha256"], "assignment_row_group": worker,
                "assignment_row_count": int(group["row_count"]),
                "assignment_row_group_digest": group["canonical_task_rows_sha256"],
            }
            # The closer is in the generation-exclusive lease here.  The WAL
            # reader still takes its own nonblocking shared fd lease as a
            # defense against out-of-protocol writers.
            scan = WorkerEventLog.open_shared(wal, expected_identity=expected_identity)
            if scan.identity is None:
                wal_inventory.append({"worker": worker, "wal_state": "absent", "abandoned_uninitialized_wal": True})
                continue
            wal_inventory.append({
                "worker": worker, "wal_state": "present", "path": str(wal.resolve()),
                "size": wal.stat().st_size, "sha256": sha256_file(wal),
                "last_committed_offset": scan.committed_offset,
                "last_commit_trailer_digest": scan.trailer_digest,
                "uncommitted_tail": scan.has_uncommitted_tail,
            })
        return {
            "assignment_path": str(assignment.resolve()), "assignment_sha256": assignment_sha,
            "assignment_index_path": str(index.resolve()), "assignment_index_sha256": index_sha,
            "workers": wal_inventory,
        }

    return seal_generation(root, target, inventory_builder=build_inventory)


def recover_generation_activation(
    snapshot_path: Path, *, round_index: int, submission_generation: str,
    expected_prep_token: str, expected_previous_generation: str | None = None,
    expected_pointer_version: int | None = None, prep_job_id: str | None = None,
    expected_previous_execution_plan_id: str | None = None,
    expected_previous_round_index: int | None = None,
) -> Path:
    """Resume one exact prepared activation; never select a latest target."""

    snapshot = _load_snapshot(snapshot_path)
    analysis, execution, target = _exact_target(
        snapshot, round_index=round_index, submission_generation=submission_generation,
        prep_token=expected_prep_token, expected_previous_generation=expected_previous_generation,
        expected_pointer_version=expected_pointer_version, prep_job_id=prep_job_id,
        expected_previous_execution_plan_id=expected_previous_execution_plan_id,
        expected_previous_round_index=expected_previous_round_index,
        validate_task_table=False,
    )
    root = Path(str(snapshot["output_dir"]))
    dispatch = classify_exact_afterany_target_read_only(root, target)
    if dispatch.kind in {"no-generation", "sealed-generation", "active-generation"}:
        return Path(dispatch.outcome_path or dispatch.closed_path or dispatch.activation_path)
    if dispatch.exit_code == SUPERSEDED_EXIT_CODE:
        raise ControlSupersededError("activation recovery target was superseded")
    if dispatch.exit_code == PROTOCOL_EXIT_CODE:
        raise ControlProtocolError("activation recovery target has a protocol conflict")
    directory = generation_dir(root, target)
    staging = directory.parent / f".{directory.name}.staging"
    if directory.exists() and staging.exists():
        raise ControlProtocolError("activation recovery found both canonical and staging directories")
    if staging.exists():
        prepared = staging / "generation.prepared.json"
        if not prepared.is_file():
            # A staging directory is not an immutable generation.  Its intent
            # freezes the target, while exact prep deterministically rebuilds
            # any unprepared files under the same schedule transaction.
            resumed = prepare_round(
                snapshot_path, round_index=round_index, prep_token=expected_prep_token,
                submission_generation=submission_generation,
                expected_previous_generation=expected_previous_generation,
                expected_pointer_version=expected_pointer_version,
                prep_job_id=prep_job_id,
                expected_previous_execution_plan_id=expected_previous_execution_plan_id,
                expected_previous_round_index=expected_previous_round_index,
            )
            activation_value = resumed.get("activation")
            if not isinstance(activation_value, str):
                raise ControlProtocolError("activation recovery resumed to a non-generation outcome")
            return Path(activation_value)
        try:
            os.rename(staging, directory)
            parent_fd = os.open(directory.parent, os.O_RDONLY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        except OSError as exc:
            raise ControlBusyError(f"activation staging promotion is indeterminate: {exc}") from exc
    prepared_path = directory / "generation.prepared.json"
    if not prepared_path.is_file():
        raise ControlBusyError("activation recovery needs a complete prepared record")
    try:
        prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
        activation = activate_generation(
            root, target,
            assignment_path=Path(str(prepared["assignment_path"])), assignment_sha256=str(prepared["assignment_sha256"]),
            assignment_index_path=Path(str(prepared["assignment_index_path"])), assignment_index_sha256=str(prepared["assignment_index_sha256"]),
            ready_path=Path(str(prepared["ready_path"])), ready_sha256=str(prepared["ready_sha256"]),
            prep_path=Path(str(prepared["prep_path"])), prep_sha256=str(prepared["prep_sha256"]),
            worker_count=int(prepared["worker_count"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ControlProtocolError("activation recovery prepared record is invalid") from exc
    return activation


def _round_directory(snapshot: Mapping[str, object], round_index: int) -> Path:
    if round_index < 1:
        raise ValueError("round_index must be >= 1")
    return Path(str(snapshot["output_dir"])) / f"round-{round_index}"


def _load_snapshot(path: Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format_version") != TABLE_FORMAT_VERSION:
        raise ValueError("unsupported legacy dynamic snapshot/result store; rebuild a worker-event-wal-v1 plan")
    if "analysis_contract" in payload:
        if payload.get("result_store_format") != "worker-event-wal-v1":
            raise ValueError("dynamic WAL snapshot is missing its result-store identity")
    elif not (
        payload.get("result_store_format") == "test-legacy-csv-v2"
        and payload.get("test_compatibility_mode") is True
    ):
        raise ValueError("unsupported legacy dynamic snapshot/result store; rebuild a worker-event-wal-v1 plan")
    return payload


def _load_contract_chain(
    snapshot: Mapping[str, object], *, validate_task_table: bool = True,
) -> tuple[AnalysisContract, DynamicExecutionContract]:
    """Load immutable contract files before accepting any task/result key."""

    try:
        analysis_path = Path(str(snapshot["analysis_contract"]))
        execution_path = Path(str(snapshot["execution_contract"]))
        analysis_raw = analysis_path.read_bytes(); execution_raw = execution_path.read_bytes()
        analysis_payload = json.loads(analysis_raw.decode("utf-8"))
        execution_payload = json.loads(execution_raw.decode("utf-8"))
        if (
            analysis_raw != canonical_json_bytes(analysis_payload) + b"\n"
            or execution_raw != canonical_json_bytes(execution_payload) + b"\n"
        ):
            raise ContractError("immutable contract file is not canonical")
        analysis = AnalysisContract.from_payload(analysis_payload)
        execution = DynamicExecutionContract.from_payload(execution_payload)
    except (KeyError, OSError, UnicodeDecodeError, json.JSONDecodeError, ContractError) as exc:
        raise ValueError("dynamic snapshot lacks a valid immutable contract chain") from exc
    if (
        snapshot.get("analysis_id") != analysis.analysis_id
        or snapshot.get("analysis_contract_sha256") != analysis.sha256
        or snapshot.get("execution_plan_id") != execution.execution_plan_id
        or snapshot.get("execution_contract_sha256") != execution.sha256
        or execution.payload.get("analysis_id") != analysis.analysis_id
        or execution.payload.get("analysis_contract_sha256") != analysis.sha256
    ):
        raise ValueError("dynamic snapshot immutable contract identity mismatch")
    table_path = Path(str(snapshot["task_table"]))
    if execution.payload.get("task_table_path") != str(table_path.resolve()):
        raise ValueError("dynamic snapshot task table path mismatch")
    if validate_task_table and execution.payload.get("task_table_file_sha256") != sha256_file(table_path):
        raise ValueError("dynamic snapshot task table checksum mismatch")
    return analysis, execution


def _exact_target(
    snapshot: Mapping[str, object], *, round_index: int, submission_generation: str,
    prep_token: str, expected_previous_generation: str | None,
    expected_pointer_version: int | None, prep_job_id: str | None,
    expected_previous_execution_plan_id: str | None = None,
    expected_previous_round_index: int | None = None,
    validate_task_table: bool = True,
) -> tuple[AnalysisContract, DynamicExecutionContract, ActivationTarget]:
    analysis, execution = _load_contract_chain(snapshot, validate_task_table=validate_task_table)
    target = ActivationTarget(
        analysis_id=analysis.analysis_id,
        execution_plan_id=execution.execution_plan_id,
        execution_contract_sha256=execution.sha256,
        round_index=int(round_index),
        submission_generation=str(submission_generation),
        expected_previous_generation=expected_previous_generation,
        expected_pointer_version=(int(round_index) - 1 if expected_pointer_version is None else int(expected_pointer_version)),
        prep_job_id=str(prep_job_id or prep_token),
        prep_token=_validated_prep_token(prep_token),
        expected_previous_execution_plan_id=expected_previous_execution_plan_id,
        expected_previous_round_index=expected_previous_round_index,
    )
    return analysis, execution, target


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
        CREATE TABLE terminal_payloads (
            {_QUEUE_KEY_SQL},
            status TEXT NOT NULL,
            payload TEXT NOT NULL,
            PRIMARY KEY ({_QUEUE_KEY_SQL})
        ) WITHOUT ROWID;
        CREATE TABLE attempts (
            ingest_order INTEGER PRIMARY KEY,
            execution_plan_id TEXT NOT NULL,
            round_index INTEGER NOT NULL,
            submission_generation TEXT NOT NULL,
            worker_index INTEGER NOT NULL,
            sequence INTEGER NOT NULL,
            row_id TEXT NOT NULL
        );
        CREATE INDEX attempts_order ON attempts (execution_plan_id, round_index, submission_generation, worker_index, sequence, ingest_order);
        CREATE TABLE completed_rows (row_id TEXT PRIMARY KEY) WITHOUT ROWID;
        CREATE TABLE crashed_rows (row_id TEXT PRIMARY KEY) WITHOUT ROWID;
        CREATE TABLE too_long_rows (row_id TEXT PRIMARY KEY) WITHOUT ROWID;
        CREATE TABLE aborted_rows (row_id TEXT PRIMARY KEY, reason_code TEXT NOT NULL) WITHOUT ROWID;
        CREATE TABLE failed_attempt_rows (row_id TEXT PRIMARY KEY) WITHOUT ROWID;
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
    wal_inputs = tuple(sorted(
        output_dir.glob("executions/*/round-*/generation-*/worker-*.events.wal")
    ))
    inputs = (*_result_shards(output_dir), *_attempt_logs(output_dir), *wal_inputs)
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
        "INSERT INTO attempts (execution_plan_id, round_index, submission_generation, worker_index, sequence, row_id) "
        "VALUES (?, ?, ?, ?, ?, ?)"
    )
    for path in _attempt_logs(output_dir):
        try:
            with path.open(encoding="utf-8") as source:
                for line in source:
                    try:
                        item = json.loads(line)
                        connection.execute(statement, (
                            "__legacy__", int(item["round"]), "__legacy__", int(item["worker_index"]),
                            int(item["sequence"]), str(item["row_id"]),
                        ))
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                        # A Slurm SIGKILL can leave only the final JSON line malformed.
                        continue
        except (OSError, UnicodeError) as exc:
            raise ValueError(f"cannot read attempt log {path}: {exc}") from exc
        connection.commit()


def _iter_sealed_wal_scans(
    output_dir: Path, *, analysis: AnalysisContract,
    frozen_frontier: Sequence[Mapping[str, object]] | None = None,
):
    """Yield only fully validated, shared-locked WALs behind closed markers."""

    root = Path(output_dir) / "executions"
    if frozen_frontier is None:
        if not root.exists():
            return
        markers = sorted(root.glob("*/round-*/generation-*/generation.closed.json"))
    else:
        markers = [Path(str(item["closed_path"])) for item in frozen_frontier]
    for marker in markers:
        if not marker.is_file():
            raise ControlProtocolError(f"sealed generation marker is missing: {marker}")
        try:
            closed = json.loads(marker.read_text(encoding="utf-8"))
            activation = json.loads((marker.parent / "generation.activation.json").read_text(encoding="utf-8"))
            target = ActivationTarget(
                analysis_id=str(closed["analysis_id"]),
                execution_plan_id=str(closed["execution_plan_id"]),
                execution_contract_sha256=str(closed["execution_contract_sha256"]),
                round_index=int(closed["round"]),
                submission_generation=str(closed["submission_generation"]),
                expected_previous_generation=(
                    None if activation.get("expected_previous_generation_or_null") is None
                    else str(activation["expected_previous_generation_or_null"])
                ),
                expected_pointer_version=int(activation["expected_pointer_version"]),
                prep_job_id=str(closed["prep_job_id"]),
                prep_token=str(closed["prep_token"]),
                expected_previous_execution_plan_id=(
                    None if activation.get("expected_previous_execution_plan_id_or_null") is None
                    else str(activation["expected_previous_execution_plan_id_or_null"])
                ),
                expected_previous_round_index=(
                    None if activation.get("expected_previous_round_or_null") is None
                    else int(activation["expected_previous_round_or_null"])
                ),
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ControlProtocolError("sealed generation control records are invalid") from exc
        if target.analysis_id != analysis.analysis_id:
            raise ControlProtocolError("sealed generation belongs to another analysis")
        execution_file = Path(output_dir) / "execution-contracts" / f"{target.execution_plan_id}.json"
        try:
            execution_raw = execution_file.read_bytes()
            execution_payload = json.loads(execution_raw.decode("utf-8"))
            if execution_raw != canonical_json_bytes(execution_payload) + b"\n":
                raise ContractError("sealed execution contract is not canonical")
            contract = DynamicExecutionContract.from_payload(execution_payload)
        except (OSError, ValueError, json.JSONDecodeError, ContractError) as exc:
            raise ControlProtocolError("sealed generation execution contract is invalid") from exc
        if (
            contract.sha256 != target.execution_contract_sha256
            or contract.payload.get("analysis_id") != analysis.analysis_id
            or contract.payload.get("analysis_contract_sha256") != analysis.sha256
        ):
            raise ControlProtocolError("sealed generation execution contract scope mismatch")
        dispatch = classify_exact_afterany_target_read_only(Path(output_dir), target)
        if dispatch.kind != "sealed-generation" or dispatch.closed_path is None:
            raise ControlProtocolError("closed marker is not an exact sealed generation")
        if dispatch.closed_path.resolve() != marker.resolve():
            raise ControlProtocolError("sealed generation marker is not the exact control target")
        if frozen_frontier is not None:
            expected_sha = next((item.get("closed_sha256") for item in frozen_frontier if Path(str(item.get("closed_path"))).resolve() == marker.resolve()), None)
            if expected_sha != sha256_file(marker):
                raise ControlProtocolError("frozen sealed marker checksum mismatch")
        inventory = closed.get("inventory")
        if not isinstance(inventory, Mapping) or not isinstance(inventory.get("workers"), list):
            raise ControlProtocolError("sealed generation has no immutable inventory")
        assignment = Path(str(inventory["assignment_path"])); index = Path(str(inventory["assignment_index_path"]))
        index_payload = json.loads(index.read_text(encoding="utf-8"))
        groups = {int(group["worker"]): group for group in index_payload.get("row_groups", [])}
        for item in sorted(inventory["workers"], key=lambda value: int(value["worker"])):
            if not isinstance(item, Mapping):
                raise ControlProtocolError("sealed worker inventory entry is invalid")
            worker = int(item["worker"])
            wal_path = marker.parent / f"worker-{worker}.events.wal"
            if item.get("wal_state") == "absent":
                # ``classify`` above invokes the canonical closed validator,
                # including byte-level validation of an abandoned empty inode.
                continue
            if item.get("wal_state") != "present" or worker not in groups:
                raise ControlProtocolError("sealed worker inventory scope is invalid")
            group = groups[worker]
            expected_identity = {
                "wal_format": WAL_FORMAT, "analysis_id": target.analysis_id,
                "execution_plan_id": target.execution_plan_id,
                "execution_contract_sha256": target.execution_contract_sha256,
                "round": target.round_index, "submission_generation": target.submission_generation,
                "worker": worker, "workers": len(groups),
                "assignment_path": str(assignment.resolve()), "assignment_sha256": inventory["assignment_sha256"],
                "assignment_index_path": str(index.resolve()), "assignment_index_sha256": inventory["assignment_index_sha256"],
                "assignment_row_group": worker, "assignment_row_count": int(group["row_count"]),
                "assignment_row_group_digest": group["canonical_task_rows_sha256"],
            }
            scan = WorkerEventLog.open_shared(wal_path, expected_identity=expected_identity)
            yield marker, wal_path, scan


def _index_queue_wals(
    connection: sqlite3.Connection, output_dir: Path, *, analysis: AnalysisContract,
    frozen_frontier: Sequence[Mapping[str, object]] | None = None,
) -> None:
    """Stream durable WAL facts into the existing local-only SQLite index."""

    complete_insert = f"INSERT OR IGNORE INTO completed ({_QUEUE_KEY_SQL}) VALUES (?, ?, ?, ?, ?)"
    terminal_insert = (
        f"INSERT OR IGNORE INTO terminal_payloads ({_QUEUE_KEY_SQL}, status, payload) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)"
    )
    terminal_select = (
        "SELECT status, payload FROM terminal_payloads WHERE "
        + " AND ".join(f'\"{column}\"=?' for column in ("model", "seed", "draw", "N", "K"))
    )
    attempt_insert = "INSERT INTO attempts (execution_plan_id, round_index, submission_generation, worker_index, sequence, row_id) VALUES (?, ?, ?, ?, ?, ?)"
    for marker, wal_path, scan in _iter_sealed_wal_scans(
        output_dir, analysis=analysis, frozen_frontier=frozen_frontier,
    ) or ():
        identity = scan.identity or {}
        try:
            execution_plan_id = str(identity["execution_plan_id"])
            generation = str(identity["submission_generation"])
            round_index = int(identity["round"]); worker_index = int(identity["worker"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"WAL has invalid scope identity: {wal_path}") from exc
        terminal: set[tuple[int, str]] = set()
        starts: list[tuple[int, str]] = []
        for record in scan.records:
            if record.event_type == TASK_STARTED:
                starts.append((record.sequence, str(record.row_id)))
            elif record.event_type == TASK_ABORTED:
                terminal.add((record.sequence, str(record.row_id)))
                try:
                    abort = json.loads(record.payload.decode("utf-8"))
                    reason = str(abort["reason_code"])
                except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise ValueError("TASK_ABORTED payload is corrupt") from exc
                connection.execute("INSERT OR IGNORE INTO aborted_rows VALUES (?, ?)", (str(record.row_id), reason))
            elif record.event_type == TASK_RESULT:
                terminal.add((record.sequence, str(record.row_id)))
                header, public_rows = decode_public_rows(record.payload)
                required = {"model", "seed", "draw", "N", "K", "status"}
                if not required.issubset(header):
                    raise ValueError("TASK_RESULT public schema lacks required key/status columns")
                for public in public_rows:
                    key = _csv_key(public)
                    row_id = str(record.row_id)
                    expected = connection.execute(
                        'SELECT row_id FROM expected WHERE "model"=? AND "seed"=? AND "draw"=? AND "N"=? AND "K"=?',
                        key,
                    ).fetchone()
                    if expected is None or str(expected[0]) != row_id:
                        raise ValueError("TASK_RESULT row ID/public key is outside the frozen task design")
                    status = public.get("status")
                    if status in TERMINAL_STATUSES:
                        stable_payload = json.dumps(
                            [public[column] for column in header],
                            ensure_ascii=False, separators=(",", ":"),
                        )
                        inserted = connection.execute(
                            terminal_insert, (*key, str(status), stable_payload),
                        ).rowcount
                        if not inserted:
                            previous = connection.execute(terminal_select, key).fetchone()
                            if previous != (str(status), stable_payload):
                                raise ControlProtocolError("conflicting durable terminal RESULT payload")
                        connection.execute(complete_insert, key)
                    elif status == "failed":
                        connection.execute("INSERT OR IGNORE INTO failed_attempt_rows VALUES (?)", (row_id,))
                    else:
                        raise ValueError("TASK_RESULT has unsupported public status")
        for sequence, row_id in starts:
            if (sequence, row_id) not in terminal:
                connection.execute(attempt_insert, (execution_plan_id, round_index, generation, worker_index, sequence, row_id))
        connection.commit()


def _sealed_history_digest(output_dir: Path, *, analysis_id: str) -> str:
    frontier: list[dict[str, object]] = []
    root = Path(output_dir) / "executions"
    if root.exists():
        for marker in sorted(root.glob("*/round-*/generation-*/generation.closed.json")):
            payload = json.loads(marker.read_text(encoding="utf-8"))
            if payload.get("analysis_id") != analysis_id:
                raise ValueError("sealed history includes a different analysis")
            frontier.append({"closed_path": str(marker.resolve()), "closed_sha256": sha256_file(marker)})
    return hashlib.sha256(canonical_json_bytes(frontier)).hexdigest()


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

    connection.execute("CREATE TABLE final_attempts (execution_plan_id TEXT, round_index INTEGER, submission_generation TEXT, worker_index INTEGER, row_id TEXT, PRIMARY KEY (execution_plan_id, round_index, submission_generation, worker_index)) WITHOUT ROWID")
    prior_group: tuple[str, int, str, int] | None = None
    final_sequence: int | None = None
    final_row_id: str | None = None
    cursor = connection.execute(
        "SELECT execution_plan_id, round_index, submission_generation, worker_index, sequence, row_id FROM attempts "
        "ORDER BY execution_plan_id, round_index, submission_generation, worker_index, sequence, ingest_order"
    )
    for execution_plan_id, round_index, generation, worker_index, sequence, row_id in cursor:
        group = (str(execution_plan_id), int(round_index), str(generation), int(worker_index))
        if group != prior_group:
            if prior_group is not None and final_row_id is not None:
                connection.execute(
                    "INSERT INTO final_attempts VALUES (?, ?, ?, ?, ?)",
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
        connection.execute("INSERT INTO final_attempts VALUES (?, ?, ?, ?, ?)", (*prior_group, final_row_id))
    # ``attempts`` contains only durable START records without a matching
    # RESULT/ABORTED.  The final one in each worker invocation is interrupted
    # too; retaining it only for the too-long reducer previously hid it from
    # verification diagnostics.
    connection.execute("INSERT OR IGNORE INTO crashed_rows SELECT DISTINCT row_id FROM attempts")
    connection.execute("CREATE TABLE final_rounds (execution_plan_id TEXT, row_id TEXT, round_index INTEGER, PRIMARY KEY (execution_plan_id, row_id, round_index)) WITHOUT ROWID")
    connection.execute("INSERT INTO final_rounds SELECT DISTINCT execution_plan_id, row_id, round_index FROM final_attempts")
    connection.execute("""
        INSERT INTO too_long_rows (row_id)
        SELECT DISTINCT first.row_id FROM final_rounds first
        JOIN final_rounds second
          ON second.execution_plan_id=first.execution_plan_id AND second.row_id=first.row_id AND second.round_index=first.round_index + 1
        JOIN final_rounds third
          ON third.execution_plan_id=first.execution_plan_id AND third.row_id=first.row_id AND third.round_index=first.round_index + 2
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
) -> tuple[Path, list[int], list[dict[str, object]]]:
    """Publish exactly one modulo-ordered Parquet row group for every worker."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    writer = pq.ParquetWriter(temporary, _empty_arrow_table().schema, compression="zstd")
    assigned_rows: list[int] = []
    row_groups: list[dict[str, object]] = []
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
            row_groups.append({
                "worker": worker_index,
                "row_count": len(rows),
                "canonical_task_rows_sha256": task_row_digest(rows),
            })
            writer.write_table(_arrow_table(rows) if rows else _empty_arrow_table())
    finally:
        writer.close()
    _durable_replace(temporary, path)
    os.chmod(path, 0o444)
    return path, assigned_rows, row_groups


def _assignment_ready_path(round_dir: Path) -> Path:
    return Path(round_dir) / "assignment.ready.json"


def _validated_prep_token(value: str) -> str:
    if not isinstance(value, str) or not value or any(character.isspace() for character in value):
        raise ValueError("prep token must be a non-empty, whitespace-free string")
    return value


def _require_ready_assignment(
    snapshot: Mapping[str, object], *, round_index: int, workers: int,
    expected_prep_token: str,
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
        or payload.get("prep_token") != expected_prep_token
    ):
        raise RuntimeError(
            f"assignment readiness record is invalid for round {round_index}; prep may have failed: {ready}"
        )
    if not assignment.is_file():
        raise RuntimeError(
            f"assignment is not ready for round {round_index}; prep may have failed: {assignment}"
        )
    return assignment


# Compatibility is intentionally restricted to in-process/test snapshots made
# by the old helper without any contract fields.  On-disk v1/v2 snapshots are
# rejected by ``_load_snapshot``; production planning always supplies the
# contract chain and therefore cannot reach these helpers.
def _legacy_prepare_round(snapshot_path: Path, *, round_index: int, prep_token: str, tmp_dir: Path | None) -> dict[str, object]:
    payload = _load_snapshot(snapshot_path)
    output_dir = Path(str(payload["output_dir"])); workers = int(payload["workers"])
    table_path = Path(str(payload["task_table"])); round_dir = _round_directory(payload, round_index); round_dir.mkdir(parents=True, exist_ok=True)
    stage_directory: Path | None = None
    with _queue_index(payload, explicit_tmp_dir=tmp_dir, phase="preparation") as connection:
        _index_queue_expected_design(connection, table_path)
        completed_model_keys = _index_queue_completed_shards(connection, output_dir)
        _index_queue_attempts(connection, output_dir); _index_completed_rows(connection); _classify_queue_attempts(connection); _queue_incomplete_rows(connection)
        todo_rows = _stage_queue_todo(connection, table_path, workers=workers)
        assignment, assigned_rows, _ = _write_assignment_from_queue_index(round_dir / "assignment.parquet", connection, workers=workers)
        crashed = _queue_row_ids(connection, "crashed_rows"); too_long = _queue_row_ids(connection, "too_long_rows")
    stats: dict[str, object] = {"round": round_index, "workers": workers, "todo_rows": todo_rows, "assigned_rows": assigned_rows, "completed_model_keys": completed_model_keys, "crashed_row_ids": crashed, "too_long_row_ids": too_long, "assignment": str(assignment), "prep_token": prep_token}
    write_json_atomic(round_dir / "crashed.json", {"round": round_index, "row_ids": crashed}); write_json_atomic(round_dir / "too-long.json", {"round": round_index, "row_ids": too_long}); write_json_atomic(round_dir / "prep.json", stats)
    write_json_atomic(_assignment_ready_path(round_dir), {"format_version": TABLE_FORMAT_VERSION, "round": round_index, "workers": workers, "assignment": str(assignment.resolve()), "prep_token": prep_token})
    return stats


def _legacy_run_slice(snapshot_path: Path, *, round_index: int, worker_index: int, expected_prep_token: str) -> Path:
    payload = _load_snapshot(snapshot_path); workers = int(payload["workers"])
    if not 0 <= worker_index < workers:
        raise IndexError("worker_index is outside the frozen worker count")
    assignment = _require_ready_assignment(payload, round_index=round_index, workers=workers, expected_prep_token=_validated_prep_token(expected_prep_token))
    rows = read_row_group(assignment, worker_index); output = _round_directory(payload, round_index) / f"worker-{worker_index}.csv"
    _sweep_slice_temporaries(output, round_index=round_index, worker_index=worker_index)
    header: list[str] | None = None; materialized: list[dict[str, str]] = []
    if output.exists(): header, materialized = _read_csv_keys(output)
    by_key = {_csv_key(row): row for row in materialized}
    if len(by_key) != len(materialized): raise ValueError("worker output contains duplicate keys")
    completed = {key for key, row in by_key.items() if row.get("status") in TERMINAL_STATUSES}; config = _config_from_json(payload["config"])
    for sequence, row in enumerate(pending_rows(rows, completed)):
        _append_attempt(_round_directory(payload, round_index) / "attempts" / f"worker-{worker_index}.jsonl", round_index=round_index, worker_index=worker_index, sequence=sequence, row_id=row.row_id)
        row_out = output.parent / f".{output.stem}.round-{round_index}.worker-{worker_index}.{row.row_id}.csv"
        row_config = replace(config, out=row_out, models=row.models, n_grid=(row.n_samples,), k_grid=(row.k_features,), n_seeds=1, n_draws=1, repeat_plan=((row.seed, row.draw),), n_jobs=1)
        run_nk_grid(row_config, execution_pairs=((row.seed, row.draw),), exact_output_path=True, defer_failure_policy=True)
        current_header, current_rows = _read_csv_keys(row_out)
        if header is None: header = current_header
        elif current_header != header: raise ValueError("worker rows produced inconsistent CSV headers")
        for current in current_rows: by_key[_csv_key(current)] = current
        _write_materialized_rows(output, header, by_key, execution={"mode": "slice", "round": round_index, "worker_index": worker_index}, expected_rows=len(expected_model_keys(rows)))
        row_out.unlink(missing_ok=True); manifest_path(row_out).unlink(missing_ok=True)
    if header is None and rows: raise RuntimeError("slice produced no rows")
    if header is None: write_json_atomic(manifest_path(output), {"format_version": TABLE_FORMAT_VERSION, "execution": {"mode": "slice", "round": round_index, "worker_index": worker_index}, "completion": {"expected_rows": 0, "materialized_rows": 0}})
    return output


def _legacy_verify_rounds(snapshot_path: Path, *, tmp_dir: Path | None) -> dict[str, object]:
    payload = _load_snapshot(snapshot_path); output_dir = Path(str(payload["output_dir"])); table_path = Path(str(payload["task_table"]))
    with _queue_index(payload, explicit_tmp_dir=tmp_dir, phase="verification") as connection:
        expected = _index_queue_expected_design(connection, table_path); complete = _index_queue_completed_shards(connection, output_dir); _index_queue_attempts(connection, output_dir); _index_completed_rows(connection); _classify_queue_attempts(connection)
        result = {"expected_model_keys": expected, "completed_model_keys": complete, "missing_model_keys": _queue_missing_model_keys(connection), "crashed_row_ids": _queue_row_ids(connection, "crashed_rows"), "too_long_row_ids": _queue_row_ids(connection, "too_long_rows")}
    write_json_atomic(output_dir / "verification.json", result)
    return result


def prepare_round(
    snapshot_path: Path, *, round_index: int, prep_token: str,
    tmp_dir: Path | None = None,
    submission_generation: str | None = None,
    expected_previous_generation: str | None = None,
    expected_pointer_version: int | None = None,
    prep_job_id: str | None = None,
    expected_previous_execution_plan_id: str | None = None,
    expected_previous_round_index: int | None = None,
    _schedule_locked: bool = False,
) -> dict[str, object]:
    """Prepare one immutable generation, or publish the exact todo=0 outcome."""

    prep_token = _validated_prep_token(prep_token)
    payload = _load_snapshot(snapshot_path)
    if "analysis_contract" not in payload:
        return _legacy_prepare_round(snapshot_path, round_index=round_index, prep_token=prep_token, tmp_dir=tmp_dir)
    analysis, execution = _load_contract_chain(payload)
    output_dir = Path(str(payload["output_dir"])); workers = int(payload["workers"])
    if workers != int(execution.payload["worker_count"]):
        raise ValueError("snapshot worker count differs from immutable execution contract")
    config = _config_from_json(payload["config"])
    if config.n_jobs != 1:
        raise ValueError("dynamic worker config must freeze model_n_jobs=1")
    generation = submission_generation or uuid.uuid4().hex
    target = ActivationTarget(
        analysis_id=analysis.analysis_id,
        execution_plan_id=execution.execution_plan_id,
        execution_contract_sha256=execution.sha256,
        round_index=int(round_index),
        submission_generation=str(generation),
        expected_previous_generation=expected_previous_generation,
        expected_pointer_version=(int(round_index) - 1 if expected_pointer_version is None else int(expected_pointer_version)),
        prep_job_id=str(prep_job_id or prep_token),
        prep_token=prep_token,
        expected_previous_execution_plan_id=expected_previous_execution_plan_id,
        expected_previous_round_index=expected_previous_round_index,
    )
    # Keep one exclusive analysis schedule lease from the predecessor/history
    # read through durable outcome publication or pointer CAS.  The recursive
    # entry is deliberately private: it avoids a second fd/lock while keeping
    # the public preparation API small and exact-target based.
    if not _schedule_locked:
        with schedule_transaction(output_dir):
            return prepare_round(
                snapshot_path, round_index=round_index, prep_token=prep_token,
                tmp_dir=tmp_dir, submission_generation=target.submission_generation,
                expected_previous_generation=expected_previous_generation,
                expected_pointer_version=target.expected_pointer_version,
                prep_job_id=target.prep_job_id, _schedule_locked=True,
                expected_previous_execution_plan_id=target.expected_previous_execution_plan_id,
                expected_previous_round_index=target.expected_previous_round_index,
            )
    table_path = Path(str(payload["task_table"]))
    predecessor_kind, predecessor = predecessor_gate(output_dir, target)
    if predecessor_kind == "no-generation":
        assert isinstance(predecessor, tuple)
        prior_outcome = predecessor[1]
        assert isinstance(prior_outcome, Mapping)
        raw_frontier = prior_outcome.get("sealed_history_frontier")
        if not isinstance(raw_frontier, list):
            raise ControlProtocolError("predecessor no-generation outcome lacks frozen history")
        frozen_frontier = tuple(dict(item) for item in raw_frontier if isinstance(item, Mapping))
        if len(frozen_frontier) != len(raw_frontier):
            raise ControlProtocolError("predecessor no-generation outcome history is malformed")
    else:
        assert isinstance(predecessor, tuple)
        frozen_frontier = frozen_history_from_closed(
            output_dir, closed_sha256=predecessor[0], closed_path=predecessor[1],
        )
    with _queue_index(payload, explicit_tmp_dir=tmp_dir, phase="preparation") as connection:
        _index_queue_expected_design(connection, table_path)
        _index_queue_wals(connection, output_dir, analysis=analysis, frozen_frontier=frozen_frontier)
        _index_completed_rows(connection)
        _classify_queue_attempts(connection)
        if _queue_row_ids(connection, "aborted_rows"):
            raise ControlProtocolError("durable TASK_ABORTED exists; refusing to publish a new ready assignment")
        _queue_incomplete_rows(connection)
        todo_rows = _stage_queue_todo(connection, table_path, workers=workers)
        completed_model_keys = int(connection.execute("SELECT COUNT(*) FROM completed").fetchone()[0])
        crashed = _queue_row_ids(connection, "crashed_rows")
        too_long = _queue_row_ids(connection, "too_long_rows")
        if todo_rows == 0:
            assignment = None; assigned_rows: list[int] = []; row_groups: list[dict[str, object]] = []
        else:
            if predecessor_kind == "no-generation":
                raise ControlProtocolError(
                    "todo reappeared after an exact no-generation predecessor; refusing activation"
                )
            directory = generation_dir(output_dir, target)
            # The intent is the first durable generation artefact.  Build all
            # mutable preparation output in a private sibling and promote it
            # only after every immutable child has been fsynced.
            intent_file = publish_activation_intent(output_dir, target, lease_held=True)
            if directory.exists():
                raise ControlProtocolError("generation directory already exists before immutable activation")
            stage_directory = directory.parent / f".{directory.name}.staging"
            try:
                stage_directory.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise ControlBusyError("activation staging directory cannot be resumed") from exc
            # The lease inode is a generation artefact, not an activation
            # afterthought.  Create and sync it inside private staging before
            # any prepared record can make this directory promotable.
            lease_path = stage_directory / "generation.lease"
            if lease_path.exists() and not lease_path.is_file():
                raise ControlProtocolError("activation staging lease is not a regular file")
            lease_descriptor = os.open(lease_path, os.O_RDWR | os.O_CREAT, 0o640)
            try:
                os.fsync(lease_descriptor)
            finally:
                os.close(lease_descriptor)
            _fsync_directory(stage_directory)
            assignment, assigned_rows, row_groups = _write_assignment_from_queue_index(
                stage_directory / "assignment.parquet", connection, workers=workers,
            )
    if todo_rows == 0:
        receipt = publish_no_generation_outcome(output_dir, target, lease_held=True)
        return {
            "round": round_index, "todo_rows": 0, "no_generation": True,
            "prep_outcome": str(receipt), "submission_generation": target.submission_generation,
            "prep_token": prep_token,
        }
    assert assignment is not None
    assert stage_directory is not None
    directory = generation_dir(output_dir, target)
    stage_assignment = assignment
    canonical_assignment = directory / stage_assignment.name
    assignment_sha = sha256_file(assignment)
    index = stage_directory / "assignment.index.json"
    canonical_index = directory / index.name
    index_payload = {
        "execution_plan_id": execution.execution_plan_id,
        "execution_contract_sha256": execution.sha256,
        "submission_generation": target.submission_generation,
        "round": round_index,
        "workers": workers,
        "assignment_file_sha256": assignment_sha,
        "row_groups": row_groups,
    }
    immutable_json_bytes(index, index_payload)
    index_sha = sha256_file(index)
    stats: dict[str, object] = {
        "round": round_index, "workers": workers, "todo_rows": todo_rows,
        "assigned_rows": assigned_rows, "completed_model_keys": completed_model_keys,
        "interrupted_row_ids": crashed, "too_long_row_ids": too_long,
        "assignment": str(canonical_assignment.resolve()), "assignment_index": str(canonical_index.resolve()),
        "submission_generation": target.submission_generation, "prep_token": prep_token,
        "prep_job_id": target.prep_job_id,
    }
    prep = stage_directory / "prep.json"; immutable_json_bytes(prep, stats)
    ready = stage_directory / "assignment.ready.json"
    ready_payload = {
        "format_version": TABLE_FORMAT_VERSION, "execution_plan_id": execution.execution_plan_id,
        "execution_contract_sha256": execution.sha256, "round": round_index, "workers": workers,
        "submission_generation": target.submission_generation, "assignment": str(canonical_assignment.resolve()),
        "assignment_sha256": assignment_sha, "assignment_index": str(canonical_index.resolve()),
        "assignment_index_sha256": index_sha, "prep_token": prep_token, "prep_job_id": target.prep_job_id,
    }
    immutable_json_bytes(ready, ready_payload)
    ready_sha = sha256_file(ready)
    # ``prepared`` is part of the immutable staging directory, not a record
    # synthesized after canonical promotion.  ``activate_generation`` will
    # only byte-verify this exact record before publishing activation/CAS.
    prepared = stage_directory / "generation.prepared.json"
    prepared_payload = {
        "prepared_format_version": 1,
        "analysis_id": target.analysis_id,
        "execution_plan_id": target.execution_plan_id,
        "execution_contract_sha256": target.execution_contract_sha256,
        "round": target.round_index,
        "submission_generation": target.submission_generation,
        "intent_sha256": sha256_file(intent_file),
        "assignment_path": str(canonical_assignment.resolve()),
        "assignment_sha256": assignment_sha,
        "assignment_index_path": str(canonical_index.resolve()),
        "assignment_index_sha256": index_sha,
        "ready_path": str((directory / ready.name).resolve()),
        "ready_sha256": ready_sha,
        "prep_path": str((directory / prep.name).resolve()),
        "prep_sha256": sha256_file(prep),
        "worker_count": workers,
    }
    immutable_json_bytes(prepared, prepared_payload)
    # Directory rename is the generation-artifact publication boundary.  The
    # parent fsync makes it durable before the prepared/activation records.
    try:
        os.rename(stage_directory, directory)
    except FileExistsError as exc:
        raise ControlBusyError("canonical generation appeared while staging") from exc
    parent_fd = os.open(directory.parent, os.O_RDONLY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    assignment = directory / stage_assignment.name
    index = directory / index.name
    prep = directory / prep.name
    ready = directory / ready.name
    prep_sha = sha256_file(prep)
    activation = activate_generation(
        output_dir, target, assignment_path=assignment, assignment_sha256=assignment_sha,
        assignment_index_path=index, assignment_index_sha256=index_sha, ready_path=ready,
        ready_sha256=ready_sha, prep_path=prep, prep_sha256=prep_sha, worker_count=workers,
        lease_held=True,
    )
    stats["activation"] = str(activation)
    return stats


def verify_rounds(
    snapshot_path: Path, *, tmp_dir: Path | None = None,
    round_index: int | None = None, submission_generation: str | None = None,
    expected_prep_token: str | None = None, expected_previous_generation: str | None = None,
    expected_pointer_version: int | None = None, prep_job_id: str | None = None,
    expected_previous_execution_plan_id: str | None = None,
    expected_previous_round_index: int | None = None,
) -> dict[str, object]:
    """Verify only an exact sealed/outcome target and its frozen WAL history."""

    payload = _load_snapshot(snapshot_path)
    if "analysis_contract" not in payload:
        return _legacy_verify_rounds(snapshot_path, tmp_dir=tmp_dir)
    if round_index is None or submission_generation is None or expected_prep_token is None:
        raise ValueError("verify requires exact last round, generation, and prep token")
    analysis, execution, target = _exact_target(
        payload, round_index=round_index, submission_generation=submission_generation,
        prep_token=expected_prep_token, expected_previous_generation=expected_previous_generation,
        expected_pointer_version=expected_pointer_version, prep_job_id=prep_job_id,
        expected_previous_execution_plan_id=expected_previous_execution_plan_id,
        expected_previous_round_index=expected_previous_round_index,
    )
    output_dir = Path(str(payload["output_dir"])); table_path = Path(str(payload["task_table"]))
    dispatch = classify_exact_afterany_target_read_only(output_dir, target)
    if dispatch.kind not in {"sealed-generation", "no-generation"}:
        if dispatch.exit_code == SUPERSEDED_EXIT_CODE:
            raise ControlSupersededError("verify target was superseded")
        if dispatch.exit_code == PROTOCOL_EXIT_CODE:
            raise ControlProtocolError("verify target has protocol conflict")
        raise ControlBusyError("verify target is not sealed")
    frozen_frontier = frozen_sealed_history(output_dir, target, dispatch)
    history_digest = hashlib.sha256(canonical_json_bytes(list(frozen_frontier))).hexdigest()
    with _queue_index(payload, explicit_tmp_dir=tmp_dir, phase="verification") as connection:
        expected_model_keys = _index_queue_expected_design(connection, table_path)
        _index_queue_wals(
            connection, output_dir, analysis=analysis,
            frozen_frontier=frozen_frontier,
        )
        _index_completed_rows(connection); _classify_queue_attempts(connection)
        completed_model_keys = int(connection.execute("SELECT COUNT(*) FROM completed").fetchone()[0])
        aborted = _queue_row_ids(connection, "aborted_rows")
        result: dict[str, object] = {
            "expected_model_keys": expected_model_keys,
            "completed_model_keys": completed_model_keys,
            "missing_model_keys": _queue_missing_model_keys(connection),
            "failed_attempt_row_ids": _queue_row_ids(connection, "failed_attempt_rows"),
            "aborted_tasks": aborted,
            "interrupted_row_ids": _queue_row_ids(connection, "crashed_rows"),
            "too_long_row_ids": _queue_row_ids(connection, "too_long_rows"),
        }
    exit_code = PROTOCOL_EXIT_CODE if result["aborted_tasks"] else (VERIFY_INCOMPLETE_EXIT_CODE if result["missing_model_keys"] or result["interrupted_row_ids"] or result["too_long_row_ids"] else SUCCESS_EXIT_CODE)
    result.update({"complete": exit_code == SUCCESS_EXIT_CODE, "exit_code": exit_code, "sealed_history_digest_sha256": history_digest, "dispatch_kind": dispatch.kind})
    receipt = publish_verification_receipt(
        output_dir, target, dispatch=dispatch, sealed_history_digest_sha256=history_digest,
        complete=bool(result["complete"]), exit_code=exit_code, extra=result,
    )
    result["verification_receipt"] = str(receipt)
    return result


def _verification_failure_counts(result: Mapping[str, object]) -> dict[str, int]:
    return {
        "missing_model_keys": int(result["missing_model_keys"]),
        "interrupted_row_ids": len(result["interrupted_row_ids"]),
        "too_long_row_ids": len(result["too_long_row_ids"]),
        "aborted_tasks": len(result["aborted_tasks"]),
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
        writer = csv.DictWriter(handle, fieldnames=header or [], lineterminator="\n")
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


def _estimated_finalization_temp_bytes(
    table_path: Path, shards: Sequence[Path], *, sealed_wal_bytes: int = 0,
) -> int:
    """Conservatively budget the expanded SQLite keys and result payloads."""

    try:
        task_bytes = Path(table_path).stat().st_size
        shard_bytes = sum(path.stat().st_size for path in shards) + int(sealed_wal_bytes)
    except OSError as exc:
        raise FinalizationError(f"cannot size finalization inputs: {exc}") from exc
    return max(
        FINALIZATION_MIN_TEMP_BYTES,
        # Parquet keys are compressed; 12x reserves their expanded B-tree.
        # Three shard copies cover payload/index pages plus rollback journal.
        FINALIZATION_MIN_TEMP_BYTES + 12 * task_bytes + 3 * shard_bytes,
    )


def _preflight_finalization_space(
    tmp_base: Path, table_path: Path, shards: Sequence[Path], *, sealed_wal_bytes: int = 0,
) -> tuple[int, int]:
    estimated = _estimated_finalization_temp_bytes(table_path, shards, sealed_wal_bytes=sealed_wal_bytes)
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


def _frozen_wal_inventory_bytes(frozen_frontier: Sequence[Mapping[str, object]]) -> int:
    """Sum immutable inventory sizes before finalizer creates its first temp file."""

    total = 0
    for item in frozen_frontier:
        try:
            closed = json.loads(Path(str(item["closed_path"])).read_text(encoding="utf-8"))
            workers = closed["inventory"]["workers"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise FinalizationError("cannot read frozen WAL inventory for temp preflight") from exc
        if not isinstance(workers, list):
            raise FinalizationError("frozen WAL inventory workers are invalid")
        for worker in workers:
            if not isinstance(worker, Mapping):
                raise FinalizationError("frozen WAL inventory worker is invalid")
            if worker.get("wal_state") == "present":
                size = worker.get("size")
                if not isinstance(size, int) or size < 0:
                    raise FinalizationError("frozen WAL inventory size is invalid")
                total += size
    return total


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
            writer = csv.writer(destination, lineterminator="\n")
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


def _write_sealed_wal_csv(
    output: Path, *, history_root: Path, analysis: AnalysisContract,
    frozen_frontier: Sequence[Mapping[str, object]],
) -> tuple[Path, int]:
    """Materialize one local temporary CSV from sealed WAL RESULT payloads."""

    target = Path(output)
    header: tuple[str, ...] | None = None
    count = 0
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer: csv.DictWriter | None = None
        for _, _, scan in _iter_sealed_wal_scans(
            history_root, analysis=analysis, frozen_frontier=frozen_frontier,
        ) or ():
            for record in scan.records:
                if record.event_type != TASK_RESULT:
                    continue
                current_header, rows = decode_public_rows(record.payload)
                if header is None:
                    header = current_header
                    writer = csv.DictWriter(handle, fieldnames=list(header), lineterminator="\n")
                    writer.writeheader()
                elif current_header != header:
                    raise FinalizationError("sealed WAL RESULT payload headers differ")
                assert writer is not None
                writer.writerows(rows); count += len(rows)
    if header is None:
        raise FinalizationError("sealed history has no RESULT rows")
    return target, count


def finalize_snapshot(
    snapshot_path: Path, *, tmp_dir: Path | None = None,
    round_index: int | None = None, submission_generation: str | None = None,
    expected_prep_token: str | None = None, expected_previous_generation: str | None = None,
    expected_pointer_version: int | None = None, prep_job_id: str | None = None,
    expected_previous_execution_plan_id: str | None = None,
    expected_previous_round_index: int | None = None,
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
    legacy = "analysis_contract" not in snapshot
    if legacy:
        analysis = None; execution = None; target = None; dispatch = None
    else:
        if round_index is None or submission_generation is None or expected_prep_token is None:
            raise FinalizationError("finalizer requires an exact verification target")
        analysis, execution, target = _exact_target(
            snapshot, round_index=round_index, submission_generation=submission_generation,
            prep_token=expected_prep_token, expected_previous_generation=expected_previous_generation,
            expected_pointer_version=expected_pointer_version, prep_job_id=prep_job_id,
            expected_previous_execution_plan_id=expected_previous_execution_plan_id,
            expected_previous_round_index=expected_previous_round_index,
        )
        dispatch = classify_exact_afterany_target_read_only(output_dir, target)
        if dispatch.kind not in {"sealed-generation", "no-generation"}:
            raise FinalizationError("finalizer target is not exact sealed/no-generation dispatch")
        frozen_frontier = frozen_sealed_history(output_dir, target, dispatch)
        history_digest = hashlib.sha256(canonical_json_bytes(list(frozen_frontier))).hexdigest()
        validate_exact_verification_receipt(output_dir, target, expected_dispatch=dispatch, sealed_history_digest_sha256=history_digest)
    config = snapshot.get("config")
    if not isinstance(config, Mapping) or not isinstance(config.get("out"), str):
        raise FinalizationError("snapshot config.out is required for finalization")
    output = Path(str(config["out"])).expanduser().resolve()
    shards: tuple[Path, ...] = tuple(sorted(output_dir.glob("round-*/worker-*.csv"))) if legacy else ()
    tmp_base = _resolve_finalization_tmp_base(snapshot, tmp_dir)
    sealed_wal_bytes = 0 if legacy else _frozen_wal_inventory_bytes(frozen_frontier)
    available_bytes, estimated_bytes = _preflight_finalization_space(
        tmp_base, table_path, shards, sealed_wal_bytes=sealed_wal_bytes,
    )
    run_dir = Path(tempfile.mkdtemp(prefix="nk-grid-finalize-", dir=tmp_base))
    database = run_dir / "index.sqlite"
    connection: sqlite3.Connection | None = None
    try:
        if legacy:
            wal_rows = 0
        else:
            assert analysis is not None
            wal_csv, wal_rows = _write_sealed_wal_csv(
                run_dir / "sealed-results.csv", history_root=output_dir,
                analysis=analysis, frozen_frontier=frozen_frontier,
            )
            shards = (wal_csv,)
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
            "wal_result_rows": wal_rows,
            "frozen_wal_input_bytes": sealed_wal_bytes,
            "analysis_id": None if analysis is None else analysis.analysis_id,
            "execution_plan_ids": [] if execution is None else [execution.execution_plan_id],
            "verification_receipt": None if target is None else str(verification_path(output_dir, target)),
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
    analysis_contract: AnalysisContract | None = None,
    execution_contract: DynamicExecutionContract | None = None,
    task_summary: TaskTableSummary | None = None,
    cell_spec_repo_root: Path | None = None,
) -> Path:
    if workers < 1:
        raise ValueError("workers must be positive")
    table = Path(table_path).resolve()
    try:
        source = pq.ParquetFile(table, memory_map=True)
        _validate_task_table_columns(source.schema_arrow.names)
        if source.metadata.num_rows < 1:
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
    if (analysis_contract is None) != (execution_contract is None):
        raise ValueError("analysis and execution contracts must be supplied together")
    if analysis_contract is not None and execution_contract is not None:
        if execution_contract.payload.get("analysis_id") != analysis_contract.analysis_id:
            raise ValueError("execution contract does not bind analysis contract")
        payload["analysis_id"] = analysis_contract.analysis_id
        payload["analysis_contract_sha256"] = analysis_contract.sha256
        payload["analysis_contract"] = str((Path(output_dir) / "analysis-contract.json").resolve())
        payload["execution_plan_id"] = execution_contract.execution_plan_id
        payload["execution_contract_sha256"] = execution_contract.sha256
        payload["execution_contract"] = str((Path(output_dir) / "execution-contracts" / f"{execution_contract.execution_plan_id}.json").resolve())
        payload["cell_spec_repo_root"] = str((cell_spec_repo_root or Path(__file__).resolve().parents[2]).resolve())
        payload["result_store_format"] = "worker-event-wal-v1"
    else:
        # Direct unit tests retain a deliberately explicit CSV compatibility
        # fixture.  It cannot be mistaken for an old production WAL plan.
        payload["result_store_format"] = "test-legacy-csv-v2"
        payload["test_compatibility_mode"] = True
    if task_summary is not None:
        payload["task_table_file_sha256"] = task_summary.task_table_file_sha256
        payload["task_design_digest"] = task_summary.task_design_digest
        payload["expected_task_rows"] = task_summary.expected_task_rows
        payload["expected_model_rows"] = task_summary.expected_model_rows
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
    parser.add_argument("command", choices=("prep", "recover-activation", "run", "close", "verify", "finalize"))
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--round", type=int)
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--tmp-dir", type=Path)
    parser.add_argument("--prep-token")
    parser.add_argument("--expected-prep-token")
    parser.add_argument("--generation")
    parser.add_argument("--expected-previous-generation")
    parser.add_argument("--expected-previous-execution-plan-id")
    parser.add_argument("--expected-previous-round", type=int)
    parser.add_argument("--expected-pointer-version", type=int)
    parser.add_argument("--prep-job-id")
    args = parser.parse_args(argv)
    try:
        if args.command == "prep":
            if args.round is None or args.prep_token is None or args.generation is None:
                parser.error("prep requires --round, --prep-token, and --generation")
            print(json.dumps(prepare_round(
                args.snapshot, round_index=args.round, prep_token=args.prep_token,
                tmp_dir=args.tmp_dir, submission_generation=args.generation,
                expected_previous_generation=args.expected_previous_generation,
                expected_pointer_version=args.expected_pointer_version, prep_job_id=args.prep_job_id,
                expected_previous_execution_plan_id=args.expected_previous_execution_plan_id,
                expected_previous_round_index=args.expected_previous_round,
            ), sort_keys=True))
        elif args.command == "recover-activation":
            if args.round is None or args.expected_prep_token is None or args.generation is None:
                parser.error("recover-activation requires --round, --expected-prep-token, and --generation")
            print(recover_generation_activation(
                args.snapshot, round_index=args.round, submission_generation=args.generation,
                expected_prep_token=args.expected_prep_token,
                expected_previous_generation=args.expected_previous_generation,
                expected_pointer_version=args.expected_pointer_version, prep_job_id=args.prep_job_id,
                expected_previous_execution_plan_id=args.expected_previous_execution_plan_id,
                expected_previous_round_index=args.expected_previous_round,
            ))
        elif args.command == "run":
            if args.round is None or args.worker_index is None or args.expected_prep_token is None or args.generation is None:
                parser.error("run requires --round, --worker-index, --expected-prep-token, and --generation")
            print(run_slice(
                args.snapshot, round_index=args.round, worker_index=args.worker_index,
                expected_prep_token=args.expected_prep_token, submission_generation=args.generation,
                expected_previous_generation=args.expected_previous_generation,
                expected_pointer_version=args.expected_pointer_version, prep_job_id=args.prep_job_id,
                expected_previous_execution_plan_id=args.expected_previous_execution_plan_id,
                expected_previous_round_index=args.expected_previous_round,
            ))
        elif args.command == "close":
            if args.round is None or args.expected_prep_token is None or args.generation is None:
                parser.error("close requires --round, --expected-prep-token, and --generation")
            print(close_generation(
                args.snapshot, round_index=args.round, submission_generation=args.generation,
                expected_prep_token=args.expected_prep_token,
                expected_previous_generation=args.expected_previous_generation,
                expected_pointer_version=args.expected_pointer_version, prep_job_id=args.prep_job_id,
                expected_previous_execution_plan_id=args.expected_previous_execution_plan_id,
                expected_previous_round_index=args.expected_previous_round,
            ))
        elif args.command == "verify":
            legacy = "analysis_contract" not in _load_snapshot(args.snapshot)
            if not legacy and (args.round is None or args.expected_prep_token is None or args.generation is None):
                parser.error("verify requires --round, --expected-prep-token, and --generation")
            result = verify_rounds(
                args.snapshot, tmp_dir=args.tmp_dir, round_index=args.round,
                submission_generation=args.generation, expected_prep_token=args.expected_prep_token,
                expected_previous_generation=args.expected_previous_generation,
                expected_pointer_version=args.expected_pointer_version, prep_job_id=args.prep_job_id,
                expected_previous_execution_plan_id=args.expected_previous_execution_plan_id,
                expected_previous_round_index=args.expected_previous_round,
            )
            print(json.dumps(result, sort_keys=True), flush=True)
            if legacy:
                if any(_verification_failure_counts({
                    "missing_model_keys": result["missing_model_keys"],
                    "interrupted_row_ids": result["crashed_row_ids"],
                    "too_long_row_ids": result["too_long_row_ids"],
                    "aborted_tasks": [],
                }).values()):
                    print(
                        "verification incomplete: "
                        + "; ".join((
                            f"missing_model_keys={result['missing_model_keys']}",
                            f"crashed_row_ids={len(result['crashed_row_ids'])}",
                            f"too_long_row_ids={len(result['too_long_row_ids'])}",
                        )),
                        file=sys.stderr,
                        flush=True,
                    )
                    raise SystemExit(VERIFY_INCOMPLETE_EXIT_CODE)
            elif result["exit_code"] != SUCCESS_EXIT_CODE:
                raise SystemExit(int(result["exit_code"]))
        else:
            legacy = "analysis_contract" not in _load_snapshot(args.snapshot)
            if not legacy and (args.round is None or args.expected_prep_token is None or args.generation is None):
                parser.error("finalize requires exact --round, --expected-prep-token, and --generation")
            print(json.dumps(finalize_snapshot(
                args.snapshot, tmp_dir=args.tmp_dir, round_index=args.round,
                submission_generation=args.generation, expected_prep_token=args.expected_prep_token,
                expected_previous_generation=args.expected_previous_generation,
                expected_pointer_version=args.expected_pointer_version, prep_job_id=args.prep_job_id,
                expected_previous_execution_plan_id=args.expected_previous_execution_plan_id,
                expected_previous_round_index=args.expected_previous_round,
            ), sort_keys=True))
    except json.JSONDecodeError:
        # Preserve the public CLI's long-standing malformed-user-input
        # behaviour; committed protocol artefacts are normalized below.
        raise
    except (ControlProtocolError, WALProtocolError, FinalizationError, ContractError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(PROTOCOL_EXIT_CODE) from exc
    except (ControlBusyError, WALBusyError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(RETRYABLE_EXIT_CODE) from exc
    except ControlSupersededError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(SUPERSEDED_EXIT_CODE) from exc


if __name__ == "__main__":
    main()
