"""Experiment identity and safe CSV checkpoint helpers."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import sqlite3
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import fcntl
import numpy as np
import pandas as pd


MODEL_ENV_KEYS = (
    "BART_N_BURN",
    "BART_N_SAMPLES",
    "BART_N_TREES",
    "BART_THIN",
    "LGBM_MAX_ROUNDS",
    "RF_MAX_FEATURES",
    "RF_MIN_SAMPLES_LEAF",
    "RF_N_ESTIMATORS",
    "XGB_MAX_ROUNDS",
)

CHECKPOINT_KEY_COLUMNS = ["model", "seed", "draw", "N", "K"]
CHECKPOINT_SORT_COLUMNS = ["model", "seed", "draw", "N", "K"]
CHECKPOINT_RESUME_COLUMNS = [
    "experiment_id",
    *CHECKPOINT_KEY_COLUMNS,
    "status",
]
# Each execution batch remains its own small WAL shard.  A single writer for a
# given output path periodically replaces groups of loose shards with one
# compact shard; readers may see both copies during publication and rely on the
# normal checkpoint-key deduplication below.
CHECKPOINT_COMPACTION_LOOSE_PARTS = 50
CHECKPOINT_SQLITE_INSERT_ROWS = 2_000
MANIFEST_SCHEMA_VERSION = "1"
EXPERIMENT_IDENTITY_VERSION = 3
COMPLETED_CHECKPOINT_STATUSES = frozenset({"ok", "skipped"})
SERIAL_OUTER_MODELS = frozenset({"lightgbm", "super_learner"})


@dataclass(frozen=True)
class CheckpointSummary:
    """Low-memory run summary produced by the on-disk checkpoint reducer."""

    experiment_id: str
    materialized_rows: int
    ok_rows: int
    skipped_rows: int
    failed_rows: int
    diagnostics: dict[str, Any]

    @property
    def completed_rows(self) -> int:
        return self.ok_rows + self.skipped_rows


@dataclass(frozen=True)
class CheckpointMaterialization:
    """Result of atomically materializing checkpoint shards."""

    path: Path
    source_parts: int
    columns: tuple[str, ...]
    summary: CheckpointSummary
    backend: str = "sqlite_streaming"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(directory: Path) -> None:
    """Persist directory-entry changes before deleting an older authority."""

    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdir_durable(directory: Path) -> None:
    """Create a directory tree and durably publish every new directory entry."""

    missing: list[Path] = []
    cursor = directory
    while not cursor.exists():
        missing.append(cursor)
        parent = cursor.parent
        if parent == cursor:
            break
        cursor = parent
    directory.mkdir(parents=True, exist_ok=True)
    # ``mkdir(parents=True)`` may create more than the leaf. Persist both each
    # new directory and the directory entry that names it in its parent before
    # a compacted shard is allowed to supersede older sources.
    for created in reversed(missing):
        _fsync_directory(created)
        _fsync_directory(created.parent)


def _write_dataframe_atomic(frame: pd.DataFrame, target: Path) -> Path:
    """Write, fsync and atomically publish one CSV file."""

    _mkdir_durable(target.parent)
    token = uuid.uuid4().hex
    tmp = target.parent / f".{target.name}.{token}.tmp"
    while True:
        try:
            descriptor = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
            break
        except FileExistsError:
            token = uuid.uuid4().hex
            tmp = target.parent / f".{target.name}.{token}.tmp"
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
        _fsync_directory(target.parent)
        return target
    finally:
        tmp.unlink(missing_ok=True)


@contextmanager
def _atomic_text_writer(target: Path) -> Iterator[Any]:
    """Yield a text handle and atomically publish it only after fsync."""

    _mkdir_durable(target.parent)
    token = uuid.uuid4().hex
    tmp = target.parent / f".{target.name}.{token}.tmp"
    descriptor = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    published = False
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
        _fsync_directory(target.parent)
        published = True
    finally:
        if not published:
            tmp.unlink(missing_ok=True)


@contextmanager
def output_run_lock(out_path: Path) -> Iterator[Path]:
    """Fail fast when another process is already writing this output.

    Keep persistent lock inodes in a hidden directory so output folders remain
    uncluttered without introducing the race caused by deleting flock files.
    """

    lock_path = out_path.parent / ".locks" / f"{out_path.name}.run.lock"
    _mkdir_durable(lock_path.parent)
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o666)
    acquired = False
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Another NK Grid worker already holds the output lease: {lock_path}"
            ) from exc
        yield lock_path
    finally:
        try:
            if acquired:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def build_experiment_metadata(
    *,
    kind: str,
    data_path: Path,
    outcome: str,
    test_size: float,
    split_seed: int,
    algorithm_version: str = "legacy",
    extra: dict[str, Any] | None = None,
    split_mode: str = "internal_random",
    test_data_path: Path | None = None,
    experiment_identity_version: int = EXPERIMENT_IDENTITY_VERSION,
) -> dict[str, Any]:
    """Return stable metadata that distinguishes result-producing inputs."""

    settings = {
        "experiment_identity_version": int(experiment_identity_version),
        "kind": kind,
        "algorithm_version": algorithm_version,
        "data_sha256": file_sha256(data_path),
        "test_data_sha256": (
            file_sha256(test_data_path) if test_data_path is not None else ""
        ),
        "outcome": outcome,
        "split_mode": split_mode,
        "split_seed": int(split_seed),
        "extra": extra or {},
    }
    if split_mode == "internal_random":
        settings["test_size"] = float(test_size)
    elif split_mode != "external_test":
        raise ValueError(f"Unknown split_mode: {split_mode!r}")
    encoded = json.dumps(settings, sort_keys=True, separators=(",", ":"))
    experiment_id = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]
    return {
        "experiment_id": experiment_id,
        "experiment_identity_version": int(experiment_identity_version),
        "experiment_kind": kind,
        "algorithm_version": algorithm_version,
        "data_sha256": settings["data_sha256"],
        "test_data_sha256": settings["test_data_sha256"],
        "data_path": str(data_path.resolve()),
        "outcome": outcome,
        "test_size": float(test_size) if split_mode == "internal_random" else None,
        "split_mode": split_mode,
        "split_seed": int(split_seed),
    }


def _validate_checkpoint_source(
    frame: pd.DataFrame,
    source: Path,
    *,
    allow_empty: bool,
) -> None:
    """Fail fast on a malformed checkpoint source before concatenation."""

    required = {"experiment_id", *CHECKPOINT_KEY_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"Malformed checkpoint source {source}: missing required columns "
            f"{missing}"
        )
    if frame.empty:
        if not allow_empty:
            raise ValueError(f"Malformed checkpoint source {source}: shard is empty")
        return
    for column in ("experiment_id", "model"):
        values = frame[column]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise ValueError(
                f"Malformed checkpoint source {source}: {column} contains "
                "missing or empty values"
            )
    for column in ("seed", "draw", "N", "K"):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        valid = (
            numeric.notna()
            & np.isfinite(numeric.to_numpy(dtype=float))
            & numeric.mod(1).eq(0)
        )
        if not valid.all():
            raise ValueError(
                f"Malformed checkpoint source {source}: {column} must contain "
                "finite integers"
            )
    if "status" in frame:
        allowed = {"ok", "skipped", "failed"}
        status = frame["status"]
        if status.isna().any() or not set(status.astype(str)).issubset(allowed):
            raise ValueError(
                f"Malformed checkpoint source {source}: status must contain "
                "only ok, skipped or failed"
            )
    else:
        # Pre-status checkpoints treated every persisted row as completed.
        # Normalize here so a legacy CSV can be migrated into WAL shards and
        # safely coexist with status-bearing retries.
        frame["status"] = "ok"


def _read_checkpoint_source(
    source: Path,
    *,
    resume_projection: bool,
    allow_empty: bool,
) -> pd.DataFrame:
    read_csv_kwargs: dict[str, Any] = {}
    if resume_projection:
        wanted = frozenset(CHECKPOINT_RESUME_COLUMNS)
        # A callable keeps legacy checkpoints without ``status`` readable while
        # asking pandas not to parse the dozens of metric/diagnostic columns.
        read_csv_kwargs["usecols"] = lambda column: column in wanted
    try:
        frame = pd.read_csv(source, **read_csv_kwargs)
    except (
        OSError,
        UnicodeError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
    ) as exc:
        raise ValueError(f"Malformed checkpoint source {source}: {exc}") from exc
    _validate_checkpoint_source(frame, source, allow_empty=allow_empty)
    return frame


def _load_checkpoint(
    path: Path,
    *,
    resume_projection: bool,
) -> pd.DataFrame:
    """Load checkpoint rows, optionally projecting to resume-only columns."""

    parts = checkpoint_parts(path)
    if parts:
        frames = [
            _read_checkpoint_source(
                part,
                resume_projection=resume_projection,
                allow_empty=False,
            )
            for part in parts
        ]
        frame = pd.concat(
            frames,
            ignore_index=True,
            sort=False,
        )
        frame = _prefer_completed_checkpoint_rows(
            frame, ["experiment_id", *CHECKPOINT_KEY_COLUMNS]
        )
        return frame.sort_values(
            ["experiment_id", *CHECKPOINT_SORT_COLUMNS]
        ).reset_index(drop=True)
    if not path.exists():
        return pd.DataFrame()
    return _read_checkpoint_source(
        path,
        resume_projection=resume_projection,
        allow_empty=True,
    )


def load_checkpoint(path: Path) -> pd.DataFrame:
    """Load all authoritative checkpoint columns for merge/materialization."""

    return _load_checkpoint(path, resume_projection=False)


def load_checkpoint_index(path: Path) -> pd.DataFrame:
    """Load only identity, cell-key and status columns needed for resume."""

    return _load_checkpoint(path, resume_projection=True)


def _prefer_completed_checkpoint_rows(
    frame: pd.DataFrame, key_columns: list[str]
) -> pd.DataFrame:
    """Deduplicate cells without allowing a failed retry to hide a success."""

    if frame.empty:
        return frame
    if "status" not in frame:
        return frame.drop_duplicates(key_columns, keep="last")
    priority_column = "_checkpoint_status_priority"
    prioritized = frame.assign(
        **{
            priority_column: frame["status"]
            .isin(COMPLETED_CHECKPOINT_STATUSES)
            .astype("int8")
        }
    ).sort_values(priority_column, kind="stable")
    return prioritized.drop_duplicates(key_columns, keep="last").drop(
        columns=priority_column
    )


def rows_for_experiment(frame: pd.DataFrame, experiment_id: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame[frame["experiment_id"].eq(experiment_id)]


def add_metadata(row: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    return {**metadata, **row}


def write_checkpoint(
    existing: pd.DataFrame,
    rows: Iterable[dict[str, Any]],
    out_path: Path,
    *,
    key_columns: list[str],
    sort_columns: list[str],
) -> None:
    """Atomically merge rows without deduplicating across experiments."""

    new_rows = pd.DataFrame(list(rows))
    frames = [frame for frame in (existing, new_rows) if not frame.empty]
    if not frames:
        return
    result = pd.concat(frames, ignore_index=True)
    result = _prefer_completed_checkpoint_rows(
        result, ["experiment_id", *key_columns]
    )
    _write_dataframe_atomic(
        result.sort_values(["experiment_id", *sort_columns]),
        out_path,
    )


def checkpoint_parts_dir(out_path: Path) -> Path:
    """Return ``result.parts`` for ``result.csv``."""

    return out_path.with_suffix(".parts")


def checkpoint_loose_parts_dir(out_path: Path) -> Path:
    return checkpoint_parts_dir(out_path) / "loose"


def checkpoint_compact_parts_dir(out_path: Path) -> Path:
    return checkpoint_parts_dir(out_path) / "compact"


def _checkpoint_part_sort_key(path: Path) -> tuple[int, int | str, str]:
    token = path.name.removeprefix("part-").split("-", 1)[0]
    if token.isdigit():
        return (1, int(token), path.name)
    return (0, path.stat().st_mtime_ns, path.name)


def checkpoint_parts(out_path: Path) -> list[Path]:
    directory = checkpoint_parts_dir(out_path)
    if not directory.exists():
        return []
    parts = [
        *directory.glob("part-*.csv"),
        *checkpoint_loose_parts_dir(out_path).glob("part-*.csv"),
        *checkpoint_compact_parts_dir(out_path).glob("part-*.csv"),
    ]
    return sorted(parts, key=_checkpoint_part_sort_key)


def _new_checkpoint_path(directory: Path, *, compact: bool) -> Path:
    kind = "-compact" if compact else ""
    return (
        directory
        / f"part-{time.time_ns():020d}{kind}-{uuid.uuid4().hex}.csv"
    )


def _write_checkpoint_frame_atomic(
    frame: pd.DataFrame,
    directory: Path,
    *,
    compact: bool,
) -> Path:
    """Publish a complete CSV shard with an atomic same-directory rename."""

    return _write_dataframe_atomic(
        frame,
        _new_checkpoint_path(directory, compact=compact),
    )


def _loose_checkpoint_parts(out_path: Path) -> list[Path]:
    root = checkpoint_parts_dir(out_path)
    parts = [
        part
        for part in root.glob("part-*.csv")
        if "-compact-" not in part.name
    ]
    parts.extend(checkpoint_loose_parts_dir(out_path).glob("part-*.csv"))
    return sorted(parts, key=_checkpoint_part_sort_key)


def _remove_compacted_sources(sources: Iterable[Path]) -> None:
    for source in sources:
        source.unlink(missing_ok=True)


def compact_checkpoint_parts(
    out_path: Path,
    *,
    loose_part_threshold: int = CHECKPOINT_COMPACTION_LOOSE_PARTS,
) -> Path | None:
    """Atomically compact one group of loose WAL shards.

    This operation assumes exactly one writer/compactor for a given
    ``out_path``.  It first publishes the compact shard with an atomic rename
    and only then removes the exact loose source files.  A process interruption
    before publication leaves all sources intact; interruption during cleanup
    leaves the published shard plus some duplicate sources.  ``load_checkpoint``
    and ``load_checkpoint_index`` safely deduplicate that latter state.
    """

    if loose_part_threshold < 2:
        raise ValueError("loose_part_threshold must be at least 2")
    loose_parts = _loose_checkpoint_parts(out_path)
    if len(loose_parts) < loose_part_threshold:
        return None
    sources = loose_parts[:loose_part_threshold]
    frames = [
        _read_checkpoint_source(
            source,
            resume_projection=False,
            allow_empty=False,
        )
        for source in sources
    ]
    compacted = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )
    compact_part = _write_checkpoint_frame_atomic(
        compacted,
        checkpoint_compact_parts_dir(out_path),
        compact=True,
    )
    _remove_compacted_sources(sources)
    return compact_part


def write_checkpoint_part(
    rows: Iterable[dict[str, Any]],
    out_path: Path,
) -> Path | None:
    """Atomically append one WAL shard and periodically compact loose shards."""

    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return None
    part = _write_checkpoint_frame_atomic(
        frame,
        checkpoint_loose_parts_dir(out_path),
        compact=False,
    )
    compacted = compact_checkpoint_parts(out_path)
    return compacted or part


def seed_checkpoint_parts_from_csv(out_path: Path) -> Path | None:
    """Stream-copy a legacy/final CSV into the authoritative shard layout."""

    if checkpoint_parts(out_path):
        return None
    if not out_path.exists():
        return None
    _checkpoint_header(out_path)
    target = _new_checkpoint_path(
        checkpoint_compact_parts_dir(out_path),
        compact=True,
    )
    with (
        out_path.open(encoding="utf-8", newline="") as source,
        _atomic_text_writer(target) as destination,
    ):
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            destination.write(chunk)
    return target


def _sql_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _checkpoint_sources(out_path: Path) -> list[Path]:
    parts = checkpoint_parts(out_path)
    if parts:
        return parts
    return [out_path] if out_path.exists() else []


def _checkpoint_header(source: Path) -> list[str]:
    try:
        with source.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
    except StopIteration as exc:
        raise ValueError(
            f"Malformed checkpoint source {source}: file is empty"
        ) from exc
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"Malformed checkpoint source {source}: {exc}") from exc
    if not header or any(not column for column in header):
        raise ValueError(
            f"Malformed checkpoint source {source}: header contains an empty column"
        )
    duplicates = sorted(
        {column for column in header if header.count(column) > 1}
    )
    if duplicates:
        raise ValueError(
            f"Malformed checkpoint source {source}: duplicate columns {duplicates}"
        )
    required = {"experiment_id", *CHECKPOINT_KEY_COLUMNS}
    missing = sorted(required - set(header))
    if missing:
        raise ValueError(
            f"Malformed checkpoint source {source}: missing required columns "
            f"{missing}"
        )
    return header


def _checkpoint_union_columns(sources: Iterable[Path]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for source in sources:
        for column in _checkpoint_header(source):
            if column not in seen:
                columns.append(column)
                seen.add(column)
    if "status" not in seen:
        columns.append("status")
    return columns


def _checkpoint_integer(
    value: Any,
    *,
    source: Path,
    column: str,
) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Malformed checkpoint source {source}: {column} must contain "
            "finite integers"
        ) from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(
            f"Malformed checkpoint source {source}: {column} must contain "
            "finite integers"
        )
    return int(numeric)


def _iter_checkpoint_rows(
    source: Path,
    *,
    allow_empty: bool,
) -> Iterator[dict[str, Any]]:
    """Stream and validate one CSV checkpoint without creating a DataFrame."""

    header = _checkpoint_header(source)
    has_status = "status" in header
    row_count = 0
    try:
        with source.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for raw in reader:
                extra = raw.get(None)
                if extra:
                    raise ValueError(
                        f"Malformed checkpoint source {source}: row contains "
                        "more fields than the header"
                    )
                for column in ("experiment_id", "model"):
                    value = raw.get(column)
                    if value is None or not str(value).strip():
                        raise ValueError(
                            f"Malformed checkpoint source {source}: {column} "
                            "contains missing or empty values"
                        )
                for column in ("seed", "draw", "N", "K"):
                    raw[column] = _checkpoint_integer(
                        raw.get(column),
                        source=source,
                        column=column,
                    )
                if has_status:
                    status = str(raw.get("status", "")).strip()
                    if status not in {"ok", "skipped", "failed"}:
                        raise ValueError(
                            f"Malformed checkpoint source {source}: status must "
                            "contain only ok, skipped or failed"
                        )
                    raw["status"] = status
                else:
                    raw["status"] = "ok"
                row_count += 1
                yield raw
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"Malformed checkpoint source {source}: {exc}") from exc
    if row_count == 0 and not allow_empty:
        raise ValueError(f"Malformed checkpoint source {source}: shard is empty")


def _sqlite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _sqlite_truth(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (bool, int, float)):
        return int(bool(value))
    return int(str(value).strip().lower() in {"1", "1.0", "true", "yes"})


def _checkpoint_sqlite_path(out_path: Path) -> Path:
    return out_path.parent / (
        f".{out_path.name}.materialize-{uuid.uuid4().hex}.sqlite"
    )


def _create_checkpoint_table(
    connection: sqlite3.Connection,
    columns: list[str],
) -> None:
    internal = "__nk_status_priority"
    if internal in columns:
        raise ValueError(f"Checkpoint column name is reserved: {internal}")
    definitions = [
        (
            f"{_sql_identifier(column)} INTEGER NOT NULL"
            if column in {"seed", "draw", "N", "K"}
            else f"{_sql_identifier(column)} TEXT"
        )
        for column in columns
    ]
    definitions.append(f"{_sql_identifier(internal)} INTEGER NOT NULL")
    primary_key = ", ".join(
        _sql_identifier(column)
        for column in ("experiment_id", *CHECKPOINT_KEY_COLUMNS)
    )
    connection.execute(
        "CREATE TABLE checkpoint_rows ("
        + ", ".join(definitions)
        + f", PRIMARY KEY ({primary_key})) WITHOUT ROWID"
    )


def _checkpoint_insert_sql(columns: list[str]) -> str:
    internal = "__nk_status_priority"
    insert_columns = [*columns, internal]
    names = ", ".join(_sql_identifier(column) for column in insert_columns)
    placeholders = ", ".join("?" for _ in insert_columns)
    key_columns = {"experiment_id", *CHECKPOINT_KEY_COLUMNS}
    assignments = ", ".join(
        f"{_sql_identifier(column)} = excluded.{_sql_identifier(column)}"
        for column in insert_columns
        if column not in key_columns
    )
    conflict_columns = ", ".join(
        _sql_identifier(column)
        for column in ("experiment_id", *CHECKPOINT_KEY_COLUMNS)
    )
    priority = _sql_identifier(internal)
    return (
        f"INSERT INTO checkpoint_rows ({names}) VALUES ({placeholders}) "
        f"ON CONFLICT ({conflict_columns}) "
        f"DO UPDATE SET {assignments} "
        f"WHERE excluded.{priority} >= checkpoint_rows.{priority}"
    )


def _ingest_checkpoint_sources(
    connection: sqlite3.Connection,
    sources: list[Path],
    columns: list[str],
    *,
    out_path: Path,
) -> None:
    insert_sql = _checkpoint_insert_sql(columns)
    pending_rows: list[tuple[Any, ...]] = []
    for source in sources:
        allow_empty = len(sources) == 1 and source == out_path
        for row in _iter_checkpoint_rows(source, allow_empty=allow_empty):
            status = str(row["status"])
            pending_rows.append(
                (
                    *(row.get(column) for column in columns),
                    int(status in COMPLETED_CHECKPOINT_STATUSES),
                )
            )
            if len(pending_rows) >= CHECKPOINT_SQLITE_INSERT_ROWS:
                with connection:
                    connection.executemany(insert_sql, pending_rows)
                pending_rows.clear()
    if pending_rows:
        with connection:
            connection.executemany(insert_sql, pending_rows)


def _numeric_column_summary(
    connection: sqlite3.Connection,
    *,
    column: str,
    experiment_id: str,
    model: str,
) -> tuple[int, float | None, float | None, float | None, float | None]:
    identifier = _sql_identifier(column)
    expression = f"nk_float({identifier})"
    where = (
        "experiment_id = ? AND model = ? "
        f"AND {expression} IS NOT NULL"
    )
    count, total, minimum, maximum = connection.execute(
        f"SELECT COUNT(*), SUM({expression}), MIN({expression}), "
        f"MAX({expression}) FROM checkpoint_rows WHERE {where}",
        (experiment_id, model),
    ).fetchone()
    count = int(count)
    median: float | None = None
    if count:
        middle = connection.execute(
            f"SELECT {expression} FROM checkpoint_rows WHERE {where} "
            f"ORDER BY {expression} LIMIT 2 OFFSET ?",
            (experiment_id, model, (count - 1) // 2),
        ).fetchall()
        values = [float(row[0]) for row in middle]
        median = values[0] if count % 2 else sum(values[:2]) / 2.0
    return (
        count,
        float(total) if total is not None else None,
        median,
        float(minimum) if minimum is not None else None,
        float(maximum) if maximum is not None else None,
    )


def _sqlite_diagnostics_summary(
    connection: sqlite3.Connection,
    *,
    experiment_id: str,
    columns: set[str],
) -> dict[str, Any]:
    def boolean_count(column: str, *, model: str | None = None) -> int:
        if column not in columns:
            return 0
        parameters: list[Any] = [experiment_id]
        where = "experiment_id = ? AND status = 'ok'"
        if model is not None:
            where += " AND model = ?"
            parameters.append(model)
        value = connection.execute(
            f"SELECT COALESCE(SUM(nk_truth({_sql_identifier(column)})), 0) "
            f"FROM checkpoint_rows WHERE {where}",
            parameters,
        ).fetchone()[0]
        return int(value)

    def nonconverged_count(*, model: str | None = None) -> int:
        if "converged" not in columns:
            return 0
        parameters: list[Any] = [experiment_id]
        where = "experiment_id = ? AND status = 'ok'"
        if model is not None:
            where += " AND model = ?"
            parameters.append(model)
        value = connection.execute(
            "SELECT COALESCE(SUM(CASE WHEN "
            f"nk_truth({_sql_identifier('converged')}) = 1 THEN 0 ELSE 1 END), 0) "
            f"FROM checkpoint_rows WHERE {where}",
            parameters,
        ).fetchone()[0]
        return int(value)

    result: dict[str, Any] = {
        "constant_prediction_rows": boolean_count("constant_prediction"),
        "underdetermined_rows": boolean_count("underdetermined"),
        "nonconverged_rows": nonconverged_count(),
    }
    models = [
        str(row[0])
        for row in connection.execute(
            "SELECT DISTINCT model FROM checkpoint_rows "
            "WHERE experiment_id = ? ORDER BY model",
            (experiment_id,),
        )
    ]
    by_model: dict[str, Any] = {}
    for model in models:
        ok_rows = int(
            connection.execute(
                "SELECT COUNT(*) FROM checkpoint_rows "
                "WHERE experiment_id = ? AND model = ? AND status = 'ok'",
                (experiment_id, model),
            ).fetchone()[0]
        )
        model_summary: dict[str, Any] = {
            "rows": ok_rows,
            "constant_prediction_rows": boolean_count(
                "constant_prediction", model=model
            ),
            "underdetermined_rows": boolean_count(
                "underdetermined", model=model
            ),
            "nonconverged_rows": nonconverged_count(model=model),
        }
        if "_fit_seconds" in columns:
            count, total, median, _, _ = _numeric_column_summary(
                connection,
                column="_fit_seconds",
                experiment_id=experiment_id,
                model=model,
            )
            model_summary["fit_seconds_total"] = total if count else 0.0
            model_summary["fit_seconds_median"] = median
        if "_best_rounds" in columns:
            count, _, median, minimum, maximum = _numeric_column_summary(
                connection,
                column="_best_rounds",
                experiment_id=experiment_id,
                model=model,
            )
            model_summary["best_rounds"] = (
                {
                    "min": int(minimum),
                    "median": median,
                    "max": int(maximum),
                }
                if count
                else None
            )
        by_model[model] = model_summary
    result["by_model"] = by_model
    return result


def _checkpoint_summary(
    connection: sqlite3.Connection,
    *,
    experiment_id: str,
    columns: set[str],
) -> CheckpointSummary:
    status_counts = {
        str(status): int(count)
        for status, count in connection.execute(
            "SELECT status, COUNT(*) FROM checkpoint_rows "
            "WHERE experiment_id = ? GROUP BY status",
            (experiment_id,),
        )
    }
    return CheckpointSummary(
        experiment_id=experiment_id,
        materialized_rows=sum(status_counts.values()),
        ok_rows=status_counts.get("ok", 0),
        skipped_rows=status_counts.get("skipped", 0),
        failed_rows=status_counts.get("failed", 0),
        diagnostics=_sqlite_diagnostics_summary(
            connection,
            experiment_id=experiment_id,
            columns=columns,
        ),
    )


def _write_materialized_checkpoint(
    connection: sqlite3.Connection,
    *,
    out_path: Path,
    columns: list[str],
    drop_output_columns: Iterable[str],
) -> None:
    dropped = set(drop_output_columns)
    output_columns = [column for column in columns if column not in dropped]
    selected = ", ".join(_sql_identifier(column) for column in output_columns)
    ordered = ", ".join(
        _sql_identifier(column)
        for column in ("experiment_id", *CHECKPOINT_SORT_COLUMNS)
    )
    cursor = connection.execute(
        f"SELECT {selected} FROM checkpoint_rows ORDER BY {ordered}"
    )
    with _atomic_text_writer(out_path) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(output_columns)
        while True:
            rows = cursor.fetchmany(10_000)
            if not rows:
                break
            writer.writerows(rows)


def merge_checkpoint_parts(
    out_path: Path,
    *,
    experiment_id: str | None = None,
    key_columns: list[str] | None = None,
    sort_columns: list[str] | None = None,
    drop_output_columns: list[str] | None = None,
) -> CheckpointMaterialization:
    """Materialize shards with bounded RAM using a temporary SQLite reducer."""

    if (key_columns or CHECKPOINT_KEY_COLUMNS) != CHECKPOINT_KEY_COLUMNS:
        raise ValueError("SQLite materialization requires the canonical checkpoint keys")
    if (sort_columns or CHECKPOINT_SORT_COLUMNS) != CHECKPOINT_SORT_COLUMNS:
        raise ValueError("SQLite materialization requires the canonical checkpoint order")
    sources = _checkpoint_sources(out_path)
    if not sources:
        if experiment_id is None:
            raise ValueError("experiment_id is required for an empty checkpoint")
        return CheckpointMaterialization(
            path=out_path,
            source_parts=0,
            columns=(),
            summary=CheckpointSummary(
                experiment_id=experiment_id,
                materialized_rows=0,
                ok_rows=0,
                skipped_rows=0,
                failed_rows=0,
                diagnostics={
                    "constant_prediction_rows": 0,
                    "underdetermined_rows": 0,
                    "nonconverged_rows": 0,
                    "by_model": {},
                },
            ),
        )
    columns = _checkpoint_union_columns(sources)
    sqlite_path = _checkpoint_sqlite_path(out_path)
    connection = sqlite3.connect(sqlite_path)
    connection.create_function("nk_float", 1, _sqlite_float, deterministic=True)
    connection.create_function("nk_truth", 1, _sqlite_truth, deterministic=True)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA cache_size=-262144")
        connection.execute("PRAGMA locking_mode=EXCLUSIVE")
        _create_checkpoint_table(connection, columns)
        _ingest_checkpoint_sources(
            connection,
            sources,
            columns,
            out_path=out_path,
        )
        if experiment_id is None:
            experiment_ids = [
                str(row[0])
                for row in connection.execute(
                    "SELECT DISTINCT experiment_id FROM checkpoint_rows"
                )
            ]
            if len(experiment_ids) != 1:
                raise ValueError(
                    "experiment_id must be supplied when a checkpoint contains "
                    f"{len(experiment_ids)} experiment identities"
                )
            experiment_id = experiment_ids[0]
        summary = _checkpoint_summary(
            connection,
            experiment_id=experiment_id,
            columns=set(columns),
        )
        _write_materialized_checkpoint(
            connection,
            out_path=out_path,
            columns=columns,
            drop_output_columns=drop_output_columns or [],
        )
        return CheckpointMaterialization(
            path=out_path,
            source_parts=len(sources),
            columns=tuple(
                column
                for column in columns
                if column not in set(drop_output_columns or [])
            ),
            summary=summary,
        )
    finally:
        connection.close()
        sqlite_path.unlink(missing_ok=True)


def verify_materialized_checkpoint(
    path: Path,
    *,
    experiment_id: str,
    expected_rows: int,
) -> None:
    """Stream-verify row count, status, uniqueness and sort order."""

    count = 0
    previous_key: tuple[str, int, int, int, int] | None = None
    for row in _iter_checkpoint_rows(path, allow_empty=True):
        if str(row["experiment_id"]) != experiment_id:
            continue
        status = str(row["status"])
        if status not in COMPLETED_CHECKPOINT_STATUSES:
            raise RuntimeError(
                "Final CSV verification failed: target experiment contains "
                f"non-completed status {status!r}."
            )
        key = (
            str(row["model"]),
            int(row["seed"]),
            int(row["draw"]),
            int(row["N"]),
            int(row["K"]),
        )
        if previous_key is not None and key <= previous_key:
            reason = "duplicate" if key == previous_key else "unsorted"
            raise RuntimeError(
                "Final CSV verification failed: target experiment contains "
                f"{reason} checkpoint keys."
            )
        previous_key = key
        count += 1
    if count != expected_rows:
        raise RuntimeError(
            "Final CSV verification failed: expected "
            f"{expected_rows} rows for experiment {experiment_id}, found {count}."
        )


def retire_checkpoint_parts(out_path: Path) -> Path | None:
    """Atomically make active shards invisible before best-effort deletion."""

    directory = checkpoint_parts_dir(out_path)
    if not directory.exists():
        return None
    retired = directory.with_name(
        f".{directory.name}.retired-{uuid.uuid4().hex}"
    )
    os.replace(directory, retired)
    _fsync_directory(directory.parent)
    return retired


def manifest_path(out_path: Path) -> Path:
    return out_path.with_suffix(".manifest.json")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write and durably publish a JSON object with a same-directory rename."""

    _mkdir_durable(path.parent)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    token = uuid.uuid4().hex
    tmp = path.parent / f".{path.name}.{token}.tmp"
    descriptor = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _fsync_directory(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def git_state(project_dir: Path) -> dict[str, Any]:
    """Return the commit and dirty state without making Git a runtime requirement."""

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_dir,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=normal"],
                cwd=project_dir,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def core_environment() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for package, key in (
        ("scikit-learn", "scikit_learn"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("lightgbm", "lightgbm"),
        ("xgboost", "xgboost"),
    ):
        try:
            versions[key] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[key] = "not-installed"
    return versions


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def diagnostics_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Return compact run-level QA counts without duplicating cell diagnostics."""

    ok = frame[frame["status"].eq("ok")] if "status" in frame else frame
    result: dict[str, Any] = {}
    for column, key in (
        ("constant_prediction", "constant_prediction_rows"),
        ("underdetermined", "underdetermined_rows"),
    ):
        result[key] = int(ok[column].fillna(False).astype(bool).sum()) if column in ok else 0
    if "converged" in ok:
        result["nonconverged_rows"] = int(
            (~ok["converged"].fillna(False).astype(bool)).sum()
        )
    else:
        result["nonconverged_rows"] = 0
    by_model: dict[str, Any] = {}
    if "model" in frame:
        for model, all_group in frame.groupby("model", sort=True):
            group = all_group[all_group["status"].eq("ok")] if "status" in all_group else all_group
            summary = {
                "rows": int(len(group)),
                "constant_prediction_rows": int(
                    group.get("constant_prediction", pd.Series(False, index=group.index))
                    .fillna(False)
                    .astype(bool)
                    .sum()
                ),
                "underdetermined_rows": int(
                    group.get("underdetermined", pd.Series(False, index=group.index))
                    .fillna(False)
                    .astype(bool)
                    .sum()
                ),
                "nonconverged_rows": (
                    int(
                        (~group["converged"].fillna(False).astype(bool)).sum()
                    )
                    if "converged" in group
                    else 0
                ),
            }
            if "_fit_seconds" in all_group:
                timings = pd.to_numeric(all_group["_fit_seconds"], errors="coerce").dropna()
                summary["fit_seconds_total"] = float(timings.sum())
                summary["fit_seconds_median"] = float(timings.median()) if len(timings) else None
            if "_best_rounds" in all_group:
                rounds = pd.to_numeric(all_group["_best_rounds"], errors="coerce").dropna()
                summary["best_rounds"] = (
                    {
                        "min": int(rounds.min()),
                        "median": float(rounds.median()),
                        "max": int(rounds.max()),
                    }
                    if len(rounds)
                    else None
                )
            by_model[str(model)] = summary
    result["by_model"] = by_model
    return result


def parallel_preference(models: Iterable[str]) -> str:
    """Isolate BART's process-global RNG when jobs run concurrently."""

    return "processes" if "bart" in set(models) else "threads"


def effective_outer_n_jobs(model_name: str, configured_n_jobs: int) -> int:
    """Return safe Joblib parallelism for one selected model."""

    return 1 if model_name in SERIAL_OUTER_MODELS else configured_n_jobs


def model_run_settings(models: Iterable[str]) -> dict[str, Any]:
    """Capture model selection and environment overrides that affect fitted results."""

    return {
        "models": sorted(set(models)),
        "environment_overrides": {
            key: os.environ.get(key) for key in MODEL_ENV_KEYS if key in os.environ
        },
    }
