"""Experiment identity and safe CSV checkpoint helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
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
MANIFEST_SCHEMA_VERSION = "1"
EXPERIMENT_IDENTITY_VERSION = 3
COMPLETED_CHECKPOINT_STATUSES = frozenset({"ok", "skipped"})
SERIAL_OUTER_MODELS = frozenset({"lightgbm", "super_learner"})


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
def output_run_lock(out_path: Path) -> Iterator[Path]:
    """Fail fast when another process is already writing this output."""

    lock_path = out_path.with_suffix(out_path.suffix + ".run.lock")
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


def merge_checkpoint_parts(
    out_path: Path,
    *,
    key_columns: list[str] | None = None,
    sort_columns: list[str] | None = None,
    drop_output_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Atomically materialize the final CSV from append-only checkpoint shards."""

    key_columns = key_columns or CHECKPOINT_KEY_COLUMNS
    sort_columns = sort_columns or CHECKPOINT_SORT_COLUMNS
    frame = load_checkpoint(out_path)
    if frame.empty:
        return frame
    frame = frame.drop_duplicates(["experiment_id", *key_columns], keep="last")
    frame = frame.sort_values(["experiment_id", *sort_columns]).reset_index(drop=True)
    _write_dataframe_atomic(
        frame.drop(columns=drop_output_columns or [], errors="ignore"),
        out_path,
    )
    return frame


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
                "nonconverged_rows": int(
                    (~group.get("converged", pd.Series(False, index=group.index))
                    .fillna(False)
                    .astype(bool))
                    .sum()
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
