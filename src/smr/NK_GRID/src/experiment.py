"""Experiment identity and safe CSV checkpoint helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

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
MANIFEST_SCHEMA_VERSION = "1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_experiment_metadata(
    *,
    kind: str,
    data_path: Path,
    outcome: str,
    test_size: float,
    split_seed: int,
    algorithm_version: str = "legacy",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return stable metadata that distinguishes result-producing inputs."""

    settings = {
        "kind": kind,
        "algorithm_version": algorithm_version,
        "data_sha256": file_sha256(data_path),
        "outcome": outcome,
        "test_size": float(test_size),
        "split_seed": int(split_seed),
        "extra": extra or {},
    }
    encoded = json.dumps(settings, sort_keys=True, separators=(",", ":"))
    experiment_id = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]
    return {
        "experiment_id": experiment_id,
        "experiment_kind": kind,
        "algorithm_version": algorithm_version,
        "data_sha256": settings["data_sha256"],
        "data_path": str(data_path.resolve()),
        "outcome": outcome,
        "test_size": float(test_size),
        "split_seed": int(split_seed),
    }


def load_checkpoint(path: Path) -> pd.DataFrame:
    """Load authoritative shards, falling back to a legacy/final CSV."""

    parts = checkpoint_parts(path)
    if parts:
        frame = pd.concat((pd.read_csv(part) for part in parts), ignore_index=True)
        if frame.empty:
            return frame
        frame = frame.drop_duplicates(
            ["experiment_id", *CHECKPOINT_KEY_COLUMNS], keep="last"
        )
        return frame.sort_values(
            ["experiment_id", *CHECKPOINT_SORT_COLUMNS]
        ).reset_index(drop=True)
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if not frame.empty and "experiment_id" not in frame:
        raise ValueError(
            f"Existing checkpoint lacks experiment metadata: {path}. "
            "Remove it or choose a new --out path before resuming."
        )
    return frame


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
    result = result.drop_duplicates(["experiment_id", *key_columns], keep="last")
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    result.sort_values(["experiment_id", *sort_columns]).to_csv(tmp, index=False)
    tmp.replace(out_path)


def checkpoint_parts_dir(out_path: Path) -> Path:
    """Return ``result.parts`` for ``result.csv``."""

    return out_path.with_suffix(".parts")


def checkpoint_parts(out_path: Path) -> list[Path]:
    directory = checkpoint_parts_dir(out_path)
    return sorted(directory.glob("part-*.csv")) if directory.exists() else []


def write_checkpoint_part(rows: Iterable[dict[str, Any]], out_path: Path) -> Path | None:
    """Atomically append one immutable checkpoint shard."""

    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return None
    directory = checkpoint_parts_dir(out_path)
    directory.mkdir(parents=True, exist_ok=True)
    existing = checkpoint_parts(out_path)
    next_index = 1
    if existing:
        next_index = max(int(path.stem.split("-")[-1]) for path in existing) + 1
    part = directory / f"part-{next_index:06d}.csv"
    tmp = part.with_suffix(part.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(part)
    return part


def merge_checkpoint_parts(
    out_path: Path,
    *,
    key_columns: list[str] | None = None,
    sort_columns: list[str] | None = None,
    drop_output_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Atomically materialize the final CSV from immutable checkpoint shards."""

    key_columns = key_columns or CHECKPOINT_KEY_COLUMNS
    sort_columns = sort_columns or CHECKPOINT_SORT_COLUMNS
    frame = load_checkpoint(out_path)
    if frame.empty:
        return frame
    frame = frame.drop_duplicates(["experiment_id", *key_columns], keep="last")
    frame = frame.sort_values(["experiment_id", *sort_columns]).reset_index(drop=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    frame.drop(columns=drop_output_columns or [], errors="ignore").to_csv(tmp, index=False)
    tmp.replace(out_path)
    return frame


def manifest_path(out_path: Path) -> Path:
    return out_path.with_suffix(".manifest.json")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


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


def model_run_settings(models: Iterable[str]) -> dict[str, Any]:
    """Capture model selection and environment overrides that affect fitted results."""

    return {
        "models": sorted(set(models)),
        "environment_overrides": {
            key: os.environ.get(key) for key in MODEL_ENV_KEYS if key in os.environ
        },
    }
