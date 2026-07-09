"""Experiment identity and safe CSV checkpoint helpers."""

from __future__ import annotations

import hashlib
import json
import os
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
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return stable metadata that distinguishes result-producing inputs."""

    settings = {
        "kind": kind,
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
        "data_sha256": settings["data_sha256"],
        "data_path": str(data_path.resolve()),
        "outcome": outcome,
        "test_size": float(test_size),
        "split_seed": int(split_seed),
    }


def load_checkpoint(path: Path) -> pd.DataFrame:
    """Load a checkpoint, refusing legacy rows that cannot be identified safely."""

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
