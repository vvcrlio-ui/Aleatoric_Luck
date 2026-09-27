"""Experiment identity and run-settings helpers."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Iterable


MODEL_ENV_KEYS = (
    "LGBM_MAX_ROUNDS",
    "RF_MAX_FEATURES",
    "RF_MIN_SAMPLES_LEAF",
    "RF_N_ESTIMATORS",
    "XGB_MAX_ROUNDS",
)

SERIAL_OUTER_MODELS = frozenset({"lightgbm", "super_learner"})


def build_experiment_metadata(
    *,
    kind: str,
    experiment_id: str,
    data_version: str,
    model_spec_version: str,
    outcome: str,
    test_size: float,
    split_seed: int,
    algorithm_version: str = "legacy",
    semantic_contract: dict[str, Any],
    split_mode: str = "internal_random",
) -> dict[str, Any]:
    """Return researcher-declared identity and directly comparable semantics."""

    if split_mode not in {"internal_random", "external_test"}:
        raise ValueError(f"Unknown split_mode: {split_mode!r}")
    return {
        "experiment_id": experiment_id,
        "identity": {
            "mode": "explicit-v1",
            "experiment_id": experiment_id,
            "data_version": data_version,
            "model_spec_version": model_spec_version,
        },
        "semantic_contract": semantic_contract,
        "experiment_kind": kind,
        "algorithm_version": algorithm_version,
        "outcome": outcome,
        "test_size": float(test_size) if split_mode == "internal_random" else None,
        "split_mode": split_mode,
        "split_seed": int(split_seed),
    }


def add_metadata(row: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    return {**metadata, **row}


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


def model_run_settings(models: Iterable[str]) -> dict[str, Any]:
    """Capture model selection and environment overrides that affect fitted results."""

    return {
        "models": sorted(set(models)),
        "environment_overrides": {
            key: os.environ.get(key) for key in MODEL_ENV_KEYS if key in os.environ
        },
    }
