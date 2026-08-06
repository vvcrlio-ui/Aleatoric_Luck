"""Lightweight spawn entry points for calibration memory measurement.

This module must remain import-cheap: a spawned interpreter imports the module
that defines its target before invoking the target.  Do not add project,
NumPy, pandas, scikit-learn, or other third-party imports here.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _join_cgroup(cgroup_path: str | None) -> bool:
    if cgroup_path is None:
        return False
    try:
        (Path(cgroup_path) / "cgroup.procs").write_text(
            str(os.getpid()), encoding="ascii"
        )
        return True
    except OSError:
        return False


def _record_import_state(path: str | None) -> None:
    if path is not None:
        Path(path).write_text(
            f"numpy={int('numpy' in sys.modules)}\n"
            f"pandas={int('pandas' in sys.modules)}\n",
            encoding="ascii",
        )


def import_state_probe_target(cgroup_path: str | None, observation_path: str) -> None:
    """Test target proving cgroup join precedes heavyweight project imports."""

    _join_cgroup(cgroup_path)
    _record_import_state(observation_path)


def cell_worker_target(
    cgroup_path: str | None,
    result_path: str,
    schema_path: str,
    outcome: str,
    seed: int,
    model_name: str,
    n: int,
    k: int,
    draw: int,
    max_seconds: float,
    observation_path: str | None,
    probe_only: int,
) -> None:
    cgroup_joined = _join_cgroup(cgroup_path)
    _record_import_state(observation_path)
    if probe_only:
        return
    from aleatoric_nk_grid.calibrate_cost import _run_cell_worker_after_join

    _run_cell_worker_after_join(
        result_path,
        cgroup_joined,
        schema_path,
        outcome,
        seed,
        model_name,
        n,
        k,
        draw,
        max_seconds,
    )


def task_cell_worker_target(
    cgroup_path: str | None,
    schema_path: str,
    outcome: str,
    model_name: str,
    n: int,
    k: int,
    seed: int,
    draw: int,
    max_seconds: float,
    observation_path: str | None,
    probe_only: int,
) -> None:
    _join_cgroup(cgroup_path)
    _record_import_state(observation_path)
    if probe_only:
        return
    from aleatoric_nk_grid.calibrate_cost import _run_task_cell_after_join

    _run_task_cell_after_join(
        schema_path, outcome, model_name, n, k, seed, draw, max_seconds
    )
