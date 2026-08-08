from __future__ import annotations

import json
from pathlib import Path

import pytest

from aleatoric_nk_grid.chunk_planning import (
    MEMORY_FRAME_COPIES,
    ClusterPolicy,
    build_dynamic_plan,
    check_memory_measurement,
    expanded_columns_for_k,
    implied_frame_copies,
    peak_memory_bytes,
)
from aleatoric_nk_grid.nk_grid import NKGridConfig


def _schema(path: Path) -> Path:
    path.write_text(json.dumps({"sources": [
        {"unit_type": "continuous", "features": [{"name": "x0"}]},
        {"unit_type": "onehot_group", "features": [{"name": "x1a"}, {"name": "x1b"}, {"name": "x1c"}]},
        {"unit_type": "continuous", "features": [{"name": "x2"}, {"name": "x3"}]},
    ]}), encoding="utf-8")
    return path


def _config(tmp_path: Path) -> NKGridConfig:
    return NKGridConfig(schema=_schema(tmp_path / "schema.json"), out=tmp_path / "out.csv", outcome="y", models=("ols", "super_learner"), seed=1, test_size=0.2, n_seeds=1, n_draws=1, n_sizes_n=1, n_sizes_k=1, max_n=20, max_k=3, batch_size=1, n_jobs=8, repeat_plan=((1, 0),))


def _cluster(**changes: object) -> ClusterPolicy:
    values: dict[str, object] = {"workers": 4, "rounds": 3, "partition": "long", "time_limit": "12:00:00", "account": "project", "constraint": "arch"}
    values.update(changes)
    return ClusterPolicy(**values)


def test_expanded_columns_not_source_count_drives_memory(tmp_path):
    schema = _schema(tmp_path / "schema.json")
    expanded = expanded_columns_for_k(schema, 2)
    assert expanded == 4
    assert peak_memory_bytes(100, expanded) > peak_memory_bytes(100, 2)


def test_memory_formula_and_probe_fail_closed_above_twelve_copies():
    measured = peak_memory_bytes(100, 6)
    checked = check_memory_measurement(measured, n_samples=100, expanded_columns=6)
    assert checked["implied_frame_copies"] == MEMORY_FRAME_COPIES
    assert implied_frame_copies(measured, n_samples=100, expanded_columns=6) == MEMORY_FRAME_COPIES
    with pytest.raises(RuntimeError, match="report and stop"):
        check_memory_measurement(peak_memory_bytes(100, 6, frame_copies=13), n_samples=100, expanded_columns=6)


def test_plan_requires_account_and_never_splits_super_learner(tmp_path):
    with pytest.raises(ValueError, match="account"):
        build_dynamic_plan(_config(tmp_path), n_grid=(10,), k_grid=(1,), cluster=_cluster(account=""), table_path=tmp_path / "tasks.parquet", snapshot_path=tmp_path / "snapshot.json", output_dir=tmp_path / "out", panel="p")
    plan = build_dynamic_plan(_config(tmp_path), n_grid=(10, 20), k_grid=(1, 2), cluster=_cluster(), table_path=tmp_path / "tasks.parquet", snapshot_path=tmp_path / "snapshot.json", output_dir=tmp_path / "out", panel="p")
    assert plan["submission"]["array"] == "0-3%4"
    assert "--cpus-per-task=1" in plan["submission"]["sbatch_args"]
    assert "--account=project" in plan["submission"]["sbatch_args"]
    assert "--constraint=arch" in plan["submission"]["sbatch_args"]
    assert plan["memory"]["frame_copies"] == 12
