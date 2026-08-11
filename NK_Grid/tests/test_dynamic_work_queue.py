from __future__ import annotations

import json
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

from conftest import write_repo_schema_bundle as write_schema_bundle
import aleatoric_nk_grid.flat_task_table as ft

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
from aleatoric_nk_grid.flat_task_table import (
    PublicSchemaMismatchError,
    ResultKeySetError,
    ResultProjectionError,
    _aborted_reason,
    _load_snapshot,
)
from aleatoric_nk_grid.worker_event_wal import WALFrameTooLarge


def _schema(path: Path) -> Path:
    path.write_text(json.dumps({"sources": [
        {"unit_type": "continuous", "features": [{"name": "x0"}]},
        {"unit_type": "onehot_group", "features": [{"name": "x1a"}, {"name": "x1b"}, {"name": "x1c"}]},
        {"unit_type": "continuous", "features": [{"name": "x2"}, {"name": "x3"}]},
    ]}), encoding="utf-8")
    return path


def _config(tmp_path: Path) -> NKGridConfig:
    frame = pd.DataFrame({
        "x0": np.arange(40, dtype=float), "x1": np.arange(40, dtype=float),
        "x2": np.arange(40, dtype=float), "y": np.arange(40, dtype=float),
    })
    schema = write_schema_bundle(tmp_path / "input", frame, predictors=["x0", "x1", "x2"])
    return NKGridConfig(schema=schema, out=tmp_path / "out.csv", outcome="y", models=("ols", "super_learner"), seed=1, test_size=0.2, n_seeds=1, n_draws=1, n_sizes_n=1, n_sizes_k=1, max_n=20, max_k=3, batch_size=1, n_jobs=8, repeat_plan=((1, 0),))


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


def test_aborted_reason_mapping_is_typed_and_exhaustive():
    assert _aborted_reason(WALFrameTooLarge("x")) == "RESULT_FRAME_TOO_LARGE"
    assert _aborted_reason(PublicSchemaMismatchError("x")) == "PUBLIC_SCHEMA_MISMATCH"
    assert _aborted_reason(ResultProjectionError("x")) == "RESULT_PROJECTION_FAILED"
    assert _aborted_reason(ResultKeySetError("x")) == "RESULT_KEY_SET_MISMATCH"
    assert _aborted_reason(UnicodeError("x")) == "RESULT_ENCODING_FAILED"
    assert _aborted_reason(RuntimeError("RESULT_KEY_SET_MISMATCH")) == "RESULT_PROTOCOL_VIOLATION"


def test_serialized_compatibility_flags_cannot_enable_legacy_csv_execution(tmp_path):
    snapshot = tmp_path / "forged-legacy.json"
    snapshot.write_text(
        json.dumps({
            "format_version": 2,
            "result_store_format": "test-legacy-csv-v2",
            "test_compatibility_mode": True,
            "task_table": str(tmp_path / "tasks.parquet"),
            "output_dir": str(tmp_path / "out"),
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported legacy dynamic snapshot"):
        _load_snapshot(snapshot)


def test_production_cli_rejects_legacy_snapshot_before_any_csv_state_machine(
    tmp_path,
):
    snapshot = tmp_path / "forged-legacy.json"
    snapshot.write_text(
        json.dumps({
            "format_version": 2,
            "result_store_format": "test-legacy-csv-v2",
            "test_compatibility_mode": True,
        }),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as stopped:
        ft.main([
            "verify", "--snapshot", str(snapshot), "--round", "1",
            "--generation", "g1", "--expected-prep-token", "prep",
        ])
    assert stopped.value.code == 6


def test_plan_requires_account_and_never_splits_super_learner(tmp_path):
    with pytest.raises(ValueError, match="account"):
        build_dynamic_plan(_config(tmp_path), n_grid=(10,), k_grid=(1,), cluster=_cluster(account=""), table_path=tmp_path / "tasks.parquet", snapshot_path=tmp_path / "snapshot.json", output_dir=tmp_path / "out", panel="p")
    plan = build_dynamic_plan(
        _config(tmp_path), n_grid=(10, 20), k_grid=(1, 2),
        cluster=_cluster(
            preparation_memory="3G", preparation_time_limit="01:00:00",
            preparation_tmp_dir=str(tmp_path / "prep-scratch"),
            verification_memory="3G", verification_time_limit="01:00:00",
            verification_tmp_dir=str(tmp_path / "verify-scratch"),
            finalization_memory="4G", finalization_time_limit="02:00:00",
            finalization_tmp_dir=str(tmp_path / "scratch"),
        ),
        table_path=tmp_path / "tasks.parquet",
        snapshot_path=tmp_path / "snapshot.json",
        output_dir=tmp_path / "out", panel="p",
    )
    assert plan["submission"]["array"] == "0-3%4"
    assert "--cpus-per-task=1" in plan["submission"]["sbatch_args"]
    assert "--account=project" in plan["submission"]["sbatch_args"]
    assert "--constraint=arch" in plan["submission"]["sbatch_args"]
    assert "--mem=3G" in plan["preparation"]["sbatch_args"]
    assert "--time=01:00:00" in plan["preparation"]["sbatch_args"]
    assert plan["preparation"]["tmp_dir"] == str(tmp_path / "prep-scratch")
    assert "--mem=3G" in plan["verification"]["sbatch_args"]
    assert "--time=01:00:00" in plan["verification"]["sbatch_args"]
    assert plan["verification"]["tmp_dir"] == str(tmp_path / "verify-scratch")
    assert "--mem=4G" in plan["finalization"]["sbatch_args"]
    assert "--time=02:00:00" in plan["finalization"]["sbatch_args"]
    assert plan["finalization"]["tmp_dir"] == str(tmp_path / "scratch")
    snapshot = json.loads((tmp_path / "snapshot.json").read_text())
    assert snapshot["preparation"]["tmp_dir"] == str((tmp_path / "prep-scratch").resolve())
    assert snapshot["verification"]["tmp_dir"] == str((tmp_path / "verify-scratch").resolve())
    assert snapshot["finalization"]["tmp_dir"] == str((tmp_path / "scratch").resolve())
    assert plan["memory"]["frame_copies"] == 12
