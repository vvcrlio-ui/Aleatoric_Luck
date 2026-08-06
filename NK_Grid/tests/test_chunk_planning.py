"""Stage B: calibrated cost model, chunk budget, resources and acceptance.

Every fixture builds its own calibration document.  Nothing here reads a real
schema, a real calibration run, or any dataset-specific shape.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aleatoric_nk_grid import chunk_planning as cp
from aleatoric_nk_grid.calibrated_cost_model import (
    CALIBRATION_FORMAT_VERSION,
    CalibrationUnusable,
    CalibratedCostModel,
    PowerLaw,
)
from aleatoric_nk_grid.flat_task_table import (
    TaskRow,
    pack_lpt,
    preprocess_mode_for_group,
)


# ``log_c=0, a=1, b=1`` makes every prediction exactly ``K * N``, so the tests
# assert arithmetic rather than restating the implementation.
UNIT_FIT = {"log_c": 0.0, "a": 1.0, "b": 1.0, "r2": 0.99, "residual_range": [0.0, 0.0], "n_points": 9}


def _fit(scale: float = 1.0, *, r2: float = 0.99) -> dict[str, object]:
    import math

    return {"log_c": math.log(scale), "a": 1.0, "b": 1.0, "r2": r2, "residual_range": [0.0, 0.0], "n_points": 9}


def _calibration(
    tmp_path: Path,
    *,
    version: int = CALIBRATION_FORMAT_VERSION,
    fit_cost: dict[str, object] | None = None,
    preprocess_cost: dict[str, object] | None = None,
    peak_rss: dict[str, object] | None = None,
    t0_total: float = 6.0,
    validation: list[dict[str, object]] | None = None,
    suspect: int = 0,
    task_peak_status: str = "measured",
    task_peak_bytes: int | None = 8 * 1024 ** 3,
) -> Path:
    payload = {
        "format_version": version,
        "t0_seconds": {"total": {"median": t0_total}},
        "fit_cost": fit_cost if fit_cost is not None else {"ols": _fit(1.0), "super_learner": _fit(10.0)},
        "preprocess_cost": preprocess_cost
        if preprocess_cost is not None
        else {"imputed": _fit(0.5), "passthrough": _fit(0.01)},
        "peak_rss_bytes": peak_rss
        if peak_rss is not None
        else {"ols": _fit(1000.0), "super_learner": _fit(5000.0)},
        "memory_measurement": {
            "cell_memory_scope_suspect_count": suspect,
            "task_cgroup_peak_n_jobs_8": {"status": task_peak_status, "bytes": task_peak_bytes},
        },
        "validation": validation
        if validation is not None
        else [{"n": 10, "k": 10, "model": "ols", "ratio": 1.1, "status": "ok"}],
    }
    path = tmp_path / "cost_model.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _row(models: tuple[str, ...], *, group: str = "imputed_core", n: int = 10, k: int = 20, row_id: str = "r") -> TaskRow:
    return TaskRow(row_id=row_id, seed=1, draw=2, n_samples=n, k_features=k, group=group, models=models)


# ---------------------------------------------------------------------------
# Loading and refusal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("version", [1, 2, 3, 4, 6, None])
def test_only_the_current_calibration_format_is_accepted(tmp_path, version):
    path = _calibration(tmp_path, version=version)
    with pytest.raises(CalibrationUnusable, match="format_version"):
        CalibratedCostModel.from_json(path)


def test_out_of_band_validation_points_block_packing(tmp_path):
    path = _calibration(
        tmp_path,
        validation=[
            {"n": 10, "k": 10, "model": "ols", "ratio": 1.0, "status": "ok"},
            {"n": 4000, "k": 8000, "model": "ols", "ratio": 0.07, "status": "ok"},
        ],
    )
    with pytest.raises(CalibrationUnusable, match="outside predicted/actual band"):
        CalibratedCostModel.from_json(path)
    inspected = CalibratedCostModel.from_json(path, require_validation=False)
    assert inspected.validation_checked == 2
    assert len(inspected.validation_failures) == 1


def test_calibration_without_any_completed_validation_point_is_refused(tmp_path):
    path = _calibration(
        tmp_path,
        validation=[{"n": 10, "k": 10, "model": "ols", "ratio": None, "status": "censored"}],
    )
    with pytest.raises(CalibrationUnusable, match="never checked"):
        CalibratedCostModel.from_json(path)


def test_missing_t0_is_refused(tmp_path):
    path = _calibration(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["t0_seconds"] = {}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CalibrationUnusable, match="t0_seconds"):
        CalibratedCostModel.from_json(path)


def test_low_r2_models_are_reported_and_can_be_enforced(tmp_path):
    path = _calibration(tmp_path, fit_cost={"ols": _fit(1.0, r2=0.7), "super_learner": _fit(10.0)})
    model = CalibratedCostModel.from_json(path)
    assert model.low_r2_models == ("ols",)
    with pytest.raises(CalibrationUnusable, match="r2 below"):
        CalibratedCostModel.from_json(path, min_r2=0.8)


# ---------------------------------------------------------------------------
# B1: cost function
# ---------------------------------------------------------------------------


def test_group_cost_counts_preprocessing_once_and_every_model_fit(tmp_path):
    model = CalibratedCostModel.from_json(_calibration(tmp_path))
    row = _row(("ols", "super_learner"), n=10, k=20)
    # preprocess 0.5*K*N + ols 1*K*N + super_learner 10*K*N = 11.5 * 200
    assert model.row_cost_seconds(row) == pytest.approx(11.5 * 200)


def test_passthrough_group_uses_the_passthrough_preprocessing_mode(tmp_path):
    path = _calibration(tmp_path, fit_cost={"ols": _fit(1.0), "lightgbm": _fit(2.0)})
    model = CalibratedCostModel.from_json(path)
    row = _row(("lightgbm",), group="passthrough", n=10, k=20)
    assert model.row_cost_seconds(row) == pytest.approx((0.01 + 2.0) * 200)


def test_unknown_execution_group_is_rejected():
    with pytest.raises(ValueError, match="unknown execution group"):
        preprocess_mode_for_group("something_else")


def test_model_without_a_fit_names_itself(tmp_path):
    model = CalibratedCostModel.from_json(_calibration(tmp_path))
    with pytest.raises(CalibrationUnusable, match="ridge"):
        model.row_cost_seconds(_row(("ols", "ridge")))


def test_cost_function_drives_the_stage_a_packer(tmp_path):
    model = CalibratedCostModel.from_json(_calibration(tmp_path))
    rows = [
        _row(("ols",), n=10, k=10, row_id="cheap"),
        _row(("ols",), n=100, k=100, row_id="dear"),
    ]
    packed = pack_lpt(rows, budget=1e9, cost_function=model.cost_function())
    costs = {row.row_id: row.est_cost for row in packed}
    assert costs["cheap"] == pytest.approx(1.5 * 100)
    assert costs["dear"] == pytest.approx(1.5 * 10_000)


# ---------------------------------------------------------------------------
# B2: chunk budget
# ---------------------------------------------------------------------------


def test_shortest_budget_that_amortises_t0_is_chosen(tmp_path):
    model = CalibratedCostModel.from_json(_calibration(tmp_path, t0_total=6.0))
    assert cp.choose_t_target_seconds(model) == 1800


def test_budget_escalates_when_startup_is_expensive(tmp_path):
    model = CalibratedCostModel.from_json(_calibration(tmp_path, t0_total=120.0))
    assert cp.choose_t_target_seconds(model) == 3600


def test_unamortisable_startup_is_an_error_not_a_silent_long_chunk(tmp_path):
    model = CalibratedCostModel.from_json(_calibration(tmp_path, t0_total=1000.0))
    with pytest.raises(CalibrationUnusable, match="exceeds"):
        cp.choose_t_target_seconds(model)


# ---------------------------------------------------------------------------
# B3: super_learner split
# ---------------------------------------------------------------------------


def test_multi_cpu_split_requires_the_unmeasured_parallel_efficiency(tmp_path):
    model = CalibratedCostModel.from_json(_calibration(tmp_path))
    with pytest.raises(CalibrationUnusable, match="parallel efficiency"):
        cp.split_super_learner_tradeoff(
            model, n_samples=10, k_features=20, core_models=("ols",), super_learner_cpus=8
        )


def test_single_cpu_split_costs_exactly_one_extra_preprocessing_pass(tmp_path):
    model = CalibratedCostModel.from_json(_calibration(tmp_path))
    tradeoff = cp.split_super_learner_tradeoff(
        model, n_samples=10, k_features=20, core_models=("ols",), super_learner_cpus=1
    )
    assert tradeoff.extra_core_seconds == pytest.approx(0.5 * 200)
    # Serial class stops carrying Super Learner's footprint: 5000*K*N -> 1000*K*N.
    assert tradeoff.memory_reduction_bytes == pytest.approx(4000 * 200)


def test_split_threshold_is_none_when_no_k_frees_enough_memory(tmp_path):
    model = CalibratedCostModel.from_json(_calibration(tmp_path))
    assert (
        cp.split_super_learner_min_k(
            model,
            n_samples=10,
            k_grid=(10, 20, 40),
            core_models=("ols",),
            super_learner_cpus=1,
            min_memory_reduction_bytes=10 ** 12,
        )
        is None
    )


def test_split_threshold_is_the_first_k_clearing_the_policy_bar(tmp_path):
    model = CalibratedCostModel.from_json(_calibration(tmp_path))
    # reduction = 4000 * K * N, N = 10 -> clears 1_200_000 first at K = 30.
    assert (
        cp.split_super_learner_min_k(
            model,
            n_samples=10,
            k_grid=(10, 20, 30, 40),
            core_models=("ols",),
            super_learner_cpus=1,
            min_memory_reduction_bytes=1_200_000,
        )
        == 30
    )


# ---------------------------------------------------------------------------
# B4: resource requests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_bytes,expected", [(1024 ** 3, "1G"), (1024 ** 2, "1M"), (1536 * 1024 ** 2, "1536M"), (1, "1M")]
)
def test_slurm_memory_rounds_up_to_whole_mebibytes(num_bytes, expected):
    assert cp.format_slurm_memory(num_bytes) == expected


def test_resource_request_is_fully_specified_per_class(tmp_path):
    model = CalibratedCostModel.from_json(_calibration(tmp_path))
    packed = pack_lpt([_row(("ols",), n=10, k=20)], budget=1e9, cost_function=model.cost_function())
    requests = cp.resource_requests(
        packed,
        model,
        cpus_per_task={"serial": 1},
        partition={"serial": "long"},
        time_limit={"serial": "01:00:00"},
        memory_safety_factor=1.5,
    )
    request = requests["serial"]
    assert (request.partition, request.cpus_per_task, request.time_limit) == ("long", 1, "01:00:00")
    # 1000 * K * N * 1.5 bytes rounded up to whole MiB.
    assert request.memory == "1M"


def test_missing_cluster_value_for_a_used_class_is_an_error(tmp_path):
    model = CalibratedCostModel.from_json(_calibration(tmp_path))
    packed = pack_lpt([_row(("ols",))], budget=1e9, cost_function=model.cost_function())
    with pytest.raises(ValueError, match="no partition supplied"):
        cp.resource_requests(
            packed,
            model,
            cpus_per_task={"serial": 1},
            partition={},
            time_limit={"serial": "01:00:00"},
            memory_safety_factor=1.0,
        )


def test_suspect_memory_measurements_block_mem_sizing(tmp_path):
    model = CalibratedCostModel.from_json(_calibration(tmp_path, suspect=3))
    packed = pack_lpt([_row(("ols",))], budget=1e9, cost_function=model.cost_function())
    with pytest.raises(CalibrationUnusable, match="memory_scope_suspect"):
        cp.resource_requests(
            packed,
            model,
            cpus_per_task={"serial": 1},
            partition={"serial": "long"},
            time_limit={"serial": "01:00:00"},
            memory_safety_factor=1.0,
        )


def test_multi_cpu_class_needs_the_measured_concurrent_task_peak(tmp_path):
    model = CalibratedCostModel.from_json(
        _calibration(tmp_path, task_peak_status="not_measured", task_peak_bytes=None)
    )
    packed = pack_lpt(
        [_row(("super_learner",), group="super_learner")], budget=1e9, cost_function=model.cost_function()
    )
    with pytest.raises(CalibrationUnusable, match="concurrent task memory peak"):
        cp.resource_requests(
            packed,
            model,
            cpus_per_task={"super_learner": 8},
            partition={"super_learner": "long"},
            time_limit={"super_learner": "01:00:00"},
            memory_safety_factor=1.0,
        )


def test_multi_cpu_class_is_floored_at_the_measured_task_peak(tmp_path):
    model = CalibratedCostModel.from_json(_calibration(tmp_path, task_peak_bytes=4 * 1024 ** 3))
    packed = pack_lpt(
        [_row(("super_learner",), group="super_learner", n=1, k=1)],
        budget=1e9,
        cost_function=model.cost_function(),
    )
    requests = cp.resource_requests(
        packed,
        model,
        cpus_per_task={"super_learner": 8},
        partition={"super_learner": "long"},
        time_limit={"super_learner": "02:00:00"},
        memory_safety_factor=1.0,
    )
    assert requests["super_learner"].memory == "4G"


# ---------------------------------------------------------------------------
# B5: acceptance reports
# ---------------------------------------------------------------------------


def test_packing_report_separates_rows_larger_than_the_budget(tmp_path):
    model = CalibratedCostModel.from_json(_calibration(tmp_path))
    rows = [_row(("ols",), n=10, k=10, row_id=f"small-{index}") for index in range(6)]
    rows.append(_row(("ols",), n=200, k=200, row_id="huge"))
    budget = 1000.0
    packed = pack_lpt(rows, budget=budget, cost_function=model.cost_function())
    report = cp.packing_report(packed, budget=budget)
    assert report["oversized_chunk_count"] == 1
    assert report["p95_over_p50"] is not None
    assert report["meets_acceptance"] is True


def test_t0_share_uses_the_median_chunk(tmp_path):
    model = CalibratedCostModel.from_json(_calibration(tmp_path, t0_total=15.0))
    rows = [_row(("ols",), n=10, k=10, row_id=f"r{index}") for index in range(4)]
    packed = pack_lpt(rows, budget=150.0, cost_function=model.cost_function())
    # each row costs 1.5 * 100 = 150, so every chunk holds exactly one row
    assert cp.t0_share(model, packed) == pytest.approx(0.1)


def test_compare_budgets_reports_both_acceptance_criteria(tmp_path):
    model = CalibratedCostModel.from_json(_calibration(tmp_path, t0_total=6.0))
    rows = [_row(("ols",), n=10, k=10, row_id=f"r{index}") for index in range(40)]
    comparison = cp.compare_budgets(rows, model, budgets=(1800.0, 3600.0))
    assert [entry["budget"] for entry in comparison] == [1800.0, 3600.0]
    assert comparison[0]["chunk_count"] >= comparison[1]["chunk_count"]
    assert all(entry["meets_t0_acceptance"] for entry in comparison)


def test_power_law_prediction_rejects_degenerate_scale():
    law = PowerLaw(log_c=0.0, a=1.0, b=1.0, r2=1.0)
    with pytest.raises(ValueError):
        law.predict(0, 10)


# ---------------------------------------------------------------------------
# Packing by resource class, and the end-to-end plan
# ---------------------------------------------------------------------------


def test_super_learner_rows_never_share_a_chunk_with_serial_rows(tmp_path):
    model = CalibratedCostModel.from_json(_calibration(tmp_path))
    rows = [
        _row(("ols",), row_id="serial-a", n=1, k=1),
        _row(("ols",), row_id="serial-b", n=1, k=1),
        _row(("super_learner",), group="super_learner", row_id="sl-a", n=1, k=1),
        _row(("super_learner",), group="super_learner", row_id="sl-b", n=1, k=1),
    ]
    packed, ranges = cp.pack_by_resource_class(rows, budget=1e9, cost_function=model.cost_function())
    by_chunk: dict[int, set[str]] = {}
    for row in packed:
        by_chunk.setdefault(row.chunk_id, set()).add(row.group)
    assert all(len(groups) == 1 for groups in by_chunk.values())
    assert set(ranges) == {"serial", "super_learner"}
    # Contiguous, non-overlapping ranges so each class is one --array=a-b.
    (serial_first, serial_last) = ranges["serial"]
    (sl_first, sl_last) = ranges["super_learner"]
    assert serial_last < sl_first
    assert sl_last == max(row.chunk_id for row in packed)


def test_naive_single_pass_packing_would_have_demoted_super_learner(tmp_path):
    """Documents why Stage B may not simply call ``pack_lpt`` over all rows."""

    model = CalibratedCostModel.from_json(_calibration(tmp_path))
    rows = [
        _row(("ols",), row_id="serial-a", n=1, k=1),
        _row(("super_learner",), group="super_learner", row_id="sl-a", n=1, k=1),
    ]
    mixed = pack_lpt(rows, budget=1e9, cost_function=model.cost_function())
    assert len({row.chunk_id for row in mixed}) == 1
    from aleatoric_nk_grid.flat_task_table import resource_class_for_rows

    assert resource_class_for_rows(mixed) == "serial"  # the demotion this design avoids


def _planning_config(tmp_path: Path):
    from aleatoric_nk_grid.nk_grid import NKGridConfig

    return NKGridConfig(
        schema=tmp_path / "schema.json",
        out=tmp_path / "result.csv",
        outcome="y",
        models=("ols", "super_learner"),
        seed=11,
        test_size=0.3,
        n_seeds=1,
        n_draws=1,
        n_sizes_n=1,
        n_sizes_k=1,
        max_n=20,
        max_k=2,
        batch_size=4,
        n_jobs=1,
    )


def _cluster() -> cp.ClusterPolicy:
    return cp.ClusterPolicy(
        cpus_per_task={"serial": 1, "super_learner": 8},
        partition={"serial": "long", "super_learner": "long"},
        time_limit={"serial": "01:00:00", "super_learner": "02:00:00"},
        memory_safety_factor=1.25,
        max_concurrent={"serial": 200, "super_learner": 20},
    )


def test_end_to_end_plan_writes_table_snapshot_and_submission_ranges(tmp_path):
    calibration = _calibration(tmp_path)
    plan = cp.build_calibrated_plan(
        _planning_config(tmp_path),
        calibration_path=calibration,
        n_grid=(10, 20),
        k_grid=(5, 10),
        cluster=_cluster(),
        table_path=tmp_path / "table.parquet",
        snapshot_path=tmp_path / "snapshot.json",
        output_dir=tmp_path / "chunks",
        panel="unit-test-panel",
        split_super_learner_min_k=1,  # always split, so both classes exist
    )
    assert Path(plan["task_table"]).exists()
    assert Path(plan["snapshot"]).exists()
    assert plan["t_target_seconds"] == 1800
    classes = {entry["resource_class"] for entry in plan["submissions"]}
    assert classes == {"serial", "super_learner"}
    for entry in plan["submissions"]:
        assert entry["array"].endswith("%200") or entry["array"].endswith("%20")
        assert any(arg.startswith("--partition=") for arg in entry["sbatch_args"])
        assert any(arg.startswith("--mem=") for arg in entry["sbatch_args"])
        assert any(arg.startswith("--time=") for arg in entry["sbatch_args"])
    assert plan["resource_requests"]["super_learner"]["cpus_per_task"] == 8


def test_plan_refuses_to_emit_when_startup_dominates(tmp_path):
    # t0 = 200s against ~300s median chunks: two thirds of the wall clock would
    # be interpreter startup, which acceptance 6 exists to prevent.
    calibration = _calibration(tmp_path, t0_total=200.0)
    with pytest.raises(CalibrationUnusable, match="acceptance"):
        cp.build_calibrated_plan(
            _planning_config(tmp_path),
            calibration_path=calibration,
            n_grid=(10,),
            k_grid=(5,),
            cluster=_cluster(),
            table_path=tmp_path / "t.parquet",
            snapshot_path=tmp_path / "s.json",
            output_dir=tmp_path / "chunks",
            panel="unit-test-panel",
            split_super_learner_min_k=1,
            t_target_seconds=100,  # chunks far too short to amortise t0=6s
        )


def test_plan_can_be_inspected_when_it_fails_acceptance(tmp_path):
    plan = cp.build_calibrated_plan(
        _planning_config(tmp_path),
        calibration_path=_calibration(tmp_path, t0_total=200.0),
        n_grid=(10,),
        k_grid=(5,),
        cluster=_cluster(),
        table_path=tmp_path / "t.parquet",
        snapshot_path=tmp_path / "s.json",
        output_dir=tmp_path / "chunks",
        panel="unit-test-panel",
        split_super_learner_min_k=1,
        t_target_seconds=100,
        strict=False,
    )
    assert plan["packing"]["meets_t0_acceptance"] is False


def test_directory_entry_budget_blocks_a_plan_that_would_flood_one_directory(tmp_path):
    with pytest.raises(ValueError, match="entries in one output directory"):
        cp.build_calibrated_plan(
            _planning_config(tmp_path),
            calibration_path=_calibration(tmp_path),
            n_grid=(10, 20),
            k_grid=(5, 10),
            cluster=_cluster(),
            table_path=tmp_path / "t.parquet",
            snapshot_path=tmp_path / "s.json",
            output_dir=tmp_path / "chunks",
            panel="unit-test-panel",
            split_super_learner_min_k=1,
            max_directory_entries=2,
        )


def test_plan_reports_directory_entries_when_no_budget_is_stated(tmp_path):
    plan = cp.build_calibrated_plan(
        _planning_config(tmp_path),
        calibration_path=_calibration(tmp_path),
        n_grid=(10,),
        k_grid=(5,),
        cluster=_cluster(),
        table_path=tmp_path / "t.parquet",
        snapshot_path=tmp_path / "s.json",
        output_dir=tmp_path / "chunks",
        panel="unit-test-panel",
        split_super_learner_min_k=1,
    )
    filesystem = plan["filesystem"]
    assert filesystem["output_directory_entries"] == filesystem["chunk_count"] * 2
    assert filesystem["within_directory_budget"] is None


def test_cli_writes_a_plan_from_one_request_file(tmp_path):
    from dataclasses import fields

    config = _planning_config(tmp_path)
    config_payload = {}
    for field in fields(config):
        value = getattr(config, field.name)
        if isinstance(value, Path):
            config_payload[field.name] = str(value)
        elif isinstance(value, tuple):
            config_payload[field.name] = [list(item) if isinstance(item, tuple) else item for item in value]
        else:
            config_payload[field.name] = value
    request = {
        "calibration": str(_calibration(tmp_path)),
        "config": config_payload,
        "n_grid": [10],
        "k_grid": [5],
        "panel": "unit-test-panel",
        "task_table": str(tmp_path / "t.parquet"),
        "snapshot": str(tmp_path / "s.json"),
        "output_dir": str(tmp_path / "chunks"),
        "split_super_learner_min_k": 1,
        "cluster": {
            "cpus_per_task": {"serial": 1, "super_learner": 8},
            "partition": {"serial": "long", "super_learner": "long"},
            "time_limit": {"serial": "01:00:00", "super_learner": "02:00:00"},
            "memory_safety_factor": 1.25,
            "max_concurrent": {"serial": 200, "super_learner": 20},
        },
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    plan_path = tmp_path / "plan.json"
    cp.main(["--request", str(request_path), "--plan-out", str(plan_path)])
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["row_count"] > 0
    assert Path(plan["task_table"]).exists()


def test_planned_table_runs_through_the_stage_a_worker_unchanged(tmp_path):
    """The calibrated plan must produce a table Stage A executes bit-for-bit."""

    import numpy as np
    import pandas as pd
    import yaml
    from conftest import write_schema_bundle
    from aleatoric_nk_grid.flat_task_table import run_snapshot_chunk
    from aleatoric_nk_grid.nk_grid import METRIC_COLUMNS, NKGridConfig, run_nk_grid

    source = Path(__file__).resolve().parents[1] / "model_params.yaml"
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["regression"]["lightgbm"].update({"max_rounds": 2, "cv_folds": 2})
    fast_params = tmp_path / "fast_model_params.yaml"
    fast_params.write_text(yaml.safe_dump(payload), encoding="utf-8")

    values = np.arange(40, dtype=float)
    frame = pd.DataFrame({"y": values * 2 + values % 5, "x1": values, "x2": values % 3})
    schema = write_schema_bundle(
        tmp_path / "input",
        frame,
        predictors=["x1", "x2"],
        imputation={
            "continuous": "median",
            "ordinal": "most_frequent",
            "onehot_group": "atomic_mode",
            "model_overrides": {"lightgbm": "passthrough", "xgboost": "passthrough"},
        },
    )
    config = NKGridConfig(
        schema=schema, out=tmp_path / "direct.csv", outcome="y", models=("ols", "lightgbm"),
        seed=17, test_size=0.25, n_seeds=1, n_draws=1, n_sizes_n=1, n_sizes_k=1,
        max_n=12, max_k=1, min_n=10, batch_size=2, n_jobs=1,
        repeat_plan=((17, 0),), n_grid=(10, 12), k_grid=(1,), rerun_completed=False,
        model_params=fast_params,
    )
    plan = cp.build_calibrated_plan(
        config,
        calibration_path=_calibration(
            tmp_path, t0_total=0.01, fit_cost={"ols": _fit(1.0), "lightgbm": _fit(1.0)},
            peak_rss={"ols": _fit(1000.0), "lightgbm": _fit(1000.0)},
        ),
        n_grid=(10, 12),
        k_grid=(1,),
        cluster=cp.ClusterPolicy(
            cpus_per_task={"serial": 1}, partition={"serial": "long"},
            time_limit={"serial": "01:00:00"}, memory_safety_factor=1.0,
            max_concurrent={"serial": 50},
        ),
        table_path=tmp_path / "tasks.parquet",
        snapshot_path=tmp_path / "snapshot.json",
        output_dir=tmp_path / "chunks",
        panel="unit-test-panel",
        t_target_seconds=1800,
    )
    outputs = [run_snapshot_chunk(Path(plan["snapshot"]), chunk_id)
               for chunk_id in range(plan["filesystem"]["chunk_count"])]
    run_nk_grid(config)

    sort_columns = ["model", "seed", "draw", "N", "K"]
    planned = pd.concat([pd.read_csv(path) for path in outputs]).sort_values(sort_columns).reset_index(drop=True)
    direct = pd.read_csv(config.out).sort_values(sort_columns).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        planned.loc[:, [*sort_columns, *METRIC_COLUMNS]],
        direct.loc[:, [*sort_columns, *METRIC_COLUMNS]],
        check_exact=True,
    )
