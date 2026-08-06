from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from aleatoric_nk_grid_measure_worker import cell_worker_target, task_cell_worker_target

from aleatoric_nk_grid import calibrate_cost as cc


def _allocate_bytes_for_peak(byte_count: int, seconds: float = 0.15) -> None:
    """Module-level so multiprocessing ``spawn`` can import it in a child."""

    payload = bytearray(byte_count)
    payload[::4096] = b"\x01" * len(payload[::4096])
    time.sleep(seconds)


def _test_shape(
    n_sources: int,
    *,
    onehot_sources: int | None = None,
    onehot_dtype: str = "float64",
    continuous_dtype: str = "float64",
) -> cc.PanelShape:
    """Small explicit shape for tests that do not exercise schema parsing."""

    n_onehot = n_sources // 2 if onehot_sources is None else onehot_sources
    sources = tuple(
        [cc.ShapeSource("continuous", (continuous_dtype,)) for _ in range(n_sources - n_onehot)]
        + [cc.ShapeSource("onehot_group", (onehot_dtype, onehot_dtype, onehot_dtype)) for _ in range(n_onehot)]
    )
    expanded_count = sum(source.width for source in sources)
    return cc.PanelShape(
        schema_path="<test-shape>",
        sources=sources,
        dtype_source="declared",
        dtype_metadata_declared=expanded_count,
        dtype_metadata_total=expanded_count,
    )


# ---------------------------------------------------------------------------
# 1. Power-law regression recovers known exponents
# ---------------------------------------------------------------------------


def test_fit_power_law_recovers_known_exponents_noise_free():
    rng = np.random.default_rng(0)
    n = rng.integers(1, 5000, size=60).astype(float)
    k = rng.integers(1, 5000, size=60).astype(float)
    c, a, b = 3.5, 0.85, 0.15
    t = c * (k**a) * (n**b)

    fit = cc.fit_power_law(n, k, t)

    assert abs(fit.a - a) < 1e-6
    assert abs(fit.b - b) < 1e-6
    assert abs(fit.log_c - np.log(c)) < 1e-6
    assert fit.r2 > 1 - 1e-9


def test_fit_power_law_recovers_known_exponents_with_noise():
    rng = np.random.default_rng(1)
    n = rng.integers(1, 5000, size=200).astype(float)
    k = rng.integers(1, 5000, size=200).astype(float)
    c, a, b = 1.2, 0.7, 0.4
    t = c * (k**a) * (n**b) * (1.0 + rng.normal(0, 0.02, size=200))

    fit = cc.fit_power_law(n, k, t)

    assert abs(fit.a - a) < 0.05
    assert abs(fit.b - b) < 0.05
    assert abs(fit.log_c - np.log(c)) < 0.1
    assert fit.r2 > 0.9


def test_fit_power_law_rejects_too_few_points():
    with pytest.raises(ValueError):
        cc.fit_power_law([1.0, 2.0], [1.0, 2.0], [1.0, 2.0])


# ---------------------------------------------------------------------------
# 2. Synthetic data determinism
# ---------------------------------------------------------------------------


def test_synthetic_bundle_generation_is_deterministic(tmp_path):
    params = cc.SyntheticDataParams(n_train=120, shape=_test_shape(15), seed=42)
    schema_a, stats_a = cc.generate_synthetic_bundle(tmp_path / "a", params)
    schema_b, stats_b = cc.generate_synthetic_bundle(tmp_path / "b", params)

    frame_a = pd.read_parquet(tmp_path / "a" / "train.parquet")
    frame_b = pd.read_parquet(tmp_path / "b" / "train.parquet")

    pd.testing.assert_frame_equal(frame_a, frame_b, check_exact=True)
    assert stats_a == stats_b
    assert schema_a.read_text(encoding="utf-8") == schema_b.read_text(encoding="utf-8")


def test_synthetic_bundle_generation_differs_across_seeds(tmp_path):
    params_a = cc.SyntheticDataParams(n_train=80, shape=_test_shape(10), seed=1)
    params_b = cc.SyntheticDataParams(n_train=80, shape=_test_shape(10), seed=2)
    cc.generate_synthetic_bundle(tmp_path / "a", params_a)
    cc.generate_synthetic_bundle(tmp_path / "b", params_b)

    frame_a = pd.read_parquet(tmp_path / "a" / "train.parquet")
    frame_b = pd.read_parquet(tmp_path / "b" / "train.parquet")
    assert not frame_a.equals(frame_b)


def test_t0_component_consistency_flags_more_than_ten_percent_gap():
    consistent = cc.t0_component_consistency(10.0, 10.9)
    inconsistent = cc.t0_component_consistency(10.0, 12.0)

    assert consistent["exceeds_tolerance"] is False
    assert inconsistent["exceeds_tolerance"] is True
    assert inconsistent["relative_difference"] == pytest.approx(2.0 / 12.0)


def test_payload_lists_every_fit_below_the_r2_explanation_threshold():
    low = cc.PowerLawFit(0.0, 1.0, 1.0, 0.79, (0.0, 0.0), 3)
    acceptable = cc.PowerLawFit(0.0, 1.0, 1.0, 0.80, (0.0, 0.0), 3)
    payload = cc.build_calibration_payload(
        synthetic_params=cc.SyntheticDataParams(n_train=10, shape=_test_shape(4), seed=0),
        synthetic_stats={"n_expanded_predictors": 8, "observed_missingness": {}},
        t0_seconds={}, fit_cost={"ridge": low, "ols": acceptable}, preprocess_cost={},
        peak_rss={}, validation=[], censored=[], raw_measurements=[],
        thread_env_report={"ok": True, "values": {}},
    )

    assert payload["fit_quality"]["models_below_r2_threshold"] == [
        {"model": "ridge", "r2": 0.79}
    ]


def test_parallel_efficiency_measurement_uses_injected_clock_without_real_workers(monkeypatch):
    session = SimpleNamespace(schema_path=Path("synthetic-schema.json"), outcome="y")
    requests = tuple(cc.CellMeasurementRequest("ols", 10, 5) for _ in range(8))
    observed_jobs: list[int] = []
    clock_values = iter((0.0, 16.0, 20.0, 24.0))

    def _fake_run(worker_args, *, n_jobs):
        assert len(worker_args) == 8
        observed_jobs.append(n_jobs)

    monkeypatch.setattr(cc, "_run_parallel_efficiency_cells", _fake_run)
    measurement = cc.measure_parallel_efficiency(
        session, requests, n_reps=1, clock=lambda: next(clock_values)
    )

    assert observed_jobs == [1, 8]
    assert measurement["t1_seconds"]["median"] == 16.0
    assert measurement["t8_seconds"]["median"] == 4.0
    assert measurement["eta"] == pytest.approx(0.5)
    assert measurement["worker_start_method"] == "spawn"


def test_synthetic_observed_missingness_exposes_an_all_integer_zero_missing_panel(tmp_path):
    shape = _test_shape(8, onehot_sources=4, onehot_dtype="int64", continuous_dtype="int64")
    _, stats = cc.generate_synthetic_bundle(
        tmp_path / "integer-panel", cc.SyntheticDataParams(n_train=30, shape=shape, seed=4)
    )

    observed = stats["observed_missingness"]
    assert observed["expanded_columns_with_nan_fraction"] == 0.0
    assert observed["missing_cells_fraction"] == 0.0


def test_small_calibration_collects_fit_telemetry_from_real_fits(tmp_path):
    params = cc.SyntheticDataParams(n_train=100, shape=_test_shape(8), seed=11)
    schema_path, stats = cc.generate_synthetic_bundle(tmp_path / "bundle", params)
    session = cc.build_session(schema_path, "y", seed=0)
    # Test-only reduced settings retain each production estimator path while
    # making this a small telemetry smoke test rather than a calibration run.
    session.model_params["xgboost"].update(max_rounds=5, cv_folds=2)
    session.model_params["shallow_neural_network"].update(
        hidden_layer_sizes=[3], max_iter=5, n_alphas=2, max_cv_folds=2
    )
    measurements = []
    try:
        for model_name in ("lasso", "ridge", "xgboost", "shallow_neural_network"):
            measurement = cc._measure_one_cell_in_process(
                session, model_name=model_name, n=20, k=4, seed=0, draw=0, max_seconds=60
            )
            measurement.stage = "A"
            measurements.append(measurement)
    finally:
        cc.close_session(session)

    payload = cc.build_calibration_payload(
        synthetic_params=params,
        synthetic_stats=stats,
        t0_seconds={}, fit_cost={}, preprocess_cost={}, peak_rss={}, validation=[], censored=[],
        raw_measurements=measurements, thread_env_report={"ok": True, "values": {}},
    )
    telemetry = payload["telemetry"]

    assert telemetry["status"] == "collected"
    assert telemetry["fields_with_observations"] == {
        "converged": True, "best_rounds": True, "solver": True
    }
    assert all(
        {"converged", "best_rounds", "solver"}.issubset(row)
        for row in payload["raw_measurements"]
    )


# ---------------------------------------------------------------------------
# 3. Censoring path
# ---------------------------------------------------------------------------


def test_measure_one_cell_records_censoring_and_excludes_from_regression(
    tmp_path, monkeypatch
):
    params = cc.SyntheticDataParams(n_train=100, shape=_test_shape(10), seed=0)
    schema_path, _ = cc.generate_synthetic_bundle(tmp_path / "bundle", params)
    session = cc.build_session(schema_path, "y", seed=0)

    def _always_times_out(**kwargs):
        import time

        time.sleep(1.0)
        return {"predictions": None, "fit_seconds": 1.0, "best_rounds": None, "converged": True, "peak_rss_bytes": 0}

    monkeypatch.setattr(cc, "_fit_predict_model_cell", _always_times_out)

    # The public entry point is a fresh spawn process.  Exercise the budget
    # implementation directly here; monkeypatches do not cross a spawn
    # boundary (which is exactly the isolation the calibration needs).
    with pytest.raises(cc.MeasurementCensored):
        cc._measure_one_cell_in_process(
            session,
            model_name="ols",
            n=10,
            k=5,
            seed=0,
            draw=0,
            max_seconds=0.05,
        )

    monkeypatch.setattr(
        cc, "measure_one_cell",
        lambda *args, **kwargs: (_ for _ in ()).throw(cc.MeasurementCensored("test")),
    )
    raw, censored = cc.run_stage_a(
        session,
        n_grid=(10,),
        k_grid=(5,),
        n_reps=1,
        max_seconds=0.05,
        models=("ols",),
    )
    assert raw == []
    assert len(censored) == 1
    assert censored[0] == {
        "n": 10,
        "k": 5,
        "model": "ols",
        "rep": 0,
        "max_seconds": 0.05,
        "stage": "A",
    }

    fits = cc.fit_all_models(raw, models=("ols",))
    assert fits["ols"] is None

    payload = cc.build_calibration_payload(
        synthetic_params=params,
        synthetic_stats={"n_expanded_predictors": 10},
        t0_seconds={},
        fit_cost=fits,
        preprocess_cost={},
        peak_rss={},
        validation=[],
        censored=censored,
        raw_measurements=raw,
        thread_env_report={"ok": True, "values": {}},
    )
    assert payload["censored"] == censored
    assert payload["raw_measurements"] == []


# ---------------------------------------------------------------------------
# 4. Calibration file round-trip
# ---------------------------------------------------------------------------


def test_calibration_file_round_trip_recomputes_fit_cost(tmp_path):
    rng = np.random.default_rng(7)
    raw = []
    true_fit = {}
    for model_name, (c, a, b) in {
        "ols": (1.0, 0.5, 0.5),
        "ridge": (2.0, 0.7, 0.2),
    }.items():
        true_fit[model_name] = (c, a, b)
        for n in (10, 100, 1000):
            for k in (10, 100):
                t = c * (k**a) * (n**b)
                raw.append(
                    cc.RawMeasurement(
                        model=model_name,
                        n=n,
                        k=k,
                        rep=0,
                        fit_seconds=t,
                        preprocess_seconds=0.01,
                        preprocess_mode="imputed",
                        peak_rss_bytes=1_000_000,
                        stage="A",
                    )
                )

    fit_cost = cc.fit_all_models(raw, models=("ols", "ridge"))
    raw[0].memory_scope_suspect = True
    raw[0].preprocess_vectorized = True
    payload = cc.build_calibration_payload(
        synthetic_params=cc.SyntheticDataParams(n_train=10, shape=_test_shape(10), seed=0),
        synthetic_stats={
            "n_expanded_predictors": 10,
            "observed_missingness": {
                "expanded_columns_with_nan_fraction": 0.0,
                "missing_cells_fraction": 0.0,
            },
        },
        t0_seconds={
            "import": {"median": 1.18, "min": 1.156, "max": 1.234, "n_reps": 5},
            "load": {"median": 0.1, "min": 0.09, "max": 0.11, "n_reps": 5},
            "split": {"median": 0.01, "min": 0.009, "max": 0.011, "n_reps": 5},
            "orders": {"median": 0.001, "min": 0.0009, "max": 0.0011, "n_reps": 5},
            "total": {"median": 1.3, "n_reps": 5},
        },
        fit_cost=fit_cost,
        preprocess_cost={},
        peak_rss={},
        validation=[],
        censored=[],
        raw_measurements=raw,
        thread_env_report={"ok": True, "values": {name: "1" for name in cc.THREAD_ENV_VARS}},
        task_cgroup_peak_n_jobs_8=cc.MemoryPeak(
            bytes=987_654_321,
            method=cc.MEMORY_METHOD_CGROUP_PEAK,
            sampling_interval_seconds=None,
        ),
    )

    out_path = cc.write_calibration_file(payload, tmp_path, date="2026-07-31")
    assert out_path.name == "cost_model_2026-07-31.json"

    reloaded = cc.read_calibration_file(out_path)
    assert reloaded["format_version"] == cc.FORMAT_VERSION

    required_top_level = {
        "format_version",
        "created_at_utc",
        "git_commit",
        "environment",
        "synthetic_data",
        "t0_seconds",
        "fit_cost",
        "preprocess_cost",
        "peak_rss_bytes",
        "memory_measurement",
        "validation",
        "censored",
        "raw_measurements",
    }
    assert required_top_level.issubset(reloaded.keys())
    assert reloaded["t0_seconds"]["import"]["median"] == 1.18
    assert reloaded["memory_measurement"]["cell_memory_scope_suspect_count"] == 1
    assert reloaded["raw_measurements"][0]["memory_scope_suspect"] is True
    assert reloaded["raw_measurements"][0]["preprocess_vectorized"] is True
    task_peak = reloaded["memory_measurement"]["task_cgroup_peak_n_jobs_8"]
    assert task_peak == {
        "status": "measured",
        "bytes": 987_654_321,
        "method": cc.MEMORY_METHOD_CGROUP_PEAK,
        "sampling_interval_seconds": None,
        "sampling_interval_max_seconds": None,
        "samples": 0,
    }

    recomputed = cc.recompute_fit_cost_from_raw(
        reloaded["raw_measurements"], models=("ols", "ridge")
    )
    for model_name, (c, a, b) in true_fit.items():
        fit = recomputed[model_name]
        assert fit is not None
        assert abs(fit.a - a) < 1e-6
        assert abs(fit.b - b) < 1e-6
        assert abs(fit.log_c - np.log(c)) < 1e-6
        assert abs(fit.log_c - reloaded["fit_cost"][model_name]["log_c"]) < 1e-9
        assert abs(fit.a - reloaded["fit_cost"][model_name]["a"]) < 1e-9
        assert abs(fit.b - reloaded["fit_cost"][model_name]["b"]) < 1e-9


# ---------------------------------------------------------------------------
# 4A. Round-2 memory measurement: independent spawn RSS plus cgroup fallbacks
# ---------------------------------------------------------------------------


def test_fresh_spawn_process_peak_does_not_inherit_parent_high_water():
    # This reproduces the rejected shape in reverse: the parent runs both
    # calls, but each reported high-water mark belongs only to its own spawn
    # child.  The small second workload must not report the first workload's
    # allocation as would RUSAGE_SELF in a reused harness process.
    large = cc.measure_process_peak_rss(_allocate_bytes_for_peak, 48 * 1024 * 1024)
    small = cc.measure_process_peak_rss(_allocate_bytes_for_peak, 1 * 1024 * 1024)
    assert large > 0
    assert small > 0
    assert small < large * 0.85


def test_cell_measurement_records_distinct_spawn_and_whole_tree_numbers(tmp_path, monkeypatch):
    params = cc.SyntheticDataParams(n_train=80, shape=_test_shape(8), seed=0)
    schema_path, _ = cc.generate_synthetic_bundle(tmp_path / "bundle", params)
    session = cc.build_session(schema_path, "y", seed=0)
    # The desktop sandbox intentionally denies ``ps``.  Mock the platform
    # primitive so this test covers the collection *scope* without depending
    # on a cgroup-enabled Linux host or sandbox process inspection rights.
    monkeypatch.setattr(cc, "_process_tree_rss_bytes", lambda pid: 321_000_000)
    measurement = cc.measure_one_cell(
        session, model_name="ols", n=10, k=5, seed=0, draw=0, max_seconds=60
    )
    assert measurement.process_peak_rss_bytes > 0
    assert measurement.cell_cgroup_peak_bytes > 0
    assert measurement.peak_rss_bytes == measurement.cell_cgroup_peak_bytes
    assert measurement.cell_memory_method in {
        cc.MEMORY_METHOD_CGROUP_PEAK,
        cc.MEMORY_METHOD_CGROUP_CURRENT,
        cc.MEMORY_METHOD_PROCESS_TREE,
    }
    if measurement.cell_memory_method == cc.MEMORY_METHOD_CGROUP_PEAK:
        assert measurement.cell_memory_sampling_interval_seconds is None
    else:
        assert measurement.cell_memory_sampling_interval_seconds is not None
        assert measurement.cell_memory_sampling_interval_seconds >= cc.MEMORY_SAMPLE_INTERVAL_SECONDS
        assert measurement.cell_memory_sampling_interval_max_seconds is not None
        assert measurement.cell_memory_sampling_interval_max_seconds >= (
            measurement.cell_memory_sampling_interval_seconds
        )


def test_macos_fallback_is_explicit_tree_rss_with_sampling_interval(tmp_path, monkeypatch):
    params = cc.SyntheticDataParams(n_train=80, shape=_test_shape(8), seed=1)
    schema_path, _ = cc.generate_synthetic_bundle(tmp_path / "bundle", params)
    session = cc.build_session(schema_path, "y", seed=0)
    # This is the normal macOS state, forced so the assertion is portable on
    # Linux CI as well: no cgroup may be claimed when one is unavailable.
    monkeypatch.setattr(cc, "_create_measurement_cgroup", lambda: None)
    monkeypatch.setattr(cc, "_process_tree_rss_bytes", lambda pid: 123_000_000)
    measurement = cc.measure_one_cell(
        session, model_name="ols", n=10, k=5, seed=0, draw=0, max_seconds=60
    )
    assert measurement.cell_memory_method == cc.MEMORY_METHOD_PROCESS_TREE
    assert measurement.cell_memory_sampling_interval_seconds is not None
    assert measurement.cell_memory_sampling_interval_max_seconds is not None
    assert measurement.cell_cgroup_peak_bytes > 0


def test_cell_spawn_target_observes_no_numpy_or_pandas_before_join(tmp_path):
    observation = tmp_path / "cell-import-state.txt"
    process = mp.get_context("spawn").Process(
        target=cell_worker_target,
        args=(None, str(tmp_path / "unused-result.json"), "schema.yaml", "y", 0,
              "ols", 10, 5, 0, 60.0, str(observation), 1),
    )
    process.start()
    process.join()
    assert process.exitcode == 0
    assert observation.read_text(encoding="ascii") == "numpy=0\npandas=0\n"


def test_task_spawn_target_observes_no_numpy_or_pandas_before_join(tmp_path):
    observation = tmp_path / "task-import-state.txt"
    process = mp.get_context("spawn").Process(
        target=task_cell_worker_target,
        args=(None, "schema.yaml", "y", "ols", 10, 5, 0, 0, 60.0,
              str(observation), 1),
    )
    process.start()
    process.join()
    assert process.exitcode == 0
    assert observation.read_text(encoding="ascii") == "numpy=0\npandas=0\n"


def test_memory_scope_suspect_marks_only_an_underreported_tree_peak():
    assert cc._memory_scope_suspect(99, 100) is True
    assert cc._memory_scope_suspect(100, 100) is False
    assert cc._memory_scope_suspect(101, 100) is False


def test_linux_tree_sampler_reads_proc_without_invoking_ps(tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    for pid, ppid, pages in ((100, 1, 3), (101, 100, 5), (200, 1, 99)):
        entry = proc / str(pid)
        entry.mkdir(parents=True)
        (entry / "stat").write_text(f"{pid} (worker name) S {ppid} 0 0", encoding="utf-8")
        (entry / "statm").write_text(f"10 {pages} 0 0 0 0 0", encoding="ascii")
    monkeypatch.setattr(cc.os, "sysconf", lambda name: 4096)
    assert cc._linux_process_tree_rss_bytes(100, proc) == (3 + 5) * 4096


def test_task_peak_uses_eight_spawn_workers_and_returns_its_own_scope(tmp_path, monkeypatch):
    monkeypatch.setattr(cc, "_create_measurement_cgroup", lambda: None)
    monkeypatch.setattr(cc, "_process_tree_rss_bytes", lambda pid: 2_000_000)
    temporary = [tmp_path / f"task-import-state-{index}" for index in range(8)]
    peak = cc._measure_task_peak_n_jobs_8(
        [("schema.yaml", "y", "ols", 10, 5, 0, 0, 60.0)] * 8,
        observation_paths=[str(path) for path in temporary],
        probe_only=1,
    )
    assert peak.bytes > 0
    assert peak.method in {
        cc.MEMORY_METHOD_CGROUP_PEAK,
        cc.MEMORY_METHOD_CGROUP_CURRENT,
        cc.MEMORY_METHOD_PROCESS_TREE,
    }
    if peak.method == cc.MEMORY_METHOD_CGROUP_PEAK:
        assert peak.sampling_interval_seconds is None
    else:
        assert peak.sampling_interval_seconds is not None
        assert peak.sampling_interval_max_seconds is not None
    for path in temporary:
        assert path.read_text(encoding="ascii") == "numpy=0\npandas=0\n"


def test_task_memory_cli_requires_an_explicit_eight_cell_task_shape():
    specs = ",".join(["ols:10:5"] * 8)
    requests = cc.parse_task_memory_cells(specs, max_seconds=12.5)
    assert len(requests) == 8
    assert requests[0] == cc.CellMeasurementRequest("ols", 10, 5, max_seconds=12.5)
    with pytest.raises(ValueError, match="exactly eight"):
        cc.parse_task_memory_cells("ols:10:5", max_seconds=12.5)


@pytest.mark.parametrize("old_version", (1, 2, 3, 4))
def test_calibration_reader_rejects_old_calibration_formats(tmp_path, old_version):
    old_path = tmp_path / f"old-v{old_version}.json"
    old_path.write_text(json.dumps({"format_version": old_version}), encoding="utf-8")
    with pytest.raises(ValueError, match="expected 5"):
        cc.read_calibration_file(old_path)


# ---------------------------------------------------------------------------
# 5. Environment-variable assertion
# ---------------------------------------------------------------------------


def test_enforce_thread_env_refuses_to_start_when_not_production_set():
    bad_env = {name: "4" for name in cc.THREAD_ENV_VARS}
    with pytest.raises(RuntimeError):
        cc.enforce_thread_env(strict=True, env=bad_env)


def test_enforce_thread_env_warns_and_records_when_not_strict(capsys):
    bad_env = dict.fromkeys(cc.THREAD_ENV_VARS)  # unset
    report = cc.enforce_thread_env(strict=False, env=bad_env)
    assert report["ok"] is False
    captured = capsys.readouterr()
    assert "WARNING" in captured.err


def test_check_thread_env_ok_when_all_set_to_one():
    good_env = {name: "1" for name in cc.THREAD_ENV_VARS}
    report = cc.check_thread_env(good_env)
    assert report["ok"] is True
    # Should not raise.
    cc.enforce_thread_env(strict=True, env=good_env)


# ---------------------------------------------------------------------------
# 6. No-private-data-access
# ---------------------------------------------------------------------------


def _write_path_guard_schema(
    schema_dir: Path,
    *,
    table: str,
    extra: dict[str, object] | None = None,
) -> Path:
    schema_dir.mkdir(parents=True)
    document: dict[str, object] = {"table": table, "task": "regression"}
    if extra:
        document.update(extra)
    schema_path = schema_dir / "panel.json"
    schema_path.write_text(json.dumps(document), encoding="utf-8")
    return schema_path


def test_build_session_rejects_cross_directory_data_before_load(tmp_path, monkeypatch):
    schema_dir = tmp_path / "unseen_cohort" / "schema"
    data_path = tmp_path / "unseen_cohort" / "observations" / "table.parquet"
    schema_path = _write_path_guard_schema(
        schema_dir, table="../observations/table.parquet"
    )
    monkeypatch.setattr(cc, "_default_calibration_read_roots", lambda: (schema_dir.resolve(),))
    load_called = False

    def _unexpected_load(*args, **kwargs):
        nonlocal load_called
        load_called = True
        raise AssertionError("load_input must not run before the path guard")

    monkeypatch.setattr(cc, "load_input", _unexpected_load)
    with pytest.raises(cc.PrivateDataAccessError) as caught:
        cc.build_session(schema_path, "y")

    assert load_called is False
    assert str(data_path.resolve()) in str(caught.value)
    assert str(schema_dir.resolve()) in str(caught.value)


def test_guard_is_dataset_count_agnostic(tmp_path, monkeypatch):
    schema_dir = tmp_path / "new_dataset_number_n" / "metadata"
    schema_path = _write_path_guard_schema(
        schema_dir, table="../restricted/data.arrow"
    )
    monkeypatch.setattr(cc, "_default_calibration_read_roots", lambda: (schema_dir.resolve(),))
    with pytest.raises(cc.PrivateDataAccessError, match="data.arrow"):
        cc.guard_not_private_data(schema_path)


def test_guard_discovers_future_path_field_without_field_name_changes(tmp_path, monkeypatch):
    schema_dir = tmp_path / "metadata"
    schema_path = _write_path_guard_schema(
        schema_dir,
        table="local.parquet",
        extra={"future_auxiliary_binary": "../restricted/future_payload.bin"},
    )
    monkeypatch.setattr(cc, "_default_calibration_read_roots", lambda: (schema_dir.resolve(),))
    with pytest.raises(cc.PrivateDataAccessError, match="future_payload.bin"):
        cc.guard_not_private_data(schema_path)


def test_guard_allows_tmp_bundle_and_explicit_additional_root(tmp_path, monkeypatch):
    tmp_schema = _write_path_guard_schema(tmp_path / "tmp-bundle", table="table.parquet")
    assert cc.guard_not_private_data(tmp_schema) == tmp_schema.resolve()

    explicit_root = tmp_path / "explicit-bundle"
    explicit_schema = _write_path_guard_schema(explicit_root, table="table.parquet")
    monkeypatch.setattr(cc, "_default_calibration_read_roots", lambda: ())
    assert cc.guard_not_private_data(
        explicit_schema, allowed_roots=(explicit_root,)
    ) == explicit_schema.resolve()


def test_build_session_never_opens_private_data_paths(tmp_path, monkeypatch):
    opened_paths: list[str] = []
    real_read_parquet = pd.read_parquet
    real_read_csv = pd.read_csv

    def _tracking_read_parquet(path, *args, **kwargs):
        opened_paths.append(str(path))
        return real_read_parquet(path, *args, **kwargs)

    def _tracking_read_csv(path, *args, **kwargs):
        opened_paths.append(str(path))
        return real_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", _tracking_read_parquet)
    monkeypatch.setattr(pd, "read_csv", _tracking_read_csv)

    params = cc.SyntheticDataParams(n_train=60, shape=_test_shape(8), seed=0)
    schema_path, _ = cc.generate_synthetic_bundle(tmp_path / "bundle", params)
    cc.build_session(schema_path, "y", seed=0)

    bundle_root = schema_path.parent.resolve()
    assert opened_paths
    assert all(cc._is_within(Path(path).resolve(), bundle_root) for path in opened_paths)


def _write_feature_universe_schema(
    path: Path,
    source_specs: list[tuple[str, tuple[str | None, ...]]],
) -> Path:
    sources = []
    for source_order, (unit_type, dtypes) in enumerate(source_specs):
        sources.append(
            {
                "source": f"source_{source_order}",
                "source_order": source_order,
                "unit_type": unit_type,
                "features": [
                    {
                        "feature": f"f_{source_order}_{feature_order}",
                        **({"dtype": dtype} if dtype is not None else {}),
                    }
                    for feature_order, dtype in enumerate(dtypes)
                ],
            }
        )
    path.write_text(json.dumps({"predictors": [], "sources": sources}), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("source_specs", "expected_counts", "expected_onehot_fraction"),
    [
        (
            [("continuous", ("float64",)), ("onehot_group", ("float64", "float64"))],
            {"float64": 3},
            0.5,
        ),
        (
            [
                ("continuous", ("float32",)),
                ("onehot_group", ("int64", "int64", "int64")),
                ("onehot_group", ("int64", "int64")),
            ],
            {"float32": 1, "int64": 5},
            2 / 3,
        ),
        (
            [("continuous", ("float32",)), ("continuous", ("float64",)), ("onehot_group", ("int16", "int16"))],
            {"float32": 1, "float64": 1, "int16": 2},
            1 / 3,
        ),
    ],
)
def test_shape_from_schema_and_synthetic_bundle_preserve_parameterized_dtypes(
    tmp_path, source_specs, expected_counts, expected_onehot_fraction
):
    schema = _write_feature_universe_schema(tmp_path / "shape.feature_universe.json", source_specs)
    shape = cc.shape_from_schema(schema)

    assert shape.n_sources == len(source_specs)
    assert shape.expanded_dtype_counts == expected_counts
    assert shape.onehot_source_fraction == pytest.approx(expected_onehot_fraction)
    params = cc.SyntheticDataParams(n_train=20, shape=shape, seed=4)
    cc.generate_synthetic_bundle(tmp_path / "bundle", params)
    generated = pd.read_parquet(tmp_path / "bundle" / "train.parquet")
    assert {
        str(dtype): sum(generated[column].dtype == dtype for column in generated.columns if column != "y")
        for dtype in generated.drop(columns="y").dtypes.unique()
    } == expected_counts


def test_shape_without_dtype_metadata_fails_closed(tmp_path):
    schema = _write_feature_universe_schema(
        tmp_path / "missing-dtypes.feature_universe.json",
        [("continuous", (None,)), ("onehot_group", (None, None))],
    )

    with pytest.raises(ValueError, match="declares dtype for 0/3"):
        cc.shape_from_schema(schema)


def test_external_dtype_profile_drives_generated_panel_and_is_recorded(tmp_path):
    schema = _write_feature_universe_schema(
        tmp_path / "profiled.feature_universe.json",
        [("continuous", (None,)), ("continuous", (None,)), ("onehot_group", (None, None, None))],
    )
    profile_path = tmp_path / "dtype-profile.json"
    profile_path.write_text(json.dumps({"int64": 3, "float64": 2}), encoding="utf-8")
    profile = cc.read_feature_dtype_profile(profile_path)
    shape = cc.shape_from_schema(
        schema,
        feature_dtype_profile=profile,
        dtype_profile_path=profile_path,
    )
    params = cc.SyntheticDataParams(n_train=20, shape=shape, seed=8)
    _, stats = cc.generate_synthetic_bundle(tmp_path / "bundle", params)
    generated = pd.read_parquet(tmp_path / "bundle" / "train.parquet").drop(columns="y")

    assert {str(dtype): int((generated.dtypes == dtype).sum()) for dtype in generated.dtypes.unique()} == {
        "float64": 2,
        "int64": 3,
    }
    description = stats["shape"]
    assert description["dtype_source"] == "external_profile"
    assert description["dtype_metadata_coverage"] == "0/5"
    assert description["dtype_profile_path"] == str(profile_path.resolve())


@pytest.mark.parametrize(
    ("source_specs", "profile", "expected_dtypes"),
    [
        (
            [("onehot_group", (None, None, None)), ("continuous", (None,)), ("continuous", (None,))],
            {"continuous": {"float64": 2}, "onehot_group": {"int64": 3}},
            [("int64", "int64", "int64"), ("float64",), ("float64",)],
        ),
        (
            [("continuous", (None,)), ("onehot_group", (None, None)), ("onehot_group", (None, None))],
            {"continuous": {"float32": 1}, "onehot_group": {"int16": 4}},
            [("float32",), ("int16", "int16"), ("int16", "int16")],
        ),
    ],
)
def test_unit_type_grouped_dtype_profile_is_schema_order_independent(
    tmp_path, source_specs, profile, expected_dtypes
):
    schema = _write_feature_universe_schema(tmp_path / "grouped.json", source_specs)
    shape = cc.shape_from_schema(schema, feature_dtype_profile=profile)

    assert [source.feature_dtypes for source in shape.sources] == expected_dtypes
    assert shape.dtype_profile_allocation_rule == (
        "unit_type_grouped_dtype_name_sorted_then_schema_source_order_within_unit_type"
    )


def test_unit_type_dtype_profile_requires_complete_structure_coverage(tmp_path):
    schema = _write_feature_universe_schema(
        tmp_path / "incomplete-grouped.json",
        [("continuous", (None,)), ("onehot_group", (None, None))],
    )

    with pytest.raises(ValueError, match="exactly match schema unit types"):
        cc.shape_from_schema(schema, feature_dtype_profile={"continuous": {"float64": 1}})


def test_json_distinguishes_declared_defaulted_and_partial_dtype_sources(tmp_path):
    schemas = {
        "declared": _write_feature_universe_schema(
            tmp_path / "declared.json", [("onehot_group", ("float64", "float64"))]
        ),
        "defaulted_float64": _write_feature_universe_schema(
            tmp_path / "defaulted.json", [("onehot_group", (None, None))]
        ),
        "declared_with_assumed_float64": _write_feature_universe_schema(
            tmp_path / "partial.json", [("onehot_group", ("int64", None))]
        ),
    }
    expected_coverage = {
        "declared": "2/2",
        "defaulted_float64": "0/2",
        "declared_with_assumed_float64": "1/2",
    }
    for expected_source, schema in schemas.items():
        shape = cc.shape_from_schema(
            schema,
            assume_feature_dtype=(None if expected_source == "declared" else "float64"),
        )
        payload = cc.build_calibration_payload(
            synthetic_params=cc.SyntheticDataParams(n_train=10, shape=shape, seed=0),
            synthetic_stats={"n_expanded_predictors": 2},
            t0_seconds={},
            fit_cost={},
            preprocess_cost={},
            peak_rss={},
            validation=[],
            censored=[],
            raw_measurements=[],
            thread_env_report={"ok": True, "values": {}},
        )
        recorded = payload["synthetic_data"]["panel_shape"]
        assert recorded["dtype_source"] == expected_source
        assert recorded["dtype_metadata_coverage"] == expected_coverage[expected_source]


def test_k_grid_uses_each_schema_shape_maximum_and_has_interior_probes(tmp_path):
    small = cc.shape_from_schema(
        _write_feature_universe_schema(
            tmp_path / "small.feature_universe.json",
            [("continuous", ("float64",))] * 40,
        )
    )
    large = cc.shape_from_schema(
        _write_feature_universe_schema(
            tmp_path / "large.feature_universe.json",
            [("continuous", ("float64",))] * 4000,
        )
    )
    small_grid = cc.k_grid_from_shape(small, base_anchors=(10,), intermediate_points=2)
    large_grid = cc.k_grid_from_shape(large, base_anchors=(10, 100, 1000), intermediate_points=2)

    assert small_grid[-1] == 40
    assert large_grid[-1] == 4000
    assert sum(10 < point < 40 for point in small_grid) == 2
    assert sum(1000 < point < 4000 for point in large_grid) == 2


def test_shape_from_existing_structure_schema_is_self_consistent():
    schema = next((cc.repo_root() / "SMR" / "schema").glob("*.feature_universe.json"))
    document = json.loads(schema.read_text(encoding="utf-8"))
    shape = cc.shape_from_schema(schema, assume_feature_dtype="float64")

    assert shape.n_sources == len(document["sources"])
    assert shape.dtype_source == "defaulted_float64"
    assert shape.dtype_metadata_coverage == f"0/{len(document['predictors'])}"
    assert sum(shape.expanded_dtype_counts.values()) == sum(
        len(source["features"]) for source in document["sources"]
    )


def test_shape_export_passes_only_its_schema_directory_to_private_data_guard(tmp_path, monkeypatch):
    schema = _write_feature_universe_schema(
        tmp_path / "shape.feature_universe.json", [("continuous", ("float64",))]
    )
    calls = []

    def _guard(path, *, allowed_roots=()):
        calls.append((Path(path), tuple(Path(root) for root in allowed_roots)))
        return Path(path).resolve()

    monkeypatch.setattr(cc, "guard_not_private_data", _guard)
    cc.shape_from_schema(schema)

    assert calls == [(schema.resolve(), (schema.parent.resolve(),))]


def test_cli_requires_explicit_shape_schema():
    with pytest.raises(SystemExit):
        cc.parse_args(["--n-train", "20"])


def test_cli_accepts_external_dtype_profile_and_rejects_conflicting_assumption(tmp_path):
    profile = tmp_path / "profile.json"
    args = cc.parse_args(
        [
            "--shape-schema",
            "shape.json",
            "--n-train",
            "20",
            "--feature-dtype-profile",
            str(profile),
        ]
    )
    assert args.feature_dtype_profile == profile
    with pytest.raises(SystemExit):
        cc.parse_args(
            [
                "--shape-schema",
                "shape.json",
                "--n-train",
                "20",
                "--feature-dtype-profile",
                str(profile),
                "--assume-feature-dtype",
                "float64",
            ]
        )


# ---------------------------------------------------------------------------
# Round-1 review regression tests (F1: SUBPROCESS_MODELS must track the
# engine's own SERIAL_OUTER_MODELS, never a hand-copied set; F2: those models
# must be measured through the real isolated subprocess so peak RSS reflects
# the child process that actually does the fitting).
# ---------------------------------------------------------------------------


def test_subprocess_model_set_tracks_the_engine():
    from aleatoric_nk_grid.experiment import SERIAL_OUTER_MODELS

    assert cc.SUBPROCESS_MODELS == SERIAL_OUTER_MODELS
    # Lock in the concrete membership too, so a change to the engine's set
    # shows up here as a meaningful diff, not just an opaque equality change.
    assert cc.SUBPROCESS_MODELS == frozenset({"lightgbm", "super_learner"})


def test_measure_one_cell_routes_subprocess_models_through_isolated_runner(
    tmp_path, monkeypatch
):
    params = cc.SyntheticDataParams(n_train=100, shape=_test_shape(10), seed=0)
    schema_path, _ = cc.generate_synthetic_bundle(tmp_path / "bundle", params)
    session = cc.build_session(schema_path, "y", seed=0)

    calls: dict[str, list[str]] = {"native": [], "inline": []}

    def _fake_native(runner, *, fit_arguments, on_native_crash, on_native_timeout):
        assert runner is session.native_runner
        calls["native"].append(fit_arguments["model_name"])
        return {
            "predictions": np.zeros(len(fit_arguments["X_test"])),
            "fit_seconds": 0.01,
            "best_rounds": None,
            "converged": True,
            "peak_rss_bytes": 123_456_789,
        }

    def _fake_inline(**kwargs):
        calls["inline"].append(kwargs["model_name"])
        return {
            "predictions": np.zeros(len(kwargs["X_test"])),
            "fit_seconds": 0.01,
            "best_rounds": None,
            "converged": True,
            "peak_rss_bytes": 111,
        }

    monkeypatch.setattr(cc, "_run_native_model_cell_locked", _fake_native)
    monkeypatch.setattr(cc, "_fit_predict_model_cell", _fake_inline)

    for model in ("lightgbm", "super_learner"):
        measurement = cc._measure_one_cell_in_process(
            session, model_name=model, n=10, k=5, seed=0, draw=0, max_seconds=60
        )
        assert measurement.fit_seconds == 0.01

    # xgboost is NOT in SUBPROCESS_MODELS (that was F1's bug: xgboost doesn't
    # go through the isolated subprocess in production) so it must take the
    # inline path, same as any other non-native model.
    for model in ("ols", "xgboost"):
        measurement = cc._measure_one_cell_in_process(
            session, model_name=model, n=10, k=5, seed=0, draw=0, max_seconds=60
        )
        assert measurement.fit_seconds == 0.01

    assert set(calls["native"]) == {"lightgbm", "super_learner"}
    assert set(calls["inline"]) == {"ols", "xgboost"}
    assert session.native_runner is not None

    cc.close_session(session)
    assert session.native_runner is None


def test_measure_one_cell_discards_native_runner_on_failure(tmp_path, monkeypatch):
    params = cc.SyntheticDataParams(n_train=60, shape=_test_shape(8), seed=0)
    schema_path, _ = cc.generate_synthetic_bundle(tmp_path / "bundle", params)
    session = cc.build_session(schema_path, "y", seed=0)

    def _boom(runner, *, fit_arguments, on_native_crash, on_native_timeout):
        raise RuntimeError("simulated native subprocess crash")

    monkeypatch.setattr(cc, "_run_native_model_cell_locked", _boom)

    with pytest.raises(RuntimeError, match="simulated native subprocess crash"):
        cc._measure_one_cell_in_process(
            session, model_name="lightgbm", n=10, k=5, seed=0, draw=0, max_seconds=60
        )

    # The runner must be discarded (not left in a possibly-corrupted,
    # mid-request state) so a subsequent call starts a fresh worker.
    assert session.native_runner is None


def test_synthetic_data_section_records_all_params_and_placeholder_flag(tmp_path):
    params = cc.SyntheticDataParams(
        n_train=50,
        shape=_test_shape(6, onehot_sources=3),
        seed=3,
        missing_rate_continuous=0.2,
        missing_rate_group=0.15,
    )
    _, stats = cc.generate_synthetic_bundle(tmp_path / "bundle", params)
    payload = cc.build_calibration_payload(
        synthetic_params=params,
        synthetic_stats=stats,
        t0_seconds={},
        fit_cost={},
        preprocess_cost={},
        peak_rss={},
        validation=[],
        censored=[],
        raw_measurements=[],
        thread_env_report={"ok": True, "values": {}},
    )
    synthetic_data = payload["synthetic_data"]
    assert synthetic_data["panel_shape"] == params.shape.as_dict()
    assert synthetic_data["missing_rate_continuous"] == 0.2
    assert synthetic_data["missing_rate_group"] == 0.15
    assert synthetic_data["outcome"] == params.outcome
    assert synthetic_data["observed_missingness"] == stats["observed_missingness"]
    assert payload["parallel_efficiency"] == {"status": "not_measured"}
    assert payload["telemetry"] == {
        "status": "not_measured",
        "by_model_k": [],
        "fields_with_observations": {
            "converged": False,
            "best_rounds": False,
            "solver": False,
        },
    }
    assert payload["fit_quality"]["models_below_r2_threshold"] == []
    assert payload["wall_clock_seconds"] == {"status": "not_measured"}
