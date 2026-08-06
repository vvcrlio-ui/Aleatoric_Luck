from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aleatoric_nk_grid import calibrate_cost as cc


def _allocate_bytes_for_peak(byte_count: int, seconds: float = 0.15) -> None:
    """Module-level so multiprocessing ``spawn`` can import it in a child."""

    payload = bytearray(byte_count)
    payload[::4096] = b"\x01" * len(payload[::4096])
    time.sleep(seconds)


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
    params = cc.SyntheticDataParams(n_train=120, n_sources=15, seed=42)
    schema_a, stats_a = cc.generate_synthetic_bundle(tmp_path / "a", params)
    schema_b, stats_b = cc.generate_synthetic_bundle(tmp_path / "b", params)

    frame_a = pd.read_parquet(tmp_path / "a" / "train.parquet")
    frame_b = pd.read_parquet(tmp_path / "b" / "train.parquet")

    pd.testing.assert_frame_equal(frame_a, frame_b, check_exact=True)
    assert stats_a == stats_b
    assert schema_a.read_text(encoding="utf-8") == schema_b.read_text(encoding="utf-8")


def test_synthetic_bundle_generation_differs_across_seeds(tmp_path):
    params_a = cc.SyntheticDataParams(n_train=80, n_sources=10, seed=1)
    params_b = cc.SyntheticDataParams(n_train=80, n_sources=10, seed=2)
    cc.generate_synthetic_bundle(tmp_path / "a", params_a)
    cc.generate_synthetic_bundle(tmp_path / "b", params_b)

    frame_a = pd.read_parquet(tmp_path / "a" / "train.parquet")
    frame_b = pd.read_parquet(tmp_path / "b" / "train.parquet")
    assert not frame_a.equals(frame_b)


# ---------------------------------------------------------------------------
# 3. Censoring path
# ---------------------------------------------------------------------------


def test_measure_one_cell_records_censoring_and_excludes_from_regression(
    tmp_path, monkeypatch
):
    params = cc.SyntheticDataParams(n_train=100, n_sources=10, seed=0)
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
    payload = cc.build_calibration_payload(
        synthetic_params=cc.SyntheticDataParams(n_train=10, n_sources=10, seed=0),
        synthetic_stats={"n_expanded_predictors": 10},
        t0_seconds={
            "import": cc.T_IMPORT_PREMEASURED,
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
    assert reloaded["t0_seconds"]["import"]["source"].startswith("pre-measured")
    assert reloaded["memory_measurement"]["cell_memory_scope_suspect_count"] == 1
    assert reloaded["raw_measurements"][0]["memory_scope_suspect"] is True
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
    params = cc.SyntheticDataParams(n_train=80, n_sources=8, seed=0)
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
    params = cc.SyntheticDataParams(n_train=80, n_sources=8, seed=1)
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


def test_cell_worker_joins_precreated_cgroup_before_building_session(tmp_path, monkeypatch):
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    cgroup_procs = cgroup / "cgroup.procs"
    cgroup_procs.touch()
    observed: list[str] = []

    class Connection:
        message: dict[str, object] | None = None

        def send(self, message):
            self.message = message

        def close(self):
            pass

    connection = Connection()

    def _build_session(*args, **kwargs):
        observed.append(cgroup_procs.read_text(encoding="ascii"))
        return object()

    monkeypatch.setattr(cc, "build_session", _build_session)
    monkeypatch.setattr(cc, "_measure_one_cell_in_process", lambda *args, **kwargs: cc.RawMeasurement(
        model="ols", n=10, k=5, rep=0, fit_seconds=0.0, preprocess_seconds=0.0,
        preprocess_mode="imputed", peak_rss_bytes=0, stage="",
    ))
    monkeypatch.setattr(cc, "close_session", lambda session: None)
    monkeypatch.setattr(cc, "_process_peak_rss_bytes", lambda: 123)

    cc._cell_worker(connection, str(cgroup), "schema.yaml", "y", 0, "ols", 10, 5, 0, 60)

    assert observed == [str(os.getpid())]
    assert connection.message is not None
    assert connection.message["cgroup_joined"] is True


def test_memory_scope_suspect_marks_only_an_underreported_tree_peak():
    assert cc._memory_scope_suspect(99, 100) is True
    assert cc._memory_scope_suspect(100, 100) is False
    assert cc._memory_scope_suspect(101, 100) is False


def test_task_worker_joins_precreated_cgroup_before_workload(tmp_path):
    cgroup = tmp_path / "task-cgroup"
    cgroup.mkdir()
    cgroup_procs = cgroup / "cgroup.procs"
    cgroup_procs.touch()
    observed: list[str] = []

    def _workload():
        observed.append(cgroup_procs.read_text(encoding="ascii"))

    cc._task_worker(str(cgroup), _workload, ())
    assert observed == [str(os.getpid())]


def test_linux_tree_sampler_reads_proc_without_invoking_ps(tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    for pid, ppid, pages in ((100, 1, 3), (101, 100, 5), (200, 1, 99)):
        entry = proc / str(pid)
        entry.mkdir(parents=True)
        (entry / "stat").write_text(f"{pid} (worker name) S {ppid} 0 0", encoding="utf-8")
        (entry / "statm").write_text(f"10 {pages} 0 0 0 0 0", encoding="ascii")
    monkeypatch.setattr(cc.os, "sysconf", lambda name: 4096)
    assert cc._linux_process_tree_rss_bytes(100, proc) == (3 + 5) * 4096


def test_task_peak_uses_eight_spawn_workers_and_returns_its_own_scope(monkeypatch):
    monkeypatch.setattr(cc, "_create_measurement_cgroup", lambda: None)
    monkeypatch.setattr(cc, "_process_tree_rss_bytes", lambda pid: 2_000_000)
    peak = cc._measure_task_peak_n_jobs_8(
        _allocate_bytes_for_peak,
        [(2 * 1024 * 1024, 0.2)] * 8,
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


def test_task_memory_cli_requires_an_explicit_eight_cell_task_shape():
    specs = ",".join(["ols:10:5"] * 8)
    requests = cc.parse_task_memory_cells(specs, max_seconds=12.5)
    assert len(requests) == 8
    assert requests[0] == cc.CellMeasurementRequest("ols", 10, 5, max_seconds=12.5)
    with pytest.raises(ValueError, match="exactly eight"):
        cc.parse_task_memory_cells("ols:10:5", max_seconds=12.5)


def test_calibration_reader_rejects_invalid_v1_peak_rss_file(tmp_path):
    old_path = tmp_path / "old.json"
    old_path.write_text(json.dumps({"format_version": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="Earlier peak_rss_bytes formats are invalid"):
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


def test_guard_rejects_smr_and_ffcws_data_paths():
    with pytest.raises(cc.PrivateDataAccessError):
        cc.guard_not_private_data(Path("/Users/x/Aleatoric_Luck/SMR/data/table.csv"))
    with pytest.raises(cc.PrivateDataAccessError):
        cc.guard_not_private_data(Path("/Users/x/Aleatoric_Luck/FFCWS/data/ard/table.parquet"))


def test_guard_allows_schema_and_scratch_paths(tmp_path):
    # FFCWS/schema is allowed (it holds only structure, not observations).
    schema_dir = cc.repo_root() / "FFCWS" / "schema"
    if schema_dir.exists():
        cc.guard_not_private_data(schema_dir / "README.md")
    cc.guard_not_private_data(tmp_path / "synthetic.parquet")


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

    params = cc.SyntheticDataParams(n_train=60, n_sources=8, seed=0)
    schema_path, _ = cc.generate_synthetic_bundle(tmp_path / "bundle", params)
    cc.build_session(schema_path, "y", seed=0)

    for path in opened_paths:
        assert "SMR/data" not in path
        assert "FFCWS/data" not in path


def test_onehot_group_size_pool_only_reads_schema_directory():
    sizes, provenance = cc.onehot_group_size_pool()
    assert len(sizes) > 0
    assert "data" not in provenance or "FFCWS/schema" in provenance


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
    params = cc.SyntheticDataParams(n_train=100, n_sources=10, seed=0)
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
    params = cc.SyntheticDataParams(n_train=60, n_sources=8, seed=0)
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
        n_sources=6,
        seed=3,
        continuous_fraction=0.5,
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
    assert synthetic_data["continuous_fraction"] == 0.5
    assert synthetic_data["missing_rate_continuous"] == 0.2
    assert synthetic_data["missing_rate_group"] == 0.15
    assert synthetic_data["outcome"] == params.outcome
    assert synthetic_data["missing_rates_are_unverified_placeholders"] is True
