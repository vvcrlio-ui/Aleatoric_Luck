from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aleatoric_nk_grid import calibrate_cost as cc


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

    with pytest.raises(cc.MeasurementCensored):
        cc.measure_one_cell(
            session,
            model_name="ols",
            n=10,
            k=5,
            seed=0,
            draw=0,
            max_seconds=0.05,
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
    )

    out_path = cc.write_calibration_file(payload, tmp_path, date="2026-07-31")
    assert out_path.name == "cost_model_2026-07-31.json"

    reloaded = json.loads(out_path.read_text(encoding="utf-8"))

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
        "validation",
        "censored",
        "raw_measurements",
    }
    assert required_top_level.issubset(reloaded.keys())
    assert reloaded["t0_seconds"]["import"]["source"].startswith("pre-measured")

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
        measurement = cc.measure_one_cell(
            session, model_name=model, n=10, k=5, seed=0, draw=0, max_seconds=60
        )
        assert measurement.peak_rss_bytes == 123_456_789

    # xgboost is NOT in SUBPROCESS_MODELS (that was F1's bug: xgboost doesn't
    # go through the isolated subprocess in production) so it must take the
    # inline path, same as any other non-native model.
    for model in ("ols", "xgboost"):
        measurement = cc.measure_one_cell(
            session, model_name=model, n=10, k=5, seed=0, draw=0, max_seconds=60
        )
        assert measurement.peak_rss_bytes == 111

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
        cc.measure_one_cell(
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
