from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from aleatoric_nk_grid.experiment import (
    checkpoint_parts_dir,
    write_checkpoint_part,
)
from aleatoric_nk_grid.nk_grid import (
    CLASSIFICATION_METRIC_COLUMNS,
    NKGridConfig,
    RunFailureThresholdExceeded,
    _ols_is_underdetermined,
    _positive_class_probability,
    _select_output_path,
    _timestamped_out_path,
    compute_classification_metrics,
    draw_orders,
    estimate_run_size,
    run_nk_grid,
    split_frame,
)

from conftest import write_schema_bundle


MODEL_PARAMS = Path(__file__).resolve().parents[1] / "model_params.yaml"


def _config(schema: Path, out: Path, **overrides) -> NKGridConfig:
    values = {
        "schema": schema,
        "out": out,
        "outcome": "y",
        "models": ("ols",),
        "seed": 123,
        "test_size": 0.3,
        "n_seeds": 1,
        "n_draws": 1,
        "n_sizes_n": 1,
        "n_sizes_k": 1,
        "max_n": 0,
        "max_k": 0,
        "batch_size": 10,
        "n_jobs": 1,
        "min_n": 10,
        "model_params": MODEL_PARAMS,
    }
    values.update(overrides)
    return NKGridConfig(**values)


def _regression_frame(rows: int = 40) -> pd.DataFrame:
    x = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "y": 2.0 * x - 0.5 * (x % 4) + 3.0,
            "X_a": x,
            "X_b": x % 4,
        }
    )


def test_engine_smoke_records_identity_and_k_diagnostics(tmp_path):
    frame = _regression_frame()
    frame.loc[::7, "X_b"] = np.nan
    schema = write_schema_bundle(
        tmp_path / "input", frame, predictors=["X_a", "X_b"]
    )
    out = tmp_path / "result.csv"
    returned = run_nk_grid(_config(schema, out))
    assert returned == out
    result = pd.read_csv(out)
    assert result.loc[0, "status"] == "ok"
    assert result.loc[0, "K"] == 2
    assert result.loc[0, "K_expanded"] == 2
    assert result.loc[0, "K_unobserved"] == 0
    assert result.loc[0, "experiment_id"] == "nkgrid-test-v1"
    manifest = json.loads(out.with_suffix(".manifest.json").read_text())
    assert manifest["failure_policy"]["passed"] is True
    assert manifest["failure_policy"]["denominator"] == 1
    assert manifest["identity"]["mode"] == "explicit-v1"
    assert manifest["semantic_contract"]["kind"] == "nk_grid"
    assert not checkpoint_parts_dir(out).exists()


def test_all_selected_sources_unobserved_is_fixed_skip_and_zero_denominator_failure(
    tmp_path,
):
    frame = _regression_frame(30)[["y", "X_a"]]
    split = split_frame(
        frame, ["X_a"], "y", test_size=0.3, seed=123
    )
    order = draw_orders(
        split.X_train.index,
        ["X_a"],
        seed=123,
        draw=0,
    ).row_index
    observed_row = int(order[-1])
    frame["X_a"] = np.nan
    frame.loc[observed_row, "X_a"] = 1.0
    schema = write_schema_bundle(
        tmp_path / "input", frame, predictors=["X_a"]
    )
    out = tmp_path / "result.csv"
    config = _config(schema, out, max_n=10, min_n=10)
    with pytest.raises(
        RunFailureThresholdExceeded, match="denominator is zero"
    ):
        run_nk_grid(config)
    result = pd.read_csv(out)
    assert result.loc[0, "status"] == "skipped"
    assert result.loc[0, "error"] == "all_selected_sources_unobserved"
    assert result.loc[0, "K"] == 1
    assert result.loc[0, "K_unobserved"] == 1
    assert checkpoint_parts_dir(out).exists()


def test_failed_ratio_raises_after_persisting_checkpoint_and_manifest(tmp_path):
    schema = write_schema_bundle(
        tmp_path / "input",
        _regression_frame(),
        predictors=["X_a", "X_b"],
    )
    out = tmp_path / "result.csv"
    with patch(
        "aleatoric_nk_grid.nk_grid.make_model",
        side_effect=RuntimeError("synthetic fit failure"),
    ):
        with pytest.raises(
            RunFailureThresholdExceeded, match="failed_ratio"
        ):
            run_nk_grid(_config(schema, out))
    result = pd.read_csv(out)
    assert result.loc[0, "status"] == "failed"
    assert "synthetic fit failure" in result.loc[0, "error"]
    manifest = json.loads(out.with_suffix(".manifest.json").read_text())
    assert manifest["failure_policy"]["failed_count"] == 1
    assert manifest["failure_policy"]["failed_ratio"] == 1.0
    assert manifest["failure_policy"]["passed"] is False
    assert checkpoint_parts_dir(out).exists()


def test_external_test_size_does_not_change_experiment_identity(tmp_path):
    train = _regression_frame()
    test = _regression_frame(12)
    train.insert(0, "id", np.arange(len(train)))
    test.insert(0, "id", 1000 + np.arange(len(test)))
    schema = write_schema_bundle(
        tmp_path / "input",
        train,
        test=test,
        split_mode="external_test",
        predictors=["X_a", "X_b"],
        id_column="id",
    )
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    run_nk_grid(_config(schema, first, test_size=0.1))
    run_nk_grid(_config(schema, second, test_size=0.9))
    first_id = pd.read_csv(first).loc[0, "experiment_id"]
    second_id = pd.read_csv(second).loc[0, "experiment_id"]
    assert first_id == second_id


def test_binary_classification_path_runs_with_schema_task(tmp_path):
    frame = _regression_frame(60)
    frame["y"] = (np.arange(len(frame)) % 2).astype(int)
    schema = write_schema_bundle(
        tmp_path / "input",
        frame,
        predictors=["X_a", "X_b"],
        task="classification",
    )
    out = tmp_path / "classification.csv"
    run_nk_grid(_config(schema, out, models=("ridge",)))
    row = pd.read_csv(out).iloc[0]
    assert row["status"] == "ok"
    assert row["task"] == "classification"
    assert 0.0 <= row["accuracy"] <= 1.0


@pytest.mark.parametrize("bad_score", [np.nan, np.inf, -np.inf])
def test_nonfinite_classification_scores_never_produce_apparently_valid_metrics(
    bad_score,
):
    metrics = compute_classification_metrics(
        [0, 1, 1],
        [bad_score, 0.8, 0.9],
        [0, 1, 0, 1],
    )
    assert set(metrics) == set(CLASSIFICATION_METRIC_COLUMNS)
    assert all(np.isnan(value) for value in metrics.values())


def test_nonfinite_predict_proba_fails_the_cell_contract():
    class BadClassifier:
        classes_ = np.array([0, 1])

        def predict_proba(self, X):
            return np.array([[0.4, 0.6], [np.nan, np.nan]])

    with pytest.raises(ValueError, match="non-finite"):
        _positive_class_probability(BadClassifier(), np.zeros((2, 1)))


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"n_seeds": 0}, "n_seeds must be at least 1"),
        ({"batch_size": 0}, "batch_size must be at least 1"),
        ({"models": ()}, "models must not be empty"),
        ({"models": ("ols", "ols")}, "models must not contain duplicates"),
        ({"n_jobs": 0}, "n_jobs must not be zero"),
    ],
)
def test_invalid_run_controls_fail_before_dry_run_arithmetic(overrides, match):
    config = _config(Path("schema.json"), Path("out.csv"), **overrides)
    with pytest.raises(ValueError, match=match):
        estimate_run_size(config)


@pytest.mark.parametrize(
    "model,overrides,error",
    [
        (
            "ridge",
            {"min_n": 1, "max_n": 1},
            "below minimum N for ridge's internal CV",
        ),
    ],
)
def test_floor_skips_do_not_run_full_preprocessing(
    tmp_path, model, overrides, error
):
    schema = write_schema_bundle(
        tmp_path / "input",
        _regression_frame(),
        predictors=["X_a", "X_b"],
    )
    out = tmp_path / f"{model}.csv"
    config = _config(schema, out, models=(model,), **overrides)
    with patch(
        "aleatoric_nk_grid.nk_grid.preprocess_cell",
        side_effect=AssertionError("preprocess_cell must not run"),
    ):
        with pytest.raises(RunFailureThresholdExceeded, match="denominator is zero"):
            run_nk_grid(config)
    row = pd.read_csv(out).iloc[0]
    assert row["status"] == "skipped"
    assert row["error"].startswith(error)
    assert row["K_unobserved"] == 0


def test_all_unobserved_skip_happens_before_any_preprocessing(tmp_path):
    # The BART N/K floor used to sit right behind this branch, so this test also
    # asserted a precedence between the two. With BART gone the only remaining
    # floor is the CV one, which cannot co-trigger at this N, so precedence is
    # no longer observable; the early-skip behaviour itself still is.
    frame = _regression_frame(30)[["y", "X_a"]]
    split = split_frame(frame, ["X_a"], "y", test_size=0.3, seed=123)
    order = draw_orders(
        split.X_train.index, ["X_a"], seed=123, draw=0
    ).row_index
    frame["X_a"] = np.nan
    frame.loc[int(order[-1]), "X_a"] = 1.0
    schema = write_schema_bundle(
        tmp_path / "input", frame, predictors=["X_a"]
    )
    out = tmp_path / "unobserved.csv"
    config = _config(
        schema,
        out,
        models=("ols",),
        min_n=10,
        max_n=10,
    )
    with patch(
        "aleatoric_nk_grid.nk_grid.preprocess_cell",
        side_effect=AssertionError("preprocess_cell must not run"),
    ):
        with pytest.raises(RunFailureThresholdExceeded, match="denominator is zero"):
            run_nk_grid(config)
    row = pd.read_csv(out).iloc[0]
    assert row["error"] == "all_selected_sources_unobserved"
    assert row["K_unobserved"] == 1


def test_completed_preset_can_be_reused_without_recomputation(tmp_path):
    schema = write_schema_bundle(
        tmp_path / "input",
        _regression_frame(),
        predictors=["X_a", "X_b"],
    )
    config = _config(
        schema,
        tmp_path / "result.csv",
        preset="dev",
        rerun_completed=False,
    )
    first = run_nk_grid(config)
    with (
        patch(
            "aleatoric_nk_grid.nk_grid.make_model",
            side_effect=AssertionError("completed run must not be recomputed"),
        ),
        patch(
            "aleatoric_nk_grid.nk_grid.load_checkpoint",
            side_effect=AssertionError("verified complete output needs no full read"),
        ),
        patch(
            "aleatoric_nk_grid.nk_grid.merge_checkpoint_parts",
            side_effect=AssertionError("verified complete output needs no rewrite"),
        ),
    ):
        second = run_nk_grid(config)
    assert second == first
    assert len(list(tmp_path.glob("result_dev_*.csv"))) == 1


@pytest.mark.parametrize("corruption", ["duplicate", "extra"])
def test_completed_output_with_non_exact_cell_index_fails_integrity_check(
    tmp_path,
    corruption,
):
    schema = write_schema_bundle(
        tmp_path / "input",
        _regression_frame(),
        predictors=["X_a", "X_b"],
    )
    config = _config(
        schema,
        tmp_path / "result.csv",
        rerun_completed=False,
    )
    out = run_nk_grid(config)
    result = pd.read_csv(out)
    corrupt = result.iloc[[0]].copy()
    if corruption == "extra":
        corrupt["K"] = corrupt["K"] + 1
    pd.concat([result, corrupt], ignore_index=True).to_csv(out, index=False)

    with patch(
        "aleatoric_nk_grid.nk_grid.load_checkpoint",
        side_effect=AssertionError("corrupt complete output must not be rewritten"),
    ):
        with pytest.raises(RuntimeError, match="projected-index integrity"):
            run_nk_grid(config)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("completion", []),
        ("completion", {"status": "complete", "completed_rows": "not-an-int"}),
        ("design", []),
    ],
)
def test_completed_output_with_malformed_manifest_structure_is_rebuilt(
    tmp_path,
    field,
    invalid_value,
):
    schema = write_schema_bundle(
        tmp_path / "input",
        _regression_frame(),
        predictors=["X_a", "X_b"],
    )
    config = _config(
        schema,
        tmp_path / "result.csv",
        rerun_completed=False,
    )
    out = run_nk_grid(config)
    manifest_path = out.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = invalid_value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    run_nk_grid(config)

    rebuilt = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert rebuilt["completion"]["status"] == "complete"
    assert isinstance(rebuilt["design"], dict)


def test_preset_resume_reuses_matching_manifest_before_first_checkpoint(tmp_path):
    declared = tmp_path / "result.csv"
    candidate = tmp_path / "result_dev_20260724-120000.csv"
    candidate.with_suffix(".manifest.json").write_text(
        json.dumps({"experiment_id": "matching-experiment"}),
        encoding="utf-8",
    )

    selected = _select_output_path(
        declared,
        preset="dev",
        experiment_id="matching-experiment",
        jobs=[("ols", 123, 0, 10, 2)],
        rerun_completed=True,
    )

    assert selected == candidate


def test_preset_resume_retries_failed_only_checkpoint_in_place(tmp_path):
    declared = tmp_path / "result.csv"
    candidate = tmp_path / "result_dev_20260724-120000.csv"
    pd.DataFrame(
        [
            {
                "experiment_id": "matching-experiment",
                "model": "ols",
                "seed": 123,
                "draw": 0,
                "N": 10,
                "K": 2,
                "status": "failed",
            }
        ]
    ).to_csv(candidate, index=False)

    selected = _select_output_path(
        declared,
        preset="dev",
        experiment_id="matching-experiment",
        jobs=[("ols", 123, 0, 10, 2)],
        rerun_completed=True,
    )

    assert selected == candidate


def test_preset_resume_discovers_orphan_checkpoint_parts(tmp_path):
    declared = tmp_path / "result.csv"
    candidate = tmp_path / "result_dev_20260724-120000.csv"
    write_checkpoint_part(
        [
            {
                "experiment_id": "matching-experiment",
                "model": "ols",
                "seed": 123,
                "draw": 0,
                "N": 10,
                "K": 1,
                "status": "ok",
            }
        ],
        candidate,
    )

    selected = _select_output_path(
        declared,
        preset="dev",
        experiment_id="matching-experiment",
        jobs=[
            ("ols", 123, 0, 10, 1),
            ("ols", 123, 0, 10, 2),
        ],
        rerun_completed=False,
    )

    assert not candidate.exists()
    assert not candidate.with_suffix(".manifest.json").exists()
    assert selected == candidate


def test_preset_scan_skips_newer_unrelated_corrupt_candidate(tmp_path):
    declared = tmp_path / "result.csv"
    valid = tmp_path / "result_dev_20260724-110000.csv"
    corrupt = tmp_path / "result_dev_20260724-120000.csv"
    job = ("ols", 123, 0, 10, 2)
    pd.DataFrame(
        [
            {
                "experiment_id": "matching-experiment",
                "model": job[0],
                "seed": job[1],
                "draw": job[2],
                "N": job[3],
                "K": job[4],
                "status": "failed",
            }
        ]
    ).to_csv(valid, index=False)
    corrupt.write_bytes(b"")
    corrupt_mtime = max(corrupt.stat().st_mtime, valid.stat().st_mtime + 1.0)
    os.utime(corrupt, (corrupt_mtime, corrupt_mtime))

    selected = _select_output_path(
        declared,
        preset="dev",
        experiment_id="matching-experiment",
        jobs=[job],
        rerun_completed=False,
    )

    assert selected == valid


def test_preset_scan_skips_non_object_manifest_on_unreadable_candidate(tmp_path):
    declared = tmp_path / "result.csv"
    valid = tmp_path / "result_dev_20260724-110000.csv"
    invalid = tmp_path / "result_dev_20260724-120000.csv"
    job = ("ols", 123, 0, 10, 2)
    pd.DataFrame(
        [
            {
                "experiment_id": "matching-experiment",
                "model": job[0],
                "seed": job[1],
                "draw": job[2],
                "N": job[3],
                "K": job[4],
                "status": "failed",
            }
        ]
    ).to_csv(valid, index=False)
    invalid.write_bytes(b"")
    invalid.with_suffix(".manifest.json").write_text("[]", encoding="utf-8")
    invalid_mtime = max(invalid.stat().st_mtime, valid.stat().st_mtime + 1.0)
    os.utime(invalid, (invalid_mtime, invalid_mtime))

    selected = _select_output_path(
        declared,
        preset="dev",
        experiment_id="matching-experiment",
        jobs=[job],
        rerun_completed=False,
    )

    assert selected == valid


def test_preset_scan_does_not_resume_manifest_only_non_object_json(tmp_path):
    declared = tmp_path / "result.csv"
    candidate = tmp_path / "result_dev_20260724-120000.csv"
    fresh = tmp_path / "result_dev_20260724-130000.csv"
    candidate.with_suffix(".manifest.json").write_text("[]", encoding="utf-8")

    with patch(
        "aleatoric_nk_grid.nk_grid._timestamped_out_path",
        return_value=fresh,
    ):
        selected = _select_output_path(
            declared,
            preset="dev",
            experiment_id="matching-experiment",
            jobs=[("ols", 123, 0, 10, 2)],
            rerun_completed=False,
        )

    assert selected == fresh


def test_completed_preset_starts_new_output_when_rerun_is_requested(tmp_path):
    declared = tmp_path / "result.csv"
    candidate = tmp_path / "result_dev_20260724-120000.csv"
    fresh = tmp_path / "result_dev_20260724-130000.csv"
    job = ("ols", 123, 0, 10, 2)
    pd.DataFrame(
        [
            {
                "experiment_id": "matching-experiment",
                "model": job[0],
                "seed": job[1],
                "draw": job[2],
                "N": job[3],
                "K": job[4],
                "status": "ok",
            }
        ]
    ).to_csv(candidate, index=False)

    with patch(
        "aleatoric_nk_grid.nk_grid._timestamped_out_path",
        return_value=fresh,
    ):
        selected = _select_output_path(
            declared,
            preset="dev",
            experiment_id="matching-experiment",
            jobs=[job],
            rerun_completed=True,
        )

    assert selected == fresh


def test_completed_newest_preset_does_not_revive_older_partial_on_rerun(tmp_path):
    declared = tmp_path / "result.csv"
    older = tmp_path / "result_dev_20260724-110000.csv"
    newest = tmp_path / "result_dev_20260724-120000.csv"
    fresh = tmp_path / "result_dev_20260724-130000.csv"
    jobs = [
        ("ols", 123, 0, 10, 1),
        ("ols", 123, 0, 10, 2),
    ]
    columns = ("model", "seed", "draw", "N", "K")
    pd.DataFrame(
        [
            {
                "experiment_id": "matching-experiment",
                **dict(zip(columns, jobs[0])),
                "status": "ok",
            }
        ]
    ).to_csv(older, index=False)
    pd.DataFrame(
        [
            {
                "experiment_id": "matching-experiment",
                **dict(zip(columns, job)),
                "status": "ok",
            }
            for job in jobs
        ]
    ).to_csv(newest, index=False)
    older_mtime = older.stat().st_mtime
    newest_mtime = max(newest.stat().st_mtime, older_mtime + 1.0)
    os.utime(newest, (newest_mtime, newest_mtime))

    with patch(
        "aleatoric_nk_grid.nk_grid._timestamped_out_path",
        return_value=fresh,
    ):
        selected = _select_output_path(
            declared,
            preset="dev",
            experiment_id="matching-experiment",
            jobs=jobs,
            rerun_completed=True,
        )

    assert selected == fresh


def test_timestamped_output_avoids_manifest_and_parts_only_collisions(tmp_path):
    stem = "result"
    preset = "dev"
    suffix = ".csv"
    timestamp = "20260724-120000"
    first = tmp_path / f"{stem}_{preset}_{timestamp}{suffix}"
    first.with_suffix(".manifest.json").write_text("{}", encoding="utf-8")
    second = tmp_path / f"{stem}_{preset}_20260724-120001{suffix}"

    with (
        patch(
            "aleatoric_nk_grid.nk_grid.datetime",
        ) as datetime_mock,
        patch("aleatoric_nk_grid.nk_grid.time.sleep") as sleep_mock,
    ):
        datetime_mock.now.return_value.strftime.side_effect = [
            timestamp,
            "20260724-120001",
        ]
        selected = _timestamped_out_path(tmp_path, stem, preset, suffix)

    assert selected == second
    sleep_mock.assert_called_once_with(1.0)


def test_scheduler_n_jobs_override_does_not_change_experiment_identity(tmp_path):
    schema = write_schema_bundle(
        tmp_path / "input",
        _regression_frame(),
        predictors=["X_a", "X_b"],
    )
    output = tmp_path / "result.csv"
    first_config = _config(
        schema,
        output,
        preset="dev",
        rerun_completed=False,
        n_jobs=1,
    )
    second_config = _config(
        schema,
        output,
        preset="dev",
        rerun_completed=False,
        n_jobs=8,
    )

    first = run_nk_grid(first_config)
    with patch(
        "aleatoric_nk_grid.nk_grid.make_model",
        side_effect=AssertionError("scheduler-only override must reuse the run"),
    ):
        second = run_nk_grid(second_config)

    assert second == first
    manifest = json.loads(
        first.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["design"]["parallelism"]["configured_outer_n_jobs"] == 8


def test_checkpoint_boundary_stop_is_resumable(tmp_path):
    schema = write_schema_bundle(
        tmp_path / "input",
        _regression_frame(),
        predictors=["X_a", "X_b"],
    )
    out = tmp_path / "result.csv"
    config = _config(
        schema,
        out,
        n_draws=3,
        batch_size=1,
        rerun_completed=False,
    )

    first = run_nk_grid(config, stop_after_batch=lambda: True)

    partial = pd.read_csv(first)
    partial_manifest = json.loads(
        out.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    assert len(partial) == 1
    assert partial_manifest["completion"]["status"] == "incomplete"
    assert partial_manifest["failure_policy"]["passed"] is None
    assert partial_manifest["termination"] == {
        "reason": "graceful_stop_after_batch",
        "resumable": True,
    }
    assert checkpoint_parts_dir(out).exists()

    second = run_nk_grid(config)

    completed = pd.read_csv(second)
    completed_manifest = json.loads(
        out.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    assert len(completed) == 3
    assert completed_manifest["completion"]["status"] == "complete"
    assert completed_manifest["failure_policy"]["passed"] is True
    assert not checkpoint_parts_dir(out).exists()


def test_resume_migrates_main_csv_when_parts_directory_is_empty(tmp_path):
    schema = write_schema_bundle(
        tmp_path / "input",
        _regression_frame(),
        predictors=["X_a", "X_b"],
    )
    out = tmp_path / "result.csv"
    config = _config(
        schema,
        out,
        n_draws=3,
        batch_size=1,
        rerun_completed=False,
    )
    run_nk_grid(config, stop_after_batch=lambda: True)
    assert len(pd.read_csv(out)) == 1

    shutil.rmtree(checkpoint_parts_dir(out))
    checkpoint_parts_dir(out).mkdir()

    run_nk_grid(config)

    result = pd.read_csv(out)
    assert len(result) == 3
    assert result["draw"].tolist() == [0, 1, 2]


def test_legacy_statusless_main_csv_can_mix_with_new_retry_shards(tmp_path):
    schema = write_schema_bundle(
        tmp_path / "input",
        _regression_frame(),
        predictors=["X_a", "X_b"],
    )
    out = tmp_path / "result.csv"
    config = _config(
        schema,
        out,
        n_draws=3,
        batch_size=1,
        rerun_completed=False,
    )
    run_nk_grid(config, stop_after_batch=lambda: True)
    legacy = pd.read_csv(out).drop(columns=["status"])
    legacy.to_csv(out, index=False)
    shutil.rmtree(checkpoint_parts_dir(out))

    run_nk_grid(config)

    result = pd.read_csv(out)
    assert len(result) == 3
    assert result["status"].eq("ok").all()


def test_interrupted_shard_cleanup_cannot_hide_verified_final_csv(tmp_path):
    schema = write_schema_bundle(
        tmp_path / "input",
        _regression_frame(),
        predictors=["X_a", "X_b"],
    )
    out = tmp_path / "result.csv"
    config = _config(schema, out, rerun_completed=False)

    with patch(
        "aleatoric_nk_grid.nk_grid.shutil.rmtree",
        side_effect=OSError("synthetic cleanup interruption"),
    ):
        run_nk_grid(config)

    assert out.exists()
    assert not checkpoint_parts_dir(out).exists()
    assert list(tmp_path.glob(".result.parts.retired-*"))
    with (
        patch(
            "aleatoric_nk_grid.nk_grid.make_model",
            side_effect=AssertionError("verified final CSV must remain authoritative"),
        ),
        patch(
            "aleatoric_nk_grid.nk_grid.merge_checkpoint_parts",
            side_effect=AssertionError("verified final CSV must not be rewritten"),
        ),
    ):
        assert run_nk_grid(config) == out


def test_scheduler_stop_defers_full_csv_materialization_until_resume(tmp_path):
    schema = write_schema_bundle(
        tmp_path / "input",
        _regression_frame(),
        predictors=["X_a", "X_b"],
    )
    out = tmp_path / "result.csv"
    config = _config(
        schema,
        out,
        n_draws=3,
        batch_size=1,
        rerun_completed=False,
    )

    with patch(
        "aleatoric_nk_grid.nk_grid.merge_checkpoint_parts",
        side_effect=AssertionError("cooperative stop must not materialize full CSV"),
    ):
        returned = run_nk_grid(
            config,
            stop_after_batch=lambda: True,
            defer_materialization_on_stop=True,
        )

    assert returned == out
    assert not out.exists()
    partial_manifest = json.loads(
        out.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    assert partial_manifest["completion"]["status"] == "incomplete"
    assert partial_manifest["completion"]["materialized_rows"] == 1
    assert partial_manifest["output"]["csv_materialization_deferred"] is True
    assert partial_manifest["diagnostics"]["deferred_until_completion"] is True
    assert checkpoint_parts_dir(out).exists()

    run_nk_grid(config)

    completed = pd.read_csv(out)
    assert len(completed) == 3
    assert not checkpoint_parts_dir(out).exists()


def test_stop_request_after_final_batch_finishes_without_resume_marker(tmp_path):
    schema = write_schema_bundle(
        tmp_path / "input",
        _regression_frame(),
        predictors=["X_a", "X_b"],
    )
    out = tmp_path / "result.csv"

    run_nk_grid(_config(schema, out), stop_after_batch=lambda: True)

    manifest = json.loads(
        out.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["completion"]["status"] == "complete"
    assert manifest["failure_policy"]["passed"] is True
    assert "termination" not in manifest
    assert not checkpoint_parts_dir(out).exists()


def test_scheduler_stop_after_final_cell_defers_materialization_and_requeues(
    tmp_path,
):
    schema = write_schema_bundle(
        tmp_path / "input",
        _regression_frame(),
        predictors=["X_a", "X_b"],
    )
    out = tmp_path / "result.csv"
    config = _config(schema, out, rerun_completed=False)

    run_nk_grid(
        config,
        stop_after_batch=lambda: True,
        defer_materialization_on_stop=True,
    )

    assert not out.exists()
    manifest = json.loads(
        out.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["completion"]["status"] == "incomplete"
    assert manifest["completion"]["checkpoint_rows_complete"] is True
    assert manifest["termination"]["reason"] == (
        "graceful_stop_before_materialization"
    )
    assert checkpoint_parts_dir(out).exists()

    run_nk_grid(config)

    assert len(pd.read_csv(out)) == 1
    assert not checkpoint_parts_dir(out).exists()


def test_underdetermined_uses_varying_expanded_columns():
    expanded_onehot = pd.DataFrame(np.eye(3), columns=["c0", "c1", "c2"])
    assert _ols_is_underdetermined(expanded_onehot) is True
    assert _ols_is_underdetermined(expanded_onehot[["c0", "c1"]]) is False
