import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import nk_grid
from model_registry import make_model
from nk_grid import (
    NKGridConfig,
    compute_regression_metrics,
    log2_size_grid,
    parse_args,
    run_nk_grid,
)
from run_panels import PRESETS, resolve_panel


def _regression_frame(n_rows: int, *, offset: float = 0.0) -> pd.DataFrame:
    index = np.arange(n_rows, dtype=float)
    x_a = index + offset
    x_b = ((index * 7) % 11) - 5
    return pd.DataFrame(
        {
            "challengeID": np.arange(n_rows) + int(offset * 1000),
            "y": 2.5 * x_a - 1.25 * x_b + 3.0,
            "X_a": x_a,
            "X_b": x_b,
        }
    )


def _classification_frame(n_rows: int, *, offset: float = 0.0) -> pd.DataFrame:
    index = np.arange(n_rows, dtype=float)
    x_a = index + offset
    x_b = ((index * 5) % 13) - 6
    return pd.DataFrame(
        {
            "challengeID": np.arange(n_rows) + int(offset * 1000),
            "y": ((index.astype(int) % 4) >= 2).astype(int),
            "X_a": x_a,
            "X_b": x_b,
        }
    )


def _write_csv(tmp_path: Path, name: str, frame: pd.DataFrame) -> Path:
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


def _config(tmp_path: Path, **overrides) -> NKGridConfig:
    values = {
        "data": tmp_path / "train.csv",
        "test_data": tmp_path / "test.csv",
        "out": tmp_path / "results.csv",
        "dataset": "synthetic",
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
        "predictor_prefix": ("X_",),
    }
    values.update(overrides)
    return NKGridConfig(**values)


def test_n_grid_starts_at_ten_while_k_grid_still_starts_at_one():
    n_grid = log2_size_grid(100, 5, min_size=10)
    k_grid = log2_size_grid(100, 5)

    assert n_grid.tolist() == [10, 18, 32, 56, 100]
    assert k_grid[0] == 1


@pytest.mark.parametrize("preset_name", ["dev", "production"])
def test_sweep_presets_set_minimum_n_to_ten(preset_name):
    assert PRESETS[preset_name]["min_n"] == 10


def test_resolve_panel_accepts_relative_test_path(tmp_path):
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    panel = {
        "name": "ffc_gpa",
        "preset": "dev",
        "data": "../data/ffc_train_gpa.csv",
        "test": "../data/ffc_test_gpa.csv",
        "out": "../out/gpa.csv",
        "dataset": "ffc",
        "outcome": "gpa",
        "models": ["ols"],
    }

    name, config = resolve_panel(panel, manifest_dir)

    assert name == "ffc_gpa"
    assert config.test_data == manifest_dir / "../data/ffc_test_gpa.csv"


def test_resolve_panel_defaults_test_data_to_none(tmp_path):
    panel = {
        "name": "internal",
        "preset": "dev",
        "data": "train.csv",
        "out": "out.csv",
        "dataset": "synthetic",
        "outcome": "y",
        "models": ["ols"],
    }

    _, config = resolve_panel(panel, tmp_path)

    assert config.test_data is None


def test_parse_args_accepts_test_data(tmp_path):
    argv = [
        "nk_grid.py",
        "--data",
        str(tmp_path / "train.csv"),
        "--test-data",
        str(tmp_path / "test.csv"),
        "--outcome",
        "y",
        "--out",
        str(tmp_path / "out.csv"),
        "--models",
        "ols",
        "--predictor-prefix",
        "X_",
    ]

    with patch.object(sys, "argv", argv):
        config = parse_args()

    assert config.test_data == tmp_path / "test.csv"


def test_feature_manifest_groups_expanded_columns_under_source_variables(tmp_path):
    train = _regression_frame(40).rename(columns={"X_a": "X_a", "X_b": "C_b__0"})
    train["M_a__missing"] = 0
    train["C_b__1"] = 1 - (train["C_b__0"] > 0).astype(int)
    test = _regression_frame(15, offset=100.0).rename(
        columns={"X_a": "X_a", "X_b": "C_b__0"}
    )
    test["M_a__missing"] = 0
    test["C_b__1"] = 1 - (test["C_b__0"] > 0).astype(int)

    train_path = _write_csv(tmp_path, "train.csv", train)
    test_path = _write_csv(tmp_path, "test.csv", test)
    manifest_path = tmp_path / "feature_manifest.csv"
    pd.DataFrame(
        {
            "source_column": ["a", "a", "b", "b"],
            "feature_name": ["X_a", "M_a__missing", "C_b__0", "C_b__1"],
            "keep": [True, True, True, True],
        }
    ).to_csv(manifest_path, index=False)
    out_path = tmp_path / "grouped_results.csv"

    run_nk_grid(
        _config(
            tmp_path,
            data=train_path,
            test_data=test_path,
            out=out_path,
            predictor_prefix=("X_", "C_", "M_"),
            feature_manifest=manifest_path,
        )
    )

    row = pd.read_csv(out_path).iloc[0]
    assert row["status"] == "ok"
    assert row["K"] == 2
    assert row["K_expanded"] == 4
    assert row["n_features_total"] == 2
    assert row["n_expanded_features_total"] == 4


def test_external_mode_aligns_test_predictors_and_uses_all_test_rows(tmp_path):
    train = _regression_frame(40)
    test = _regression_frame(15, offset=100.0)
    test = test[["challengeID", "y", "X_b", "X_a"]]
    train_path = _write_csv(tmp_path, "train.csv", train)
    test_path = _write_csv(tmp_path, "test.csv", test)
    out_path = tmp_path / "results.csv"

    run_nk_grid(
        _config(
            tmp_path,
            data=train_path,
            test_data=test_path,
            out=out_path,
            n_seeds=2,
            n_sizes_n=2,
            n_sizes_k=2,
            batch_size=8,
        )
    )

    results = pd.read_csv(out_path)
    assert set(results["split_mode"]) == {"external_test"}
    assert results["n_test_total"].eq(15).all()
    assert results["n_train_total"].eq(40).all()
    assert results["test_data_sha256"].nunique() == 1

    full_rows = results[(results["N"] == 40) & (results["K"] == 2)]
    assert len(full_rows) == 2
    model = make_model("ols", seed=nk_grid._model_seed(123, 0, 40, 2), n_jobs=1)
    model.fit(train[["X_a", "X_b"]], train["y"])
    preds = model.predict(test[["X_a", "X_b"]])
    expected = compute_regression_metrics(test["y"], preds, train["y"])
    assert full_rows["rmse"].tolist() == pytest.approx([expected["rmse"], expected["rmse"]])

    assert results["N"].ge(10).all()
    subsampled = results[(results["N"] == 10) & (results["K"] == 1)].sort_values("seed")
    assert len(subsampled) == 2
    assert subsampled["rmse"].nunique() == 2


def test_external_mode_drops_missing_outcomes_before_scanning(tmp_path):
    train = _regression_frame(41)
    test = _regression_frame(16, offset=50.0)
    train.loc[40, "y"] = np.nan
    test.loc[15, "y"] = np.nan
    train_path = _write_csv(tmp_path, "train.csv", train)
    test_path = _write_csv(tmp_path, "test.csv", test)
    out_path = tmp_path / "results.csv"

    run_nk_grid(_config(tmp_path, data=train_path, test_data=test_path, out=out_path))

    results = pd.read_csv(out_path)
    assert results["n_train_total"].tolist() == [40]
    assert results["n_test_total"].tolist() == [15]


def test_external_mode_rejects_missing_test_predictor(tmp_path):
    train_path = _write_csv(tmp_path, "train.csv", _regression_frame(10))
    test = _regression_frame(5, offset=20.0).drop(columns=["X_b"])
    test_path = _write_csv(tmp_path, "test.csv", test)

    with pytest.raises(ValueError, match="test data.*missing predictor.*X_b"):
        run_nk_grid(_config(tmp_path, data=train_path, test_data=test_path))


def test_external_mode_rejects_missing_outcome_column(tmp_path):
    cases = [
        (
            "training data",
            _regression_frame(10).drop(columns=["y"]),
            _regression_frame(5, offset=20.0),
        ),
        (
            "test data",
            _regression_frame(10),
            _regression_frame(5, offset=20.0).drop(columns=["y"]),
        ),
    ]

    for label, train, test in cases:
        train_path = _write_csv(tmp_path, f"{label}_train.csv", train)
        test_path = _write_csv(tmp_path, f"{label}_test.csv", test)
        with pytest.raises(KeyError, match=f"Outcome not found in {label}"):
            run_nk_grid(_config(tmp_path, data=train_path, test_data=test_path))


def test_external_mode_warns_when_test_size_is_ignored(tmp_path):
    train_path = _write_csv(tmp_path, "train.csv", _regression_frame(10))
    test_path = _write_csv(tmp_path, "test.csv", _regression_frame(5, offset=20.0))

    with patch.object(nk_grid, "log_progress") as log_progress:
        run_nk_grid(
            _config(
                tmp_path,
                data=train_path,
                test_data=test_path,
                test_size=0.5,
                max_n=10,
                max_k=1,
            ),
            max_jobs=0,
        )

    messages = [call.args[0] for call in log_progress.call_args_list]
    assert any("test_size=0.5" in message and "ignored" in message for message in messages)


def test_external_mode_supports_classification_with_fixed_test_rows(tmp_path):
    train_path = _write_csv(tmp_path, "train.csv", _classification_frame(40))
    test_path = _write_csv(tmp_path, "test.csv", _classification_frame(15, offset=100.0))
    out_path = tmp_path / "classification.csv"

    run_nk_grid(
        _config(
            tmp_path,
            data=train_path,
            test_data=test_path,
            out=out_path,
            task="classification",
            models=("ridge",),
        )
    )

    results = pd.read_csv(out_path)
    row = results.iloc[0]
    assert row["status"] == "ok"
    assert row["task"] == "classification"
    assert row["n_test_total"] == 15
    assert np.isfinite(row["accuracy"])


def test_classification_one_class_training_sample_is_skipped(tmp_path):
    train = _classification_frame(12)
    train["y"] = 0
    test = _classification_frame(8, offset=100.0)
    train_path = _write_csv(tmp_path, "train.csv", train)
    test_path = _write_csv(tmp_path, "test.csv", test)
    out_path = tmp_path / "one_class_classification.csv"

    run_nk_grid(
        _config(
            tmp_path,
            data=train_path,
            test_data=test_path,
            out=out_path,
            task="classification",
            models=("ridge",),
            n_seeds=1,
            n_draws=1,
            n_sizes_n=1,
            n_sizes_k=1,
            max_n=10,
            max_k=1,
        )
    )

    row = pd.read_csv(out_path).iloc[0]
    assert row["status"] == "skipped"
    assert "single-class training sample" in row["error"]
    assert pd.isna(row["roc_auc"])


def test_external_and_internal_runs_have_distinct_experiments_in_same_checkpoint(tmp_path):
    train_path = _write_csv(tmp_path, "train.csv", _regression_frame(40))
    test_path = _write_csv(tmp_path, "test.csv", _regression_frame(15, offset=100.0))
    out_path = tmp_path / "checkpoint.csv"

    common = {
        "data": train_path,
        "out": out_path,
        "models": ("ols",),
        "n_seeds": 1,
        "n_draws": 1,
        "n_sizes_n": 1,
        "n_sizes_k": 1,
        "max_n": 0,
        "max_k": 0,
        "batch_size": 10,
        "n_jobs": 1,
        "predictor_prefix": ("X_",),
    }
    run_nk_grid(_config(tmp_path, test_data=None, **common))
    run_nk_grid(_config(tmp_path, test_data=test_path, **common))

    results = pd.read_csv(out_path)
    assert len(results) == 2
    assert results["experiment_id"].nunique() == 2
    assert set(results["split_mode"]) == {"internal_random", "external_test"}
