from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aleatoric_nk_grid.model_registry import (
    BartPyRegressor,
    load_model_params,
    make_model,
    resolved_model_params,
)


MODEL_PARAMS = Path(__file__).resolve().parents[1] / "model_params.yaml"


def _document() -> dict:
    return yaml.safe_load(MODEL_PARAMS.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "model,field",
    [("xgboost", "eval_metric"), ("lightgbm", "metric")],
)
def test_cv_regression_metric_is_fixed_to_rmse_at_load_time(
    tmp_path, model, field
):
    document = _document()
    document["regression"][model][field] = "mae"
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match=r"requires .*='rmse'"):
        load_model_params(path, task="regression", models=(model,))


def test_cv_regressor_missing_required_parameter_fails_at_load_time(tmp_path):
    document = _document()
    document["regression"]["xgboost"].pop("cv_folds")
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="Missing required parameters.*cv_folds"):
        load_model_params(path, task="regression", models=("xgboost",))


def test_locked_cv_regression_parameters_load_with_rmse():
    selected = load_model_params(
        MODEL_PARAMS,
        task="regression",
        models=("xgboost", "lightgbm"),
    )
    assert selected["xgboost"]["eval_metric"] == "rmse"
    assert selected["lightgbm"]["metric"] == "rmse"


def test_bart_environment_override_is_resolved_at_model_construction(
    monkeypatch,
):
    selected = load_model_params(
        MODEL_PARAMS, task="regression", models=("bart",)
    )
    monkeypatch.setenv("BART_N_TREES", "17")
    resolved = resolved_model_params(selected)
    model = make_model(
        "bart",
        seed=1,
        n_jobs=1,
        task="regression",
        params=selected["bart"],
    )
    assert resolved["bart"]["n_trees"] == 17
    assert model.n_trees == 17
    # Direct construction has static documented defaults and no hidden
    # import-time environment state.
    assert BartPyRegressor().n_trees == 200
