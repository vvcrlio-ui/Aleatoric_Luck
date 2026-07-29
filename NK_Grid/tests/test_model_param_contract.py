from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aleatoric_nk_grid.model_registry import (
    MODEL_NAMES,
    REMOVED_MODEL_NAMES,
    load_algorithm_version,
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


def test_elastic_net_search_keeps_range_with_reduced_alpha_count():
    assert load_algorithm_version(MODEL_PARAMS) == "nk-grid-v5-adapter-3"
    selected = load_model_params(
        MODEL_PARAMS,
        task="regression",
        models=("elastic_net",),
    )["elastic_net"]
    assert selected["alpha_log10_min"] == -4
    assert selected["alpha_log10_max"] == 1
    assert selected["n_alphas"] == 20
    assert selected["l1_ratio"] == [0.1, 0.5, 0.9]
    assert selected["max_cv_folds"] == 5


def test_removed_model_is_absent_from_the_model_space():
    assert "bart" not in MODEL_NAMES
    assert "bart" in REMOVED_MODEL_NAMES
    for task in ("regression", "classification"):
        assert "bart" not in _document()[task]


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_requesting_a_removed_model_fails_with_a_self_explaining_error(task):
    # A stale panel must fail loudly rather than silently producing results for
    # one model fewer than it asked for.
    with pytest.raises(ValueError, match=r"removed from the model space"):
        load_model_params(MODEL_PARAMS, task=task, models=("bart",))
    with pytest.raises(ValueError, match=r"plans/remove-bart\.md"):
        make_model("bart", seed=1, n_jobs=1, task=task, params={})


def test_removed_model_environment_overrides_are_gone(monkeypatch):
    # BART_* used to be resolved at construction time; nothing may read them now.
    monkeypatch.setenv("BART_N_TREES", "17")
    selected = load_model_params(
        MODEL_PARAMS, task="regression", models=("random_forest",)
    )
    resolved = resolved_model_params(selected)
    assert "bart" not in resolved
    assert all("n_trees" not in params for params in resolved.values())
