from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aleatoric_nk_grid.model_registry import (
    MODEL_NAMES,
    REMOVED_MODEL_NAMES,
    SUPPORTED_MODEL_NAMES,
    load_algorithm_version,
    load_model_params,
    make_model,
    resolved_model_params,
)


MODEL_PARAMS = Path(__file__).resolve().parents[1] / "model_params.yaml"
REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PARAM_PATHS = (
    MODEL_PARAMS,
    REPO_ROOT / "FFCWS" / "model_params.yaml",
    REPO_ROOT / "SMR" / "model_params.yaml",
)
REMOVED_MODELS = tuple(sorted(REMOVED_MODEL_NAMES))
EXPECTED_REMOVED_MODEL_COUNT = 2
EXPECTED_RIDGE_GRID = {
    "alpha_log10_min": -4,
    "alpha_log10_max": 6,
    "n_alphas": 63,
    "scoring": "neg_mean_squared_error",
}
EXPECTED_SUPER_LEARNER_RIDGE_GRID = {
    "ridge_alpha_log10_min": -4,
    "ridge_alpha_log10_max": 6,
    "ridge_n_alphas": 63,
    "ridge_scoring": "neg_mean_squared_error",
}


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


def test_super_learner_missing_ridge_grid_fails_at_load_time(tmp_path):
    document = _document()
    document["regression"]["super_learner"].pop("ridge_n_alphas")
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(
        ValueError,
        match=r"Missing required parameters.*ridge_n_alphas",
    ):
        load_model_params(path, task="regression", models=("super_learner",))


@pytest.mark.parametrize("params_path", MODEL_PARAM_PATHS)
def test_ridge_rejects_removed_max_cv_folds_at_load_time(tmp_path, params_path):
    document = yaml.safe_load(params_path.read_text(encoding="utf-8"))
    document["regression"]["ridge"]["max_cv_folds"] = 5
    stale = tmp_path / params_path.name
    stale.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="max_cv_folds"):
        load_model_params(stale, task="regression", models=("ridge",))


def test_locked_cv_regression_parameters_load_with_rmse():
    selected = load_model_params(
        MODEL_PARAMS,
        task="regression",
        models=("xgboost", "lightgbm"),
    )
    assert selected["xgboost"]["eval_metric"] == "rmse"
    assert selected["lightgbm"]["metric"] == "rmse"


@pytest.mark.parametrize("params_path", MODEL_PARAM_PATHS)
@pytest.mark.parametrize("task", ["regression", "classification"])
def test_model_param_contract_covers_model_space_exactly(params_path, task):
    assert load_algorithm_version(params_path) == "nk-grid-v6-fold-local-1"
    document = yaml.safe_load(params_path.read_text(encoding="utf-8"))
    assert set(document[task]) == set(SUPPORTED_MODEL_NAMES)
    selected = load_model_params(
        params_path,
        task=task,
        models=MODEL_NAMES,
    )
    assert len(MODEL_NAMES) == 9
    assert set(selected) == set(MODEL_NAMES)


@pytest.mark.parametrize("params_path", MODEL_PARAM_PATHS)
def test_regression_ridge_grids_are_consistent_across_parameter_files(params_path):
    selected = load_model_params(
        params_path,
        task="regression",
        models=("ridge", "super_learner"),
    )
    assert selected["ridge"] == EXPECTED_RIDGE_GRID
    assert {
        key: selected["super_learner"][key]
        for key in EXPECTED_SUPER_LEARNER_RIDGE_GRID
    } == EXPECTED_SUPER_LEARNER_RIDGE_GRID


@pytest.mark.parametrize("params_path", MODEL_PARAM_PATHS)
def test_mlp_iteration_limit_distribution_is_locked(params_path):
    values = [
        line.strip() for line in params_path.read_text(encoding="utf-8").splitlines()
    ]
    assert values.count("max_iter: 500") == 3
    assert values.count("max_iter: 2000") == 1


def test_removed_model_registry_covers_expected_retirements():
    assert len(REMOVED_MODELS) == EXPECTED_REMOVED_MODEL_COUNT


@pytest.mark.parametrize("removed", REMOVED_MODELS)
def test_removed_model_request_fails_with_self_explanatory_message(removed):
    with pytest.raises(ValueError, match="removed from the model space"):
        load_model_params(MODEL_PARAMS, task="regression", models=(removed,))


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
