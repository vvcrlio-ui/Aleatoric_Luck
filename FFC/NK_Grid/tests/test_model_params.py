import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model_registry import load_model_params
from run_panels import resolved_panels


ACTIVE_MODELS = {
    "ols",
    "ridge",
    "lasso",
    "elastic_net",
    "random_forest",
    "xgboost",
    "lightgbm",
}


def _write_yaml(path: Path, value) -> Path:
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    return path


def test_default_model_params_cover_active_models_for_both_tasks():
    path = ROOT / "model_params.yaml"

    regression = load_model_params(path, task="regression", models=ACTIVE_MODELS)
    classification = load_model_params(
        path, task="classification", models=ACTIVE_MODELS
    )

    assert set(regression) == ACTIVE_MODELS
    assert set(classification) == ACTIVE_MODELS
    assert regression["random_forest"]["n_estimators"] == 500
    assert regression["xgboost"]["max_rounds"] == 90
    assert classification["elastic_net"]["l1_ratio"] == 0.5


def test_manifest_level_model_params_path_is_applied_to_panels(tmp_path):
    params_path = _write_yaml(
        tmp_path / "params.yaml",
        {"regression": {"ols": {"fit_intercept": True}}},
    )
    manifest_path = _write_yaml(
        tmp_path / "panels.yaml",
        {
            "model_params": params_path.name,
            "panels": [
                {
                    "name": "example",
                    "data": "train.csv",
                    "out": "out.csv",
                    "dataset": "synthetic",
                    "outcome": "y",
                    "models": ["ols"],
                }
            ],
        },
    )

    [(name, config)] = resolved_panels(manifest_path)

    assert name == "example"
    assert config.model_params == params_path


def test_model_params_missing_selected_model_has_clear_error(tmp_path):
    path = _write_yaml(tmp_path / "params.yaml", {"regression": {"ols": {}}})

    with pytest.raises(
        ValueError, match="regression.*missing selected model.*ridge"
    ):
        load_model_params(path, task="regression", models=["ols", "ridge"])


def test_unknown_model_parameter_has_clear_error(tmp_path):
    path = _write_yaml(
        tmp_path / "params.yaml",
        {"regression": {"ols": {"not_a_real_parameter": 1}}},
    )
    with pytest.raises(
        ValueError, match="Invalid parameters.*regression.*ols.*not_a_real_parameter"
    ):
        load_model_params(path, task="regression", models=["ols"])
