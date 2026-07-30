from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aleatoric_nk_grid.model_registry import (
    SUPPORTED_MODEL_NAMES,
    load_model_params,
    make_model,
)
from aleatoric_nk_grid.preprocessing import SourceGroup, preprocess_cell


MODEL_PARAMS = Path(__file__).resolve().parents[1] / "model_params.yaml"
IMPUTATION = {
    "continuous": "median",
    "ordinal": "most_frequent",
    "onehot_group": "atomic_mode",
    "model_overrides": {"xgboost": "passthrough", "lightgbm": "passthrough"},
}


def _fast_params(model_name: str) -> dict:
    params = deepcopy(
        load_model_params(
            MODEL_PARAMS, task="regression", models=(model_name,)
        )[model_name]
    )
    if model_name in {"ridge", "lasso"}:
        params["n_alphas"] = 5
        params["max_cv_folds"] = 2
        if model_name != "ridge":
            params["max_iter"] = min(int(params.get("max_iter", 100)), 100)
    elif model_name in {"random_forest", "extra_trees"}:
        params.update(n_estimators=8, min_samples_leaf=1)
    elif model_name == "xgboost":
        params.update(max_rounds=3, cv_folds=2)
    elif model_name == "lightgbm":
        params.update(
            max_rounds=3,
            cv_folds=2,
            early_stopping_rounds=1,
            min_data_in_leaf=2,
        )
    elif model_name == "shallow_neural_network":
        params.update(
            hidden_layer_sizes=[4],
            max_iter=30,
            early_stopping=False,
        )
    elif model_name == "super_learner":
        params.update(
            cv=2,
            n_estimators=8,
            min_samples_leaf=1,
            hidden_layer_sizes=[4],
            max_iter=30,
            lgbm_n_estimators=5,
            lgbm_min_data_in_leaf=2,
        )
    return params


@pytest.mark.parametrize("model_name", SUPPORTED_MODEL_NAMES)
def test_registered_model_predictions_are_invariant_to_full_missing_prior(
    model_name,
):
    rng = np.random.default_rng(2026)
    useful_train = rng.normal(size=30)
    useful_test = rng.normal(size=8)
    y = 2.0 * useful_train + rng.normal(scale=0.05, size=30)
    raw_train = pd.DataFrame(
        {"useful": useful_train, "full_missing": np.nan}
    )
    raw_test = pd.DataFrame(
        {"useful": useful_test, "full_missing": np.nan}
    )

    def prepared(prior: float):
        groups = (
            SourceGroup(
                name="useful",
                features=("useful",),
                source_order=0,
                unit_type="continuous",
                source_prior=0.0,
            ),
            SourceGroup(
                name="full_missing",
                features=("full_missing",),
                source_order=1,
                unit_type="continuous",
                source_prior=prior,
            ),
        )
        return preprocess_cell(
            raw_train,
            raw_test,
            groups,
            IMPUTATION,
            model_name=model_name,
        )

    first = prepared(-3.0)
    second = prepared(1_000.0)
    try:
        first_model = make_model(
            model_name,
            seed=81,
            n_jobs=1,
            task="regression",
            params=_fast_params(model_name),
        )
        second_model = make_model(
            model_name,
            seed=81,
            n_jobs=1,
            task="regression",
            params=_fast_params(model_name),
        )
        first_model.fit(first.X_train, y)
        second_model.fit(second.X_train, y)
    except ImportError as exc:
        pytest.skip(str(exc))
    np.testing.assert_allclose(
        first_model.predict(first.X_test),
        second_model.predict(second.X_test),
        rtol=1e-10,
        atol=1e-12,
    )
