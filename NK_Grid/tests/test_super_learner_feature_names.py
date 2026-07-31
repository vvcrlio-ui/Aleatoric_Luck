from __future__ import annotations

import re
import warnings
from copy import deepcopy
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error
from sklearn.pipeline import make_pipeline

from aleatoric_nk_grid.model_registry import load_model_params, make_model


MODEL_PARAMS = Path(__file__).resolve().parents[1] / "model_params.yaml"
FEATURE_NAME_WARNINGS = {
    "regression": (
        "X does not have valid feature names, but LGBMRegressor was fitted "
        "with feature names"
    ),
    "classification": (
        "X does not have valid feature names, but LGBMClassifier was fitted "
        "with feature names"
    ),
}


def _super_learner_params(task: str) -> dict:
    params = deepcopy(
        load_model_params(
            MODEL_PARAMS,
            task=task,
            models=("super_learner",),
        )["super_learner"]
    )
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


def _data(task: str) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(20260729)
    X = pd.DataFrame(
        rng.normal(size=(36, 3)),
        columns=["signal_alpha", "signal_beta", "signal_gamma"],
    )
    X.loc[::8, "signal_beta"] = np.nan
    score = (
        1.3 * X["signal_alpha"].to_numpy()
        - 0.7 * X["signal_gamma"].to_numpy()
    )
    if task == "regression":
        y = score + rng.normal(scale=0.1, size=len(X))
    else:
        y = (score > np.median(score)).astype(int)
    return X, y


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_super_learner_lightgbm_uses_feature_names_consistently(task):
    X, y = _data(task)
    model = make_model(
        "super_learner",
        seed=17,
        n_jobs=1,
        task=task,
        params=_super_learner_params(task),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X, y)
        if task == "regression":
            model.predict(X.iloc[:6])
        else:
            model.predict(X.iloc[:6])
            model.predict_proba(X.iloc[:6])

    target = [
        item
        for item in caught
        if item.category is UserWarning
        and str(item.message) == FEATURE_NAME_WARNINGS[task]
    ]
    assert target == []


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_lightgbm_pandas_imputer_is_numerically_equivalent(task):
    X, y = _data(task)
    estimator_type = (
        lgb.LGBMRegressor if task == "regression" else lgb.LGBMClassifier
    )
    estimator_params = {
        "n_estimators": 8,
        "learning_rate": 0.05,
        "num_leaves": 7,
        "min_data_in_leaf": 2,
        "n_jobs": 1,
        "random_state": 17,
        "verbosity": -1,
    }
    ndarray_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        estimator_type(**estimator_params),
    )
    pandas_pipeline = make_pipeline(
        SimpleImputer(strategy="median").set_output(transform="pandas"),
        estimator_type(**estimator_params),
    )
    ndarray_pipeline.fit(X, y)
    pandas_pipeline.fit(X, y)

    warning_message = FEATURE_NAME_WARNINGS[task]
    with pytest.warns(
        UserWarning,
        match=f"^{re.escape(warning_message)}$",
    ):
        old_predictions = ndarray_pipeline.predict(X)
        old_probabilities = (
            ndarray_pipeline.predict_proba(X)
            if task == "classification"
            else None
        )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        new_predictions = pandas_pipeline.predict(X)
        new_probabilities = (
            pandas_pipeline.predict_proba(X)
            if task == "classification"
            else None
        )
    target = [
        item
        for item in caught
        if item.category is UserWarning and str(item.message) == warning_message
    ]
    assert target == []

    np.testing.assert_allclose(
        new_predictions,
        old_predictions,
        rtol=0.0,
        atol=0.0,
    )
    if task == "regression":
        assert mean_squared_error(y, new_predictions) == pytest.approx(
            mean_squared_error(y, old_predictions),
            rel=0.0,
            abs=0.0,
        )
    else:
        assert new_probabilities is not None
        assert old_probabilities is not None
        np.testing.assert_allclose(
            new_probabilities,
            old_probabilities,
            rtol=0.0,
            atol=0.0,
        )
        assert accuracy_score(y, new_predictions) == accuracy_score(
            y, old_predictions
        )
        assert log_loss(y, new_probabilities) == pytest.approx(
            log_loss(y, old_probabilities),
            rel=0.0,
            abs=0.0,
        )
