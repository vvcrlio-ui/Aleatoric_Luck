"""Shared evaluation metrics for prediction-quality experiments."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error

METRIC_DEFINITION_VERSION = "regression-denominators-v2"


def regression_denominators(y_test, y_pred, y_train):
    """Raw MSE and explicitly named denominators; legacy skill is unchanged."""
    test, pred, train = (np.asarray(values, dtype=float).reshape(-1) for values in (y_test, y_pred, y_train))
    if not len(test) or not len(train) or len(test) != len(pred):
        raise ValueError("regression metrics require nonempty aligned arrays")
    if not all(np.isfinite(values).all() for values in (test, pred, train)):
        raise ValueError("regression metrics require finite targets and predictions")
    mse = float(np.mean((test - pred) ** 2))
    null = float(np.mean((test - train.mean()) ** 2))
    variance = float(np.mean((test - test.mean()) ** 2))
    return {"mse": mse, "null_mse_train_mean": null, "test_target_variance": variance,
            "skill_train_mean": 1 - mse / null if null > 0 else np.nan,
            "r2_test_mean": 1 - mse / variance if variance > 0 else np.nan}


def training_mean_null_mse(y_test, y_train) -> float:
    """MSE of a null model trained on the supplied training outcomes."""

    return float(
        mean_squared_error(
            y_test,
            np.full(len(y_test), np.asarray(y_train, dtype=float).mean()),
        )
    )


def r2_against_training_mean(mse: float, y_test, y_train) -> float:
    """Paper-aligned test R-squared with the training mean as denominator."""

    return 1.0 - float(mse) / training_mean_null_mse(y_test, y_train)
