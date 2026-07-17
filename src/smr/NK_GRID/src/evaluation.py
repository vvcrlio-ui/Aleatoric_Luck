"""Evaluation metrics used by the NLSY replication scripts."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error


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
