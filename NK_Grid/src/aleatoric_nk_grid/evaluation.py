"""Shared evaluation metrics for prediction-quality experiments."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, mean_squared_error

METRIC_DEFINITION_VERSION = "regression-denominators-v3"
CLASSIFICATION_METRIC_DEFINITION_VERSION = "classification-v3"


def regression_denominators(y_test, y_pred, y_train, y_train_full):
    """Raw MSE and explicitly named denominators; legacy skill is unchanged.

    ``y_train`` is the N sampled training rows. ``y_train_full`` is the whole training pool the
    rows were drawn from, so its null MSE is one constant across every N, K, seed and draw.
    """
    test, pred, train, full = (np.asarray(values, dtype=float).reshape(-1)
                               for values in (y_test, y_pred, y_train, y_train_full))
    if not len(test) or not len(train) or not len(full) or len(test) != len(pred):
        raise ValueError("regression metrics require nonempty aligned arrays")
    if not all(np.isfinite(values).all() for values in (test, pred, train, full)):
        raise ValueError("regression metrics require finite targets and predictions")
    mse = float(np.mean((test - pred) ** 2))
    null = float(np.mean((test - train.mean()) ** 2))
    full_null = float(np.mean((test - full.mean()) ** 2))
    variance = float(np.mean((test - test.mean()) ** 2))
    return {"mse": mse, "null_mse_train_mean": null, "test_target_variance": variance,
            "skill_train_mean": 1 - mse / null if null > 0 else np.nan,
            "r2_test_mean": 1 - mse / variance if variance > 0 else np.nan,
            "null_mse_full_train_mean": full_null,
            "r2_full_train_mean": 1 - mse / full_null if full_null > 0 else np.nan}


def _mcfadden_pseudo_r2(y_true, clipped, y_null) -> float:
    """McFadden pseudo R-squared against a null that predicts the positive rate of ``y_null``."""

    positive_rate = float(np.mean(y_null)) if len(y_null) else np.nan
    if len(np.unique(y_true)) != 2 or not np.isfinite(positive_rate) or not 0.0 < positive_rate < 1.0:
        return np.nan
    model_loglik = float(np.sum(y_true * np.log(clipped) + (1 - y_true) * np.log(1 - clipped)))
    null_loglik = float(np.sum(y_true * np.log(positive_rate) + (1 - y_true) * np.log(1 - positive_rate)))
    return 1.0 - model_loglik / null_loglik if null_loglik != 0 else np.nan


def classification_denominators(y_test, y_score, y_train, y_train_full):
    """Brier score and the scores that compare a model's finite probabilities with a null.

    ``y_train`` is the N sampled training rows. ``y_train_full`` is the whole training pool the
    rows were drawn from, so its null is one constant across every N, K, seed and draw.
    ``brier_full_train_r2`` is the Fragile Families Challenge R-squared (Salganik et al. 2020, eq. 1):
    the Brier score rescaled by that of predicting the training pool's positive rate.
    """
    y_true = np.asarray(y_test, dtype=int)
    score = np.asarray(y_score, dtype=float)
    train = np.asarray(y_train, dtype=int)
    full = np.asarray(y_train_full, dtype=int)
    clipped = np.clip(score, 1e-15, 1 - 1e-15)
    brier = float(brier_score_loss(y_true, score))
    null_brier = float(np.mean((y_true - np.mean(full)) ** 2)) if len(full) else np.nan
    return {"brier": brier,
            "mcfadden_pseudo_r2": float(_mcfadden_pseudo_r2(y_true, clipped, train)),
            "mcfadden_full_train_pseudo_r2": float(_mcfadden_pseudo_r2(y_true, clipped, full)),
            "brier_full_train_r2": 1.0 - brier / null_brier if null_brier > 0 else np.nan}


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
