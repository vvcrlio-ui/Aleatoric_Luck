"""Metrics and diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred, null_mse: float | None = None) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = float(mean_squared_error(y_true, y_pred))
    row = {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan"),
    }
    if null_mse is not None and null_mse > 0:
        row["relative_null_improvement"] = float(1 - mse / null_mse)
    return row


def null_mse(y_train, y_holdout) -> tuple[float, float]:
    mean = float(np.mean(y_train))
    preds = np.full(len(y_holdout), mean)
    return mean, float(mean_squared_error(y_holdout, preds))


def range_diagnostics(
    predictions: pd.DataFrame,
    y_train: pd.Series,
    y_holdout: pd.Series,
    target_col: str = "prediction",
) -> pd.DataFrame:
    train_min, train_max = float(y_train.min()), float(y_train.max())
    holdout_min, holdout_max = float(y_holdout.min()), float(y_holdout.max())
    rows = []
    group_cols = ["model"]
    if "imputation_strategy" in predictions.columns:
        group_cols.append("imputation_strategy")
    for group_key, group in predictions.groupby(group_cols):
        if isinstance(group_key, tuple):
            model = group_key[0]
            imputation_strategy = group_key[1]
        else:
            model = group_key
            imputation_strategy = None
        pred = group[target_col].astype(float)
        row = {
            "model": model,
            "train_target_min": train_min,
            "train_target_max": train_max,
            "holdout_target_min": holdout_min,
            "holdout_target_max": holdout_max,
            "prediction_min": float(pred.min()),
            "prediction_max": float(pred.max()),
            "share_below_train_range": float((pred < train_min).mean()),
            "share_above_train_range": float((pred > train_max).mean()),
            "share_below_holdout_range": float((pred < holdout_min).mean()),
            "share_above_holdout_range": float((pred > holdout_max).mean()),
        }
        if imputation_strategy is not None:
            row["imputation_strategy"] = imputation_strategy
        rows.append(row)
    return pd.DataFrame(rows)
