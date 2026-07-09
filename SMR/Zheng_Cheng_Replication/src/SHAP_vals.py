"""Fit the source-style XGBoost model and export mean absolute SHAP values."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

MATPLOTLIB_CACHE = Path(os.environ.get("TMPDIR", "/tmp")) / "aleatoric-matplotlib"
MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE))

import shap
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Compute XGBoost SHAP importance.")
    parser.add_argument("--data", default=str(ROOT / "data" / "asample2_withlag.csv"))
    parser.add_argument("--outcome", default="Cm_lhourlywage")
    parser.add_argument("--out", default=str(ROOT / "outputs" / "shap_importance.csv"))
    parser.add_argument("--metrics-out", default=str(ROOT / "outputs" / "shap_model_metrics.csv"))
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"NLSY analysis data not found: {data_path}")
    out_path = Path(args.out)
    metrics_path = Path(args.metrics_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    predictors = [col for col in df.columns if col.startswith(("Aset", "Bset"))]
    X_train, X_test, y_train, y_test = train_test_split(
        df[predictors],
        df[args.outcome],
        test_size=args.test_size,
        random_state=args.seed,
    )
    model = xgb.XGBRegressor(
        n_estimators=90,
        max_depth=2,
        learning_rate=0.3,
        objective="reg:squarederror",
        random_state=args.seed,
        verbosity=0,
        n_jobs=args.n_jobs,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    pd.DataFrame(
        [
            {
                "model": "xgboost_source_fixed_90",
                "mse": mean_squared_error(y_test, preds),
                "r2_test_mean_baseline": r2_score(y_test, preds),
                "seed": args.seed,
            }
        ]
    ).to_csv(metrics_path, index=False)

    shap_values = shap.TreeExplainer(model).shap_values(X_train)
    importance = pd.Series(
        np.abs(shap_values).mean(axis=0), index=predictors, name="mean_abs_shap"
    ).sort_values(ascending=False)
    importance.rename_axis("feature").reset_index().to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
