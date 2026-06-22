"""Source-aligned cumulative A/B predictor-set comparison."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_registry import MODEL_NAMES, make_model


PREDICTOR_SETS = {
    "c1_Aset1": ("Aset1",),
    "c2_Aset1_Aset2": ("Aset1", "Aset2"),
    "c3_Aset1_Aset2_Bset1": ("Aset1", "Aset2", "Bset1"),
    "c4_Aset1_Aset2_Bset1_Bset2": ("Aset1", "Aset2", "Bset1", "Bset2"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replicate cumulative predictor-set comparisons."
    )
    parser.add_argument("--data", default=str(ROOT / "data" / "asample2_withlag.csv"))
    parser.add_argument("--outcome", default="Cm_lhourlywage")
    parser.add_argument("--out", default=str(ROOT / "outputs" / "overall_prediction.csv"))
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ols", "ridge", "lasso", "xgboost", "bart"],
        choices=MODEL_NAMES,
    )
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path)
    if args.outcome not in df:
        raise KeyError(f"Outcome not found: {args.outcome}")

    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=args.test_size, random_state=args.seed
    )
    y_train = df.iloc[train_idx][args.outcome]
    y_test = df.iloc[test_idx][args.outcome]
    null_mse = mean_squared_error(y_test, np.full(len(y_test), y_train.mean()))

    set_columns = {}
    for label, prefixes in PREDICTOR_SETS.items():
        cols = [col for col in df.columns if col.startswith(prefixes)]
        if not cols:
            raise ValueError(f"No predictors found for {label}: {prefixes}")
        set_columns[label] = cols

    def run_one(model_name: str, set_name: str) -> dict:
        cols = set_columns[set_name]
        try:
            model = make_model(model_name, seed=args.seed, n_jobs=1)
            model.fit(df.iloc[train_idx][cols], y_train)
            pred = model.predict(df.iloc[test_idx][cols])
            mse = mean_squared_error(y_test, pred)
            return {
                "model": model_name,
                "set": set_name,
                "outcome": args.outcome,
                "n_predictors": len(cols),
                "mse": mse,
                "r2": r2_score(y_test, pred),
                "r2_train_mean_baseline": 1 - mse / null_mse,
                "null_mse": null_mse,
                "seed": args.seed,
                "status": "ok",
                "error": "",
            }
        except Exception as exc:
            return {
                "model": model_name,
                "set": set_name,
                "outcome": args.outcome,
                "n_predictors": len(cols),
                "mse": np.nan,
                "r2": np.nan,
                "r2_train_mean_baseline": np.nan,
                "null_mse": null_mse,
                "seed": args.seed,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }

    jobs = [(model, set_name) for model in args.models for set_name in set_columns]
    rows = Parallel(n_jobs=args.n_jobs, batch_size=1, prefer="threads")(
        delayed(run_one)(*job) for job in jobs
    )
    pd.DataFrame(rows).sort_values(["model", "set"]).to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
