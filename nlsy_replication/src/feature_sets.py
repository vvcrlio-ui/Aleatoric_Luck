"""Random feature-count experiments across all A/B predictors."""

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


def parse_args():
    parser = argparse.ArgumentParser(description="Run random feature-count experiments.")
    parser.add_argument("--data", default=str(ROOT / "data" / "asample2_withlag.csv"))
    parser.add_argument("--outcome", default="Cm_lhourlywage")
    parser.add_argument("--out", default=str(ROOT / "outputs" / "feature_sets.csv"))
    parser.add_argument("--models", nargs="+", default=["xgboost"], choices=MODEL_NAMES)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--n-sizes", type=int, default=20)
    parser.add_argument("--n-draws", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )
    return parser.parse_args()


def _write_checkpoint(existing: pd.DataFrame, rows: list[dict], out_path: Path) -> None:
    frames = [frame for frame in (existing, pd.DataFrame(rows)) if not frame.empty]
    if not frames:
        return
    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(["model", "k", "seed"], keep="last")
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    result.sort_values(["model", "k", "seed"]).to_csv(tmp, index=False)
    tmp.replace(out_path)


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
    predictors = [col for col in df.columns if col.startswith(("Aset", "Bset"))]
    if not predictors:
        raise ValueError("No Aset/Bset predictors found in the input data.")

    X_train, X_test, y_train, y_test = train_test_split(
        df[predictors],
        df[args.outcome],
        test_size=args.test_size,
        random_state=args.seed,
    )
    feature_names = np.array(X_train.columns)
    sizes = np.unique(
        np.clip(
            np.logspace(0, np.log2(len(feature_names)), num=args.n_sizes, base=2).astype(int),
            1,
            len(feature_names),
        )
    )

    def run_one(model_name: str, k: int, draw_seed: int) -> dict:
        try:
            rng = np.random.default_rng(draw_seed)
            cols = rng.choice(feature_names, size=k, replace=False)
            model = make_model(model_name, seed=draw_seed, n_jobs=1)
            model.fit(X_train.loc[:, cols], y_train)
            preds = model.predict(X_test.loc[:, cols])
            return {
                "model": model_name,
                "k": int(k),
                "seed": draw_seed,
                "n_features_total": len(feature_names),
                "mse": mean_squared_error(y_test, preds),
                "r2": r2_score(y_test, preds),
                "status": "ok",
                "error": "",
            }
        except Exception as exc:
            return {
                "model": model_name,
                "k": int(k),
                "seed": draw_seed,
                "n_features_total": len(feature_names),
                "mse": np.nan,
                "r2": np.nan,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }

    jobs = [
        (model_name, int(k), args.seed + draw)
        for model_name in args.models
        for k in sizes
        for draw in range(args.n_draws)
    ]
    existing = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
    completed = set()
    if not existing.empty:
        ok = existing[existing.get("status", "ok").eq("ok")] if "status" in existing else existing
        completed = set(zip(ok["model"], ok["k"].astype(int), ok["seed"].astype(int)))
    pending = [job for job in jobs if job not in completed]
    new_rows: list[dict] = []
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        new_rows.extend(
            Parallel(n_jobs=args.n_jobs, batch_size=1, prefer="threads")(
                delayed(run_one)(*job) for job in batch
            )
        )
        _write_checkpoint(existing, new_rows, out_path)


if __name__ == "__main__":
    main()
