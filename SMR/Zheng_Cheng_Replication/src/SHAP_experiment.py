"""Compare top-k and bottom-k SHAP-ranked feature sets."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment import (
    add_metadata,
    build_experiment_metadata,
    file_sha256,
    load_checkpoint,
    rows_for_experiment,
    write_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run SHAP-ordering experiments.")
    parser.add_argument("--data", default=str(ROOT / "data" / "asample2_withlag.csv"))
    parser.add_argument("--importance", default=str(ROOT / "outputs" / "shap_importance.csv"))
    parser.add_argument("--out", default=str(ROOT / "outputs" / "shap_ordering_results.csv"))
    parser.add_argument("--outcome", default="Cm_lhourlywage")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )
    return parser.parse_args()


def evaluate_feature_count(
    k: int,
    ordered_features: list[str],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    seed: int,
) -> list[dict]:
    """Return both top-k and bottom-k results; neither direction is skipped."""

    rows = []
    selections = {
        "high_to_low": ordered_features[:k],
        "low_to_high": ordered_features[-k:],
    }
    for direction, features in selections.items():
        try:
            model = xgb.XGBRegressor(
                n_estimators=90,
                max_depth=2,
                learning_rate=0.3,
                objective="reg:squarederror",
                random_state=seed,
                verbosity=0,
                n_jobs=1,
            )
            model.fit(X_train[features], y_train)
            preds = model.predict(X_test[features])
            rows.append(
                {
                    "k": k,
                    "direction": direction,
                    "mse": mean_squared_error(y_test, preds),
                    "r2_test_mean_baseline": r2_score(y_test, preds),
                    "status": "ok",
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "k": k,
                    "direction": direction,
                    "mse": float("nan"),
                    "r2_test_mean_baseline": float("nan"),
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return rows


def main():
    args = parse_args()
    data_path = Path(args.data)
    importance_path = Path(args.importance)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / "logs" / "shap_experiment.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    if not data_path.exists():
        raise FileNotFoundError(f"NLSY analysis data not found: {data_path}")
    if not importance_path.exists():
        raise FileNotFoundError(
            f"SHAP importance not found: {importance_path}. Run SHAP_vals.py first."
        )

    df = pd.read_csv(data_path)
    ordered = pd.read_csv(importance_path)["feature"].tolist()
    missing = [feature for feature in ordered if feature not in df]
    if missing:
        raise KeyError(f"SHAP file contains {len(missing)} features absent from data.")
    metadata = build_experiment_metadata(
        kind="shap_ordering",
        data_path=data_path,
        outcome=args.outcome,
        test_size=args.test_size,
        split_seed=args.seed,
        extra={"importance_sha256": file_sha256(importance_path)},
    )
    X_train, X_test, y_train, y_test = train_test_split(
        df[ordered],
        df[args.outcome],
        test_size=args.test_size,
        random_state=args.seed,
    )
    existing = load_checkpoint(out_path)
    current = rows_for_experiment(existing, metadata["experiment_id"])
    completed = set()
    if not current.empty:
        ok = current[current["status"].eq("ok")] if "status" in current else current
        completed = set(zip(ok["k"].astype(int), ok["direction"]))
    pending = [
        k
        for k in range(1, len(ordered) + 1)
        if (k, "high_to_low") not in completed or (k, "low_to_high") not in completed
    ]
    rows: list[dict] = []
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        batch_results = Parallel(n_jobs=args.n_jobs, batch_size=1)(
            delayed(evaluate_feature_count)(
                k, ordered, X_train, X_test, y_train, y_test, args.seed
            )
            for k in batch
        )
        for result in batch_results:
            rows.extend(add_metadata(row, metadata) for row in result)
        write_checkpoint(
            existing,
            rows,
            out_path,
            key_columns=["k", "direction"],
            sort_columns=["k", "direction"],
        )
        logging.info(
            "Saved %d/%d feature counts",
            min(start + len(batch), len(pending)),
            len(pending),
        )


if __name__ == "__main__":
    main()
