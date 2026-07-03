"""Training-sample learning curves and guarded power-law extrapolation."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation import r2_against_training_mean, training_mean_null_mse
from experiment import (
    add_metadata,
    build_experiment_metadata,
    load_checkpoint,
    model_run_settings,
    parallel_preference,
    rows_for_experiment,
    write_checkpoint,
)
from model_registry import MODEL_NAMES, make_model


def power_law(n, c, alpha, epsilon):
    return c * np.power(n, -alpha) + epsilon


def fit_curve(n: np.ndarray, mse: np.ndarray):
    n = np.asarray(n, dtype=float)
    mse = np.asarray(mse, dtype=float)
    p0 = [max(mse) - min(mse), 0.5, max(0.0, min(mse) * 0.9)]
    return curve_fit(
        power_law,
        n,
        mse,
        p0=p0,
        bounds=([0.0, 1e-8, 0.0], [np.inf, 5.0, np.inf]),
        maxfev=50000,
    )


def fit_power_law(
    results_df: pd.DataFrame,
    bootstrap_iterations: int = 500,
    seed: int = 12345,
) -> pd.DataFrame:
    """Fit each model independently; one failed fit never aborts the run."""

    rows = []
    usable = results_df
    if "status" in usable:
        usable = usable[usable["status"].eq("ok")]
    for model_name, group in usable.groupby("model"):
        try:
            grouped = {
                int(n): values["mse"].dropna().to_numpy(dtype=float)
                for n, values in group.groupby("n_samples")
            }
            grouped = {n: values for n, values in grouped.items() if len(values)}
            if len(grouped) < 4:
                raise ValueError("At least four distinct training sizes are required.")
            ns = np.array(sorted(grouped), dtype=float)
            means = np.array([grouped[int(n)].mean() for n in ns])
            params, covariance = fit_curve(ns, means)
            c_hat, alpha_hat, epsilon_hat = [float(value) for value in params]
            covariance_se = float(np.sqrt(np.diag(covariance))[2])

            rng = np.random.default_rng(seed)
            boot_eps = []
            failed = 0
            for _ in range(bootstrap_iterations):
                try:
                    boot_means = np.array(
                        [
                            rng.choice(
                                grouped[int(n)],
                                size=len(grouped[int(n)]),
                                replace=True,
                            ).mean()
                            for n in ns
                        ]
                    )
                    boot_eps.append(float(fit_curve(ns, boot_means)[0][2]))
                except Exception:
                    failed += 1
            if not boot_eps:
                raise RuntimeError("All bootstrap power-law fits failed.")
            ci_low, ci_high = np.percentile(boot_eps, [2.5, 97.5])
            fail_rate = failed / max(1, bootstrap_iterations)
            status = "stable"
            reasons = []
            if epsilon_hat <= 1e-10:
                reasons.append("epsilon_at_lower_bound")
            if alpha_hat <= 1e-6 or alpha_hat >= 4.99:
                reasons.append("alpha_at_bound")
            if not np.isfinite(covariance_se) or covariance_se > max(epsilon_hat, 1e-8):
                reasons.append("epsilon_weakly_identified")
            if fail_rate > 0.10:
                reasons.append("bootstrap_fail_rate_gt_10pct")
            if reasons:
                status = "unstable"
            rows.append(
                {
                    "model": model_name,
                    "c": c_hat,
                    "alpha": alpha_hat,
                    "epsilon": epsilon_hat,
                    "epsilon_covariance_se": covariance_se,
                    "epsilon_ci_low": float(ci_low),
                    "epsilon_ci_high": float(ci_high),
                    "bootstrap_fail_rate": fail_rate,
                    "status": status,
                    "diagnostic": ";".join(reasons),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "model": model_name,
                    "c": np.nan,
                    "alpha": np.nan,
                    "epsilon": np.nan,
                    "epsilon_covariance_se": np.nan,
                    "epsilon_ci_low": np.nan,
                    "epsilon_ci_high": np.nan,
                    "bootstrap_fail_rate": 1.0,
                    "status": "fit_failed",
                    "diagnostic": f"{type(exc).__name__}: {exc}",
                }
            )
    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Run sample-size learning curves.")
    parser.add_argument("--data", default=str(ROOT / "data" / "asample2_withlag.csv"))
    parser.add_argument("--outcome", default="Cm_lhourlywage")
    parser.add_argument("--out", default=str(ROOT / "outputs" / "sample_size.csv"))
    parser.add_argument(
        "--power-law-out",
        default=str(ROOT / "outputs" / "sample_size_power_law.csv"),
    )
    parser.add_argument("--models", nargs="+", default=["xgboost"], choices=MODEL_NAMES)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--n-sizes", type=int, default=10)
    parser.add_argument("--n-draws", type=int, default=20)
    parser.add_argument("--min-fraction", type=float, default=0.1)
    parser.add_argument("--max-fraction", type=float, default=1.0)
    parser.add_argument("--bootstrap-iterations", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--skip-power-law", action="store_true")
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
    power_path = Path(args.power_law_out)
    power_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    if args.outcome not in df:
        raise KeyError(f"Outcome not found: {args.outcome}")
    predictors = [col for col in df.columns if col.startswith(("Aset", "Bset"))]
    if not predictors:
        raise ValueError("No Aset/Bset predictors found in the input data.")
    metadata = build_experiment_metadata(
        kind="sample_size",
        data_path=data_path,
        outcome=args.outcome,
        test_size=args.test_size,
        split_seed=args.seed,
        extra={
            "n_sizes": args.n_sizes,
            "min_fraction": args.min_fraction,
            "max_fraction": args.max_fraction,
            **model_run_settings(args.models),
        },
    )
    X_train, X_test, y_train, y_test = train_test_split(
        df[predictors],
        df[args.outcome],
        test_size=args.test_size,
        random_state=args.seed,
    )
    train_sizes = np.unique(
        np.clip(
            np.round(
                np.linspace(args.min_fraction, args.max_fraction, args.n_sizes)
                * len(X_train)
            ).astype(int),
            2,
            len(X_train),
        )
    )

    def run_one(model_name: str, n_samples: int, draw_seed: int) -> dict:
        try:
            if n_samples >= len(X_train):
                X_sub, y_sub = X_train, y_train
            else:
                X_sub = X_train.sample(
                    n=n_samples, replace=False, random_state=draw_seed
                )
                y_sub = y_train.loc[X_sub.index]
            model = make_model(model_name, seed=draw_seed, n_jobs=1)
            model.fit(X_sub, y_sub)
            preds = model.predict(X_test)
            mse = mean_squared_error(y_test, preds)
            subset_null_mse = training_mean_null_mse(y_test, y_sub)
            return add_metadata(
                {
                    "model": model_name,
                    "n_samples": int(n_samples),
                    "seed": draw_seed,
                    "n_train_total": len(X_train),
                    "mse": mse,
                    "null_mse_train_subset": subset_null_mse,
                    "r2_test_mean_baseline": r2_score(y_test, preds),
                    "r2_train_mean_baseline": r2_against_training_mean(
                        mse, y_test, y_sub
                    ),
                    "status": "ok",
                    "error": "",
                },
                metadata,
            )
        except Exception as exc:
            return add_metadata(
                {
                    "model": model_name,
                    "n_samples": int(n_samples),
                    "seed": draw_seed,
                    "n_train_total": len(X_train),
                    "mse": np.nan,
                    "null_mse_train_subset": np.nan,
                    "r2_test_mean_baseline": np.nan,
                    "r2_train_mean_baseline": np.nan,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                metadata,
            )

    jobs = [
        (model, int(n), args.seed + draw)
        for model in args.models
        for n in train_sizes
        for draw in range(args.n_draws)
    ]
    existing = load_checkpoint(out_path)
    current = rows_for_experiment(existing, metadata["experiment_id"])
    completed = set()
    if not current.empty:
        ok = current[current["status"].eq("ok")] if "status" in current else current
        completed = set(
            zip(ok["model"], ok["n_samples"].astype(int), ok["seed"].astype(int))
        )
    pending = [job for job in jobs if job not in completed]
    rows: list[dict] = []
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        rows.extend(
            Parallel(
                n_jobs=args.n_jobs,
                batch_size=1,
                prefer=parallel_preference(args.models),
            )(
                delayed(run_one)(*job) for job in batch
            )
        )
        write_checkpoint(
            existing,
            rows,
            out_path,
            key_columns=["model", "n_samples", "seed"],
            sort_columns=["model", "n_samples", "seed"],
        )

    if not out_path.exists():
        write_checkpoint(
            existing,
            rows,
            out_path,
            key_columns=["model", "n_samples", "seed"],
            sort_columns=["model", "n_samples", "seed"],
        )
    if not args.skip_power_law:
        results = pd.read_csv(out_path)
        results = rows_for_experiment(results, metadata["experiment_id"])
        fits = fit_power_law(
            results,
            bootstrap_iterations=args.bootstrap_iterations,
            seed=args.seed,
        )
        for key, value in metadata.items():
            fits[key] = value
        fits.to_csv(power_path, index=False)


if __name__ == "__main__":
    main()
