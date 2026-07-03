"""Joint N x K sweeps for long-format prediction quality tables."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_pinball_loss,
    mean_squared_error,
    median_absolute_error,
)
from sklearn.model_selection import train_test_split

try:
    from sklearn.metrics import d2_absolute_error_score
except ImportError:

    def d2_absolute_error_score(y_true, y_pred) -> float:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        numerator = np.sum(np.abs(y_true - y_pred))
        denominator = np.sum(np.abs(y_true - np.median(y_true)))
        if denominator == 0:
            return np.nan
        return 1.0 - numerator / denominator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation import r2_against_training_mean
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


METRIC_COLUMNS = (
    "r2_test",
    "skill_score_pct",
    "rmse",
    "mae",
    "medae",
    "max_error",
    "nrmse",
    "spearman_rho",
    "pearson_r",
    "kendall_tau",
    "ccc",
    "explained_variance",
    "mean_bias",
    "median_bias",
    "pinball_q10",
    "pinball_q90",
    "d2_absolute_error",
)


@dataclass(frozen=True)
class NKGridConfig:
    data: Path
    out: Path
    dataset: str
    outcome: str
    models: tuple[str, ...]
    seed: int
    test_size: float
    n_seeds: int
    n_draws: int
    n_sizes_n: int
    n_sizes_k: int
    max_n: int
    max_k: int
    batch_size: int
    n_jobs: int
    group_split_col: str | None = None


@dataclass(frozen=True)
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


@dataclass(frozen=True)
class DrawOrders:
    row_index: np.ndarray
    feature_names: np.ndarray


def log2_size_grid(total: int, n_sizes: int, max_size: int | None = None) -> np.ndarray:
    """Return unique integer sizes on the shared base-2 log grid."""

    if total < 1:
        raise ValueError("total must be at least 1")
    if n_sizes < 1:
        raise ValueError("n_sizes must be at least 1")
    upper = int(total if max_size is None or max_size <= 0 else min(total, max_size))
    upper = max(1, upper)
    if n_sizes == 1:
        return np.array([upper], dtype=int)
    return np.unique(
        np.clip(
            np.round(np.logspace(0, np.log2(upper), num=n_sizes, base=2)).astype(int),
            1,
            upper,
        )
    )


def split_frame(
    frame: pd.DataFrame,
    predictors: Sequence[str],
    outcome: str,
    *,
    test_size: float,
    seed: int,
) -> SplitData:
    X_train, X_test, y_train, y_test = train_test_split(
        frame.loc[:, list(predictors)],
        frame[outcome],
        test_size=test_size,
        random_state=seed,
    )
    return SplitData(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def draw_orders(
    train_index: Sequence,
    feature_names: Sequence[str],
    *,
    seed: int,
    draw: int,
) -> DrawOrders:
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(draw)]))
    rows = np.asarray(list(train_index))
    features = np.asarray(list(feature_names))
    return DrawOrders(
        row_index=rows[rng.permutation(len(rows))],
        feature_names=features[rng.permutation(len(features))],
    )


def _as_float_array(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _bounded_statistic(result) -> float:
    if isinstance(result, tuple):
        result = result[0]
    statistic = getattr(result, "statistic", result)
    return float(statistic) if np.isfinite(statistic) else np.nan


def _correlation_statistic(y_true: np.ndarray, y_pred: np.ndarray, func) -> float:
    if len(y_true) < 2 or len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return np.nan
    return _bounded_statistic(func(y_true, y_pred))


def _concordance_correlation_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_mean = float(np.mean(y_true))
    pred_mean = float(np.mean(y_pred))
    true_var = float(np.var(y_true))
    pred_var = float(np.var(y_pred))
    covariance = float(np.mean((y_true - true_mean) * (y_pred - pred_mean)))
    denominator = true_var + pred_var + (true_mean - pred_mean) ** 2
    if denominator == 0:
        return np.nan
    return 2.0 * covariance / denominator


def compute_regression_metrics(y_test, y_pred, y_train) -> dict[str, float]:
    """Compute continuous-outcome metrics for one fitted model run."""

    y_true = _as_float_array(y_test)
    preds = _as_float_array(y_pred)
    train = _as_float_array(y_train)
    mse = float(mean_squared_error(y_true, preds))
    rmse = float(np.sqrt(mse))
    y_range = float(np.max(y_true) - np.min(y_true))
    r2_test = float(r2_against_training_mean(mse, y_true, train))
    return {
        "r2_test": r2_test,
        "skill_score_pct": 100.0 * r2_test,
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, preds)),
        "medae": float(median_absolute_error(y_true, preds)),
        "max_error": float(max_error(y_true, preds)),
        "nrmse": rmse / y_range if y_range > 0 else np.nan,
        "spearman_rho": _correlation_statistic(y_true, preds, stats.spearmanr),
        "pearson_r": _correlation_statistic(y_true, preds, stats.pearsonr),
        "kendall_tau": _correlation_statistic(y_true, preds, stats.kendalltau),
        "ccc": float(_concordance_correlation_coefficient(y_true, preds)),
        "explained_variance": float(explained_variance_score(y_true, preds)),
        "mean_bias": float(np.mean(preds - y_true)),
        "median_bias": float(np.median(preds - y_true)),
        "pinball_q10": float(mean_pinball_loss(y_true, preds, alpha=0.10)),
        "pinball_q90": float(mean_pinball_loss(y_true, preds, alpha=0.90)),
        "d2_absolute_error": float(d2_absolute_error_score(y_true, preds)),
    }


def _empty_metrics() -> dict[str, float]:
    return {column: np.nan for column in METRIC_COLUMNS}


def _model_seed(seed: int, draw: int, n_samples: int, k_features: int) -> int:
    return int(
        np.random.SeedSequence(
            [int(seed), int(draw), int(n_samples), int(k_features)]
        ).generate_state(1)[0]
    )


def _predictor_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in frame.columns if col.startswith(("Aset", "Bset"))]


def _base_row(
    *,
    dataset: str,
    outcome: str,
    model_name: str,
    seed: int,
    draw: int,
    n_samples: int,
    k_features: int,
    n_train_total: int,
    n_features_total: int,
) -> dict:
    return {
        "dataset": dataset,
        "outcome": outcome,
        "model": model_name,
        "seed": int(seed),
        "draw": int(draw),
        "N": int(n_samples),
        "K": int(k_features),
        "split_random_state": int(seed),
        "n_train_total": int(n_train_total),
        "n_features_total": int(n_features_total),
    }


def run_nk_grid(config: NKGridConfig, *, max_jobs: int | None = None) -> None:
    if config.group_split_col:
        raise NotImplementedError(
            "--group-split-col is reserved for the sibling-clustering confirmation item."
        )
    data_path = Path(config.data)
    if not data_path.exists():
        raise FileNotFoundError(f"NLSY analysis data not found: {data_path}")

    out_path = Path(config.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(data_path)
    if config.outcome not in frame:
        raise KeyError(f"Outcome not found: {config.outcome}")
    predictors = _predictor_columns(frame)
    if not predictors:
        raise ValueError("No Aset/Bset predictors found in the input data.")

    metadata = build_experiment_metadata(
        kind="nk_grid",
        data_path=data_path,
        outcome=config.outcome,
        test_size=config.test_size,
        split_seed=config.seed,
        extra={
            "dataset": config.dataset,
            "n_seeds": config.n_seeds,
            "n_draws": config.n_draws,
            "n_sizes_n": config.n_sizes_n,
            "n_sizes_k": config.n_sizes_k,
            "max_n": config.max_n,
            "max_k": config.max_k,
            "group_split_col": config.group_split_col,
            **model_run_settings(config.models),
        },
    )

    split_seeds = [config.seed + offset for offset in range(config.n_seeds)]
    splits = {
        seed: split_frame(
            frame,
            predictors,
            config.outcome,
            test_size=config.test_size,
            seed=seed,
        )
        for seed in split_seeds
    }
    n_grid = log2_size_grid(
        len(next(iter(splits.values())).X_train),
        config.n_sizes_n,
        config.max_n,
    )
    k_grid = log2_size_grid(len(predictors), config.n_sizes_k, config.max_k)

    jobs = [
        (model_name, seed, draw, int(n_samples), int(k_features))
        for seed in split_seeds
        for draw in range(config.n_draws)
        for k_features in k_grid
        for n_samples in n_grid
        for model_name in config.models
    ]

    existing = load_checkpoint(out_path)
    current = rows_for_experiment(existing, metadata["experiment_id"])
    completed = set()
    if not current.empty:
        ok = current[current["status"].eq("ok")] if "status" in current else current
        completed = set(
            zip(
                ok["model"],
                ok["seed"].astype(int),
                ok["draw"].astype(int),
                ok["N"].astype(int),
                ok["K"].astype(int),
            )
        )
    pending = [job for job in jobs if job not in completed]
    if max_jobs is not None:
        pending = pending[: int(max_jobs)]

    def run_one(
        model_name: str,
        seed: int,
        draw: int,
        n_samples: int,
        k_features: int,
    ) -> dict:
        split = splits[seed]
        row = _base_row(
            dataset=config.dataset,
            outcome=config.outcome,
            model_name=model_name,
            seed=seed,
            draw=draw,
            n_samples=n_samples,
            k_features=k_features,
            n_train_total=len(split.X_train),
            n_features_total=len(predictors),
        )
        try:
            orders = draw_orders(split.X_train.index, predictors, seed=seed, draw=draw)
            selected_rows = orders.row_index[:n_samples]
            selected_cols = orders.feature_names[:k_features]
            X_sub = split.X_train.loc[selected_rows, selected_cols]
            y_sub = split.y_train.loc[selected_rows]
            X_test = split.X_test.loc[:, selected_cols]
            model = make_model(
                model_name,
                seed=_model_seed(seed, draw, n_samples, k_features),
                n_jobs=1,
            )
            model.fit(X_sub, y_sub)
            preds = model.predict(X_test)
            return add_metadata(
                {
                    **row,
                    **compute_regression_metrics(split.y_test, preds, y_sub),
                    "status": "ok",
                    "error": "",
                },
                metadata,
            )
        except Exception as exc:
            return add_metadata(
                {
                    **row,
                    **_empty_metrics(),
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                metadata,
            )

    rows: list[dict] = []
    for start in range(0, len(pending), config.batch_size):
        batch = pending[start : start + config.batch_size]
        rows.extend(
            Parallel(
                n_jobs=config.n_jobs,
                batch_size=1,
                prefer=parallel_preference(config.models),
            )(delayed(run_one)(*job) for job in batch)
        )
        write_checkpoint(
            existing,
            rows,
            out_path,
            key_columns=["model", "seed", "draw", "N", "K"],
            sort_columns=["model", "seed", "draw", "N", "K"],
        )


def parse_args() -> NKGridConfig:
    parser = argparse.ArgumentParser(
        description="Run joint log-scale N x K prediction-quality sweeps."
    )
    parser.add_argument("--data", default=str(ROOT / "data" / "asample2_withlag.csv"))
    parser.add_argument("--outcome", default="Cm_lhourlywage")
    parser.add_argument("--out", default=str(ROOT / "outputs" / "nk_grid.csv"))
    parser.add_argument("--dataset", default="asample2_withlag")
    parser.add_argument("--models", nargs="+", default=["xgboost"], choices=MODEL_NAMES)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--n-seeds", type=int, default=2)
    parser.add_argument("--n-draws", type=int, default=2)
    parser.add_argument("--n-sizes-n", type=int, default=4)
    parser.add_argument("--n-sizes-k", type=int, default=4)
    parser.add_argument("--max-n", type=int, default=100, help="Use <=0 for full train set.")
    parser.add_argument("--max-k", type=int, default=100, help="Use <=0 for all features.")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--group-split-col", default=None)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )
    args = parser.parse_args()
    return NKGridConfig(
        data=Path(args.data),
        out=Path(args.out),
        dataset=args.dataset,
        outcome=args.outcome,
        models=tuple(args.models),
        seed=args.seed,
        test_size=args.test_size,
        n_seeds=args.n_seeds,
        n_draws=args.n_draws,
        n_sizes_n=args.n_sizes_n,
        n_sizes_k=args.n_sizes_k,
        max_n=args.max_n,
        max_k=args.max_k,
        batch_size=args.batch_size,
        n_jobs=args.n_jobs,
        group_split_col=args.group_split_col,
    )


def main() -> None:
    run_nk_grid(parse_args())


if __name__ == "__main__":
    main()
