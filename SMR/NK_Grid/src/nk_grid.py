"""Joint N x K sweeps for long-format prediction quality tables."""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    explained_variance_score,
    f1_score,
    log_loss,
    max_error,
    mean_absolute_error,
    mean_pinball_loss,
    mean_squared_error,
    median_absolute_error,
    roc_auc_score,
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

try:
    from .helpers_logging import log_progress
except ImportError:
    from helpers_logging import log_progress

from evaluation import r2_against_training_mean
from experiment import (
    add_metadata,
    build_experiment_metadata,
    file_sha256,
    load_checkpoint,
    model_run_settings,
    parallel_preference,
    rows_for_experiment,
    write_checkpoint,
)
from model_registry import (
    DEFAULT_MODEL_PARAMS_PATH,
    SUPPORTED_MODEL_NAMES,
    load_model_params,
    make_model,
)


REGRESSION_CV_MIN_N = {
    "ridge": 2,
    "lasso": 2,
    "elastic_net": 2,
    "lightgbm": 5,
    # Keep this aligned with super_learner.cv in model_params.yaml.
    "super_learner": 5,
}


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
    "pinball_q05",
    "pinball_q25",
    "pinball_q50",
    "pinball_q75",
    "pinball_q95",
    "ks_statistic",
    "wasserstein_distance",
    "top_decile_hit_rate",
    "bottom_decile_hit_rate",
    "rsr",
    "cv_rmse",
    "mase",
    "pearson_r2",
)

CLASSIFICATION_METRIC_COLUMNS = (
    "roc_auc",
    "pr_auc",
    "brier",
    "log_loss",
    "balanced_accuracy",
    "f1",
    "accuracy",
    "mcfadden_pseudo_r2",
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
    min_n: int = 10
    model_params: Path = DEFAULT_MODEL_PARAMS_PATH
    test_data: Path | None = None
    group_split_col: str | None = None
    task: str = "regression"
    bart_min_n: int = 10
    bart_min_k: int = 2
    predictor_prefix: tuple[str, ...] = ("Aset", "Bset")
    preset: str | None = None


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


def log2_size_grid(
    total: int,
    n_sizes: int,
    max_size: int | None = None,
    *,
    min_size: int = 1,
) -> np.ndarray:
    """Return unique integer sizes on the shared base-2 log grid."""

    if total < 1:
        raise ValueError("total must be at least 1")
    if n_sizes < 1:
        raise ValueError("n_sizes must be at least 1")
    if min_size < 1:
        raise ValueError("min_size must be at least 1")
    upper = int(total if max_size is None or max_size <= 0 else min(total, max_size))
    if upper < min_size:
        raise ValueError(
            f"grid upper bound {upper} is below minimum size {min_size}"
        )
    if n_sizes == 1:
        return np.array([upper], dtype=int)
    return np.unique(
        np.clip(
            np.round(
                np.logspace(
                    np.log2(min_size), np.log2(upper), num=n_sizes, base=2
                )
            ).astype(int),
            min_size,
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
    task: str = "regression",
) -> SplitData:
    y = frame[outcome]
    X_train, X_test, y_train, y_test = train_test_split(
        frame.loc[:, list(predictors)],
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y if task == "classification" else None,
    )
    return SplitData(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def external_test_split(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    predictors: Sequence[str],
    outcome: str,
) -> SplitData:
    for label, frame in (("training data", train_frame), ("test data", test_frame)):
        if outcome not in frame:
            raise KeyError(f"Outcome not found in {label}: {outcome}")
        for predictor in predictors:
            if predictor not in frame:
                raise KeyError(f"Predictor not found in {label}: {predictor}")

    train_complete = train_frame.dropna(subset=[outcome])
    test_complete = test_frame.dropna(subset=[outcome])
    predictor_list = list(predictors)
    return SplitData(
        X_train=train_complete.loc[:, predictor_list],
        X_test=test_complete.loc[:, predictor_list],
        y_train=train_complete[outcome],
        y_test=test_complete[outcome],
    )


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
    mae = float(mean_absolute_error(y_true, preds))
    y_range = float(np.max(y_true) - np.min(y_true))
    try:
        r2_test = float(r2_against_training_mean(mse, y_true, train))
    except ZeroDivisionError:
        r2_test = np.nan
    pearson_r = _correlation_statistic(y_true, preds, stats.pearsonr)
    y_std = float(np.std(y_true))
    y_mean = float(np.mean(y_true))
    train_mean_absolute_error = float(np.mean(np.abs(y_true - np.mean(train))))
    top_true = y_true >= np.quantile(y_true, 0.90)
    top_pred = preds >= np.quantile(preds, 0.90)
    bottom_true = y_true <= np.quantile(y_true, 0.10)
    bottom_pred = preds <= np.quantile(preds, 0.10)
    return {
        "r2_test": r2_test,
        "skill_score_pct": 100.0 * r2_test,
        "rmse": rmse,
        "mae": mae,
        "medae": float(median_absolute_error(y_true, preds)),
        "max_error": float(max_error(y_true, preds)),
        "nrmse": rmse / y_range if y_range > 0 else np.nan,
        "spearman_rho": _correlation_statistic(y_true, preds, stats.spearmanr),
        "pearson_r": pearson_r,
        "kendall_tau": _correlation_statistic(y_true, preds, stats.kendalltau),
        "ccc": float(_concordance_correlation_coefficient(y_true, preds)),
        "explained_variance": float(explained_variance_score(y_true, preds)),
        "mean_bias": float(np.mean(preds - y_true)),
        "median_bias": float(np.median(preds - y_true)),
        "pinball_q10": float(mean_pinball_loss(y_true, preds, alpha=0.10)),
        "pinball_q90": float(mean_pinball_loss(y_true, preds, alpha=0.90)),
        "d2_absolute_error": float(d2_absolute_error_score(y_true, preds)),
        "pinball_q05": float(mean_pinball_loss(y_true, preds, alpha=0.05)),
        "pinball_q25": float(mean_pinball_loss(y_true, preds, alpha=0.25)),
        "pinball_q50": float(mean_pinball_loss(y_true, preds, alpha=0.50)),
        "pinball_q75": float(mean_pinball_loss(y_true, preds, alpha=0.75)),
        "pinball_q95": float(mean_pinball_loss(y_true, preds, alpha=0.95)),
        "ks_statistic": float(stats.ks_2samp(y_true, preds).statistic),
        "wasserstein_distance": float(stats.wasserstein_distance(y_true, preds)),
        "top_decile_hit_rate": (
            float(np.sum(top_true & top_pred) / np.sum(top_true))
            if np.sum(top_true) > 0
            else np.nan
        ),
        "bottom_decile_hit_rate": (
            float(np.sum(bottom_true & bottom_pred) / np.sum(bottom_true))
            if np.sum(bottom_true) > 0
            else np.nan
        ),
        "rsr": rmse / y_std if y_std != 0 else np.nan,
        "cv_rmse": rmse / y_mean if y_mean != 0 else np.nan,
        "mase": (
            mae / train_mean_absolute_error
            if train_mean_absolute_error != 0
            else np.nan
        ),
        "pearson_r2": pearson_r**2 if np.isfinite(pearson_r) else np.nan,
    }


def compute_classification_metrics(y_test, y_score, y_train) -> dict[str, float]:
    """Compute binary classification metrics from positive-class probabilities."""

    y_true = np.asarray(y_test, dtype=int)
    score = np.asarray(y_score, dtype=float)
    train = np.asarray(y_train, dtype=int)
    labels = (score >= 0.5).astype(int)
    has_two_test_classes = len(np.unique(y_true)) == 2
    finite_scores = np.all(np.isfinite(score))
    if finite_scores:
        clipped = np.clip(score, 1e-15, 1 - 1e-15)
    else:
        clipped = score

    positive_rate = float(np.mean(train)) if len(train) else np.nan
    if (
        has_two_test_classes
        and finite_scores
        and np.isfinite(positive_rate)
        and 0.0 < positive_rate < 1.0
    ):
        model_loglik = float(
            np.sum(y_true * np.log(clipped) + (1 - y_true) * np.log(1 - clipped))
        )
        null_loglik = float(
            np.sum(
                y_true * np.log(positive_rate)
                + (1 - y_true) * np.log(1 - positive_rate)
            )
        )
        mcfadden = 1.0 - model_loglik / null_loglik if null_loglik != 0 else np.nan
    else:
        mcfadden = np.nan

    return {
        "roc_auc": (
            float(roc_auc_score(y_true, score))
            if has_two_test_classes and finite_scores
            else np.nan
        ),
        "pr_auc": (
            float(average_precision_score(y_true, score))
            if has_two_test_classes and finite_scores
            else np.nan
        ),
        "brier": float(brier_score_loss(y_true, score)) if finite_scores else np.nan,
        "log_loss": (
            float(log_loss(y_true, clipped, labels=[0, 1]))
            if has_two_test_classes and finite_scores
            else np.nan
        ),
        "balanced_accuracy": (
            float(balanced_accuracy_score(y_true, labels))
            if has_two_test_classes
            else np.nan
        ),
        "f1": float(f1_score(y_true, labels, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, labels)),
        "mcfadden_pseudo_r2": float(mcfadden),
    }


def _empty_metrics() -> dict[str, float]:
    return {column: np.nan for column in METRIC_COLUMNS}


def _empty_classification_metrics() -> dict[str, float]:
    return {column: np.nan for column in CLASSIFICATION_METRIC_COLUMNS}


def _model_seed(seed: int, draw: int, n_samples: int, k_features: int) -> int:
    return int(
        np.random.SeedSequence(
            [int(seed), int(draw), int(n_samples), int(k_features)]
        ).generate_state(1)[0]
    )


def _completed_jobs_for_experiment(existing: pd.DataFrame, experiment_id: str) -> set[tuple]:
    current = rows_for_experiment(existing, experiment_id)
    if current.empty:
        return set()
    ok = (
        current[current["status"].isin(("ok", "skipped"))]
        if "status" in current
        else current
    )
    return set(
        zip(
            ok["model"],
            ok["seed"].astype(int),
            ok["draw"].astype(int),
            ok["N"].astype(int),
            ok["K"].astype(int),
        )
    )


def _timestamped_out_path(directory: Path, stem: str, preset: str, suffix: str) -> Path:
    while True:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = directory / f"{stem}_{preset}_{timestamp}{suffix}"
        if not out_path.exists():
            return out_path
        time.sleep(1.0)


def _select_output_path(
    declared: Path,
    *,
    preset: str | None,
    experiment_id: str,
    jobs: list[tuple],
) -> Path:
    if preset is None:
        return declared

    # With a panel preset, config.out is only a template for directory/stem.
    # Actual writes go to {stem}_{preset}_{timestamp}{suffix}.
    directory = declared.parent
    stem = declared.stem
    suffix = declared.suffix
    all_jobs = set(jobs)
    candidates = sorted(
        directory.glob(f"{stem}_{preset}_*{suffix}"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        existing = load_checkpoint(candidate)
        completed = _completed_jobs_for_experiment(existing, experiment_id)
        if completed and not all_jobs.issubset(completed):
            return candidate
    return _timestamped_out_path(directory, stem, preset, suffix)


def _predictor_columns(
    frame: pd.DataFrame, prefixes: Sequence[str] = ("Aset", "Bset")
) -> list[str]:
    prefix_tuple = tuple(prefixes)
    return [col for col in frame.columns if col.startswith(prefix_tuple)]


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
    n_test_total: int,
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
        "n_test_total": int(n_test_total),
        "n_features_total": int(n_features_total),
    }


def _validate_classification_outcome(frame: pd.DataFrame, outcome: str) -> None:
    classes = set(pd.Series(frame[outcome]).dropna().unique())
    if not classes.issubset({0, 1, False, True}) or len(classes) > 2:
        raise ValueError(
            "--task classification requires a binary 0/1 outcome column. "
            "The real employment column name could not be verified locally; "
            "pass the confirmed column via --outcome."
        )


def _positive_class_probability(model, X) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        return np.full(len(X), np.nan)
    probabilities = np.asarray(model.predict_proba(X), dtype=float)
    if probabilities.ndim != 2:
        return np.full(len(X), np.nan)
    classes = np.asarray(getattr(model, "classes_", []))
    if classes.size and 1 in classes:
        return probabilities[:, int(np.where(classes == 1)[0][0])]
    if probabilities.shape[1] == 2:
        return probabilities[:, 1]
    if classes.size == 1:
        return np.ones(len(X)) if classes[0] == 1 else np.zeros(len(X))
    return np.full(len(X), np.nan)


def run_nk_grid(config: NKGridConfig, *, max_jobs: int | None = None) -> None:
    if config.task not in {"regression", "classification"}:
        raise ValueError("task must be 'regression' or 'classification'")
    if config.group_split_col:
        raise NotImplementedError(
            "--group-split-col is reserved for the sibling-clustering confirmation item."
        )
    data_path = Path(config.data)
    if not data_path.exists():
        raise FileNotFoundError(f"NLSY analysis data not found: {data_path}")

    frame = pd.read_csv(data_path)
    model_params_path = Path(config.model_params)
    selected_model_params = load_model_params(
        model_params_path,
        task=config.task,
        models=config.models,
    )
    if config.outcome not in frame:
        raise KeyError(f"Outcome not found: {config.outcome}")
    predictors = _predictor_columns(frame, config.predictor_prefix)
    if not predictors:
        raise ValueError(
            "No predictor columns found in the input data for prefixes "
            f"{list(config.predictor_prefix)}. Pass --predictor-prefix to match "
            "your dataset's feature column names."
        )
    log_progress(
        "loaded data "
        f"path={data_path} rows={len(frame)} predictors={len(predictors)} "
        f"outcome={config.outcome} task={config.task}"
    )
    if config.task == "classification":
        _validate_classification_outcome(frame, config.outcome)

    split_mode = "internal_random"
    test_data_sha256 = ""
    test_path: Path | None = None
    fixed_split: SplitData | None = None
    if config.test_data is not None:
        split_mode = "external_test"
        test_path = Path(config.test_data)
        if not test_path.exists():
            raise FileNotFoundError(f"NLSY external test data not found: {test_path}")
        test_frame = pd.read_csv(test_path)
        if config.outcome not in test_frame:
            raise KeyError(f"Outcome not found: {config.outcome}")
        missing_predictors = [col for col in predictors if col not in test_frame]
        if missing_predictors:
            raise ValueError(
                "External test data is missing predictor columns from the "
                f"training data: {missing_predictors}"
            )
        if config.task == "classification":
            _validate_classification_outcome(test_frame, config.outcome)
        if not np.isclose(config.test_size, 0.3):
            log_progress(
                "external test data supplied; ignoring --test-size because the "
                "test split is fixed by --test-data"
            )
        fixed_split = external_test_split(frame, test_frame, predictors, config.outcome)
        test_data_sha256 = file_sha256(test_path)
        log_progress(
            "loaded external test data "
            f"path={test_path} rows={len(test_frame)} "
            f"usable_test_rows={len(fixed_split.X_test)}"
        )

    metadata_extra = {
        "dataset": config.dataset,
        "n_seeds": config.n_seeds,
        "n_draws": config.n_draws,
        "n_sizes_n": config.n_sizes_n,
        "n_sizes_k": config.n_sizes_k,
        "min_n": config.min_n,
        "max_n": config.max_n,
        "max_k": config.max_k,
        "predictor_prefix": ",".join(config.predictor_prefix),
        "group_split_col": config.group_split_col,
        "bart_min_n": config.bart_min_n,
        "bart_min_k": config.bart_min_k,
        "split_mode": split_mode,
        "test_data_sha256": test_data_sha256,
        "model_params_path": str(model_params_path),
        "model_params_sha256": file_sha256(model_params_path),
        "model_params": selected_model_params,
        **model_run_settings(config.models),
    }
    if config.task == "classification":
        metadata_extra["task"] = config.task
    metadata = build_experiment_metadata(
        kind="nk_grid" if config.task == "regression" else "nk_grid_classification",
        data_path=data_path,
        outcome=config.outcome,
        test_size=config.test_size,
        split_seed=config.seed,
        extra=metadata_extra,
    )
    metadata["split_mode"] = split_mode
    metadata["test_data_sha256"] = test_data_sha256

    split_seeds = [config.seed + offset for offset in range(config.n_seeds)]
    if fixed_split is None:
        splits = {
            seed: split_frame(
                frame,
                predictors,
                config.outcome,
                test_size=config.test_size,
                seed=seed,
                task=config.task,
            )
            for seed in split_seeds
        }
    else:
        splits = {seed: fixed_split for seed in split_seeds}
    n_grid = log2_size_grid(
        len(next(iter(splits.values())).X_train),
        config.n_sizes_n,
        config.max_n,
        min_size=config.min_n,
    )
    k_grid = log2_size_grid(len(predictors), config.n_sizes_k, config.max_k)
    log_progress(
        "grid "
        f"N={n_grid.tolist()} K={k_grid.tolist()} "
        f"seeds={split_seeds} draws={config.n_draws} models={list(config.models)}"
    )

    jobs = [
        (model_name, seed, draw, int(n_samples), int(k_features))
        for seed in split_seeds
        for draw in range(config.n_draws)
        for k_features in k_grid
        for n_samples in n_grid
        for model_name in config.models
    ]

    out_path = _select_output_path(
        Path(config.out),
        preset=config.preset,
        experiment_id=metadata["experiment_id"],
        jobs=jobs,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_checkpoint(out_path)
    completed = _completed_jobs_for_experiment(existing, metadata["experiment_id"])
    pending = [job for job in jobs if job not in completed]
    if max_jobs is not None:
        pending = pending[: int(max_jobs)]
    log_progress(
        f"jobs total={len(jobs)} completed={len(completed)} "
        f"pending={len(pending)} batch_size={config.batch_size} n_jobs={config.n_jobs}"
    )

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
            n_test_total=len(split.X_test),
            n_features_total=len(predictors),
        )
        if (
            model_name == "bart"
            and (n_samples < config.bart_min_n or k_features < config.bart_min_k)
        ):
            return add_metadata(
                {
                    **row,
                    **(
                        _empty_metrics()
                        if config.task == "regression"
                        else _empty_classification_metrics()
                    ),
                    **({"task": config.task} if config.task == "classification" else {}),
                    "status": "skipped",
                    "error": "below BART minimum N/K floor",
                },
                metadata,
            )
        if (
            config.task == "regression"
            and model_name in REGRESSION_CV_MIN_N
            and n_samples < REGRESSION_CV_MIN_N[model_name]
        ):
            min_n = REGRESSION_CV_MIN_N[model_name]
            return add_metadata(
                {
                    **row,
                    **_empty_metrics(),
                    "status": "skipped",
                    "error": (
                        f"below minimum N for {model_name}'s internal CV "
                        f"(requires N>={min_n})"
                    ),
                },
                metadata,
            )
        try:
            orders = draw_orders(split.X_train.index, predictors, seed=seed, draw=draw)
            selected_rows = orders.row_index[:n_samples]
            selected_cols = orders.feature_names[:k_features]
            X_sub = split.X_train.loc[selected_rows, selected_cols]
            y_sub = split.y_train.loc[selected_rows]
            X_test = split.X_test.loc[:, selected_cols]
            if config.task == "classification" and len(np.unique(y_sub)) < 2:
                return add_metadata(
                    {
                        **row,
                        **_empty_classification_metrics(),
                        "task": config.task,
                        "status": "skipped",
                        "error": "single-class training sample for classification",
                    },
                    metadata,
                )
            if config.task == "classification" and model_name == "super_learner":
                min_class_count = int(y_sub.value_counts().min())
                if min_class_count < 2:
                    return add_metadata(
                        {
                            **row,
                            **_empty_classification_metrics(),
                            "task": config.task,
                            "status": "skipped",
                            "error": (
                                "below minimum per-class count for "
                                "super_learner CV"
                            ),
                        },
                        metadata,
                    )
            model = make_model(
                model_name,
                seed=_model_seed(seed, draw, n_samples, k_features),
                n_jobs=1,
                task=config.task,
                params=selected_model_params[model_name],
            )
            model.fit(X_sub, y_sub)
            if config.task == "classification":
                scores = _positive_class_probability(model, X_test)
                return add_metadata(
                    {
                        **row,
                        "task": config.task,
                        **compute_classification_metrics(split.y_test, scores, y_sub),
                        "status": "ok",
                        "error": "",
                    },
                    metadata,
                )
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
                    **(
                        _empty_metrics()
                        if config.task == "regression"
                        else _empty_classification_metrics()
                    ),
                    **({"task": config.task} if config.task == "classification" else {}),
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                metadata,
            )

    rows: list[dict] = []
    total_batches = int(np.ceil(len(pending) / config.batch_size)) if pending else 0
    for batch_index, start in enumerate(range(0, len(pending), config.batch_size), start=1):
        batch = pending[start : start + config.batch_size]
        log_progress(
            f"batch {batch_index}/{total_batches} starting "
            f"jobs={len(batch)} first={batch[0]}"
        )
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
        new_rows = rows[-len(batch) :] if batch else []
        ok_count = sum(row.get("status") == "ok" for row in new_rows)
        failed_count = sum(row.get("status") == "failed" for row in new_rows)
        skipped_count = sum(row.get("status") == "skipped" for row in new_rows)
        log_progress(
            f"batch {batch_index}/{total_batches} wrote checkpoint "
            f"new_rows={len(new_rows)} ok={ok_count} failed={failed_count} "
            f"skipped={skipped_count} total_new_rows={len(rows)} out={out_path}"
        )
    if not pending:
        log_progress("no pending jobs; checkpoint is already complete")


def parse_args() -> NKGridConfig:
    parser = argparse.ArgumentParser(
        description="Run joint log-scale N x K prediction-quality sweeps."
    )
    parser.add_argument("--data", default=str(ROOT / "data" / "asample2_withlag.csv"))
    parser.add_argument("--test-data", default=None)
    parser.add_argument("--task", default="regression", choices=("regression", "classification"))
    parser.add_argument("--outcome", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--dataset", default="asample2_withlag")
    parser.add_argument(
        "--models", nargs="+", default=["xgboost"], choices=SUPPORTED_MODEL_NAMES
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--n-seeds", type=int, default=2)
    parser.add_argument("--n-draws", type=int, default=2)
    parser.add_argument("--n-sizes-n", type=int, default=4)
    parser.add_argument("--n-sizes-k", type=int, default=4)
    parser.add_argument("--min-n", type=int, default=10)
    parser.add_argument("--max-n", type=int, default=100, help="Use <=0 for full train set.")
    parser.add_argument("--max-k", type=int, default=100, help="Use <=0 for all features.")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--bart-min-n", type=int, default=10)
    parser.add_argument("--bart-min-k", type=int, default=2)
    parser.add_argument("--model-params", default=str(DEFAULT_MODEL_PARAMS_PATH))
    parser.add_argument("--group-split-col", default=None)
    parser.add_argument(
        "--predictor-prefix",
        nargs="+",
        default=["Aset", "Bset"],
        help=(
            "Column-name prefixes that select the predictor (feature) columns. "
            "Defaults to the Zheng-Cheng Aset/Bset naming; pass your own dataset's "
            "prefixes to run on another paper's data."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )
    args = parser.parse_args()
    if args.outcome is None:
        raise ValueError(
            "--outcome is required: pass the name of the outcome column to predict "
            "(a continuous column for --task regression, or a binary 0/1 column for "
            "--task classification)."
        )
    outcome = args.outcome
    out = args.out
    if out is None:
        out = str(
            ROOT
            / "outputs"
            / ("nk_grid.csv" if args.task == "regression" else "nk_grid_clf.csv")
        )
    return NKGridConfig(
        data=Path(args.data),
        test_data=Path(args.test_data) if args.test_data is not None else None,
        out=Path(out),
        dataset=args.dataset,
        outcome=outcome,
        models=tuple(args.models),
        seed=args.seed,
        test_size=args.test_size,
        n_seeds=args.n_seeds,
        n_draws=args.n_draws,
        n_sizes_n=args.n_sizes_n,
        n_sizes_k=args.n_sizes_k,
        min_n=args.min_n,
        max_n=args.max_n,
        max_k=args.max_k,
        batch_size=args.batch_size,
        n_jobs=args.n_jobs,
        group_split_col=args.group_split_col,
        task=args.task,
        model_params=Path(args.model_params),
        bart_min_n=args.bart_min_n,
        bart_min_k=args.bart_min_k,
        predictor_prefix=tuple(args.predictor_prefix),
    )


def main() -> None:
    run_nk_grid(parse_args())


if __name__ == "__main__":
    main()
