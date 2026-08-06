"""Joint N x K sweeps for long-format prediction quality tables."""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
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

ROOT = Path(__file__).resolve().parents[2]

from .evaluation import r2_against_training_mean
from .experiment import (
    CHECKPOINT_COMPACTION_LOOSE_PARTS,
    CHECKPOINT_KEY_COLUMNS,
    CheckpointSummary,
    SERIAL_OUTER_MODELS,
    add_metadata,
    build_experiment_metadata,
    checkpoint_parts,
    checkpoint_parts_dir,
    core_environment,
    diagnostics_summary,
    git_state,
    load_checkpoint,  # compatibility alias; production resume uses projected index
    load_checkpoint_index,
    manifest_path,
    merge_checkpoint_parts,
    model_run_settings,
    output_run_lock,
    rows_for_experiment,
    seed_checkpoint_parts_from_csv,
    retire_checkpoint_parts,
    utc_now,
    verify_materialized_checkpoint,
    write_checkpoint_part,
    write_json_atomic,
)
from .helpers_logging import log_progress
from .ingest import LoadedInput, load_input
from .model_registry import (
    DEFAULT_MODEL_PARAMS_PATH,
    SUPPORTED_MODEL_NAMES,
    load_algorithm_version,
    load_model_params,
    make_model,
    reject_removed_model,
    resolved_model_params,
)
from .native_process import IsolatedProcessRunner
from .preprocessing import (
    SourceGroup,
    count_unobserved_sources,
    preprocess_cell,
)
from .validate_input import REGRESSION_CV_MIN_N, validate_input


LARGE_RUN_THRESHOLD = 250_000

# Row-level metadata is deliberately scalar-only. The complete artifact-level
# identity and semantic contract belong in the sidecar manifest, where
# _manifest_payload() records them once instead of once per checkpoint row.
ROW_METADATA_FIELDS = (
    "experiment_id",
    "experiment_kind",
    "algorithm_version",
    "outcome",
    "test_size",
    "split_mode",
    "split_seed",
)

# Super Learner fits each of its 4 base learners once per CV fold plus one final
# refit on the full subsample: 4 x (cv + 1) with cv=5.
SUPER_LEARNER_FITS_PER_CELL = 24


_NATIVE_RUNNER_LOCK = threading.Lock()


def _run_native_model_cell_locked(
    runner: IsolatedProcessRunner,
    *,
    fit_arguments: dict[str, Any],
    on_native_crash: Callable[[int, BaseException], None],
    on_native_timeout: Callable[[int, BaseException], None],
) -> dict[str, Any]:
    """Serialize access to the one reusable native-model subprocess."""

    with _NATIVE_RUNNER_LOCK:
        return runner.run(
            _fit_predict_model_cell,
            **fit_arguments,
            on_native_crash=on_native_crash,
            on_native_timeout=on_native_timeout,
        )


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
    schema: Path
    out: Path
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
    failed_abs_threshold: int = 50
    failed_ratio_threshold: float = 0.05
    native_process_max_attempts: int = 2
    native_process_timeout_seconds: float = 21_600.0
    preset: str | None = None
    allow_large_run: bool = False
    dry_run: bool = False
    rerun_completed: bool = True
    # Direct construction is used by the test/dev API. Production manifests
    # always override these explicit values.
    experiment_id: str = "nkgrid-test-v1"
    data_version: str = "test-data-v1"
    model_spec_version: str = "nkgrid-test-models-v1"
    repeat_plan: tuple[tuple[int, int], ...] | None = None
    n_grid: tuple[int, ...] | None = None
    k_grid: tuple[int, ...] | None = None


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


def resolve_repeat_pairs(config: NKGridConfig) -> tuple[tuple[int, int], ...]:
    """Resolve legacy counts or explicit absolute pairs into one representation."""

    if config.repeat_plan is not None:
        if config.n_seeds != 1 or config.n_draws != 1:
            raise ValueError("repeat_plan cannot be combined with n_seeds or n_draws")
        pairs = tuple((seed, draw) for seed, draw in config.repeat_plan)
    else:
        pairs = tuple(
            (config.seed + offset, draw)
            for offset in range(config.n_seeds)
            for draw in range(config.n_draws)
        )
    group_repeat_pairs_by_seed(pairs)
    return tuple(sorted(pairs))


def group_repeat_pairs_by_seed(
    repeat_pairs: Sequence[tuple[int, int]],
) -> dict[int, tuple[int, ...]]:
    """Validate and group absolute repeat pairs without silently deduplicating."""

    grouped: dict[int, list[int]] = {}
    seen: set[tuple[int, int]] = set()
    for pair in repeat_pairs:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("repeat_plan entries must be (seed, draw) pairs")
        seed, draw = pair
        if isinstance(seed, bool) or isinstance(draw, bool) or not isinstance(seed, int) or not isinstance(draw, int) or seed < 0 or draw < 0:
            raise ValueError("repeat_plan seed and draw must be non-negative integers")
        if (seed, draw) in seen:
            raise ValueError(f"repeat_plan contains duplicate pair ({seed}, {draw})")
        seen.add((seed, draw))
        grouped.setdefault(seed, []).append(draw)
    if not grouped:
        raise ValueError("repeat_plan must not be empty")
    return {seed: tuple(sorted(draws)) for seed, draws in sorted(grouped.items())}


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
    row_index = rows[rng.permutation(len(rows))]
    ordered_features = features[rng.permutation(len(features))]
    return DrawOrders(
        row_index=row_index,
        feature_names=ordered_features,
    )


def _freeze_draw_orders(orders: DrawOrders) -> DrawOrders:
    """Protect an order shared by cached execution without changing public API."""

    orders.row_index.setflags(write=False)
    orders.feature_names.setflags(write=False)
    return orders


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
    has_two_test_classes = len(np.unique(y_true)) == 2
    finite_scores = np.all(np.isfinite(score))
    if not finite_scores:
        return _empty_classification_metrics()
    labels = (score >= 0.5).astype(int)
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


def _empty_diagnostics() -> dict[str, float | bool]:
    return {
        "K_varying": np.nan,
        "constant_prediction": False,
        "underdetermined": False,
        "converged": False,
        "_fit_seconds": np.nan,
        "_best_rounds": np.nan,
        "_preprocess_seconds": 0.0,
        "_preprocess_computed": False,
        "_preprocess_vectorized": False,
        "_slice_seconds": 0.0,
        "_cell_wall_seconds": 0.0,
        "_peak_rss_bytes": 0,
    }


def _process_peak_rss_bytes() -> int:
    """Return this process's peak resident set size in bytes."""

    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def _constant_prediction(values: Sequence[float]) -> bool:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return bool(len(np.unique(finite)) < 2)


def _ols_is_underdetermined(X: pd.DataFrame) -> bool:
    """Diagnose p>=N using varying expanded columns, not source count."""

    expanded_varying = int(X.nunique(dropna=True).gt(1).sum())
    return expanded_varying >= len(X)


def _model_converged(model) -> bool:
    """Read deterministic iteration limits through fitted estimator wrappers."""

    statuses: list[bool] = []
    seen: set[int] = set()

    def visit(estimator) -> None:
        if estimator is None or id(estimator) in seen:
            return
        seen.add(id(estimator))

        if hasattr(estimator, "n_iter_") and hasattr(estimator, "max_iter"):
            iterations = np.asarray(getattr(estimator, "n_iter_"), dtype=float)
            finite = iterations[np.isfinite(iterations)]
            if finite.size:
                statuses.append(bool(np.max(finite) < float(estimator.max_iter)))

        if hasattr(estimator, "steps"):
            for _, step in estimator.steps:
                visit(step)
        if hasattr(estimator, "regressor_"):
            visit(estimator.regressor_)
        if hasattr(estimator, "model_"):
            visit(estimator.model_)
        if hasattr(estimator, "final_estimator_"):
            visit(estimator.final_estimator_)
            for fitted_estimator in getattr(estimator, "estimators_", ()):
                visit(fitted_estimator)

    visit(model)
    return all(statuses) if statuses else True


def _model_best_rounds(model) -> float:
    """Find a fitted boosting round count through common wrapper layers."""

    for attribute in ("best_rounds_", "best_iteration_"):
        if hasattr(model, attribute):
            value = getattr(model, attribute)
            if value is not None:
                return float(value)
    if hasattr(model, "steps") and model.steps:
        return _model_best_rounds(model.steps[-1][1])
    if hasattr(model, "regressor_"):
        return _model_best_rounds(model.regressor_)
    return np.nan


def _model_attribute_values(model, attribute: str) -> list[Any]:
    """Read one fitted attribute through common estimator wrappers.

    The traversal is deliberately model-agnostic and shared by all fit-state
    diagnostics so a wrapper added to one diagnostic cannot silently be absent
    from another.
    """

    values: list[Any] = []
    seen: set[int] = set()

    def visit(estimator) -> None:
        if estimator is None or id(estimator) in seen:
            return
        seen.add(id(estimator))
        value = getattr(estimator, attribute, None)
        if value is not None:
            values.append(value)
        if hasattr(estimator, "steps"):
            for _, step in estimator.steps:
                visit(step)
        for child_attribute in ("regressor_", "model_", "final_estimator_"):
            visit(getattr(estimator, child_attribute, None))
        for fitted_estimator in getattr(estimator, "estimators_", ()):
            visit(fitted_estimator)

    visit(model)
    return values


def _model_solver(model) -> str | None:
    """Find an estimator's fitted solver through common wrapper layers.

    ``solver_`` records the solver actually selected by estimators that accept
    an automatic choice, so it wins globally over a configured ``solver``.
    The traversal is deliberately duck-typed and model-agnostic.
    """

    selected = _model_attribute_values(model, "solver_")
    configured = _model_attribute_values(model, "solver")
    value = selected[0] if selected else (configured[0] if configured else None)
    return None if value is None else str(value)


def _model_iterations(model) -> float:
    """Return the largest finite observed ``n_iter_``, or NaN when absent."""

    iterations: list[float] = []
    for value in _model_attribute_values(model, "n_iter_"):
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            continue
        finite = array[np.isfinite(array)]
        if finite.size:
            iterations.append(float(np.max(finite)))
    return max(iterations) if iterations else np.nan


def _model_alpha(model) -> float:
    """Return the first finite fitted ``alpha_``, or NaN when absent."""

    for value in _model_attribute_values(model, "alpha_"):
        try:
            alpha = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(alpha):
            return alpha
    return np.nan


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


def _checkpoint_index_exactly_matches_jobs(
    existing: pd.DataFrame,
    experiment_id: str,
    jobs: list[tuple],
    completed: set[tuple],
) -> bool:
    """Reject duplicate, failed, missing, or out-of-design rows on fast reuse."""

    current = rows_for_experiment(existing, experiment_id)
    if len(current) != len(jobs):
        return False
    if current.duplicated(CHECKPOINT_KEY_COLUMNS).any():
        return False
    if "status" not in current or not current["status"].isin(("ok", "skipped")).all():
        return False
    return len(completed) == len(jobs) and all(job in completed for job in jobs)


def _identity_path_segment(experiment_id: str) -> str:
    """Validate that experiment_id is safe as a filename segment."""

    if experiment_id in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9._-]+", experiment_id):
        raise ValueError(
            "experiment_id must be a non-empty filename segment containing only "
            "ASCII letters, digits, dots, underscores, or hyphens"
        )
    return experiment_id


def _timestamped_out_path(directory: Path, stem: str, segment: str, suffix: str) -> Path:
    while True:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = directory / f"{stem}_{segment}_{timestamp}{suffix}"
        if not any(
            (
                out_path.exists(),
                manifest_path(out_path).exists(),
                checkpoint_parts_dir(out_path).exists(),
            )
        ):
            return out_path
        time.sleep(1.0)


def _select_output_path(
    declared: Path,
    *,
    preset: str | None,
    experiment_id: str,
    jobs: list[tuple],
    rerun_completed: bool,
) -> Path:
    if preset is None:
        # This distinguishes direct CLI output from panel-managed output paths.
        return declared

    segment = _identity_path_segment(experiment_id)

    # With a panel preset, config.out is only a template for directory/stem.
    # Actual writes use experiment_id as their resumable namespace segment.
    directory = declared.parent
    stem = declared.stem
    suffix = declared.suffix
    candidates_by_path = {
        path: path.stat().st_mtime
        for path in directory.glob(f"{stem}_{segment}_*{suffix}")
    }
    for candidate_manifest in directory.glob(f"{stem}_{segment}_*.manifest.json"):
        candidate = candidate_manifest.with_name(
            candidate_manifest.name.removesuffix(".manifest.json") + suffix
        )
        candidates_by_path[candidate] = max(
            candidates_by_path.get(candidate, float("-inf")),
            candidate_manifest.stat().st_mtime,
        )
    for candidate_parts in directory.glob(f"{stem}_{segment}_*.parts"):
        if not candidate_parts.is_dir():
            continue
        candidate = candidate_parts.with_suffix(suffix)
        candidates_by_path[candidate] = max(
            candidates_by_path.get(candidate, float("-inf")),
            candidate_parts.stat().st_mtime,
        )
    candidates = sorted(
        candidates_by_path,
        key=lambda path: candidates_by_path[path],
        reverse=True,
    )
    for candidate in candidates:
        candidate_manifest_path = manifest_path(candidate)
        candidate_manifest_id = None
        if candidate_manifest_path.exists():
            try:
                candidate_manifest = json.loads(
                    candidate_manifest_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                candidate_manifest = None
            if isinstance(candidate_manifest, dict):
                candidate_manifest_id = candidate_manifest.get("experiment_id")
        if (
            candidate_manifest_id is not None
            and candidate_manifest_id != experiment_id
        ):
            continue
        try:
            existing = load_checkpoint_index(candidate)
        except ValueError:
            if candidate_manifest_id == experiment_id:
                raise
            log_progress(
                "ignoring unreadable preset candidate without a matching "
                f"manifest identity: {candidate}"
            )
            continue
        current = rows_for_experiment(existing, experiment_id)
        if current.empty:
            # A manifest is written before the first batch. Reuse that path
            # after an early interruption, but only when no checkpoint rows
            # contradict the manifest's identity.
            if existing.empty and _read_prior_manifest(
                manifest_path(candidate), experiment_id
            ) is not None:
                return candidate
            continue
        completed = _completed_jobs_for_experiment(existing, experiment_id)
        if not all(job in completed for job in jobs):
            # Matching rows are resumable even when every prior attempt failed:
            # failed cells remain pending and a successful retry supersedes
            # them during checkpoint deduplication.
            return candidate
        if not rerun_completed:
            log_progress(
                f"already complete; reusing prior preset output: {candidate}"
            )
            return candidate
        log_progress(
            "already complete; rerun_completed=true so a new timestamped "
            f"output will be created instead of reusing {candidate}"
        )
        # The newest matching run is complete. Do not fall through and revive
        # an older partial run when the caller explicitly requested a rerun.
        return _timestamped_out_path(directory, stem, segment, suffix)
    return _timestamped_out_path(directory, stem, segment, suffix)


def estimate_run_size(config: NKGridConfig) -> dict[str, int | str]:
    """Return a conservative pre-data estimate for panel dry-runs."""

    _validate_config(config)
    top_level = (
        len(config.models)
        * len(resolve_repeat_pairs(config))
        * len(config.n_grid or tuple(range(config.n_sizes_n)))
        * len(config.k_grid or tuple(range(config.n_sizes_k)))
    )
    super_cells = (
        top_level // len(config.models)
        if "super_learner" in config.models
        else 0
    )
    checkpoint_writes = int(np.ceil(top_level / config.batch_size))
    compact_parts, loose_parts = divmod(
        checkpoint_writes,
        CHECKPOINT_COMPACTION_LOOSE_PARTS,
    )
    stable_checkpoint_parts = compact_parts + loose_parts
    peak_checkpoint_parts = max(
        stable_checkpoint_parts,
        (
            compact_parts + CHECKPOINT_COMPACTION_LOOSE_PARTS
            if compact_parts
            else checkpoint_writes
        ),
    )
    return {
        "top_level_model_cells": int(top_level),
        "expected_output_rows": int(top_level),
        "estimated_super_learner_internal_fits": int(
            super_cells * SUPER_LEARNER_FITS_PER_CELL
        ),
        "estimated_checkpoint_writes": checkpoint_writes,
        # Backward-compatible key now describes the stable physical shard
        # count after automatic WAL compaction, not the number of writes.
        "estimated_checkpoint_parts": stable_checkpoint_parts,
        "estimated_peak_checkpoint_parts": peak_checkpoint_parts,
        "checkpoint_compaction_loose_parts": CHECKPOINT_COMPACTION_LOOSE_PARTS,
        "max_uncheckpointed_cells": min(
            int(config.batch_size),
            int(top_level),
        ),
        "materialization_backend": "sqlite_streaming",
    }


def _validate_config(config: NKGridConfig) -> None:
    """Reject invalid run controls before dry-run arithmetic or data loading."""

    for field in (
        "n_seeds",
        "n_draws",
        "n_sizes_n",
        "n_sizes_k",
        "min_n",
        "batch_size",
    ):
        if int(getattr(config, field)) < 1:
            raise ValueError(f"{field} must be at least 1")
    if config.n_jobs == 0:
        raise ValueError("n_jobs must not be zero")
    if not config.models:
        raise ValueError("models must not be empty")
    if len(config.models) != len(set(config.models)):
        raise ValueError("models must not contain duplicates")
    for model_name in config.models:
        reject_removed_model(model_name)
    unknown_models = sorted(set(config.models) - set(SUPPORTED_MODEL_NAMES))
    if unknown_models:
        raise ValueError(f"Unknown model(s): {', '.join(unknown_models)}")
    if config.failed_abs_threshold < 0:
        raise ValueError("failed_abs_threshold must be non-negative")
    if not 0.0 <= config.failed_ratio_threshold <= 1.0:
        raise ValueError("failed_ratio_threshold must be in [0, 1]")
    if config.native_process_max_attempts < 1:
        raise ValueError("native_process_max_attempts must be at least 1")
    if config.native_process_timeout_seconds <= 0:
        raise ValueError("native_process_timeout_seconds must be greater than zero")
    for field in ("experiment_id", "data_version", "model_spec_version"):
        value = getattr(config, field)
        if not isinstance(value, str) or not value or len(value) > 80 or not value.isascii() or not all(char.isalnum() or char in "._-" for char in value):
            raise ValueError(f"{field} must contain 1-80 ASCII letters, digits, dots, underscores or hyphens")
    group_repeat_pairs_by_seed(resolve_repeat_pairs(config))


def _relative_path(path: Path) -> str:
    try:
        return os.path.relpath(path.resolve(), ROOT)
    except OSError:
        return str(path)


def _parallelism_payload(config: NKGridConfig) -> dict[str, Any]:
    """Describe the scheduler policy actually used by one array worker."""

    selected = set(config.models)
    native = selected & set(SERIAL_OUTER_MODELS)
    super_learner = selected == {"super_learner"}
    return {
        "outer_cell_n_jobs": 1,
        "model_internal_n_jobs": int(config.n_jobs) if super_learner else 1,
        "base_estimator_n_jobs": 1,
        "configured_outer_n_jobs": 1,
        "chunk_policy": {
            "parallel_unit": "cell_group",
            "prefer": "serial",
            "contains_native": bool(native),
            "n_jobs_rule": "one array worker executes its groups serially",
            "native_calls_serialized": bool(native),
        },
        "native_process_isolated_models": sorted(native),
        "native_process_max_attempts": int(config.native_process_max_attempts),
        "native_process_timeout_seconds": float(
            config.native_process_timeout_seconds
        ),
    }


def _manifest_payload(
    *,
    config: NKGridConfig,
    metadata: dict,
    out_path: Path,
    data_path: Path,
    test_path: Path | None,
    model_params_path: Path,
    selected_model_params: dict,
    frame: pd.DataFrame,
    predictors: Sequence[str],
    split_seeds: list[int],
    execution_pairs: Sequence[tuple[int, int]],
    splits: dict[int, SplitData],
    n_grid: np.ndarray,
    k_grid: np.ndarray,
    expected_rows: int,
    results: pd.DataFrame | None,
    result_summary: CheckpointSummary | None = None,
    started_at: str,
    dataset: str,
    task: str,
    schema_path: Path,
    semantic_contract: Mapping[str, Any],
    seed_shard_execution: bool = False,
) -> dict:
    if result_summary is not None:
        if result_summary.experiment_id != metadata["experiment_id"]:
            raise ValueError("Checkpoint summary does not match the current experiment")
        materialized_rows = int(result_summary.materialized_rows)
        ok_count = int(result_summary.ok_rows)
        skipped_count = int(result_summary.skipped_rows)
        failed = int(result_summary.failed_rows)
        completed = int(result_summary.completed_rows)
        diagnostics = result_summary.diagnostics
    else:
        if results is None:
            raise ValueError("results or result_summary is required")
        current_results = rows_for_experiment(results, metadata["experiment_id"])
        statuses = current_results.get("status", pd.Series(dtype=str))
        materialized_rows = int(len(current_results))
        ok_count = int(statuses.eq("ok").sum())
        skipped_count = int(statuses.eq("skipped").sum())
        failed = int(statuses.eq("failed").sum())
        completed = ok_count + skipped_count
        diagnostics = diagnostics_summary(current_results)
    if materialized_rows != expected_rows:
        completion_status = "incomplete"
    elif failed:
        completion_status = "complete_with_failures"
    else:
        completion_status = "complete"
    return {
        "schema_version": "1",
        "experiment_id": metadata["experiment_id"],
        "algorithm_version": metadata["algorithm_version"],
        "created_at": started_at,
        "updated_at": utc_now(),
        "task": task,
        "outcome": config.outcome,
        "dataset": dataset,
        "identity": metadata["identity"],
        "semantic_contract": semantic_contract,
        "schema": {
            "path": _relative_path(schema_path),
        },
        "git": git_state(ROOT),
        "data": {
            "input_path": _relative_path(data_path),
            "test_path": _relative_path(test_path) if test_path is not None else None,
            "rows": int(len(frame)),
            "features": int(len(predictors)),
            "train_rows": int(len(next(iter(splits.values())).X_train)),
            "test_rows": int(len(next(iter(splits.values())).X_test)),
        },
        "design": {
            "preset": config.preset,
            "test_size": float(config.test_size),
            "split_mode": metadata["split_mode"],
            "split_seeds": split_seeds,
            "repeat_plan": [
                {"seed": seed, "draw": draw} for seed, draw in resolve_repeat_pairs(config)
            ],
            "n_grid": [int(value) for value in n_grid],
            "k_grid": [int(value) for value in k_grid],
            "models": list(config.models),
            "parallelism": _parallelism_payload(config),
            "checkpointing": {
                "batch_size": int(config.batch_size),
                "loose_parts_per_compaction": int(
                    CHECKPOINT_COMPACTION_LOOSE_PARTS
                ),
                "materialization_backend": "sqlite_streaming",
            },
        },
        "execution": {
            "mode": "seed-shard" if seed_shard_execution else "monolithic",
            "seed": split_seeds[0] if len(split_seeds) == 1 else None,
            "draws": [draw for _, draw in execution_pairs] if len(split_seeds) == 1 else None,
            "expected_rows": int(expected_rows),
        },
        "model_parameters": {
            "source": _relative_path(model_params_path),
            "resolved": resolved_model_params(selected_model_params),
        },
        "environment": core_environment(),
        "output": {
            "csv": out_path.name,
            "parts_directory": checkpoint_parts_dir(out_path).name,
            "checkpoint_parts_deleted": False,
        },
        "completion": {
            "expected_rows": int(expected_rows),
            "materialized_rows": materialized_rows,
            "completed_rows": completed,
            "failed_rows": failed,
            "status": completion_status,
        },
        "failure_policy": {
            "failed_abs_threshold": int(config.failed_abs_threshold),
            "failed_ratio_threshold": float(config.failed_ratio_threshold),
            "failed_count": failed,
            "ok_count": ok_count,
            "skipped_count": skipped_count,
            "denominator": ok_count + failed,
            "failed_ratio": (
                float(failed / (ok_count + failed))
                if ok_count + failed
                else None
            ),
        },
        "diagnostics": diagnostics,
    }


def _prune_checkpoint_parts(out_path: Path, manifest: dict) -> bool:
    """Delete shards only after the persisted final artifacts pass QA."""

    completion = manifest["completion"]
    status = completion["status"]
    if status != "complete":
        log_progress(
            f"checkpoint cleanup skipped: completion status is {status!r}; "
            "shards are retained for resume"
        )
        return False
    expected = int(completion["expected_rows"])
    counts_are_complete = (
        int(completion["materialized_rows"]) == expected
        and int(completion["completed_rows"]) == expected
        and int(completion["failed_rows"]) == 0
    )
    if not counts_are_complete:
        log_progress(
            "checkpoint cleanup skipped: manifest row counts or failure count "
            "did not pass verification"
        )
        return False
    directory = checkpoint_parts_dir(out_path)
    if not directory.exists():
        return False

    try:
        verify_materialized_checkpoint(
            out_path,
            experiment_id=manifest["experiment_id"],
            expected_rows=expected,
        )
    except Exception as exc:
        raise RuntimeError(
            "Final CSV streaming verification failed; checkpoint shards were "
            "retained."
        ) from exc

    retired = retire_checkpoint_parts(out_path)
    if retired is None:
        return False
    try:
        shutil.rmtree(retired)
        log_progress(
            "deleted checkpoint shards after verified-complete run: "
            f"{directory.name}"
        )
    except OSError as exc:
        # The atomic rename already made the final CSV authoritative. A stale
        # hidden tombstone is a storage leak, not a resume/data-loss hazard.
        log_progress(
            f"checkpoint shards retired but cleanup was incomplete: "
            f"{retired.name} ({type(exc).__name__}: {exc})"
        )
    return True


def _read_prior_manifest(path: Path, experiment_id: str) -> dict | None:
    """Return the previous manifest for this experiment, if it is readable."""

    if not path.exists():
        return None
    try:
        prior = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(prior, dict):
        return None
    return prior if prior.get("experiment_id") == experiment_id else None


def _first_contract_difference(previous: Any, current: Any, path: str = "semantic_contract") -> str | None:
    if isinstance(previous, Mapping) and isinstance(current, Mapping):
        for key in sorted(set(previous) | set(current)):
            if key not in previous or key not in current:
                return f"{path}.{key}: checkpoint={previous.get(key)!r} current={current.get(key)!r}"
            difference = _first_contract_difference(previous[key], current[key], f"{path}.{key}")
            if difference:
                return difference
        return None
    if previous != current:
        return f"{path}: checkpoint={previous!r} current={current!r}"
    return None


def _require_resumable_manifest(prior: dict, metadata: Mapping[str, Any]) -> None:
    identity = prior.get("identity")
    expected = metadata["identity"]
    if not isinstance(identity, Mapping) or identity.get("mode") != "explicit-v1":
        raise ValueError("Existing manifest is not explicit-v1 and cannot be resumed")
    for field in ("experiment_id", "data_version", "model_spec_version"):
        if identity.get(field) != expected[field]:
            raise ValueError(f"identity.{field}: checkpoint={identity.get(field)!r} current={expected[field]!r}")
    difference = _first_contract_difference(prior.get("semantic_contract"), metadata["semantic_contract"])
    if difference:
        raise ValueError(difference)


def _verified_complete_artifacts(
    out_path: Path,
    experiment_id: str,
    expected_rows: int,
) -> bool:
    prior = _read_prior_manifest(manifest_path(out_path), experiment_id)
    if prior is None or not out_path.exists() or checkpoint_parts(out_path):
        return False
    completion = prior.get("completion")
    design = prior.get("design")
    if not isinstance(completion, dict) or not isinstance(design, dict):
        return False
    try:
        return (
            completion.get("status") == "complete"
            and int(completion.get("completed_rows", -1)) == expected_rows
            and int(completion.get("materialized_rows", -1)) == expected_rows
        )
    except (TypeError, ValueError):
        return False


def _preserve_prior_timings(payload: dict, prior: dict | None) -> dict:
    """Carry per-model timings forward when shards can no longer supply them.

    Timing diagnostics live only in checkpoint shards, so once the shards are
    pruned a later merge cannot recompute them. Keep whatever the previous
    manifest recorded instead of dropping the fields.
    """

    if prior is None:
        return payload
    prior_diagnostics = prior.get("diagnostics")
    if not isinstance(prior_diagnostics, dict):
        return payload
    prior_models = prior_diagnostics.get("by_model")
    if not isinstance(prior_models, dict):
        return payload
    for model, summary in payload.get("diagnostics", {}).get("by_model", {}).items():
        prior_summary = prior_models.get(model, {})
        if not isinstance(prior_summary, dict):
            continue
        for key in (
            "fit_seconds_total",
            "fit_seconds_median",
            "preprocess_seconds_total",
            "cell_wall_seconds_total",
            "peak_rss_bytes_max",
            "best_rounds",
        ):
            if key not in summary and key in prior_summary:
                summary[key] = prior_summary[key]
    return payload


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
    k_expanded: int,
    n_expanded_features_total: int,
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
        "K_expanded": int(k_expanded),
        "n_expanded_features_total": int(n_expanded_features_total),
        "K_unobserved": np.nan,
    }


class RunFailureThresholdExceeded(RuntimeError):
    """Raised after artifacts are persisted and the run failure policy fails."""


def _failure_policy_violation(payload: dict) -> str | None:
    policy = payload["failure_policy"]
    denominator = int(policy["denominator"])
    failed = int(policy["failed_count"])
    if denominator == 0:
        return "failure-policy denominator is zero (no ok/failed cells)"
    ratio = float(policy["failed_ratio"])
    if failed > int(policy["failed_abs_threshold"]):
        return (
            f"failed_count={failed} exceeds "
            f"failed_abs_threshold={policy['failed_abs_threshold']}"
        )
    if ratio > float(policy["failed_ratio_threshold"]):
        return (
            f"failed_ratio={ratio:.6f} exceeds "
            f"failed_ratio_threshold={policy['failed_ratio_threshold']:.6f}"
        )
    return None


def _positive_class_probability(model, X) -> np.ndarray:
    # Callers guarantee a two-class training sample (single-class cells are
    # skipped upstream), so the fitted classifier exposes predict_proba with
    # both classes. A contract violation raises here and surfaces as a failed
    # cell rather than silently producing NaN metrics.
    probabilities = np.asarray(model.predict_proba(X), dtype=float)
    classes = np.asarray(getattr(model, "classes_", []))
    if probabilities.ndim != 2:
        raise ValueError(
            f"predict_proba must return a 2D array; got shape={probabilities.shape}"
        )
    if classes.size and 1 in classes:
        positive = probabilities[:, int(np.where(classes == 1)[0][0])]
    elif probabilities.ndim == 2 and probabilities.shape[1] == 2:
        positive = probabilities[:, 1]
    else:
        raise ValueError(
            f"cannot locate positive class in predict_proba output "
            f"(classes_={classes}, shape={probabilities.shape})"
        )
    if not np.isfinite(positive).all():
        raise ValueError("predict_proba returned non-finite positive-class scores")
    return positive


def _fit_predict_model_cell(
    *,
    model_name: str,
    model_seed: int,
    task: str,
    params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    model_n_jobs: int = 1,
) -> dict[str, Any]:
    """Fit and predict one cell; safe to execute in an isolated subprocess."""

    model = make_model(
        model_name,
        seed=model_seed,
        n_jobs=model_n_jobs,
        task=task,
        params=params,
    )
    fit_started = time.perf_counter()
    model.fit(X_train, y_train)
    predictions = (
        _positive_class_probability(model, X_test)
        if task == "classification"
        else np.asarray(model.predict(X_test))
    )
    return {
        "predictions": predictions,
        "fit_seconds": time.perf_counter() - fit_started,
        "best_rounds": _model_best_rounds(model),
        "converged": _model_converged(model),
        "solver": _model_solver(model),
        "iterations": _model_iterations(model),
        "alpha": _model_alpha(model),
        "peak_rss_bytes": _process_peak_rss_bytes(),
    }


def run_nk_grid(
    config: NKGridConfig,
    *,
    execution_pairs: tuple[tuple[int, int], ...] | None = None,
    defer_failure_policy: bool = False,
    max_jobs: int | None = None,
    allow_large_run: bool | None = None,
    dry_run: bool | None = None,
    stop_after_batch: Callable[[], bool] | None = None,
    defer_materialization_on_stop: bool = False,
    exact_output_path: bool = False,
) -> Path | dict[str, int | str]:
    """Run one output under an advisory cross-process writer lease."""

    effective_dry_run = config.dry_run if dry_run is None else dry_run
    if effective_dry_run:
        return _run_nk_grid_locked(
            config,
            max_jobs=max_jobs,
            allow_large_run=allow_large_run,
            dry_run=True,
            stop_after_batch=stop_after_batch,
            defer_materialization_on_stop=defer_materialization_on_stop,
            exact_output_path=exact_output_path,
            execution_pairs=execution_pairs,
            defer_failure_policy=defer_failure_policy,
        )
    with output_run_lock(Path(config.out)):
        return _run_nk_grid_locked(
            config,
            max_jobs=max_jobs,
            allow_large_run=allow_large_run,
            dry_run=False,
            stop_after_batch=stop_after_batch,
            defer_materialization_on_stop=defer_materialization_on_stop,
            exact_output_path=exact_output_path,
            execution_pairs=execution_pairs,
            defer_failure_policy=defer_failure_policy,
        )


def _run_nk_grid_locked(
    config: NKGridConfig,
    *,
    max_jobs: int | None = None,
    allow_large_run: bool | None = None,
    dry_run: bool | None = None,
    stop_after_batch: Callable[[], bool] | None = None,
    defer_materialization_on_stop: bool = False,
    exact_output_path: bool = False,
    execution_pairs: tuple[tuple[int, int], ...] | None = None,
    defer_failure_policy: bool = False,
) -> Path | dict[str, int | str]:
    allow_large_run = config.allow_large_run if allow_large_run is None else allow_large_run
    dry_run = config.dry_run if dry_run is None else dry_run
    if max_jobs is not None and max_jobs < 0:
        raise ValueError("max_jobs must be non-negative")
    repeat_pairs = resolve_repeat_pairs(config)
    shard_execution_requested = execution_pairs is not None
    if execution_pairs is None:
        execution_pairs = repeat_pairs
    else:
        execution_pairs = tuple(
            (int(seed), int(draw)) for seed, draw in execution_pairs
        )
        if not execution_pairs or not set(execution_pairs).issubset(
            set(repeat_pairs)
        ):
            raise ValueError(
                "execution_pairs must be a non-empty subset of repeat_plan"
            )
        if len({seed for seed, _ in execution_pairs}) != 1:
            raise ValueError(
                "seed-shard execution_pairs must contain exactly one seed"
            )
    if exact_output_path and not shard_execution_requested:
        raise ValueError(
            "exact_output_path is reserved for explicit single-seed "
            "shard execution"
        )
    declared_size = estimate_run_size(config)
    if dry_run:
        print(json.dumps(declared_size, indent=2, sort_keys=True))
        return declared_size
    if (
        declared_size["top_level_model_cells"] > LARGE_RUN_THRESHOLD
        and not allow_large_run
    ):
        raise ValueError(
            "Large run requires --allow-large-run: declared grid contains "
            f"{declared_size['top_level_model_cells']:,} top-level model cells, "
            f"above the {LARGE_RUN_THRESHOLD:,} safety threshold."
        )
    model_params_path = Path(config.model_params)
    raw_loaded = load_input(config.schema, config.outcome)
    if (
        raw_loaded.schema.split_mode == "internal_random"
        and not 0.0 < config.test_size < 1.0
    ):
        raise ValueError("test_size must be strictly between 0 and 1")
    loaded, source_definitions = validate_input(
        raw_loaded,
        config.outcome,
        models=config.models,
        min_n=config.min_n,
        test_size=config.test_size,
        seed=config.seed,
    )
    schema = loaded.schema
    task = schema.task
    dataset = schema.dataset
    data_path = schema.table
    test_path = schema.test_table
    frame = loaded.train
    predictors = list(loaded.predictors)
    feature_units = [group.name for group in source_definitions]
    feature_groups = {
        group.name: list(group.features) for group in source_definitions
    }
    groups_by_name = {group.name: group for group in source_definitions}
    selected_model_params = load_model_params(
        model_params_path,
        task=task,
        models=config.models,
    )
    resolved_selected_model_params = resolved_model_params(selected_model_params)
    algorithm_version = load_algorithm_version(model_params_path)
    log_progress(
        "loaded data "
        f"path={data_path} rows={len(frame)} sources={len(feature_units)} "
        f"predictors={len(predictors)} outcome={config.outcome} task={task}"
    )
    split_mode = schema.split_mode
    fixed_split: SplitData | None = None
    if split_mode == "external_test":
        assert loaded.test is not None and test_path is not None
        if not np.isclose(config.test_size, 0.3):
            log_progress(
                "external test schema supplied; ignoring test_size because the "
                "test split is fixed by schema.test_table"
            )
        fixed_split = external_test_split(
            frame, loaded.test, predictors, config.outcome
        )
        log_progress(
            "loaded external test data "
            f"path={test_path} rows={len(loaded.test)} "
            f"usable_test_rows={len(fixed_split.X_test)}"
        )

    semantic_contract = {
        "kind": "nk_grid" if task == "regression" else "nk_grid_classification",
        "algorithm_version": algorithm_version,
        "dataset": dataset,
        "outcome": config.outcome,
        "task": task,
        "split_mode": split_mode,
        "split_seed": config.seed,
        "test_size": config.test_size if split_mode == "internal_random" else None,
        "predictors": predictors,
        "model": list(config.models),
        "resolved_model_params": resolved_selected_model_params,
        "imputation": dict(schema.imputation),
        "feature_universe": dict(schema.semantic_contract.get("feature_universe", {})),
        "environment_overrides": model_run_settings(config.models),
    }
    metadata = build_experiment_metadata(
        kind="nk_grid" if task == "regression" else "nk_grid_classification",
        experiment_id=config.experiment_id,
        data_version=config.data_version,
        model_spec_version=config.model_spec_version,
        outcome=config.outcome,
        test_size=config.test_size,
        split_seed=config.seed,
        algorithm_version=algorithm_version,
        semantic_contract=semantic_contract,
        split_mode=split_mode,
    )
    row_metadata = {field: metadata[field] for field in ROW_METADATA_FIELDS}

    split_seeds = sorted({seed for seed, _ in execution_pairs})
    if fixed_split is None:
        splits = {
            seed: split_frame(
                frame,
                predictors,
                config.outcome,
                test_size=config.test_size,
                seed=seed,
                task=task,
            )
            for seed in split_seeds
        }
    else:
        splits = {seed: fixed_split for seed in split_seeds}
    n_grid = np.asarray(config.n_grid, dtype=int) if config.n_grid else log2_size_grid(
        len(next(iter(splits.values())).X_train),
        config.n_sizes_n,
        config.max_n,
        min_size=config.min_n,
    )
    k_grid = np.asarray(config.k_grid, dtype=int) if config.k_grid else log2_size_grid(len(feature_units), config.n_sizes_k, config.max_k)
    log_progress(
        "grid "
        f"N={n_grid.tolist()} K={k_grid.tolist()} "
        f"seeds={split_seeds} repeat_pairs={list(execution_pairs)} models={list(config.models)}"
    )

    jobs = [
        (model_name, seed, draw, int(n_samples), int(k_features))
        for seed, draw in execution_pairs
        for k_features in k_grid
        for n_samples in n_grid
        for model_name in config.models
    ]
    expected_rows = len(jobs)
    if expected_rows > LARGE_RUN_THRESHOLD and not allow_large_run:
        raise ValueError(
            f"Large run requires --allow-large-run: {expected_rows:,} top-level model "
            f"cells exceeds the {LARGE_RUN_THRESHOLD:,} safety threshold."
        )
    state = git_state(ROOT)
    if config.preset == "production" and state.get("dirty") is not False:
        raise ValueError(
            "Production runs require a clean Git worktree; commit or stash changes first."
        )

    out_path = (
        Path(config.out)
        if exact_output_path
        else _select_output_path(
            Path(config.out),
            preset=config.preset,
            experiment_id=metadata["experiment_id"],
            jobs=jobs,
            rerun_completed=config.rerun_completed,
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if exact_output_path and manifest_path(out_path).exists():
        try:
            exact_prior = json.loads(
                manifest_path(out_path).read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                "Existing exact-output manifest cannot be parsed"
            ) from exc
        if not isinstance(exact_prior, dict):
            raise ValueError(
                "Existing exact-output manifest must be a JSON object"
            )
        _require_resumable_manifest(exact_prior, metadata)
    existing_index = load_checkpoint_index(out_path)
    indexed_completed = _completed_jobs_for_experiment(
        existing_index,
        metadata["experiment_id"],
    )
    if (
        not config.rerun_completed
        and _verified_complete_artifacts(
            out_path,
            metadata["experiment_id"],
            expected_rows,
        )
    ):
        if not _checkpoint_index_exactly_matches_jobs(
            existing_index,
            metadata["experiment_id"],
            jobs,
            indexed_completed,
        ):
            raise RuntimeError(
                "Completed output failed projected-index integrity: expected "
                "exactly one ok/skipped row for every current model-cell key "
                "and no out-of-design keys. Refusing silent reuse or rewrite."
            )
        completed_manifest_path = manifest_path(out_path)
        completed_manifest = json.loads(
            completed_manifest_path.read_text(encoding="utf-8")
        )
        completed_manifest["updated_at"] = utc_now()
        completed_manifest["design"]["parallelism"] = _parallelism_payload(
            config
        )
        write_json_atomic(completed_manifest_path, completed_manifest)
        log_progress(f"already complete; no-op reuse of verified output: {out_path}")
        return out_path
    # Resume planning needs only identity, cell keys and status. Avoid loading
    # every metric column for a multi-million-row checkpoint.
    existing = existing_index
    completed = _completed_jobs_for_experiment(existing, metadata["experiment_id"])
    pending = [job for job in jobs if job not in completed]
    if max_jobs is not None:
        pending = pending[: int(max_jobs)]
    if pending and not existing.empty and not checkpoint_parts(out_path):
        seed_checkpoint_parts_from_csv(out_path)
    started_at = utc_now()
    current_manifest_path = manifest_path(out_path)
    prior_manifest = _read_prior_manifest(
        current_manifest_path, metadata["experiment_id"]
    )
    if prior_manifest is not None:
        _require_resumable_manifest(prior_manifest, metadata)
        started_at = prior_manifest.get("created_at", started_at)
    initial_manifest = _manifest_payload(
        config=config,
        metadata=metadata,
        out_path=out_path,
        data_path=data_path,
        test_path=test_path,
        model_params_path=model_params_path,
        selected_model_params=selected_model_params,
        frame=frame,
        predictors=predictors,
        split_seeds=split_seeds,
        execution_pairs=execution_pairs,
        splits=splits,
        n_grid=n_grid,
        k_grid=k_grid,
        expected_rows=expected_rows,
        results=existing,
        started_at=started_at,
        dataset=dataset,
        task=task,
        schema_path=schema.path,
        semantic_contract=semantic_contract,
        seed_shard_execution=shard_execution_requested,
    )
    write_json_atomic(
        current_manifest_path,
        _preserve_prior_timings(initial_manifest, prior_manifest),
    )
    log_progress(
        f"jobs total={expected_rows} completed={len(completed)} "
        f"pending={len(pending)} batch_size={config.batch_size} "
        f"chunk_policy={_parallelism_payload(config)['chunk_policy']}"
    )
    # The pending list is now authoritative. Release the full design list,
    # projected index and completed-key set before model fitting so a resumed
    # production task does not retain several duplicate multi-million-cell
    # structures for the lifetime of the run.
    del existing_index, indexed_completed, existing, completed, jobs

    @lru_cache(maxsize=8)
    def cached_draw_orders(seed: int, draw: int) -> DrawOrders:
        """Bounded run-local cache consumed only in the parent process."""

        split = splits[seed]
        return _freeze_draw_orders(
            draw_orders(
                split.X_train.index,
                feature_units,
                seed=seed,
                draw=draw,
            )
        )

    native_process_runner = IsolatedProcessRunner(
        max_attempts=config.native_process_max_attempts,
        timeout_seconds=config.native_process_timeout_seconds,
    )

    def run_cell_group(
        seed: int,
        draw: int,
        n_samples: int,
        k_features: int,
        models: Sequence[str],
        *,
        orders: DrawOrders | None = None,
    ) -> list[dict]:
        split = splits[seed]
        if orders is None:
            orders = draw_orders(
                split.X_train.index, feature_units, seed=seed, draw=draw
            )
        selected_rows = orders.row_index[:n_samples]
        selected_units = [str(unit) for unit in orders.feature_names[:k_features]]
        selected_cols = [
            feature for unit in selected_units for feature in feature_groups[unit]
        ]
        selected_groups = [groups_by_name[unit] for unit in selected_units]
        slice_started = time.perf_counter()
        try:
            X_sub_raw = split.X_train.loc[selected_rows, selected_cols]
            y_sub = split.y_train.loc[selected_rows]
            X_test_raw = split.X_test.loc[:, selected_cols]
        except Exception as exc:
            slice_seconds = time.perf_counter() - slice_started
            failed_rows = []
            for position, model_name in enumerate(models):
                row = _base_row(
                    dataset=dataset,
                    outcome=config.outcome,
                    model_name=model_name,
                    seed=seed,
                    draw=draw,
                    n_samples=n_samples,
                    k_features=k_features,
                    n_train_total=len(split.X_train),
                    n_test_total=len(split.X_test),
                    n_features_total=len(feature_units),
                    k_expanded=len(selected_cols),
                    n_expanded_features_total=len(predictors),
                )
                diagnostics = _empty_diagnostics()
                diagnostics["_slice_seconds"] = (
                    slice_seconds if position == 0 else 0.0
                )
                diagnostics["_peak_rss_bytes"] = _process_peak_rss_bytes()
                failed_rows.append(
                    add_metadata(
                        {
                            **row,
                            **(
                                _empty_metrics()
                                if task == "regression"
                                else _empty_classification_metrics()
                            ),
                            **diagnostics,
                            **(
                                {"task": task}
                                if task == "classification"
                                else {}
                            ),
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                        row_metadata,
                    )
                )
            return failed_rows

        slice_seconds = time.perf_counter() - slice_started
        unobserved = count_unobserved_sources(X_sub_raw, selected_groups)
        prepared: dict[str, Any] = {}
        preparation_errors: dict[str, Exception] = {}

        def run_model(model_name: str, position: int) -> dict:
            model_started = time.perf_counter()
            row = _base_row(
                dataset=dataset,
                outcome=config.outcome,
                model_name=model_name,
                seed=seed,
                draw=draw,
                n_samples=n_samples,
                k_features=k_features,
                n_train_total=len(split.X_train),
                n_test_total=len(split.X_test),
                n_features_total=len(feature_units),
                k_expanded=len(selected_cols),
                n_expanded_features_total=len(predictors),
            )
            row["K_unobserved"] = unobserved
            diagnostics = _empty_diagnostics()
            diagnostics["_slice_seconds"] = (
                slice_seconds if position == 0 else 0.0
            )

            def result_row(
                metrics: dict[str, Any],
                *,
                status: str,
                error: str,
                peak_rss_bytes: int | None = None,
            ) -> dict:
                diagnostics["_cell_wall_seconds"] = (
                    time.perf_counter() - model_started
                )
                diagnostics["_peak_rss_bytes"] = (
                    _process_peak_rss_bytes()
                    if peak_rss_bytes is None
                    else int(peak_rss_bytes)
                )
                return add_metadata(
                    {
                        **row,
                        **metrics,
                        **diagnostics,
                        **(
                            {"task": task}
                            if task == "classification"
                            else {}
                        ),
                        "status": status,
                        "error": error,
                    },
                    row_metadata,
                )

            empty_metrics = (
                _empty_metrics()
                if task == "regression"
                else _empty_classification_metrics()
            )
            if unobserved == k_features:
                return result_row(
                    empty_metrics,
                    status="skipped",
                    error="all_selected_sources_unobserved",
                )
            if (
                task == "regression"
                and model_name in REGRESSION_CV_MIN_N
                and n_samples < REGRESSION_CV_MIN_N[model_name]
            ):
                min_required = REGRESSION_CV_MIN_N[model_name]
                return result_row(
                    empty_metrics,
                    status="skipped",
                    error=(
                        f"below minimum N for {model_name}'s internal CV "
                        f"(requires N>={min_required})"
                    ),
                )
            try:
                mode = (
                    "passthrough"
                    if schema.imputation["model_overrides"].get(model_name)
                    == "passthrough"
                    else "imputed"
                )
                if mode in preparation_errors:
                    raise preparation_errors[mode]
                if mode not in prepared:
                    preprocess_started = time.perf_counter()
                    diagnostics["_preprocess_computed"] = True
                    try:
                        prepared_cell = preprocess_cell(
                            X_sub_raw,
                            X_test_raw,
                            selected_groups,
                            schema.imputation,
                            model_name=model_name,
                        )
                    except Exception as exc:
                        preparation_errors[mode] = exc
                        raise
                    finally:
                        diagnostics["_preprocess_seconds"] = (
                            time.perf_counter() - preprocess_started
                        )
                    if prepared_cell.K_unobserved != unobserved:
                        mismatch = RuntimeError(
                            "preprocessing changed the precomputed "
                            "K_unobserved count"
                        )
                        preparation_errors[mode] = mismatch
                        raise mismatch
                    prepared[mode] = prepared_cell
                prepared_cell = prepared[mode]
                diagnostics["_preprocess_vectorized"] = bool(
                    prepared_cell.X_train.attrs.get("_preprocess_vectorized", False)
                )
                X_prepared = prepared_cell.X_train
                X_test_prepared = prepared_cell.X_test
                k_varying = int(
                    sum(
                        X_prepared.loc[:, list(group.features)]
                        .nunique(dropna=True)
                        .gt(1)
                        .any()
                        for group in selected_groups
                    )
                )
                diagnostics["K_varying"] = k_varying
                diagnostics["underdetermined"] = bool(
                    task == "regression"
                    and model_name == "ols"
                    and _ols_is_underdetermined(X_prepared)
                )
                if task == "classification" and len(np.unique(y_sub)) < 2:
                    return result_row(
                        empty_metrics,
                        status="skipped",
                        error="single-class training sample for classification",
                    )
                if task == "classification" and model_name == "super_learner":
                    min_class_count = int(y_sub.value_counts().min())
                    if min_class_count < 2:
                        return result_row(
                            empty_metrics,
                            status="skipped",
                            error=(
                                "below minimum per-class count for "
                                "super_learner CV"
                            ),
                        )
                if model_name in {"lightgbm", "super_learner"}:
                    log_progress(
                        "cell starting "
                        f"model={model_name} seed={seed} draw={draw} "
                        f"N={n_samples} K={k_features}"
                    )
                if model_name in SERIAL_OUTER_MODELS:
                    X_fit = X_prepared
                    X_test_fit = X_test_prepared
                else:
                    X_fit = X_prepared.copy(deep=True)
                    X_test_fit = X_test_prepared.copy(deep=True)
                fit_arguments = {
                    "model_name": model_name,
                    "model_seed": _model_seed(
                        seed, draw, n_samples, k_features
                    ),
                    "model_n_jobs": config.n_jobs if model_name == "super_learner" else 1,
                    "task": task,
                    "params": selected_model_params[model_name],
                    "X_train": X_fit,
                    "y_train": y_sub,
                    "X_test": X_test_fit,
                }
                if model_name in SERIAL_OUTER_MODELS:
                    fit_result = _run_native_model_cell_locked(
                        native_process_runner,
                        fit_arguments=fit_arguments,
                        on_native_crash=lambda attempt, exc: log_progress(
                            "native subprocess crashed while running "
                            "isolated cell "
                            f"attempt={attempt}/"
                            f"{config.native_process_max_attempts} "
                            f"model={model_name} seed={seed} draw={draw} "
                            f"N={n_samples} K={k_features} error={exc}"
                        ),
                        on_native_timeout=lambda attempt, exc: log_progress(
                            "native subprocess timed out while running "
                            "isolated cell "
                            f"attempt={attempt}/"
                            f"{config.native_process_max_attempts} "
                            f"timeout_seconds="
                            f"{config.native_process_timeout_seconds:g} "
                            f"model={model_name} seed={seed} draw={draw} "
                            f"N={n_samples} K={k_features} error={exc}"
                        ),
                    )
                else:
                    fit_result = _fit_predict_model_cell(**fit_arguments)
                del fit_arguments, X_fit, X_test_fit
                predictions = np.asarray(fit_result["predictions"])
                diagnostics["_fit_seconds"] = fit_result["fit_seconds"]
                diagnostics["_best_rounds"] = fit_result["best_rounds"]
                diagnostics["converged"] = fit_result["converged"]
                diagnostics["constant_prediction"] = _constant_prediction(
                    predictions
                )
                if task == "classification":
                    metrics = compute_classification_metrics(
                        split.y_test, predictions, y_sub
                    )
                else:
                    metrics = compute_regression_metrics(
                        split.y_test, predictions, y_sub
                    )
                return result_row(
                    metrics,
                    status="ok",
                    error="",
                    peak_rss_bytes=fit_result["peak_rss_bytes"],
                )
            except Exception as exc:
                return result_row(
                    empty_metrics,
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )

        try:
            return [
                run_model(model_name, position)
                for position, model_name in enumerate(models)
            ]
        finally:
            prepared.clear()
            preparation_errors.clear()
            del X_sub_raw, y_sub, X_test_raw

    pending_cell_groups: dict[
        tuple[int, int, int, int], list[tuple[str, int, int, int, int]]
    ] = {}
    for job in pending:
        pending_cell_groups.setdefault(job[1:], []).append(job)
    total_batches = (
        int(np.ceil(len(pending) / config.batch_size)) if pending else 0
    )
    graceful_stop = False
    stop_before_materialization = False
    processed_rows = 0
    checkpoint_buffer: list[dict] = []
    checkpoint_batch_index = 0
    # Array workers are the only outer concurrency layer.  Keep one complete
    # cell group together so its imputation cache remains shared, but execute
    # groups serially: no joblib windows and therefore no window barrier.
    for cell_key, cell_jobs in pending_cell_groups.items():
        seed, draw, n_samples, k_features = cell_key
        cell_rows = run_cell_group(
            seed,
            draw,
            n_samples,
            k_features,
            tuple(job[0] for job in cell_jobs),
            orders=cached_draw_orders(seed, draw),
        )
        checkpoint_buffer.extend(cell_rows)
        while len(checkpoint_buffer) >= config.batch_size:
            checkpoint_batch_index += 1
            batch_rows = checkpoint_buffer[: config.batch_size]
            del checkpoint_buffer[: config.batch_size]
            log_progress(
                f"batch {checkpoint_batch_index}/{total_batches} starting "
                f"jobs={len(batch_rows)}"
            )
            part = write_checkpoint_part(batch_rows, out_path)
            ok_count = sum(row.get("status") == "ok" for row in batch_rows)
            failed_count = sum(
                row.get("status") == "failed" for row in batch_rows
            )
            skipped_count = sum(
                row.get("status") == "skipped" for row in batch_rows
            )
            log_progress(
                f"batch {checkpoint_batch_index}/{total_batches} wrote "
                f"checkpoint new_rows={len(batch_rows)} ok={ok_count} "
                f"failed={failed_count} skipped={skipped_count} "
                f"part={part.name if part else 'none'} out={out_path}"
            )
            processed_rows += len(batch_rows)
            if stop_after_batch is not None and stop_after_batch():
                if processed_rows < len(pending):
                    graceful_stop = True
                    log_progress(
                        "graceful stop requested; latest batch is checkpointed "
                        "and remaining cells will resume on the next invocation"
                    )
                    break
                if defer_materialization_on_stop:
                    graceful_stop = True
                    stop_before_materialization = True
                    log_progress(
                        "graceful stop arrived after the final cell checkpoint; "
                        "full CSV materialization is deferred to the next invocation"
                    )
                    break
                log_progress(
                    "graceful stop arrived after the final pending batch; "
                    "the run will finalize without requeue"
                )
        if graceful_stop:
            break
    if checkpoint_buffer and not graceful_stop:
        checkpoint_batch_index += 1
        batch_rows = checkpoint_buffer
        log_progress(
            f"batch {checkpoint_batch_index}/{total_batches} starting "
            f"jobs={len(batch_rows)}"
        )
        part = write_checkpoint_part(batch_rows, out_path)
        ok_count = sum(row.get("status") == "ok" for row in batch_rows)
        failed_count = sum(row.get("status") == "failed" for row in batch_rows)
        skipped_count = sum(row.get("status") == "skipped" for row in batch_rows)
        log_progress(
            f"batch {checkpoint_batch_index}/{total_batches} wrote checkpoint "
            f"new_rows={len(batch_rows)} ok={ok_count} failed={failed_count} "
            f"skipped={skipped_count} "
            f"part={part.name if part else 'none'} out={out_path}"
        )
        processed_rows += len(batch_rows)
        if stop_after_batch is not None and stop_after_batch():
            if defer_materialization_on_stop:
                graceful_stop = True
                stop_before_materialization = True
                log_progress(
                    "graceful stop arrived after the final cell checkpoint; "
                    "full CSV materialization is deferred to the next invocation"
                )
            else:
                log_progress(
                    "graceful stop arrived after the final pending batch; "
                    "the run will finalize without requeue"
                )
    native_process_runner.close()
    if not pending:
        log_progress("no pending jobs; checkpoint is already complete")
    if (
        not graceful_stop
        and defer_materialization_on_stop
        and stop_after_batch is not None
        and stop_after_batch()
    ):
        graceful_stop = True
        stop_before_materialization = True
        log_progress(
            "graceful stop observed before full CSV materialization; "
            "finalization is deferred to the next invocation"
        )
    materialization_deferred = graceful_stop and defer_materialization_on_stop
    if materialization_deferred:
        # Slurm's advance-signal watchdog should wait only for the current
        # atomic checkpoint, not a full multi-million-row CSV rewrite.
        results = load_checkpoint_index(out_path)
        result_summary = None
    else:
        materialization = merge_checkpoint_parts(
            out_path,
            experiment_id=metadata["experiment_id"],
            drop_output_columns=[
                "_fit_seconds",
                "_best_rounds",
                "_preprocess_seconds",
                "_preprocess_computed",
                "_slice_seconds",
                "_cell_wall_seconds",
                "_peak_rss_bytes",
            ],
        )
        results = None
        result_summary = materialization.summary
    final_manifest = _manifest_payload(
        config=config,
        metadata=metadata,
        out_path=out_path,
        data_path=data_path,
        test_path=test_path,
        model_params_path=model_params_path,
        selected_model_params=selected_model_params,
        frame=frame,
        predictors=predictors,
        split_seeds=split_seeds,
        execution_pairs=execution_pairs,
        splits=splits,
        n_grid=n_grid,
        k_grid=k_grid,
        expected_rows=expected_rows,
        results=results,
        result_summary=result_summary,
        started_at=started_at,
        dataset=dataset,
        task=task,
        schema_path=schema.path,
        semantic_contract=semantic_contract,
        seed_shard_execution=shard_execution_requested,
    )
    _preserve_prior_timings(final_manifest, prior_manifest)
    if materialization_deferred:
        final_manifest["output"]["csv_materialization_deferred"] = True
        final_manifest["diagnostics"] = {
            "deferred_until_completion": True,
            "by_model": {},
        }
    if graceful_stop:
        violation = None
        if stop_before_materialization:
            final_manifest["completion"]["status"] = "incomplete"
            final_manifest["completion"]["checkpoint_rows_complete"] = True
        final_manifest["failure_policy"]["passed"] = None
        final_manifest["failure_policy"]["violation"] = None
        final_manifest["termination"] = {
            "reason": (
                "graceful_stop_before_materialization"
                if stop_before_materialization
                else "graceful_stop_after_batch"
            ),
            "resumable": True,
        }
    else:
        violation = None if defer_failure_policy else _failure_policy_violation(final_manifest)
        final_manifest["failure_policy"]["passed"] = None if defer_failure_policy else violation is None
        final_manifest["failure_policy"]["violation"] = None if defer_failure_policy else violation
    write_json_atomic(current_manifest_path, final_manifest)
    persisted_manifest = json.loads(current_manifest_path.read_text(encoding="utf-8"))
    if violation is None and _prune_checkpoint_parts(out_path, persisted_manifest):
        persisted_manifest["output"]["checkpoint_parts_deleted"] = True
        write_json_atomic(current_manifest_path, persisted_manifest)
    if violation is not None:
        raise RunFailureThresholdExceeded(
            f"{violation}; output and checkpoint shards were retained at {out_path}"
        )
    return out_path


def parse_args() -> NKGridConfig:
    parser = argparse.ArgumentParser(
        description="Run joint log-scale N x K prediction-quality sweeps."
    )
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--outcome", required=True)
    parser.add_argument("--out", required=True, type=Path)
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
    parser.add_argument("--failed-abs-threshold", type=int, default=50)
    parser.add_argument("--failed-ratio-threshold", type=float, default=0.05)
    parser.add_argument("--native-process-max-attempts", type=int, default=2)
    parser.add_argument(
        "--native-process-timeout-seconds",
        type=float,
        default=21_600.0,
        help="Kill and retry an isolated native-model cell after this deadline.",
    )
    parser.add_argument("--model-params", default=str(DEFAULT_MODEL_PARAMS_PATH))
    parser.add_argument("--allow-large-run", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--rerun-completed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For preset runs, create a new timestamped output when an identical "
            "completed run exists (use --no-rerun-completed to reuse it)."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")),
    )
    args = parser.parse_args()
    return NKGridConfig(
        schema=args.schema,
        out=args.out,
        outcome=args.outcome,
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
        model_params=Path(args.model_params),
        failed_abs_threshold=args.failed_abs_threshold,
        failed_ratio_threshold=args.failed_ratio_threshold,
        native_process_max_attempts=args.native_process_max_attempts,
        native_process_timeout_seconds=args.native_process_timeout_seconds,
        allow_large_run=args.allow_large_run,
        dry_run=args.dry_run,
        rerun_completed=args.rerun_completed,
    )


def main() -> None:
    run_nk_grid(parse_args())


if __name__ == "__main__":
    main()
