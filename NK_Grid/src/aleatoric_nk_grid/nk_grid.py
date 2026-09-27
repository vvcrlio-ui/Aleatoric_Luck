"""Joint N x K sweeps for long-format prediction quality tables."""

from __future__ import annotations
from .phase_timing import timed_phase
from .grid_contract import select_grid_points, validate_size_grid
from .config import (
    NKGridConfig,
    execution_groups_for_models, group_repeat_pairs_by_seed, resolve_repeat_pairs,
)

import json
import resource
import sys
import threading
import time
from dataclasses import dataclass
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
    d2_absolute_error_score,
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

ROOT = Path(__file__).resolve().parents[2]

from .evaluation import r2_against_training_mean, regression_denominators, METRIC_DEFINITION_VERSION
from .execution_contract import (
    CellExecutionSpec,
    ContractError,
    git_repository_root,
    resolve_repo_locator,
    sha256_file,
)
from .experiment import (
    SERIAL_OUTER_MODELS,
    add_metadata,
    build_experiment_metadata,
    git_state,
    model_run_settings,
)
from .helpers_logging import log_progress
from .ingest import LoadedInput, load_input
from .model_registry import (
    SUPPORTED_MODEL_NAMES,
    load_algorithm_version,
    load_model_params,
    make_model,
    reject_removed_model,
    resolved_model_params,
)
from .native_process import IsolatedProcessRunner
from .preprocessing import (
    FoldPreprocessor,
    SourceGroup,
    count_unobserved_sources,
    count_varying_sources,
    preprocess_cell,
    sampling_units,
)
from .validate_input import REGRESSION_CV_MIN_N, validate_input


LARGE_RUN_THRESHOLD = 250_000

# Row-level metadata is deliberately scalar-only. The complete artifact-level
# identity and semantic contract are recorded once per run, not once per row.
ROW_METADATA_FIELDS = (
    "experiment_id",
    "experiment_kind",
    "algorithm_version",
    "outcome",
    "test_size",
    "split_mode",
    "split_seed",
)


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
    "mse", "null_mse_train_mean", "test_target_variance", "skill_train_mean", "r2_test_mean",
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

BASE_RESULT_COLUMNS = (
    "dataset", "outcome", "model", "seed", "draw", "N", "K",
    "split_random_state", "n_train_total", "n_test_total", "n_features_total",
    "K_expanded", "n_expanded_features_total", "K_unobserved",
)
STABLE_DIAGNOSTIC_RESULT_COLUMNS = (
    "mlp_diagnostics_json",
    "K_varying", "constant_prediction", "underdetermined", "converged",
    "_preprocess_vectorized",
)


def public_result_columns(task: str) -> tuple[str, ...]:
    """Return the byte-serialized result schema for one immutable analysis."""

    if task == "regression":
        metrics = METRIC_COLUMNS
        task_column: tuple[str, ...] = ()
    elif task == "classification":
        metrics = CLASSIFICATION_METRIC_COLUMNS
        task_column = ("task",)
    else:
        raise ValueError(f"unsupported NK-grid task {task!r}")
    # ``outcome`` originates in metadata and is then overwritten by the base
    # row without changing insertion order.  Deduplicate exactly as Python's
    # dictionary expansion does before CSV serialization.
    return tuple(dict.fromkeys((*ROW_METADATA_FIELDS, *BASE_RESULT_COLUMNS, *metrics, *STABLE_DIAGNOSTIC_RESULT_COLUMNS, *task_column, "status", "error")))


def _frozen_input_provenance_for_schema(schema: Any) -> dict[str, dict[str, str]]:
    """Freeze numeric inputs with the same fields used by dynamic planning."""

    definition_value = Path(str(schema.feature_universe["definition_file"]))
    definition = definition_value if definition_value.is_absolute() else (schema.path.parent / definition_value).resolve()
    candidates = {
        "training_table": schema.table,
        "external_test_table": schema.test_table,
        "feature_manifest": schema.feature_manifest,
        "feature_universe_definition": definition,
        "provenance": schema.table.parent / "provenance.json",
    }
    frozen: dict[str, dict[str, str]] = {}
    for name, candidate in candidates.items():
        if candidate is None:
            continue
        path = Path(candidate).resolve()
        if name == "provenance" and not path.exists():
            continue
        if not path.is_file():
            raise ContractError(f"numeric input provenance file is missing: {name}={path}")
        frozen[name] = {"path": str(path), "sha256": sha256_file(path)}
    if not frozen:
        raise ContractError("CellExecutionSpec found no numeric input provenance")
    return frozen


def project_public_result(row: Mapping[str, object], *, header: Sequence[str]) -> dict[str, object]:
    """Project one computed row into the immutable public result codec.

    Session computation intentionally returns local diagnostic telemetry too.
    Workers must never choose a different subset or preserve whatever dict
    order happened to be produced by a model; this one projection is the
    boundary shared with ``public_result_columns`` and final CSV equality.
    """

    columns = tuple(str(column) for column in header)
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("public result header must be non-empty and unique")
    missing = [column for column in columns if column not in row]
    if missing:
        raise ValueError(f"computed result lacks public columns: {missing}")
    return {column: row[column] for column in columns}

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


@dataclass(frozen=True)
class SplitIndexes:
    """A split's stable row labels, not a copied predictor DataFrame."""

    train_index: pd.Index
    test_index: pd.Index
    external_test: bool


class SplitIndexManager:
    """Lazily cache only train/test labels for each seed, never seed × predictor frames."""

    def __init__(
        self,
        *,
        frame: pd.DataFrame,
        external_frame: pd.DataFrame | None,
        predictors: Sequence[str],
        outcome: str,
        test_size: float,
        task: str,
    ) -> None:
        self.frame = frame
        self.external_frame = external_frame
        self.predictors = tuple(str(value) for value in predictors)
        self.outcome = str(outcome)
        self.test_size = float(test_size)
        self.task = str(task)
        self._cache: dict[int, SplitIndexes] = {}
        if external_frame is not None:
            fixed = external_test_split(
                frame,
                external_frame,
                self.predictors,
                self.outcome,
            )
            self._external = SplitIndexes(
                train_index=fixed.X_train.index.copy(),
                test_index=fixed.X_test.index.copy(),
                external_test=True,
            )
        else:
            self._external = None

    def for_seed(self, seed: int) -> SplitIndexes:
        if self._external is not None:
            return self._external
        frozen = self._cache.get(int(seed))
        if frozen is not None:
            return frozen
        target = self.frame[self.outcome]
        train_index, test_index = train_test_split(
            self.frame.index,
            test_size=self.test_size,
            random_state=int(seed),
            stratify=target if self.task == "classification" else None,
        )
        frozen = SplitIndexes(
            pd.Index(train_index),
            pd.Index(test_index),
            False,
        )
        self._cache[int(seed)] = frozen
        return frozen


def log2_size_grid(
    total: int,
    n_sizes: int,
    max_size: int | None = None,
    *,
    min_size: int = 1,
) -> np.ndarray:
    """Return exactly ``n_sizes`` distinct integer sizes spanning ``[min_size, upper]``.

    Sizes follow a base-2 log grid; rounding collisions are shifted to the nearest
    free integer, both endpoints are exact, and an interval too small to hold
    ``n_sizes`` distinct integers raises ``ValueError``.
    """

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
    if upper - min_size + 1 < n_sizes:
        raise ValueError(
            f"cannot place {n_sizes} distinct integer sizes in [{min_size}, {upper}]"
        )
    ideal = np.round(
        np.logspace(np.log2(min_size), np.log2(upper), num=n_sizes, base=2)
    ).astype(int)
    out = ideal.copy()
    out[0] = min_size
    for i in range(1, n_sizes):
        out[i] = max(ideal[i], out[i - 1] + 1)
    out[-1] = upper
    for i in range(n_sizes - 2, -1, -1):
        out[i] = min(out[i], out[i + 1] - 1)
    return out


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
        **regression_denominators(y_true, preds, train),
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


def _empty_diagnostics() -> dict[str, float | bool | str]:
    return {
        "mlp_diagnostics_json": "",
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


def _validate_config(config: NKGridConfig) -> None:
    """Reject invalid run controls before dry-run arithmetic or data loading."""

    if config.grid_selection not in ("all", "min_middle_max"):
        raise ValueError("grid_selection must be all or min_middle_max")
    if type(config.checkpoint_retention) is not str or config.checkpoint_retention not in {"keep", "delete"}:
        raise ValueError("checkpoint_retention must be keep or delete")

    for name in ("n_grid", "k_grid"):
        if getattr(config, name) is not None:
            validate_size_grid(getattr(config, name), name)
    if config.grid_selection == "min_middle_max":
        for grid_name, count_name in (("n_grid", "n_sizes_n"), ("k_grid", "n_sizes_k")):
            values = getattr(config, grid_name)
            count = len(values) if values is not None else getattr(config, count_name)
            if count < 3:
                raise ValueError(f"{grid_name} needs at least three production grid points for pilot")
    for field in (
        "n_seeds",
        "n_draws",
        "n_sizes_n",
        "n_sizes_k",
        "min_n",
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
    if config.native_process_max_attempts < 1:
        raise ValueError("native_process_max_attempts must be at least 1")
    if config.native_process_timeout_seconds <= 0:
        raise ValueError("native_process_timeout_seconds must be greater than zero")
    for field in ("experiment_id", "data_version", "model_spec_version"):
        value = getattr(config, field)
        if not isinstance(value, str) or not value or len(value) > 80 or not value.isascii() or not all(char.isalnum() or char in "._-" for char in value):
            raise ValueError(f"{field} must contain 1-80 ASCII letters, digits, dots, underscores or hyphens")
    group_repeat_pairs_by_seed(resolve_repeat_pairs(config))


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
    preprocessor=None,
    prediction_request=None,
) -> dict[str, Any]:
    """Fit and predict one cell; safe to execute in an isolated subprocess."""

    if prediction_request and model_name != "super_learner":
        from .prediction_training import train_base_predictions
        fitted = train_base_predictions(model_name=model_name, model_seed=model_seed,
            task=task, params=params, X_train=X_train, y_train=y_train, X_test=X_test,
            preprocessor=preprocessor, n_jobs=model_n_jobs,
            mode=prediction_request["mode"], oof_folds=prediction_request.get("oof_folds", 5),
            resume_folds=prediction_request.get("resume_folds"),
            persist_fold=prediction_request.get("persist_fold"),
            pipeline_id=prediction_request.get("pipeline_id"),
            formal_params=prediction_request.get("formal_params"))
        model = fitted.pop("model")
        meta = fitted["metadata"]
        return {"predictions": fitted["predictions"], "prediction_cache_data": fitted,
                "status": meta["status"], "reason": meta["reason"],
                "mlp_diagnostics_json": "", "fit_seconds": meta["full_fit_seconds"] + meta["oof_fit_seconds"],
                "best_rounds": _model_best_rounds(model), "converged": meta["converged"],
                "solver": _model_solver(model), "iterations": _model_iterations(model),
                "alpha": _model_alpha(model), "peak_rss_bytes": _process_peak_rss_bytes()}
    model = make_model(
        model_name,
        seed=model_seed,
        n_jobs=model_n_jobs,
        task=task,
        params=params,
        preprocessor=preprocessor,
    )
    if prediction_request and model_name == "super_learner":
        model.capture_predictions = True
    fit_started = time.perf_counter()
    model.fit(X_train, y_train)
    predictions = (
        _positive_class_probability(model, X_test)
        if task == "classification"
        else np.asarray(model.predict(X_test))
    )
    output = {
        "predictions": predictions,
        "mlp_diagnostics_json": json.dumps(model.diagnostics_, sort_keys=True, separators=(",", ":"), allow_nan=False) if hasattr(model, "diagnostics_") else "",
        "fit_seconds": time.perf_counter() - fit_started,
        "best_rounds": _model_best_rounds(model),
        "converged": _model_converged(model),
        "solver": _model_solver(model),
        "iterations": _model_iterations(model),
        "alpha": _model_alpha(model),
        "peak_rss_bytes": _process_peak_rss_bytes(),
    }
    if prediction_request and model_name == "super_learner":
        from .prediction_training import formal_sl_predictions
        output["prediction_cache_data"] = formal_sl_predictions(model, X_test, predictions)
        output["prediction_cache_data"]["metadata"].update(status="ok", reason="")
    return output


@timed_phase("input.resolve_grids")
def resolve_input_grids(config, loaded, source_definitions):
    """Resolve the design against validated outcome-specific split capacities."""
    manager = SplitIndexManager(
        frame=loaded.train, external_frame=loaded.test if loaded.schema.split_mode == "external_test" else None,
        predictors=loaded.predictors, outcome=config.outcome,
        test_size=config.test_size, task=loaded.schema.task,
    )
    seeds = tuple(dict.fromkeys(seed for seed, _ in resolve_repeat_pairs(config)))
    capacity = len(manager.for_seed(seeds[0]).train_index)
    units = sampling_units(source_definitions)
    n_grid = config.n_grid if config.n_grid is not None else log2_size_grid(
        capacity, config.n_sizes_n, config.max_n, min_size=config.min_n)
    k_grid = config.k_grid if config.k_grid is not None else log2_size_grid(
        len(units), config.n_sizes_k, config.max_k)
    for seed in seeds:
        validate_size_grid(n_grid, f"N (seed={seed})", len(manager.for_seed(seed).train_index))
    validate_size_grid(k_grid, "K", len(units))
    n_grid = select_grid_points(n_grid, config.grid_selection, "N")
    k_grid = select_grid_points(k_grid, config.grid_selection, "K")
    return np.asarray(n_grid, dtype=int), np.asarray(k_grid, dtype=int)


class NKGridExecutionSession:
    """One validated input/model/native-runner lifetime for many cell groups.

    It has no output-path, manifest, checkpoint, lock, assignment, round, or
    generation knowledge.  Workers use ``run_cell_group`` as their only
    numerical primitive.
    """

    def __init__(
        self,
        *,
        config: NKGridConfig,
        spec: CellExecutionSpec | None,
        repo_root: Path | None = None,
        loaded: LoadedInput,
        source_definitions: Sequence[SourceGroup],
        selected_model_params: Mapping[str, Mapping[str, object]],
        algorithm_version: str,
    ) -> None:
        self.config = config
        self.spec = spec
        self.repo_root = Path(repo_root).resolve() if repo_root is not None else git_repository_root(ROOT)
        self.loaded = loaded
        self.source_definitions = tuple(source_definitions)
        self.schema = loaded.schema
        self.task = self.schema.task
        self.dataset = self.schema.dataset
        self.data_path = self.schema.table
        self.test_path = self.schema.test_table
        self.frame = loaded.train
        self.external_frame = loaded.test
        self.predictors = tuple(loaded.predictors)
        units = sampling_units(self.source_definitions)
        self.feature_units = tuple(unit.name for unit in units)
        self.feature_groups = {unit.name: unit.features for unit in units}
        self.groups_by_unit = {unit.name: unit.groups for unit in units}
        self.selected_model_params = dict(selected_model_params)
        self.algorithm_version = algorithm_version
        self.split_manager = SplitIndexManager(
            frame=self.frame,
            external_frame=self.external_frame if self.schema.split_mode == "external_test" else None,
            predictors=self.predictors,
            outcome=config.outcome,
            test_size=config.test_size,
            task=self.task,
        )
        self.repeat_pairs = resolve_repeat_pairs(config)
        self.n_grid, self.k_grid = resolve_input_grids(config, loaded, source_definitions)
        self.semantic_contract = {
            "metric_definition_version": METRIC_DEFINITION_VERSION if self.task == "regression" else "classification-v1",
            "kind": "nk_grid" if self.task == "regression" else "nk_grid_classification",
            "algorithm_version": algorithm_version,
            "dataset": self.dataset,
            "outcome": config.outcome,
            "task": self.task,
            "split_mode": self.schema.split_mode,
            "split_seed": config.seed,
            "test_size": config.test_size if self.schema.split_mode == "internal_random" else None,
            "predictors": list(self.predictors),
            "model": list(config.models),
            "resolved_model_params": resolved_model_params(self.selected_model_params),
            "imputation": dict(self.schema.imputation),
            "feature_universe": dict(self.schema.semantic_contract.get("feature_universe", {})),
            "environment_overrides": model_run_settings(config.models),
        }
        self.metadata = build_experiment_metadata(
            kind="nk_grid" if self.task == "regression" else "nk_grid_classification",
            experiment_id=config.experiment_id,
            data_version=config.data_version,
            model_spec_version=config.model_spec_version,
            outcome=config.outcome,
            test_size=config.test_size,
            split_seed=config.seed,
            algorithm_version=algorithm_version,
            semantic_contract=self.semantic_contract,
            split_mode=self.schema.split_mode,
        )
        self.row_metadata = {field: self.metadata[field] for field in ROW_METADATA_FIELDS}
        self._draw_orders: dict[tuple[int, int], DrawOrders] = {}
        self._runner = IsolatedProcessRunner(
            max_attempts=config.native_process_max_attempts,
            timeout_seconds=config.native_process_timeout_seconds,
        )
        self._closed = False
        self._prediction_request = None
        self._validate_spec()

    @classmethod
    def _open_config(
        cls, config: NKGridConfig, *, spec: CellExecutionSpec | None = None,
        repo_root: Path | None = None, input_store=None,
    ) -> "NKGridExecutionSession":
        _validate_config(config)
        def validated_input():
            raw_loaded = load_input(config.schema, config.outcome)
            if raw_loaded.schema.split_mode == "internal_random" and not 0.0 < config.test_size < 1.0:
                raise ValueError("test_size must be strictly between 0 and 1")
            return validate_input(raw_loaded, config.outcome, models=config.models, min_n=config.min_n,
                test_size=config.test_size, seed=config.seed,
                require_id=bool(config.prediction_cache))
        # This contains validated raw frames and fixed source definitions only.
        # Fitted preprocessing remains exclusively inside the original folds.
        loaded, source_definitions = (input_store.load_or_build('validated-raw-input-v1', validated_input)[0]
                                      if input_store is not None else validated_input())
        selected_model_params = load_model_params(config.model_params, task=loaded.schema.task, models=config.models)
        return cls(
            config=config, spec=spec, repo_root=repo_root, loaded=loaded, source_definitions=source_definitions,
            selected_model_params=selected_model_params,
            algorithm_version=load_algorithm_version(config.model_params),
        )

    @classmethod
    def open(cls, spec: CellExecutionSpec, *, repo_root: Path | None = None, input_store=None) -> "NKGridExecutionSession":
        """Open a session from the session-only canonical execution spec."""

        spec = CellExecutionSpec.from_payload(spec.payload)
        root = git_repository_root(ROOT) if repo_root is None else git_repository_root(repo_root)
        schema, model_params = spec.resolve_inputs(repo_root=root)
        value = spec.payload
        config = NKGridConfig(
            schema=schema,
            out=root / ".nk-grid-session-no-output.csv",
            outcome=str(value["outcome"]),
            models=tuple(str(item) for item in value["models"]),
            seed=int(value["split_seed"]),
            test_size=float(value["test_size"]),
            n_seeds=1,
            n_draws=1,
            n_sizes_n=len(value["resolved_n_grid"]),
            n_sizes_k=len(value["resolved_k_grid"]),
            max_n=max(int(item) for item in value["resolved_n_grid"]),
            max_k=max(int(item) for item in value["resolved_k_grid"]),
            batch_size=1,
            n_jobs=int(value["model_n_jobs"]),
            min_n=int(value["min_n"]),
            model_params=model_params,
            native_process_max_attempts=int(value["native_process_max_attempts"]),
            native_process_timeout_seconds=float(value["native_process_timeout_seconds"]),
            preset=value.get("preset") if isinstance(value.get("preset"), str) else None,
            experiment_id=str(value["experiment_id"]),
            data_version=str(value["data_version"]),
            model_spec_version=str(value["model_spec_version"]),
            repeat_plan=tuple((int(pair[0]), int(pair[1])) for pair in value["resolved_repeat_plan"]),
            n_grid=tuple(int(item) for item in value["resolved_n_grid"]),
            k_grid=tuple(int(item) for item in value["resolved_k_grid"]),
            **({"prediction_cache": value["prediction_cache"]} if "prediction_cache" in value else {}),
            **({"execution": value["execution"]} if "execution" in value else {}),
        )
        return cls._open_config(config, spec=spec, repo_root=root, input_store=input_store)

    @classmethod
    def open_from_config(cls, config: NKGridConfig) -> "NKGridExecutionSession":
        """Open a session from a resolved config, as run preparation does."""

        return cls._open_config(config)

    def _validate_spec(self) -> None:
        if self.spec is None:
            return
        value = self.spec.payload
        if tuple(int(item) for item in value["resolved_n_grid"]) != tuple(int(item) for item in self.n_grid):
            raise ContractError("CellExecutionSpec resolved N grid does not match validated input")
        if tuple(int(item) for item in value["resolved_k_grid"]) != tuple(int(item) for item in self.k_grid):
            raise ContractError("CellExecutionSpec resolved K grid does not match validated input")
        if tuple((int(pair[0]), int(pair[1])) for pair in value["resolved_repeat_plan"]) != self.repeat_pairs:
            raise ContractError("CellExecutionSpec repeat plan does not match validated configuration")
        if int(value["model_n_jobs"]) != int(self.config.n_jobs):
            raise ContractError("CellExecutionSpec model_n_jobs does not match training configuration")
        expected_algorithm = value.get("algorithm_version")
        if expected_algorithm is not None and expected_algorithm != self.algorithm_version:
            raise ContractError("CellExecutionSpec algorithm version mismatch")
        expected_params = value.get("resolved_model_params")
        if expected_params and expected_params != resolved_model_params(self.selected_model_params):
            raise ContractError("CellExecutionSpec model parameter mismatch")
        expected_commit = value.get("git_commit")
        actual_git = git_state(ROOT)
        if actual_git.get("commit") != expected_commit:
            raise ContractError("CellExecutionSpec git commit mismatch")
        if bool(value["require_clean_worktree"]) and actual_git.get("dirty") is not False:
            raise ContractError("CellExecutionSpec requires a clean Git worktree")
        if value["environment_overrides"] != model_run_settings(self.config.models):
            raise ContractError("CellExecutionSpec environment override mismatch")
        expected_groups = [
            {"k_features": int(k_features), "groups": [
                {"group": group, "models": list(group_models)}
                for group, group_models in execution_groups_for_models(self.config.models)
            ]}
            for k_features in self.k_grid
        ]
        if value["execution_groups"] != expected_groups:
            raise ContractError("CellExecutionSpec execution groups mismatch")
        for name, entry in dict(value["input_provenance"]).items():
            try:
                resolve_repo_locator(
                    str(entry["path"]), str(entry["sha256"]),
                    repo_root=self.repo_root,
                )
            except (OSError, ContractError) as exc:
                raise ContractError(f"CellExecutionSpec provenance input is unavailable: {name}") from exc

    def _orders(self, seed: int, draw: int, train_index: pd.Index) -> DrawOrders:
        key = (int(seed), int(draw))
        existing = self._draw_orders.get(key)
        if existing is None:
            if len(self._draw_orders) >= 8:
                self._draw_orders.pop(next(iter(self._draw_orders)))
            existing = _freeze_draw_orders(draw_orders(train_index, self.feature_units, seed=seed, draw=draw))
            self._draw_orders[key] = existing
        return existing

    def run_cell_group(
        self, *, seed: int, draw: int, n_samples: int, k_features: int, models: Sequence[str],
    ) -> list[dict[str, object]]:
        """Run one frozen cell group without creating any persistence artefact."""

        if self._closed:
            raise RuntimeError("NKGridExecutionSession is closed")
        validate_size_grid((n_samples,), "N")
        validate_size_grid((k_features,), "K", len(self.feature_units))
        frozen_models = tuple(str(model) for model in models)
        if not frozen_models or len(frozen_models) != len(set(frozen_models)) or not set(frozen_models).issubset(self.config.models):
            raise ValueError("cell group models must be a unique subset of the frozen model list")
        if (int(seed), int(draw)) not in self.repeat_pairs:
            raise ValueError("cell group seed/draw is outside the frozen repeat plan")
        if int(n_samples) not in set(map(int, self.n_grid)) or int(k_features) not in set(map(int, self.k_grid)):
            raise ValueError("cell group N/K is outside the frozen resolved grid")
        indexes = self.split_manager.for_seed(int(seed))
        validate_size_grid((n_samples,), "N", len(indexes.train_index))
        orders = self._orders(int(seed), int(draw), indexes.train_index)
        selected_rows = orders.row_index[: int(n_samples)]
        selected_units = [str(unit) for unit in orders.feature_names[: int(k_features)]]
        if len(selected_rows) != n_samples or len(selected_units) != k_features:
            raise ValueError("sampled N/K does not match declared N/K")
        selected_cols = [feature for unit in selected_units for feature in self.feature_groups[unit]]
        selected_groups = [
            group
            for unit in selected_units
            for group in self.groups_by_unit[unit]
        ]
        test_frame = self.external_frame if indexes.external_test else self.frame
        if test_frame is None:
            raise RuntimeError("validated external test frame is missing")
        started = time.perf_counter()
        try:
            X_sub_raw = self.frame.loc[selected_rows, selected_cols]
            y_sub = self.frame.loc[selected_rows, self.config.outcome]
            X_test_raw = test_frame.loc[indexes.test_index, selected_cols]
            y_test = test_frame.loc[indexes.test_index, self.config.outcome]
        except Exception as exc:
            return self._slice_failure_rows(
                exc, frozen_models, seed=int(seed), draw=int(draw), n_samples=int(n_samples),
                k_features=int(k_features), slice_seconds=time.perf_counter() - started,
                n_train_total=len(indexes.train_index), n_test_total=len(indexes.test_index),
            )
        slice_seconds = time.perf_counter() - started
        unobserved = count_unobserved_sources(X_sub_raw, selected_groups)
        prepared: dict[str, object] = {}; preparation_errors: dict[str, Exception] = {}
        try:
            rows: list[dict[str, object]] = []
            for position, model_name in enumerate(frozen_models):
                rows.append(self._run_model(
                    model_name=model_name, position=position, seed=int(seed), draw=int(draw),
                    n_samples=int(n_samples), k_features=int(k_features), X_sub_raw=X_sub_raw,
                    y_sub=y_sub, X_test_raw=X_test_raw, y_test=y_test,
                    selected_groups=selected_groups,
                    unobserved=unobserved, slice_seconds=slice_seconds, prepared=prepared,
                    preparation_errors=preparation_errors, n_train_total=len(indexes.train_index),
                    n_test_total=len(indexes.test_index),
                ))
            return rows
        finally:
            prepared.clear(); preparation_errors.clear()

    def run_prediction_cell(self, *, seed, draw, n_samples, k_features, model,
                            pipeline_id=None, mode="holdout_oof", oof_folds=5,
                            resume_folds=None, persist_fold=None):
        """Execute one explicit cache task and expose exact ordered sample maps.

        Persistence is the worker's responsibility. This method never marks an
        unpersisted prediction successful at the queue level. Formal SL capture
        is a separate P1 interface, never a base-stage task in base_then_sl.
        """
        from .prediction_training import BASE_LIBRARY
        expected = (f"reported-sl4-{self.task}-v1" if model == "super_learner"
                    else f"{BASE_LIBRARY}/{model}")
        formal_prefix = f"reported-sl4-{self.task}-v1/"
        is_formal = bool(pipeline_id and pipeline_id.startswith(formal_prefix))
        if pipeline_id is not None and pipeline_id != expected and not is_formal:
            raise ValueError(f"unknown prediction pipeline {pipeline_id!r}; expected {expected!r}")
        if self._prediction_request is not None:
            raise RuntimeError("prediction session does not permit nested tasks")
        if mode not in {"holdout", "holdout_oof"}:
            raise ValueError("prediction mode must be holdout or holdout_oof")
        inputs = self.prediction_cell_inputs(seed=seed, draw=draw, n_samples=n_samples,
            k_features=k_features, model=model, oof_folds=oof_folds, pipeline_id=pipeline_id)
        self._prediction_request = {"mode": mode, "oof_folds": oof_folds,
                                    "resume_folds": resume_folds, "persist_fold": persist_fold,
                                    "pipeline_id": pipeline_id,
                                    "formal_params": self.selected_model_params.get("super_learner") if is_formal else None}
        try:
            row = self.run_cell_group(seed=seed, draw=draw, n_samples=n_samples,
                                      k_features=k_features, models=(model,))[0]
        finally:
            self._prediction_request = None
        captured = row.pop("_prediction_cache_data", None)
        if captured is None:
            captured = {"arrays": {}, "metadata": {"status": row["status"], "reason": row.get("error", "")}}
        captured["metadata"].update(inputs["metadata"])
        return {"row": row, "arrays": captured["arrays"], "metadata": captured["metadata"],
                "sample_arrays": inputs["sample_arrays"]}

    def prediction_cell_inputs(self, *, seed, draw, n_samples, k_features, model, oof_folds=5, pipeline_id=None):
        """Return exact identity inputs without fitting any estimator."""
        from .prediction_training import BASE_LIBRARY, make_oof_folds
        formal_prefix = f"reported-sl4-{self.task}-v1/"
        is_formal = bool(pipeline_id and pipeline_id.startswith(formal_prefix))
        if is_formal:
            if "super_learner" not in self.selected_model_params:
                raise ValueError("formal SL pipeline requires configured super_learner parameters")
            oof_folds = int(self.selected_model_params["super_learner"]["cv"])
            internal = pipeline_id[len(formal_prefix):]
            aliases = {"ridge": "ridge", "logistic": "ols", "lightgbm": "lightgbm",
                       "extra_trees": "extra_trees", "shallow_nn": "shallow_neural_network"}
            valid = {"ridge", "extra_trees", "lightgbm", "shallow_nn"} if self.task == "regression" else {"logistic", "lightgbm", "extra_trees", "shallow_nn"}
            if internal not in valid or aliases[internal] != model:
                raise ValueError("formal base recipe does not match the declared public model alias")
        if (int(seed), int(draw)) not in self.repeat_pairs:
            raise ValueError("prediction identity seed/draw outside frozen plan")
        if int(n_samples) not in self.n_grid or int(k_features) not in self.k_grid or model not in self.config.models:
            raise ValueError("prediction identity N/K/model outside frozen plan")
        indexes = self.split_manager.for_seed(int(seed))
        order = self._orders(int(seed), int(draw), indexes.train_index)
        train_index = order.row_index[:int(n_samples)]
        units = [str(unit) for unit in order.feature_names[:int(k_features)]]
        features = [name for unit in units for name in self.feature_groups[unit]]
        if not self.schema.id_column:
            raise ValueError("prediction cache requires an explicit validated sample ID column")
        holdout_frame = self.external_frame if indexes.external_test else self.frame
        def identifiers(values):
            array = values.to_numpy()
            return array.astype(str) if array.dtype.kind == "O" else array
        sample_arrays = {
            "train_ids": identifiers(self.frame.loc[train_index, self.schema.id_column]),
            "holdout_ids": identifiers(holdout_frame.loc[indexes.test_index, self.schema.id_column]),
            "train_positions": np.asarray(train_index, dtype=str),
            "holdout_positions": np.asarray(indexes.test_index, dtype=str),
            "y_train": self.frame.loc[train_index, self.config.outcome].to_numpy(dtype=np.float64),
            "y_holdout": holdout_frame.loc[indexes.test_index, self.config.outcome].to_numpy(dtype=np.float64),
            "feature_names": np.asarray(features, dtype=str), "source_names": np.asarray(units, dtype=str)}
        try:
            folds = make_oof_folds(sample_arrays["y_train"], self.task, oof_folds)
            sample_arrays["oof_fold"] = np.full(n_samples, -1, dtype=np.int64)
            for fold, (_, valid) in enumerate(folds):
                sample_arrays["oof_fold"][valid] = fold
            fold_status = "ok"
        except ValueError as exc:
            folds = ()
            fold_status = str(exc)
        expected = pipeline_id if is_formal else (f"reported-sl4-{self.task}-v1" if model == "super_learner"
                    else f"{BASE_LIBRARY}/{model}")
        return {"sample_arrays": sample_arrays, "folds": folds,
                "metadata": {"pipeline_id": expected, "model_name": internal if is_formal else model, "task": self.task,
                    "model_seed": _model_seed(seed, draw, n_samples, k_features),
                    "model_params": resolved_model_params({"super_learner": self.selected_model_params["super_learner"]}) if is_formal else resolved_model_params({model: self.selected_model_params[model]}),
                    "imputation": dict(self.schema.imputation), "fold_rule": "ordered-stratified-v1" if self.task == "classification" else "ordered-kfold-v1",
                    "oof_folds": oof_folds, "fold_status": fold_status}}

    def _base(self, *, model_name: str, seed: int, draw: int, n_samples: int, k_features: int, n_train_total: int, n_test_total: int) -> dict[str, object]:
        return _base_row(
            dataset=self.dataset, outcome=self.config.outcome, model_name=model_name,
            seed=seed, draw=draw, n_samples=n_samples, k_features=k_features,
            n_train_total=n_train_total, n_test_total=n_test_total,
            n_features_total=len(self.feature_units),
            k_expanded=0, n_expanded_features_total=len(self.predictors),
        )

    def _slice_failure_rows(self, exc: Exception, models: Sequence[str], *, seed: int, draw: int, n_samples: int, k_features: int, slice_seconds: float, n_train_total: int, n_test_total: int) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        metrics = _empty_metrics() if self.task == "regression" else _empty_classification_metrics()
        for position, model_name in enumerate(models):
            row = self._base(model_name=model_name, seed=seed, draw=draw, n_samples=n_samples, k_features=k_features, n_train_total=n_train_total, n_test_total=n_test_total)
            diagnostics = _empty_diagnostics(); diagnostics["_slice_seconds"] = slice_seconds if position == 0 else 0.0; diagnostics["_peak_rss_bytes"] = _process_peak_rss_bytes()
            result.append(add_metadata({**row, **metrics, **diagnostics, **({"task": self.task} if self.task == "classification" else {}), "status": "failed", "error": f"{type(exc).__name__}: {exc}"}, self.row_metadata))
        return result

    def _run_model(self, *, model_name: str, position: int, seed: int, draw: int, n_samples: int, k_features: int, X_sub_raw: pd.DataFrame, y_sub: pd.Series, X_test_raw: pd.DataFrame, y_test: pd.Series, selected_groups: Sequence[SourceGroup], unobserved: int, slice_seconds: float, prepared: dict[str, object], preparation_errors: dict[str, Exception], n_train_total: int, n_test_total: int) -> dict[str, object]:
        if len(X_sub_raw) != n_samples or len(y_sub) != n_samples or len(sampling_units(selected_groups)) != k_features:
            raise ValueError("fit input does not match declared N/K")
        if X_sub_raw.shape[1] != sum(len(g.features) for g in selected_groups):
            raise ValueError("fit input does not match expanded feature count")
        model_started = time.perf_counter()
        row = self._base(model_name=model_name, seed=seed, draw=draw, n_samples=n_samples, k_features=k_features, n_train_total=n_train_total, n_test_total=n_test_total)
        row["K_expanded"] = X_sub_raw.shape[1]; row["K_unobserved"] = unobserved
        diagnostics = _empty_diagnostics(); diagnostics["_slice_seconds"] = slice_seconds if position == 0 else 0.0
        empty_metrics = _empty_metrics() if self.task == "regression" else _empty_classification_metrics()

        def result(metrics: dict[str, object], *, status: str, error: str, peak: int | None = None) -> dict[str, object]:
            diagnostics["_cell_wall_seconds"] = time.perf_counter() - model_started
            diagnostics["_peak_rss_bytes"] = _process_peak_rss_bytes() if peak is None else int(peak)
            return add_metadata({**row, **metrics, **diagnostics, **({"task": self.task} if self.task == "classification" else {}), "status": status, "error": error}, self.row_metadata)

        if unobserved == k_features:
            return result(empty_metrics, status="skipped", error="all_selected_sources_unobserved")
        formal_recipe = bool(self._prediction_request and str(self._prediction_request.get("pipeline_id", "")).startswith(f"reported-sl4-{self.task}-v1/"))
        if not formal_recipe and self.task == "regression" and model_name in REGRESSION_CV_MIN_N and n_samples < REGRESSION_CV_MIN_N[model_name]:
            return result(empty_metrics, status="skipped", error=f"below minimum N for {model_name}'s internal CV (requires N>={REGRESSION_CV_MIN_N[model_name]})")
        try:
            mode = "passthrough" if self.schema.imputation["model_overrides"].get(model_name) == "passthrough" else "imputed"
            if mode in preparation_errors:
                raise preparation_errors[mode]
            if mode not in prepared:
                preprocess_started = time.perf_counter(); diagnostics["_preprocess_computed"] = True
                try:
                    prepared_cell = preprocess_cell(X_sub_raw, X_test_raw, selected_groups, self.schema.imputation, model_name=model_name)
                except Exception as exc:
                    preparation_errors[mode] = exc; raise
                finally:
                    diagnostics["_preprocess_seconds"] = time.perf_counter() - preprocess_started
                if prepared_cell.K_unobserved != unobserved:
                    mismatch = RuntimeError("preprocessing changed the precomputed K_unobserved count"); preparation_errors[mode] = mismatch; raise mismatch
                prepared[mode] = prepared_cell
            prepared_cell = prepared[mode]
            diagnostics["_preprocess_vectorized"] = bool(prepared_cell.X_train.attrs.get("_preprocess_vectorized", False))
            X_prepared = prepared_cell.X_train; X_test_prepared = prepared_cell.X_test
            diagnostics["K_varying"] = count_varying_sources(
                X_prepared, selected_groups
            )
            diagnostics["underdetermined"] = bool(self.task == "regression" and model_name == "ols" and _ols_is_underdetermined(X_prepared))
            if self.task == "classification" and len(np.unique(y_sub)) < 2:
                return result(empty_metrics, status="skipped", error="single-class training sample for classification")
            if self.task == "classification" and model_name == "super_learner" and int(y_sub.value_counts().min()) < 2:
                return result(empty_metrics, status="skipped", error="below minimum per-class count for super_learner CV")
            if model_name in {"lightgbm", "super_learner"}:
                log_progress(f"cell starting model={model_name} seed={seed} draw={draw} N={n_samples} K={k_features}")
            # The outer transformed matrices above are diagnostics only. CV
            # must receive original missingness, including all-missing sources.
            X_fit = X_sub_raw.copy(deep=True)
            X_test_fit = X_test_raw.copy(deep=True)
            arguments = {"model_name": model_name, "model_seed": _model_seed(seed, draw, n_samples, k_features), "model_n_jobs": self.config.n_jobs if model_name == "super_learner" else 1, "task": self.task, "params": self.selected_model_params[model_name], "X_train": X_fit, "y_train": y_sub, "X_test": X_test_fit, "preprocessor": FoldPreprocessor(tuple(selected_groups), self.schema.imputation, model_name)}
            if self._prediction_request:
                arguments["prediction_request"] = self._prediction_request
                if formal_recipe:
                    arguments["preprocessor"] = FoldPreprocessor(tuple(selected_groups), self.schema.imputation, "super_learner")
            if model_name in SERIAL_OUTER_MODELS and not (self._prediction_request and self._prediction_request.get("persist_fold")):
                fit = _run_native_model_cell_locked(self._runner, fit_arguments=arguments, on_native_crash=lambda attempt, exc: log_progress(f"native subprocess crashed attempt={attempt}/{self.config.native_process_max_attempts} model={model_name} seed={seed} draw={draw} N={n_samples} K={k_features} error={exc}"), on_native_timeout=lambda attempt, exc: log_progress(f"native subprocess timed out attempt={attempt}/{self.config.native_process_max_attempts} model={model_name} seed={seed} draw={draw} N={n_samples} K={k_features} error={exc}"))
            else:
                fit = _fit_predict_model_cell(**arguments)
            if fit.get("status") == "skipped":
                skipped = result(empty_metrics, status="skipped", error=fit["reason"])
                skipped["_prediction_cache_data"] = fit["prediction_cache_data"]
                return skipped
            predictions = np.asarray(fit["predictions"])
            diagnostics["mlp_diagnostics_json"] = fit.get("mlp_diagnostics_json", "")
            diagnostics["_fit_seconds"] = fit["fit_seconds"]; diagnostics["_best_rounds"] = fit["best_rounds"]; diagnostics["converged"] = fit["converged"]; diagnostics["constant_prediction"] = _constant_prediction(predictions)
            metrics = compute_classification_metrics(y_test, predictions, y_sub) if self.task == "classification" else compute_regression_metrics(y_test, predictions, y_sub)
            completed = result(
                metrics, status="ok", error="", peak=int(fit["peak_rss_bytes"])
            )
            if "prediction_cache_data" in fit:
                completed["_prediction_cache_data"] = fit["prediction_cache_data"]
            return completed
        except Exception as exc:
            return result(empty_metrics, status="failed", error=f"{type(exc).__name__}: {exc}")

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._runner.close()
            self._draw_orders.clear()

    def __enter__(self) -> "NKGridExecutionSession":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
