"""Model constructors for the shared N×K engine."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    LassoCV,
    LinearRegression,
    LogisticRegression,
    RidgeCV,
)
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAMES = (
    "ols",
    "ridge",
    "lasso",
    "random_forest",
    "xgboost",
    "lightgbm",
    "shallow_neural_network",
    "extra_trees",
    "super_learner",
)

# Models that were deliberately retired from the model space. Naming one is a
# hard error rather than a silent drop, so a stale panel fails loudly instead of
# quietly producing results for one model fewer than it asked for.
REMOVED_MODEL_NAMES = {
    "bart": "BART was removed from the model space; see plans/remove-bart.md",
    "elastic_net": "elastic_net was removed from the model space; see plans/remove-elastic-net.md",
}
SUPPORTED_MODEL_NAMES = MODEL_NAMES


def reject_removed_model(name: str) -> None:
    """Raise a self-explaining error for a model that no longer exists."""

    reason = REMOVED_MODEL_NAMES.get(str(name).lower())
    if reason is not None:
        raise ValueError(reason)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PARAMS_PATH = ROOT / "model_params.yaml"

MODEL_PARAM_KEYS = {
    "regression": {
        "ols": {"fit_intercept"},
        "ridge": {
            "alpha_log10_min", "alpha_log10_max", "n_alphas", "scoring",
        },
        "lasso": {
            "alpha_log10_min", "alpha_log10_max", "n_alphas",
            "max_cv_folds", "max_iter",
        },
        "random_forest": {"n_estimators", "max_features", "min_samples_leaf"},
        "extra_trees": {"n_estimators", "max_features", "min_samples_leaf"},
        "shallow_neural_network": {
            "hidden_layer_sizes", "activation", "solver",
            "alpha_log10_min", "alpha_log10_max", "n_alphas", "max_cv_folds",
            "learning_rate_init", "max_iter", "early_stopping",
            "validation_fraction", "n_iter_no_change",
        },
        "super_learner": {
            "cv", "passthrough", "n_estimators", "max_features",
            "min_samples_leaf", "hidden_layer_sizes", "alpha",
            "ridge_alpha_log10_min", "ridge_alpha_log10_max",
            "ridge_n_alphas", "ridge_scoring",
            "learning_rate_init", "max_iter", "positive",
            "lgbm_n_estimators", "lgbm_learning_rate", "lgbm_num_leaves",
            "lgbm_min_data_in_leaf",
        },
        "xgboost": {
            "objective", "eval_metric", "max_depth", "eta", "max_rounds",
            "cv_folds",
        },
        "lightgbm": {
            "objective", "metric", "learning_rate", "num_leaves",
            "min_data_in_leaf", "verbosity", "max_rounds", "cv_folds",
            "early_stopping_rounds",
        },
    },
    "classification": {
        "ols": {"C", "l1_ratio", "solver", "max_iter"},
        "ridge": {"C", "l1_ratio", "solver", "max_iter"},
        "lasso": {"penalty", "C", "l1_ratio", "solver", "max_iter"},
        "random_forest": {"n_estimators", "max_features", "min_samples_leaf"},
        "extra_trees": {"n_estimators", "max_features", "min_samples_leaf"},
        "shallow_neural_network": {
            "hidden_layer_sizes", "activation", "solver", "alpha",
            "learning_rate_init", "max_iter", "early_stopping",
            "validation_fraction", "n_iter_no_change",
        },
        "super_learner": {
            "cv", "passthrough", "n_estimators", "max_features",
            "min_samples_leaf", "hidden_layer_sizes", "alpha",
            "learning_rate_init", "max_iter", "C",
            "lgbm_n_estimators", "lgbm_learning_rate", "lgbm_num_leaves",
            "lgbm_min_data_in_leaf",
        },
        "xgboost": {
            "objective", "eval_metric", "max_depth", "learning_rate",
            "n_estimators",
        },
        "lightgbm": {
            "objective", "learning_rate", "num_leaves", "min_data_in_leaf",
            "n_estimators", "verbosity",
        },
    },
}


def _validated_params(
    task: str,
    model_name: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    reject_removed_model(model_name)
    allowed = MODEL_PARAM_KEYS.get(task, {}).get(model_name)
    if task == "regression" and model_name == "super_learner":
        allowed = allowed | {"mlp_batch_size", "diagnostics"}
        params = {"mlp_batch_size": "auto", "diagnostics": False, **params}
        batch = params["mlp_batch_size"]
        if not ((isinstance(batch, str) and batch in {"auto", "full"}) or
                (type(batch) is int and batch > 0)):
            raise ValueError("mlp_batch_size must be auto, full, or a positive integer")
        if type(params["diagnostics"]) is not bool:
            raise ValueError("diagnostics must be boolean")
    if allowed is None:
        reject_removed_model(model_name)
        raise ValueError(
            f"Unknown {task} model '{model_name}'. Choose from: "
            f"{', '.join(SUPPORTED_MODEL_NAMES)}"
        )
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise ValueError(
            f"Invalid parameters for {task} model '{model_name}': "
            f"{', '.join(unknown)}"
        )
    if task == "regression" and model_name in {
        "xgboost", "lightgbm", "super_learner",
    }:
        missing = sorted(allowed - set(params))
        if missing:
            raise ValueError(
                f"Missing required parameters for {task} model '{model_name}': "
                f"{', '.join(missing)}"
            )
    if task == "regression" and model_name in {"xgboost", "lightgbm"}:
        metric_field = "eval_metric" if model_name == "xgboost" else "metric"
        if params[metric_field] != "rmse":
            raise ValueError(
                f"{task} model '{model_name}' requires {metric_field}='rmse'; "
                f"got {params[metric_field]!r}"
            )
    return dict(params)


def load_model_params(
    path: Path,
    *,
    task: str,
    models: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Load task-specific parameters for exactly the selected models."""

    params_path = Path(path)
    if not params_path.exists():
        raise FileNotFoundError(f"Model parameter YAML not found: {params_path}")
    try:
        with params_path.open(encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid model parameter YAML {params_path}: {exc}") from exc

    if not isinstance(document, dict):
        raise ValueError(f"Model parameter YAML must contain a mapping: {params_path}")
    task_params = document.get(task)
    if not isinstance(task_params, dict):
        raise ValueError(
            f"Model parameter YAML is missing a '{task}' mapping: {params_path}"
        )

    normalized_models = [str(model).lower() for model in models]
    # Reject retired models before the generic "missing from YAML" path, so a
    # stale panel is told why the model is gone instead of that it is absent.
    for model_name in normalized_models:
        reject_removed_model(model_name)
    missing = sorted(set(normalized_models) - set(task_params))
    if missing:
        raise ValueError(
            f"Model parameter YAML {task} section is missing selected model(s): "
            f"{', '.join(missing)}"
        )

    selected: dict[str, dict[str, Any]] = {}
    for model_name in normalized_models:
        model_params = task_params[model_name]
        if not isinstance(model_params, dict):
            raise ValueError(
                f"Parameters for {task} model '{model_name}' must be a mapping."
            )
        selected[model_name] = _validated_params(task, model_name, model_params)
    return selected


def load_algorithm_version(path: Path) -> str:
    """Load the manually maintained methodological version from the YAML root."""

    params_path = Path(path)
    try:
        with params_path.open(encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid model parameter YAML {params_path}: {exc}") from exc
    version = document.get("algorithm_version") if isinstance(document, dict) else None
    if not isinstance(version, str) or not version.strip():
        raise ValueError(
            f"Model parameter YAML requires a non-empty algorithm_version: {params_path}"
        )
    return version.strip()


def resolved_model_params(
    params: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Apply documented environment overrides for manifest recording."""

    return {
        model_name: _apply_environment_overrides(model_name, model_params)
        for model_name, model_params in params.items()
    }


def _apply_environment_overrides(
    model_name: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Preserve the established cluster environment overrides."""

    result = dict(params)
    env_key = None
    parameter = None
    if model_name == "xgboost":
        env_key = "XGB_MAX_ROUNDS"
        parameter = "max_rounds" if "max_rounds" in result else "n_estimators"
    elif model_name == "lightgbm":
        env_key = "LGBM_MAX_ROUNDS"
        parameter = "max_rounds" if "max_rounds" in result else "n_estimators"
    if env_key is not None and parameter is not None and env_key in os.environ:
        result[parameter] = int(os.environ[env_key])

    if model_name == "random_forest":
        if "RF_N_ESTIMATORS" in os.environ:
            result["n_estimators"] = int(os.environ["RF_N_ESTIMATORS"])
        if "RF_MAX_FEATURES" in os.environ:
            result["max_features"] = os.environ["RF_MAX_FEATURES"]
        if "RF_MIN_SAMPLES_LEAF" in os.environ:
            result["min_samples_leaf"] = int(os.environ["RF_MIN_SAMPLES_LEAF"])
    return result


def _select_cv_round(curve, patience=None):
    """First strict minimum, optionally replaying aggregate early stopping."""
    values = np.asarray(curve, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("CV requires a complete finite metric curve")
    best = 0
    for index in range(1, len(values)):
        if values[index] < values[best]:
            best = index
        if patience is not None and patience > 0 and index - best >= patience:
            break
    return best + 1


class XGBoostCVRegressor(BaseEstimator, RegressorMixin):
    """Source-aligned XGBoost: depth 2, eta .3, CV-selected rounds <= 90.

    CV folds are generated from a private ``RandomState`` and passed explicitly
    to ``xgboost.cv``.  This avoids its module-global NumPy RNG, which is shared
    by concurrent cell workers even when both ``seed`` and ``nthread`` are fixed.
    """

    def __init__(
        self,
        seed: int,
        n_jobs: int,
        *,
        objective: str,
        eval_metric: str,
        max_depth: int,
        eta: float,
        max_rounds: int,
        cv_folds: int,
    ):
        self.seed = seed
        self.n_jobs = n_jobs
        self.objective = objective
        self.eval_metric = eval_metric
        self.max_depth = max_depth
        self.eta = eta
        self.max_rounds = max_rounds
        self.cv_folds = cv_folds

    def fit(self, X, y):
        import xgboost as xgb

        dtrain = xgb.DMatrix(X, label=np.asarray(y, dtype=float))
        self.params_ = {
            "objective": self.objective,
            "eval_metric": self.eval_metric,
            "max_depth": self.max_depth,
            "eta": self.eta,
            "nthread": self.n_jobs,
            "seed": self.seed,
        }
        # ``xgboost.cv`` otherwise calls ``np.random.seed(seed)`` then uses the
        # module-global generator to shuffle folds.  Concurrent callers race on
        # that shared generator, so give it the same legacy MT19937 permutation
        # from an instance-local generator instead.
        fold_order = np.random.RandomState(self.seed).permutation(len(y))
        test_folds = np.array_split(fold_order, self.cv_folds)
        folds = [
            (
                np.concatenate(
                    [test_folds[index] for index in range(self.cv_folds) if index != fold]
                ),
                test_folds[fold],
            )
            for fold in range(self.cv_folds)
        ]
        if len(y) < self.cv_folds:
            raise ValueError("XGBoost requires at least cv_folds training rows")
        curves = []
        for train_index, valid_index in folds:
            fold_train = dtrain.slice(train_index)
            fold_valid = dtrain.slice(valid_index)
            booster = None
            try:
                history = {}
                booster = xgb.train(
                    self.params_, fold_train, num_boost_round=self.max_rounds,
                    evals=[(fold_valid, "valid")], evals_result=history, verbose_eval=False,
                )
                curves.append(history["valid"]["rmse"])
            finally:
                del booster, fold_train, fold_valid
        self.cv_curve_ = np.mean(np.asarray(curves, dtype=float), axis=0)
        self.best_rounds_ = _select_cv_round(self.cv_curve_)
        self.model_ = xgb.train(
            self.params_, dtrain, num_boost_round=self.best_rounds_
        )
        return self

    def predict(self, X):
        import xgboost as xgb

        return self.model_.predict(xgb.DMatrix(X))


class LightGBMCVRegressor(BaseEstimator, RegressorMixin):
    """LightGBM extension with CV-selected boosting rounds."""

    def __init__(
        self,
        seed: int,
        n_jobs: int,
        *,
        objective: str,
        metric: str,
        learning_rate: float,
        num_leaves: int,
        min_data_in_leaf: int,
        verbosity: int,
        max_rounds: int,
        cv_folds: int,
        early_stopping_rounds: int,
    ):
        self.seed = seed
        self.n_jobs = n_jobs
        self.objective = objective
        self.metric = metric
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_data_in_leaf = min_data_in_leaf
        self.verbosity = verbosity
        self.max_rounds = max_rounds
        self.cv_folds = cv_folds
        self.early_stopping_rounds = early_stopping_rounds

    def fit(self, X, y):
        import lightgbm as lgb

        train = lgb.Dataset(X, label=np.asarray(y, dtype=float))
        self.params_ = {
            "objective": self.objective,
            "metric": self.metric,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "num_threads": self.n_jobs,
            "seed": self.seed,
            "verbosity": self.verbosity,
        }
        if len(y) < self.cv_folds:
            raise ValueError("LightGBM requires at least cv_folds training rows")
        # Preserve 4.6.0 _make_n_folds, including its remainder behavior and
        # full-data bin construction. Fold-local bins belong to M4, not M2.
        train._update_params(self.params_).construct()
        order = np.random.RandomState(self.seed).permutation(len(y))
        step = len(y) // self.cv_folds
        valid_indices = [order[i:i + step] for i in range(0, len(y), step)]
        curves = []
        for fold in range(self.cv_folds):
            train_index = np.concatenate([valid_indices[i] for i in range(self.cv_folds) if i != fold])
            fold_train = train.subset(sorted(train_index))
            fold_valid = train.subset(sorted(valid_indices[fold]))
            booster = None
            try:
                history = {}
                booster = lgb.train(
                    self.params_, fold_train, num_boost_round=self.max_rounds,
                    valid_sets=[fold_valid], valid_names=["valid"],
                    callbacks=[lgb.record_evaluation(history)],
                )
                curves.append(history["valid"]["rmse"])
            finally:
                del booster, fold_train, fold_valid
        self.cv_curve_ = np.mean(np.asarray(curves, dtype=float), axis=0)
        self.best_rounds_ = _select_cv_round(self.cv_curve_, self.early_stopping_rounds)
        self.model_ = lgb.train(
            self.params_, train, num_boost_round=self.best_rounds_
        )
        return self

    def predict(self, X):
        return np.asarray(self.model_.predict(X), dtype=float)


class AdaptiveRidgeCV(BaseEstimator, RegressorMixin):
    """Select ridge alpha with exact leave-one-out CV.

    Leaving ``RidgeCV.cv`` unset activates its analytic leave-one-out path,
    which reuses one matrix decomposition across all rows and alpha values
    (about 15x faster than the former 5-fold grid search).  Lasso and MLP
    cannot use this shortcut: variable selection and nonlinear fitting mean
    neither is a linear smoother with a ridge-style hat-matrix identity.
    """

    def __init__(
        self,
        *,
        alpha_log10_min: float,
        alpha_log10_max: float,
        n_alphas: int,
        scoring: str,
    ):
        self.alpha_log10_min = alpha_log10_min
        self.alpha_log10_max = alpha_log10_max
        self.n_alphas = n_alphas
        self.scoring = scoring

    def fit(self, X, y):
        if len(y) < 2:
            raise ValueError("Ridge requires at least two training rows.")
        self.model_ = RidgeCV(
            alphas=np.logspace(
                self.alpha_log10_min, self.alpha_log10_max, self.n_alphas
            ),
            scoring=self.scoring,
        ).fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


class AdaptiveLassoCV(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        seed: int,
        n_jobs: int,
        *,
        alpha_log10_min: float,
        alpha_log10_max: float,
        n_alphas: int,
        max_cv_folds: int,
        max_iter: int,
    ):
        self.seed = seed
        self.n_jobs = n_jobs
        self.alpha_log10_min = alpha_log10_min
        self.alpha_log10_max = alpha_log10_max
        self.n_alphas = n_alphas
        self.max_cv_folds = max_cv_folds
        self.max_iter = max_iter

    def fit(self, X, y):
        cv = min(self.max_cv_folds, len(y))
        if cv < 2:
            raise ValueError("Lasso requires at least two training rows.")
        self.model_ = LassoCV(
            alphas=np.logspace(
                self.alpha_log10_min, self.alpha_log10_max, self.n_alphas
            ),
            cv=cv,
            max_iter=self.max_iter,
            n_jobs=self.n_jobs,
            random_state=self.seed,
        ).fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


class AdaptiveMLPRegressor(BaseEstimator, RegressorMixin):
    """MLP whose L2 penalty is chosen per fit by internal K-fold CV.

    Mirrors AdaptiveLassoCV/RidgeCV: the alpha grid is the locked contract;
    the value a cell actually uses is selected inside that cell's training
    rows only, so no test-split information leaks into the choice. A fixed
    alpha cannot suit both (N=10, K=100) and (N=4242, K=1) corners of the
    grid, which is why the other penalized models already CV their strength.
    """

    def __init__(
        self,
        seed: int,
        *,
        hidden_layer_sizes: Sequence[int],
        activation: str,
        solver: str,
        learning_rate_init: float,
        max_iter: int,
        early_stopping: bool,
        alpha_log10_min: float,
        alpha_log10_max: float,
        n_alphas: int,
        max_cv_folds: int,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 10,
    ):
        self.seed = seed
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.alpha_log10_min = alpha_log10_min
        self.alpha_log10_max = alpha_log10_max
        self.n_alphas = n_alphas
        self.max_cv_folds = max_cv_folds
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change

    def _mlp(self, alpha: float) -> MLPRegressor:
        return MLPRegressor(
            hidden_layer_sizes=tuple(self.hidden_layer_sizes),
            activation=self.activation,
            solver=self.solver,
            alpha=alpha,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            random_state=self.seed,
        )

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if len(y) < 2:
            raise ValueError(
                "shallow_neural_network requires at least two training rows."
            )
        alphas = np.logspace(
            self.alpha_log10_min, self.alpha_log10_max, self.n_alphas
        )
        cv = min(self.max_cv_folds, len(y))
        # shuffle=False keeps fold membership a pure function of row order,
        # so a cell's result stays reproducible bit-for-bit.
        folds = tuple(KFold(n_splits=cv, shuffle=False).split(X))
        mean_mse = []
        for alpha in alphas:
            fold_mse = []
            for train_rows, val_rows in folds:
                model = self._mlp(alpha).fit(X[train_rows], y[train_rows])
                residual = model.predict(X[val_rows]) - y[val_rows]
                fold_mse.append(float(np.mean(residual**2)))
            mean_mse.append(float(np.mean(fold_mse)))
        # argmin ties resolve to the smallest alpha; deterministic either way.
        self.alpha_ = float(alphas[int(np.argmin(mean_mse))])
        self.cv_mse_ = tuple(mean_mse)
        self.model_ = self._mlp(self.alpha_).fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X, dtype=float))


class FitBatchMLPRegressor(MLPRegressor):
    """Resolve full at each actual fit; retain constructor policy for clone."""

    def fit(self, X, y, sample_weight=None):
        policy = self.batch_size
        self.fit_n_ = len(y)
        self.effective_batch_size_ = min(200, len(y)) if policy == "auto" else (
            len(y) if policy == "full" else min(policy, len(y)))
        self.batch_size = len(y) if policy == "full" else policy
        try:
            return super().fit(X, y, sample_weight=sample_weight)
        finally:
            self.batch_size = policy


class AdaptiveStackingRegressor(BaseEstimator, RegressorMixin):
    """Compact Super Learner with out-of-fold base-model predictions."""

    def __init__(
        self,
        seed: int,
        n_jobs: int,
        *,
        cv: int,
        passthrough: bool,
        n_estimators: int,
        max_features: str | float | int | None,
        min_samples_leaf: int,
        hidden_layer_sizes: Sequence[int],
        alpha: float,
        ridge_alpha_log10_min: float,
        ridge_alpha_log10_max: float,
        ridge_n_alphas: int,
        ridge_scoring: str,
        learning_rate_init: float,
        max_iter: int,
        positive: bool,
        lgbm_n_estimators: int,
        lgbm_learning_rate: float,
        lgbm_num_leaves: int,
        lgbm_min_data_in_leaf: int,
        mlp_batch_size: str | int = "auto",
        diagnostics: bool = False,
    ):
        self.seed = seed
        self.n_jobs = n_jobs
        self.cv = cv
        self.passthrough = passthrough
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.ridge_alpha_log10_min = ridge_alpha_log10_min
        self.ridge_alpha_log10_max = ridge_alpha_log10_max
        self.ridge_n_alphas = ridge_n_alphas
        self.ridge_scoring = ridge_scoring
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.positive = positive
        self.lgbm_n_estimators = lgbm_n_estimators
        self.lgbm_learning_rate = lgbm_learning_rate
        self.lgbm_num_leaves = lgbm_num_leaves
        self.lgbm_min_data_in_leaf = lgbm_min_data_in_leaf
        self.mlp_batch_size = mlp_batch_size
        self.diagnostics = diagnostics

    def fit(self, X, y):
        if self.passthrough and np.asarray(pd.isna(X)).any():
            raise ValueError(
                "Super Learner passthrough=True does not support NaN values in X; "
                "impute X before fitting or set passthrough=False."
            )
        cv = min(self.cv, len(y))
        if cv < 2:
            raise ValueError("Super Learner requires at least two training rows.")
        import lightgbm as lgb

        estimators = [
            (
                "ridge",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                    RidgeCV(
                        alphas=np.logspace(
                            self.ridge_alpha_log10_min,
                            self.ridge_alpha_log10_max,
                            self.ridge_n_alphas,
                        ),
                        scoring=self.ridge_scoring,
                    ),
                ),
            ),
            (
                "extra_trees",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    ExtraTreesRegressor(
                        n_estimators=self.n_estimators,
                        max_features=self.max_features,
                        min_samples_leaf=self.min_samples_leaf,
                        n_jobs=1,
                        random_state=self.seed,
                    ),
                ),
            ),
            (
                "lightgbm",
                make_pipeline(
                    SimpleImputer(strategy="median").set_output(
                        transform="pandas"
                    ),
                    lgb.LGBMRegressor(
                        n_estimators=self.lgbm_n_estimators,
                        learning_rate=self.lgbm_learning_rate,
                        num_leaves=self.lgbm_num_leaves,
                        min_data_in_leaf=self.lgbm_min_data_in_leaf,
                        n_jobs=1,
                        random_state=self.seed,
                        verbosity=-1,
                    ),
                ),
            ),
            (
                "shallow_nn",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                    TransformedTargetRegressor(
                        regressor=FitBatchMLPRegressor(
                            batch_size=self.mlp_batch_size,
                            hidden_layer_sizes=tuple(self.hidden_layer_sizes),
                            alpha=self.alpha,
                            learning_rate_init=self.learning_rate_init,
                            max_iter=self.max_iter,
                            random_state=self.seed,
                        ),
                        transformer=StandardScaler(),
                    ),
                ),
            ),
        ]
        self.model_ = StackingRegressor(
            estimators=estimators,
            final_estimator=LinearRegression(positive=self.positive),
            cv=cv,
            passthrough=self.passthrough,
            n_jobs=self.n_jobs,
        ).fit(X, y)
        if self.diagnostics:
            # Explicit diagnostic runs replay deterministic folds to inspect
            # fitted estimators; this extra work is never enabled by default.
            records = []
            target = np.asarray(y)
            for fold, (train_rows, valid_rows) in enumerate(KFold(cv).split(X)):
                train_X = X.iloc[train_rows] if hasattr(X, "iloc") else X[train_rows]
                valid_X = X.iloc[valid_rows] if hasattr(X, "iloc") else X[valid_rows]
                for name, estimator in estimators:
                    fitted = clone(estimator).fit(train_X, target[train_rows])
                    record = {"phase": "oof", "fold": fold, "model": name, "N": len(train_rows),
                              "mse": float(np.mean((fitted.predict(valid_X) - target[valid_rows]) ** 2))}
                    if name == "shallow_nn":
                        mlp = fitted.steps[-1][1].regressor_
                        record.update(batch=mlp.effective_batch_size_, iterations=mlp.n_iter_,
                                      reached_max_iter=mlp.n_iter_ >= mlp.max_iter)
                    records.append(record)
            mlp = self.model_.named_estimators_["shallow_nn"].steps[-1][1].regressor_
            records.append({"phase": "full", "model": "shallow_nn", "N": mlp.fit_n_,
                            "batch": mlp.effective_batch_size_, "iterations": mlp.n_iter_,
                            "reached_max_iter": mlp.n_iter_ >= mlp.max_iter})
            self.diagnostics_ = {"fits": records, "coefficients": self.model_.final_estimator_.coef_.tolist(),
                                 "intercept": float(self.model_.final_estimator_.intercept_),
                                 "note": "diagnostic fold replay; epochs do not imply equal optimizer steps"}
        return self

    def predict(self, X):
        return self.model_.predict(X)


class AdaptiveStackingClassifier(BaseEstimator, ClassifierMixin):
    """Classification counterpart of the out-of-fold Super Learner."""

    def __init__(
        self,
        seed: int,
        n_jobs: int,
        *,
        cv: int,
        passthrough: bool,
        n_estimators: int,
        max_features: str | float | int | None,
        min_samples_leaf: int,
        hidden_layer_sizes: Sequence[int],
        alpha: float,
        learning_rate_init: float,
        max_iter: int,
        C: float,
        lgbm_n_estimators: int,
        lgbm_learning_rate: float,
        lgbm_num_leaves: int,
        lgbm_min_data_in_leaf: int,
    ):
        self.seed = seed
        self.n_jobs = n_jobs
        self.cv = cv
        self.passthrough = passthrough
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.C = C
        self.lgbm_n_estimators = lgbm_n_estimators
        self.lgbm_learning_rate = lgbm_learning_rate
        self.lgbm_num_leaves = lgbm_num_leaves
        self.lgbm_min_data_in_leaf = lgbm_min_data_in_leaf

    def fit(self, X, y):
        if self.passthrough and np.asarray(pd.isna(X)).any():
            raise ValueError(
                "Super Learner passthrough=True does not support NaN values in X; "
                "impute X before fitting or set passthrough=False."
            )
        _, counts = np.unique(np.asarray(y), return_counts=True)
        cv = min(self.cv, int(counts.min())) if len(counts) >= 2 else 0
        if cv < 2:
            raise ValueError(
                "Super Learner classification requires at least two rows per class."
            )
        import lightgbm as lgb

        estimators = [
            (
                "logistic",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                    LogisticRegression(max_iter=self.max_iter, random_state=self.seed),
                ),
            ),
            (
                "lightgbm",
                make_pipeline(
                    SimpleImputer(strategy="median").set_output(
                        transform="pandas"
                    ),
                    lgb.LGBMClassifier(
                        n_estimators=self.lgbm_n_estimators,
                        learning_rate=self.lgbm_learning_rate,
                        num_leaves=self.lgbm_num_leaves,
                        min_data_in_leaf=self.lgbm_min_data_in_leaf,
                        n_jobs=1,
                        random_state=self.seed,
                        verbosity=-1,
                    ),
                ),
            ),
            (
                "extra_trees",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    ExtraTreesClassifier(
                        n_estimators=self.n_estimators,
                        max_features=self.max_features,
                        min_samples_leaf=self.min_samples_leaf,
                        n_jobs=1,
                        random_state=self.seed,
                    ),
                ),
            ),
            (
                "shallow_nn",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                    MLPClassifier(
                        hidden_layer_sizes=tuple(self.hidden_layer_sizes),
                        alpha=self.alpha,
                        learning_rate_init=self.learning_rate_init,
                        max_iter=self.max_iter,
                        random_state=self.seed,
                    ),
                ),
            ),
        ]
        self.model_ = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(
                C=self.C, max_iter=self.max_iter, random_state=self.seed
            ),
            cv=cv,
            stack_method="predict_proba",
            passthrough=self.passthrough,
            n_jobs=self.n_jobs,
        ).fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def _make_classification_model(
    model_name: str,
    seed: int,
    n_jobs: int,
    params: Mapping[str, Any],
):
    name = model_name.lower()
    if name == "xgboost":
        import xgboost as xgb

        return xgb.XGBClassifier(
            **params,
            n_jobs=n_jobs,
            random_state=seed,
        )
    if name == "lightgbm":
        import lightgbm as lgb

        return lgb.LGBMClassifier(
            **params,
            n_jobs=n_jobs,
            random_state=seed,
        )
    if name == "ols":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                **params,
                random_state=seed,
            ),
        )
    if name == "ridge":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                **params,
                random_state=seed,
            ),
        )
    if name == "lasso":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                **params,
                random_state=seed,
            ),
        )
    if name == "random_forest":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                **params,
                n_jobs=n_jobs,
                random_state=seed,
            ),
        )
    if name == "extra_trees":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                **params,
                n_jobs=n_jobs,
                random_state=seed,
            ),
        )
    if name == "shallow_neural_network":
        neural_params = dict(params)
        neural_params["hidden_layer_sizes"] = tuple(
            neural_params["hidden_layer_sizes"]
        )
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            MLPClassifier(**neural_params, random_state=seed),
        )
    if name == "super_learner":
        return AdaptiveStackingClassifier(
            seed=seed,
            n_jobs=n_jobs,
            **params,
        )
    reject_removed_model(name)
    raise ValueError(
        f"Unknown model '{model_name}'. Choose from: "
        f"{', '.join(SUPPORTED_MODEL_NAMES)}"
    )


def make_model(
    model_name: str,
    seed: int,
    n_jobs: int = 1,
    task: str = "regression",
    params: Mapping[str, Any] | None = None,
):
    """Construct one model using source-aligned or documented extension settings."""

    if task not in {"regression", "classification"}:
        raise ValueError("task must be 'regression' or 'classification'")

    name = model_name.lower()
    reject_removed_model(name)
    if params is None:
        params = load_model_params(
            DEFAULT_MODEL_PARAMS_PATH,
            task=task,
            models=[name],
        )[name]
    resolved_params = _validated_params(task, name, params)
    resolved_params = _apply_environment_overrides(name, resolved_params)

    if task == "classification":
        return _make_classification_model(
            name,
            seed=seed,
            n_jobs=n_jobs,
            params=resolved_params,
        )

    if name == "xgboost":
        return XGBoostCVRegressor(
            seed=seed,
            n_jobs=n_jobs,
            **resolved_params,
        )
    if name == "lightgbm":
        return LightGBMCVRegressor(
            seed=seed,
            n_jobs=n_jobs,
            **resolved_params,
        )
    if name == "ols":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LinearRegression(**resolved_params),
        )
    if name == "ridge":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            AdaptiveRidgeCV(**resolved_params),
        )
    if name == "lasso":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            AdaptiveLassoCV(seed=seed, n_jobs=n_jobs, **resolved_params),
        )
    if name == "random_forest":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                **resolved_params,
                n_jobs=n_jobs,
                random_state=seed,
            ),
        )
    if name == "extra_trees":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesRegressor(
                **resolved_params,
                n_jobs=n_jobs,
                random_state=seed,
            ),
        )
    if name == "shallow_neural_network":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            TransformedTargetRegressor(
                regressor=AdaptiveMLPRegressor(seed=seed, **resolved_params),
                transformer=StandardScaler(),
            ),
        )
    if name == "super_learner":
        return AdaptiveStackingRegressor(
            seed=seed,
            n_jobs=n_jobs,
            **resolved_params,
        )
    reject_removed_model(name)
    raise ValueError(
        f"Unknown model '{model_name}'. Choose from: "
        f"{', '.join(SUPPORTED_MODEL_NAMES)}"
    )
