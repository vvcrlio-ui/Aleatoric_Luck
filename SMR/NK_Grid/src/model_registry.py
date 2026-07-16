"""Model constructors for the NLSY replication and extensions."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
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
    ElasticNetCV,
    LassoCV,
    LinearRegression,
    LogisticRegression,
    RidgeCV,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MATPLOTLIB_CACHE = Path(os.environ.get("TMPDIR", "/tmp")) / "aleatoric-matplotlib"
MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE))


MODEL_NAMES = (
    "ols",
    "ridge",
    "lasso",
    "elastic_net",
    "random_forest",
    "xgboost",
    "lightgbm",
    "shallow_neural_network",
    "extra_trees",
    "super_learner",
)

# BART belongs to the original replication rather than the expanded model
# space. Keep it constructible so existing runs and checkpoints remain usable.
LEGACY_MODEL_NAMES = (
    "bart",
)
SUPPORTED_MODEL_NAMES = MODEL_NAMES + LEGACY_MODEL_NAMES

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PARAMS_PATH = ROOT.parent / "NK_Grid" / "model_params.yaml"

MODEL_PARAM_KEYS = {
    "regression": {
        "ols": {"fit_intercept"},
        "ridge": {
            "alpha_log10_min", "alpha_log10_max", "n_alphas",
            "max_cv_folds", "scoring",
        },
        "lasso": {
            "alpha_log10_min", "alpha_log10_max", "n_alphas",
            "max_cv_folds", "max_iter",
        },
        "elastic_net": {
            "alpha_log10_min", "alpha_log10_max", "n_alphas", "l1_ratio",
            "max_cv_folds", "max_iter",
        },
        "random_forest": {"n_estimators", "max_features", "min_samples_leaf"},
        "extra_trees": {"n_estimators", "max_features", "min_samples_leaf"},
        "shallow_neural_network": {
            "hidden_layer_sizes", "activation", "solver", "alpha",
            "learning_rate_init", "max_iter", "early_stopping",
        },
        "super_learner": {
            "cv", "passthrough", "n_estimators", "max_features",
            "min_samples_leaf", "hidden_layer_sizes", "alpha",
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
        "bart": {"n_trees", "n_samples", "n_burn", "thin", "n_chains"},
    },
    "classification": {
        "ols": {"C", "l1_ratio", "solver", "max_iter"},
        "ridge": {"C", "l1_ratio", "solver", "max_iter"},
        "lasso": {"penalty", "C", "l1_ratio", "solver", "max_iter"},
        "elastic_net": {"penalty", "C", "solver", "l1_ratio", "max_iter"},
        "random_forest": {"n_estimators", "max_features", "min_samples_leaf"},
        "extra_trees": {"n_estimators", "max_features", "min_samples_leaf"},
        "shallow_neural_network": {
            "hidden_layer_sizes", "activation", "solver", "alpha",
            "learning_rate_init", "max_iter", "early_stopping",
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
        "bart": set(),
    },
}

_BART_RANDOM_LOCK = threading.Lock()


def _validated_params(
    task: str,
    model_name: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    allowed = MODEL_PARAM_KEYS.get(task, {}).get(model_name)
    if allowed is None:
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
    if model_name == "bart":
        bart_env = {
            "BART_N_TREES": ("n_trees", int),
            "BART_N_SAMPLES": ("n_samples", int),
            "BART_N_BURN": ("n_burn", int),
            "BART_THIN": ("thin", float),
        }
        for key, (parameter_name, converter) in bart_env.items():
            if key in os.environ:
                result[parameter_name] = converter(os.environ[key])
    return result


class XGBoostCVRegressor(BaseEstimator, RegressorMixin):
    """Source-aligned XGBoost: depth 2, eta .3, CV-selected rounds <= 90."""

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
        cv = xgb.cv(
            self.params_,
            dtrain,
            num_boost_round=self.max_rounds,
            nfold=self.cv_folds,
            seed=self.seed,
            shuffle=True,
            verbose_eval=False,
        )
        self.best_rounds_ = int(cv["test-rmse-mean"].idxmin()) + 1
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
        cv = lgb.cv(
            self.params_,
            train,
            num_boost_round=self.max_rounds,
            nfold=self.cv_folds,
            stratified=False,
            seed=self.seed,
            callbacks=[
                lgb.early_stopping(self.early_stopping_rounds, verbose=False)
            ],
        )
        metric_key = next(key for key in cv if key.endswith("rmse-mean"))
        self.best_rounds_ = int(np.argmin(cv[metric_key])) + 1
        self.model_ = lgb.train(
            self.params_, train, num_boost_round=self.best_rounds_
        )
        return self

    def predict(self, X):
        return np.asarray(self.model_.predict(X), dtype=float)


class AdaptiveRidgeCV(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        *,
        alpha_log10_min: float,
        alpha_log10_max: float,
        n_alphas: int,
        max_cv_folds: int,
        scoring: str,
    ):
        self.alpha_log10_min = alpha_log10_min
        self.alpha_log10_max = alpha_log10_max
        self.n_alphas = n_alphas
        self.max_cv_folds = max_cv_folds
        self.scoring = scoring

    def fit(self, X, y):
        cv = min(self.max_cv_folds, len(y))
        if cv < 2:
            raise ValueError("Ridge requires at least two training rows.")
        self.model_ = RidgeCV(
            alphas=np.logspace(
                self.alpha_log10_min, self.alpha_log10_max, self.n_alphas
            ),
            cv=cv,
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


class AdaptiveElasticNetCV(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        seed: int,
        n_jobs: int,
        *,
        alpha_log10_min: float,
        alpha_log10_max: float,
        n_alphas: int,
        l1_ratio: Sequence[float],
        max_cv_folds: int,
        max_iter: int,
    ):
        self.seed = seed
        self.n_jobs = n_jobs
        self.alpha_log10_min = alpha_log10_min
        self.alpha_log10_max = alpha_log10_max
        self.n_alphas = n_alphas
        self.l1_ratio = l1_ratio
        self.max_cv_folds = max_cv_folds
        self.max_iter = max_iter

    def fit(self, X, y):
        cv = min(self.max_cv_folds, len(y))
        if cv < 2:
            raise ValueError("Elastic Net requires at least two training rows.")
        self.model_ = ElasticNetCV(
            alphas=np.logspace(
                self.alpha_log10_min, self.alpha_log10_max, self.n_alphas
            ),
            l1_ratio=self.l1_ratio,
            cv=cv,
            max_iter=self.max_iter,
            n_jobs=self.n_jobs,
            random_state=self.seed,
        ).fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


class BartPyRegressor(BaseEstimator, RegressorMixin):
    """Small sklearn-compatible wrapper around a BartPy implementation."""

    def __init__(
        self,
        n_trees: int = int(os.environ.get("BART_N_TREES", "200")),
        n_samples: int = int(os.environ.get("BART_N_SAMPLES", "1000")),
        n_burn: int = int(os.environ.get("BART_N_BURN", "100")),
        thin: float = float(os.environ.get("BART_THIN", "1.0")),
        n_chains: int = 1,
        n_jobs: int = 1,
        random_state: int | None = None,
    ):
        self.n_trees = n_trees
        self.n_samples = n_samples
        self.n_burn = n_burn
        self.thin = thin
        self.n_chains = n_chains
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y):
        try:
            from bartpy.sklearnmodel import SklearnModel
        except ImportError as exc:
            try:
                from bartpy2.sklearnmodel import SklearnModel
            except ImportError:
                raise ImportError(
                    "BART requires bartpy2 or a compatible bartpy package."
                ) from exc

        self.feature_names_ = list(getattr(X, "columns", [])) or [
            f"x{i}" for i in range(np.asarray(X).shape[1])
        ]
        self.imputer_ = SimpleImputer(strategy="median")
        X_imputed = self.imputer_.fit_transform(X)
        X_frame = pd.DataFrame(X_imputed, columns=self.feature_names_)
        kwargs = {
            "n_trees": self.n_trees,
            "n_samples": self.n_samples,
            "n_burn": self.n_burn,
            "thin": self.thin,
        }
        try:
            self.model_ = SklearnModel(
                **kwargs, n_chains=self.n_chains, n_jobs=self.n_jobs
            )
        except TypeError:
            self.model_ = SklearnModel(**kwargs)
        # BartPy uses NumPy's process-global RNG. Protect and restore it for
        # direct threaded use; experiment scripts additionally use processes.
        with _BART_RANDOM_LOCK:
            previous_state = np.random.get_state()
            try:
                if self.random_state is not None:
                    np.random.seed(self.random_state)
                self.model_.fit(X_frame, np.asarray(y))
            finally:
                np.random.set_state(previous_state)
        return self

    def predict(self, X):
        X_imputed = self.imputer_.transform(X)
        X_frame = pd.DataFrame(X_imputed, columns=self.feature_names_)
        return np.asarray(self.model_.predict(X_frame))


class BartPyClassifier(BaseEstimator):
    """Placeholder for BART classification until a supported backend is available."""

    def __init__(self, random_state: int | None = None, n_jobs: int = 1):
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        raise NotImplementedError("BART classification is not supported by this backend.")


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
        learning_rate_init: float,
        max_iter: int,
        positive: bool,
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
        self.positive = positive
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
                    RidgeCV(alphas=np.logspace(-4, 4, 50)),
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
                        n_jobs=self.n_jobs,
                        random_state=self.seed,
                    ),
                ),
            ),
            (
                "lightgbm",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    lgb.LGBMRegressor(
                        n_estimators=self.lgbm_n_estimators,
                        learning_rate=self.lgbm_learning_rate,
                        num_leaves=self.lgbm_num_leaves,
                        min_data_in_leaf=self.lgbm_min_data_in_leaf,
                        n_jobs=self.n_jobs,
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
                        regressor=MLPRegressor(
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
                    SimpleImputer(strategy="median"),
                    lgb.LGBMClassifier(
                        n_estimators=self.lgbm_n_estimators,
                        learning_rate=self.lgbm_learning_rate,
                        num_leaves=self.lgbm_num_leaves,
                        min_data_in_leaf=self.lgbm_min_data_in_leaf,
                        n_jobs=self.n_jobs,
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
                        n_jobs=self.n_jobs,
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
    if name == "elastic_net":
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
    if name == "bart":
        return BartPyClassifier(
            random_state=seed,
            n_jobs=n_jobs,
            **params,
        )
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
    if name == "elastic_net":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            AdaptiveElasticNetCV(seed=seed, n_jobs=n_jobs, **resolved_params),
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
        neural_params = dict(resolved_params)
        neural_params["hidden_layer_sizes"] = tuple(
            neural_params["hidden_layer_sizes"]
        )
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            TransformedTargetRegressor(
                regressor=MLPRegressor(**neural_params, random_state=seed),
                transformer=StandardScaler(),
            ),
        )
    if name == "super_learner":
        return AdaptiveStackingRegressor(
            seed=seed,
            n_jobs=n_jobs,
            **resolved_params,
        )
    if name == "bart":
        return BartPyRegressor(
            random_state=seed,
            n_jobs=n_jobs,
            **resolved_params,
        )
    raise ValueError(
        f"Unknown model '{model_name}'. Choose from: "
        f"{', '.join(SUPPORTED_MODEL_NAMES)}"
    )
