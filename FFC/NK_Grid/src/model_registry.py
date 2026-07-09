"""Model constructors for the NLSY replication and extensions."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNetCV,
    LassoCV,
    LinearRegression,
    LogisticRegression,
    RidgeCV,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MATPLOTLIB_CACHE = Path(os.environ.get("TMPDIR", "/tmp")) / "aleatoric-matplotlib"
MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE))


MODEL_NAMES = (
    "xgboost",
    "lightgbm",
    "ols",
    "ridge",
    "lasso",
    "elastic_net",
    "random_forest",
    "bart",
)

_BART_RANDOM_LOCK = threading.Lock()


def _median_imputer() -> SimpleImputer:
    return SimpleImputer(strategy="median", keep_empty_features=True)


class XGBoostCVRegressor(BaseEstimator, RegressorMixin):
    """Source-aligned XGBoost: depth 2, eta .3, CV-selected rounds <= 90."""

    def __init__(self, seed: int = 333, n_jobs: int = 1, max_rounds: int = 90):
        self.seed = seed
        self.n_jobs = n_jobs
        self.max_rounds = max_rounds

    def fit(self, X, y):
        import xgboost as xgb

        dtrain = xgb.DMatrix(X, label=np.asarray(y, dtype=float))
        self.params_ = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "max_depth": 2,
            "eta": 0.3,
            "nthread": self.n_jobs,
            "seed": self.seed,
        }
        cv = xgb.cv(
            self.params_,
            dtrain,
            num_boost_round=self.max_rounds,
            nfold=5,
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

    def __init__(self, seed: int = 333, n_jobs: int = 1, max_rounds: int = 200):
        self.seed = seed
        self.n_jobs = n_jobs
        self.max_rounds = max_rounds

    def fit(self, X, y):
        import lightgbm as lgb

        train = lgb.Dataset(X, label=np.asarray(y, dtype=float))
        self.params_ = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.05,
            "num_leaves": 10,
            "min_data_in_leaf": 20,
            "num_threads": self.n_jobs,
            "seed": self.seed,
            "verbosity": -1,
        }
        cv = lgb.cv(
            self.params_,
            train,
            num_boost_round=self.max_rounds,
            nfold=5,
            stratified=False,
            seed=self.seed,
            callbacks=[lgb.early_stopping(10, verbose=False)],
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
    def fit(self, X, y):
        cv = min(5, len(y))
        if cv < 2:
            raise ValueError("Ridge requires at least two training rows.")
        self.model_ = RidgeCV(
            alphas=np.logspace(-4, 4, 50),
            cv=cv,
            scoring="neg_mean_squared_error",
        ).fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


class AdaptiveLassoCV(BaseEstimator, RegressorMixin):
    def __init__(self, seed: int = 12345, n_jobs: int = 1):
        self.seed = seed
        self.n_jobs = n_jobs

    def fit(self, X, y):
        cv = min(5, len(y))
        if cv < 2:
            raise ValueError("Lasso requires at least two training rows.")
        self.model_ = LassoCV(
            alphas=np.logspace(-4, 1, 50),
            cv=cv,
            max_iter=20000,
            n_jobs=self.n_jobs,
            random_state=self.seed,
        ).fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


class AdaptiveElasticNetCV(BaseEstimator, RegressorMixin):
    def __init__(self, seed: int = 12345, n_jobs: int = 1):
        self.seed = seed
        self.n_jobs = n_jobs

    def fit(self, X, y):
        cv = min(5, len(y))
        if cv < 2:
            raise ValueError("Elastic Net requires at least two training rows.")
        self.model_ = ElasticNetCV(
            alphas=np.logspace(-4, 1, 50),
            l1_ratio=[0.1, 0.5, 0.9],
            cv=cv,
            max_iter=20000,
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
        self.imputer_ = _median_imputer()
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


def _make_classification_model(model_name: str, seed: int, n_jobs: int = 1):
    name = model_name.lower()
    if name == "xgboost":
        import xgboost as xgb

        return xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            max_depth=2,
            learning_rate=0.3,
            n_estimators=int(os.environ.get("XGB_MAX_ROUNDS", "90")),
            n_jobs=n_jobs,
            random_state=seed,
        )
    if name == "lightgbm":
        import lightgbm as lgb

        return lgb.LGBMClassifier(
            objective="binary",
            learning_rate=0.05,
            num_leaves=10,
            min_data_in_leaf=20,
            n_estimators=int(os.environ.get("LGBM_MAX_ROUNDS", "200")),
            n_jobs=n_jobs,
            random_state=seed,
            verbosity=-1,
        )
    if name == "ols":
        return make_pipeline(
            _median_imputer(),
            StandardScaler(),
            LogisticRegression(
                C=1e12,
                l1_ratio=0.0,
                solver="lbfgs",
                max_iter=20000,
                random_state=seed,
            ),
        )
    if name == "ridge":
        return make_pipeline(
            _median_imputer(),
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                l1_ratio=0.0,
                solver="lbfgs",
                max_iter=20000,
                random_state=seed,
            ),
        )
    if name == "lasso":
        return make_pipeline(
            _median_imputer(),
            StandardScaler(),
            LogisticRegression(
                penalty="l1",
                C=1.0,
                l1_ratio=1.0,
                solver="saga",
                max_iter=20000,
                random_state=seed,
            ),
        )
    if name == "elastic_net":
        return make_pipeline(
            _median_imputer(),
            StandardScaler(),
            LogisticRegression(
                penalty="elasticnet",
                C=1.0,
                solver="saga",
                l1_ratio=0.5,
                max_iter=20000,
                random_state=seed,
            ),
        )
    if name == "random_forest":
        return make_pipeline(
            _median_imputer(),
            RandomForestClassifier(
                n_estimators=int(os.environ.get("RF_N_ESTIMATORS", "500")),
                max_features=os.environ.get("RF_MAX_FEATURES", "sqrt"),
                min_samples_leaf=int(os.environ.get("RF_MIN_SAMPLES_LEAF", "1")),
                n_jobs=n_jobs,
                random_state=seed,
            ),
        )
    if name == "bart":
        return BartPyClassifier(random_state=seed, n_jobs=n_jobs)
    raise ValueError(f"Unknown model '{model_name}'. Choose from: {', '.join(MODEL_NAMES)}")


def make_model(model_name: str, seed: int, n_jobs: int = 1, task: str = "regression"):
    """Construct one model using source-aligned or documented extension settings."""

    if task == "classification":
        return _make_classification_model(model_name, seed=seed, n_jobs=n_jobs)
    if task != "regression":
        raise ValueError("task must be 'regression' or 'classification'")

    name = model_name.lower()
    if name == "xgboost":
        return XGBoostCVRegressor(
            seed=seed,
            n_jobs=n_jobs,
            max_rounds=int(os.environ.get("XGB_MAX_ROUNDS", "90")),
        )
    if name == "lightgbm":
        return LightGBMCVRegressor(
            seed=seed,
            n_jobs=n_jobs,
            max_rounds=int(os.environ.get("LGBM_MAX_ROUNDS", "200")),
        )
    if name == "ols":
        return make_pipeline(
            _median_imputer(),
            StandardScaler(),
            LinearRegression(),
        )
    if name == "ridge":
        return make_pipeline(
            _median_imputer(),
            StandardScaler(),
            AdaptiveRidgeCV(),
        )
    if name == "lasso":
        return make_pipeline(
            _median_imputer(),
            StandardScaler(),
            AdaptiveLassoCV(seed=seed, n_jobs=n_jobs),
        )
    if name == "elastic_net":
        return make_pipeline(
            _median_imputer(),
            StandardScaler(),
            AdaptiveElasticNetCV(seed=seed, n_jobs=n_jobs),
        )
    if name == "random_forest":
        return make_pipeline(
            _median_imputer(),
            RandomForestRegressor(
                n_estimators=int(os.environ.get("RF_N_ESTIMATORS", "500")),
                max_features=os.environ.get("RF_MAX_FEATURES", "sqrt"),
                min_samples_leaf=int(os.environ.get("RF_MIN_SAMPLES_LEAF", "1")),
                n_jobs=n_jobs,
                random_state=seed,
            ),
        )
    if name == "bart":
        return BartPyRegressor(random_state=seed, n_jobs=n_jobs)
    raise ValueError(f"Unknown model '{model_name}'. Choose from: {', '.join(MODEL_NAMES)}")
