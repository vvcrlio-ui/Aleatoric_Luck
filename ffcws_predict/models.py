"""Model registry for material hardship prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, RidgeCV
from sklearn.model_selection import KFold, cross_val_score


MODEL_NAMES = (
    "mean_baseline",
    "ols",
    "ridge",
    "lasso",
    "elastic_net",
    "random_forest",
    "xgboost",
    "lightgbm",
)


@dataclass
class FittedModel:
    name: str
    estimator: Any
    params: dict[str, Any]
    cv_used: bool = False

    def predict(self, X):
        return np.asarray(self.estimator.predict(X), dtype=float)


class MeanBaselineRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_, dtype=float)


class RandomForestCVRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, seed: int = 333, n_jobs: int = 1):
        self.seed = seed
        self.n_jobs = n_jobs

    def fit(self, X, y):
        grid = [
            {"min_samples_leaf": 1, "max_features": "sqrt"},
            {"min_samples_leaf": 2, "max_features": "sqrt"},
            {"min_samples_leaf": 5, "max_features": "sqrt"},
            {"min_samples_leaf": 2, "max_features": 0.5},
        ]
        cv = KFold(
            n_splits=min(5, max(2, len(y) // 20)),
            shuffle=True,
            random_state=self.seed,
        )
        best_score = float("inf")
        best_params: dict[str, Any] | None = None
        for params in grid:
            model = RandomForestRegressor(
                n_estimators=200,
                n_jobs=self.n_jobs,
                random_state=self.seed,
                **params,
            )
            score = -cross_val_score(
                model, X, y, scoring="neg_root_mean_squared_error", cv=cv
            ).mean()
            if score < best_score:
                best_score = float(score)
                best_params = params
        assert best_params is not None
        self.best_params_ = best_params
        self.model_ = RandomForestRegressor(
            n_estimators=300,
            n_jobs=self.n_jobs,
            random_state=self.seed,
            **best_params,
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


class XGBoostCVRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, seed: int = 333, n_jobs: int = 1):
        self.seed = seed
        self.n_jobs = n_jobs

    def fit(self, X, y):
        import xgboost as xgb

        dtrain = xgb.DMatrix(X, label=np.asarray(y, dtype=float))
        grid = [
            {"max_depth": 2, "eta": 0.05},
            {"max_depth": 2, "eta": 0.1},
            {"max_depth": 3, "eta": 0.05},
            {"max_depth": 3, "eta": 0.1},
        ]
        best = None
        best_score = float("inf")
        for params in grid:
            run_params = {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "max_depth": params["max_depth"],
                "eta": params["eta"],
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "nthread": self.n_jobs,
                "seed": self.seed,
            }
            cv = xgb.cv(
                run_params,
                dtrain,
                num_boost_round=200,
                nfold=min(5, max(2, len(y) // 20)),
                early_stopping_rounds=10,
                seed=self.seed,
                verbose_eval=False,
            )
            score = float(cv["test-rmse-mean"].min())
            rounds = int(cv["test-rmse-mean"].idxmin()) + 1
            if score < best_score:
                best_score = score
                best = (run_params, rounds)
        assert best is not None
        self.best_params_, self.best_rounds_ = best
        self.model_ = xgb.train(self.best_params_, dtrain, num_boost_round=self.best_rounds_)
        return self

    def predict(self, X):
        import xgboost as xgb

        return self.model_.predict(xgb.DMatrix(X))


class LightGBMCVRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, seed: int = 333, n_jobs: int = 1):
        self.seed = seed
        self.n_jobs = n_jobs

    def fit(self, X, y):
        from lightgbm import LGBMRegressor
        from sklearn.model_selection import KFold, cross_val_score

        grid = [
            {"num_leaves": 7, "learning_rate": 0.05},
            {"num_leaves": 15, "learning_rate": 0.05},
            {"num_leaves": 15, "learning_rate": 0.1},
        ]
        cv = KFold(n_splits=min(5, max(2, len(y) // 20)), shuffle=True, random_state=self.seed)
        best_score = float("inf")
        best_params: dict[str, Any] | None = None
        for params in grid:
            model = LGBMRegressor(
                objective="regression",
                n_estimators=200,
                random_state=self.seed,
                n_jobs=self.n_jobs,
                verbosity=-1,
                **params,
            )
            score = -cross_val_score(
                model, X, y, scoring="neg_root_mean_squared_error", cv=cv
            ).mean()
            if score < best_score:
                best_score = float(score)
                best_params = params
        assert best_params is not None
        self.best_params_ = best_params
        self.model_ = LGBMRegressor(
            objective="regression",
            n_estimators=200,
            random_state=self.seed,
            n_jobs=self.n_jobs,
            verbosity=-1,
            **best_params,
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


def _cv_folds(n_samples: int) -> int:
    return min(5, max(2, n_samples // 3))


def make_estimator(name: str, seed: int, n_jobs: int = 1, n_samples: int = 100):
    cv = _cv_folds(n_samples)
    if name == "mean_baseline":
        return MeanBaselineRegressor(), {"strategy": "train_mean"}, False
    if name == "ols":
        return LinearRegression(), {}, False
    if name == "ridge":
        return RidgeCV(alphas=np.logspace(-4, 4, 30), cv=cv), {"cv": cv}, True
    if name == "lasso":
        return (
            LassoCV(
                alphas=np.logspace(-4, 1, 30),
                cv=cv,
                max_iter=20000,
                n_jobs=n_jobs,
                random_state=seed,
            ),
            {"cv": cv},
            True,
        )
    if name == "elastic_net":
        return (
            ElasticNetCV(
                alphas=np.logspace(-4, 1, 30),
                l1_ratio=[0.1, 0.5, 0.9],
                cv=cv,
                max_iter=20000,
                n_jobs=n_jobs,
                random_state=seed,
            ),
            {"cv": cv, "l1_ratio": [0.1, 0.5, 0.9]},
            True,
        )
    if name == "random_forest":
        return RandomForestCVRegressor(seed=seed, n_jobs=n_jobs), {"internal_cv": True}, True
    if name == "xgboost":
        return XGBoostCVRegressor(seed=seed, n_jobs=n_jobs), {"internal_cv": True}, True
    if name == "lightgbm":
        return LightGBMCVRegressor(seed=seed, n_jobs=n_jobs), {"internal_cv": True}, True
    raise ValueError(f"Unknown model '{name}'. Choose from: {', '.join(MODEL_NAMES)}")


def fit_model(name: str, X, y, seed: int, n_jobs: int = 1) -> FittedModel:
    estimator, params, cv_used = make_estimator(
        name, seed, n_jobs, n_samples=len(y)
    )
    estimator.fit(X, y)
    fitted_params = dict(params)
    for attr in ("best_params_", "best_rounds_", "alpha_", "l1_ratio_"):
        if hasattr(estimator, attr):
            value = getattr(estimator, attr)
            if isinstance(value, np.generic):
                value = value.item()
            fitted_params[attr.rstrip("_")] = value
    return FittedModel(name=name, estimator=estimator, params=fitted_params, cv_used=cv_used)
