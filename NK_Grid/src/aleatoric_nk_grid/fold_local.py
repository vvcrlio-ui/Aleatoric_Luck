"""Versioned complete-pipeline regression CV; legacy models remain M2 oracles."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class FoldLocalRidge(RegressorMixin, BaseEstimator):
    """Explicit complete-pipeline LOO; no N-dependent fold substitution."""
    def __init__(self, preprocessor, alpha_log10_min, alpha_log10_max, n_alphas, scoring):
        self.preprocessor = preprocessor
        self.alpha_log10_min = alpha_log10_min
        self.alpha_log10_max = alpha_log10_max
        self.n_alphas = n_alphas
        self.scoring = scoring

    def fit(self, X, y):
        if self.scoring != 'neg_mean_squared_error':
            raise ValueError('fold-local Ridge currently requires the declared neg_mean_squared_error scoring')
        y = np.asarray(y)
        self.alphas_ = np.logspace(self.alpha_log10_min, self.alpha_log10_max, self.n_alphas)
        self.cv_predictions_ = np.empty((len(y), len(self.alphas_)))
        for train, valid in LeaveOneOut().split(X):
            process = make_pipeline(clone(self.preprocessor), StandardScaler()).fit(X.iloc[train])
            train_X, valid_X = process.transform(X.iloc[train]), process.transform(X.iloc[valid])
            # One fold-specific SVD serves the unchanged alpha grid. This is
            # complete-pipeline LOO, not full-N analytic RidgeCV.
            center = train_X.mean(axis=0)
            target_mean = y[train].mean()
            u, singular, vt = np.linalg.svd(train_X - center, full_matrices=False)
            keep = singular > 1e-15  # sklearn Ridge's SVD rank cutoff
            singular = singular[keep]
            projection = (valid_X - center) @ vt[keep].T
            target = u[:, keep].T @ (y[train] - target_mean)
            factors = singular[:, None] / (singular[:, None] ** 2 + self.alphas_[None, :])
            self.cv_predictions_[valid, :] = (projection * target) @ factors + target_mean
        self.cv_mse_ = np.mean((self.cv_predictions_ - y[:, None]) ** 2, axis=0)
        self.alpha_ = float(self.alphas_[np.argmin(self.cv_mse_)])
        self.model_ = make_pipeline(clone(self.preprocessor), StandardScaler(), Ridge(alpha=self.alpha_)).fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


class FoldLocalLasso(RegressorMixin, BaseEstimator):
    def __init__(self, preprocessor, seed, n_jobs, alpha_log10_min, alpha_log10_max, n_alphas, max_cv_folds, max_iter):
        self.preprocessor = preprocessor
        self.seed = seed
        self.n_jobs = n_jobs
        self.alpha_log10_min = alpha_log10_min
        self.alpha_log10_max = alpha_log10_max
        self.n_alphas = n_alphas
        self.max_cv_folds = max_cv_folds
        self.max_iter = max_iter

    def fit(self, X, y):
        y = np.asarray(y)
        # LassoCV sorts alphas descending, including its first-minimum tie rule.
        self.alphas_ = np.logspace(self.alpha_log10_min, self.alpha_log10_max, self.n_alphas)[::-1]
        losses = []
        for train, valid in KFold(min(self.max_cv_folds, len(y))).split(X):
            process = make_pipeline(clone(self.preprocessor), StandardScaler()).fit(X.iloc[train])
            train_X, valid_X = process.transform(X.iloc[train]), process.transform(X.iloc[valid])
            losses.append([float(np.mean((Lasso(alpha=alpha, max_iter=self.max_iter, random_state=self.seed)
                .fit(train_X, y[train]).predict(valid_X) - y[valid]) ** 2)) for alpha in self.alphas_])
        self.cv_mse_ = np.mean(losses, axis=0)
        self.alpha_ = float(self.alphas_[np.argmin(self.cv_mse_)])
        self.model_ = make_pipeline(clone(self.preprocessor), StandardScaler(),
            Lasso(alpha=self.alpha_, max_iter=self.max_iter, random_state=self.seed)).fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


class FoldLocalMLP(RegressorMixin, BaseEstimator):
    def __init__(self, preprocessor, seed, params):
        self.preprocessor = preprocessor
        self.seed = seed
        self.params = params

    def _estimator(self, alpha):
        from .model_registry import AdaptiveMLPRegressor
        mlp = AdaptiveMLPRegressor(seed=self.seed, **self.params)._mlp(alpha)
        return make_pipeline(clone(self.preprocessor), StandardScaler(),
                             TransformedTargetRegressor(regressor=mlp, transformer=StandardScaler()))

    def fit(self, X, y):
        y = np.asarray(y)
        alphas = np.logspace(self.params['alpha_log10_min'], self.params['alpha_log10_max'], self.params['n_alphas'])
        folds = tuple(KFold(min(self.params['max_cv_folds'], len(y))).split(X))
        self.cv_mse_ = [np.mean([np.mean((self._estimator(alpha).fit(X.iloc[t], y[t]).predict(X.iloc[v]) - y[v]) ** 2)
                                      for t, v in folds]) for alpha in alphas]
        self.alpha_ = float(alphas[np.argmin(self.cv_mse_)])
        self.model_ = self._estimator(self.alpha_).fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)
