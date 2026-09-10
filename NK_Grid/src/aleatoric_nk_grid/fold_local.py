"""Versioned complete-pipeline regression CV; legacy models remain M2 oracles."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge, Lasso, lasso_path
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .mlp_estimator import build_mlp_regressor
from .svd_fallback import ridge_svd


class FoldLocalRidge(RegressorMixin, BaseEstimator):
    """Five-fold complete-pipeline CV, with one SVD per fold for all alphas.

    Folds preserve row order. With fewer than five samples, use N folds;
    select by the unweighted mean of fold MSEs, as in GridSearchCV.
    """
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
        if len(y) < 2:
            raise ValueError('Ridge CV requires at least two training rows')
        self.n_splits_ = min(5, len(y))
        self.svd_fallback_count_ = 0
        self.alphas_ = np.logspace(self.alpha_log10_min, self.alpha_log10_max, self.n_alphas)
        self.cv_predictions_ = np.empty((len(y), len(self.alphas_)))
        fold_losses = []
        for train, valid in KFold(self.n_splits_, shuffle=False).split(X):
            process = make_pipeline(clone(self.preprocessor), StandardScaler()).fit(X.iloc[train])
            train_X, valid_X = process.transform(X.iloc[train]), process.transform(X.iloc[valid])
            # One fold-specific SVD serves the unchanged alpha grid. This is
            # complete-pipeline five-fold CV, not full-N analytic RidgeCV.
            center = train_X.mean(axis=0)
            target_mean = y[train].mean()
            (u, singular, vt), fallback_used = ridge_svd(train_X - center)
            self.svd_fallback_count_ += int(fallback_used)
            keep = singular > 1e-15  # sklearn Ridge's SVD rank cutoff
            singular = singular[keep]
            projection = (valid_X - center) @ vt[keep].T
            target = u[:, keep].T @ (y[train] - target_mean)
            factors = singular[:, None] / (singular[:, None] ** 2 + self.alphas_[None, :])
            self.cv_predictions_[valid, :] = (projection * target) @ factors + target_mean
            fold_losses.append(np.mean((self.cv_predictions_[valid] - y[valid, None]) ** 2, axis=0))
        self.fold_mse_ = np.asarray(fold_losses)
        self.cv_mse_ = self.fold_mse_.mean(axis=0)
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
            # Keep LassoCV's descending, warm-started coordinate-descent path
            # rather than introducing independent zero starts at every alpha.
            center, target_mean = train_X.mean(axis=0), y[train].mean()
            _, coefficients, _ = lasso_path(train_X - center, y[train] - target_mean,
                alphas=self.alphas_, max_iter=self.max_iter, random_state=self.seed)
            predictions = (valid_X - center) @ coefficients + target_mean
            losses.append(np.mean((predictions - y[valid, None]) ** 2, axis=0))
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
        mlp = build_mlp_regressor(seed=self.seed, alpha=alpha, params=self.params)
        return make_pipeline(clone(self.preprocessor), StandardScaler(),
                             TransformedTargetRegressor(regressor=mlp, transformer=StandardScaler()))

    def fit(self, X, y):
        from .mlp_batch_cv import validate_batch
        validate_batch(self.params.get('mlp_batch_size', 'auto'),
                       self.params.get('mlp_batch_candidates', (32, 64, 128, 256)),
                       self.params['max_cv_folds'])
        if self.params.get('mlp_batch_size', 'auto') == 'cv':
            from .mlp_batch_cv import BatchSearchMLP
            self.search_ = BatchSearchMLP(
                build_mlp_regressor(seed=self.seed, alpha=0.0, params=self.params),
                np.logspace(self.params['alpha_log10_min'], self.params['alpha_log10_max'], self.params['n_alphas']),
                self.params.get('mlp_batch_candidates', (32, 64, 128, 256)),
                self.params['max_cv_folds'], self.preprocessor).fit(X, y)
            self.alpha_, self.batch_size_ = self.search_.alpha_, self.search_.batch_size_
            self.cv_mse_, self.diagnostics_ = self.search_.cv_mse_, self.search_.diagnostics_
            self.model_ = self.search_.model_
            return self
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
