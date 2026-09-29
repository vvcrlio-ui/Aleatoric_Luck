"""Versioned complete-pipeline CV; legacy regression models remain M2 oracles."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LogisticRegression, lasso_path
from sklearn.metrics import log_loss
from .robust_linear import Ridge
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .mlp_estimator import build_mlp_classifier, build_mlp_regressor
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
    def __init__(self, preprocessor, seed, n_jobs, alpha_log10_min, alpha_log10_max, n_alphas, max_cv_folds, max_iter, alpha_scale='absolute', tol=1e-4):
        self.preprocessor = preprocessor
        self.seed = seed
        self.n_jobs = n_jobs
        self.alpha_log10_min = alpha_log10_min
        self.alpha_log10_max = alpha_log10_max
        self.n_alphas = n_alphas
        self.max_cv_folds = max_cv_folds
        self.max_iter = max_iter
        self.alpha_scale = alpha_scale
        self.tol = tol

    def fit(self, X, y):
        if self.alpha_scale == 'relative':
            return self._fit_relative(X, y)
        if self.alpha_scale != 'absolute':
            raise ValueError('alpha_scale must be absolute or relative')
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

    def _fit_relative(self, X, y):
        """Select a ratio using training-fold alpha_max; refit at full-N scale."""
        import json
        from sklearn.dummy import DummyRegressor

        y = np.asarray(y, dtype=float)
        if len(y) < 2 or not np.isfinite(y).all():
            raise ValueError('Relative Lasso requires at least two finite targets')
        if (self.max_cv_folds < 2 or self.n_alphas < 2 or self.max_iter < 1
                or not np.isfinite(self.tol) or self.tol <= 0
                or not np.isfinite([self.alpha_log10_min, self.alpha_log10_max]).all()
                or not self.alpha_log10_min < self.alpha_log10_max <= 0):
            raise ValueError('Invalid relative Lasso search/stopping parameters')
        self.ratios_ = np.logspace(self.alpha_log10_max, self.alpha_log10_min, self.n_alphas)
        self.n_splits_ = min(self.max_cv_folds, len(y))
        self.cv_predictions_ = np.empty((len(y), self.n_alphas))
        losses, maxima, iterations, gaps = [], [], [], []
        for train, valid in KFold(self.n_splits_, shuffle=False).split(X):
            process = make_pipeline(clone(self.preprocessor), StandardScaler()).fit(X.iloc[train])
            train_X, valid_X = process.transform(X.iloc[train]), process.transform(X.iloc[valid])
            center, target_mean = train_X.mean(axis=0), y[train].mean()
            centered = np.asfortranarray(train_X - center)
            target = y[train] - target_mean
            alpha_max = float(np.max(np.abs(centered.T @ target), initial=0.) / len(train))
            maxima.append(alpha_max)
            if alpha_max == 0.:
                predictions = np.full((len(valid), self.n_alphas), target_mean)
                n_iter, dual_gaps = np.zeros(self.n_alphas, dtype=int), np.zeros(self.n_alphas)
            else:
                _, coefficients, dual_gaps, n_iter = lasso_path(centered, target,
                    alphas=self.ratios_ * alpha_max, max_iter=self.max_iter,
                    tol=self.tol, random_state=self.seed, return_n_iter=True)
                predictions = (valid_X - center) @ coefficients + target_mean
            self.cv_predictions_[valid] = predictions
            losses.append(np.mean((predictions - y[valid, None]) ** 2, axis=0))
            iterations.append(np.asarray(n_iter, dtype=int))
            gaps.append(np.asarray(dual_gaps, dtype=float))
        self.fold_alpha_max_ = np.asarray(maxima)
        self.fold_mse_ = np.asarray(losses)
        self.fold_n_iter_ = np.asarray(iterations)
        self.fold_dual_gaps_ = np.asarray(gaps)
        self.cv_mse_ = self.fold_mse_.mean(axis=0)
        selected = int(np.argmin(self.cv_mse_))  # ties prefer stronger regularization
        self.selected_ratio_ = float(self.ratios_[selected])
        self.boundary_ = 'weak' if selected == self.n_alphas - 1 else ('strong' if selected == 0 else 'interior')
        process = make_pipeline(clone(self.preprocessor), StandardScaler()).fit(X)
        full_X = process.transform(X)
        centered_y = y - y.mean()
        self.alpha_max_ = float(np.max(np.abs((full_X - full_X.mean(axis=0)).T @ centered_y), initial=0.) / len(y))
        self.alpha_ = self.selected_ratio_ * self.alpha_max_
        estimator = (DummyRegressor(strategy='mean') if self.alpha_max_ == 0. else
            Lasso(alpha=self.alpha_, max_iter=self.max_iter, tol=self.tol, random_state=self.seed))
        estimator.fit(full_X, y)
        self.model_ = make_pipeline(process, estimator)
        self.n_iter_ = np.append(self.fold_n_iter_.ravel(), getattr(estimator, 'n_iter_', 0))
        # Small machine-readable diagnostics persist in each independent worker log.
        self.lasso_diagnostics_ = dict(protocol='relative-lasso-3fold-v1', folds=self.n_splits_,
            n_samples=len(y), expanded_columns=int(full_X.shape[1]),
            ratios=self.ratios_.tolist(), fold_alpha_max=self.fold_alpha_max_.tolist(),
            fold_iterations=self.fold_n_iter_.tolist(), fold_dual_gaps=self.fold_dual_gaps_.tolist(),
            cv_mse=self.cv_mse_.tolist(), selected_ratio=self.selected_ratio_, alpha=self.alpha_,
            boundary=self.boundary_, max_iter_hits=int(np.sum(self.n_iter_ >= self.max_iter)),
            refit_iterations=int(getattr(estimator, 'n_iter_', 0)), tol=self.tol)
        print(json.dumps({'lasso_diagnostics': self.lasso_diagnostics_}, allow_nan=False), flush=True)
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


# Classification twins of the regression CV above. Folds are ordered and
# stratified, as in the OOF split, so every fold holds both classes; the number
# of folds shrinks to the minority count. Selection minimizes mean fold log
# loss, the Bernoulli counterpart of mean fold MSE.

def classification_preprocessor(preprocessor):
    return preprocessor if preprocessor is not None else SimpleImputer(strategy="median", keep_empty_features=True)


def stratified_cv_folds(y, folds, model_name, shuffle_seed=None):
    """Ordered folds, or seeded shuffled ones where the regression twin shuffles."""
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    if not np.array_equal(classes, [0, 1]):
        raise ValueError("single-class training sample for classification")
    count = min(folds, int(counts.min()))
    if count < 2:
        raise ValueError(f"below minimum per-class count for {model_name}'s internal CV")
    splitter = StratifiedKFold(count, shuffle=shuffle_seed is not None, random_state=shuffle_seed)
    return tuple(splitter.split(np.empty((len(y), 0)), y))


def _rows(X, indexes):
    return X.iloc[indexes] if hasattr(X, "iloc") else np.asarray(X)[indexes]


def _fold_log_loss(y, probability):
    return float(log_loss(y, probability, labels=[0, 1]))


class _Classifier(ClassifierMixin, BaseEstimator):
    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


class FoldLocalLogisticRidge(_Classifier):
    """L2 logistic twin of FoldLocalRidge: five stratified folds, same alpha grid.

    Ridge regression minimizes the squared-error sum, twice the Gaussian negative
    log-likelihood, plus alpha*||w||^2. The same alpha on the Bernoulli negative
    log-likelihood is sklearn's C = 1/alpha. Each fold walks the grid from the
    strongest penalty down, warm-starting every fit from the previous solution.
    """
    def __init__(self, preprocessor, alpha_log10_min, alpha_log10_max, n_alphas, max_iter):
        self.preprocessor = preprocessor
        self.alpha_log10_min = alpha_log10_min
        self.alpha_log10_max = alpha_log10_max
        self.n_alphas = n_alphas
        self.max_iter = max_iter

    def _logistic(self, alpha, warm_start=False):
        return LogisticRegression(C=1. / alpha, l1_ratio=0., solver="lbfgs",
                                  max_iter=self.max_iter, warm_start=warm_start)

    def fit(self, X, y):
        y = np.asarray(y, dtype=int)
        process = classification_preprocessor(self.preprocessor)
        folds = stratified_cv_folds(y, 5, "ridge")
        self.n_splits_ = len(folds)
        self.alphas_ = np.logspace(self.alpha_log10_min, self.alpha_log10_max, self.n_alphas)
        losses = []
        for train, valid in folds:
            scaled = make_pipeline(clone(process), StandardScaler()).fit(_rows(X, train))
            train_X, valid_X = scaled.transform(_rows(X, train)), scaled.transform(_rows(X, valid))
            model = self._logistic(self.alphas_[-1], warm_start=True)
            fold = np.empty(len(self.alphas_))
            for index in range(len(self.alphas_) - 1, -1, -1):
                model.set_params(C=1. / self.alphas_[index]).fit(train_X, y[train])
                fold[index] = _fold_log_loss(y[valid], model.predict_proba(valid_X)[:, 1])
            losses.append(fold)
        self.fold_log_loss_ = np.asarray(losses)
        self.cv_log_loss_ = self.fold_log_loss_.mean(axis=0)
        self.alpha_ = float(self.alphas_[np.argmin(self.cv_log_loss_)])
        self.model_ = make_pipeline(clone(process), StandardScaler(), self._logistic(self.alpha_)).fit(X, y)
        self.classes_ = self.model_.classes_
        return self


class FoldLocalLogisticLasso(_Classifier):
    """L1 logistic twin of FoldLocalLasso's absolute alpha scale.

    Lasso regression minimizes the mean Gaussian negative log-likelihood plus
    alpha*||w||_1. The same objective on the Bernoulli likelihood is sklearn's
    C = 1/(n*alpha), with n the rows of that fit. As in LassoCV, alphas run from
    the strongest penalty down with warm starts, and ties keep the stronger one.
    """
    def __init__(self, preprocessor, seed, alpha_log10_min, alpha_log10_max, n_alphas,
                 max_cv_folds, max_iter, tol=1e-4):
        self.preprocessor = preprocessor
        self.seed = seed
        self.alpha_log10_min = alpha_log10_min
        self.alpha_log10_max = alpha_log10_max
        self.n_alphas = n_alphas
        self.max_cv_folds = max_cv_folds
        self.max_iter = max_iter
        self.tol = tol

    def _logistic(self, n, alpha, warm_start=False):
        return LogisticRegression(C=1. / (n * alpha), l1_ratio=1., solver="saga", max_iter=self.max_iter,
                                  tol=self.tol, random_state=self.seed, warm_start=warm_start)

    def fit(self, X, y):
        y = np.asarray(y, dtype=int)
        process = classification_preprocessor(self.preprocessor)
        folds = stratified_cv_folds(y, self.max_cv_folds, "lasso")
        self.n_splits_ = len(folds)
        self.alphas_ = np.logspace(self.alpha_log10_min, self.alpha_log10_max, self.n_alphas)[::-1]
        losses = []
        for train, valid in folds:
            scaled = make_pipeline(clone(process), StandardScaler()).fit(_rows(X, train))
            train_X, valid_X = scaled.transform(_rows(X, train)), scaled.transform(_rows(X, valid))
            model = self._logistic(len(train), self.alphas_[0], warm_start=True)
            fold = []
            for alpha in self.alphas_:
                model.set_params(C=1. / (len(train) * alpha)).fit(train_X, y[train])
                fold.append(_fold_log_loss(y[valid], model.predict_proba(valid_X)[:, 1]))
            losses.append(fold)
        self.fold_log_loss_ = np.asarray(losses)
        self.cv_log_loss_ = self.fold_log_loss_.mean(axis=0)
        self.alpha_ = float(self.alphas_[np.argmin(self.cv_log_loss_)])
        self.model_ = make_pipeline(clone(process), StandardScaler(),
                                    self._logistic(len(y), self.alpha_)).fit(X, y)
        self.classes_ = self.model_.classes_
        return self


class FoldLocalMLPClassifier(_Classifier):
    """FoldLocalMLP's alpha grid and folds, scored by log loss on 0/1 labels."""
    def __init__(self, preprocessor, seed, params):
        self.preprocessor = preprocessor
        self.seed = seed
        self.params = params

    def _estimator(self, alpha):
        return make_pipeline(clone(classification_preprocessor(self.preprocessor)), StandardScaler(),
                             build_mlp_classifier(seed=self.seed, alpha=alpha, params=self.params))

    def fit(self, X, y):
        y = np.asarray(y, dtype=int)
        folds = stratified_cv_folds(y, self.params["max_cv_folds"], "shallow_neural_network")
        alphas = np.logspace(self.params["alpha_log10_min"], self.params["alpha_log10_max"], self.params["n_alphas"])
        self.cv_log_loss_ = [np.mean([_fold_log_loss(y[v], self._estimator(alpha).fit(_rows(X, t), y[t])
                                                     .predict_proba(_rows(X, v))[:, 1]) for t, v in folds])
                             for alpha in alphas]
        self.alpha_ = float(alphas[np.argmin(self.cv_log_loss_)])
        self.model_ = self._estimator(self.alpha_).fit(X, y)
        self.classes_ = self.model_.classes_
        return self
