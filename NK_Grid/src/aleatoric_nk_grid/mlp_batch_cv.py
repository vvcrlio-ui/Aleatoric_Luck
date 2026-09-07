"""Serial, complete-pipeline MLP selection on training rows only."""
from __future__ import annotations

import time
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch


BATCH_CV_RULE = "mlp-batch-cv-v1-ordered-mean-fold-mse-no-shuffle-no-dedup"
DEFAULT_BATCH_CANDIDATES = (32, 64, 128, 256)


def validate_batch(policy, candidates, folds):
    if not ((isinstance(policy, str) and policy in {"auto", "full", "cv"})
            or (type(policy) is int and policy > 0)):
        raise ValueError("mlp_batch_size must be auto, full, cv, or a positive integer")
    if not isinstance(candidates, (list, tuple)) or not candidates:
        raise ValueError("mlp_batch_candidates must be a nonempty list or tuple")
    if any(type(b) is not int or b <= 0 for b in candidates):
        raise ValueError("mlp_batch_candidates must contain positive integers (not bool)")
    if len(set(candidates)) != len(candidates):
        raise ValueError("mlp_batch_candidates must not contain duplicates")
    if type(folds) is not int or folds < 2:
        raise ValueError("MLP CV folds must be an integer >= 2")


def rows(X, indices):
    return X.iloc[indices] if hasattr(X, "iloc") else np.asarray(X)[indices]


class BatchSearchMLP(RegressorMixin, BaseEstimator):
    """A single base learner, including all transformations in every inner fit.

    Only scalar diagnostics survive candidate fits. Candidate failures are
    explicit; final refit errors propagate. Constructor parameters are untouched
    so sklearn clone and pickle preserve the selection contract.
    """
    def __init__(self, mlp, alphas, mlp_batch_candidates=DEFAULT_BATCH_CANDIDATES,
                 max_cv_folds=3, preprocessor=None):
        self.mlp = mlp
        self.alphas = alphas
        self.mlp_batch_candidates = mlp_batch_candidates
        self.max_cv_folds = max_cv_folds
        self.preprocessor = preprocessor

    def _estimator(self, alpha, batch):
        process = (clone(self.preprocessor) if self.preprocessor is not None
                   else SimpleImputer(strategy="median", keep_empty_features=True))
        mlp = clone(self.mlp).set_params(alpha=float(alpha), batch_size=batch)
        return make_pipeline(process, StandardScaler(),
                             TransformedTargetRegressor(regressor=mlp, transformer=StandardScaler()))

    def fit(self, X, y):
        validate_batch("cv", self.mlp_batch_candidates, self.max_cv_folds)
        y = np.asarray(y, dtype=float).ravel()
        if not len(y) or len(X) != len(y) or not np.isfinite(y).all():
            raise ValueError("MLP requires nonempty matching X/y and finite targets")
        alphas = np.asarray(self.alphas, dtype=float)
        if alphas.ndim != 1 or not len(alphas) or not np.isfinite(alphas).all() or (alphas < 0).any():
            raise ValueError("MLP alphas must be a nonempty finite nonnegative sequence")
        started = time.perf_counter()
        self.fit_count_ = 0
        self.candidate_scores_ = []
        folds = tuple(KFold(min(self.max_cv_folds, len(y))).split(X)) if len(y) >= 3 else ()
        if folds and min(len(t) for t, _ in folds) < 2:
            folds = ()
        self.fallback_reason_ = (None if folds else
            "inner CV would have fewer than 2 training rows; first declared alpha/batch")
        best_score = float("inf")
        self.alpha_, self.batch_size_ = float(alphas[0]), self.mlp_batch_candidates[0]
        for alpha in alphas if folds else ():
            for batch in self.mlp_batch_candidates:
                record = {"alpha": float(alpha), "batch": batch, "folds": [], "mean_mse": None, "error": None}
                for train, valid in folds:
                    fitted = None
                    try:
                        self.fit_count_ += 1
                        fitted = self._estimator(alpha, batch).fit(rows(X, train), y[train])
                        prediction = fitted.predict(rows(X, valid))
                        if prediction.shape != y[valid].shape or not np.isfinite(prediction).all():
                            raise ValueError("nonfinite or invalid validation prediction")
                        mse = float(np.mean((prediction - y[valid]) ** 2))
                        if not np.isfinite(mse):
                            raise ValueError("nonfinite validation MSE")
                        mlp = fitted[-1].regressor_
                        record["folds"].append({"mse": mse, "N": len(train),
                            "effective_batch": mlp.effective_batch_size_, "iterations": mlp.n_iter_})
                        if mlp.solver == "lbfgs":
                            record["folds"][-1].update(solver="lbfgs", batch_size_applicable=False)
                        del mlp
                    except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
                        record["error"] = f"{type(exc).__name__}: {exc}"
                        break
                    finally:
                        del fitted
                if record["error"] is None:
                    record["mean_mse"] = float(np.mean([f["mse"] for f in record["folds"]]))
                    if not np.isfinite(record["mean_mse"]):
                        record["mean_mse"] = None
                        record["error"] = "ValueError: nonfinite mean validation MSE"
                    elif record["mean_mse"] < best_score:
                        best_score = record["mean_mse"]
                        self.alpha_, self.batch_size_ = float(alpha), batch
                self.candidate_scores_.append(record)
        if folds and not np.isfinite(best_score):
            raise ValueError(f"All MLP alpha/batch candidates failed: {self.candidate_scores_}")
        self.cv_mse_ = tuple(r["mean_mse"] for r in self.candidate_scores_)
        self.fit_count_ += 1
        self.model_ = self._estimator(self.alpha_, self.batch_size_).fit(X, y)
        if not np.isfinite(self.model_.predict(X)).all():
            raise ValueError("nonfinite final MLP prediction")
        mlp = self.model_[-1].regressor_
        # Timings belong to invocation telemetry, not immutable RESULT payloads.
        self.fit_seconds_ = time.perf_counter() - started
        self.diagnostics_ = {"rule": BATCH_CV_RULE, "selected_batch": self.batch_size_,
            "effective_batch": mlp.effective_batch_size_, "alpha": self.alpha_,
            "iterations": mlp.n_iter_, "max_iter": mlp.max_iter, "N": len(y),
            "convergence_warnings": mlp.convergence_warnings_,
            "fallback_reason": self.fallback_reason_, "candidates": self.candidate_scores_,
            "fit_count": self.fit_count_, "fit_count_kind": "pipeline fit attempts including final refit"}
        if mlp.solver == "lbfgs":
            self.diagnostics_.update(solver="lbfgs", batch_size_applicable=False)
        return self

    def predict(self, X):
        return self.model_.predict(X)


class SerialBatchStack(RegressorMixin, BaseEstimator):
    """Regression stacking with diagnostics from actual OOF fits, without replay."""
    def __init__(self, estimators, cv, positive, passthrough):
        self.estimators = estimators
        self.cv = cv
        self.positive = positive
        self.passthrough = passthrough

    def fit(self, X, y):
        y = np.asarray(y)
        folds = tuple(KFold(self.cv).split(X))
        oof = np.empty((len(y), len(self.estimators)))
        self.estimators_, self.named_estimators_ = [], Bunch()
        self.mlp_fits_ = []
        for column, (name, estimator) in enumerate(self.estimators):
            for fold, (train, valid) in enumerate(folds):
                fitted = clone(estimator).fit(rows(X, train), y[train])
                oof[valid, column] = fitted.predict(rows(X, valid))
                if name == "shallow_nn":
                    self.mlp_fits_.append({"phase": "oof", "fold": fold, **fitted.diagnostics_})
                del fitted
            fitted = clone(estimator).fit(X, y)
            self.estimators_.append(fitted)
            self.named_estimators_[name] = fitted
            if name == "shallow_nn":
                self.mlp_fits_.append({"phase": "full", **fitted.diagnostics_})
        if not np.isfinite(oof).all():
            raise ValueError("nonfinite stacking OOF predictions")
        design = np.column_stack((oof, X)) if self.passthrough else oof
        self.final_estimator_ = LinearRegression(positive=self.positive).fit(design, y)
        return self

    def predict(self, X):
        predictions = np.column_stack([est.predict(X) for est in self.estimators_])
        design = np.column_stack((predictions, X)) if self.passthrough else predictions
        return self.final_estimator_.predict(design)
