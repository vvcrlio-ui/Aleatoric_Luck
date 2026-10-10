"""SL7 combination of base-model predictions. This module fits no base model."""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize, nnls
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, ClassifierMixin


class MissingBasePredictions(ValueError):
    """SL7 cannot be combined because a required base prediction is unavailable."""


class NonNegativeLogisticRegression(ClassifierMixin, BaseEstimator):
    """NNLS's classification twin: unpenalized logistic, nonnegative weights.

    Minimizes mean log loss over a free intercept and nonnegative coefficients
    with L-BFGS-B, starting from the intercept-only model.
    """

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
        if not np.array_equal(np.unique(y), [0, 1]):
            raise ValueError("nonnegative logistic combiner requires both classes [0, 1]")

        def objective(theta):
            eta = theta[0] + X @ theta[1:]
            residual = expit(eta) - y
            loss = float(np.mean(np.logaddexp(0., eta) - y * eta))
            return loss, np.concatenate(([residual.mean()], X.T @ residual / len(y)))

        start = np.zeros(X.shape[1] + 1)
        start[0] = logit(y.mean())
        result = minimize(objective, start, jac=True, method="L-BFGS-B",
                          bounds=[(None, None)] + [(0., None)] * X.shape[1])
        self.intercept_ = result.x[:1]
        self.coef_ = result.x[None, 1:]
        self.objective_ = float(result.fun)
        self.converged_ = bool(result.success)
        self.message_ = str(result.message)
        self.classes_ = np.asarray([0, 1])
        return self

    def predict_proba(self, X):
        probability = expit(self.intercept_[0] + np.asarray(X, dtype=np.float64) @ self.coef_[0])
        return np.column_stack((1. - probability, probability))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= .5).astype(int)


def recombine_predictions(*, oof_predictions, holdout_predictions, y_train,
                          model_names, selected_models, task="regression",
                          combiner_config=None, variant_id, statuses=None):
    """Learn from OOF and training labels only; holdout labels are not accepted.

    Nonnegative coefficients include a fitted intercept and are not normalized:
    least squares for regression, log loss on positive-class probabilities for
    classification. Required missing columns are never dropped.
    """
    names, selected = tuple(model_names), tuple(selected_models)
    if not variant_id or not selected or len(set(selected)) != len(selected) or len(set(names)) != len(names):
        raise ValueError("variant ID and unique nonempty model lists are required")
    missing = set(selected) - set(names)
    if missing:
        raise MissingBasePredictions(f"required model columns missing: {sorted(missing)}")
    if statuses is not None:
        unavailable = {name: statuses.get(name, "missing") for name in selected if statuses.get(name) != "ok"}
        if unavailable:
            raise MissingBasePredictions(f"required model columns unavailable: {unavailable}")
    Z, P, y = np.asarray(oof_predictions), np.asarray(holdout_predictions), np.asarray(y_train, dtype=np.float64)
    if Z.dtype != np.float64 or P.dtype != np.float64:
        raise ValueError("prediction arrays must be float64")
    if Z.ndim != 2 or P.ndim != 2 or Z.shape[1] != len(names) or P.shape[1] != len(names) or y.shape != (Z.shape[0],):
        raise ValueError("OOF/holdout/model/sample dimensions do not align")
    columns = [names.index(name) for name in selected]
    Z, P = Z[:, columns], P[:, columns]
    if not np.isfinite(Z).all() or not np.isfinite(P).all() or not np.isfinite(y).all():
        raise MissingBasePredictions("nonfinite required predictions/labels")
    config = dict(combiner_config or {})
    started = time.perf_counter()
    convergence = []
    if task == "regression":
        if config not in ({}, {"rule": "nnls-intercept-v1"}):
            raise ValueError("regression combiner must explicitly use nnls-intercept-v1")
        center = Z.mean(axis=0)
        weights, _ = nnls(Z - center, y - y.mean())
        intercept = float(y.mean() - center @ weights)
        prediction = P @ weights + intercept
        objective = float(np.mean((Z @ weights + intercept - y) ** 2))
        config = {"rule": "nnls-intercept-v1"}
        classes = []
    elif task == "classification":
        if config not in ({}, {"rule": "nonnegative-logistic-intercept-v1"}):
            raise ValueError("classification combiner must explicitly use nonnegative-logistic-intercept-v1")
        if not np.array_equal(np.unique(y), [0, 1]):
            raise MissingBasePredictions("classification combiner requires both classes [0, 1]")
        if np.any((Z < 0) | (Z > 1)) or np.any((P < 0) | (P > 1)):
            raise ValueError("classification base predictions must be probabilities")
        estimator = NonNegativeLogisticRegression().fit(Z, y)
        if not estimator.converged_:
            convergence = [estimator.message_]
        weights = estimator.coef_[0]
        intercept = float(estimator.intercept_[0])
        prediction = estimator.predict_proba(P)[:, 1]
        objective = None
        config = {"rule": "nonnegative-logistic-intercept-v1"}
        classes = estimator.classes_.tolist()
    else:
        raise ValueError("task must be regression or classification")
    return {"holdout_prediction": np.asarray(prediction, dtype=np.float64),
            "coefficients": np.asarray(weights, dtype=np.float64), "intercept": intercept,
            "metadata": {"variant_id": variant_id, "selected_models": list(selected),
                         "task": task, "combiner": config, "classes": classes,
                         "base_fit_count": 0, "combiner_fit_count": 1,
                         "combiner_seconds": time.perf_counter() - started,
                         "oof_objective_mse": objective,
                         "converged": not convergence, "convergence_warnings": convergence}}
