"""Versioned base prediction training and passive capture of reported SL fits.

No cache lookup is performed here: callers must verify the complete immutable
identity before supplying resumed folds. Arrays stay off the result RPC channel.
"""
from __future__ import annotations

import time
import warnings
from collections.abc import Mapping

import numpy as np
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold, check_cv

BASE_MODELS = ("ols", "ridge", "lasso", "random_forest", "xgboost", "lightgbm",
               "shallow_neural_network", "extra_trees")
BASE_LIBRARY = "standalone8-v1"
FORMAL_REGRESSION_MODELS = ("ridge", "extra_trees", "lightgbm", "shallow_nn")


def rows(X, indexes):
    return X.iloc[indexes] if hasattr(X, "iloc") else np.asarray(X)[indexes]


class _CaptureStack:
    """Intercept the actual matrix passed to sklearn's final estimator.

    This keeps sklearn's fit ordering and all model settings unchanged. Capturing
    predictions never replays cross-validation or a base estimator fit.
    """
    def fit(self, X, y, **kwargs):
        self._capturing_oof = True
        try:
            answer = super().fit(X, y, **kwargs)
        finally:
            self._capturing_oof = False
        splitter = check_cv(self.cv, y=y, classifier=isinstance(self, StackingClassifier))
        self.oof_fold_ = np.full(len(y), -1, dtype=np.int64)
        for fold, (_, valid) in enumerate(splitter.split(X, y)):
            self.oof_fold_[valid] = fold
        self.base_model_names_ = tuple(name for name, est in self.estimators if est != "drop")
        return answer

    def _concatenate_predictions(self, X, predictions):
        design = super()._concatenate_predictions(X, predictions)
        if self._capturing_oof:
            self.oof_predictions_ = np.array(design[:, :len(self.estimators)], dtype=np.float64, copy=True)
        return design


class CapturingStackingRegressor(_CaptureStack, StackingRegressor):
    pass


class CapturingStackingClassifier(_CaptureStack, StackingClassifier):
    pass


def formal_sl_predictions(model, X_holdout, reported_prediction=None):
    """Extract actual reported SL inputs/parameters after an opt-in captured fit."""
    stack = model.model_
    if not hasattr(stack, "oof_predictions_"):
        raise ValueError("formal SL prediction capture was not enabled before fit")
    names = list(stack.base_model_names_)
    holdout = np.asarray(stack.transform(X_holdout), dtype=np.float64)[:, :len(names)]
    meta = stack.final_estimator_
    task = "classification" if hasattr(model, "predict_proba") else "regression"
    if reported_prediction is None:
        reported_prediction = predict_values(model, X_holdout, task)
    return {
        "arrays": {"base_oof_prediction": np.asarray(stack.oof_predictions_, dtype=np.float64),
                   "base_holdout_prediction": holdout,
                   "holdout_prediction": np.asarray(reported_prediction, dtype=np.float64),
                   "oof_fold": np.asarray(stack.oof_fold_, dtype=np.int64)},
        "metadata": {"pipeline_id": "reported-sl4-" + task + "-v1", "model_names": names,
                     "coefficients": np.asarray(meta.coef_).tolist(),
                     "intercept": np.asarray(meta.intercept_).tolist(),
                     "classes": np.asarray(getattr(model, "classes_", [])).tolist(),
                     "passthrough": bool(model.passthrough), "task": task,
                     "base_fit_count": len(names) * (len(np.unique(stack.oof_fold_)) + 1)},
    }


def predict_values(model, X, task):
    if task == "classification":
        classes = np.asarray(model.classes_)
        if not np.array_equal(classes, [0, 1]):
            raise ValueError(f"FFC probability cache requires class mapping [0, 1], got {classes}")
        values = np.asarray(model.predict_proba(X), dtype=np.float64)[:, 1]
        if np.any((values < 0) | (values > 1)):
            raise ValueError("classifier returned values outside [0, 1]")
    else:
        values = np.asarray(model.predict(X), dtype=np.float64)
    if values.shape != (len(X),) or not np.isfinite(values).all():
        raise ValueError("invalid/nonfinite prediction; cache cannot be complete")
    return values


def make_oof_folds(y, task, folds=5):
    if type(folds) is not int or folds < 2:
        raise ValueError("OOF folds must be an integer >= 2")
    y = np.asarray(y)
    if task == "classification":
        classes, counts = np.unique(y, return_counts=True)
        if not np.array_equal(classes, [0, 1]):
            raise ValueError("single-class training sample for classification")
        count = min(folds, int(counts.min()))
        if count < 2:
            raise ValueError("below minimum per-class count for OOF CV")
        splitter = StratifiedKFold(count, shuffle=False)
    elif task == "regression":
        count = min(folds, len(y))
        if count < 2:
            raise ValueError("OOF requires at least two training rows")
        splitter = KFold(count, shuffle=False)
    else:
        raise ValueError("task must be regression or classification")
    return tuple((train.astype(np.int64), valid.astype(np.int64))
                 for train, valid in splitter.split(np.empty((len(y), 0)), y))


def _selected_parameters(model):
    """Small auditable selected values from each nested fitted estimator."""
    result = []
    seen = set()
    def visit(estimator, path):
        if id(estimator) in seen:
            return
        seen.add(id(estimator))
        entry = {"path": path, "type": type(estimator).__name__}
        for name in ("alpha_", "best_rounds_", "best_iteration_", "n_iter_", "batch_size_",
                     "effective_batch_size_", "fit_count_", "n_splits_", "convergence_warnings_"):
            value = getattr(estimator, name, None)
            if value is not None:
                entry[name] = np.asarray(value).tolist()
        if len(entry) > 2:
            result.append(entry)
        for attribute in ("model_", "regressor_", "best_estimator_"):
            child = getattr(estimator, attribute, None)
            if child is not None:
                visit(child, path + "." + attribute)
        for name, child in getattr(estimator, "steps", []):
            visit(child, path + "." + name)
    visit(model, "model")
    return result


def train_base_predictions(*, model_name, model_seed, task, params, X_train,
                           y_train, X_test, preprocessor=None, n_jobs=1,
                           oof_folds=5, mode="holdout_oof", resume_folds=None,
                           persist_fold=None, pipeline_id=None, formal_params=None):
    """Fit each full/OOF pipeline once, sharing the full prediction with scoring.

    ``persist_fold`` runs immediately after a valid prediction, before the next
    fit, allowing durable recovery of folds and the full fit (-1). Resume entries
    contain ``prediction``, ``positions`` and ``metadata`` and are supplied only
    after the caller has verified their complete cache identity.
    """
    from .model_registry import make_model
    from .validate_input import REGRESSION_CV_MIN_N
    if model_name not in BASE_MODELS:
        raise ValueError("base stage cannot fit super_learner or unknown models")
    if mode not in {"holdout", "holdout_oof"}:
        raise ValueError("prediction training mode must be holdout or holdout_oof")
    if type(oof_folds) is not int or oof_folds < 2:
        raise ValueError("OOF folds must be an integer >= 2")
    formal_prefix = f"reported-sl4-{task}-v1/"
    is_formal = bool(pipeline_id and pipeline_id.startswith(formal_prefix))
    formal_template = None
    process = preprocessor if preprocessor is not None else SimpleImputer(strategy="median", keep_empty_features=True)
    if is_formal:
        if formal_params is None:
            raise ValueError("formal base recipe requires frozen formal SL parameters")
        formal_model = make_model("super_learner", seed=model_seed, n_jobs=n_jobs, task=task,
                                  params=formal_params, preprocessor=process)
        templates = dict(formal_model.base_estimators())
        internal_name = pipeline_id[len(formal_prefix):]
        if internal_name not in templates:
            raise ValueError(f"unknown formal SL base recipe {pipeline_id}")
        formal_template = templates[internal_name]
        oof_folds = int(formal_params["cv"])
    elif pipeline_id not in (None, f"{BASE_LIBRARY}/{model_name}"):
        raise ValueError(f"unknown base pipeline {pipeline_id}")
    if not hasattr(X_train, "iloc"):
        import pandas as pd
        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test, columns=X_train.columns)
    y = np.asarray(y_train, dtype=np.float64)
    if y.shape != (len(X_train),) or not np.isfinite(y).all():
        raise ValueError("nonfinite labels or mismatching training order")
    metadata = {"pipeline_id": pipeline_id or f"{BASE_LIBRARY}/{model_name}", "model_name": internal_name if is_formal else model_name,
                "task": task, "status": "ok", "reason": "", "mode": mode,
                "classes": [0, 1] if task == "classification" else [],
                "holdout_status": "ok", "oof_status": "ok" if mode == "holdout_oof" else "not_required",
                "base_fit_count": 0, "fit_count_kind": "outer complete-pipeline fits; excludes internal tuning",
                "full_fit_seconds": 0., "oof_fit_seconds": 0., "resumed_fits": 0,
                "folds": [], "converged": True}
    try:
        if task == "classification" and not np.array_equal(np.unique(y), [0, 1]):
            raise ValueError("single-class training sample for classification")
        minimum = ((2 if internal_name == "ridge" else 1) if is_formal else REGRESSION_CV_MIN_N.get(model_name, 1)) if task == "regression" else 1
        if len(y) < minimum:
            raise ValueError(f"below minimum N for {model_name}'s internal CV in full training (requires N>={minimum})")
    except ValueError as exc:
        metadata.update(status="skipped", reason=str(exc), holdout_status="skipped", oof_status="skipped", oof_reason=str(exc))
        return {"arrays": {}, "metadata": metadata, "model": None, "predictions": None}
    try:
        folds = make_oof_folds(y, task, oof_folds) if mode == "holdout_oof" else ()
        if any(len(train) < minimum for train, _ in folds):
            raise ValueError(f"below minimum N for {model_name}'s internal CV in OOF training (requires N>={minimum})")
    except ValueError as exc:
        # A legal OOF skip cannot erase a valid independent full prediction.
        folds = ()
        metadata.update(oof_status="skipped", oof_reason=str(exc))
    resumed = resume_folds or {}
    unknown = set(resumed) - ({-1} | set(range(len(folds))))
    if unknown:
        raise ValueError(f"resumed fold IDs not in frozen split: {unknown}")
    oof = np.empty(len(y), dtype=np.float64)
    fold_ids = np.full(len(y), -1, dtype=np.int64)
    full_model = None
    def fit_one(fold, train, valid):
        nonlocal full_model
        if fold in resumed:
            record = resumed[fold]
            expected = np.arange(len(X_test)) if fold == -1 else valid
            if not np.array_equal(record["positions"], expected):
                raise ValueError("resumed prediction positions differ from frozen split")
            prediction = np.asarray(record["prediction"])
            if prediction.dtype != np.float64 or prediction.shape != (len(expected),) or not np.isfinite(prediction).all():
                raise ValueError("invalid resumed prediction")
            metadata["resumed_fits"] += 1
            metadata["folds"].append(record["metadata"])
            metadata["converged"] &= bool(record["metadata"].get("converged", True))
            return prediction
        estimator = (clone(formal_template) if is_formal else
            make_model(model_name, seed=model_seed, n_jobs=n_jobs, task=task,
                       params=params, preprocessor=process))
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            estimator.fit(rows(X_train, train), y[train])
            prediction = predict_values(estimator, X_test if fold == -1 else rows(X_train, valid), task)
        seconds = time.perf_counter() - started
        convergence = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
        selected = _selected_parameters(estimator)
        converged = not convergence and not any(item.get("convergence_warnings_", 0) for item in selected)
        record = {"fold": fold, "train_size": len(train), "seconds": seconds,
                  "selected_parameters": selected, "converged": converged,
                  "convergence_warnings": convergence}
        metadata["folds"].append(record)
        metadata["base_fit_count"] += 1
        metadata["full_fit_seconds" if fold == -1 else "oof_fit_seconds"] += seconds
        metadata["converged"] &= converged
        if persist_fold is not None:
            persist_fold(fold, {"prediction": prediction,
                               "positions": np.arange(len(X_test), dtype=np.int64) if fold == -1 else valid,
                               "metadata": record})
        if fold == -1:
            full_model = estimator
        return prediction
    # Full first: same independent fit and score, before additional OOF work.
    holdout = fit_one(-1, np.arange(len(y)), None)
    for fold, (train, valid) in enumerate(folds):
        if np.intersect1d(train, valid).size:
            raise RuntimeError("OOF training/validation overlap")
        oof[valid] = fit_one(fold, train, valid)
        fold_ids[valid] = fold
    arrays = {"holdout_prediction": holdout}
    if folds:
        if np.any(fold_ids < 0) or not np.isfinite(oof).all():
            raise RuntimeError("incomplete OOF predictions")
        arrays.update(oof_prediction=oof, oof_fold=fold_ids)
    return {"arrays": arrays, "metadata": metadata, "model": full_model, "predictions": holdout}
