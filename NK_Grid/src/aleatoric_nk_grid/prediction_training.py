"""Base-model test and out-of-fold predictions for one cell."""
from __future__ import annotations

import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold

BASE_MODELS = ("ols", "ridge", "lasso", "random_forest", "xgboost", "lightgbm",
               "shallow_neural_network", "extra_trees")
BASE_LIBRARY = "standalone8-v1"
# Their OOF fits reuse the full fit's ``selection_`` instead of searching.
TUNED_MODELS = ("ridge", "lasso", "xgboost", "lightgbm", "shallow_neural_network")


def rows(X, indexes):
    return X.iloc[indexes] if hasattr(X, "iloc") else np.asarray(X)[indexes]


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
                           y_train, X_test, preprocessor=None, oof_folds=5):
    """Fit one model on all N rows, then once per OOF fold, sharing the full prediction with scoring.

    OOF fits of tuned models reuse the full fit's selected value (recorded as
    fold -1's ``oof_selection``) and do not repeat its parameter search.
    """
    from .model_registry import make_model
    from .validate_input import CLASSIFICATION_CV_MODELS, REGRESSION_CV_MIN_N
    if model_name not in BASE_MODELS:
        raise ValueError("base stage cannot fit super_learner or unknown models")
    if type(oof_folds) is not int or oof_folds < 2:
        raise ValueError("OOF folds must be an integer >= 2")
    process = preprocessor if preprocessor is not None else SimpleImputer(strategy="median", keep_empty_features=True)
    if not hasattr(X_train, "iloc"):
        import pandas as pd
        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test, columns=X_train.columns)
    y = np.asarray(y_train, dtype=np.float64)
    if y.shape != (len(X_train),) or not np.isfinite(y).all():
        raise ValueError("nonfinite labels or mismatching training order")
    metadata = {"pipeline_id": f"{BASE_LIBRARY}/{model_name}", "model_name": model_name,
                "task": task, "status": "ok", "reason": "",
                "classes": [0, 1] if task == "classification" else [],
                "holdout_status": "ok", "oof_status": "ok",
                "base_fit_count": 0, "fit_count_kind": "outer complete-pipeline fits; excludes internal tuning",
                "full_fit_seconds": 0., "oof_fit_seconds": 0., "folds": [], "converged": True}
    minimum = REGRESSION_CV_MIN_N.get(model_name, 1)
    classification_cv = model_name in CLASSIFICATION_CV_MODELS
    def check_cv_rows(labels, stage):
        if task == "regression" and len(labels) < minimum:
            raise ValueError(f"below minimum N for {model_name}'s internal CV in {stage} training (requires N>={minimum})")
        if task == "classification" and classification_cv and np.bincount(labels.astype(int), minlength=2).min() < 2:
            raise ValueError(f"below minimum per-class count for {model_name}'s internal CV in {stage} training")
    try:
        if task == "classification" and not np.array_equal(np.unique(y), [0, 1]):
            raise ValueError("single-class training sample for classification")
        check_cv_rows(y, "full")
    except ValueError as exc:
        metadata.update(status="skipped", reason=str(exc), holdout_status="skipped", oof_status="skipped", oof_reason=str(exc))
        return {"arrays": {}, "metadata": metadata, "model": None, "predictions": None}
    try:
        folds = make_oof_folds(y, task, oof_folds)
        for train, _ in folds:
            check_cv_rows(y[train], "OOF")
    except ValueError as exc:
        # A legal OOF skip cannot erase a valid independent full prediction.
        folds = ()
        metadata.update(oof_status="skipped", oof_reason=str(exc))
    oof = np.empty(len(y), dtype=np.float64)
    fold_ids = np.full(len(y), -1, dtype=np.int64)
    full_model = None
    fixed = model_name in TUNED_MODELS
    selection = None
    def fit_one(fold, train, valid):
        nonlocal full_model
        estimator = make_model(model_name, seed=model_seed, task=task, params=params,
                               preprocessor=process, selected=selection)
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
        if fold == -1 and fixed:
            record["oof_selection"] = estimator.selection_
        metadata["folds"].append(record)
        metadata["base_fit_count"] += 1
        metadata["full_fit_seconds" if fold == -1 else "oof_fit_seconds"] += seconds
        metadata["converged"] &= converged
        if fold == -1:
            full_model = estimator
        return prediction
    # Full first: same independent fit and score, before additional OOF work.
    holdout = fit_one(-1, np.arange(len(y)), None)
    if fixed:
        selection = metadata["folds"][0]["oof_selection"]
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
