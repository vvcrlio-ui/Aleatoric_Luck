"""Model constructors for the shared N×K engine."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import DEFAULT_MODEL_PARAMS_PATH
from .robust_linear import LinearRegression

MODEL_NAMES = (
    "ols",
    "ridge",
    "lasso",
    "random_forest",
    "xgboost",
    "lightgbm",
    "shallow_neural_network",
    "extra_trees",
    "super_learner",
)

# Models that were deliberately retired from the model space. Naming one is a
# hard error rather than a silent drop, so a stale panel fails loudly instead of
# quietly producing results for one model fewer than it asked for.
REMOVED_MODEL_NAMES = {
    "bart": "BART was removed from the model space; remove 'bart' from the requested models",
    "elastic_net": "elastic_net was removed from the model space; remove 'elastic_net' from the requested models",
}
SUPPORTED_MODEL_NAMES = MODEL_NAMES


def reject_removed_model(name: str) -> None:
    """Raise a self-explaining error for a model that no longer exists."""

    reason = REMOVED_MODEL_NAMES.get(str(name).lower())
    if reason is not None:
        raise ValueError(reason)


MODEL_PARAM_KEYS = {
    "regression": {
        "ols": {"fit_intercept"},
        "ridge": {
            "alpha_log10_min", "alpha_log10_max", "n_alphas", "scoring",
        },
        "lasso": {
            "alpha_log10_min", "alpha_log10_max", "n_alphas",
            "max_cv_folds", "max_iter", "alpha_scale", "tol",
        },
        "random_forest": {"n_estimators", "max_features", "min_samples_leaf"},
        "extra_trees": {"n_estimators", "max_features", "min_samples_leaf"},
        "shallow_neural_network": {
            "hidden_layer_sizes", "activation", "solver",
            "alpha_log10_min", "alpha_log10_max", "n_alphas", "max_cv_folds",
            "learning_rate_init", "max_iter", "early_stopping",
            "validation_fraction", "n_iter_no_change",
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
    },
    "classification": {
        "ols": {"C", "l1_ratio", "solver", "max_iter"},
        "ridge": {"alpha_log10_min", "alpha_log10_max", "n_alphas", "max_iter"},
        "lasso": {
            "alpha_log10_min", "alpha_log10_max", "n_alphas",
            "max_cv_folds", "max_iter", "alpha_scale", "tol",
        },
        "random_forest": {"n_estimators", "max_features", "min_samples_leaf"},
        "extra_trees": {"n_estimators", "max_features", "min_samples_leaf"},
        "shallow_neural_network": {
            "hidden_layer_sizes", "activation", "solver",
            "alpha_log10_min", "alpha_log10_max", "n_alphas", "max_cv_folds",
            "learning_rate_init", "max_iter", "early_stopping",
            "validation_fraction", "n_iter_no_change",
            "mlp_batch_size", "mlp_l2_normalization", "tol",
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
    },
}


def _validated_params(
    task: str,
    model_name: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    reject_removed_model(model_name)
    allowed = MODEL_PARAM_KEYS.get(task, {}).get(model_name)
    if task == "regression" and model_name == "shallow_neural_network":
        allowed = allowed | {"mlp_batch_size", "mlp_batch_candidates", "mlp_l2_normalization", "tol"}
        params = {"mlp_batch_size": "auto", "mlp_l2_normalization": "batch", "tol": 1e-4, **params}
        if params["mlp_l2_normalization"] not in {"batch", "fit_samples", "effective_batch"}:
            raise ValueError("mlp_l2_normalization must be batch, fit_samples or effective_batch")
        from .mlp_batch_cv import DEFAULT_BATCH_CANDIDATES, validate_batch
        params = {"mlp_batch_candidates": list(DEFAULT_BATCH_CANDIDATES), **params}
        validate_batch(params["mlp_batch_size"], params["mlp_batch_candidates"], params.get("max_cv_folds"))
    if task == "classification" and model_name == "shallow_neural_network":
        params = {"mlp_batch_size": "auto", "mlp_l2_normalization": "batch", "tol": 1e-4, **params}
        if params["mlp_l2_normalization"] not in {"batch", "fit_samples", "effective_batch"}:
            raise ValueError("mlp_l2_normalization must be batch, fit_samples or effective_batch")
        policy = params["mlp_batch_size"]
        if not (policy in {"auto", "full", "balanced"} if isinstance(policy, str)
                else type(policy) is int and policy > 0):
            raise ValueError("classification mlp_batch_size must be auto, full, balanced, or a positive integer")
    if allowed is None:
        reject_removed_model(model_name)
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
    if model_name in {"xgboost", "lightgbm"}:
        missing = sorted(allowed - set(params))
        if missing:
            raise ValueError(
                f"Missing required parameters for {task} model '{model_name}': "
                f"{', '.join(missing)}"
            )
    if model_name in {"xgboost", "lightgbm"}:
        metric_field = "eval_metric" if model_name == "xgboost" else "metric"
        metric = ("rmse" if task == "regression" else
                  "logloss" if model_name == "xgboost" else "binary_logloss")
        if params[metric_field] != metric:
            raise ValueError(
                f"{task} model '{model_name}' requires {metric_field}={metric!r}; "
                f"got {params[metric_field]!r}"
            )
    return dict(params)


def load_model_params(
    path: Path,
    *,
    task: str,
    models: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Load task-specific parameters for exactly the selected models.

    The Super Learner has no parameters of its own: SL7 is combined from the
    base models' predictions.
    """

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

    normalized_models = [str(model).lower() for model in models if str(model).lower() != "super_learner"]
    # Reject retired models before the generic "missing from YAML" path, so a
    # stale panel is told why the model is gone instead of that it is absent.
    for model_name in normalized_models:
        reject_removed_model(model_name)
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


def load_algorithm_version(path: Path) -> str:
    """Load the manually maintained methodological version from the YAML root."""

    params_path = Path(path)
    try:
        with params_path.open(encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid model parameter YAML {params_path}: {exc}") from exc
    version = document.get("algorithm_version") if isinstance(document, dict) else None
    if not isinstance(version, str) or not version.strip():
        raise ValueError(
            f"Model parameter YAML requires a non-empty algorithm_version: {params_path}"
        )
    return version.strip()


def resolved_model_params(
    params: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Record environment overrides and the implemented CV rule in identity."""

    from .mlp_batch_cv import BATCH_CV_RULE
    from .mlp_estimator import BALANCED_BATCH_RULE
    resolved = {
        model_name: _apply_environment_overrides(model_name, model_params)
        for model_name, model_params in params.items()
    }
    for model_params in resolved.values():
        if model_params.get("mlp_batch_size") == "cv":
            model_params["mlp_batch_cv_rule"] = BATCH_CV_RULE
        elif model_params.get("mlp_batch_size") == "balanced":
            model_params["mlp_batch_rule"] = BALANCED_BATCH_RULE
    return resolved


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
    return result


def _select_cv_round(curve, patience=None):
    """First strict minimum, optionally replaying aggregate early stopping."""
    values = np.asarray(curve, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("CV requires a complete finite metric curve")
    best = 0
    for index in range(1, len(values)):
        if values[index] < values[best]:
            best = index
        if patience is not None and patience > 0 and index - best >= patience:
            break
    return best + 1


class XGBoostCVRegressor(BaseEstimator, RegressorMixin):
    """Source-aligned XGBoost: depth 2, eta .3, CV-selected rounds <= 90.

    CV folds are generated from a private ``RandomState`` and passed explicitly
    to ``xgboost.cv``.  This avoids its module-global NumPy RNG, which is shared
    by concurrent cell workers even when both ``seed`` and ``nthread`` are fixed.
    A given ``selected`` number of rounds skips the CV and refits once;
    ``selection_`` holds the rounds used.
    """

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
        preprocessor,
        selected=None,
    ):
        self.seed = seed
        self.n_jobs = n_jobs
        self.objective = objective
        self.eval_metric = eval_metric
        self.max_depth = max_depth
        self.eta = eta
        self.max_rounds = max_rounds
        self.cv_folds = cv_folds
        self.preprocessor = preprocessor
        self.selected = selected

    def fit(self, X, y):
        import xgboost as xgb

        self.params_ = {
            "objective": self.objective,
            "eval_metric": self.eval_metric,
            "max_depth": self.max_depth,
            "eta": self.eta,
            "nthread": self.n_jobs,
            "seed": self.seed,
        }
        self.best_rounds_ = self.selection_ = self._search(X, y) if self.selected is None else int(self.selected)
        self.preprocessor_ = clone(self.preprocessor).fit(X)
        dtrain = xgb.DMatrix(self.preprocessor_.transform(X), label=np.asarray(y))
        self.model_ = xgb.train(
            self.params_, dtrain, num_boost_round=self.best_rounds_
        )
        return self

    def _search(self, X, y):
        import xgboost as xgb

        # ``xgboost.cv`` otherwise calls ``np.random.seed(seed)`` then uses the
        # module-global generator to shuffle folds.  Concurrent callers race on
        # that shared generator, so give it the same legacy MT19937 permutation
        # from an instance-local generator instead.
        fold_order = np.random.RandomState(self.seed).permutation(len(y))
        test_folds = np.array_split(fold_order, self.cv_folds)
        folds = [
            (
                np.concatenate(
                    [test_folds[index] for index in range(self.cv_folds) if index != fold]
                ),
                test_folds[fold],
            )
            for fold in range(self.cv_folds)
        ]
        if len(y) < self.cv_folds:
            raise ValueError("XGBoost requires at least cv_folds training rows")
        curves = []
        for train_index, valid_index in folds:
            process = clone(self.preprocessor).fit(X.iloc[train_index])
            fold_train = xgb.DMatrix(process.transform(X.iloc[train_index]), label=np.asarray(y)[train_index])
            fold_valid = xgb.DMatrix(process.transform(X.iloc[valid_index]), label=np.asarray(y)[valid_index])
            booster = None
            try:
                history = {}
                booster = xgb.train(
                    self.params_, fold_train, num_boost_round=self.max_rounds,
                    evals=[(fold_valid, "valid")], evals_result=history, verbose_eval=False,
                )
                curves.append(history["valid"]["rmse"])
            finally:
                del booster, fold_train, fold_valid
        self.cv_curve_ = np.mean(np.asarray(curves, dtype=float), axis=0)
        return _select_cv_round(self.cv_curve_)

    def predict(self, X):
        import xgboost as xgb

        return self.model_.predict(xgb.DMatrix(self.preprocessor_.transform(X)))


class LightGBMCVRegressor(BaseEstimator, RegressorMixin):
    """LightGBM extension with CV-selected boosting rounds.

    A given ``selected`` number of rounds skips the CV and refits once;
    ``selection_`` holds the rounds used.
    """

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
        preprocessor,
        selected=None,
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
        self.preprocessor = preprocessor
        self.selected = selected

    def fit(self, X, y):
        import lightgbm as lgb

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
        self.best_rounds_ = self.selection_ = self._search(X, y) if self.selected is None else int(self.selected)
        self.preprocessor_ = clone(self.preprocessor).fit(X)
        train = lgb.Dataset(self.preprocessor_.transform(X), label=np.asarray(y))
        self.model_ = lgb.train(
            self.params_, train, num_boost_round=self.best_rounds_
        )
        return self

    def _search(self, X, y):
        import lightgbm as lgb

        if len(y) < self.cv_folds:
            raise ValueError("LightGBM requires at least cv_folds training rows")
        # Preserve 4.6.0 _make_n_folds, including its remainder behavior.
        order = np.random.RandomState(self.seed).permutation(len(y))
        step = len(y) // self.cv_folds
        valid_indices = [order[i:i + step] for i in range(0, len(y), step)]
        curves = []
        for fold in range(self.cv_folds):
            train_index = sorted(np.concatenate([valid_indices[i] for i in range(self.cv_folds) if i != fold]))
            valid_index = sorted(valid_indices[fold])
            process = clone(self.preprocessor).fit(X.iloc[train_index])
            fold_train = lgb.Dataset(process.transform(X.iloc[train_index]), label=np.asarray(y)[train_index])
            fold_valid = lgb.Dataset(process.transform(X.iloc[valid_index]), label=np.asarray(y)[valid_index], reference=fold_train)
            booster = None
            try:
                history = {}
                booster = lgb.train(
                    self.params_, fold_train, num_boost_round=self.max_rounds,
                    valid_sets=[fold_valid], valid_names=["valid"],
                    callbacks=[lgb.record_evaluation(history)],
                )
                curves.append(history["valid"]["rmse"])
            finally:
                del booster, fold_train, fold_valid
        self.cv_curve_ = np.mean(np.asarray(curves, dtype=float), axis=0)
        return _select_cv_round(self.cv_curve_, self.early_stopping_rounds)

    def predict(self, X):
        return np.asarray(self.model_.predict(self.preprocessor_.transform(X)), dtype=float)


class _BoostingCVClassifier(BaseEstimator, ClassifierMixin):
    """Binary twins of the CV regressors: rounds selected on stratified fold log loss.

    As in the regressors, the folds are shuffled with the model seed. Every fold
    refits the preprocessing on its own training rows. The fitted booster
    predicts the positive-class probability. A given ``selected`` number of
    rounds skips the CV and refits once; ``selection_`` holds the rounds used.
    """

    def _fold_curve(self, train_X, train_y, valid_X, valid_y):
        raise NotImplementedError

    def _refit(self, X, y, rounds):
        raise NotImplementedError

    def fit(self, X, y):
        from .fold_local import _rows, classification_preprocessor, stratified_cv_folds

        y = np.asarray(y, dtype=int)
        process = classification_preprocessor(self.preprocessor)
        if self.selected is None:
            curves = []
            for train, valid in stratified_cv_folds(y, self.cv_folds, self._model_name, shuffle_seed=self.seed):
                fitted = clone(process).fit(_rows(X, train))
                curves.append(self._fold_curve(fitted.transform(_rows(X, train)), y[train],
                                               fitted.transform(_rows(X, valid)), y[valid]))
            self.cv_curve_ = np.mean(np.asarray(curves, dtype=float), axis=0)
            self.best_rounds_ = _select_cv_round(self.cv_curve_, getattr(self, "early_stopping_rounds", None))
        else:
            self.best_rounds_ = int(self.selected)
        self.selection_ = self.best_rounds_
        self.preprocessor_ = clone(process).fit(X)
        self.model_ = self._refit(self.preprocessor_.transform(X), y, self.best_rounds_)
        self.classes_ = np.asarray([0, 1])
        return self

    def predict_proba(self, X):
        probability = np.asarray(self._positive(self.preprocessor_.transform(X)), dtype=float)
        return np.column_stack((1. - probability, probability))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= .5).astype(int)


class XGBoostCVClassifier(_BoostingCVClassifier):
    _model_name = "xgboost"

    def __init__(self, seed: int, n_jobs: int, *, objective: str, eval_metric: str,
                 max_depth: int, eta: float, max_rounds: int, cv_folds: int, preprocessor=None,
                 selected=None):
        self.seed = seed
        self.n_jobs = n_jobs
        self.objective = objective
        self.eval_metric = eval_metric
        self.max_depth = max_depth
        self.eta = eta
        self.max_rounds = max_rounds
        self.cv_folds = cv_folds
        self.preprocessor = preprocessor
        self.selected = selected

    def _params(self):
        return {"objective": self.objective, "eval_metric": self.eval_metric, "max_depth": self.max_depth,
                "eta": self.eta, "nthread": self.n_jobs, "seed": self.seed}

    def _fold_curve(self, train_X, train_y, valid_X, valid_y):
        import xgboost as xgb

        history = {}
        xgb.train(self._params(), xgb.DMatrix(train_X, label=train_y), num_boost_round=self.max_rounds,
                  evals=[(xgb.DMatrix(valid_X, label=valid_y), "valid")], evals_result=history,
                  verbose_eval=False)
        return history["valid"][self.eval_metric]

    def _refit(self, X, y, rounds):
        import xgboost as xgb

        return xgb.train(self._params(), xgb.DMatrix(X, label=y), num_boost_round=rounds)

    def _positive(self, X):
        import xgboost as xgb

        return self.model_.predict(xgb.DMatrix(X))


class LightGBMCVClassifier(_BoostingCVClassifier):
    _model_name = "lightgbm"

    def __init__(self, seed: int, n_jobs: int, *, objective: str, metric: str,
                 learning_rate: float, num_leaves: int, min_data_in_leaf: int, verbosity: int,
                 max_rounds: int, cv_folds: int, early_stopping_rounds: int, preprocessor=None,
                 selected=None):
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
        self.preprocessor = preprocessor
        self.selected = selected

    def _params(self):
        return {"objective": self.objective, "metric": self.metric, "learning_rate": self.learning_rate,
                "num_leaves": self.num_leaves, "min_data_in_leaf": self.min_data_in_leaf,
                "num_threads": self.n_jobs, "seed": self.seed, "verbosity": self.verbosity}

    def _fold_curve(self, train_X, train_y, valid_X, valid_y):
        import lightgbm as lgb

        history = {}
        train = lgb.Dataset(train_X, label=train_y)
        lgb.train(self._params(), train, num_boost_round=self.max_rounds,
                  valid_sets=[lgb.Dataset(valid_X, label=valid_y, reference=train)],
                  valid_names=["valid"], callbacks=[lgb.record_evaluation(history)])
        return history["valid"][self.metric]

    def _refit(self, X, y, rounds):
        import lightgbm as lgb

        return lgb.train(self._params(), lgb.Dataset(X, label=y), num_boost_round=rounds)

    def _positive(self, X):
        return self.model_.predict(X)


def _make_classification_model(
    model_name: str,
    seed: int,
    n_jobs: int,
    params: Mapping[str, Any],
    preprocessor,
    selected=None,
):
    """Every classifier refits the typed preprocessing inside each CV fold."""
    from .fold_local import (FoldLocalLogisticLasso, FoldLocalLogisticRidge,
                             FoldLocalMLPClassifier, classification_preprocessor)

    name = model_name.lower()
    process = classification_preprocessor(preprocessor)
    if name == "xgboost":
        return XGBoostCVClassifier(seed=seed, n_jobs=n_jobs, preprocessor=process, selected=selected, **params)
    if name == "lightgbm":
        return LightGBMCVClassifier(seed=seed, n_jobs=n_jobs, preprocessor=process, selected=selected, **params)
    if name == "ols":
        return make_pipeline(
            clone(process),
            StandardScaler(),
            LogisticRegression(**params, random_state=seed),
        )
    if name == "ridge":
        return FoldLocalLogisticRidge(preprocessor=process, selected=selected, **params)
    if name == "lasso":
        return FoldLocalLogisticLasso(preprocessor=process, seed=seed, selected=selected, **params)
    if name == "random_forest":
        return make_pipeline(
            clone(process),
            RandomForestClassifier(**params, n_jobs=n_jobs, random_state=seed),
        )
    if name == "extra_trees":
        return make_pipeline(
            clone(process),
            ExtraTreesClassifier(**params, n_jobs=n_jobs, random_state=seed),
        )
    if name == "shallow_neural_network":
        return FoldLocalMLPClassifier(process, seed, params, selected)
    reject_removed_model(name)
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
    *,
    preprocessor,
    selected=None,
):
    """Construct one base model; every model refits ``preprocessor`` on the rows it is fitted on.

    ``selected`` is a full-sample fit's ``selection_`` (its chosen penalty,
    Lasso ratio, MLP alpha and batch size, or number of rounds); the tuned
    fold-local models then refit at it without searching.
    """

    if task not in {"regression", "classification"}:
        raise ValueError("task must be 'regression' or 'classification'")

    name = model_name.lower()
    reject_removed_model(name)
    if params is None:
        params = load_model_params(
            DEFAULT_MODEL_PARAMS_PATH,
            task=task,
            models=[name],
        )[name]
    resolved_params = _validated_params(task, name, params)
    resolved_params = _apply_environment_overrides(name, resolved_params)

    if task == "classification":
        return _make_classification_model(name, seed, n_jobs, resolved_params, preprocessor, selected)
    from .fold_local import FoldLocalRidge, FoldLocalLasso, FoldLocalMLP
    if name == "ridge":
        return FoldLocalRidge(preprocessor=preprocessor, selected=selected, **resolved_params)
    if name == "lasso":
        return FoldLocalLasso(preprocessor=preprocessor, seed=seed, n_jobs=n_jobs, selected=selected,
                              **resolved_params)
    if name == "shallow_neural_network":
        return FoldLocalMLP(preprocessor, seed, resolved_params, selected)
    if name in {"xgboost", "lightgbm"}:
        constructor = {"xgboost": XGBoostCVRegressor, "lightgbm": LightGBMCVRegressor}[name]
        return constructor(seed=seed, n_jobs=n_jobs, preprocessor=preprocessor, selected=selected,
                           **resolved_params)
    # Models without nested parameter selection fit this chain once.
    if name == "ols":
        estimator = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                  LinearRegression(**resolved_params))
    elif name == "random_forest":
        estimator = make_pipeline(SimpleImputer(strategy="median"),
                                  RandomForestRegressor(**resolved_params, n_jobs=n_jobs, random_state=seed))
    elif name == "extra_trees":
        estimator = make_pipeline(SimpleImputer(strategy="median"),
                                  ExtraTreesRegressor(**resolved_params, n_jobs=n_jobs, random_state=seed))
    else:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: "
            f"{', '.join(SUPPORTED_MODEL_NAMES)}"
        )
    return make_pipeline(clone(preprocessor), estimator)
