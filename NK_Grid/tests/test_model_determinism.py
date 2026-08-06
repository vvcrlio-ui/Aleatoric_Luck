from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import pytest

from aleatoric_nk_grid.model_registry import (
    SUPPORTED_MODEL_NAMES,
    load_model_params,
)
from aleatoric_nk_grid.nk_grid import (
    NKGridConfig,
    _fit_predict_model_cell,
    run_nk_grid,
)

from conftest import write_schema_bundle


MODEL_PARAMS = Path(__file__).resolve().parents[1] / "model_params.yaml"
REPEATS = 24


def _fast_model_params(model_name: str, task: str) -> dict:
    """Return small test-only parameters without bypassing model code paths."""

    params = deepcopy(
        load_model_params(
            MODEL_PARAMS, task=task, models=[model_name]
        )[model_name]
    )
    if model_name in {"ridge", "lasso"}:
        if task == "regression":
            params.update(n_alphas=2, max_cv_folds=2)
        if model_name == "lasso" or task == "classification":
            params["max_iter"] = 100
    elif model_name in {"random_forest", "extra_trees"}:
        params.update(n_estimators=8, min_samples_leaf=1)
    elif model_name == "xgboost":
        if task == "regression":
            # Keep >1 rounds so xgboost.cv still constructs and evaluates folds.
            params.update(max_rounds=60, cv_folds=2)
        else:
            params["n_estimators"] = 3
    elif model_name == "lightgbm":
        if task == "regression":
            params.update(
                max_rounds=200,
                cv_folds=2,
                early_stopping_rounds=1,
                min_data_in_leaf=2,
            )
        else:
            params.update(n_estimators=3, min_data_in_leaf=2)
    elif model_name == "shallow_neural_network":
        params.update(
            hidden_layer_sizes=[4],
            max_iter=30,
        )
        if task == "regression":
            # Both stay >1 to exercise AdaptiveMLPRegressor's CV selection.
            params.update(n_alphas=2, max_cv_folds=2)
    elif model_name == "super_learner":
        # Retain the same base-estimator family and the stacking CV pathway.
        params.update(
            cv=2,
            n_estimators=8,
            min_samples_leaf=1,
            hidden_layer_sizes=[4],
            max_iter=30,
            lgbm_n_estimators=5,
            lgbm_min_data_in_leaf=2,
        )
    return params


def _data(n_samples: int, task: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    rng = np.random.RandomState(20260804 + n_samples)
    X = pd.DataFrame(
        rng.normal(size=(n_samples, 5)), columns=[f"x{index}" for index in range(5)]
    )
    X.loc[X.index[::11], "x3"] = np.nan
    signal = 1.25 * X["x0"].fillna(0.0) - 0.75 * X["x1"].fillna(0.0)
    if task == "classification":
        # Strict alternation guarantees at least two observations per class at
        # N=10, which is the Super Learner's smallest valid CV case.
        y = pd.Series(np.arange(n_samples) % 2)
    else:
        y = pd.Series(signal + rng.normal(scale=0.05, size=n_samples))
    return X, y, X.iloc[: min(7, n_samples)].copy()


def _fit_repeatedly(
    *,
    model_name: str,
    task: str,
    n_samples: int,
    outer_n_jobs: int,
) -> list[dict]:
    X_train, y_train, X_test = _data(n_samples, task)
    params = _fast_model_params(model_name, task)
    return Parallel(n_jobs=outer_n_jobs, prefer="threads")(
        delayed(_fit_predict_model_cell)(
            model_name=model_name,
            model_seed=918,
            model_n_jobs=1,
            task=task,
            params=params,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
        )
        for _ in range(REPEATS)
    )


@pytest.mark.parametrize("task", ("regression", "classification"))
@pytest.mark.parametrize("n_samples", (10, 20, 200))
@pytest.mark.parametrize("model_name", SUPPORTED_MODEL_NAMES)
def test_all_registered_models_are_bitwise_deterministic(
    model_name: str,
    task: str,
    n_samples: int,
):
    """All registry models are exact across serial and threaded cell execution."""

    for outer_n_jobs in (1, 2, 4, 8):
        results = _fit_repeatedly(
            model_name=model_name,
            task=task,
            n_samples=n_samples,
            outer_n_jobs=outer_n_jobs,
        )
        reference = results[0]
        assert all(
            np.array_equal(result["predictions"], reference["predictions"])
            for result in results[1:]
        ), (
            f"{model_name=} {task=} {n_samples=} {outer_n_jobs=} produced "
            "non-identical predictions"
        )
        reference_rounds = reference["best_rounds"]
        if np.isnan(reference_rounds):
            assert all(np.isnan(result["best_rounds"]) for result in results[1:])
        else:
            assert all(
                result["best_rounds"] == reference_rounds
                for result in results[1:]
            )


def _toy_panel(rows: int = 48) -> pd.DataFrame:
    values = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "y": 1.75 * values + (values % 3),
            "X_a": values % 7,
            "X_b": (values * 3) % 11,
        }
    )


def test_xgboost_toy_panel_final_csv_is_bitwise_identical_across_five_runs(
    tmp_path,
):
    schema = write_schema_bundle(
        tmp_path / "input", _toy_panel(), predictors=["X_a", "X_b"]
    )
    outputs = []
    for index in range(5):
        out = tmp_path / f"xgboost-{index}.csv"
        run_nk_grid(
            NKGridConfig(
                schema=schema,
                out=out,
                outcome="y",
                models=("xgboost",),
                seed=321,
                test_size=0.25,
                n_seeds=1,
                n_draws=1,
                n_sizes_n=1,
                n_sizes_k=1,
                min_n=10,
                max_n=10,
                max_k=1,
                batch_size=20,
                n_jobs=8,
                model_params=MODEL_PARAMS,
            ),
            exact_output_path=True,
            execution_pairs=((321, 0),),
        )
        outputs.append(out)

    reference = outputs[0].read_bytes()
    assert all(path.read_bytes() == reference for path in outputs[1:])
    manifest = json.loads(
        outputs[0].with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["design"]["parallelism"]["configured_outer_n_jobs"] == 1
    assert manifest["design"]["parallelism"]["chunk_policy"]["n_jobs_rule"] == (
        "one array worker executes its groups serially"
    )
