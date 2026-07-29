from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from aleatoric_nk_grid.nk_grid import NKGridConfig, run_nk_grid

from conftest import write_schema_bundle


MODEL_PARAMS = Path(__file__).resolve().parents[1] / "model_params.yaml"


def test_external_typed_manifest_runs_k_by_source(tmp_path):
    train_n = 40
    test_n = 12
    x_train = np.arange(train_n, dtype=float)
    x_test = np.arange(test_n, dtype=float) + 50
    train_category = np.arange(train_n) % 2
    test_category = np.arange(test_n) % 2
    train = pd.DataFrame(
        {
            "id": np.arange(train_n),
            "y": 1.5 * x_train + train_category,
            "X_value": x_train,
            "C_zero": (train_category == 0).astype(float),
            "C_one": (train_category == 1).astype(float),
        }
    )
    test = pd.DataFrame(
        {
            "id": 100 + np.arange(test_n),
            "y": 1.5 * x_test + test_category,
            "X_value": x_test,
            "C_zero": (test_category == 0).astype(float),
            "C_one": (test_category == 1).astype(float),
        }
    )
    train.loc[[2, 3], ["C_zero", "C_one"]] = np.nan
    test.loc[[1], ["C_zero", "C_one"]] = np.nan
    manifest = pd.DataFrame(
        {
            "source_column": ["value", "category", "category"],
            "feature_name": ["X_value", "C_zero", "C_one"],
            "keep": [True, True, True],
            "source_order": [0, 1, 1],
            "feature_order": [0, 0, 1],
            "unit_type": ["continuous", "onehot_group", "onehot_group"],
            "drop_first": [False, False, False],
            "is_reference": [False, True, False],
            "reference_level": [np.nan, 0, 0],
            "level_value": [np.nan, 0, 1],
            "ordinal_levels": [np.nan, np.nan, np.nan],
            "source_prior": [0.0, np.nan, np.nan],
        }
    )
    schema = write_schema_bundle(
        tmp_path / "input",
        train,
        test=test,
        split_mode="external_test",
        predictors=["X_value", "C_zero", "C_one"],
        manifest=manifest,
        id_column="id",
    )
    out = tmp_path / "result.csv"
    run_nk_grid(
        NKGridConfig(
            schema=schema,
            out=out,
            outcome="y",
            models=("ols",),
            seed=123,
            test_size=0.3,
            n_seeds=1,
            n_draws=1,
            n_sizes_n=1,
            n_sizes_k=1,
            min_n=10,
            max_n=0,
            max_k=0,
            batch_size=10,
            n_jobs=1,
            model_params=MODEL_PARAMS,
        )
    )
    row = pd.read_csv(out).iloc[0]
    assert row["status"] == "ok"
    assert row["K"] == 2
    assert row["K_expanded"] == 3
    assert row["n_features_total"] == 2
    assert row["n_expanded_features_total"] == 3
    assert row["K_unobserved"] == 0
    assert pd.read_csv(out)["K_unobserved"].groupby(pd.read_csv(out)["N"]).mean().iloc[0] == 0
