from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from aleatoric_nk_grid.nk_grid import NKGridConfig, run_nk_grid

from conftest import write_schema_bundle


REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_SRC = REPO_ROOT / "SMR" / "NK_Grid" / "src"
MODEL_PARAMS = REPO_ROOT / "SMR" / "NK_Grid" / "model_params.yaml"

LEGACY_COMPARISON_COLUMNS = frozenset(
    {
        "dataset",
        "outcome",
        "model",
        "seed",
        "draw",
        "N",
        "K",
        "split_random_state",
        "n_train_total",
        "n_test_total",
        "n_features_total",
        "r2_test",
        "skill_score_pct",
        "rmse",
        "mae",
        "medae",
        "max_error",
        "nrmse",
        "spearman_rho",
        "pearson_r",
        "kendall_tau",
        "ccc",
        "explained_variance",
        "mean_bias",
        "median_bias",
        "pinball_q10",
        "pinball_q90",
        "d2_absolute_error",
        "pinball_q05",
        "pinball_q25",
        "pinball_q50",
        "pinball_q75",
        "pinball_q95",
        "ks_statistic",
        "wasserstein_distance",
        "top_decile_hit_rate",
        "bottom_decile_hit_rate",
        "rsr",
        "cv_rmse",
        "mase",
        "pearson_r2",
        "K_varying",
        "underdetermined",
        "constant_prediction",
        "converged",
        "status",
        "error",
    }
)
ALIGNMENT_COLUMNS = ["dataset", "outcome", "model", "seed", "draw", "N", "K"]


def test_smr_continuous_path_matches_legacy_on_literal_whitelist(tmp_path):
    rng = np.random.default_rng(732)
    predictors = [f"X_{index}" for index in range(5)]
    frame = pd.DataFrame(
        rng.normal(size=(60, len(predictors))), columns=predictors
    )
    frame["y"] = (
        3.0
        + 1.7 * frame["X_0"]
        - 0.8 * frame["X_1"]
        + rng.normal(scale=0.2, size=len(frame))
    )
    data_path = tmp_path / "legacy_input.csv"
    frame.to_csv(data_path, index=False)
    legacy_out = tmp_path / "legacy.csv"
    new_out = tmp_path / "new.csv"
    legacy_script = """
import sys
from pathlib import Path
from NK_Grid.src.nk_grid import NKGridConfig, run_nk_grid
run_nk_grid(NKGridConfig(
    data=Path(sys.argv[1]), out=Path(sys.argv[2]), dataset="synthetic",
    outcome="y", models=("ols", "ridge"), seed=314, test_size=0.3,
    n_seeds=1, n_draws=2, n_sizes_n=2, n_sizes_k=2, min_n=10,
    max_n=30, max_k=4, batch_size=20, n_jobs=1,
    model_params=Path(sys.argv[3]), predictor_prefix=("X_",),
))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO_ROOT / "SMR")
    subprocess.run(
        [
            sys.executable,
            "-c",
            legacy_script,
            str(data_path),
            str(legacy_out),
            str(MODEL_PARAMS),
        ],
        cwd=REPO_ROOT / "SMR",
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    schema = write_schema_bundle(
        tmp_path / "schema_bundle", frame, predictors=predictors
    )
    run_nk_grid(
        NKGridConfig(
            schema=schema,
            out=new_out,
            outcome="y",
            models=("ols", "ridge"),
            seed=314,
            test_size=0.3,
            n_seeds=1,
            n_draws=2,
            n_sizes_n=2,
            n_sizes_k=2,
            min_n=10,
            max_n=30,
            max_k=4,
            batch_size=20,
            n_jobs=1,
            model_params=MODEL_PARAMS,
        )
    )
    old = pd.read_csv(legacy_out)
    new = pd.read_csv(new_out)
    assert LEGACY_COMPARISON_COLUMNS.issubset(old.columns)
    assert LEGACY_COMPARISON_COLUMNS.issubset(new.columns)
    ordered = ALIGNMENT_COLUMNS + sorted(
        LEGACY_COMPARISON_COLUMNS - set(ALIGNMENT_COLUMNS)
    )
    old = old.loc[:, ordered].sort_values(ALIGNMENT_COLUMNS).reset_index(drop=True)
    new = new.loc[:, ordered].sort_values(ALIGNMENT_COLUMNS).reset_index(drop=True)
    assert_frame_equal(
        new,
        old,
        check_exact=False,
        rtol=1e-9,
        atol=1e-12,
        check_dtype=False,
    )
