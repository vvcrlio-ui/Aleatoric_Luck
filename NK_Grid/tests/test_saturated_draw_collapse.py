"""Regression coverage for collapsing draw-order artifacts at saturated cells."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import aleatoric_nk_grid.nk_grid as nk_grid
from aleatoric_nk_grid.experiment import manifest_path
from aleatoric_nk_grid.flat_task_table import (
    build_rows,
    expected_model_keys,
    finalize_chunk_shards,
    pack_lpt,
    run_chunk,
    write_task_table,
)
from aleatoric_nk_grid.nk_grid import (
    METRIC_COLUMNS,
    NKGridConfig,
    draws_are_degenerate,
    enumerate_jobs,
    estimate_run_size,
    pairs_for_point,
    run_nk_grid,
    split_frame,
)
from conftest import write_schema_bundle


@pytest.mark.parametrize(
    ("n_samples", "k_features", "n_total", "k_total", "expected"),
    [
        (8, 4, 8, 4, True),
        (9, 5, 8, 4, True),
        (7, 4, 8, 4, False),
        (8, 3, 8, 4, False),
    ],
)
def test_draw_degeneracy_requires_both_saturated_dimensions(
    n_samples, k_features, n_total, k_total, expected,
):
    assert draws_are_degenerate(
        n_samples,
        k_features,
        n_train_total=n_total,
        n_feature_units=k_total,
    ) is expected


def test_saturated_point_keeps_the_smallest_draw_for_every_seed():
    pairs = ((22, 5), (11, 4), (22, 1), (11, 2))
    assert pairs_for_point(
        pairs, 8, 4, n_train_total=8, n_feature_units=4,
    ) == ((11, 2), (22, 1))
    # A grid point below either real population bound remains byte-for-byte
    # equivalent as a sequence; this is not generic duplicate-row removal.
    assert pairs_for_point(
        pairs, 7, 4, n_train_total=8, n_feature_units=4,
    ) == pairs


def test_no_point_collapses_when_the_grid_does_not_reach_the_real_totals():
    pairs = ((11, 0), (11, 2), (22, 0), (22, 2))
    n_grid = (3, 7)
    k_grid = (1, 4)
    jobs = enumerate_jobs(
        ("ols",), pairs, n_grid, k_grid,
        n_train_total=8, n_feature_units=4,
    )
    assert len(jobs) == len(pairs) * len(n_grid) * len(k_grid)
    assert all(
        pairs_for_point(
            pairs, n, k, n_train_total=8, n_feature_units=4,
        ) == pairs
        for n in n_grid for k in k_grid
    )


def test_direct_job_enumeration_keeps_the_legacy_relative_order():
    models = ("ols", "ridge")
    pairs = ((9, 3), (2, 4), (9, 1), (2, 0))
    n_grid = (5, 8)
    k_grid = (1, 4)
    old_order = [
        (model, seed, draw, n, k)
        for seed, draw in pairs
        for k in k_grid
        for n in n_grid
        for model in models
    ]
    actual = enumerate_jobs(
        models, pairs, n_grid, k_grid, n_train_total=8, n_feature_units=4,
    )
    assert actual == [job for job in old_order if job in set(actual)]


def _saturated_fixture(tmp_path: Path, *, out_name: str, models: tuple[str, ...]):
    values = np.arange(24, dtype=float)
    predictors = ["x1", "x2", "x3", "x4"]
    frame = pd.DataFrame({
        "y": values * 1.7 + values % 5,
        "x1": values,
        "x2": values % 3,
        "x3": values % 7,
        "x4": (values * 2) % 11,
    })
    schema = write_schema_bundle(tmp_path / "input", frame, predictors=predictors)
    train_total = len(split_frame(
        frame, predictors, "y", test_size=0.25, seed=17, task="regression",
    ).X_train)
    feature_total = len(predictors)
    model_params = Path(__file__).resolve().parents[1] / "model_params.yaml"
    if "random_forest" in models:
        payload = yaml.safe_load(model_params.read_text(encoding="utf-8"))
        payload["regression"]["random_forest"]["n_estimators"] = 3
        model_params = tmp_path / "fast-model-params.yaml"
        model_params.write_text(yaml.safe_dump(payload), encoding="utf-8")
    config = NKGridConfig(
        schema=schema,
        out=tmp_path / out_name,
        outcome="y",
        models=models,
        seed=17,
        test_size=0.25,
        n_seeds=1,
        n_draws=1,
        n_sizes_n=1,
        n_sizes_k=1,
        max_n=train_total,
        max_k=feature_total,
        batch_size=2,
        n_jobs=1,
        repeat_plan=((17, 2), (17, 0), (23, 2), (23, 0)),
        n_grid=(10, train_total),
        k_grid=(1, feature_total),
        rerun_completed=False,
        model_params=model_params,
    )
    return config, train_total, feature_total


def _key_set(frame: pd.DataFrame) -> set[tuple[str, int, int, int, int]]:
    return {
        (str(row.model), int(row.seed), int(row.draw), int(row.N), int(row.K))
        for row in frame.loc[:, ["model", "seed", "draw", "N", "K"]].itertuples(index=False)
    }


def test_saturated_draw_collapse_preserves_survivor_metrics_and_manifest(tmp_path, monkeypatch):
    before, train_total, feature_total = _saturated_fixture(
        tmp_path / "before", out_name="before.csv", models=("ols", "random_forest"),
    )
    # This emulates the pre-change design only.  It deliberately leaves the
    # fitting path untouched so metrics on surviving keys are a true comparison.
    monkeypatch.setattr(
        nk_grid,
        "pairs_for_point",
        lambda repeat_pairs, n_samples, k_features, **kwargs: tuple(repeat_pairs),
    )
    before_out = run_nk_grid(before)
    monkeypatch.undo()

    after = replace(before, out=tmp_path / "after" / "after.csv")
    after_out = run_nk_grid(after)
    before_frame = pd.read_csv(before_out)
    after_frame = pd.read_csv(after_out)
    before_keys = _key_set(before_frame)
    after_keys = _key_set(after_frame)
    dropped = before_keys - after_keys
    expected_dropped = {
        (model, seed, draw, train_total, feature_total)
        for model in after.models
        for seed in (17, 23)
        for draw in (2,)
    }
    assert after_keys < before_keys
    assert dropped == expected_dropped
    assert all((key[3], key[4]) == (train_total, feature_total) for key in dropped)

    key_columns = ["model", "seed", "draw", "N", "K"]
    left = before_frame[before_frame.apply(
        lambda row: (row["model"], int(row["seed"]), int(row["draw"]), int(row["N"]), int(row["K"])) in after_keys,
        axis=1,
    )].sort_values(key_columns).reset_index(drop=True)
    right = after_frame.sort_values(key_columns).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        left.loc[:, [*key_columns, *METRIC_COLUMNS]],
        right.loc[:, [*key_columns, *METRIC_COLUMNS]],
        check_exact=True,
    )
    assert estimate_run_size(
        after,
        n_grid=after.n_grid,
        k_grid=after.k_grid,
        n_train_total=train_total,
        n_feature_units=feature_total,
    )["expected_output_rows"] == len(after_frame)

    collapsed_manifest = json.loads(manifest_path(after_out).read_text(encoding="utf-8"))
    assert collapsed_manifest["execution"]["expected_rows"] == len(after_frame)
    assert collapsed_manifest["draw_collapse"] == {
        "n_train_total": train_total,
        "n_feature_units": feature_total,
        "collapsed_points": [{
            "n": train_total, "k": feature_total,
            "pairs_before": 4, "pairs_after": 2,
        }],
        "rows_dropped": 4,
    }

    unsaturated = replace(
        after,
        out=tmp_path / "unsaturated" / "result.csv",
        models=("ols",),
        n_grid=(10, train_total - 1),
    )
    unsaturated_out = run_nk_grid(unsaturated)
    unsaturated_manifest = json.loads(manifest_path(unsaturated_out).read_text(encoding="utf-8"))
    assert unsaturated_manifest["draw_collapse"] == {
        "n_train_total": train_total,
        "n_feature_units": feature_total,
        "collapsed_points": [],
        "rows_dropped": 0,
    }


def test_flat_table_and_direct_design_match_and_finalize_after_collapse(tmp_path):
    config, train_total, feature_total = _saturated_fixture(
        tmp_path, out_name="direct.csv", models=("ols",),
    )
    jobs = enumerate_jobs(
        config.models,
        config.repeat_plan or (),
        config.n_grid or (),
        config.k_grid or (),
        n_train_total=train_total,
        n_feature_units=feature_total,
    )
    rows = pack_lpt(build_rows(
        config,
        n_grid=config.n_grid or (),
        k_grid=config.k_grid or (),
        n_train_total=train_total,
        n_feature_units=feature_total,
    ), budget=1)
    assert set(jobs) == expected_model_keys(rows)

    table = write_task_table(tmp_path / "tasks.parquet", rows)
    outputs = {
        chunk_id: run_chunk(table, chunk_id, config, output=tmp_path / f"chunk-{chunk_id}.csv")
        for chunk_id in range(max(row.chunk_id for row in rows) + 1)
    }
    final = finalize_chunk_shards(table, outputs, tmp_path / "final.csv")
    direct = run_nk_grid(config)
    key_columns = ["model", "seed", "draw", "N", "K"]
    left = pd.read_csv(final).sort_values(key_columns).reset_index(drop=True)
    right = pd.read_csv(direct).sort_values(key_columns).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        left.loc[:, [*key_columns, *METRIC_COLUMNS]],
        right.loc[:, [*key_columns, *METRIC_COLUMNS]],
        check_exact=True,
    )
