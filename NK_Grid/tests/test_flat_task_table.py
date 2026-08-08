from __future__ import annotations

import csv
import inspect
import stat
from pathlib import Path
from unittest.mock import patch

import pytest
import aleatoric_nk_grid.flat_task_table as flat_task_table
import aleatoric_nk_grid.nk_grid as nk_grid
import yaml

from aleatoric_nk_grid.flat_task_table import (
    ResourceRequest,
    TaskRow,
    build_rows,
    execution_groups,
    expected_model_keys,
    finalize_chunk_shards,
    pack_lpt,
    pending_rows,
    read_chunk,
    resource_class_for_rows,
    resource_request,
    measure_chunk_read_rss,
    run_chunk,
    sbatch_resource_args,
    run_snapshot_chunk,
    write_chunk_snapshot,
    write_task_table,
    write_synthetic_task_table,
)
from aleatoric_nk_grid.nk_grid import NKGridConfig
from aleatoric_nk_grid.seed_shards import (
    build_chunk_finalizer_map,
    finalize_chunk_shards as finalize_chunk_shards_via_seed_shards,
)
from conftest import write_schema_bundle
import numpy as np
import pandas as pd


def _config(tmp_path: Path) -> NKGridConfig:
    return NKGridConfig(
        schema=tmp_path / "schema.json", out=tmp_path / "result.csv", outcome="y",
        models=("ols", "ridge", "lightgbm", "super_learner"), seed=11,
        test_size=0.3, n_seeds=2, n_draws=2, n_sizes_n=1, n_sizes_k=1,
        max_n=20, max_k=2, batch_size=4, n_jobs=1,
    )


def _fast_model_params(tmp_path: Path) -> Path:
    source = Path(__file__).resolve().parents[1] / "model_params.yaml"
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    regression = payload["regression"]
    regression["lightgbm"].update({"max_rounds": 2, "cv_folds": 2})
    regression["xgboost"].update({"max_rounds": 2, "cv_folds": 2})
    regression["super_learner"].update({
        "cv": 2, "n_estimators": 1, "max_iter": 2, "lgbm_n_estimators": 1,
    })
    path = tmp_path / "fast-model-params.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_task_table_is_reproducible_and_has_one_row_group_per_chunk(tmp_path):
    rows = build_rows(_config(tmp_path), n_grid=(10, 20), k_grid=(1, 2))
    first = pack_lpt(rows, budget=2)
    second = pack_lpt(build_rows(_config(tmp_path), n_grid=(10, 20), k_grid=(1, 2)), budget=2)
    assert first == second
    path = write_task_table(tmp_path / "tasks.parquet", first)
    loaded = tuple(row for chunk in range(max(row.chunk_id for row in first) + 1) for row in read_chunk(path, chunk))
    assert loaded == first
    assert all(len(row.models) >= 1 for row in loaded)
    assert all("ols" not in row.models or "ridge" in row.models for row in loaded if row.group == "imputed_core")
    assert not path.stat().st_mode & stat.S_IWUSR
    assert "sqlite" not in inspect.getsource(flat_task_table).lower()


def test_rss_harness_streams_synthetic_design_and_reads_one_chunk(tmp_path):
    path = write_synthetic_task_table(
        tmp_path / "synthetic.parquet", row_count=5_000, rows_per_chunk=1_000,
    )
    measured = measure_chunk_read_rss(path, chunk_id=2)
    assert measured["rows"] == 1_000
    assert measured["rss_delta_bytes"] >= 0


def test_chunk_read_rss_uses_the_engine_peak_rss_conversion():
    assert flat_task_table._process_peak_rss_bytes is nk_grid._process_peak_rss_bytes


def test_lpt_obeys_budget_except_single_over_budget_rows():
    rows = tuple(TaskRow(str(index), 1, 0, 10, index + 1, "imputed_core", ("ols",)) for index in range(4))
    costs = {"0": 6.0, "1": 4.0, "2": 3.0, "3": 2.0}
    packed = pack_lpt(rows, budget=5, cost_function=lambda row: costs[row.row_id])
    totals: dict[int, float] = {}
    counts: dict[int, int] = {}
    for row in packed:
        totals[row.chunk_id] = totals.get(row.chunk_id, 0.0) + row.est_cost
        counts[row.chunk_id] = counts.get(row.chunk_id, 0) + 1
    assert all(total <= 5 or counts[chunk] == 1 for chunk, total in totals.items())


def test_super_learner_split_is_threshold_controlled_without_default():
    models = ("ols", "super_learner", "lightgbm")
    assert execution_groups(models, k_features=100) == (
        ("imputed_core", ("ols", "super_learner")), ("passthrough", ("lightgbm",)),
    )
    assert execution_groups(models, k_features=100, split_super_learner_min_k=100) == (
        ("imputed_core", ("ols",)), ("super_learner", ("super_learner",)),
        ("passthrough", ("lightgbm",)),
    )


def test_resume_is_key_set_difference_not_table_position():
    rows = (
        TaskRow("a", 1, 0, 10, 1, "imputed_core", ("ols", "ridge")),
        TaskRow("b", 2, 0, 10, 1, "imputed_core", ("ols",)),
    )
    assert pending_rows(rows, {("ols", 1, 0, 10, 1), ("ridge", 1, 0, 10, 1)}) == (rows[1],)


def test_finalizer_rejects_out_of_design_and_duplicate_keys(tmp_path):
    rows = pack_lpt((
        TaskRow("a", 1, 0, 10, 1, "imputed_core", ("ols",)),
        TaskRow("b", 2, 0, 10, 1, "imputed_core", ("ols",)),
    ), budget=1)
    table = write_task_table(tmp_path / "tasks.parquet", rows)
    chunks: dict[int, Path] = {}
    for chunk_id in (0, 1):
        path = tmp_path / f"chunk-{chunk_id}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["model", "seed", "draw", "N", "K", "status"])
            writer.writeheader()
            row = read_chunk(table, chunk_id)[0]
            writer.writerow({"model": "ols", "seed": row.seed, "draw": row.draw, "N": row.n_samples, "K": row.k_features, "status": "ok"})
        chunks[chunk_id] = path
    output = finalize_chunk_shards(table, chunks, tmp_path / "merged.csv")
    assert len(list(csv.DictReader(output.open(encoding="utf-8")))) == len(expected_model_keys(rows))
    with chunks[0].open("a", encoding="utf-8") as handle:
        handle.write("not-a-model,1,0,10,1,ok\n")
    with pytest.raises(ValueError, match="out-of-design"):
        finalize_chunk_shards(table, chunks, tmp_path / "bad.csv")


def test_resource_framework_requires_stage_b_values():
    serial = (TaskRow("a", 1, 0, 10, 1, "imputed_core", ("ols",)),)
    super_rows = (TaskRow("b", 1, 0, 10, 1, "super_learner", ("super_learner",)),)
    assert resource_class_for_rows(serial) == "serial"
    assert resource_class_for_rows(super_rows) == "super_learner"
    with pytest.raises(ValueError, match="No Stage-B"):
        resource_request(serial, {})
    values = {
        "serial": ResourceRequest("serial", 1, "long", "8G", "00:30:00"),
        "super_learner": ResourceRequest("super_learner", 2, "long", "16G", "00:30:00"),
    }
    assert resource_request(serial, values).cpus_per_task == 1
    assert sbatch_resource_args(serial, values) == (
        "--partition=long", "--cpus-per-task=1", "--mem=8G", "--time=00:30:00",
    )
    assert resource_request(super_rows, values).cpus_per_task == 2
    assert "--cpus-per-task=2" in sbatch_resource_args(super_rows, values)


@pytest.mark.parametrize(
    ("models", "expected_group", "n_jobs"),
    [
        (("ols", "ridge"), "imputed_core", 1),
        (("lightgbm", "xgboost"), "passthrough", 1),
        (("super_learner",), "imputed_core", 2),
    ],
    ids=("imputed", "passthrough", "super-learner-isolated"),
)
def test_chunk_execution_matches_direct_cell_group_metrics(
    tmp_path, models, expected_group, n_jobs,
):
    values = np.arange(72, dtype=float)
    frame = pd.DataFrame({
        "y": values * 2 + values % 5,
        "x1": values,
        "x2": values % 3,
        "x3": values % 7,
    })
    schema = write_schema_bundle(
        tmp_path / "input", frame, predictors=["x1", "x2", "x3"],
        imputation={
            "continuous": "median", "ordinal": "most_frequent",
            "onehot_group": "atomic_mode",
            "model_overrides": {"lightgbm": "passthrough", "xgboost": "passthrough"},
        },
    )
    config = NKGridConfig(
        schema=schema, out=tmp_path / "direct.csv", outcome="y", models=models,
        seed=17, test_size=0.25, n_seeds=1, n_draws=1, n_sizes_n=1, n_sizes_k=1,
        max_n=12, max_k=1, min_n=10, batch_size=2, n_jobs=n_jobs,
        repeat_plan=((17, 0),), n_grid=(10, 12), k_grid=(1,), rerun_completed=False,
        model_params=_fast_model_params(tmp_path),
    )
    rows = pack_lpt(build_rows(config, n_grid=(10, 12), k_grid=(1,)), budget=1)
    assert {row.group for row in rows} == {expected_group}
    assert len({row.chunk_id for row in rows}) >= 2
    table = write_task_table(tmp_path / "tasks.parquet", rows)
    chunk_outputs = {
        chunk_id: run_chunk(table, chunk_id, config, output=tmp_path / f"chunk-{chunk_id}.csv")
        for chunk_id in range(2)
    }
    chunk_output = finalize_chunk_shards(table, chunk_outputs, tmp_path / "chunk-final.csv")
    from aleatoric_nk_grid.nk_grid import METRIC_COLUMNS, run_nk_grid
    if models == ("super_learner",):
        native_calls: list[dict] = []
        from aleatoric_nk_grid import nk_grid
        original = nk_grid._run_native_model_cell_locked

        def observe(*args, **kwargs):
            native_calls.append(dict(kwargs["fit_arguments"]))
            return original(*args, **kwargs)

        with patch("aleatoric_nk_grid.nk_grid._run_native_model_cell_locked", side_effect=observe):
            run_nk_grid(config)
        assert native_calls
        assert {call["model_n_jobs"] for call in native_calls} == {2}
    else:
        run_nk_grid(config)
    sort_columns = ["model", "seed", "draw", "N", "K"]
    left = pd.read_csv(chunk_output).sort_values(sort_columns).reset_index(drop=True)
    right = pd.read_csv(config.out).sort_values(sort_columns).reset_index(drop=True)
    pd.testing.assert_frame_equal(left.loc[:, [*sort_columns, *METRIC_COLUMNS]], right.loc[:, [*sort_columns, *METRIC_COLUMNS]], check_exact=True)


def test_snapshot_freezes_chunk_array_mapping(tmp_path):
    config = _config(tmp_path)
    table = write_task_table(tmp_path / "tasks.parquet", pack_lpt(
        build_rows(config, n_grid=(10,), k_grid=(1,)), budget=1
    ))
    snapshot = write_chunk_snapshot(
        tmp_path / "snapshot.json", table_path=table, panel="panel", config=config,
        output_dir=tmp_path / "chunks",
    )
    import json
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    assert payload["chunk_count"] == 8
    assert payload["task_table"] == str(table.resolve())
    with pytest.raises(IndexError, match="outside"):
        run_snapshot_chunk(snapshot, 8)


def test_seed_shard_finalizer_map_uses_chunk_id_targets(tmp_path):
    config = _config(tmp_path)
    table = write_task_table(tmp_path / "tasks.parquet", pack_lpt(
        build_rows(config, n_grid=(10,), k_grid=(1,)), budget=1
    ))
    snapshot = write_chunk_snapshot(
        tmp_path / "snapshot.json", table_path=table, panel="panel", config=config,
        output_dir=tmp_path / "chunks",
    )
    finalizer_map = build_chunk_finalizer_map(snapshot)
    import json
    payload = json.loads(finalizer_map.read_text(encoding="utf-8"))
    assert payload["kind"] == "chunk-finalizer"
    assert payload["targets"][0]["chunk_count"] == 8
