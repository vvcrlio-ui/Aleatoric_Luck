from __future__ import annotations

import csv
import json
import stat
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import aleatoric_nk_grid.flat_task_table as ft
from aleatoric_nk_grid.flat_task_table import (
    ResourceRequest,
    TaskRow,
    assign_rows_modulo,
    build_rows,
    classify_attempts,
    expected_model_keys,
    finalize_slice_shards,
    prepare_round,
    read_row_group,
    read_task_table,
    run_slice,
    sbatch_resource_args,
    verify_rounds,
    write_task_table,
    write_work_snapshot,
)
from aleatoric_nk_grid.nk_grid import NKGridConfig


def _config(tmp_path: Path, *, models: tuple[str, ...] = ("ols",)) -> NKGridConfig:
    return NKGridConfig(
        schema=tmp_path / "schema.json", out=tmp_path / "unused.csv", outcome="y", models=models,
        seed=11, test_size=0.3, n_seeds=1, n_draws=1, n_sizes_n=1, n_sizes_k=1,
        max_n=20, max_k=2, batch_size=4, n_jobs=8, repeat_plan=((11, 0),),
    )


def _fake_run(config: NKGridConfig, **_: object) -> None:
    config.out.parent.mkdir(parents=True, exist_ok=True)
    with config.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "seed", "draw", "N", "K", "status", "metric"])
        writer.writeheader()
        for model in config.models:
            writer.writerow({"model": model, "seed": config.repeat_plan[0][0], "draw": config.repeat_plan[0][1], "N": config.n_grid[0], "K": config.k_grid[0], "status": "ok", "metric": f"{config.n_grid[0]}:{config.k_grid[0]}"})


def _snapshot(tmp_path: Path, *, workers: int = 2, n_grid: tuple[int, ...] = (10, 12), k_grid: tuple[int, ...] = (1, 2)) -> tuple[Path, tuple[TaskRow, ...]]:
    config = _config(tmp_path)
    rows = build_rows(config, n_grid=n_grid, k_grid=k_grid)
    table = write_task_table(tmp_path / "tasks.parquet", rows, rows_per_group=2)
    snapshot = write_work_snapshot(tmp_path / "snapshot.json", table_path=table, panel="test", config=config, output_dir=tmp_path / "outputs", workers=workers)
    return snapshot, rows


def test_v2_table_is_immutable_cost_free_and_streamed(tmp_path):
    snapshot, rows = _snapshot(tmp_path, workers=3)
    table = Path(json.loads(snapshot.read_text())["task_table"])
    source = pq.ParquetFile(table)
    assert source.schema.names == ["row_id", "seed", "draw", "N", "K", "group", "element"] or source.schema.names == ["row_id", "seed", "draw", "N", "K", "group", "models"]
    assert all(column not in source.schema.names for column in ("est_cost", "chunk_id"))
    assert read_task_table(table) == tuple(sorted(rows, key=lambda row: (row.k_features, row.n_samples, row.seed, row.draw, row.group, row.row_id)))
    assert not table.stat().st_mode & stat.S_IWUSR


def test_v1_table_is_rejected_fail_closed(tmp_path):
    path = tmp_path / "old.parquet"
    pq.write_table(pa.table({"row_id": ["a"], "seed": [1], "draw": [0], "N": [10], "K": [1], "group": ["imputed_core"], "models": [["ols"]], "est_cost": [1.0], "chunk_id": [0]}), path)
    with pytest.raises(ValueError, match="v1"):
        read_task_table(path)


@pytest.mark.parametrize("workers,row_count", [(1, 5), (2, 5), (7, 5), (4, 0)])
def test_prepare_round_assigns_every_todo_row_by_index_modulo(tmp_path, workers, row_count):
    snapshot, rows = _snapshot(tmp_path, workers=workers, n_grid=(10, 12, 14, 16, 18), k_grid=(1,))
    if row_count:
        rows = rows[:row_count]
        table = write_task_table(tmp_path / "short.parquet", rows)
        payload = json.loads(snapshot.read_text()); payload["task_table"] = str(table); Path(snapshot).chmod(0o644); Path(snapshot).write_text(json.dumps(payload)); Path(snapshot).chmod(0o444)
    else:
        # Mark the only model key of every row complete without manufacturing a
        # special empty main table, which production deliberately forbids.
        output = tmp_path / "outputs" / "round-0"; output.mkdir(parents=True)
        with (output / "worker-0.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["model", "seed", "draw", "N", "K", "status"]); writer.writeheader()
            for row in rows: writer.writerow({"model": "ols", "seed": row.seed, "draw": row.draw, "N": row.n_samples, "K": row.k_features, "status": "ok"})
    stats = prepare_round(snapshot, round_index=1)
    assignment = Path(stats["assignment"])
    groups = [read_row_group(assignment, index) for index in range(workers)]
    assigned = [row for group in groups for row in group]
    assert len(assigned) == len({row.row_id for row in assigned})
    expected = [] if row_count == 0 else list(rows)
    assert {row.row_id for row in assigned} == {row.row_id for row in expected}
    assert all(tuple(row.row_id for row in group) == tuple(row.row_id for row in expected[index::workers]) for index, group in enumerate(groups))


def test_attempt_classification_keeps_crash_and_too_long_separate():
    crashed, too_long = classify_attempts([
        {"round": 1, "worker_index": 0, "sequence": 0, "row_id": "crashed"},
        {"round": 1, "worker_index": 0, "sequence": 1, "row_id": "later"},
        {"round": 1, "worker_index": 1, "sequence": 0, "row_id": "long"},
        {"round": 2, "worker_index": 1, "sequence": 0, "row_id": "long"},
        {"round": 3, "worker_index": 1, "sequence": 0, "row_id": "long"},
    ])
    assert crashed == {"crashed"}
    assert too_long == {"long"}


@pytest.mark.parametrize("workers,row_count", [(workers, rows) for workers in range(1, 11) for rows in (0, 1, 2, 7, 19)])
def test_modulo_assignment_is_complete_balanced_and_not_contiguous(workers, row_count):
    rows = tuple(TaskRow(str(index), 1, 0, 10, index + 1, "imputed_core", ("ols",)) for index in range(row_count))
    groups = assign_rows_modulo(rows, workers)
    assert len(groups) == workers
    assert {row.row_id for group in groups for row in group} == {row.row_id for row in rows}
    assert max((len(group) for group in groups), default=0) - min((len(group) for group in groups), default=0) <= 1
    assert all(tuple(row.row_id for row in group) == tuple(str(index) for index in range(worker, row_count, workers)) for worker, group in enumerate(groups))


def test_each_successful_cell_is_atomically_persisted_before_interruption(tmp_path, monkeypatch):
    snapshot, rows = _snapshot(tmp_path, workers=1, n_grid=(10, 12, 14), k_grid=(1,))
    prepare_round(snapshot, round_index=1)
    calls = 0

    def interrupted(config, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("simulated SIGKILL boundary")
        _fake_run(config, **kwargs)

    monkeypatch.setattr(ft, "run_nk_grid", interrupted)
    with pytest.raises(RuntimeError, match="SIGKILL"):
        run_slice(snapshot, round_index=1, worker_index=0)
    shard = tmp_path / "outputs" / "round-1" / "worker-0.csv"
    persisted = list(csv.DictReader(shard.open(encoding="utf-8")))
    assert len(persisted) == 2
    assert json.loads(ft.manifest_path(shard).read_text())["completion"]["materialized_rows"] == 2
    assert all(not path.name.endswith(".tmp") for path in shard.parent.iterdir())
    monkeypatch.setattr(ft, "run_nk_grid", _fake_run)
    run_slice(snapshot, round_index=1, worker_index=0)
    assert len(list(csv.DictReader(shard.open(encoding="utf-8")))) == len(expected_model_keys(rows))


def test_real_two_round_recovery_converges_to_one_shot_output(tmp_path, monkeypatch):
    snapshot, rows = _snapshot(tmp_path, workers=2)
    monkeypatch.setattr(ft, "run_nk_grid", _fake_run)
    first = prepare_round(snapshot, round_index=1)
    assert first["todo_rows"] == len(rows)
    run_slice(snapshot, round_index=1, worker_index=0)
    second = prepare_round(snapshot, round_index=2)
    assert second["todo_rows"] == len(rows) - len(read_row_group(Path(first["assignment"]), 0))
    run_slice(snapshot, round_index=2, worker_index=0)
    run_slice(snapshot, round_index=2, worker_index=1)
    result = verify_rounds(snapshot)
    assert result["missing_model_keys"] == 0
    merged = finalize_slice_shards(Path(json.loads(snapshot.read_text())["task_table"]), (tmp_path / "outputs" / "round-1" / "worker-0.csv", tmp_path / "outputs" / "round-2" / "worker-0.csv", tmp_path / "outputs" / "round-2" / "worker-1.csv"), tmp_path / "merged.csv")
    one_shot = tmp_path / "one-shot.csv"
    with one_shot.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "seed", "draw", "N", "K", "status", "metric"]); writer.writeheader()
        for row in sorted(rows, key=lambda row: (row.n_samples, row.k_features)):
            writer.writerow({"model": "ols", "seed": row.seed, "draw": row.draw, "N": row.n_samples, "K": row.k_features, "status": "ok", "metric": f"{row.n_samples}:{row.k_features}"})
    pd.testing.assert_frame_equal(pd.read_csv(merged).sort_values(["N", "K"]).reset_index(drop=True), pd.read_csv(one_shot).sort_values(["N", "K"]).reset_index(drop=True), check_exact=True)


def test_resource_request_is_single_core_and_records_account_constraint():
    request = ResourceRequest(1, "long", "8G", "12:00:00", "proj", "cpu-a")
    assert sbatch_resource_args(request) == ("--partition=long", "--cpus-per-task=1", "--mem=8G", "--time=12:00:00", "--account=proj", "--constraint=cpu-a")
    with pytest.raises(ValueError, match="one CPU"):
        sbatch_resource_args(ResourceRequest(2, "long", "8G", "12:00:00", "proj", "none"))
