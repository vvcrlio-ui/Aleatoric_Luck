from __future__ import annotations

import csv
import gc
import tracemalloc
from pathlib import Path

import pyarrow.parquet as pq

import aleatoric_nk_grid.flat_task_table as ft
from aleatoric_nk_grid.flat_task_table import (
    TaskRow,
    prepare_round,
    verify_rounds,
    write_task_table,
    write_work_snapshot,
)
from aleatoric_nk_grid.nk_grid import NKGridConfig


def _config(root: Path) -> NKGridConfig:
    return NKGridConfig(
        schema=root / "unused-schema.json",
        out=root / "final.csv",
        outcome="y",
        models=("ols",),
        seed=0,
        test_size=0.2,
        n_seeds=1,
        n_draws=1,
        n_sizes_n=1,
        n_sizes_k=1,
        max_n=10,
        max_k=1,
        batch_size=1,
        n_jobs=1,
        repeat_plan=((0, 0),),
    )


def _snapshot(root: Path, row_count: int, *, workers: int) -> tuple[Path, tuple[TaskRow, ...]]:
    root.mkdir(parents=True)
    rows = tuple(
        TaskRow(f"row-{seed}", seed, 0, 10, 1, "imputed_core", ("ols",))
        for seed in range(row_count)
    )
    table = write_task_table(root / "tasks.parquet", rows, rows_per_group=2_048)
    snapshot = write_work_snapshot(
        root / "snapshot.json",
        table_path=table,
        panel="synthetic",
        config=_config(root),
        output_dir=root / "outputs",
        workers=workers,
        preparation_tmp_dir=root / "prep-scratch",
        verification_tmp_dir=root / "verify-scratch",
    )
    return snapshot, rows


def _write_complete_shard(snapshot: Path, rows: tuple[TaskRow, ...]) -> None:
    output = Path(ft._load_snapshot(snapshot)["output_dir"]) / "round-1" / "worker-0.csv"
    output.parent.mkdir(parents=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "seed", "draw", "N", "K", "status", "metric"])
        for row in rows:
            writer.writerow(["ols", row.seed, row.draw, row.n_samples, row.k_features, "ok", row.seed])


def _measure_verify(root: Path, row_count: int) -> tuple[int, dict[str, object]]:
    snapshot, rows = _snapshot(root, row_count, workers=1)
    _write_complete_shard(snapshot, rows)
    del rows
    gc.collect()
    tracemalloc.start()
    try:
        result = verify_rounds(snapshot)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result["missing_model_keys"] == 0
    assert result["crashed_row_ids"] == []
    assert result["too_long_row_ids"] == []
    assert (root / "outputs" / "verification.json").exists()
    assert not list((root / "verify-scratch").glob("nk-grid-verification-*"))
    return peak, result


def _measure_prep(root: Path, row_count: int, *, workers: int) -> tuple[int, dict[str, object]]:
    snapshot, rows = _snapshot(root, row_count, workers=workers)
    del rows
    gc.collect()
    tracemalloc.start()
    try:
        result = prepare_round(snapshot, round_index=1, prep_token="prep-1")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assignment = Path(result["assignment"])
    assert result["todo_rows"] == row_count
    assert pq.ParquetFile(assignment).num_row_groups == workers
    assert (assignment.parent / "assignment.ready.json").exists()
    assert not list((root / "prep-scratch").glob("nk-grid-preparation-*"))
    return peak, result


def test_verify_rounds_streams_sqlite_index_across_tenfold_scale(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ft,
        "read_task_table",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("verify loaded all task rows")),
    )
    monkeypatch.setattr(
        ft,
        "_completed_keys",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("verify loaded completed keys")),
    )
    monkeypatch.setattr(
        ft,
        "_attempt_records",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("verify loaded attempts")),
    )
    small_peak, small = _measure_verify(tmp_path / "small", 5_000)
    large_peak, large = _measure_verify(tmp_path / "large", 50_000)
    assert large["expected_model_keys"] == 10 * small["expected_model_keys"]
    # Every cardinality-growing key lives in SQLite; Python retains one Arrow
    # batch and one streamed CSV row, so 10x rows must not approach 10x heap.
    assert large_peak < small_peak * 2.5


def test_prepare_round_streams_modulo_staging_across_tenfold_scale(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ft,
        "read_task_table",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("prep loaded all task rows")),
    )
    monkeypatch.setattr(
        ft,
        "_completed_keys",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("prep loaded completed keys")),
    )
    monkeypatch.setattr(
        ft,
        "_attempt_records",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("prep loaded attempts")),
    )
    # Keep each worker's required single Parquet row group at 500 rows while
    # scaling the total design by 10x.  This asserts the only intentional
    # materialisation is one worker slice, never all rows/todo/groups.
    small_peak, small = _measure_prep(tmp_path / "small", 5_000, workers=10)
    large_peak, large = _measure_prep(tmp_path / "large", 50_000, workers=100)
    assert large["todo_rows"] == 10 * small["todo_rows"]
    assert max(small["assigned_rows"]) == max(large["assigned_rows"]) == 500
    assert large_peak < small_peak * 2.5
