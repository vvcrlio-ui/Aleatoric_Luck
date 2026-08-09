from __future__ import annotations

import csv
import gc
import tracemalloc
from pathlib import Path

from aleatoric_nk_grid.flat_task_table import (
    TaskRow,
    finalize_snapshot,
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


def _measure_peak(root: Path, row_count: int) -> tuple[int, dict[str, object]]:
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
        output_dir=root / "shards",
        workers=1,
    )
    shard = root / "shards" / "round-1" / "worker-0.csv"
    shard.parent.mkdir(parents=True)
    with shard.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "seed", "draw", "N", "K", "status", "metric"])
        for row in rows:
            writer.writerow(["ols", row.seed, 0, 10, 1, "ok", row.seed])
    del rows
    gc.collect()

    tracemalloc.start()
    try:
        receipt = finalize_snapshot(snapshot, tmp_dir=root / "scratch")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert receipt["final_rows"] == row_count
    assert receipt["backend"] == "sqlite_streaming"
    assert 0 < receipt["temporary_bytes_used"] < receipt["estimated_temporary_bytes"]
    assert not Path(receipt["temporary_directory"]).exists()
    return peak, receipt


def test_streaming_finalizer_python_memory_is_bounded_across_tenfold_scale(tmp_path):
    small_peak, small = _measure_peak(tmp_path / "small", 5_000)
    large_peak, large = _measure_peak(tmp_path / "large", 50_000)
    assert large["rows_read"] == 10 * small["rows_read"]
    # SQLite owns the proportional index on disk. Python retains only one
    # Arrow batch and one CSV row, so a 10x design must not approach 10x heap.
    assert large_peak < small_peak * 2.5
