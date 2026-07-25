from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from aleatoric_nk_grid.experiment import (
    checkpoint_parts,
    compact_checkpoint_parts,
    load_checkpoint,
    load_checkpoint_index,
    write_checkpoint_part,
)


def _row(cell: int, *, status: str = "ok") -> dict[str, object]:
    return {
        "experiment_id": "experiment",
        "model": "ols",
        "seed": 1,
        "draw": cell,
        "N": 10,
        "K": 2,
        "status": status,
        "rmse": float(cell),
    }


def test_compaction_publishes_one_shard_and_preserves_all_rows(tmp_path):
    out = tmp_path / "result.csv"
    for cell in range(3):
        write_checkpoint_part([_row(cell)], out)

    compact = compact_checkpoint_parts(out, loose_part_threshold=3)

    assert compact is not None
    assert compact.exists()
    assert "-compact-" in compact.name
    assert checkpoint_parts(out) == [compact]
    full = load_checkpoint(out)
    assert full["draw"].tolist() == [0, 1, 2]
    projected = load_checkpoint_index(out)
    assert projected["draw"].tolist() == [0, 1, 2]
    assert "rmse" not in projected


def test_interruption_before_compact_publish_keeps_all_sources(tmp_path):
    out = tmp_path / "result.csv"
    for cell in range(3):
        write_checkpoint_part([_row(cell)], out)
    original_parts = checkpoint_parts(out)

    with (
        patch(
            "aleatoric_nk_grid.experiment.os.replace",
            side_effect=RuntimeError("interrupted before publish"),
        ),
        pytest.raises(RuntimeError, match="before publish"),
    ):
        compact_checkpoint_parts(out, loose_part_threshold=3)

    assert checkpoint_parts(out) == original_parts
    assert load_checkpoint(out)["draw"].tolist() == [0, 1, 2]


def test_interruption_after_publish_leaves_only_deduplicable_copies(tmp_path):
    out = tmp_path / "result.csv"
    for cell in range(3):
        write_checkpoint_part([_row(cell)], out)

    with (
        patch(
            "aleatoric_nk_grid.experiment._remove_compacted_sources",
            side_effect=RuntimeError("interrupted during cleanup"),
        ),
        pytest.raises(RuntimeError, match="during cleanup"),
    ):
        compact_checkpoint_parts(out, loose_part_threshold=3)

    parts_after_interruption = checkpoint_parts(out)
    assert len(parts_after_interruption) == 4
    assert sum("-compact-" in part.name for part in parts_after_interruption) == 1
    full = load_checkpoint(out)
    assert full["draw"].tolist() == [0, 1, 2]
    assert len(load_checkpoint_index(out)) == 3


def test_compaction_does_not_recompact_prior_compact_shards(tmp_path):
    out = tmp_path / "result.csv"
    for cell in range(3):
        write_checkpoint_part([_row(cell)], out)
    first_compact = compact_checkpoint_parts(out, loose_part_threshold=3)
    assert first_compact is not None

    for cell in range(3, 6):
        write_checkpoint_part([_row(cell)], out)
    second_compact = compact_checkpoint_parts(out, loose_part_threshold=3)

    assert second_compact is not None
    assert checkpoint_parts(out) == sorted([first_compact, second_compact])
    assert load_checkpoint(out)["draw"].tolist() == list(range(6))


def test_compaction_overlap_preserves_completed_status_priority(tmp_path):
    out = tmp_path / "result.csv"
    write_checkpoint_part([_row(0, status="ok")], out)
    write_checkpoint_part([_row(0, status="failed")], out)

    with (
        patch(
            "aleatoric_nk_grid.experiment._remove_compacted_sources",
            side_effect=RuntimeError("interrupted during cleanup"),
        ),
        pytest.raises(RuntimeError),
    ):
        compact_checkpoint_parts(out, loose_part_threshold=2)

    row = load_checkpoint(out).iloc[0]
    assert row["status"] == "ok"
    assert row["rmse"] == 0.0


def test_default_writer_compacts_each_fifty_loose_shards(tmp_path):
    out = tmp_path / "result.csv"

    for cell in range(100):
        write_checkpoint_part([_row(cell)], out)

    parts = checkpoint_parts(out)
    assert len(parts) == 2
    assert all("-compact-" in part.name for part in parts)
    assert load_checkpoint_index(out)["draw"].tolist() == list(range(100))
