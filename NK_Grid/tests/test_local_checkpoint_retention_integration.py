"""Linux/POSIX integration checks; portable policy tests cannot replace these."""
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fcntl", reason="real engine requires POSIX")

from aleatoric_nk_grid.experiment import checkpoint_parts, load_checkpoint, write_checkpoint_part
from aleatoric_nk_grid.nk_grid import NKGridConfig, run_nk_grid
from conftest import write_schema_bundle


def test_keep_preserves_more_than_compaction_threshold(tmp_path):
    out = tmp_path / "result.csv"
    for draw in range(101):
        write_checkpoint_part([{"experiment_id": "x", "model": "ols", "seed": 1,
            "draw": draw, "N": 10, "K": 2, "status": "ok"}], out, keep_all=True)
    assert len(checkpoint_parts(out)) == 101
    assert len(load_checkpoint(out)) == 101


@pytest.mark.parametrize("policy", ["default", "keep", "delete"])
@pytest.mark.parametrize("rerun_completed", [True, False])
def test_success_and_incomplete_local_runs(tmp_path, policy, rerun_completed):
    x = np.arange(40, dtype=float)
    schema = write_schema_bundle(tmp_path / "input", pd.DataFrame({"y": x * 2 + 1, "X_a": x, "X_b": x % 3}), predictors=["X_a", "X_b"])
    out = tmp_path / "result.csv"
    config = NKGridConfig(schema=schema, out=out, outcome="y", models=("ols",),
        seed=123, test_size=0.3, n_seeds=1, n_draws=1, n_sizes_n=2, n_sizes_k=2,
        max_n=0, max_k=0, n_grid=(10, 15), k_grid=(1, 2), batch_size=1, n_jobs=1,
        checkpoint_retention=policy, rerun_completed=rerun_completed)
    run_nk_grid(config, max_jobs=1)
    assert checkpoint_parts(out), "partial runs must retain recovery data under every policy"
    run_nk_grid(config)
    assert len(pd.read_csv(out)) == 4
    assert bool(checkpoint_parts(out)) is (policy == "keep")
    manifest = json.loads(out.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert manifest["output"]["checkpoint_retention"] == policy
    if policy == "keep":
        assert manifest["design"]["checkpointing"]["loose_parts_per_compaction"] is None
        original_parts = checkpoint_parts(out)
        run_nk_grid(replace(config, checkpoint_retention="default"))
        assert checkpoint_parts(out) == original_parts
        run_nk_grid(replace(config, checkpoint_retention="delete"))
        assert not checkpoint_parts(out)
        with pytest.raises(ValueError, match="cannot reconstruct"):
            run_nk_grid(config)
