from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from aleatoric_nk_grid.experiment import manifest_path
from aleatoric_nk_grid.nk_grid import NKGridConfig
from aleatoric_nk_grid.run_panels import config_to_json
from aleatoric_nk_grid.seed_shards import (
    SeedShardValidationError,
    build_finalizer_map,
    diagnose_missing,
    finalize_seed_shards,
    main,
)


MODEL_PARAMS = Path(__file__).resolve().parents[1] / "model_params.yaml"


def _config(tmp_path: Path, out: Path) -> NKGridConfig:
    return NKGridConfig(
        schema=tmp_path / "schema.json", out=out, outcome="y", models=("ols",),
        seed=123, test_size=0.3, n_seeds=1, n_draws=1, n_sizes_n=1,
        n_sizes_k=1, max_n=20, max_k=2, n_grid=(10,), k_grid=(1,),
        batch_size=1, n_jobs=1, min_n=5, model_params=MODEL_PARAMS,
        repeat_plan=((11, 0), (22, 0)), experiment_id="experiment",
        data_version="data-v1", model_spec_version="models-v1",
    )


def _write_shard(config: NKGridConfig, *, seed: int, status: str = "ok") -> None:
    config.out.parent.mkdir(parents=True, exist_ok=True)
    with config.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "seed", "draw", "N", "K", "status", "metric"])
        writer.writeheader()
        writer.writerow({"model": "ols", "seed": seed, "draw": 0, "N": 10, "K": 1, "status": status, "metric": "1.0"})
    manifest = {
        "identity": {"mode": "explicit-v1", "experiment_id": "experiment", "data_version": "data-v1", "model_spec_version": "models-v1"},
        "semantic_contract": {"kind": "nk_grid", "model": ["ols"]},
        "design": {"n_grid": [10], "k_grid": [1]},
        "execution": {"mode": "seed-shard", "seed": seed, "draws": [0], "expected_rows": 1},
        "completion": {"expected_rows": 1, "materialized_rows": 1, "completed_rows": 1 if status != "failed" else 0, "failed_rows": 1 if status == "failed" else 0, "status": "complete_with_failures" if status == "failed" else "complete"},
        "failure_policy": {"failed_abs_threshold": 0, "failed_ratio_threshold": 0.0},
    }
    manifest_path(config.out).write_text(json.dumps(manifest), encoding="utf-8")


def _snapshot(tmp_path: Path, *, duplicate: bool = False) -> Path:
    final = tmp_path / "final.csv"
    jobs = []
    for seed in (11, 22):
        shard = tmp_path / ".seed-shards" / "panel" / "ols" / f"seed-{seed}.csv"
        config = _config(tmp_path, shard)
        _write_shard(config, seed=seed)
        jobs.append({"panel": "panel", "model": "ols", "seed": seed, "draws": [0], "config": config_to_json(config), "final_out": str(final)})
    if duplicate:
        jobs.append(dict(jobs[0]))
    path = tmp_path / "jobs.json"
    path.write_text(json.dumps({"format_version": 1, "jobs": jobs}), encoding="utf-8")
    return path


def test_finalizer_merges_full_key_design_and_writes_aggregate_policy(tmp_path):
    snapshot = _snapshot(tmp_path)
    output = finalize_seed_shards(snapshot, panel="panel", model="ols")
    assert output == tmp_path / "final.csv"
    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    assert [(row["seed"], row["draw"]) for row in rows] == [("11", "0"), ("22", "0")]
    final_manifest = json.loads(manifest_path(output).read_text(encoding="utf-8"))
    assert final_manifest["failure_policy"]["failed_count"] == 0
    assert final_manifest["failure_policy"]["passed"] is True


def test_finalizer_rejects_duplicate_master_seed_instead_of_overwriting(tmp_path):
    snapshot = _snapshot(tmp_path, duplicate=True)
    with pytest.raises(SeedShardValidationError, match="Duplicate master seed"):
        finalize_seed_shards(snapshot, panel="panel", model="ols")


def test_missing_cli_reports_missing_as_data_not_a_command_failure(tmp_path, capsys):
    snapshot = _snapshot(tmp_path)
    shard = tmp_path / ".seed-shards" / "panel" / "ols" / "seed-22.csv"
    shard.unlink()
    manifest_path(shard).unlink()
    assert main(["missing", "--snapshot", str(snapshot)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["missing_master_indices"] == [1]
    assert diagnose_missing(snapshot)["invalid_targets"] == []


def test_finalizer_cli_returns_three_for_missing_shard(tmp_path):
    snapshot = _snapshot(tmp_path)
    shard = tmp_path / ".seed-shards" / "panel" / "ols" / "seed-22.csv"
    shard.unlink()
    manifest_path(shard).unlink()
    finalizer_map = build_finalizer_map(snapshot)
    assert main(["finalize", "--snapshot", str(snapshot), "--map", str(finalizer_map), "--index", "0"]) == 3
