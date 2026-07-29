from __future__ import annotations

import csv
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aleatoric_nk_grid.experiment import manifest_path
from aleatoric_nk_grid.nk_grid import METRIC_COLUMNS, NKGridConfig, run_nk_grid
from aleatoric_nk_grid.run_panels import config_to_json
from aleatoric_nk_grid.seed_shards import (
    SeedShardIncompleteError,
    SeedShardValidationError,
    build_finalizer_map,
    build_publish_map,
    diagnose_missing,
    finalize_seed_shards,
    main,
    publish_panel,
    recovery_indices,
)
from aleatoric_nk_grid import seed_shards
from conftest import write_schema_bundle


MODEL_PARAMS = Path(__file__).resolve().parents[1] / "model_params.yaml"
NUMERIC_EQUIVALENCE_TOLERANCES = {
    "bounded_dimensionless": {"rtol": 0.0, "atol": 1e-12},
    "r2_like_unbounded": {"rtol": 1e-12, "atol": 1e-12},
    "scale_dependent": {"rtol": 1e-10, "atol": 1e-12},
}
SCIENTIFIC_COLUMN_GROUPS = {
    "bounded_dimensionless": {
        "spearman_rho",
        "pearson_r",
        "kendall_tau",
        "ccc",
        "ks_statistic",
        "top_decile_hit_rate",
        "bottom_decile_hit_rate",
        "pearson_r2",
    },
    "r2_like_unbounded": {
        "r2_test",
        "skill_score_pct",
        "explained_variance",
        "d2_absolute_error",
        "nrmse",
        "rsr",
        "cv_rmse",
        "mase",
    },
    "scale_dependent": {
        "rmse",
        "mae",
        "medae",
        "max_error",
        "mean_bias",
        "median_bias",
        "pinball_q10",
        "pinball_q90",
        "pinball_q05",
        "pinball_q25",
        "pinball_q50",
        "pinball_q75",
        "pinball_q95",
        "wasserstein_distance",
    },
}


def _config(tmp_path: Path, out: Path, model: str = "ols") -> NKGridConfig:
    return NKGridConfig(
        schema=tmp_path / "schema.json", out=out, outcome="y", models=(model,),
        seed=123, test_size=0.3, n_seeds=1, n_draws=1, n_sizes_n=1,
        n_sizes_k=1, max_n=20, max_k=2, n_grid=(10,), k_grid=(1,),
        batch_size=1, n_jobs=1, min_n=5, model_params=MODEL_PARAMS,
        repeat_plan=((11, 0), (22, 0)), experiment_id="experiment",
        data_version="data-v1", model_spec_version="models-v1",
    )


def _write_shard(config: NKGridConfig, *, seed: int, status: str = "ok") -> None:
    model = config.models[0]
    config.out.parent.mkdir(parents=True, exist_ok=True)
    with config.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "seed", "draw", "N", "K", "status", "metric"])
        writer.writeheader()
        writer.writerow({"model": model, "seed": seed, "draw": 0, "N": 10, "K": 1, "status": status, "metric": "1.0"})
    manifest = {
        "updated_at": f"2026-07-29T00:00:{seed % 60:02d}+00:00",
        "identity": {"mode": "explicit-v1", "experiment_id": "experiment", "data_version": "data-v1", "model_spec_version": "models-v1"},
        "semantic_contract": {"kind": "nk_grid", "model": [model]},
        "design": {"n_grid": [10], "k_grid": [1], "models": [model]},
        "execution": {"mode": "seed-shard", "seed": seed, "draws": [0], "expected_rows": 1},
        "completion": {"expected_rows": 1, "materialized_rows": 1, "completed_rows": 1 if status != "failed" else 0, "failed_rows": 1 if status == "failed" else 0, "status": "complete_with_failures" if status == "failed" else "complete"},
        "failure_policy": {"failed_abs_threshold": 0, "failed_ratio_threshold": 0.0},
    }
    manifest_path(config.out).write_text(json.dumps(manifest), encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _rewrite_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _snapshot(
    tmp_path: Path,
    *,
    duplicate: bool = False,
    models: tuple[str, ...] = ("ols",),
) -> Path:
    panel_out = tmp_path / "panel.csv"
    jobs = []
    for model in models:
        model_final = (
            tmp_path / ".seed-final" / "panel" / f"{model}.csv"
        )
        for seed in (11, 22):
            shard = (
                tmp_path
                / ".seed-shards"
                / "panel"
                / model
                / f"seed-{seed}.csv"
            )
            config = _config(tmp_path, shard, model)
            _write_shard(config, seed=seed)
            jobs.append(
                {
                    "panel": "panel",
                    "model": model,
                    "seed": seed,
                    "draws": [0],
                    "config": config_to_json(config),
                    "final_out": str(model_final),
                }
            )
    if duplicate:
        jobs.append(dict(jobs[0]))
    path = tmp_path / "jobs.json"
    path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "panel_outputs": {"panel": str(panel_out)},
                "jobs": jobs,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_finalizer_merges_full_key_design_and_writes_aggregate_policy(tmp_path):
    snapshot = _snapshot(tmp_path)
    output = finalize_seed_shards(snapshot, panel="panel", model="ols")
    assert output == tmp_path / ".seed-final" / "panel" / "ols.csv"
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


def test_multi_model_panel_publication_preserves_every_model(tmp_path):
    snapshot = _snapshot(tmp_path, models=("ols", "ridge"))
    for model in ("ols", "ridge"):
        finalize_seed_shards(snapshot, panel="panel", model=model)

    output = publish_panel(
        snapshot, panel="panel", models=("ols", "ridge")
    )

    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    assert [(row["model"], row["seed"]) for row in rows] == [
        ("ols", "11"),
        ("ols", "22"),
        ("ridge", "11"),
        ("ridge", "22"),
    ]
    payload = json.loads(manifest_path(output).read_text(encoding="utf-8"))
    assert sorted(payload["failure_policy"]["models"]) == ["ols", "ridge"]
    assert payload["failure_policy"]["passed"] is True


def test_publish_cli_returns_three_when_one_model_final_is_missing(tmp_path):
    snapshot = _snapshot(tmp_path, models=("ols", "ridge"))
    finalize_seed_shards(snapshot, panel="panel", model="ols")
    publish_map = build_publish_map(snapshot)

    assert main(
        [
            "publish",
            "--snapshot",
            str(snapshot),
            "--map",
            str(publish_map),
            "--index",
            "0",
        ]
    ) == 3
    assert not (tmp_path / "panel.csv").exists()


def test_publish_rejects_incomplete_cross_model_key_set(tmp_path):
    snapshot = _snapshot(tmp_path, models=("ols", "ridge"))
    for model in ("ols", "ridge"):
        finalize_seed_shards(snapshot, panel="panel", model=model)
    ridge = tmp_path / ".seed-final" / "panel" / "ridge.csv"
    rows = list(csv.DictReader(ridge.open(encoding="utf-8")))
    with ridge.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerow(rows[0])

    with pytest.raises(SeedShardValidationError, match="incomplete"):
        publish_panel(snapshot, panel="panel", models=("ols", "ridge"))


def test_finalize_and_publish_noop_for_equivalent_existing_results(tmp_path):
    snapshot = _snapshot(tmp_path, models=("ols", "ridge"))
    model_out = finalize_seed_shards(snapshot, panel="panel", model="ols")
    model_mtime = model_out.stat().st_mtime_ns
    assert (
        finalize_seed_shards(snapshot, panel="panel", model="ols")
        == model_out
    )
    assert model_out.stat().st_mtime_ns == model_mtime
    finalize_seed_shards(snapshot, panel="panel", model="ridge")
    panel_out = publish_panel(
        snapshot, panel="panel", models=("ols", "ridge")
    )
    panel_mtime = panel_out.stat().st_mtime_ns
    assert (
        publish_panel(snapshot, panel="panel", models=("ols", "ridge"))
        == panel_out
    )
    assert panel_out.stat().st_mtime_ns == panel_mtime


@pytest.mark.parametrize("operation", ["finalize", "publish"])
def test_concurrent_finalize_and_publish_reserve_distinct_temporary_csvs(
    tmp_path, monkeypatch, operation
):
    snapshot = _snapshot(tmp_path, models=("ols", "ridge"))
    if operation == "publish":
        for model in ("ols", "ridge"):
            finalize_seed_shards(snapshot, panel="panel", model=model)
    expected_target = (
        tmp_path / ".seed-final" / "panel" / "ols.csv"
        if operation == "finalize"
        else tmp_path / "panel.csv"
    )
    expected_lock_error = (
        "Another NK Grid worker already holds the output lease: "
        f"{expected_target.parent / '.locks' / f'{expected_target.name}.run.lock'}"
    )

    barrier = threading.Barrier(2)
    reserved: list[Path] = []
    reserved_threads: set[int] = set()
    reserved_lock = threading.Lock()
    real_reserve = seed_shards._reserve_temporary_csv

    def synchronized_reserve(target, operation_name):
        path = real_reserve(target, operation_name)
        with reserved_lock:
            reserved.append(path)
            reserved_threads.add(threading.get_ident())
        barrier.wait(timeout=10)
        return path

    monkeypatch.setattr(
        seed_shards, "_reserve_temporary_csv", synchronized_reserve
    )

    def run():
        for _ in range(100):
            try:
                if operation == "finalize":
                    return finalize_seed_shards(
                        snapshot, panel="panel", model="ols"
                    )
                return publish_panel(
                    snapshot, panel="panel", models=("ols", "ridge")
                )
            except RuntimeError as exc:
                assert str(exc) == expected_lock_error
                with reserved_lock:
                    if threading.get_ident() in reserved_threads:
                        return None
                time.sleep(0.001)
        pytest.fail("concurrent operation never passed its initial no-op lock")

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: run(), range(2)))

    assert len(reserved) == 2
    assert reserved[0] != reserved[1]
    assert all(not path.exists() for path in reserved)
    assert expected_target in results
    assert expected_target.exists()
    assert manifest_path(expected_target).exists()


def test_concurrent_publication_loser_observes_whole_old_pair(tmp_path):
    target = tmp_path / "result.csv"
    target_manifest = manifest_path(target)
    target.write_text("generation\nold\n", encoding="utf-8")
    _write_json(target_manifest, {"generation": "old"})
    new_csv = seed_shards._reserve_temporary_csv(target, "concurrent-test")
    new_csv.write_text("generation\nnew\n", encoding="utf-8")
    new_manifest = seed_shards._write_temporary_manifest(
        target, {"generation": "new"}
    )
    winner_locked = threading.Event()
    loser_checked = threading.Event()
    loser_observation: list[tuple[str, dict]] = []

    def winner():
        with seed_shards.output_run_lock(target):
            winner_locked.set()
            assert loser_checked.wait(timeout=10)
            seed_shards._publish_pair(new_csv, new_manifest, target)

    def loser():
        assert winner_locked.wait(timeout=10)
        with pytest.raises(RuntimeError) as error:
            with seed_shards.output_run_lock(target):
                pass
        assert str(error.value) == (
            "Another NK Grid worker already holds the output lease: "
            f"{tmp_path / '.locks' / 'result.csv.run.lock'}"
        )
        loser_observation.append(
            (
                target.read_text(encoding="utf-8"),
                _read_json(target_manifest),
            )
        )
        loser_checked.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(winner), executor.submit(loser)]
        for future in futures:
            future.result(timeout=10)

    assert loser_observation == [
        ("generation\nold\n", {"generation": "old"})
    ]
    assert target.read_text(encoding="utf-8") == "generation\nnew\n"
    assert _read_json(target_manifest) == {"generation": "new"}


def test_finalizer_refuses_noop_when_shard_provenance_changes(tmp_path):
    snapshot = _snapshot(tmp_path)
    target = finalize_seed_shards(snapshot, panel="panel", model="ols")
    original = target.read_bytes()
    shard = tmp_path / ".seed-shards" / "panel" / "ols" / "seed-22.csv"
    rows = list(csv.DictReader(shard.open(encoding="utf-8")))
    rows[0]["metric"] = "2.0"
    _rewrite_rows(shard, rows)
    shard_manifest = _read_json(manifest_path(shard))
    shard_manifest["updated_at"] = "2026-07-29T01:00:22+00:00"
    _write_json(manifest_path(shard), shard_manifest)

    assert finalize_seed_shards(
        snapshot, panel="panel", model="ols"
    ) == target

    changed_rows = list(csv.DictReader(target.open(encoding="utf-8")))
    assert target.read_bytes() != original
    assert [row["metric"] for row in changed_rows] == ["1.0", "2.0"]
    provenance = _read_json(manifest_path(target))["input_artifacts"]
    assert provenance[1] == {
        "path": str(shard.resolve()),
        "rows": 1,
        "seed": 22,
        "updated_at": "2026-07-29T01:00:22+00:00",
    }


def test_cli_distinguishes_oserror_from_contract_violation(
    tmp_path, monkeypatch, capsys
):
    def storage_failure(*args, **kwargs):
        raise PermissionError("storage unavailable")

    monkeypatch.setattr(seed_shards, "_load_map", storage_failure)
    arguments = [
        "finalize",
        "--snapshot",
        str(tmp_path / "jobs.json"),
        "--map",
        str(tmp_path / "map.json"),
        "--index",
        "0",
    ]
    assert main(arguments) == 4
    assert capsys.readouterr().err.strip() == "storage unavailable"

    def contract_failure(*args, **kwargs):
        raise SeedShardValidationError("contract mismatch")

    monkeypatch.setattr(seed_shards, "_load_map", contract_failure)
    assert main(arguments) == 1
    assert capsys.readouterr().err.strip() == "contract mismatch"


def test_missing_reports_publication_residuals_for_snapshot_targets(tmp_path):
    snapshot = _snapshot(tmp_path)
    model_target = tmp_path / ".seed-final" / "panel" / "ols.csv"
    panel_target = tmp_path / "panel.csv"
    residuals = (
        model_target.with_name(f".{model_target.name}.publishing"),
        model_target.with_name(f".{model_target.name}.previous"),
        manifest_path(panel_target).with_name(
            f".{manifest_path(panel_target).name}.previous"
        ),
    )
    for residual in residuals:
        residual.parent.mkdir(parents=True, exist_ok=True)
        residual.touch()

    diagnosis = diagnose_missing(snapshot)

    assert diagnosis["publication_residuals"] == [
        str(path.resolve()) for path in sorted(residuals)
    ]


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing", "Missing master seed"),
        ("extra", "Unexpected master seed"),
        ("wrong_draws", "draws do not match"),
        ("duplicate_output", "Duplicate shard output"),
        ("different_final", "disagree on per-model final_out"),
    ],
)
def test_finalizer_rejects_invalid_master_seed_design(
    tmp_path, mutation, match
):
    snapshot = _snapshot(tmp_path)
    payload = _read_json(snapshot)
    if mutation == "missing":
        payload["jobs"].pop()
    elif mutation == "extra":
        payload["jobs"][-1]["seed"] = 33
    elif mutation == "wrong_draws":
        payload["jobs"][-1]["draws"] = [1]
    elif mutation == "duplicate_output":
        payload["jobs"][-1]["config"]["out"] = payload["jobs"][0]["config"]["out"]
    else:
        payload["jobs"][-1]["final_out"] = str(tmp_path / "other.csv")
    _write_json(snapshot, payload)

    with pytest.raises(SeedShardValidationError, match=match):
        finalize_seed_shards(snapshot, panel="panel", model="ols")


@pytest.mark.parametrize(
    ("mutation", "error_type", "match"),
    [
        ("identity", SeedShardValidationError, "identity.data_version differs"),
        ("contract", SeedShardValidationError, "semantic_contract"),
        ("grid", SeedShardValidationError, "resolved N/K grid differs"),
        ("execution_seed", SeedShardValidationError, "execution declaration"),
        ("execution_draws", SeedShardValidationError, "execution declaration"),
        ("incomplete", SeedShardIncompleteError, "shard is incomplete"),
        ("legacy", SeedShardValidationError, "Legacy digest field"),
    ],
)
def test_finalizer_rejects_invalid_shard_manifests(
    tmp_path, mutation, error_type, match
):
    snapshot = _snapshot(tmp_path)
    shard = tmp_path / ".seed-shards" / "panel" / "ols" / "seed-22.csv"
    shard_manifest = manifest_path(shard)
    payload = _read_json(shard_manifest)
    if mutation == "identity":
        payload["identity"]["data_version"] = "wrong"
    elif mutation == "contract":
        payload["semantic_contract"]["kind"] = "wrong"
    elif mutation == "grid":
        payload["design"]["n_grid"] = [11]
    elif mutation == "execution_seed":
        payload["execution"]["seed"] = 11
    elif mutation == "execution_draws":
        payload["execution"]["draws"] = [1]
    elif mutation == "incomplete":
        payload["completion"]["materialized_rows"] = 0
        payload["completion"]["status"] = "incomplete"
    else:
        payload["nested"] = {"file_sha256": "obsolete"}
    _write_json(shard_manifest, payload)

    with pytest.raises(error_type, match=match):
        finalize_seed_shards(snapshot, panel="panel", model="ols")


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("model", "ridge"),
        ("seed", "11"),
        ("draw", "1"),
        ("N", "11"),
        ("K", "2"),
    ],
)
def test_finalizer_rejects_wrong_or_out_of_design_cell_keys(
    tmp_path, column, value
):
    snapshot = _snapshot(tmp_path)
    shard = tmp_path / ".seed-shards" / "panel" / "ols" / "seed-22.csv"
    rows = list(csv.DictReader(shard.open(encoding="utf-8")))
    rows[0][column] = value
    _rewrite_rows(shard, rows)

    with pytest.raises(SeedShardValidationError, match="out-of-design"):
        finalize_seed_shards(snapshot, panel="panel", model="ols")


def test_finalizer_rejects_duplicate_cell_key(tmp_path):
    snapshot = _snapshot(tmp_path)
    shard = tmp_path / ".seed-shards" / "panel" / "ols" / "seed-22.csv"
    rows = list(csv.DictReader(shard.open(encoding="utf-8")))
    _rewrite_rows(shard, [rows[0], rows[0]])

    with pytest.raises(SeedShardValidationError, match="Duplicate cell key"):
        finalize_seed_shards(snapshot, panel="panel", model="ols")


def test_failure_policy_is_aggregated_per_model_and_panel(tmp_path):
    snapshot = _snapshot(tmp_path, models=("ols", "ridge"))
    failed_shard = (
        tmp_path / ".seed-shards" / "panel" / "ridge" / "seed-22.csv"
    )
    rows = list(csv.DictReader(failed_shard.open(encoding="utf-8")))
    rows[0]["status"] = "failed"
    _rewrite_rows(failed_shard, rows)
    failed_manifest = _read_json(manifest_path(failed_shard))
    failed_manifest["completion"].update(
        completed_rows=0,
        failed_rows=1,
        status="complete_with_failures",
    )
    _write_json(manifest_path(failed_shard), failed_manifest)

    ols = finalize_seed_shards(snapshot, panel="panel", model="ols")
    ridge = finalize_seed_shards(snapshot, panel="panel", model="ridge")
    assert _read_json(manifest_path(ols))["failure_policy"]["passed"] is True
    ridge_policy = _read_json(manifest_path(ridge))["failure_policy"]
    assert ridge_policy["failed_count"] == 1
    assert ridge_policy["failed_ratio"] == 0.5
    assert ridge_policy["passed"] is False

    panel = publish_panel(
        snapshot, panel="panel", models=("ols", "ridge")
    )
    policy = _read_json(manifest_path(panel))["failure_policy"]
    assert policy["failed_count"] == 1
    assert policy["passed"] is False
    assert policy["models"]["ols"]["passed"] is True
    assert policy["models"]["ridge"]["passed"] is False


def test_finalizer_removes_single_shard_diagnostics(tmp_path):
    snapshot = _snapshot(tmp_path)
    shard = tmp_path / ".seed-shards" / "panel" / "ols" / "seed-11.csv"
    payload = _read_json(manifest_path(shard))
    payload["diagnostics"] = {"only_this_seed": 1}
    _write_json(manifest_path(shard), payload)

    final = finalize_seed_shards(snapshot, panel="panel", model="ols")
    assert "diagnostics" not in _read_json(manifest_path(final))


def test_atomic_publication_failure_preserves_previous_pair_and_shards(
    tmp_path, monkeypatch
):
    snapshot = _snapshot(tmp_path)
    target = finalize_seed_shards(snapshot, panel="panel", model="ols")
    old_csv = target.read_bytes()
    old_manifest = manifest_path(target).read_bytes()
    manifest_payload = _read_json(manifest_path(target))
    manifest_payload["execution"]["mode"] = "obsolete"
    _write_json(manifest_path(target), manifest_payload)
    preserved_manifest = manifest_path(target).read_bytes()
    real_replace = os.replace

    def fail_new_manifest(source, destination):
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            source_path.name.endswith(".manifest.tmp")
            and destination_path == manifest_path(target)
        ):
            raise OSError("simulated manifest publication failure")
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_new_manifest)
    with pytest.raises(OSError, match="simulated"):
        finalize_seed_shards(snapshot, panel="panel", model="ols")

    assert target.read_bytes() == old_csv
    assert manifest_path(target).read_bytes() == preserved_manifest
    assert old_manifest != preserved_manifest
    assert not target.with_name(f".{target.name}.publishing").exists()
    for seed in (11, 22):
        shard = (
            tmp_path / ".seed-shards" / "panel" / "ols" / f"seed-{seed}.csv"
        )
        assert shard.exists()
        assert manifest_path(shard).exists()


def test_maps_are_read_only_and_reject_wrong_snapshot(tmp_path):
    snapshot = _snapshot(tmp_path, models=("ols", "ridge"))
    finalizer_map = build_finalizer_map(snapshot)
    publish_map = build_publish_map(snapshot)
    assert finalizer_map.stat().st_mode & 0o777 == 0o444
    assert publish_map.stat().st_mode & 0o777 == 0o444
    wrong = tmp_path / "wrong.json"
    wrong.write_text(snapshot.read_text(encoding="utf-8"), encoding="utf-8")
    assert main(
        [
            "finalize",
            "--snapshot",
            str(wrong),
            "--map",
            str(finalizer_map),
            "--index",
            "0",
        ]
    ) == 1


def test_recovery_indices_include_only_missing_or_incomplete_class_members(
    tmp_path,
):
    snapshot = _snapshot(
        tmp_path, models=("ols", "lightgbm", "super_learner")
    )
    ols_missing = (
        tmp_path / ".seed-shards" / "panel" / "ols" / "seed-22.csv"
    )
    ols_missing.unlink()
    manifest_path(ols_missing).unlink()
    serial_incomplete = (
        tmp_path / ".seed-shards" / "panel" / "lightgbm" / "seed-11.csv"
    )
    payload = _read_json(manifest_path(serial_incomplete))
    payload["completion"]["status"] = "incomplete"
    payload["completion"]["materialized_rows"] = 0
    _write_json(manifest_path(serial_incomplete), payload)

    assert recovery_indices(snapshot, resource_class="parallel") == (1,)
    assert recovery_indices(
        snapshot,
        resource_class="parallel",
        requested_indices=(0, 1),
    ) == (1,)
    assert recovery_indices(snapshot, resource_class="serial") == (2,)
    assert recovery_indices(
        snapshot, resource_class="super_learner"
    ) == ()


@pytest.mark.parametrize("mutation", ["identity", "legacy"])
def test_existing_incompatible_or_legacy_final_is_not_replaced(
    tmp_path, mutation
):
    snapshot = _snapshot(tmp_path)
    target = finalize_seed_shards(snapshot, panel="panel", model="ols")
    previous_csv = target.read_bytes()
    payload = _read_json(manifest_path(target))
    if mutation == "identity":
        payload["identity"]["data_version"] = "other"
    else:
        payload["file_sha256"] = "obsolete"
    _write_json(manifest_path(target), payload)
    previous_manifest = manifest_path(target).read_bytes()

    with pytest.raises(SeedShardValidationError):
        finalize_seed_shards(snapshot, panel="panel", model="ols")

    assert target.read_bytes() == previous_csv
    assert manifest_path(target).read_bytes() == previous_manifest


@pytest.mark.parametrize("mutation", ["missing_row", "invalid_status"])
def test_finalizer_rejects_incomplete_rows_and_invalid_status(
    tmp_path, mutation
):
    snapshot = _snapshot(tmp_path)
    shard = tmp_path / ".seed-shards" / "panel" / "ols" / "seed-22.csv"
    rows = list(csv.DictReader(shard.open(encoding="utf-8")))
    if mutation == "missing_row":
        with shard.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=list(rows[0])).writeheader()
        match = "Merged cell keys differ"
    else:
        rows[0]["status"] = "unknown"
        _rewrite_rows(shard, rows)
        match = "invalid status"

    with pytest.raises(SeedShardValidationError, match=match):
        finalize_seed_shards(snapshot, panel="panel", model="ols")


def test_monolithic_and_seed_shard_publication_are_numerically_equivalent(
    tmp_path,
):
    frame = pd.DataFrame(
        {
            "y": np.arange(60, dtype=float) * 1.7
            + np.arange(60, dtype=float) % 5,
            "X_a": np.arange(60, dtype=float),
            "X_b": np.arange(60, dtype=float) % 7,
        }
    )
    schema = write_schema_bundle(
        tmp_path / "input", frame, predictors=["X_a", "X_b"]
    )
    model_params = tmp_path / "model-params.yaml"
    model_params.write_text(
        "\n".join(
            [
                "algorithm_version: equivalence-v1",
                "regression:",
                "  ols:",
                "    fit_intercept: true",
                "  super_learner:",
                "    cv: 2",
                "    passthrough: false",
                "    n_estimators: 8",
                "    max_features: sqrt",
                "    min_samples_leaf: 2",
                "    hidden_layer_sizes: [4]",
                "    alpha: 0.01",
                "    learning_rate_init: 0.001",
                "    max_iter: 40",
                "    positive: true",
                "    lgbm_n_estimators: 8",
                "    lgbm_learning_rate: 0.05",
                "    lgbm_num_leaves: 4",
                "    lgbm_min_data_in_leaf: 2",
                "classification: {}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    panel_out = tmp_path / "panel.csv"
    config = NKGridConfig(
        schema=schema,
        out=panel_out,
        outcome="y",
        models=("ols", "super_learner"),
        seed=123,
        test_size=0.25,
        n_seeds=1,
        n_draws=1,
        n_sizes_n=2,
        n_sizes_k=2,
        max_n=30,
        max_k=2,
        n_grid=(20, 30),
        k_grid=(1, 2),
        batch_size=4,
        n_jobs=1,
        min_n=10,
        model_params=model_params,
        repeat_plan=((11, 0), (11, 1), (22, 0), (22, 1)),
        experiment_id="equivalence-experiment",
        data_version="synthetic-v1",
        model_spec_version="equivalence-models-v1",
    )
    monolithic_frames = []
    monolithic_manifests = []
    for model in config.models:
        monolithic_out = tmp_path / f"monolithic-{model}.csv"
        run_nk_grid(
            replace(config, models=(model,), out=monolithic_out)
        )
        monolithic_frames.append(pd.read_csv(monolithic_out))
        monolithic_manifests.append(
            _read_json(manifest_path(monolithic_out))
        )

    jobs = []
    for model in config.models:
        model_final = (
            tmp_path / ".seed-final" / "panel" / f"{model}.csv"
        )
        for seed in (11, 22):
            shard_out = (
                tmp_path
                / ".seed-shards"
                / "panel"
                / model
                / f"seed-{seed}.csv"
            )
            shard_config = replace(
                config, models=(model,), out=shard_out
            )
            run_nk_grid(
                shard_config,
                execution_pairs=((seed, 0), (seed, 1)),
                defer_failure_policy=True,
                exact_output_path=True,
            )
            jobs.append(
                {
                    "panel": "panel",
                    "model": model,
                    "seed": seed,
                    "draws": [0, 1],
                    "config": config_to_json(shard_config),
                    "final_out": str(model_final),
                }
            )
    snapshot = tmp_path / "equivalence-jobs.json"
    _write_json(
        snapshot,
        {
            "format_version": 1,
            "panel_outputs": {"panel": str(panel_out)},
            "jobs": jobs,
        },
    )
    for model in config.models:
        finalize_seed_shards(snapshot, panel="panel", model=model)
    published_out = publish_panel(
        snapshot, panel="panel", models=sorted(config.models)
    )

    key_columns = ["model", "seed", "draw", "N", "K"]
    monolithic = pd.concat(
        monolithic_frames, ignore_index=True
    ).sort_values(key_columns).reset_index(drop=True)
    published = pd.read_csv(published_out).sort_values(
        key_columns
    ).reset_index(drop=True)
    assert set().union(*SCIENTIFIC_COLUMN_GROUPS.values()) == set(
        METRIC_COLUMNS
    )
    assert list(monolithic.columns) == list(published.columns)
    telemetry = {"fit_seconds", "peak_rss_bytes"}
    scientific = set(METRIC_COLUMNS)
    for column in monolithic.columns:
        if column in telemetry or column in scientific:
            continue
        pd.testing.assert_series_equal(
            monolithic[column],
            published[column],
            check_names=False,
            check_exact=True,
        )
    for group, columns in SCIENTIFIC_COLUMN_GROUPS.items():
        tolerance = NUMERIC_EQUIVALENCE_TOLERANCES[group]
        for column in sorted(columns):
            left = monolithic[column].to_numpy(dtype=float)
            right = published[column].to_numpy(dtype=float)
            assert np.array_equal(np.isnan(left), np.isnan(right))
            np.testing.assert_allclose(
                left,
                right,
                equal_nan=True,
                **tolerance,
            )

    published_manifest = _read_json(manifest_path(published_out))
    for field in ("expected_rows", "materialized_rows", "completed_rows", "failed_rows"):
        assert sum(
            payload["completion"][field]
            for payload in monolithic_manifests
        ) == published_manifest["completion"][field]
    for field in ("failed_count", "ok_count", "skipped_count"):
        assert sum(
            payload["failure_policy"][field]
            for payload in monolithic_manifests
        ) == published_manifest["failure_policy"][field]
    assert all(
        payload["failure_policy"]["passed"]
        for payload in monolithic_manifests
    ) == published_manifest["failure_policy"]["passed"]
