"""Real POSIX lifecycle coverage. Never certified through a Windows lock shim."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fcntl", reason="dynamic lifecycle requires real POSIX flock")

import numpy as np
import pandas as pd

from conftest import write_repo_schema_bundle as write_schema_bundle
from aleatoric_nk_grid.checkpoint_retention import CheckpointArchiveError, checkpoint_archive_path
from aleatoric_nk_grid.chunk_planning import ClusterPolicy, build_dynamic_plan
from aleatoric_nk_grid.flat_task_table import (
    _generation_shared_lease,
    close_generation,
    finalize_snapshot,
    prepare_round,
    run_slice,
    verify_rounds,
)
from aleatoric_nk_grid.generation_control import ControlBusyError, ControlProtocolError, schedule_transaction
from aleatoric_nk_grid.nk_grid import NKGridConfig


def _fixture(tmp_path: Path, policy: str):
    frame = pd.DataFrame({"x": np.arange(40, dtype=float), "y": np.arange(40, dtype=float)})
    schema = write_schema_bundle(tmp_path / "input", frame, predictors=["x"])
    config = NKGridConfig(
        schema=schema, out=tmp_path / "final.csv", outcome="y", models=("ols",),
        seed=1, test_size=0.2, n_seeds=1, n_draws=1, n_sizes_n=1, n_sizes_k=1,
        max_n=10, max_k=1, batch_size=1, n_jobs=1, repeat_plan=((1, 0),), min_n=2,
        checkpoint_retention=policy,
    )
    snapshot = tmp_path / "snapshot.json"
    build_dynamic_plan(
        config, n_grid=(10,), k_grid=(1,),
        cluster=ClusterPolicy(workers=1, rounds=1, partition="test", time_limit="01:00:00", account="test", constraint="none"),
        table_path=tmp_path / "tasks.parquet", snapshot_path=snapshot,
        output_dir=tmp_path / "history", panel="generic",
    )
    target = dict(round_index=1, submission_generation="g1", expected_prep_token="job-1", prep_job_id="job-1", expected_pointer_version=0)
    prepare_round(snapshot, round_index=1, submission_generation="g1", prep_token="job-1", prep_job_id="job-1", expected_pointer_version=0)
    return snapshot, config, target


@pytest.mark.parametrize("policy", ["default", "keep", "delete"])
def test_real_dynamic_success_respects_retention_and_delete_is_terminal(tmp_path, policy):
    snapshot, config, target = _fixture(tmp_path, policy)
    wal = run_slice(snapshot, worker_index=0, **target)
    close_generation(snapshot, **target)
    assert verify_rounds(snapshot, **target)["complete"] is True
    result = finalize_snapshot(snapshot, **target)
    assert result["final_rows"] == 1
    assert wal.exists() is (policy != "delete")
    marker = checkpoint_archive_path(tmp_path / "history")
    assert marker.exists() is (policy == "delete")
    final_bytes = config.out.read_bytes()
    if policy == "delete":
        assert finalize_snapshot(snapshot, **target) == result
        assert config.out.read_bytes() == final_bytes
        for action in (lambda: run_slice(snapshot, worker_index=0, **target),
                       lambda: close_generation(snapshot, **target),
                       lambda: verify_rounds(snapshot, **target),
                       lambda: prepare_round(snapshot, round_index=1, submission_generation="g1", prep_token="job-1", prep_job_id="job-1", expected_pointer_version=0)):
            with pytest.raises(CheckpointArchiveError, match="final CSV"):
                action()


def test_incomplete_dynamic_run_does_not_archive_or_delete(tmp_path):
    snapshot, config, target = _fixture(tmp_path, "delete")
    close_generation(snapshot, **target)
    assert verify_rounds(snapshot, **target)["complete"] is False
    with pytest.raises(ControlProtocolError, match="verification receipt"):
        finalize_snapshot(snapshot, **target)
    assert not config.out.exists()
    assert not checkpoint_archive_path(tmp_path / "history").exists()


def test_dynamic_archive_exclusive_generation_lease_blocks_cleanup(tmp_path):
    snapshot, config, target = _fixture(tmp_path, "delete")
    wal = run_slice(snapshot, worker_index=0, **target)
    close_generation(snapshot, **target)
    verify_rounds(snapshot, **target)
    with _generation_shared_lease(wal.parent / "generation.lease"):
        with pytest.raises(ControlBusyError, match="lease is busy"):
            finalize_snapshot(snapshot, **target)
    assert wal.is_file()
    assert not checkpoint_archive_path(tmp_path / "history").exists()
    assert finalize_snapshot(snapshot, **target)["deleted_checkpoint_files"] == 1


def test_dynamic_archive_fast_path_rejects_changed_csv_and_wrong_target(tmp_path):
    snapshot, config, target = _fixture(tmp_path, "delete")
    run_slice(snapshot, worker_index=0, **target)
    close_generation(snapshot, **target)
    verify_rounds(snapshot, **target)
    finalize_snapshot(snapshot, **target)
    with pytest.raises(CheckpointArchiveError, match="target"):
        finalize_snapshot(snapshot, **{**target, "submission_generation": "wrong"})
    config.out.write_bytes(b"changed")
    with pytest.raises(CheckpointArchiveError, match="size/hash"):
        finalize_snapshot(snapshot, **target)


@pytest.mark.parametrize("old_policy", ["default", "keep"])
def test_old_policy_finalizer_cannot_rewrite_archived_csv_or_receipt(tmp_path, old_policy):
    snapshot, config, target = _fixture(tmp_path, "delete")
    run_slice(snapshot, worker_index=0, **target)
    close_generation(snapshot, **target)
    verify_rounds(snapshot, **target)
    expected = finalize_snapshot(snapshot, **target)
    receipt = config.out.with_name(config.out.name + ".finalization.json")
    before = (config.out.read_bytes(), receipt.read_bytes())
    # The policy is operational, not part of the scientific analysis. An older
    # caller must still observe the analysis-scoped terminal archive marker.
    snapshot_payload = json.loads(snapshot.read_text())
    snapshot_payload["config"]["checkpoint_retention"] = old_policy
    # The production snapshot is mode 0444. Use a distinct historical caller
    # fixture instead of failing on permissions before the behavior assertion.
    historical_snapshot = tmp_path / f"historical-{old_policy}-snapshot.json"
    historical_snapshot.write_text(json.dumps(snapshot_payload))
    assert finalize_snapshot(historical_snapshot, **target) == expected
    assert (config.out.read_bytes(), receipt.read_bytes()) == before
    with pytest.raises(CheckpointArchiveError, match="terminally archived"):
        verify_rounds(historical_snapshot, **target)


def test_dynamic_archive_never_deletes_during_new_active_generation(tmp_path):
    snapshot, config, target = _fixture(tmp_path, "delete")
    wal = run_slice(snapshot, worker_index=0, **target)
    close_generation(snapshot, **target)
    verify_rounds(snapshot, **target)
    # An invalid/current external pointer must not authorize deleting a valid
    # frozen history. This checks the cleanup guard before its first unlink.
    pointer = tmp_path / "history" / "active-generation.json"
    payload = json.loads(pointer.read_text())
    payload["generation_activation_path"] = str(tmp_path / "history" / "new-active" / "generation.activation.json")
    pointer.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises((CheckpointArchiveError, ControlProtocolError)):
        finalize_snapshot(snapshot, **target)
    assert wal.is_file()
    assert not checkpoint_archive_path(tmp_path / "history").exists()


@pytest.mark.parametrize("policy", ["default", "keep", "delete"])
def test_finalizer_of_every_policy_obeys_exclusive_schedule_lease(tmp_path, policy):
    snapshot, config, target = _fixture(tmp_path, policy)
    wal = run_slice(snapshot, worker_index=0, **target)
    close_generation(snapshot, **target)
    verify_rounds(snapshot, **target)
    with _generation_shared_lease(tmp_path / "history" / ".analysis-schedule.lease"):
        with pytest.raises(ControlBusyError, match="lease is busy"):
            finalize_snapshot(snapshot, **target)
    assert not config.out.exists()
    assert wal.is_file()
    assert finalize_snapshot(snapshot, **target)["final_rows"] == 1


@pytest.mark.parametrize("policy", ["default", "keep", "delete"])
@pytest.mark.parametrize("phase", ["run", "close", "verify"])
def test_readers_of_every_policy_obey_shared_schedule_lease(tmp_path, policy, phase):
    snapshot, config, target = _fixture(tmp_path, policy)
    actions = {"run": lambda: run_slice(snapshot, worker_index=0, **target),
               "close": lambda: close_generation(snapshot, **target),
               "verify": lambda: verify_rounds(snapshot, **target)}
    with schedule_transaction(tmp_path / "history"):
        with pytest.raises(ControlBusyError, match="lease is busy"):
            actions[phase]()
    assert not config.out.exists()
    assert not checkpoint_archive_path(tmp_path / "history").exists()
