from __future__ import annotations

import os
import json
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

from conftest import write_repo_schema_bundle as write_schema_bundle
from aleatoric_nk_grid.chunk_planning import ClusterPolicy, build_dynamic_plan
from aleatoric_nk_grid.execution_contract import canonical_json_bytes
from aleatoric_nk_grid.flat_task_table import (
    FinalizationError,
    close_generation,
    finalize_snapshot,
    prepare_round,
    recover_generation_activation,
    run_slice,
    verify_rounds,
)
from aleatoric_nk_grid.generation_control import (
    ActivationTarget,
    ControlBusyError,
    ControlProtocolError,
    ControlSupersededError,
    publish_activation_intent,
)
import aleatoric_nk_grid.nk_grid as ng
from aleatoric_nk_grid.nk_grid import NKGridConfig, run_nk_grid
from aleatoric_nk_grid.nk_grid import NKGridExecutionSession
from aleatoric_nk_grid.worker_event_wal import (
    TASK_ABORTED,
    TASK_RESULT,
    WAL_FORMAT,
    WALFrameTooLarge,
    WALProtocolError,
    WorkerEventLog,
    bounded_abort_payload,
    scan_wal,
)


def _identity() -> dict[str, object]:
    return {
        "wal_format": WAL_FORMAT,
        "analysis_id": "analysis",
        "execution_plan_id": "plan",
        "execution_contract_sha256": "contract",
        "round": 1,
        "submission_generation": "generation",
        "worker": 0,
        "workers": 1,
        "assignment_path": "/immutable/assignment.parquet",
        "assignment_sha256": "assignment",
        "assignment_index_path": "/immutable/assignment.index.json",
        "assignment_index_sha256": "index",
        "assignment_row_group": 0,
        "assignment_row_count": 1,
        "assignment_row_group_digest": "digest",
    }


def _row() -> dict[str, object]:
    return {"model": "ols", "seed": 1, "draw": 0, "N": 10, "K": 1, "status": "ok", "error": ""}


def _fault_plan(tmp_path: Path, *, rounds: int = 1) -> tuple[Path, dict[str, object]]:
    frame = pd.DataFrame({"x": np.arange(30, dtype=float), "y": np.arange(30, dtype=float)})
    schema = write_schema_bundle(tmp_path / "input", frame, predictors=["x"])
    config = NKGridConfig(
        schema=schema, out=tmp_path / "final.csv", outcome="y", models=("ols",),
        seed=1, test_size=0.2, n_seeds=1, n_draws=1, n_sizes_n=1,
        n_sizes_k=1, max_n=10, max_k=1, batch_size=1, n_jobs=1,
        repeat_plan=((1, 0),), min_n=2,
    )
    plan = build_dynamic_plan(
        config, n_grid=(10,), k_grid=(1,),
        cluster=ClusterPolicy(
            workers=1, rounds=rounds, partition="test", time_limit="01:00:00",
            account="test", constraint="none",
        ),
        table_path=tmp_path / "tasks.parquet", snapshot_path=tmp_path / "snapshot.json",
        output_dir=tmp_path / "out", panel="generic",
    )
    return tmp_path / "snapshot.json", plan


def test_two_phase_wal_commits_complete_cell_group_and_reopens(tmp_path: Path):
    path = tmp_path / "worker.events.wal"
    with WorkerEventLog.open_exclusive_and_repair(path, identity=_identity()) as log:
        sequence = log.commit_started(row_id="row-1")
        log.commit_result(sequence=sequence, row_id="row-1", public_rows=[_row()])
    scan = WorkerEventLog.open_shared(path, expected_identity=_identity())
    assert [record.event_type for record in scan.records] == ["TASK_STARTED", TASK_RESULT]
    assert scan.has_uncommitted_tail is False
    with WorkerEventLog.open_exclusive_and_repair(path, identity=_identity()) as resumed:
        assert resumed.committed_terminal_row_ids() == {"row-1"}


def test_writer_repairs_only_an_uncommitted_file_tail(tmp_path: Path):
    path = tmp_path / "worker.events.wal"
    with WorkerEventLog.open_exclusive_and_repair(path, identity=_identity()) as log:
        log.commit_started(row_id="row-1")
    committed = path.stat().st_size
    with path.open("ab") as handle:
        handle.write(b"NKGRID-WAL-PREPARE-1\n\x00\x00")
        handle.flush(); os.fsync(handle.fileno())
    assert scan_wal(path).has_uncommitted_tail is True
    with WorkerEventLog.open_exclusive_and_repair(path, identity=_identity()):
        pass
    assert path.stat().st_size == committed
    assert scan_wal(path).has_uncommitted_tail is False


def test_committed_payload_corruption_is_fail_closed(tmp_path: Path):
    path = tmp_path / "worker.events.wal"
    with WorkerEventLog.open_exclusive_and_repair(path, identity=_identity()) as log:
        sequence = log.commit_started(row_id="row-1")
        log.commit_result(sequence=sequence, row_id="row-1", public_rows=[_row()])
    data = bytearray(path.read_bytes())
    data[-20] ^= 1
    path.write_bytes(data)
    with pytest.raises(WALProtocolError):
        scan_wal(path)
    with pytest.raises(WALProtocolError):
        WorkerEventLog.open_exclusive_and_repair(path, identity=_identity())


def test_payload_overflow_becomes_bounded_abort_not_truncated(tmp_path: Path, monkeypatch):
    path = tmp_path / "worker.events.wal"
    with WorkerEventLog.open_exclusive_and_repair(path, identity=_identity()) as log:
        sequence = log.commit_started(row_id="row-1")
        monkeypatch.setattr("aleatoric_nk_grid.worker_event_wal.WAL_PAYLOAD_LIMIT", 40)
        with pytest.raises(WALFrameTooLarge):
            log.commit_result(sequence=sequence, row_id="row-1", public_rows=[_row()])
        payload = bounded_abort_payload(reason_code="RESULT_FRAME_TOO_LARGE", actual_bytes=100, diagnostic="full diagnostic")
        log.commit_aborted(sequence=sequence, row_id="row-1", payload=payload)
    scan = scan_wal(path)
    assert scan.records[-1].event_type == TASK_ABORTED


def test_started_metadata_limit_and_uncommitted_body_recovery(tmp_path: Path, monkeypatch):
    path = tmp_path / "worker.events.wal"
    with WorkerEventLog.open_exclusive_and_repair(path, identity=_identity()) as log:
        with pytest.raises(WALFrameTooLarge):
            log.commit_started(row_id="too-large", metadata={"diagnostic": "x" * (4 * 1024)})
        original_sync = __import__("aleatoric_nk_grid.worker_event_wal", fromlist=["_sync_fd"])._sync_fd
        def crash_before_body_fsync(_: int) -> None:
            raise RuntimeError("injected body-fsync crash")
        monkeypatch.setattr("aleatoric_nk_grid.worker_event_wal._sync_fd", crash_before_body_fsync)
        with pytest.raises(RuntimeError, match="body-fsync"):
            log.commit_started(row_id="tail")
        monkeypatch.setattr("aleatoric_nk_grid.worker_event_wal._sync_fd", original_sync)
    assert scan_wal(path).has_uncommitted_tail is True
    with WorkerEventLog.open_exclusive_and_repair(path, identity=_identity()) as resumed:
        sequence = resumed.commit_started(row_id="resumed")
        resumed.commit_result(sequence=sequence, row_id="resumed", public_rows=[_row()])
    assert scan_wal(path).has_uncommitted_tail is False


@pytest.mark.parametrize("event_type", ["START", "RESULT", "ABORTED"])
@pytest.mark.parametrize(
    "boundary",
    [
        "before_body_sync",
        "after_body_sync",
        "after_commit_trailer",
        "before_commit_sync",
        "after_commit_sync",
    ],
)
def test_each_wal_event_recovers_at_every_two_phase_sync_boundary(
    tmp_path: Path, event_type: str, boundary: str,
):
    path = tmp_path / f"{event_type}-{boundary}.wal"
    with WorkerEventLog.open_exclusive_and_repair(path, identity=_identity()) as log:
        sequence = None
        if event_type != "START":
            sequence = log.commit_started(row_id="row-1")

        def crash(observed: str) -> None:
            if observed == boundary:
                raise RuntimeError(f"injected {boundary}")

        with pytest.raises(RuntimeError, match="injected"):
            if event_type == "START":
                log.commit_started(row_id="row-1", fault=crash)
            elif event_type == "RESULT":
                assert sequence is not None
                log.commit_result(
                    sequence=sequence, row_id="row-1",
                    public_rows=[_row()], fault=crash,
                )
            else:
                assert sequence is not None
                log.commit_aborted(
                    sequence=sequence, row_id="row-1",
                    payload=bounded_abort_payload(
                        reason_code="RESULT_PROTOCOL_VIOLATION",
                        actual_bytes=0,
                        diagnostic="injected",
                    ),
                    fault=crash,
                )

    trailer_written = boundary in {
        "after_commit_trailer", "before_commit_sync", "after_commit_sync"
    }
    before = scan_wal(path)
    assert before.has_uncommitted_tail is (not trailer_written)
    with WorkerEventLog.open_exclusive_and_repair(
        path, identity=_identity()
    ):
        pass
    recovered = scan_wal(path)
    assert recovered.has_uncommitted_tail is False
    committed_types = [record.event_type for record in recovered.records]
    expected = [] if event_type == "START" else ["TASK_STARTED"]
    if trailer_written:
        expected.append(
            {
                "START": "TASK_STARTED",
                "RESULT": TASK_RESULT,
                "ABORTED": TASK_ABORTED,
            }[event_type]
        )
    assert committed_types == expected


@pytest.mark.slow
def test_wal_bytes_are_linear_and_inode_count_is_task_count_independent(
    tmp_path: Path, monkeypatch,
):
    # Preserve the production encoder/write path while avoiding 200k physical
    # fsync calls in a local complexity test.  Durability boundaries are
    # exercised independently by the crash matrix above.
    monkeypatch.setattr("aleatoric_nk_grid.worker_event_wal._sync_fd", lambda _: None)
    measurements: list[tuple[int, int, int]] = []
    for task_count in (10, 1_000, 50_000):
        root = tmp_path / str(task_count)
        path = root / "worker.events.wal"
        root.mkdir()
        with WorkerEventLog.open_exclusive_and_repair(
            path, identity=_identity()
        ) as log:
            for index in range(task_count):
                row_id = f"row-{index:05d}"
                sequence = log.commit_started(row_id=row_id)
                log.commit_result(
                    sequence=sequence, row_id=row_id, public_rows=[_row()]
                )
        inode_count = sum(1 for item in root.rglob("*") if item.is_file())
        measurements.append((task_count, path.stat().st_size, inode_count))

    assert [inode_count for _, _, inode_count in measurements] == [1, 1, 1]
    small_slope = (measurements[1][1] - measurements[0][1]) / (
        measurements[1][0] - measurements[0][0]
    )
    large_slope = (measurements[2][1] - measurements[1][1]) / (
        measurements[2][0] - measurements[1][0]
    )
    assert 0.8 <= large_slope / small_slope <= 1.25


def test_terminal_events_cannot_be_duplicated_in_one_writer(tmp_path: Path):
    path = tmp_path / "worker.events.wal"
    with WorkerEventLog.open_exclusive_and_repair(path, identity=_identity()) as log:
        sequence = log.commit_started(row_id="row-1")
        log.commit_result(sequence=sequence, row_id="row-1", public_rows=[_row()])
        with pytest.raises(WALProtocolError, match="duplicates"):
            log.commit_result(sequence=sequence, row_id="row-1", public_rows=[_row()])


def test_dynamic_worker_writes_one_wal_and_reuses_session_path(tmp_path: Path, monkeypatch):
    frame = pd.DataFrame({"x": np.arange(30, dtype=float), "y": np.arange(30, dtype=float)})
    schema = write_schema_bundle(tmp_path / "input", frame, predictors=["x"])
    config = NKGridConfig(
        schema=schema, out=tmp_path / "final.csv", outcome="y", models=("ols",),
        seed=1, test_size=0.2, n_seeds=1, n_draws=1, n_sizes_n=1, n_sizes_k=1,
        max_n=10, max_k=1, batch_size=1, n_jobs=4, repeat_plan=((1, 0),), min_n=2,
    )
    build_dynamic_plan(
        config, n_grid=(10,), k_grid=(1,),
        cluster=ClusterPolicy(workers=1, rounds=1, partition="test", time_limit="01:00:00", account="test", constraint="none"),
        table_path=tmp_path / "tasks.parquet", snapshot_path=tmp_path / "snapshot.json",
        output_dir=tmp_path / "out", panel="generic",
    )
    prepared = prepare_round(
        tmp_path / "snapshot.json", round_index=1, prep_token="job-1", prep_job_id="job-1",
        submission_generation="generation-1", expected_pointer_version=0,
    )
    assert recover_generation_activation(
        tmp_path / "snapshot.json", round_index=1, submission_generation="generation-1",
        expected_prep_token="job-1", prep_job_id="job-1", expected_pointer_version=0,
    ) == Path(str(prepared["activation"]))
    wal = run_slice(
        tmp_path / "snapshot.json", round_index=1, worker_index=0, expected_prep_token="job-1",
        prep_job_id="job-1", submission_generation="generation-1", expected_pointer_version=0,
    )
    assert wal.name == "worker-0.events.wal"
    assert not list((tmp_path / "out").rglob("*.csv"))
    close_generation(
        tmp_path / "snapshot.json", round_index=1, submission_generation="generation-1",
        expected_prep_token="job-1", prep_job_id="job-1", expected_pointer_version=0,
    )
    original_open_shared = WorkerEventLog.open_shared
    scans = 0

    def counted_open_shared(*args, **kwargs):
        nonlocal scans
        scans += 1
        return original_open_shared(*args, **kwargs)

    monkeypatch.setattr(WorkerEventLog, "open_shared", staticmethod(counted_open_shared))
    verification = verify_rounds(
        tmp_path / "snapshot.json", round_index=1, submission_generation="generation-1",
        expected_prep_token="job-1", prep_job_id="job-1", expected_pointer_version=0,
    )
    assert prepared["todo_rows"] == 1
    assert verification["exit_code"] == 0
    assert scans == 1
    receipt = Path(str(verification["verification_receipt"]))
    original_receipt = receipt.read_bytes()
    tampered = json.loads(original_receipt)
    tampered["sealed_history_digest_sha256"] = "0" * 64
    receipt.write_text(json.dumps(tampered, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    scratch = tmp_path / "final-scratch"
    with pytest.raises(ControlProtocolError, match="verification receipt"):
        finalize_snapshot(
            tmp_path / "snapshot.json", round_index=1, submission_generation="generation-1",
            expected_prep_token="job-1", prep_job_id="job-1", expected_pointer_version=0,
            tmp_dir=scratch,
        )
    assert not scratch.exists()
    assert not config.out.exists()
    receipt.write_bytes(original_receipt)
    final = finalize_snapshot(
        tmp_path / "snapshot.json", round_index=1, submission_generation="generation-1",
        expected_prep_token="job-1", prep_job_id="job-1", expected_pointer_version=0,
    )
    assert final["final_rows"] == 1


def test_dynamic_final_csv_uses_the_local_stable_public_projection(tmp_path: Path):
    """Timing/RSS telemetry may differ, but the final scientific CSV may not."""

    frame = pd.DataFrame({"x": np.arange(40, dtype=float), "y": np.arange(40, dtype=float)})
    schema = write_schema_bundle(tmp_path / "input", frame, predictors=["x"])
    config = NKGridConfig(
        schema=schema, out=tmp_path / "final.csv", outcome="y", models=("ols",),
        seed=1, test_size=0.2, n_seeds=1, n_draws=1, n_sizes_n=1, n_sizes_k=1,
        max_n=10, max_k=1, batch_size=1, n_jobs=4, repeat_plan=((1, 0),), min_n=2,
    )
    run_nk_grid(config)
    local_bytes = config.out.read_bytes()
    build_dynamic_plan(
        config, n_grid=(10,), k_grid=(1,),
        cluster=ClusterPolicy(workers=1, rounds=1, partition="test", time_limit="01:00:00", account="test", constraint="none"),
        table_path=tmp_path / "tasks.parquet", snapshot_path=tmp_path / "snapshot.json",
        output_dir=tmp_path / "out", panel="generic",
    )
    prepare_round(tmp_path / "snapshot.json", round_index=1, prep_token="job-1", prep_job_id="job-1", submission_generation="g1", expected_pointer_version=0)
    run_slice(tmp_path / "snapshot.json", round_index=1, worker_index=0, expected_prep_token="job-1", prep_job_id="job-1", submission_generation="g1", expected_pointer_version=0)
    close_generation(tmp_path / "snapshot.json", round_index=1, submission_generation="g1", expected_prep_token="job-1", prep_job_id="job-1", expected_pointer_version=0)
    verify_rounds(tmp_path / "snapshot.json", round_index=1, submission_generation="g1", expected_prep_token="job-1", prep_job_id="job-1", expected_pointer_version=0)
    finalize_snapshot(tmp_path / "snapshot.json", round_index=1, submission_generation="g1", expected_prep_token="job-1", prep_job_id="job-1", expected_pointer_version=0)
    assert config.out.read_bytes() == local_bytes


def test_sealed_abandoned_uninitialized_wal_is_a_valid_incomplete_history(tmp_path: Path):
    frame = pd.DataFrame({"x": np.arange(30, dtype=float), "y": np.arange(30, dtype=float)})
    schema = write_schema_bundle(tmp_path / "input", frame, predictors=["x"])
    config = NKGridConfig(
        schema=schema, out=tmp_path / "final.csv", outcome="y", models=("ols",),
        seed=1, test_size=0.2, n_seeds=1, n_draws=1, n_sizes_n=1, n_sizes_k=1,
        max_n=10, max_k=1, batch_size=1, n_jobs=1, repeat_plan=((1, 0),), min_n=2,
    )
    plan = build_dynamic_plan(
        config, n_grid=(10,), k_grid=(1,),
        cluster=ClusterPolicy(workers=1, rounds=1, partition="test", time_limit="01:00:00", account="test", constraint="none"),
        table_path=tmp_path / "tasks.parquet", snapshot_path=tmp_path / "snapshot.json",
        output_dir=tmp_path / "out", panel="generic",
    )
    prepare_round(tmp_path / "snapshot.json", round_index=1, prep_token="job", prep_job_id="job", submission_generation="g1", expected_pointer_version=0)
    generation = tmp_path / "out" / "executions" / str(plan["execution_plan_id"]) / "round-1" / "generation-g1"
    (generation / "worker-0.events.wal").touch()
    close_generation(tmp_path / "snapshot.json", round_index=1, submission_generation="g1", expected_prep_token="job", prep_job_id="job", expected_pointer_version=0)
    verification = verify_rounds(tmp_path / "snapshot.json", round_index=1, submission_generation="g1", expected_prep_token="job", prep_job_id="job", expected_pointer_version=0)
    assert verification["exit_code"] == 3
    assert verification["interrupted_row_ids"] == []


def test_verify_fails_closed_when_sealed_assignment_changes(tmp_path: Path):
    frame = pd.DataFrame({"x": np.arange(30, dtype=float), "y": np.arange(30, dtype=float)})
    schema = write_schema_bundle(tmp_path / "input", frame, predictors=["x"])
    config = NKGridConfig(
        schema=schema, out=tmp_path / "final.csv", outcome="y", models=("ols",),
        seed=1, test_size=0.2, n_seeds=1, n_draws=1, n_sizes_n=1, n_sizes_k=1,
        max_n=10, max_k=1, batch_size=1, n_jobs=1, repeat_plan=((1, 0),), min_n=2,
    )
    plan = build_dynamic_plan(
        config, n_grid=(10,), k_grid=(1,),
        cluster=ClusterPolicy(workers=1, rounds=1, partition="test", time_limit="01:00:00", account="test", constraint="none"),
        table_path=tmp_path / "tasks.parquet", snapshot_path=tmp_path / "snapshot.json",
        output_dir=tmp_path / "out", panel="generic",
    )
    prepare_round(tmp_path / "snapshot.json", round_index=1, prep_token="job", prep_job_id="job", submission_generation="g1", expected_pointer_version=0)
    generation = tmp_path / "out" / "executions" / str(plan["execution_plan_id"]) / "round-1" / "generation-g1"
    (generation / "worker-0.events.wal").touch()
    close_generation(tmp_path / "snapshot.json", round_index=1, submission_generation="g1", expected_prep_token="job", prep_job_id="job", expected_pointer_version=0)
    assignment = generation / "assignment.parquet"
    assignment.chmod(0o644)
    assignment.write_bytes(assignment.read_bytes() + b"tamper")
    with pytest.raises(ControlProtocolError, match="assignment checksum"):
        verify_rounds(tmp_path / "snapshot.json", round_index=1, submission_generation="g1", expected_prep_token="job", prep_job_id="job", expected_pointer_version=0)


def test_new_execution_plan_reuses_a_sealed_predecessor_history(tmp_path: Path):
    frame = pd.DataFrame({"x": np.arange(40, dtype=float), "y": np.arange(40, dtype=float)})
    schema = write_schema_bundle(tmp_path / "input", frame, predictors=["x"])
    config = NKGridConfig(
        schema=schema, out=tmp_path / "final.csv", outcome="y", models=("ols",),
        seed=1, test_size=0.2, n_seeds=1, n_draws=1, n_sizes_n=1, n_sizes_k=1,
        max_n=10, max_k=1, batch_size=1, n_jobs=1, repeat_plan=((1, 0),), min_n=2,
    )
    first = build_dynamic_plan(
        config, n_grid=(10,), k_grid=(1,),
        cluster=ClusterPolicy(workers=1, rounds=1, partition="test", time_limit="01:00:00", account="test", constraint="none"),
        table_path=tmp_path / "first.parquet", snapshot_path=tmp_path / "first.json",
        output_dir=tmp_path / "out", panel="generic",
    )
    prepare_round(tmp_path / "first.json", round_index=1, prep_token="first-job", prep_job_id="first-job", submission_generation="first-g", expected_pointer_version=0)
    run_slice(tmp_path / "first.json", round_index=1, worker_index=0, expected_prep_token="first-job", prep_job_id="first-job", submission_generation="first-g", expected_pointer_version=0)
    close_generation(tmp_path / "first.json", round_index=1, submission_generation="first-g", expected_prep_token="first-job", prep_job_id="first-job", expected_pointer_version=0)

    second = build_dynamic_plan(
        config, n_grid=(10,), k_grid=(1,),
        cluster=ClusterPolicy(workers=2, rounds=1, partition="test", time_limit="01:00:00", account="test", constraint="none"),
        table_path=tmp_path / "second.parquet", snapshot_path=tmp_path / "second.json",
        output_dir=tmp_path / "out", panel="generic",
    )
    assert first["analysis_id"] == second["analysis_id"]
    outcome = prepare_round(
        tmp_path / "second.json", round_index=1, prep_token="second-job", prep_job_id="second-job",
        submission_generation="second-g", expected_previous_generation="first-g", expected_pointer_version=1,
        expected_previous_execution_plan_id=str(first["execution_plan_id"]), expected_previous_round_index=1,
    )
    assert outcome["no_generation"] is True
    verification = verify_rounds(
        tmp_path / "second.json", round_index=1, submission_generation="second-g",
        expected_prep_token="second-job", prep_job_id="second-job", expected_previous_generation="first-g",
        expected_pointer_version=1, expected_previous_execution_plan_id=str(first["execution_plan_id"]),
        expected_previous_round_index=1,
    )
    assert verification["exit_code"] == 0


def test_exact_prep_resumes_an_empty_unprepared_staging_directory(tmp_path: Path):
    frame = pd.DataFrame({"x": np.arange(30, dtype=float), "y": np.arange(30, dtype=float)})
    schema = write_schema_bundle(tmp_path / "input", frame, predictors=["x"])
    config = NKGridConfig(
        schema=schema, out=tmp_path / "final.csv", outcome="y", models=("ols",),
        seed=1, test_size=0.2, n_seeds=1, n_draws=1, n_sizes_n=1, n_sizes_k=1,
        max_n=10, max_k=1, batch_size=1, n_jobs=1, repeat_plan=((1, 0),), min_n=2,
    )
    build_dynamic_plan(
        config, n_grid=(10,), k_grid=(1,),
        cluster=ClusterPolicy(workers=1, rounds=1, partition="test", time_limit="01:00:00", account="test", constraint="none"),
        table_path=tmp_path / "tasks.parquet", snapshot_path=tmp_path / "snapshot.json",
        output_dir=tmp_path / "out", panel="generic",
    )
    snapshot = json.loads((tmp_path / "snapshot.json").read_text(encoding="utf-8"))
    target = ActivationTarget(
        analysis_id=str(snapshot["analysis_id"]), execution_plan_id=str(snapshot["execution_plan_id"]),
        execution_contract_sha256=str(snapshot["execution_contract_sha256"]), round_index=1,
        submission_generation="g1", expected_previous_generation=None, expected_pointer_version=0,
        prep_job_id="job", prep_token="job",
    )
    publish_activation_intent(tmp_path / "out", target)
    stage = tmp_path / "out" / "executions" / str(snapshot["execution_plan_id"]) / "round-1" / ".generation-g1.staging"
    stage.mkdir(parents=True)
    # A crash before publishing any child leaves an empty staging directory;
    # exact prep may safely fill the missing deterministic children.
    prepared = prepare_round(tmp_path / "snapshot.json", round_index=1, prep_token="job", prep_job_id="job", submission_generation="g1", expected_pointer_version=0)
    assert Path(str(prepared["activation"])).is_file()


def test_exact_prep_preserves_conflicting_staging_assignment_evidence(tmp_path: Path):
    frame = pd.DataFrame({"x": np.arange(30, dtype=float), "y": np.arange(30, dtype=float)})
    schema = write_schema_bundle(tmp_path / "input", frame, predictors=["x"])
    config = NKGridConfig(
        schema=schema, out=tmp_path / "final.csv", outcome="y", models=("ols",),
        seed=1, test_size=0.2, n_seeds=1, n_draws=1, n_sizes_n=1, n_sizes_k=1,
        max_n=10, max_k=1, batch_size=1, n_jobs=1, repeat_plan=((1, 0),), min_n=2,
    )
    build_dynamic_plan(
        config, n_grid=(10,), k_grid=(1,),
        cluster=ClusterPolicy(workers=1, rounds=1, partition="test", time_limit="01:00:00", account="test", constraint="none"),
        table_path=tmp_path / "tasks.parquet", snapshot_path=tmp_path / "snapshot.json",
        output_dir=tmp_path / "out", panel="generic",
    )
    snapshot = json.loads((tmp_path / "snapshot.json").read_text(encoding="utf-8"))
    target = ActivationTarget(
        analysis_id=str(snapshot["analysis_id"]), execution_plan_id=str(snapshot["execution_plan_id"]),
        execution_contract_sha256=str(snapshot["execution_contract_sha256"]), round_index=1,
        submission_generation="g1", expected_previous_generation=None, expected_pointer_version=0,
        prep_job_id="job", prep_token="job",
    )
    publish_activation_intent(tmp_path / "out", target)
    stage = tmp_path / "out" / "executions" / str(snapshot["execution_plan_id"]) / "round-1" / ".generation-g1.staging"
    stage.mkdir(parents=True)
    conflicting = stage / "assignment.parquet"
    conflicting.write_bytes(b"partial")
    with pytest.raises(ControlProtocolError, match="staging assignment differs"):
        prepare_round(tmp_path / "snapshot.json", round_index=1, prep_token="job", prep_job_id="job", submission_generation="g1", expected_pointer_version=0)
    assert conflicting.read_bytes() == b"partial"


def test_analysis_identity_freezes_commit_environment_groups_and_input_provenance(tmp_path: Path, monkeypatch):
    frame = pd.DataFrame({"x": np.arange(30, dtype=float), "y": np.arange(30, dtype=float)})
    schema = write_schema_bundle(tmp_path / "input", frame, predictors=["x"])
    config = NKGridConfig(
        schema=schema, out=tmp_path / "final.csv", outcome="y", models=("ols",),
        seed=1, test_size=0.2, n_seeds=1, n_draws=1, n_sizes_n=1, n_sizes_k=1,
        max_n=10, max_k=1, batch_size=1, n_jobs=1, repeat_plan=((1, 0),), min_n=2,
    )
    cluster = ClusterPolicy(workers=1, rounds=1, partition="test", time_limit="01:00:00", account="test", constraint="none")
    first = build_dynamic_plan(
        config, n_grid=(10,), k_grid=(1,), cluster=cluster,
        table_path=tmp_path / "first.parquet", snapshot_path=tmp_path / "first.json", output_dir=tmp_path / "first-out", panel="generic",
    )
    contract = json.loads((tmp_path / "first-out" / "analysis-contract.json").read_text(encoding="utf-8"))
    spec = contract["cell_execution_spec"]
    assert len(spec["git_commit"]) == 40
    assert spec["resolved_model_params"]
    assert spec["execution_groups"]
    assert {"training_table", "feature_universe_definition"}.issubset(spec["input_provenance"])
    assert not Path(spec["schema_locator"]).is_absolute()
    assert all(not Path(entry["path"]).is_absolute() for entry in spec["input_provenance"].values())
    assert all(".." not in Path(entry["path"]).parts for entry in spec["input_provenance"].values())

    monkeypatch.setenv("RF_N_ESTIMATORS", "17")
    second = build_dynamic_plan(
        config, n_grid=(10,), k_grid=(1,), cluster=cluster,
        table_path=tmp_path / "second.parquet", snapshot_path=tmp_path / "second.json", output_dir=tmp_path / "second-out", panel="generic",
    )
    assert second["analysis_id"] != first["analysis_id"]


def test_worker_opens_one_execution_session_for_many_tasks(tmp_path: Path, monkeypatch):
    frame = pd.DataFrame({"x": np.arange(220, dtype=float), "y": np.arange(220, dtype=float)})
    schema = write_schema_bundle(tmp_path / "input", frame, predictors=["x"])
    repeats = tuple((seed, 0) for seed in range(1, 101))
    config = NKGridConfig(
        schema=schema, out=tmp_path / "final.csv", outcome="y", models=("ols",),
        seed=1, test_size=0.2, n_seeds=1, n_draws=1, n_sizes_n=1, n_sizes_k=1,
        max_n=10, max_k=1, batch_size=1, n_jobs=4, repeat_plan=repeats, min_n=2,
    )
    build_dynamic_plan(
        config, n_grid=(10,), k_grid=(1,),
        cluster=ClusterPolicy(workers=1, rounds=1, partition="test", time_limit="01:00:00", account="test", constraint="none"),
        table_path=tmp_path / "tasks.parquet", snapshot_path=tmp_path / "snapshot.json",
        output_dir=tmp_path / "out", panel="generic",
    )
    prepared = prepare_round(
        tmp_path / "snapshot.json", round_index=1, prep_token="job-1", prep_job_id="job-1",
        submission_generation="generation-1", expected_pointer_version=0,
    )
    snapshot = json.loads((tmp_path / "snapshot.json").read_text(encoding="utf-8"))
    opens = 0
    fit_calls = 0
    init_counts = {"load_input": 0, "validate_input": 0, "load_model_params": 0, "native_runner": 0}
    original_init = NKGridExecutionSession.__init__
    original_load_input = ng.load_input
    original_validate_input = ng.validate_input
    original_load_model_params = ng.load_model_params
    original_runner_init = ng.IsolatedProcessRunner.__init__

    def counted_init(session, *args, **kwargs):
        nonlocal opens
        opens += 1
        return original_init(session, *args, **kwargs)

    def counted_load_input(*args, **kwargs):
        init_counts["load_input"] += 1
        return original_load_input(*args, **kwargs)

    def counted_validate_input(*args, **kwargs):
        init_counts["validate_input"] += 1
        return original_validate_input(*args, **kwargs)

    def counted_load_model_params(*args, **kwargs):
        init_counts["load_model_params"] += 1
        return original_load_model_params(*args, **kwargs)

    def counted_runner_init(runner, *args, **kwargs):
        init_counts["native_runner"] += 1
        return original_runner_init(runner, *args, **kwargs)

    def deterministic_fit(**kwargs):
        nonlocal fit_calls
        fit_calls += 1
        y_train = np.asarray(kwargs["y_train"], dtype=float)
        return {
            "predictions": np.full(len(kwargs["X_test"]), float(y_train.mean())),
            "fit_seconds": 0.0,
            "best_rounds": None,
            "converged": True,
            "solver": "test-deterministic",
            "iterations": None,
            "alpha": None,
            "peak_rss_bytes": 0,
        }

    monkeypatch.setattr(ng.NKGridExecutionSession, "__init__", counted_init)
    monkeypatch.setattr(ng, "load_input", counted_load_input)
    monkeypatch.setattr(ng, "validate_input", counted_validate_input)
    monkeypatch.setattr(ng, "load_model_params", counted_load_model_params)
    monkeypatch.setattr(ng.IsolatedProcessRunner, "__init__", counted_runner_init)
    monkeypatch.setattr(ng, "_fit_predict_model_cell", deterministic_fit)
    run_slice(
        tmp_path / "snapshot.json", round_index=1, worker_index=0, expected_prep_token="job-1",
        prep_job_id="job-1", submission_generation="generation-1", expected_pointer_version=0,
    )
    wal = tmp_path / "out" / "executions" / snapshot["execution_plan_id"] / "round-1" / "generation-generation-1" / "worker-0.events.wal"
    assert prepared["todo_rows"] == 100
    assert opens == 1
    assert fit_calls == 100
    assert init_counts == {
        "load_input": 1, "validate_input": 1,
        "load_model_params": 1, "native_runner": 1,
    }
    assert len(scan_wal(wal).records) == 200


def test_production_afterany_entries_share_exact_target_lifecycle(tmp_path: Path):
    frame = pd.DataFrame({"x": np.arange(40, dtype=float), "y": np.arange(40, dtype=float)})
    schema = write_schema_bundle(tmp_path / "input", frame, predictors=["x"])
    config = NKGridConfig(
        schema=schema, out=tmp_path / "final.csv", outcome="y", models=("ols",),
        seed=1, test_size=0.2, n_seeds=1, n_draws=1, n_sizes_n=1,
        n_sizes_k=1, max_n=10, max_k=1, batch_size=1, n_jobs=1,
        repeat_plan=((1, 0),), min_n=2,
    )
    plan = build_dynamic_plan(
        config, n_grid=(10,), k_grid=(1,),
        cluster=ClusterPolicy(
            workers=1, rounds=2, partition="test", time_limit="01:00:00",
            account="test", constraint="none",
        ),
        table_path=tmp_path / "tasks.parquet",
        snapshot_path=tmp_path / "snapshot.json",
        output_dir=tmp_path / "out", panel="generic",
    )
    snapshot = tmp_path / "snapshot.json"
    prepared = prepare_round(
        snapshot, round_index=1, prep_token="job-1", prep_job_id="job-1",
        submission_generation="g1", expected_pointer_version=0,
    )
    with pytest.raises(ControlBusyError):
        verify_rounds(
            snapshot, round_index=1, submission_generation="g1",
            expected_prep_token="job-1", prep_job_id="job-1",
            expected_pointer_version=0,
        )
    wal = run_slice(
        snapshot, round_index=1, worker_index=0, expected_prep_token="job-1",
        prep_job_id="job-1", submission_generation="g1", expected_pointer_version=0,
    )
    assert wal.is_file()
    closed = close_generation(
        snapshot, round_index=1, submission_generation="g1",
        expected_prep_token="job-1", prep_job_id="job-1",
        expected_pointer_version=0,
    )
    assert closed.is_file()
    verified = verify_rounds(
        snapshot, round_index=1, submission_generation="g1",
        expected_prep_token="job-1", prep_job_id="job-1",
        expected_pointer_version=0,
    )
    assert verified["exit_code"] == 0
    final = finalize_snapshot(
        snapshot, round_index=1, submission_generation="g1",
        expected_prep_token="job-1", prep_job_id="job-1",
        expected_pointer_version=0,
    )
    assert final["final_rows"] == 1
    next_prep = prepare_round(
        snapshot, round_index=2, prep_token="job-2", prep_job_id="job-2",
        submission_generation="g2", expected_previous_generation="g1",
        expected_pointer_version=1, expected_previous_execution_plan_id=str(plan["execution_plan_id"]),
        expected_previous_round_index=1,
    )
    assert next_prep["no_generation"] is True


def test_next_prep_missing_predecessor_is_a_zero_write_failure(tmp_path: Path):
    frame = pd.DataFrame({"x": np.arange(30, dtype=float), "y": np.arange(30, dtype=float)})
    schema = write_schema_bundle(tmp_path / "input", frame, predictors=["x"])
    config = NKGridConfig(
        schema=schema, out=tmp_path / "final.csv", outcome="y", models=("ols",),
        seed=1, test_size=0.2, n_seeds=1, n_draws=1, n_sizes_n=1,
        n_sizes_k=1, max_n=10, max_k=1, batch_size=1, n_jobs=1,
        repeat_plan=((1, 0),), min_n=2,
    )
    build_dynamic_plan(
        config, n_grid=(10,), k_grid=(1,),
        cluster=ClusterPolicy(
            workers=1, rounds=2, partition="test", time_limit="01:00:00",
            account="test", constraint="none",
        ),
        table_path=tmp_path / "tasks.parquet", snapshot_path=tmp_path / "snapshot.json",
        output_dir=tmp_path / "out", panel="generic",
    )
    root = tmp_path / "out"

    def tree_state() -> dict[str, tuple[str, int, bytes | None]]:
        state: dict[str, tuple[str, int, bytes | None]] = {}
        for path in sorted(root.rglob("*")):
            relative = str(path.relative_to(root))
            if path.is_dir():
                state[relative] = ("dir", path.stat().st_ino, None)
            else:
                state[relative] = ("file", path.stat().st_ino, path.read_bytes())
        return state

    before = tree_state()
    with pytest.raises(ControlBusyError, match="predecessor pointer is absent"):
        prepare_round(
            tmp_path / "snapshot.json", round_index=2, prep_token="job-2",
            prep_job_id="job-2", submission_generation="g2",
            expected_previous_generation="g1", expected_pointer_version=1,
        )
    assert tree_state() == before


@pytest.mark.parametrize(
    "fact,expected_error",
    [
        ("missing", ControlBusyError),
        ("superseded", ControlSupersededError),
        ("corruption", ControlProtocolError),
    ],
)
def test_next_prep_non_normal_facts_are_zero_write(
    tmp_path: Path, fact: str, expected_error: type[Exception],
):
    snapshot, _ = _fault_plan(tmp_path, rounds=2)
    root = tmp_path / "out"
    if fact == "superseded":
        prepare_round(
            snapshot, round_index=1, prep_token="job-1", prep_job_id="job-1",
            submission_generation="other", expected_pointer_version=0,
        )
    elif fact == "corruption":
        prepare_round(
            snapshot, round_index=1, prep_token="job-1", prep_job_id="job-1",
            submission_generation="g1", expected_pointer_version=0,
        )
        pointer = root / "active-generation.json"
        pointer_payload = json.loads(pointer.read_text(encoding="utf-8"))
        pointer_payload["generation_activation_sha256"] = "0" * 64
        pointer.write_bytes(canonical_json_bytes(pointer_payload) + b"\n")

    def tree_state() -> dict[str, tuple[str, int, bytes | None]]:
        state: dict[str, tuple[str, int, bytes | None]] = {}
        for path in sorted(root.rglob("*")):
            relative = str(path.relative_to(root))
            state[relative] = (
                "dir", path.stat().st_ino, None,
            ) if path.is_dir() else (
                "file", path.stat().st_ino, path.read_bytes(),
            )
        return state

    before = tree_state()
    with pytest.raises(expected_error):
        prepare_round(
            snapshot, round_index=2, prep_token="job-2", prep_job_id="job-2",
            submission_generation="g2", expected_previous_generation="g1",
            expected_pointer_version=1,
        )
    assert tree_state() == before


@pytest.mark.parametrize("fact", ["missing", "superseded", "corruption"])
def test_finalizer_non_normal_exact_target_is_zero_write(tmp_path: Path, fact: str):
    snapshot, _ = _fault_plan(tmp_path)
    root = tmp_path / "out"
    if fact == "superseded":
        prepare_round(
            snapshot, round_index=1, prep_token="job-1", prep_job_id="job-1",
            submission_generation="other", expected_pointer_version=0,
        )
    elif fact == "corruption":
        prepare_round(
            snapshot, round_index=1, prep_token="job-1", prep_job_id="job-1",
            submission_generation="g1", expected_pointer_version=0,
        )
        pointer = root / "active-generation.json"
        pointer_payload = json.loads(pointer.read_text(encoding="utf-8"))
        pointer_payload["generation_activation_sha256"] = "0" * 64
        pointer.write_bytes(canonical_json_bytes(pointer_payload) + b"\n")

    def tree_state() -> dict[str, tuple[str, int, bytes | None]]:
        state: dict[str, tuple[str, int, bytes | None]] = {}
        for path in sorted(root.rglob("*")):
            relative = str(path.relative_to(root))
            state[relative] = ("dir", path.stat().st_ino, None) if path.is_dir() else (
                "file", path.stat().st_ino, path.read_bytes(),
            )
        return state

    before = tree_state()
    with pytest.raises(FinalizationError):
        finalize_snapshot(
            snapshot, round_index=1, submission_generation="g1",
            expected_prep_token="job-1", prep_job_id="job-1",
            expected_pointer_version=0, tmp_dir=tmp_path / "final-scratch",
        )
    assert tree_state() == before
    assert not (tmp_path / "final-scratch").exists()


class _InjectedPrepareCrash(RuntimeError):
    pass


@pytest.mark.parametrize(
    "fault_label",
    [
        "after_intent",
        "after_staging_lease_fsync",
        "after_staging_parent_fsync",
        "after_assignment_publish",
        "after_index_publish",
        "after_prep_publish",
        "after_ready_publish",
        "after_prepared_publish",
        "before_generation_rename",
        "after_generation_rename",
        "before_generation_parent_fsync",
        "after_generation_parent_fsync",
        "before_file_fsync",
        "after_file_fsync",
        "before_rename",
        "after_rename",
        "before_parent_fsync",
        "after_parent_fsync",
        "pointer_before_file_fsync",
        "pointer_after_file_fsync",
        "pointer_before_replace",
        "pointer_after_replace",
        "pointer_before_root_fsync",
        "pointer_after_root_fsync",
    ],
)
def test_production_prepare_fault_boundaries_recover_exact_generation(
    tmp_path: Path, fault_label: str,
):
    snapshot, _ = _fault_plan(tmp_path)
    triggered = False

    def fault(label: str) -> None:
        nonlocal triggered
        if label == fault_label and not triggered:
            triggered = True
            raise _InjectedPrepareCrash(label)

    with pytest.raises(_InjectedPrepareCrash, match=fault_label):
        prepare_round(
            snapshot, round_index=1, prep_token="job-1", prep_job_id="job-1",
            submission_generation="g1", expected_pointer_version=0, fault=fault,
        )
    recovered = recover_generation_activation(
        snapshot, round_index=1, submission_generation="g1",
        expected_prep_token="job-1", prep_job_id="job-1", expected_pointer_version=0,
    )
    assert recovered.is_file()
    assert triggered is True


@pytest.mark.parametrize(
    "fault_label",
    [
        "before_file_fsync", "after_file_fsync", "before_rename",
        "after_rename", "before_parent_fsync", "after_parent_fsync",
    ],
)
def test_production_todo_zero_outcome_fault_boundaries_recover_exact_receipt(
    tmp_path: Path, fault_label: str,
):
    snapshot, _ = _fault_plan(tmp_path, rounds=2)
    prepare_round(
        snapshot, round_index=1, prep_token="job-1", prep_job_id="job-1",
        submission_generation="g1", expected_pointer_version=0,
    )
    run_slice(
        snapshot, round_index=1, worker_index=0, expected_prep_token="job-1",
        prep_job_id="job-1", submission_generation="g1", expected_pointer_version=0,
    )
    close_generation(
        snapshot, round_index=1, submission_generation="g1",
        expected_prep_token="job-1", prep_job_id="job-1", expected_pointer_version=0,
    )
    triggered = False

    def fault(label: str) -> None:
        nonlocal triggered
        if label == fault_label and not triggered:
            triggered = True
            raise _InjectedPrepareCrash(label)

    with pytest.raises(_InjectedPrepareCrash, match=fault_label):
        prepare_round(
            snapshot, round_index=2, prep_token="job-2", prep_job_id="job-2",
            submission_generation="g2", expected_previous_generation="g1",
            expected_pointer_version=1, fault=fault,
        )
    recovered = prepare_round(
        snapshot, round_index=2, prep_token="job-2", prep_job_id="job-2",
        submission_generation="g2", expected_previous_generation="g1",
        expected_pointer_version=1,
    )
    assert recovered["no_generation"] is True
    assert Path(str(recovered["prep_outcome"])).is_file()
    assert triggered is True


@pytest.mark.parametrize("entry", [run_slice, close_generation, verify_rounds])
@pytest.mark.parametrize("fact", ["missing", "superseded", "corruption"])
def test_production_afterany_entries_fail_closed_without_control_tree_writes(
    tmp_path: Path, entry, fact: str,
):
    frame = pd.DataFrame({"x": np.arange(30, dtype=float), "y": np.arange(30, dtype=float)})
    schema = write_schema_bundle(tmp_path / "input", frame, predictors=["x"])
    config = NKGridConfig(
        schema=schema, out=tmp_path / "final.csv", outcome="y", models=("ols",),
        seed=1, test_size=0.2, n_seeds=1, n_draws=1, n_sizes_n=1,
        n_sizes_k=1, max_n=10, max_k=1, batch_size=1, n_jobs=1,
        repeat_plan=((1, 0),), min_n=2,
    )
    build_dynamic_plan(
        config, n_grid=(10,), k_grid=(1,),
        cluster=ClusterPolicy(
            workers=1, rounds=2, partition="test", time_limit="01:00:00",
            account="test", constraint="none",
        ),
        table_path=tmp_path / "tasks.parquet",
        snapshot_path=tmp_path / "snapshot.json",
        output_dir=tmp_path / "out", panel="generic",
    )
    snapshot = tmp_path / "snapshot.json"
    target_generation = "g1"
    if fact == "superseded":
        prepare_round(
            snapshot, round_index=1, prep_token="job-2", prep_job_id="job-2",
            submission_generation="g2", expected_pointer_version=0,
        )
    elif fact == "corruption":
        prepared = prepare_round(
            snapshot, round_index=1, prep_token="job-1", prep_job_id="job-1",
            submission_generation="g1", expected_pointer_version=0,
        )
        assignment = Path(str(prepared["assignment"]))
        assignment.chmod(0o644)
        assignment.write_bytes(assignment.read_bytes() + b"corruption")
        assignment.chmod(0o444)

    output_root = tmp_path / "out"
    def tree_state() -> dict[str, tuple[int, bytes]]:
        return {
            str(path.relative_to(output_root)): (path.stat().st_ino, path.read_bytes())
            for path in output_root.rglob("*") if path.is_file()
        }

    before = tree_state()
    kwargs = {
        "round_index": 1,
        "expected_prep_token": "job-1" if fact != "superseded" else "job-1",
        "prep_job_id": "job-1",
        "submission_generation": target_generation,
        "expected_pointer_version": 0,
    }
    expected_error = {
        "missing": ControlBusyError,
        "superseded": ControlSupersededError,
        "corruption": ControlProtocolError,
    }[fact]
    with pytest.raises(expected_error):
        if entry is run_slice:
            entry(snapshot, worker_index=0, **kwargs)
        else:
            entry(snapshot, **kwargs)
    assert tree_state() == before


def test_malformed_session_row_is_durably_aborted_not_interrupted(tmp_path: Path, monkeypatch):
    frame = pd.DataFrame({"x": np.arange(30, dtype=float), "y": np.arange(30, dtype=float)})
    schema = write_schema_bundle(tmp_path / "input", frame, predictors=["x"])
    config = NKGridConfig(
        schema=schema, out=tmp_path / "final.csv", outcome="y", models=("ols",),
        seed=1, test_size=0.2, n_seeds=1, n_draws=1, n_sizes_n=1, n_sizes_k=1,
        max_n=10, max_k=1, batch_size=1, n_jobs=1, repeat_plan=((1, 0),), min_n=2,
    )
    plan = build_dynamic_plan(
        config, n_grid=(10,), k_grid=(1,),
        cluster=ClusterPolicy(workers=1, rounds=1, partition="test", time_limit="01:00:00", account="test", constraint="none"),
        table_path=tmp_path / "tasks.parquet", snapshot_path=tmp_path / "snapshot.json",
        output_dir=tmp_path / "out", panel="generic",
    )
    prepare_round(tmp_path / "snapshot.json", round_index=1, prep_token="job", prep_job_id="job", submission_generation="g1", expected_pointer_version=0)

    class MalformedSession:
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def run_cell_group(self, **kwargs): return [None]

    monkeypatch.setattr(NKGridExecutionSession, "open", staticmethod(lambda *args, **kwargs: MalformedSession()))
    with pytest.raises(ControlProtocolError, match="PUBLIC_SCHEMA_MISMATCH"):
        run_slice(tmp_path / "snapshot.json", round_index=1, worker_index=0, expected_prep_token="job", prep_job_id="job", submission_generation="g1", expected_pointer_version=0)
    generation = tmp_path / "out" / "executions" / str(plan["execution_plan_id"]) / "round-1" / "generation-g1"
    scan = scan_wal(generation / "worker-0.events.wal")
    assert [record.event_type for record in scan.records] == ["TASK_STARTED", TASK_ABORTED]
    assert json.loads(scan.records[-1].payload.decode("utf-8"))["reason_code"] == "PUBLIC_SCHEMA_MISMATCH"
