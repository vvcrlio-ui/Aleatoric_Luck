from __future__ import annotations

import json
from pathlib import Path

import pytest

from aleatoric_nk_grid.generation_control import (
    ActivationTarget,
    RETRYABLE_EXIT_CODE,
    SUCCESS_EXIT_CODE,
    ControlProtocolError,
    ControlSupersededError,
    activate_generation,
    classify_exact_afterany_target_read_only,
    publish_activation_intent,
    publish_no_generation_outcome,
    seal_generation,
)
from aleatoric_nk_grid.execution_contract import canonical_json_bytes


def _target(*, generation: str = "g1", pointer_version: int = 0) -> ActivationTarget:
    return ActivationTarget(
        analysis_id="analysis", execution_plan_id="plan", execution_contract_sha256="contract",
        round_index=1, submission_generation=generation,
        expected_previous_generation=None, expected_pointer_version=pointer_version,
        prep_job_id="job", prep_token="token",
    )


def test_missing_exact_target_is_recovery_required_without_writes(tmp_path: Path):
    before = tuple(tmp_path.rglob("*"))
    dispatch = classify_exact_afterany_target_read_only(tmp_path, _target())
    assert dispatch.exit_code == RETRYABLE_EXIT_CODE
    assert tuple(tmp_path.rglob("*")) == before


def test_no_generation_outcome_is_only_todo_zero_success(tmp_path: Path):
    target = _target()
    outcome = publish_no_generation_outcome(tmp_path, target, sealed_history_frontier=[])
    dispatch = classify_exact_afterany_target_read_only(tmp_path, target)
    assert dispatch.kind == "no-generation"
    assert dispatch.exit_code == SUCCESS_EXIT_CODE
    assert dispatch.outcome_path == outcome
    assert not list(tmp_path.rglob("generation-*"))


def test_no_generation_predecessor_allows_next_exact_no_generation_without_pointer_advance(tmp_path: Path):
    first = _target(generation="g1", pointer_version=0)
    publish_no_generation_outcome(tmp_path, first, sealed_history_frontier=[])
    second = ActivationTarget(
        analysis_id="analysis", execution_plan_id="plan", execution_contract_sha256="contract",
        round_index=2, submission_generation="g2", expected_previous_generation="g1",
        expected_pointer_version=1, prep_job_id="job-2", prep_token="token-2",
    )
    publish_no_generation_outcome(tmp_path, second)
    assert classify_exact_afterany_target_read_only(tmp_path, second).kind == "no-generation"


def test_activation_pointer_is_single_commit_and_sealed_is_idempotent(tmp_path: Path):
    target = _target()
    publish_activation_intent(tmp_path, target)
    generation = tmp_path / "executions" / "plan" / "round-1" / "generation-g1"
    generation.mkdir(parents=True)
    # Production prep creates this inode in private staging before promotion.
    # The control-plane unit test constructs its promoted fixture directly.
    (generation / "generation.lease").touch()
    assignment = generation / "assignment.parquet"; assignment.write_bytes(b"assignment")
    index = generation / "assignment.index.json"
    index.write_bytes(canonical_json_bytes({"row_groups": [{"worker": 0, "row_count": 0, "canonical_task_rows_sha256": "empty"}]}) + b"\n")
    ready = generation / "assignment.ready.json"; ready.write_bytes(b"ready")
    prep = generation / "prep.json"; prep.write_bytes(b"prep")
    import hashlib
    digest = lambda file_item: hashlib.sha256(file_item.read_bytes()).hexdigest()
    activation = activate_generation(
        tmp_path, target, assignment_path=assignment, assignment_sha256=digest(assignment),
        assignment_index_path=index, assignment_index_sha256=digest(index), ready_path=ready,
        ready_sha256=digest(ready), prep_path=prep, prep_sha256=digest(prep), worker_count=1,
    )
    active = classify_exact_afterany_target_read_only(tmp_path, target)
    assert active.kind == "active-generation"
    assert active.activation_path == activation
    activation_payload = json.loads(activation.read_text(encoding="utf-8"))
    inventory = {
        "assignment_path": activation_payload["assignment_path"],
        "assignment_sha256": activation_payload["assignment_sha256"],
        "assignment_index_path": activation_payload["assignment_index_path"],
        "assignment_index_sha256": activation_payload["assignment_index_sha256"],
        "workers": [{"worker": 0, "wal_state": "absent"}],
    }
    closed = seal_generation(tmp_path, target, inventory=inventory)
    assert classify_exact_afterany_target_read_only(tmp_path, target).kind == "sealed-generation"
    assert seal_generation(tmp_path, target, inventory=inventory) == closed


def test_no_generation_corruption_or_activation_artefacts_are_protocol_not_success(tmp_path: Path):
    target = _target()
    outcome = publish_no_generation_outcome(tmp_path, target, sealed_history_frontier=[])
    payload = json.loads(outcome.read_text(encoding="utf-8"))
    payload["sealed_history_digest_sha256"] = "0" * 64
    outcome.write_bytes(canonical_json_bytes(payload) + b"\n")
    with pytest.raises(ControlProtocolError, match="sealed-history digest"):
        classify_exact_afterany_target_read_only(tmp_path, target)

    # Restore a valid exact outcome, then create a mutually-exclusive
    # generation directory.  The classifier must never reinterpret this as a
    # successful todo=0 receipt.
    outcome.unlink()
    publish_no_generation_outcome(tmp_path, target, sealed_history_frontier=[])
    (tmp_path / "executions" / "plan" / "round-1" / "generation-g1").mkdir(parents=True)
    dispatch = classify_exact_afterany_target_read_only(tmp_path, target)
    assert dispatch.kind == "protocol"
    assert dispatch.exit_code == 6


def test_pre_cas_activation_keeps_the_only_unresolved_intent_slot(tmp_path: Path):
    target = _target(generation="g1")
    publish_activation_intent(tmp_path, target)
    generation = tmp_path / "executions" / "plan" / "round-1" / "generation-g1"
    generation.mkdir(parents=True)
    (generation / "generation.lease").touch()
    assignment = generation / "assignment.parquet"; assignment.write_bytes(b"assignment")
    index = generation / "assignment.index.json"
    index.write_bytes(canonical_json_bytes({"row_groups": [{"worker": 0, "row_count": 0, "canonical_task_rows_sha256": "empty"}]}) + b"\n")
    ready = generation / "assignment.ready.json"; ready.write_bytes(b"ready")
    prep = generation / "prep.json"; prep.write_bytes(b"prep")
    import hashlib
    digest = lambda item: hashlib.sha256(item.read_bytes()).hexdigest()

    def crash_before_pointer_sync(boundary: str) -> None:
        if boundary == "pointer_before_file_fsync":
            raise RuntimeError("injected pre-CAS crash")

    with pytest.raises(RuntimeError, match="pre-CAS"):
        activate_generation(
            tmp_path, target, assignment_path=assignment, assignment_sha256=digest(assignment),
            assignment_index_path=index, assignment_index_sha256=digest(index), ready_path=ready,
            ready_sha256=digest(ready), prep_path=prep, prep_sha256=digest(prep), worker_count=1,
            fault=crash_before_pointer_sync,
        )
    second = _target(generation="g2")
    with pytest.raises(ControlSupersededError, match="another frozen target"):
        publish_activation_intent(tmp_path, second)
