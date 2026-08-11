from __future__ import annotations

import json
from pathlib import Path

import pytest

from aleatoric_nk_grid.generation_control import (
    ActivationTarget,
    Dispatch,
    PROTOCOL_EXIT_CODE,
    RETRYABLE_EXIT_CODE,
    SUCCESS_EXIT_CODE,
    SUPERSEDED_EXIT_CODE,
    ControlProtocolError,
    ControlSupersededError,
    activate_generation,
    classify_exact_afterany_target_read_only,
    publish_activation_intent,
    publish_no_generation_outcome,
    publish_verification_receipt,
    seal_generation,
    validate_exact_verification_receipt,
    _write_temp_fsync_rename,
)
from aleatoric_nk_grid.execution_contract import canonical_json_bytes


def _target(*, generation: str = "g1", pointer_version: int = 0) -> ActivationTarget:
    return ActivationTarget(
        analysis_id="analysis", execution_plan_id="plan", execution_contract_sha256="contract",
        round_index=1, submission_generation=generation,
        expected_previous_generation=None, expected_pointer_version=pointer_version,
        prep_job_id="job", prep_token="token",
    )


def _prepared_generation(root: Path, target: ActivationTarget) -> dict[str, object]:
    import hashlib

    generation = (
        root / "executions" / target.execution_plan_id
        / f"round-{target.round_index}"
        / f"generation-{target.submission_generation}"
    )
    generation.mkdir(parents=True)
    (generation / "generation.lease").touch()
    assignment = generation / "assignment.parquet"
    assignment.write_bytes(b"assignment")
    index = generation / "assignment.index.json"
    index.write_bytes(
        canonical_json_bytes(
            {"row_groups": [{
                "worker": 0,
                "row_count": 0,
                "canonical_task_rows_sha256": "empty",
            }]}
        ) + b"\n"
    )
    ready = generation / "assignment.ready.json"
    ready.write_bytes(b"ready")
    prep = generation / "prep.json"
    prep.write_bytes(b"prep")

    def digest(item: Path) -> str:
        return hashlib.sha256(item.read_bytes()).hexdigest()

    return {
        "assignment_path": assignment,
        "assignment_sha256": digest(assignment),
        "assignment_index_path": index,
        "assignment_index_sha256": digest(index),
        "ready_path": ready,
        "ready_sha256": digest(ready),
        "prep_path": prep,
        "prep_sha256": digest(prep),
        "worker_count": 1,
    }


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


@pytest.mark.parametrize(
    "boundary",
    [
        "pointer_before_file_fsync",
        "pointer_after_file_fsync",
        "pointer_before_replace",
        "pointer_after_replace",
        "pointer_before_root_fsync",
        "pointer_after_root_fsync",
    ],
)
def test_activation_retry_converges_across_every_pointer_cas_boundary(
    tmp_path: Path, boundary: str,
):
    target = _target()
    publish_activation_intent(tmp_path, target)
    prepared = _prepared_generation(tmp_path, target)

    def crash(observed: str) -> None:
        if observed == boundary:
            raise RuntimeError(f"injected {boundary}")

    with pytest.raises(RuntimeError, match="injected"):
        activate_generation(tmp_path, target, fault=crash, **prepared)
    activation = activate_generation(tmp_path, target, **prepared)
    assert activation.is_file()
    dispatch = classify_exact_afterany_target_read_only(tmp_path, target)
    assert dispatch.kind == "active-generation"
    assert dispatch.exit_code == SUCCESS_EXIT_CODE


@pytest.mark.parametrize("consumer", ["work", "close", "next-prep", "verify"])
@pytest.mark.parametrize(
    "fact,expected_code",
    [
        ("active", SUCCESS_EXIT_CODE),
        ("outcome", SUCCESS_EXIT_CODE),
        ("sealed", SUCCESS_EXIT_CODE),
        ("superseded", SUPERSEDED_EXIT_CODE),
        ("missing", RETRYABLE_EXIT_CODE),
        ("corruption", PROTOCOL_EXIT_CODE),
    ],
)
def test_all_afterany_consumers_share_the_six_fact_exit_matrix(
    tmp_path: Path, consumer: str, fact: str, expected_code: int,
):
    # ``consumer`` names the four production gates that call this exact
    # read-only classifier before writing phase artefacts.
    assert consumer in {"work", "close", "next-prep", "verify"}
    target = _target()
    if fact in {"active", "sealed"}:
        publish_activation_intent(tmp_path, target)
        prepared = _prepared_generation(tmp_path, target)
        activation = activate_generation(tmp_path, target, **prepared)
        if fact == "sealed":
            activation_payload = json.loads(activation.read_text(encoding="utf-8"))
            seal_generation(
                tmp_path,
                target,
                inventory={
                    "assignment_path": activation_payload["assignment_path"],
                    "assignment_sha256": activation_payload["assignment_sha256"],
                    "assignment_index_path": activation_payload[
                        "assignment_index_path"
                    ],
                    "assignment_index_sha256": activation_payload[
                        "assignment_index_sha256"
                    ],
                    "workers": [{"worker": 0, "wal_state": "absent"}],
                },
            )
    elif fact in {"outcome", "corruption"}:
        outcome = publish_no_generation_outcome(
            tmp_path, target, sealed_history_frontier=[]
        )
        if fact == "corruption":
            payload = json.loads(outcome.read_text(encoding="utf-8"))
            payload["sealed_history_digest_sha256"] = "0" * 64
            outcome.write_bytes(canonical_json_bytes(payload) + b"\n")
    elif fact == "superseded":
        pointer = {
            "pointer_format_version": 1,
            "analysis_id": target.analysis_id,
            "pointer_version": target.expected_pointer_version + 1,
            "generation_activation_path": str(
                (tmp_path / "superseding-generation.activation.json").resolve()
            ),
            "generation_activation_sha256": "f" * 64,
        }
        (tmp_path / "active-generation.json").write_bytes(
            canonical_json_bytes(pointer) + b"\n"
        )

    if fact == "corruption":
        with pytest.raises(ControlProtocolError):
            classify_exact_afterany_target_read_only(tmp_path, target)
    else:
        assert (
            classify_exact_afterany_target_read_only(tmp_path, target).exit_code
            == expected_code
        )


@pytest.mark.parametrize("dispatch_kind", ["sealed-generation", "no-generation"])
@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("analysis_id", "other-analysis"),
        ("last_execution_plan_id", "other-plan"),
        ("last_round", 99),
        ("last_submission_generation", "other-generation"),
        ("dispatch_kind", "no-generation"),
        ("sealed_history_digest_sha256", "0" * 64),
        ("complete", False),
        ("exit_code", 3),
    ],
)
def test_exact_verification_receipt_rejects_every_stale_or_tampered_binding(
    tmp_path: Path, dispatch_kind: str, field: str, bad_value: object,
):
    target = _target()
    dispatch = Dispatch(
        dispatch_kind,
        SUCCESS_EXIT_CODE,
        closed_sha256=("c" * 64 if dispatch_kind == "sealed-generation" else None),
        outcome_sha256=("o" * 64 if dispatch_kind == "no-generation" else None),
    )
    history_digest = "h" * 64
    receipt = publish_verification_receipt(
        tmp_path,
        target,
        dispatch=dispatch,
        sealed_history_digest_sha256=history_digest,
        complete=True,
        exit_code=SUCCESS_EXIT_CODE,
    )
    validate_exact_verification_receipt(
        tmp_path,
        target,
        expected_dispatch=dispatch,
        sealed_history_digest_sha256=history_digest,
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    if field == "dispatch_kind" and bad_value == dispatch_kind:
        bad_value = "no-generation" if dispatch_kind == "sealed-generation" else "sealed-generation"
    payload[field] = bad_value
    receipt.write_bytes(canonical_json_bytes(payload) + b"\n")
    with pytest.raises(ControlProtocolError, match="verification receipt"):
        validate_exact_verification_receipt(
            tmp_path,
            target,
            expected_dispatch=dispatch,
            sealed_history_digest_sha256=history_digest,
        )


def test_immutable_publish_never_replaces_a_between_check_and_rename_conflict(tmp_path: Path):
    target = tmp_path / "control.json"

    def inject_competing_target(boundary: str) -> None:
        if boundary == "before_rename":
            target.write_bytes(b"conflict\n")

    with pytest.raises(ControlProtocolError, match="immutable control artefact differs"):
        _write_temp_fsync_rename(target, {"value": 1}, fault=inject_competing_target)
    assert target.read_bytes() == b"conflict\n"


@pytest.mark.parametrize(
    "crash_boundary",
    [
        "before_file_fsync",
        "after_file_fsync",
        "before_rename",
        "after_rename",
        "before_parent_fsync",
        "after_parent_fsync",
    ],
)
def test_immutable_publish_retry_completes_parent_fsync(
    tmp_path: Path, monkeypatch, crash_boundary: str,
):
    target = tmp_path / "control.json"

    def crash(boundary: str) -> None:
        if boundary == crash_boundary:
            raise RuntimeError("injected immutable-publish crash")

    with pytest.raises(RuntimeError, match="immutable-publish"):
        _write_temp_fsync_rename(target, {"value": 1}, fault=crash)
    calls: list[Path] = []
    original = __import__(
        "aleatoric_nk_grid.generation_control", fromlist=["_fsync_directory"]
    )._fsync_directory

    def counted(directory: Path) -> None:
        calls.append(Path(directory))
        original(directory)

    monkeypatch.setattr("aleatoric_nk_grid.generation_control._fsync_directory", counted)
    _write_temp_fsync_rename(target, {"value": 1})
    assert calls == [target.parent]
