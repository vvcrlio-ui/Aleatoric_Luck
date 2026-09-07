"""Portable file/intent tests; these do not certify POSIX locking or Slurm."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from aleatoric_nk_grid.checkpoint_retention import (
    CheckpointArchiveError,
    archive_checkpoint_wals,
    archive_generation_paths,
    checkpoint_archive_path,
    reject_archived_checkpoints,
)
import aleatoric_nk_grid.checkpoint_retention as retention


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def completed_files(tmp_path):
    root = tmp_path / "history"
    root.mkdir()
    frontier = []
    wals = []
    for round_index in (1, 2):
        generation = root / "executions" / "plan" / f"round-{round_index}" / f"generation-g{round_index}"
        generation.mkdir(parents=True)
        (generation / "generation.lease").touch()
        activation = generation / "generation.activation.json"
        _write(activation, {"round": round_index})
        workers = []
        for worker in (0, 1):
            wal = generation / f"worker-{worker}.events.wal"
            wal.write_bytes(f"synthetic sealed WAL {round_index} {worker}".encode())
            wals.append(wal)
            workers.append({"worker": worker, "wal_state": "present", "path": str(wal.resolve()),
                            "size": wal.stat().st_size, "sha256": _hash(wal)})
        closed = generation / "generation.closed.json"
        _write(closed, {"generation_activation_sha256": _hash(activation), "inventory": {"workers": workers}})
        frontier.append({"closed_path": str(closed.resolve()), "closed_sha256": _hash(closed)})
    target = {"analysis_id": "analysis", "execution_plan_id": "plan", "round_index": 2, "submission_generation": "g2"}
    verification = root / "verifications" / "plan" / "round-2" / "g2.json"
    _write(verification, {"complete": True, "exit_code": 0, "analysis_id": "analysis",
                          "last_execution_plan_id": "plan", "last_round": 2, "last_submission_generation": "g2",
                          "sealed_history_digest_sha256": hashlib.sha256(json.dumps(frontier, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()})
    output = tmp_path / "final.csv"
    output.write_text("model,seed,draw,N,K,status,metric\nols,0,0,10,1,ok,1.5\n", encoding="utf-8")
    receipt = tmp_path / "final.csv.finalization.json"
    _write(receipt, {"status": "complete", "analysis_id": "analysis", "final_output": str(output.resolve()),
                     "final_rows": 1, "expected_model_keys": 1, "verification_receipt": str(verification.resolve())})
    return {"root": root, "target": target, "output": output, "receipt_path": receipt,
            "frozen_frontier": frontier}, wals


def test_archive_removes_exact_all_generation_wals_and_preserves_final_and_controls(completed_files):
    arguments, wals = completed_files
    root = arguments["root"]
    unrelated = root / "other-checkpoint.wal"
    unrelated.write_bytes(b"unrelated")
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file() and path not in wals}
    final_bytes = arguments["output"].read_bytes()
    result = archive_checkpoint_wals(**arguments)
    assert result["deleted_checkpoint_files"] == 4
    assert result["deleted_checkpoint_bytes"] == 4 * len(b"synthetic sealed WAL 1 0")
    assert all(not wal.exists() for wal in wals)
    assert arguments["output"].read_bytes() == final_bytes
    assert all(path.read_bytes() == content for path, content in before.items())
    assert json.loads(checkpoint_archive_path(root).read_text())["status"] == "complete"
    marker_bytes = checkpoint_archive_path(root).read_bytes()
    assert archive_checkpoint_wals(**arguments) == result
    assert checkpoint_archive_path(root).read_bytes() == marker_bytes
    assert len(archive_generation_paths(root)) == 2


@pytest.mark.parametrize("phase", ["after_archive_intent", "after_wal_unlink"])
def test_archive_interrupted_cleanup_is_explicit_and_resumable(completed_files, phase):
    arguments, wals = completed_files
    def crash(point):
        if point == phase:
            raise RuntimeError("injected archive crash")
    with pytest.raises(RuntimeError, match="injected"):
        archive_checkpoint_wals(**arguments, fault=crash)
    marker = json.loads(checkpoint_archive_path(arguments["root"]).read_text())
    assert marker["status"] == "cleanup_pending"
    assert sum(wal.exists() for wal in wals) == (4 if phase == "after_archive_intent" else 3)
    with pytest.raises(CheckpointArchiveError, match="terminally archived"):
        reject_archived_checkpoints(arguments["root"])
    retry = dict(arguments)
    retry.pop("frozen_frontier")
    result = archive_checkpoint_wals(**retry)
    assert result["deleted_checkpoint_files"] == 4
    assert all(not wal.exists() for wal in wals)


@pytest.mark.parametrize("failure", ["corrupt_wal", "missing_wal", "outside_path", "duplicate_worker", "bad_marker_hash", "bad_activation"])
def test_archive_rejects_bad_inventory_before_any_delete_or_intent(completed_files, failure):
    arguments, wals = completed_files
    closed = Path(arguments["frozen_frontier"][-1]["closed_path"])
    payload = json.loads(closed.read_text())
    if failure == "corrupt_wal":
        wals[-1].write_bytes(b"corrupt")
    elif failure == "missing_wal":
        wals[-1].unlink()
    elif failure == "outside_path":
        payload["inventory"]["workers"][-1]["path"] = str(arguments["output"])
    elif failure == "duplicate_worker":
        payload["inventory"]["workers"][-1]["worker"] = 0
    elif failure == "bad_activation":
        (closed.parent / "generation.activation.json").write_bytes(b"{}");
    if failure in {"outside_path", "duplicate_worker"}:
        _write(closed, payload)
        arguments["frozen_frontier"][-1]["closed_sha256"] = _hash(closed)
    elif failure == "bad_marker_hash":
        arguments["frozen_frontier"][-1]["closed_sha256"] = "0" * 64
    with pytest.raises(CheckpointArchiveError):
        archive_checkpoint_wals(**arguments)
    assert all(wal.exists() for wal in wals[:-1])
    assert not checkpoint_archive_path(arguments["root"]).exists()


@pytest.mark.parametrize("field,value", [("status", "failed"), ("final_rows", 0), ("analysis_id", "other")])
def test_archive_requires_complete_successful_matching_receipt(completed_files, field, value):
    arguments, wals = completed_files
    receipt = json.loads(arguments["receipt_path"].read_text())
    receipt[field] = value
    _write(arguments["receipt_path"], receipt)
    with pytest.raises(CheckpointArchiveError, match="fully verified"):
        archive_checkpoint_wals(**arguments)
    assert all(wal.exists() for wal in wals)
    assert not checkpoint_archive_path(arguments["root"]).exists()


@pytest.mark.parametrize("damaged", ["output", "receipt_path", "verification", "target", "closed", "wal", "extra_wal"])
def test_interrupted_archive_rejects_tampering_before_further_cleanup(completed_files, damaged):
    arguments, wals = completed_files
    def crash(point):
        if point == "after_archive_intent":
            raise RuntimeError("crash")
    with pytest.raises(RuntimeError):
        archive_checkpoint_wals(**arguments, fault=crash)
    if damaged == "target":
        arguments["target"] = {**arguments["target"], "submission_generation": "wrong"}
    elif damaged == "closed":
        Path(arguments["frozen_frontier"][0]["closed_path"]).write_bytes(b"{}");
    elif damaged == "verification":
        path = Path(json.loads(arguments["receipt_path"].read_text())["verification_receipt"])
        path.write_bytes(b"{}")
    elif damaged == "wal":
        wals[-1].write_bytes(b"changed")
    elif damaged == "extra_wal":
        (wals[-1].parent / "worker-999.events.wal").write_bytes(b"unexpected")
    else:
        arguments[damaged].write_bytes(b"changed")
    with pytest.raises(CheckpointArchiveError):
        archive_checkpoint_wals(**arguments)
    assert all(wal.exists() for wal in wals)


def test_archive_rejects_symlink_wal_without_touching_destination(completed_files, tmp_path):
    arguments, wals = completed_files
    outside = tmp_path / "outside"
    outside.write_bytes(wals[-1].read_bytes())
    original = wals[-1].read_bytes()
    wals[-1].unlink()
    try:
        wals[-1].symlink_to(outside)
    except OSError:
        wals[-1].write_bytes(original)
        pytest.skip("host does not permit symlink creation")
    with pytest.raises(CheckpointArchiveError, match="symlink"):
        archive_checkpoint_wals(**arguments)
    assert outside.read_bytes() == original
    assert all(wal.exists() for wal in wals)


def test_archive_requires_explicit_verified_frontier_and_no_marker_is_not_archived(tmp_path):
    reject_archived_checkpoints(tmp_path)
    with pytest.raises(CheckpointArchiveError, match="verified frozen frontier"):
        archive_checkpoint_wals(tmp_path, target={}, output=tmp_path / "out", receipt_path=tmp_path / "receipt")


def test_retry_finishes_intent_directory_sync_before_any_unlink(completed_files, monkeypatch):
    arguments, wals = completed_files
    root = arguments["root"].resolve()
    def failed_directory_sync(path):
        if path.resolve() == root:
            raise OSError("injected directory fsync failure")
    monkeypatch.setattr(retention, "_sync_directory", failed_directory_sync)
    # os.replace made intent visible, but its directory fsync failed. Both the
    # original call and its retry must stop before deleting a single checkpoint.
    for _ in range(2):
        with pytest.raises(OSError, match="fsync failure"):
            archive_checkpoint_wals(**arguments)
        assert all(wal.exists() for wal in wals)
        assert checkpoint_archive_path(root).exists()
    events = []
    monkeypatch.setattr(retention, "_sync_directory", lambda path: events.append(("sync", path.resolve())))
    original_unlink = Path.unlink
    def logged_unlink(path, *args, **kwargs):
        if path.name.endswith(".events.wal"):
            events.append(("unlink", path.resolve()))
        return original_unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", logged_unlink)
    archive_checkpoint_wals(**arguments)
    assert events[0] == ("sync", root)
    assert sum(kind == "unlink" for kind, _ in events) == 4
