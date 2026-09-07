"""Terminal archival of verified dynamic WALs, independent of POSIX locking.

The caller must hold the analysis schedule and affected generation leases.
The durable intent precedes every unlink.  Missing WALs are accepted only when
resuming that intent; this never relaxes the live generation WAL protocol.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import uuid
from pathlib import Path
from typing import Callable, Mapping, Sequence


ARCHIVE_FORMAT_VERSION = 1


class CheckpointArchiveError(ValueError):
    """A completed archive cannot be safely verified or cleaned up."""


def checkpoint_archive_path(root: Path) -> Path:
    return Path(root) / "checkpoint-archive.json"


def reject_archived_checkpoints(root: Path) -> None:
    marker = checkpoint_archive_path(root)
    if marker.exists() or marker.is_symlink():
        raise CheckpointArchiveError(
            "This run is terminally archived after successful finalization; "
            "checkpoint training/resume/verification is unavailable. The final CSV "
            "and audit records remain available. Repeat the exact finalize command "
            "to verify the CSV or finish interrupted checkpoint cleanup."
        )


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"


def _load(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise CheckpointArchiveError(f"cannot read checkpoint archive record: {path}") from exc
    if not isinstance(payload, dict):
        raise CheckpointArchiveError(f"checkpoint archive record must be an object: {path}")
    return payload


def _sync_directory(path: Path) -> None:
    # Windows cannot open directory fds this way. Portable tests exercise the
    # file/intent protocol only; production directory durability is POSIX.
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(_canonical(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_path(root: Path, path: Path, *, name: str) -> Path:
    root = Path(root).resolve()
    lexical = Path(os.path.abspath(path))
    resolved = lexical.resolve()
    if root not in resolved.parents or lexical != resolved:
        raise CheckpointArchiveError(f"{name} escapes archive root or uses a symlink: {path}")
    # resolve() is not sufficient for a final dangling symlink.
    if lexical.is_symlink():
        raise CheckpointArchiveError(f"{name} is a symlink: {path}")
    return resolved


def _binding(path: Path) -> dict[str, object]:
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise CheckpointArchiveError(f"archive input is not a regular file: {path}")
        return {"path": str(path.resolve()), "size": info.st_size, "sha256": _hash(path)}
    except OSError as exc:
        raise CheckpointArchiveError(f"cannot validate archive input: {path}") from exc


def _validate_binding(binding: Mapping[str, object], *, expected_path: Path | None = None,
                      missing_ok: bool = False) -> None:
    path = Path(str(binding.get("path", "")))
    if expected_path is not None and path != expected_path.resolve():
        raise CheckpointArchiveError("archive file path binding mismatch")
    if type(binding.get("size")) is not int or int(binding["size"]) < 0:
        raise CheckpointArchiveError("archive file size is invalid")
    if not isinstance(binding.get("sha256"), str) or re.fullmatch(r"[0-9a-f]{64}", str(binding["sha256"])) is None:
        raise CheckpointArchiveError("archive file hash is invalid")
    if missing_ok and not path.exists() and not path.is_symlink():
        return
    if _binding(path) != dict(binding):
        raise CheckpointArchiveError(f"archive input size/hash mismatch: {path}")


def _inventory(root: Path, frontier: Sequence[Mapping[str, object]], *,
               previous_wals: Sequence[Mapping[str, object]] | None = None
               ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Derive deletable names from sealed inventories, never a directory glob."""
    controls: list[dict[str, object]] = []
    wals: list[dict[str, object]] = []
    seen: set[Path] = set()
    previous = {str(item.get("path")): item for item in previous_wals or ()}
    for entry in frontier:
        marker = _safe_path(root, Path(str(entry.get("closed_path", ""))), name="closed marker")
        relative = marker.relative_to(Path(root).resolve())
        if (len(relative.parts) != 5 or relative.parts[0] != "executions"
                or not re.fullmatch(r"round-[1-9][0-9]*", relative.parts[2])
                or not relative.parts[3].startswith("generation-")
                or marker.name != "generation.closed.json" or marker in seen):
            raise CheckpointArchiveError("archive frontier contains a noncanonical or duplicate generation")
        seen.add(marker)
        binding = _binding(marker)
        if binding["sha256"] != entry.get("closed_sha256"):
            raise CheckpointArchiveError("archive closed marker checksum mismatch")
        controls.append(binding)
        closed = _load(marker)
        activation_path = _safe_path(root, marker.parent / "generation.activation.json", name="activation")
        activation = _binding(activation_path)
        if activation["sha256"] != closed.get("generation_activation_sha256"):
            raise CheckpointArchiveError("archive activation checksum mismatch")
        controls.append(activation)
        inventory = closed.get("inventory")
        workers = inventory.get("workers") if isinstance(inventory, Mapping) else None
        if not isinstance(workers, list):
            raise CheckpointArchiveError("archive sealed worker inventory is invalid")
        seen_workers: set[int] = set()
        expected_names: set[str] = set()
        for worker in workers:
            if not isinstance(worker, Mapping) or type(worker.get("worker")) is not int or int(worker["worker"]) < 0:
                raise CheckpointArchiveError("archive worker ID is invalid")
            number = int(worker["worker"])
            if number in seen_workers:
                raise CheckpointArchiveError("archive worker ID is duplicated")
            seen_workers.add(number)
            wal = _safe_path(root, marker.parent / f"worker-{number}.events.wal", name="WAL")
            expected_names.add(wal.name)
            state = worker.get("wal_state")
            if state == "present":
                if worker.get("path") != str(wal):
                    raise CheckpointArchiveError("archive WAL path differs from sealed inventory")
                item = {"path": str(wal), "size": worker.get("size"), "sha256": worker.get("sha256")}
            elif state == "absent":
                if worker.get("abandoned_uninitialized_wal") is True:
                    # A sealed abandoned inode has no committed content. Bind
                    # its bytes before deletion; it is still an exact owned WAL.
                    item = dict(previous[str(wal)]) if str(wal) in previous else _binding(wal)
                else:
                    if wal.exists() or wal.is_symlink():
                        raise CheckpointArchiveError("absent sealed WAL unexpectedly exists")
                    continue
            else:
                raise CheckpointArchiveError("archive WAL state is invalid")
            wals.append(item)
        # Enumeration is a rejection check only; it never supplies deletion
        # candidates. Unexpected files cannot be silently treated as archived.
        if any(path.name not in expected_names for path in marker.parent.glob("worker-*.events.wal")):
            raise CheckpointArchiveError("archive generation has inventory-external WAL")
    if previous_wals is not None and list(previous_wals) != wals:
        raise CheckpointArchiveError("archive WAL list differs from sealed inventory")
    return controls, wals


def archive_generation_paths(root: Path) -> tuple[Path, ...]:
    """Get confined generation lease directories before resuming an archive."""
    marker = _load(checkpoint_archive_path(root))
    frontier = marker.get("frozen_frontier")
    if not isinstance(frontier, list) or not all(isinstance(item, Mapping) for item in frontier):
        raise CheckpointArchiveError("archive frontier is invalid")
    return tuple(_safe_path(root, Path(str(item.get("closed_path", ""))), name="closed marker").parent
                 for item in frontier)


def archive_checkpoint_wals(
    root: Path, *, target: Mapping[str, object], output: Path, receipt_path: Path,
    frozen_frontier: Sequence[Mapping[str, object]] | None = None,
    fault: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Create/resume a terminal archive while caller holds exclusive leases."""
    root = Path(root).resolve()
    output = Path(output).resolve()
    receipt_path = Path(receipt_path).resolve()
    marker_path = checkpoint_archive_path(root)
    resuming = marker_path.exists() or marker_path.is_symlink()
    if marker_path.is_symlink():
        raise CheckpointArchiveError("checkpoint archive marker is a symlink")
    if resuming:
        archive = _load(marker_path)
        if (archive.get("archive_format_version") != ARCHIVE_FORMAT_VERSION
                or archive.get("status") not in {"cleanup_pending", "complete"}
                or archive.get("target") != dict(target)
                or archive.get("root") != str(root)):
            raise CheckpointArchiveError("checkpoint archive target/format mismatch")
        frontier = archive.get("frozen_frontier")
        previous_wals = archive.get("wals")
        if (not isinstance(frontier, list) or not all(isinstance(item, Mapping) for item in frontier)
                or not isinstance(previous_wals, list) or not all(isinstance(item, Mapping) for item in previous_wals)):
            raise CheckpointArchiveError("checkpoint archive inventory is invalid")
        if frozen_frontier is not None and list(frozen_frontier) != frontier:
            raise CheckpointArchiveError("checkpoint archive frontier mismatch")
        controls, wals = _inventory(root, frontier, previous_wals=previous_wals)
        if archive.get("controls") != controls:
            raise CheckpointArchiveError("checkpoint archive control records changed")
        for key, expected in (("final_output", output), ("finalization_receipt", receipt_path)):
            binding = archive.get(key)
            if not isinstance(binding, Mapping):
                raise CheckpointArchiveError("checkpoint archive file binding is missing")
            _validate_binding(binding, expected_path=expected)
        verification = archive.get("verification_receipt")
        if not isinstance(verification, Mapping):
            raise CheckpointArchiveError("checkpoint archive verification binding is missing")
        _safe_path(root, Path(str(verification.get("path", ""))), name="verification receipt")
        _validate_binding(verification)
    else:
        if frozen_frontier is None:
            raise CheckpointArchiveError("new archive requires the verified frozen frontier")
        frontier = list(frozen_frontier)
        controls, wals = _inventory(root, frontier)
        receipt = _load(receipt_path)
        verification_path = _safe_path(root, Path(str(receipt.get("verification_receipt", ""))), name="verification receipt")
        verification = _load(verification_path)
        history_digest = hashlib.sha256(_canonical(frontier).rstrip(b"\n")).hexdigest()
        if (receipt.get("status") != "complete" or receipt.get("analysis_id") != target.get("analysis_id")
                or receipt.get("final_output") != str(output)
                or type(receipt.get("final_rows")) is not int
                or receipt.get("final_rows") != receipt.get("expected_model_keys")
                or verification.get("complete") is not True or verification.get("exit_code") != 0
                or verification.get("analysis_id") != target.get("analysis_id")
                or verification.get("last_execution_plan_id") != target.get("execution_plan_id")
                or verification.get("last_round") != target.get("round_index")
                or verification.get("last_submission_generation") != target.get("submission_generation")
                or verification.get("sealed_history_digest_sha256") != history_digest):
            raise CheckpointArchiveError("only a fully verified successful finalization can be archived")
        archive = {
            "archive_format_version": ARCHIVE_FORMAT_VERSION, "status": "cleanup_pending",
            "root": str(root), "target": dict(target), "frozen_frontier": frontier,
            "controls": controls, "wals": wals, "final_output": _binding(output),
            "finalization_receipt": _binding(receipt_path),
            "verification_receipt": _binding(verification_path),
        }
    # Validate every surviving byte before touching any WAL. A damaged second
    # WAL must not cause the first good checkpoint to be deleted.
    for item in wals:
        _validate_binding(item, missing_ok=resuming)
    if not resuming:
        _write(marker_path, archive)
        if fault is not None:
            fault("after_archive_intent")
    else:
        # The prior process may have stopped after publishing intent but before
        # its directory fsync. Visibility alone cannot authorize deletion.
        _sync_directory(root)
    for item in wals:
        wal = Path(str(item["path"]))
        if wal.exists():
            # Recheck immediately before unlink as well as the full preflight.
            _safe_path(root, wal, name="WAL")
            _validate_binding(item)
            wal.unlink()
            _sync_directory(wal.parent)
            if fault is not None:
                fault("after_wal_unlink")
    # Also finish an unlink's durability boundary when its name was already
    # missing at retry entry (crash between unlink and directory fsync).
    for directory in {Path(str(item["path"])).parent for item in wals}:
        _sync_directory(directory)
    if archive["status"] != "complete":
        archive["status"] = "complete"
        _write(marker_path, archive)
    result = _load(receipt_path)
    result["checkpoint_archive"] = str(marker_path)
    result["checkpoint_retention"] = "delete"
    result["deleted_checkpoint_files"] = len(wals)
    result["deleted_checkpoint_bytes"] = sum(int(item["size"]) for item in wals)
    return result
