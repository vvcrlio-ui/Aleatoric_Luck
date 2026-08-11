"""Fail-closed activation, sealing, and exact-target dispatch control plane.

There is intentionally no "latest generation" API in this module.  Every
consumer receives an execution plan, round, generation, prep job and token,
then derives one deterministic activation/outcome path.  The active pointer
is the sole mutable commit point and its compare-and-swap runs under one
stable analysis schedule lease.
"""

from __future__ import annotations

import ctypes
import errno
import sys
import fcntl
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Mapping, Sequence

from .execution_contract import canonical_json_bytes, sha256_bytes


SUCCESS_EXIT_CODE = 0
INCOMPLETE_EXIT_CODE = 3
PROTOCOL_EXIT_CODE = 6
RETRYABLE_EXIT_CODE = 7
SUPERSEDED_EXIT_CODE = 8

POINTER_FORMAT_VERSION = 1
INTENT_FORMAT_VERSION = 1
PREPARED_FORMAT_VERSION = 1
ACTIVATION_FORMAT_VERSION = 1
CLOSED_FORMAT_VERSION = 1
OUTCOME_FORMAT_VERSION = 1
VERIFICATION_FORMAT_VERSION = 2


class ControlProtocolError(ValueError):
    exit_code = PROTOCOL_EXIT_CODE


class ControlBusyError(RuntimeError):
    exit_code = RETRYABLE_EXIT_CODE


class ControlSupersededError(RuntimeError):
    exit_code = SUPERSEDED_EXIT_CODE


@dataclass(frozen=True)
class ActivationTarget:
    analysis_id: str
    execution_plan_id: str
    execution_contract_sha256: str
    round_index: int
    submission_generation: str
    expected_previous_generation: str | None
    expected_pointer_version: int
    prep_job_id: str
    prep_token: str
    expected_previous_execution_plan_id: str | None = None
    expected_previous_round_index: int | None = None

    def validate(self) -> None:
        if not self.analysis_id or not self.execution_plan_id or not self.execution_contract_sha256:
            raise ControlProtocolError("activation target requires analysis/execution identities")
        if self.round_index < 1 or self.expected_pointer_version < 0:
            raise ControlProtocolError("activation target round/pointer version is invalid")
        if not self.submission_generation or not self.prep_job_id or not self.prep_token:
            raise ControlProtocolError("activation target requires generation and prep identity")
        if self.expected_previous_generation is None:
            if self.expected_previous_execution_plan_id is not None or self.expected_previous_round_index is not None:
                raise ControlProtocolError("initial target may not name a predecessor scope")
        elif (self.expected_previous_execution_plan_id is None) != (self.expected_previous_round_index is None):
            raise ControlProtocolError("predecessor plan and round must be frozen together")
        elif self.expected_previous_round_index is not None and self.expected_previous_round_index < 1:
            raise ControlProtocolError("predecessor round is invalid")


@dataclass(frozen=True)
class Dispatch:
    kind: str
    exit_code: int
    activation_path: Path | None = None
    activation_sha256: str | None = None
    closed_path: Path | None = None
    closed_sha256: str | None = None
    outcome_path: Path | None = None
    outcome_sha256: str | None = None


@dataclass(frozen=True)
class SealedGenerationValidation:
    """One immutable sealed-generation read, reusable by a phase consumer."""

    closed_path: Path
    closed_payload: Mapping[str, object]
    closed_sha256: str
    wal_scans: Mapping[int, object]


class GenerationValidationCache:
    """Phase-local cache: each present WAL is opened and scanned once."""

    def __init__(self) -> None:
        self._sealed: dict[tuple[Path, ActivationTarget], SealedGenerationValidation] = {}
        self._activation: dict[tuple[Path, ActivationTarget], tuple[Path, dict[str, object], str]] = {}

    def get_activation(self, root: Path, target: ActivationTarget) -> tuple[Path, dict[str, object], str] | None:
        return self._activation.get((Path(root).resolve(), target))

    def put_activation(self, root: Path, target: ActivationTarget, value: tuple[Path, dict[str, object], str]) -> None:
        self._activation[(Path(root).resolve(), target)] = value

    def get(self, root: Path, target: ActivationTarget) -> SealedGenerationValidation | None:
        return self._sealed.get((Path(root).resolve(), target))

    def put(self, root: Path, target: ActivationTarget, value: SealedGenerationValidation) -> None:
        self._sealed[(Path(root).resolve(), target)] = value


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ControlProtocolError(f"cannot read {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ControlProtocolError(f"{label} must be a JSON object: {path}")
    return payload


def _load_canonical_json(path: Path, *, label: str) -> dict[str, object]:
    """Load one committed control artefact and reject non-canonical bytes.

    A valid JSON object with changed whitespace is still a changed committed
    artefact.  Keeping this check at the control-plane boundary prevents a
    later reader from treating a payload rewrite as harmless formatting.
    """

    payload = _load_json(path, label=label)
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise ControlProtocolError(f"cannot read {label}: {path}") from exc
    if raw != canonical_json_bytes(payload) + b"\n":
        raise ControlProtocolError(f"{label} is not canonical immutable JSON")
    return payload


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(Path(directory), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, target: Path) -> None:
    """Atomically publish a filename only if it does not yet exist.

    Plain POSIX ``rename`` replaces an existing target, so it cannot publish
    immutable control records.  Linux has ``renameat2(RENAME_NOREPLACE)`` and
    macOS has ``renamex_np(RENAME_EXCL)``; unsupported filesystems fail closed
    instead of silently falling back to link/unlink or replace semantics.
    """

    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    target_bytes = os.fsencode(target)
    if sys.platform == "darwin":
        operation = getattr(libc, "renamex_np", None)
        if operation is None:
            raise ControlProtocolError("atomic no-replace rename is unavailable")
        result = operation(source_bytes, target_bytes, 0x00000004)  # RENAME_EXCL
    elif sys.platform.startswith("linux"):
        operation = getattr(libc, "renameat2", None)
        if operation is None:
            raise ControlProtocolError("atomic no-replace rename is unavailable")
        result = operation(-100, source_bytes, -100, target_bytes, 1)  # AT_FDCWD, RENAME_NOREPLACE
    else:
        raise ControlProtocolError("atomic no-replace rename is unavailable")
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise FileExistsError(error, os.strerror(error), target)
    raise ControlProtocolError(f"atomic no-replace rename failed for {target}: {os.strerror(error)}")


def _write_all(descriptor: int, payload: bytes) -> None:
    """Write a control record fully before the associated fsync boundary."""

    view = memoryview(payload)
    while view:
        try:
            written = os.write(descriptor, view)
        except InterruptedError:
            continue
        if written is None or written <= 0:
            raise ControlProtocolError("short write while publishing control artefact")
        view = view[written:]


def _write_temp_fsync_rename(
    path: Path,
    payload: Mapping[str, object],
    *,
    fault: Callable[[str], None] | None = None,
) -> str:
    """Publish one immutable canonical JSON record, preserving crash evidence."""

    target = Path(path)
    data = canonical_json_bytes(dict(payload)) + b"\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing = target.read_bytes()
        if existing != data:
            raise ControlProtocolError(f"immutable control artefact differs: {target}")
        # The prior attempt may have crashed after rename but before its
        # parent-directory fsync.  Visibility alone is not durability;
        # idempotent success must complete the missing commit boundary.
        _fsync_directory(target.parent)
        return sha256_bytes(existing)
    temporary = target.parent / f".{target.name}.tmp.{uuid.uuid4().hex}"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    try:
        _write_all(descriptor, data)
        if fault is not None:
            fault("before_file_fsync")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if fault is not None:
        fault("after_file_fsync")
        fault("before_rename")
    try:
        _rename_noreplace(temporary, target)
    except FileExistsError:
        temporary.unlink(missing_ok=True)
        existing = target.read_bytes()
        if existing != data:
            raise ControlProtocolError(f"immutable control artefact differs: {target}")
        _fsync_directory(target.parent)
        return sha256_bytes(existing)
    if fault is not None:
        fault("after_rename")
        fault("before_parent_fsync")
    _fsync_directory(target.parent)
    if fault is not None:
        fault("after_parent_fsync")
    return sha256_bytes(data)


@contextmanager
def _lease(path: Path, *, exclusive: bool, create: bool = True) -> Iterator[int]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | (os.O_CREAT if create else 0)
    try:
        descriptor = os.open(target, flags, 0o640)
    except OSError as exc:
        raise ControlBusyError(f"cannot open lease {target}: {exc}") from exc
    try:
        try:
            fcntl.flock(descriptor, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise ControlBusyError(f"lease is busy: {target}") from exc
            raise
        yield descriptor
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def analysis_schedule_lease(root: Path) -> Path:
    return Path(root) / ".analysis-schedule.lease"


def ensure_analysis_schedule_lease(root: Path) -> Path:
    """Create the analysis schedule lease during plan publication.

    Preparation and recovery must lock an existing identity-scoped inode.  In
    particular, a failed next-round predecessor check must not create a new
    control artefact merely to report code 7.
    """

    target_root = Path(root)
    target_root.mkdir(parents=True, exist_ok=True)
    target = analysis_schedule_lease(target_root)
    try:
        descriptor = os.open(target, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o640)
    except FileExistsError:
        if not target.is_file():
            raise ControlProtocolError("analysis schedule lease is not a regular file")
        return target
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(target_root)
    return target


@contextmanager
def schedule_transaction(root: Path) -> Iterator[None]:
    """Hold the one analysis schedule lease for a full prep transaction.

    The transaction begins before the sealed-history/todo read and ends only
    after its no-generation outcome or pointer CAS is durable.  Public helper
    functions accept ``lease_held=True`` solely for this context; callers must
    not compose a series of short independent leases.
    """

    with _lease(analysis_schedule_lease(Path(root)), exclusive=True, create=False):
        yield


def _round_dir(root: Path, target: ActivationTarget) -> Path:
    return Path(root) / "executions" / target.execution_plan_id / f"round-{target.round_index}"


def generation_dir(root: Path, target: ActivationTarget) -> Path:
    return _round_dir(root, target) / f"generation-{target.submission_generation}"


def outcome_path(root: Path, target: ActivationTarget) -> Path:
    return _round_dir(root, target) / "prep-outcomes" / f"{target.submission_generation}.json"


def intent_path(root: Path, target: ActivationTarget) -> Path:
    return Path(root) / "activation-intents" / f"{target.submission_generation}.json"


def activation_path(root: Path, target: ActivationTarget) -> Path:
    return generation_dir(root, target) / "generation.activation.json"


def closed_path(root: Path, target: ActivationTarget) -> Path:
    return generation_dir(root, target) / "generation.closed.json"


def pointer_path(root: Path) -> Path:
    return Path(root) / "active-generation.json"


def _pointer(root: Path) -> tuple[dict[str, object] | None, bytes | None]:
    target = pointer_path(root)
    if not target.exists():
        return None, None
    raw = target.read_bytes()
    payload = _load_json(target, label="active generation pointer")
    if payload.get("pointer_format_version") != POINTER_FORMAT_VERSION:
        raise ControlProtocolError("unsupported active generation pointer format")
    try:
        if not isinstance(payload["analysis_id"], str) or not isinstance(payload["pointer_version"], int):
            raise ValueError
        if not isinstance(payload["generation_activation_path"], str) or not isinstance(payload["generation_activation_sha256"], str):
            raise ValueError
    except (KeyError, ValueError):
        raise ControlProtocolError("active generation pointer is structurally invalid") from None
    if raw != canonical_json_bytes(payload) + b"\n":
        raise ControlProtocolError("active generation pointer is not canonical")
    return payload, raw


def _target_matches_payload(payload: Mapping[str, object], target: ActivationTarget) -> bool:
    return (
        payload.get("analysis_id") == target.analysis_id
        and payload.get("execution_plan_id") == target.execution_plan_id
        and payload.get("execution_contract_sha256") == target.execution_contract_sha256
        and payload.get("round") == target.round_index
        and payload.get("submission_generation") == target.submission_generation
        and payload.get("prep_job_id") == target.prep_job_id
        and payload.get("prep_token") == target.prep_token
    )


def _validate_outcome(root: Path, target: ActivationTarget) -> tuple[Path, dict[str, object], str] | None:
    path = outcome_path(root, target)
    if not path.exists():
        return None
    payload = _load_canonical_json(path, label="no-generation prep outcome")
    if payload.get("outcome_format_version") != OUTCOME_FORMAT_VERSION or payload.get("outcome") != "no-generation":
        raise ControlProtocolError("invalid no-generation prep outcome")
    if not _target_matches_payload(payload, target) or payload.get("todo_count") != 0 or payload.get("no_generation") is not True:
        raise ControlProtocolError("no-generation outcome identity or count mismatch")
    expected = {
        "prospective_previous_generation_or_null": target.expected_previous_generation,
        "prospective_previous_execution_plan_id_or_null": target.expected_previous_execution_plan_id,
        "prospective_previous_round_or_null": target.expected_previous_round_index,
        "prospective_pointer_version": target.expected_pointer_version,
    }
    if any(payload.get(name) != value for name, value in expected.items()):
        raise ControlProtocolError("no-generation outcome prospective predecessor mismatch")
    if not isinstance(payload.get("observed_pointer_version"), int) or int(payload["observed_pointer_version"]) < 0:
        raise ControlProtocolError("no-generation outcome observed pointer is invalid")
    observed_pointer = payload.get("observed_pointer_sha256_or_null")
    if observed_pointer is not None and (not isinstance(observed_pointer, str) or len(observed_pointer) != 64):
        raise ControlProtocolError("no-generation outcome observed pointer digest is invalid")
    frontier = payload.get("sealed_history_frontier")
    if not isinstance(frontier, list):
        raise ControlProtocolError("no-generation outcome lacks sealed-history frontier")
    frozen = _validate_frozen_frontier(root, [item for item in frontier if isinstance(item, Mapping)])
    if len(frozen) != len(frontier):
        raise ControlProtocolError("no-generation outcome sealed-history frontier is malformed")
    if payload.get("sealed_history_generation_count") != len(frozen):
        raise ControlProtocolError("no-generation outcome sealed-history count mismatch")
    if sha256_bytes(canonical_json_bytes(list(frozen))) != payload.get("sealed_history_digest_sha256"):
        raise ControlProtocolError("no-generation outcome sealed-history digest mismatch")
    predecessor_path = payload.get("observed_previous_closed_path_or_null")
    predecessor_sha = payload.get("observed_previous_closed_sha256_or_null")
    if (predecessor_path is None) != (predecessor_sha is None):
        raise ControlProtocolError("no-generation outcome predecessor reference is incomplete")
    if predecessor_path is None:
        if frozen:
            raise ControlProtocolError("no-generation outcome has history without predecessor")
    elif not frozen or frozen[-1] != {"closed_path": str(Path(str(predecessor_path)).resolve()), "closed_sha256": predecessor_sha}:
        raise ControlProtocolError("no-generation outcome predecessor does not match frozen history")
    return path, payload, _sha(path)


def _outcome_conflicts_with_generation(root: Path, target: ActivationTarget) -> bool:
    """A todo=0 receipt is mutually exclusive with every activation artefact."""

    directory = generation_dir(root, target)
    staging = directory.parent / f".{directory.name}.staging"
    return intent_path(root, target).exists() or directory.exists() or staging.exists()


def _validate_activation(
    root: Path, target: ActivationTarget, *, cache: GenerationValidationCache | None = None,
) -> tuple[Path, dict[str, object], str] | None:
    if cache is not None:
        cached = cache.get_activation(root, target)
        if cached is not None:
            return cached
    path = activation_path(root, target)
    if not path.exists():
        return None
    payload = _load_canonical_json(path, label="generation activation")
    if payload.get("activation_format_version") != ACTIVATION_FORMAT_VERSION:
        raise ControlProtocolError("invalid generation activation format")
    if not _target_matches_payload(payload, target):
        raise ControlProtocolError("generation activation identity mismatch")
    required = (
        "intent_sha256", "prepared_sha256", "assignment_sha256", "assignment_index_sha256",
        "expected_pointer_version", "observed_pointer_sha256_or_null",
        "observed_previous_closed_sha256_or_null",
    )
    if any(name not in payload for name in required):
        raise ControlProtocolError("generation activation lacks frozen fields")
    generation = generation_dir(root, target).resolve()
    if path.parent.resolve() != generation:
        raise ControlProtocolError("generation activation path is not canonical")
    intent_file = intent_path(root, target)
    intent = _load_canonical_json(intent_file, label="activation intent")
    if intent.get("intent_format_version") != INTENT_FORMAT_VERSION or not _target_matches_payload(intent, target):
        raise ControlProtocolError("activation intent identity mismatch")
    if payload.get("intent_sha256") != _sha(intent_file):
        raise ControlProtocolError("generation activation intent checksum mismatch")
    for name in (
        "expected_previous_generation_or_null", "expected_previous_execution_plan_id_or_null",
        "expected_previous_round_or_null", "expected_pointer_version",
        "observed_previous_closed_sha256_or_null", "observed_previous_closed_path_or_null",
        "observed_pointer_sha256_or_null",
    ):
        if payload.get(name) != intent.get(name):
            raise ControlProtocolError(f"generation activation differs from frozen intent field {name}")
    prepared_file = generation / "generation.prepared.json"
    prepared = _load_canonical_json(prepared_file, label="generation prepared record")
    prepared_identity = {
        "analysis_id": target.analysis_id,
        "execution_plan_id": target.execution_plan_id,
        "execution_contract_sha256": target.execution_contract_sha256,
        "round": target.round_index,
        "submission_generation": target.submission_generation,
    }
    if prepared.get("prepared_format_version") != PREPARED_FORMAT_VERSION or any(
        prepared.get(key) != value for key, value in prepared_identity.items()
    ):
        raise ControlProtocolError("generation prepared record identity mismatch")
    if payload.get("prepared_sha256") != _sha(prepared_file):
        raise ControlProtocolError("generation activation prepared checksum mismatch")
    if prepared.get("intent_sha256") != payload.get("intent_sha256"):
        raise ControlProtocolError("generation prepared record intent checksum mismatch")
    artefacts = (
        ("assignment", "assignment.parquet"),
        ("assignment_index", "assignment.index.json"),
        ("ready", "assignment.ready.json"),
        ("prep", "prep.json"),
    )
    for stem, filename in artefacts:
        actual = generation / filename
        declared_path = payload.get(f"{stem}_path") if stem != "assignment_index" else payload.get("assignment_index_path")
        declared_sha = payload.get(f"{stem}_sha256") if stem != "assignment_index" else payload.get("assignment_index_sha256")
        prepared_path = prepared.get(f"{stem}_path") if stem != "assignment_index" else prepared.get("assignment_index_path")
        prepared_sha = prepared.get(f"{stem}_sha256") if stem != "assignment_index" else prepared.get("assignment_index_sha256")
        if stem == "prep":
            # The activation deliberately carries only prepared's binding for
            # prep.json; the prepared record is the authoritative checksum.
            declared_path = prepared_path
            declared_sha = prepared_sha
        if declared_path != str(actual) or prepared_path != str(actual):
            raise ControlProtocolError(f"generation {stem} path is not canonical")
        if not actual.is_file():
            raise ControlProtocolError(f"generation {stem} file is missing")
        # A sealed consumer verifies every frozen artefact exactly once per
        # generation.  Workers validate their own row group separately, but
        # that does not relax the sealed control-plane integrity boundary.
        actual_sha = _sha(actual)
        if not isinstance(declared_sha, str) or declared_sha != prepared_sha or actual_sha != declared_sha:
            raise ControlProtocolError(f"generation {stem} checksum mismatch")
    if payload.get("generation_lease_path") != str((generation / "generation.lease")):
        raise ControlProtocolError("generation activation lease path is not canonical")
    if not (generation / "generation.lease").is_file():
        raise ControlProtocolError("generation activation lease inode is missing")
    result = (path, payload, _sha(path))
    if cache is not None:
        cache.put_activation(root, target, result)
    return result


def _validate_closed(
    root: Path, target: ActivationTarget, activation_sha256: str,
    *, cache: GenerationValidationCache | None = None,
) -> tuple[Path, dict[str, object], str] | None:
    if cache is not None:
        cached = cache.get(root, target)
        if cached is not None:
            return cached.closed_path, dict(cached.closed_payload), cached.closed_sha256
    path = closed_path(root, target)
    if not path.exists():
        return None
    payload = _load_canonical_json(path, label="generation closed marker")
    if payload.get("closed_format_version") != CLOSED_FORMAT_VERSION or not _target_matches_payload(payload, target):
        raise ControlProtocolError("generation closed marker identity mismatch")
    if payload.get("generation_activation_sha256") != activation_sha256:
        raise ControlProtocolError("generation closed marker activation mismatch")
    inventory = payload.get("inventory")
    if not isinstance(inventory, Mapping):
        raise ControlProtocolError("generation closed marker lacks immutable inventory")
    generation = generation_dir(root, target).resolve()
    activation = _load_canonical_json(activation_path(root, target), label="generation activation")
    for key in ("assignment_path", "assignment_sha256", "assignment_index_path", "assignment_index_sha256"):
        if inventory.get(key) != activation.get(key):
            raise ControlProtocolError(f"closed inventory {key} differs from activation")
    workers = inventory.get("workers")
    if not isinstance(workers, list) or inventory.get("workers") is None:
        raise ControlProtocolError("closed inventory workers are invalid")
    if inventory.get("workers") is not workers:  # keep pyright/mypy from narrowing Mapping oddly
        raise ControlProtocolError("closed inventory workers are invalid")
    expected_count = activation.get("worker_count")
    if not isinstance(expected_count, int) or expected_count < 1 or len(workers) != expected_count:
        raise ControlProtocolError("closed inventory worker count mismatch")
    if {item.get("worker") for item in workers if isinstance(item, Mapping)} != set(range(expected_count)):
        raise ControlProtocolError("closed inventory worker IDs mismatch")
    index_path = generation / "assignment.index.json"
    index_payload = _load_canonical_json(index_path, label="assignment index")
    row_groups = index_payload.get("row_groups")
    if not isinstance(row_groups, list):
        raise ControlProtocolError("assignment index lacks row groups")
    group_by_worker = {
        item.get("worker"): item for item in row_groups
        if isinstance(item, Mapping) and isinstance(item.get("worker"), int)
    }
    if set(group_by_worker) != set(range(expected_count)):
        raise ControlProtocolError("assignment index worker groups mismatch")
    from .worker_event_wal import WAL_FORMAT, WALBusyError, WALProtocolError, WorkerEventLog
    expected_wals = {generation / f"worker-{worker}.events.wal" for worker in range(expected_count)}
    actual_wals = set(generation.glob("worker-*.events.wal"))
    if actual_wals != expected_wals.intersection(actual_wals):
        raise ControlProtocolError("sealed generation has inventory-external WAL")
    scans: dict[int, object] = {}
    for item in workers:
        if not isinstance(item, Mapping):
            raise ControlProtocolError("closed inventory worker entry is invalid")
        worker = int(item["worker"])
        wal = generation / f"worker-{worker}.events.wal"
        state = item.get("wal_state")
        group = group_by_worker[worker]
        expected_identity = {
            "wal_format": WAL_FORMAT,
            "analysis_id": target.analysis_id,
            "execution_plan_id": target.execution_plan_id,
            "execution_contract_sha256": target.execution_contract_sha256,
            "round": target.round_index,
            "submission_generation": target.submission_generation,
            "worker": worker,
            "workers": expected_count,
            "assignment_path": activation["assignment_path"],
            "assignment_sha256": activation["assignment_sha256"],
            "assignment_index_path": activation["assignment_index_path"],
            "assignment_index_sha256": activation["assignment_index_sha256"],
            "assignment_row_group": worker,
            "assignment_row_count": int(group["row_count"]),
            "assignment_row_group_digest": group["canonical_task_rows_sha256"],
        }
        if state == "absent":
            if wal.exists():
                try:
                    uninitialized = WorkerEventLog.open_shared(wal, expected_identity=expected_identity)
                except WALBusyError as exc:
                    raise ControlBusyError("abandoned WAL shared lease is busy") from exc
                except WALProtocolError as exc:
                    raise ControlProtocolError("abandoned WAL is corrupt") from exc
                if (
                    uninitialized.identity is not None or uninitialized.records
                    or dict(item) != {"worker": worker, "wal_state": "absent", "abandoned_uninitialized_wal": True}
                ):
                    raise ControlProtocolError("absent WAL inventory entry conflicts with filesystem")
            elif set(item) - {"worker", "wal_state"}:
                raise ControlProtocolError("absent WAL inventory entry conflicts with filesystem")
            continue
        if state != "present" or not wal.is_file():
            raise ControlProtocolError("present WAL inventory entry is invalid")
        try:
            scan = WorkerEventLog.open_shared(wal, expected_identity=expected_identity)
        except WALBusyError as exc:
            raise ControlBusyError("sealed WAL shared lease is busy") from exc
        except WALProtocolError as exc:
            raise ControlProtocolError("sealed WAL is corrupt") from exc
        if scan.identity is None or dict(scan.identity) != expected_identity:
            raise ControlProtocolError("sealed WAL identity mismatch")
        expected = {
            "worker": worker, "wal_state": "present", "path": str(wal),
            "size": scan.file_size, "sha256": scan.file_sha256,
            "last_committed_offset": scan.committed_offset,
            "last_commit_trailer_digest": scan.trailer_digest,
            "uncommitted_tail": scan.has_uncommitted_tail,
        }
        if dict(item) != expected:
            raise ControlProtocolError("sealed WAL inventory differs from committed file")
        scans[worker] = scan
    digest = _sha(path)
    if cache is not None:
        cache.put(root, target, SealedGenerationValidation(path, dict(payload), digest, scans))
    return path, payload, digest


def classify_exact_afterany_target_read_only(
    root: Path, target: ActivationTarget, *, cache: GenerationValidationCache | None = None,
) -> Dispatch:
    """Map durable facts to the shared 0/6/7/8 afterany protocol.

    This function performs no repair, locking, WAL creation, or control-plane
    write, making it safe as the first operation in work/close/verify jobs.
    """

    target.validate()
    root = Path(root)
    outcome = _validate_outcome(root, target)
    activation = _validate_activation(root, target, cache=cache)
    if outcome is not None and _outcome_conflicts_with_generation(root, target):
        return Dispatch("protocol", PROTOCOL_EXIT_CODE)
    pointer, raw_pointer = _pointer(root)
    if outcome is not None:
        path, _, digest = outcome
        return Dispatch("no-generation", SUCCESS_EXIT_CODE, outcome_path=path, outcome_sha256=digest)
    if activation is not None:
        current_path, current, digest = activation
        closed = _validate_closed(root, target, digest, cache=cache)
        if closed is not None:
            closed_file, _, closed_digest = closed
            return Dispatch(
                "sealed-generation", SUCCESS_EXIT_CODE,
                activation_path=current_path, activation_sha256=digest,
                closed_path=closed_file, closed_sha256=closed_digest,
            )
        if pointer is not None:
            if pointer.get("analysis_id") != target.analysis_id:
                return Dispatch("protocol", PROTOCOL_EXIT_CODE)
            if (
                pointer.get("generation_activation_path") == str(current_path.resolve())
                and pointer.get("generation_activation_sha256") == digest
            ):
                if int(pointer["pointer_version"]) != target.expected_pointer_version + 1:
                    return Dispatch("protocol", PROTOCOL_EXIT_CODE)
                return Dispatch("active-generation", SUCCESS_EXIT_CODE, activation_path=current_path, activation_sha256=digest)
            if int(pointer["pointer_version"]) > target.expected_pointer_version:
                return Dispatch("protocol", PROTOCOL_EXIT_CODE)
        return Dispatch("recovery-required", RETRYABLE_EXIT_CODE)
    if pointer is not None:
        if pointer.get("analysis_id") != target.analysis_id:
            return Dispatch("protocol", PROTOCOL_EXIT_CODE)
        if int(pointer["pointer_version"]) > target.expected_pointer_version:
            return Dispatch("superseded", SUPERSEDED_EXIT_CODE)
    if intent_path(root, target).exists() or any((generation_dir(root, target).parent).glob(f".generation-{target.submission_generation}.staging")):
        return Dispatch("recovery-required", RETRYABLE_EXIT_CODE)
    return Dispatch("recovery-required", RETRYABLE_EXIT_CODE)


def _validate_frozen_frontier(root: Path, frontier: Sequence[Mapping[str, object]]) -> tuple[dict[str, object], ...]:
    """Verify an immutable oldest-to-newest closed-marker frontier."""

    root = Path(root).resolve()
    normalized: list[dict[str, object]] = []
    seen: set[Path] = set()
    for item in frontier:
        try:
            marker = Path(str(item["closed_path"])).resolve()
            digest = str(item["closed_sha256"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ControlProtocolError("sealed-history frontier entry is invalid") from exc
        if marker in seen or root not in marker.parents or marker.name != "generation.closed.json":
            raise ControlProtocolError("sealed-history frontier path is invalid")
        seen.add(marker)
        closed = _load_canonical_json(marker, label="frozen generation closed marker")
        if _sha(marker) != digest or closed.get("closed_format_version") != CLOSED_FORMAT_VERSION:
            raise ControlProtocolError("sealed-history frontier checksum mismatch")
        normalized.append({"closed_path": str(marker), "closed_sha256": digest})
    return tuple(normalized)


def _history_from_closed_marker(root: Path, marker: Path, digest: str) -> tuple[dict[str, object], ...]:
    """Follow activation-frozen predecessor links, never directory recency."""

    root = Path(root).resolve()
    chain: list[dict[str, object]] = []
    seen: set[Path] = set()
    current = Path(marker).resolve()
    current_digest = str(digest)
    while True:
        if current in seen or root not in current.parents or current.name != "generation.closed.json":
            raise ControlProtocolError("frozen sealed-history predecessor chain is invalid")
        seen.add(current)
        closed = _load_canonical_json(current, label="frozen generation closed marker")
        if _sha(current) != current_digest or closed.get("closed_format_version") != CLOSED_FORMAT_VERSION:
            raise ControlProtocolError("frozen sealed-history closed checksum mismatch")
        activation_file = current.parent / "generation.activation.json"
        activation = _load_canonical_json(activation_file, label="frozen generation activation")
        if closed.get("generation_activation_sha256") != _sha(activation_file):
            raise ControlProtocolError("frozen sealed-history activation checksum mismatch")
        chain.append({"closed_path": str(current), "closed_sha256": current_digest})
        predecessor_path = activation.get("observed_previous_closed_path_or_null")
        predecessor_digest = activation.get("observed_previous_closed_sha256_or_null")
        if predecessor_path is None and predecessor_digest is None:
            break
        if not isinstance(predecessor_path, str) or not isinstance(predecessor_digest, str):
            raise ControlProtocolError("frozen sealed-history predecessor reference is invalid")
        current = Path(predecessor_path).resolve()
        current_digest = predecessor_digest
    chain.reverse()
    return tuple(chain)


def frozen_sealed_history(
    root: Path, target: ActivationTarget, dispatch: Dispatch,
) -> tuple[dict[str, object], ...]:
    """Return the only sealed history a verify/finalizer may consume.

    Sealed targets walk the predecessor links frozen in activation intent.
    Exact no-generation targets use their durable outcome frontier.  Neither
    path enumerates a mutable ``executions`` directory.
    """

    if dispatch.kind == "sealed-generation":
        if dispatch.closed_path is None or dispatch.closed_sha256 is None:
            raise ControlProtocolError("sealed dispatch lacks closed-marker binding")
        return _history_from_closed_marker(root, dispatch.closed_path, dispatch.closed_sha256)
    if dispatch.kind == "no-generation":
        outcome = _validate_outcome(root, target)
        if outcome is None:
            raise ControlProtocolError("no-generation dispatch lacks exact outcome")
        _, payload, _ = outcome
        raw = payload.get("sealed_history_frontier")
        if not isinstance(raw, list):
            raise ControlProtocolError("no-generation outcome lacks sealed-history frontier")
        frontier = _validate_frozen_frontier(root, [item for item in raw if isinstance(item, Mapping)])
        if len(frontier) != len(raw):
            raise ControlProtocolError("no-generation outcome has malformed sealed-history frontier")
        if sha256_bytes(canonical_json_bytes(list(frontier))) != payload.get("sealed_history_digest_sha256"):
            raise ControlProtocolError("no-generation outcome sealed-history digest mismatch")
        return frontier
    raise ControlProtocolError("only sealed/outcome dispatch has frozen history")


def frozen_history_from_closed(
    root: Path, *, closed_path: Path | str | None, closed_sha256: str | None,
) -> tuple[dict[str, object], ...]:
    """Expose the exact predecessor frontier for the prep schedule transaction."""

    if closed_path is None:
        if closed_sha256 is not None:
            raise ControlProtocolError("empty predecessor path has a checksum")
        return ()
    if not isinstance(closed_sha256, str):
        raise ControlProtocolError("predecessor closed marker lacks checksum")
    return _history_from_closed_marker(Path(root), Path(closed_path), closed_sha256)


def _intent_payload(
    root: Path,
    target: ActivationTarget,
    *,
    observed_previous_closed_sha256: str | None,
    observed_previous_closed_path: str | None,
    observed_pointer_sha256: str | None,
    staging_id: str,
) -> dict[str, object]:
    return {
        "intent_format_version": INTENT_FORMAT_VERSION,
        "analysis_id": target.analysis_id,
        "execution_plan_id": target.execution_plan_id,
        "execution_contract_sha256": target.execution_contract_sha256,
        "round": target.round_index,
        "submission_generation": target.submission_generation,
        "expected_previous_generation_or_null": target.expected_previous_generation,
        "expected_previous_execution_plan_id_or_null": target.expected_previous_execution_plan_id,
        "expected_previous_round_or_null": target.expected_previous_round_index,
        "expected_pointer_version": target.expected_pointer_version,
        "observed_previous_closed_sha256_or_null": observed_previous_closed_sha256,
        "observed_previous_closed_path_or_null": observed_previous_closed_path,
        "observed_pointer_sha256_or_null": observed_pointer_sha256,
        "prep_job_id": target.prep_job_id,
        "prep_token": target.prep_token,
        "staging_id": staging_id,
        "canonical_generation_path": str(generation_dir(root, target).resolve()),
    }


def _target_from_intent(payload: Mapping[str, object]) -> ActivationTarget:
    """Decode an intent only to decide whether its reservation is resolved."""

    try:
        target = ActivationTarget(
            analysis_id=str(payload["analysis_id"]),
            execution_plan_id=str(payload["execution_plan_id"]),
            execution_contract_sha256=str(payload["execution_contract_sha256"]),
            round_index=int(payload["round"]),
            submission_generation=str(payload["submission_generation"]),
            expected_previous_generation=(
                None if payload.get("expected_previous_generation_or_null") is None
                else str(payload["expected_previous_generation_or_null"])
            ),
            expected_pointer_version=int(payload["expected_pointer_version"]),
            prep_job_id=str(payload["prep_job_id"]),
            prep_token=str(payload["prep_token"]),
            expected_previous_execution_plan_id=(
                None if payload.get("expected_previous_execution_plan_id_or_null") is None
                else str(payload["expected_previous_execution_plan_id_or_null"])
            ),
            expected_previous_round_index=(
                None if payload.get("expected_previous_round_or_null") is None
                else int(payload["expected_previous_round_or_null"])
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ControlProtocolError("activation intent target is invalid") from exc
    target.validate()
    if not _target_matches_payload(payload, target):
        raise ControlProtocolError("activation intent target identity is invalid")
    return target


def _intent_is_resolved(root: Path, payload: Mapping[str, object], pointer: Mapping[str, object] | None) -> bool:
    """Only a pointer reference or a valid closed generation resolves intent."""

    target = _target_from_intent(payload)
    activation = _validate_activation(root, target)
    if activation is None:
        return False
    activation_file, _, activation_sha = activation
    if pointer is not None and (
        pointer.get("generation_activation_path") == str(activation_file.resolve())
        and pointer.get("generation_activation_sha256") == activation_sha
    ):
        return True
    # A later successful activation may move the active pointer onward.  The
    # prior reservation remains resolved only if its immutable closed marker
    # binds this activation exactly; an activation record before CAS does not.
    return _validate_closed(root, target, activation_sha) is not None


def _check_predecessor(root: Path, target: ActivationTarget, pointer: Mapping[str, object] | None) -> str | None:
    if target.expected_previous_generation is None:
        if pointer is not None:
            raise ControlSupersededError("initial generation expected no active pointer")
        return None
    if pointer is None:
        raise ControlBusyError("expected predecessor pointer is absent")
    if int(pointer["pointer_version"]) != target.expected_pointer_version:
        if int(pointer["pointer_version"]) > target.expected_pointer_version:
            raise ControlSupersededError("pointer advanced beyond frozen predecessor version")
        raise ControlBusyError("pointer has not reached frozen predecessor version")
    activation_ref = Path(str(pointer["generation_activation_path"]))
    if not activation_ref.exists():
        raise ControlProtocolError("active pointer references missing activation")
    if _sha(activation_ref) != pointer["generation_activation_sha256"]:
        raise ControlProtocolError("active pointer activation checksum mismatch")
    predecessor_activation = _load_canonical_json(activation_ref, label="predecessor activation")
    if predecessor_activation.get("submission_generation") != target.expected_previous_generation:
        raise ControlSupersededError("pointer predecessor identity differs from frozen target")
    expected_plan = target.expected_previous_execution_plan_id or target.execution_plan_id
    if predecessor_activation.get("execution_plan_id") != expected_plan:
        raise ControlSupersededError("pointer predecessor execution plan differs from frozen target")
    if target.expected_previous_round_index is not None and predecessor_activation.get("round") != target.expected_previous_round_index:
        raise ControlSupersededError("pointer predecessor round differs from frozen target")
    predecessor_closed = activation_ref.parent / "generation.closed.json"
    if not predecessor_closed.exists():
        raise ControlBusyError("predecessor generation is not sealed")
    closed = _load_canonical_json(predecessor_closed, label="predecessor closed marker")
    if closed.get("generation_activation_sha256") != pointer["generation_activation_sha256"]:
        raise ControlProtocolError("predecessor closed marker does not bind pointer activation")
    return _sha(predecessor_closed)


def _predecessor_closed_path(pointer: Mapping[str, object] | None) -> str | None:
    if pointer is None:
        return None
    activation_ref = Path(str(pointer["generation_activation_path"])).resolve()
    return str((activation_ref.parent / "generation.closed.json").resolve())


def _expected_no_generation_outcome(
    root: Path, target: ActivationTarget,
) -> tuple[Path, dict[str, object], str] | None:
    """Read only the statically pre-submitted predecessor outcome path."""

    if target.expected_previous_generation is None:
        return None
    predecessor_plan = target.expected_previous_execution_plan_id or target.execution_plan_id
    predecessor_round = target.expected_previous_round_index or (target.round_index - 1)
    if predecessor_round < 1:
        return None
    candidate = (
        Path(root) / "executions" / predecessor_plan / f"round-{predecessor_round}"
        / "prep-outcomes" / f"{target.expected_previous_generation}.json"
    )
    if not candidate.exists():
        return None
    payload = _load_canonical_json(candidate, label="prospective predecessor no-generation outcome")
    expected = {
        "analysis_id": target.analysis_id,
        "execution_plan_id": predecessor_plan,
        "round": predecessor_round,
        "submission_generation": target.expected_previous_generation,
        "outcome": "no-generation",
        "todo_count": 0,
        "no_generation": True,
    }
    if payload.get("outcome_format_version") != OUTCOME_FORMAT_VERSION or any(
        payload.get(key) != value for key, value in expected.items()
    ):
        raise ControlProtocolError("prospective predecessor no-generation outcome identity mismatch")
    if target.expected_previous_execution_plan_id is not None:
        # A cross-plan predecessor is a sealed pointer by construction; an
        # outcome has no pointer activation to carry across plan identities.
        raise ControlProtocolError("cross-plan predecessor may not be a no-generation outcome")
    frontier = payload.get("sealed_history_frontier")
    if not isinstance(frontier, list):
        raise ControlProtocolError("prospective predecessor outcome lacks frozen frontier")
    frozen = _validate_frozen_frontier(Path(root), [item for item in frontier if isinstance(item, Mapping)])
    if len(frozen) != len(frontier) or sha256_bytes(canonical_json_bytes(list(frozen))) != payload.get("sealed_history_digest_sha256"):
        raise ControlProtocolError("prospective predecessor outcome history mismatch")
    return candidate, payload, _sha(candidate)


def predecessor_gate(root: Path, target: ActivationTarget) -> tuple[str, object | None]:
    """Apply the exact predecessor gate without looking at Slurm status.

    A no-generation predecessor is deliberately valid even though it leaves
    the active pointer unchanged; later pre-submitted rounds must be able to
    publish their own exact no-generation outcome.
    """

    root = Path(root)
    outcome = _expected_no_generation_outcome(root, target)
    if outcome is not None:
        return "no-generation", outcome
    pointer, _ = _pointer(root)
    closed_sha = _check_predecessor(root, target, pointer)
    return "sealed-generation", (closed_sha, _predecessor_closed_path(pointer))


def publish_no_generation_outcome(
    root: Path,
    target: ActivationTarget,
    *,
    sealed_history_frontier: Sequence[Mapping[str, object]] | None = None,
    fault: Callable[[str], None] | None = None,
    lease_held: bool = False,
) -> Path:
    """Atomically publish the only legal todo=0 durable success artefact."""

    target.validate()
    root = Path(root)
    lease = nullcontext() if lease_held else _lease(analysis_schedule_lease(root), exclusive=True)
    with lease:
        if intent_path(root, target).exists() or generation_dir(root, target).exists() or activation_path(root, target).exists():
            raise ControlProtocolError("no-generation outcome conflicts with activation artefacts")
        pointer, raw_pointer = _pointer(root)
        if pointer is not None and pointer.get("analysis_id") != target.analysis_id:
            raise ControlProtocolError("active pointer has a different analysis identity")
        predecessor_kind, predecessor = predecessor_gate(root, target)
        if predecessor_kind == "no-generation":
            assert isinstance(predecessor, tuple)
            _, prior_outcome, _ = predecessor
            assert isinstance(prior_outcome, Mapping)
            prior_closed = prior_outcome.get("observed_previous_closed_sha256_or_null")
            prior_closed_path = prior_outcome.get("observed_previous_closed_path_or_null")
            raw_frontier = prior_outcome.get("sealed_history_frontier")
            assert isinstance(raw_frontier, list)
            frozen_frontier = _validate_frozen_frontier(
                root, [item for item in raw_frontier if isinstance(item, Mapping)],
            )
        else:
            assert isinstance(predecessor, tuple)
            prior_closed, prior_closed_path = predecessor
            frozen_frontier = (
                () if prior_closed_path is None
                else _history_from_closed_marker(root, Path(str(prior_closed_path)), str(prior_closed))
            )
        if sealed_history_frontier is not None:
            supplied = _validate_frozen_frontier(root, sealed_history_frontier)
            if supplied != frozen_frontier:
                raise ControlProtocolError("supplied no-generation frontier is not the frozen predecessor history")
        frontier = [dict(item) for item in frozen_frontier]
        history_digest = sha256_bytes(canonical_json_bytes(frontier))
        payload = {
            "outcome_format_version": OUTCOME_FORMAT_VERSION,
            "outcome": "no-generation",
            "analysis_id": target.analysis_id,
            "execution_plan_id": target.execution_plan_id,
            "execution_contract_sha256": target.execution_contract_sha256,
            "round": target.round_index,
            "submission_generation": target.submission_generation,
            "prep_job_id": target.prep_job_id,
            "prep_token": target.prep_token,
            "prospective_previous_generation_or_null": target.expected_previous_generation,
            "prospective_previous_execution_plan_id_or_null": target.expected_previous_execution_plan_id,
            "prospective_previous_round_or_null": target.expected_previous_round_index,
            "prospective_pointer_version": target.expected_pointer_version,
            "observed_pointer_version": 0 if pointer is None else int(pointer["pointer_version"]),
            "observed_pointer_sha256_or_null": None if raw_pointer is None else sha256_bytes(raw_pointer),
            "observed_previous_closed_sha256_or_null": prior_closed,
            "observed_previous_closed_path_or_null": prior_closed_path,
            "sealed_history_generation_count": len(frontier),
            "sealed_history_frontier": frontier,
            "sealed_history_digest_sha256": history_digest,
            "todo_count": 0,
            "no_generation": True,
        }
        _write_temp_fsync_rename(outcome_path(root, target), payload, fault=fault)
        return outcome_path(root, target)


def publish_activation_intent(root: Path, target: ActivationTarget, *, lease_held: bool = False) -> Path:
    """Durably reserve one exact activation target before any generation file.

    The returned intent is deliberately the only authority for a subsequent
    staging/recovery attempt.  A second prep may resume byte-identical intent
    but may not choose another target for the same pointer version.
    """

    target.validate()
    root = Path(root)
    lease = nullcontext() if lease_held else _lease(analysis_schedule_lease(root), exclusive=True)
    with lease:
        if outcome_path(root, target).exists():
            raise ControlProtocolError("activation intent conflicts with no-generation outcome")
        pointer, raw_pointer = _pointer(root)
        if pointer is not None and pointer.get("analysis_id") != target.analysis_id:
            raise ControlProtocolError("active pointer belongs to another analysis")
        existing_target = intent_path(root, target)
        if existing_target.exists():
            existing = _load_canonical_json(existing_target, label="activation intent")
            if existing.get("intent_format_version") != INTENT_FORMAT_VERSION or not _target_matches_payload(existing, target):
                raise ControlProtocolError("existing activation intent identity mismatch")
            current_sha = None if raw_pointer is None else sha256_bytes(raw_pointer)
            if existing.get("observed_pointer_sha256_or_null") != current_sha:
                if pointer is not None and int(pointer["pointer_version"]) > target.expected_pointer_version:
                    raise ControlSupersededError("published intent was overtaken before activation")
                raise ControlBusyError("published intent pointer snapshot is not currently recoverable")
            return existing_target
        intents_root = root / "activation-intents"
        if intents_root.exists():
            unresolved: list[dict[str, object]] = []
            for candidate in sorted(intents_root.glob("*.json")):
                payload = _load_canonical_json(candidate, label="activation intent")
                if payload.get("intent_format_version") != INTENT_FORMAT_VERSION:
                    raise ControlProtocolError("activation intent format mismatch")
                if payload.get("analysis_id") != target.analysis_id:
                    raise ControlProtocolError("activation intent belongs to another analysis")
                # A published but not activated intent is an exclusive
                # schedule reservation.  Do not pick a different target or
                # silently turn a race into another generation.
                if not _intent_is_resolved(root, payload, pointer):
                    unresolved.append(payload)
            if len(unresolved) > 1:
                raise ControlProtocolError("multiple published activation intents are unresolved")
            if unresolved:
                other_version = unresolved[0].get("expected_pointer_version")
                if not isinstance(other_version, int):
                    raise ControlProtocolError("unresolved activation intent pointer version is invalid")
                if other_version == target.expected_pointer_version:
                    raise ControlSupersededError("another frozen target already owns this pointer version")
                raise ControlBusyError("another activation intent is unresolved")
        predecessor_sha = _check_predecessor(root, target, pointer)
        predecessor_path = _predecessor_closed_path(pointer)
        intent = _intent_payload(
            root, target,
            observed_previous_closed_sha256=predecessor_sha,
            observed_previous_closed_path=predecessor_path,
            observed_pointer_sha256=None if raw_pointer is None else sha256_bytes(raw_pointer),
            staging_id=target.submission_generation,
        )
        _write_temp_fsync_rename(existing_target, intent)
        return existing_target


def activate_generation(
    root: Path,
    target: ActivationTarget,
    *,
    assignment_path: Path,
    assignment_sha256: str,
    assignment_index_path: Path,
    assignment_index_sha256: str,
    ready_path: Path,
    ready_sha256: str,
    prep_path: Path,
    prep_sha256: str,
    worker_count: int,
    fault: Callable[[str], None] | None = None,
    lease_held: bool = False,
) -> Path:
    """Publish intent → prepared → activation → pointer-CAS for one target.

    Assignment construction happens before this call in a private sibling work
    directory.  This function only accepts immutable files and never replaces
    a canonical generation directory.
    """

    target.validate()
    root = Path(root)
    canonical_generation = generation_dir(root, target)
    canonical_activation = activation_path(root, target)
    lease = nullcontext() if lease_held else _lease(analysis_schedule_lease(root), exclusive=True)
    with lease:
        if outcome_path(root, target).exists():
            raise ControlProtocolError("activation conflicts with no-generation outcome")
        pointer, raw_pointer = _pointer(root)
        if pointer is not None and pointer.get("analysis_id") != target.analysis_id:
            raise ControlProtocolError("active pointer belongs to a different analysis")
        existing_intent = intent_path(root, target)
        if not existing_intent.exists():
            raise ControlBusyError("activation intent has not been durably published")
        intent = _load_canonical_json(existing_intent, label="activation intent")
        if intent.get("intent_format_version") != INTENT_FORMAT_VERSION or not _target_matches_payload(intent, target):
            raise ControlProtocolError("published activation intent identity mismatch")
        predecessor_sha = intent.get("observed_previous_closed_sha256_or_null")
        predecessor_path = intent.get("observed_previous_closed_path_or_null")
        current_pointer_sha = intent.get("observed_pointer_sha256_or_null")
        intent_sha = _sha(existing_intent)
        lease_file = canonical_generation / "generation.lease"
        if not canonical_generation.is_dir() or not lease_file.is_file():
            raise ControlBusyError("activation needs a durably promoted generation lease")
        prepared = {
            "prepared_format_version": PREPARED_FORMAT_VERSION,
            "analysis_id": target.analysis_id,
            "execution_plan_id": target.execution_plan_id,
            "execution_contract_sha256": target.execution_contract_sha256,
            "round": target.round_index,
            "submission_generation": target.submission_generation,
            "intent_sha256": intent_sha,
            "assignment_path": str(Path(assignment_path).resolve()),
            "assignment_sha256": assignment_sha256,
            "assignment_index_path": str(Path(assignment_index_path).resolve()),
            "assignment_index_sha256": assignment_index_sha256,
            "ready_path": str(Path(ready_path).resolve()),
            "ready_sha256": ready_sha256,
            "prep_path": str(Path(prep_path).resolve()),
            "prep_sha256": prep_sha256,
            "worker_count": int(worker_count),
        }
        prepared_file = canonical_generation / "generation.prepared.json"
        prepared_sha = _write_temp_fsync_rename(prepared_file, prepared, fault=fault)
        activation = {
            "activation_format_version": ACTIVATION_FORMAT_VERSION,
            "analysis_id": target.analysis_id,
            "execution_plan_id": target.execution_plan_id,
            "execution_contract_sha256": target.execution_contract_sha256,
            "round": target.round_index,
            "submission_generation": target.submission_generation,
            "prep_job_id": target.prep_job_id,
            "prep_token": target.prep_token,
            "intent_sha256": intent_sha,
            "prepared_sha256": prepared_sha,
            "generation_lease_path": str(lease_file.resolve()),
            "assignment_path": str(Path(assignment_path).resolve()),
            "assignment_sha256": assignment_sha256,
            "assignment_index_path": str(Path(assignment_index_path).resolve()),
            "assignment_index_sha256": assignment_index_sha256,
            "ready_path": str(Path(ready_path).resolve()),
            "ready_sha256": ready_sha256,
            "worker_count": int(worker_count),
            "expected_previous_generation_or_null": target.expected_previous_generation,
            "expected_previous_execution_plan_id_or_null": target.expected_previous_execution_plan_id,
            "expected_previous_round_or_null": target.expected_previous_round_index,
            "expected_pointer_version": target.expected_pointer_version,
            "observed_previous_closed_sha256_or_null": predecessor_sha,
            "observed_previous_closed_path_or_null": predecessor_path,
            "observed_pointer_sha256_or_null": current_pointer_sha,
        }
        activation_sha = _write_temp_fsync_rename(canonical_activation, activation, fault=fault)
        again_pointer, again_raw = _pointer(root)
        if (None if again_raw is None else sha256_bytes(again_raw)) != current_pointer_sha:
            if (
                again_pointer is not None
                and again_pointer.get("generation_activation_sha256") == activation_sha
                and again_pointer.get("generation_activation_path") == str(canonical_activation.resolve())
            ):
                return canonical_activation
            raise ControlSupersededError("active pointer changed before activation CAS")
        if _check_predecessor(root, target, again_pointer) != predecessor_sha:
            raise ControlSupersededError("predecessor closed marker changed before activation CAS")
        pointer_payload = {
            "pointer_format_version": POINTER_FORMAT_VERSION,
            "analysis_id": target.analysis_id,
            "pointer_version": target.expected_pointer_version + 1,
            "generation_activation_path": str(canonical_activation.resolve()),
            "generation_activation_sha256": activation_sha,
        }
        pointer_bytes = canonical_json_bytes(pointer_payload) + b"\n"
        temporary = pointer_path(root).with_name(f".active-generation.json.tmp.{uuid.uuid4().hex}")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
        try:
            _write_all(descriptor, pointer_bytes)
            if fault is not None:
                fault("pointer_before_file_fsync")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if fault is not None:
            fault("pointer_after_file_fsync")
            fault("pointer_before_replace")
        os.replace(temporary, pointer_path(root))
        if fault is not None:
            fault("pointer_after_replace")
            fault("pointer_before_root_fsync")
        _fsync_directory(root)
        if fault is not None:
            fault("pointer_after_root_fsync")
        return canonical_activation


def seal_generation(
    root: Path,
    target: ActivationTarget,
    *,
    inventory: Mapping[str, object] | None = None,
    inventory_builder: Callable[[], Mapping[str, object]] | None = None,
    diagnostic: Mapping[str, object] | None = None,
) -> Path:
    """Close precisely the active target, or read-only validate a closed one."""

    if (inventory is None) == (inventory_builder is None):
        raise ControlProtocolError("seal_generation requires exactly one inventory source")
    root = Path(root)
    dispatch = classify_exact_afterany_target_read_only(root, target)
    if dispatch.kind == "sealed-generation":
        return Path(dispatch.closed_path)
    if dispatch.kind != "active-generation":
        if dispatch.exit_code == SUPERSEDED_EXIT_CODE:
            raise ControlSupersededError("target was superseded before sealing")
        if dispatch.exit_code == PROTOCOL_EXIT_CODE:
            raise ControlProtocolError("cannot seal protocol-invalid target")
        raise ControlBusyError("target activation is not ready to seal")
    assert dispatch.activation_path is not None and dispatch.activation_sha256 is not None
    generation_lease = generation_dir(root, target) / "generation.lease"
    with _lease(analysis_schedule_lease(root), exclusive=False, create=False):
        with _lease(generation_lease, exclusive=True, create=False):
            rechecked = classify_exact_afterany_target_read_only(root, target)
            if rechecked.kind == "sealed-generation":
                return Path(rechecked.closed_path)
            if rechecked.kind != "active-generation":
                raise ControlBusyError("target changed while taking close lease")
            immutable_inventory = dict(inventory_builder() if inventory_builder is not None else inventory or {})
            payload = {
                "closed_format_version": CLOSED_FORMAT_VERSION,
                "analysis_id": target.analysis_id,
                "execution_plan_id": target.execution_plan_id,
                "execution_contract_sha256": target.execution_contract_sha256,
                "round": target.round_index,
                "submission_generation": target.submission_generation,
                "prep_job_id": target.prep_job_id,
                "prep_token": target.prep_token,
                "generation_activation_sha256": dispatch.activation_sha256,
                "inventory": immutable_inventory,
                "diagnostic": None if diagnostic is None else dict(diagnostic),
            }
            _write_temp_fsync_rename(closed_path(root, target), payload)
            return closed_path(root, target)


def verification_path(root: Path, target: ActivationTarget) -> Path:
    return Path(root) / "verifications" / target.execution_plan_id / f"round-{target.round_index}" / f"{target.submission_generation}.json"


def publish_verification_receipt(
    root: Path,
    target: ActivationTarget,
    *,
    dispatch: Dispatch,
    sealed_history_digest_sha256: str,
    complete: bool,
    exit_code: int,
    extra: Mapping[str, object] | None = None,
) -> Path:
    if dispatch.kind not in {"sealed-generation", "no-generation"}:
        raise ControlProtocolError("verification may only consume sealed generation or exact no-generation outcome")
    if (dispatch.closed_sha256 is None) == (dispatch.outcome_sha256 is None):
        raise ControlProtocolError("verification dispatch must bind exactly one target hash")
    payload = {
        "verification_format_version": VERIFICATION_FORMAT_VERSION,
        "analysis_id": target.analysis_id,
        "last_execution_plan_id": target.execution_plan_id,
        "last_round": target.round_index,
        "last_submission_generation": target.submission_generation,
        "dispatch_kind": dispatch.kind,
        "generation_closed_sha256_or_null": dispatch.closed_sha256,
        "prep_outcome_sha256_or_null": dispatch.outcome_sha256,
        "sealed_history_digest_sha256": sealed_history_digest_sha256,
        "complete": bool(complete),
        "exit_code": int(exit_code),
        **dict(extra or {}),
    }
    _write_temp_fsync_rename(verification_path(root, target), payload)
    return verification_path(root, target)


def validate_exact_verification_receipt(
    root: Path,
    target: ActivationTarget,
    *,
    expected_dispatch: Dispatch,
    sealed_history_digest_sha256: str,
) -> dict[str, object]:
    """Validate before a finalizer opens SQLite or creates a final temp file."""

    path = verification_path(root, target)
    payload = _load_canonical_json(path, label="verification receipt")
    if payload.get("verification_format_version") != VERIFICATION_FORMAT_VERSION:
        raise ControlProtocolError("verification receipt format is invalid")
    checks = {
        "analysis_id": target.analysis_id,
        "last_execution_plan_id": target.execution_plan_id,
        "last_round": target.round_index,
        "last_submission_generation": target.submission_generation,
        "dispatch_kind": expected_dispatch.kind,
        "sealed_history_digest_sha256": sealed_history_digest_sha256,
        "complete": True,
        "exit_code": SUCCESS_EXIT_CODE,
    }
    if any(payload.get(key) != value for key, value in checks.items()):
        raise ControlProtocolError("verification receipt does not match frozen finalization target")
    if payload.get("generation_closed_sha256_or_null") != expected_dispatch.closed_sha256:
        raise ControlProtocolError("verification receipt closed hash mismatch")
    if payload.get("prep_outcome_sha256_or_null") != expected_dispatch.outcome_sha256:
        raise ControlProtocolError("verification receipt outcome hash mismatch")
    return payload
