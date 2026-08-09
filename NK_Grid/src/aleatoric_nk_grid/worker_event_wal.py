"""Generation-scoped, two-sync worker event WAL.

This module deliberately knows nothing about Slurm, Parquet, or models.  A
single file is both the stable writer lease inode and the durable event stream
for one ``(plan, round, generation, worker)`` invocation.  Readers never
repair; only the exclusive writer may remove an uncommitted physical tail.
"""

from __future__ import annotations

import csv
import errno
import fcntl
import hashlib
import io
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, Sequence

from .execution_contract import canonical_json_bytes, sha256_bytes


WAL_FORMAT = "worker-event-wal-v1"
WAL_IDENTITY_LIMIT = 64 * 1024
WAL_METADATA_LIMIT = 4 * 1024
WAL_PAYLOAD_LIMIT = 8 * 1024 * 1024
ABORT_PAYLOAD_LIMIT = 4 * 1024
PREPARE_MAGIC = b"NKGRID-WAL-PREPARE-1\n"
COMMIT_MAGIC = b"NKGRID-WAL-COMMIT-1\n"
_LENGTH = struct.Struct(">I")

TASK_STARTED = "TASK_STARTED"
TASK_RESULT = "TASK_RESULT"
TASK_ABORTED = "TASK_ABORTED"
IDENTITY = "IDENTITY"
EVENT_TYPES = frozenset({TASK_STARTED, TASK_RESULT, TASK_ABORTED})
ABORT_REASONS = frozenset({
    "RESULT_FRAME_TOO_LARGE",
    "PUBLIC_SCHEMA_MISMATCH",
    "RESULT_PROJECTION_FAILED",
    "RESULT_KEY_SET_MISMATCH",
    "RESULT_ENCODING_FAILED",
    "RESULT_PROTOCOL_VIOLATION",
})


class WALProtocolError(ValueError):
    """A committed WAL frame or identity is invalid and must not be skipped."""


class WALBusyError(RuntimeError):
    """A non-blocking reader/writer lease could not be obtained."""


class WALFrameTooLarge(WALProtocolError):
    """The result needs an ABORTED record rather than truncation."""


@dataclass(frozen=True)
class WALRecord:
    event_type: str
    sequence: int
    row_id: str | None
    payload: bytes
    offset: int
    end_offset: int
    header: Mapping[str, object]


@dataclass(frozen=True)
class WALScan:
    identity: Mapping[str, object] | None
    records: tuple[WALRecord, ...]
    committed_offset: int
    has_uncommitted_tail: bool
    trailer_digest: str | None


def _sync_fd(descriptor: int) -> None:
    # fdatasync avoids a needless metadata flush for records but is not
    # universally exposed (notably some test doubles).  fsync is equivalent
    # for the protocol's durability guarantee.
    sync = getattr(os, "fdatasync", os.fsync)
    sync(descriptor)


def _write_all(descriptor: int, payload: bytes) -> None:
    """Write one protocol record completely, retrying interrupted short writes.

    A two-phase fsync only means what it says after *all* body/trailer bytes
    have reached the fd.  ``os.write`` is permitted to return a short count,
    including on a network filesystem, so a single call is not a durable
    record boundary.
    """

    view = memoryview(payload)
    while view:
        try:
            written = os.write(descriptor, view)
        except InterruptedError:
            continue
        if written is None or written <= 0:
            raise WALProtocolError("short write while publishing WAL record")
        view = view[written:]


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(Path(path).parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _lock_nonblocking(descriptor: int, operation: int) -> None:
    try:
        fcntl.flock(descriptor, operation | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise WALBusyError("WAL lease is held by another process") from exc
    except OSError as exc:
        if exc.errno in {errno.EACCES, errno.EAGAIN}:
            raise WALBusyError("WAL lease is held by another process") from exc
        raise


def _decode_json(value: bytes, *, label: str, maximum: int) -> dict[str, object]:
    if len(value) > maximum:
        raise WALProtocolError(f"{label} exceeds protocol limit")
    try:
        decoded = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WALProtocolError(f"invalid {label} JSON") from exc
    if not isinstance(decoded, dict):
        raise WALProtocolError(f"{label} must be a JSON object")
    return decoded


def _read_exact(handle, length: int) -> bytes | None:
    value = handle.read(length)
    if not value:
        return None
    if len(value) != length:
        raise EOFError
    return value


def _record_header(
    *, event_type: str, sequence: int, row_id: str | None, payload: bytes,
) -> tuple[bytes, dict[str, object]]:
    if event_type != IDENTITY and event_type not in EVENT_TYPES:
        raise WALProtocolError(f"unsupported WAL event type {event_type!r}")
    if event_type == IDENTITY:
        if sequence != -1 or row_id is not None:
            raise WALProtocolError("identity header has invalid sequence or row ID")
        maximum = WAL_IDENTITY_LIMIT
    elif event_type in {TASK_STARTED, TASK_ABORTED}:
        if sequence < 0 or not isinstance(row_id, str) or not row_id:
            raise WALProtocolError("event record requires non-negative sequence and row ID")
        maximum = WAL_METADATA_LIMIT if event_type == TASK_STARTED else ABORT_PAYLOAD_LIMIT
    else:
        if sequence < 0 or not isinstance(row_id, str) or not row_id:
            raise WALProtocolError("event record requires non-negative sequence and row ID")
        maximum = WAL_PAYLOAD_LIMIT
    if len(payload) > maximum:
        raise WALFrameTooLarge(
            f"{event_type} payload is {len(payload)} bytes; protocol maximum is {maximum}"
        )
    provisional = {
        "event_type": event_type,
        "payload_length": len(payload),
        "payload_sha256": sha256_bytes(payload),
        "row_id": row_id,
        "sequence": sequence,
        "wal_format": WAL_FORMAT,
    }
    encoded_without_hash = canonical_json_bytes(provisional)
    header = {**provisional, "header_sha256": sha256_bytes(encoded_without_hash)}
    encoded = canonical_json_bytes(header)
    if len(encoded) > WAL_METADATA_LIMIT:
        raise WALProtocolError("record metadata exceeds 4 KiB protocol limit")
    return encoded, header


def _encode_frame(header: bytes, payload: bytes, trailer: bytes | None = None) -> bytes:
    value = PREPARE_MAGIC + _LENGTH.pack(len(header)) + header + payload
    if trailer is not None:
        value += COMMIT_MAGIC + _LENGTH.pack(len(trailer)) + trailer
    return value


def _commit_trailer(header: Mapping[str, object]) -> bytes:
    payload = {
        "header_sha256": header["header_sha256"],
        "payload_sha256": header["payload_sha256"],
        "sequence": header["sequence"],
        "wal_format": WAL_FORMAT,
    }
    encoded = canonical_json_bytes(payload)
    if len(encoded) > WAL_METADATA_LIMIT:
        raise WALProtocolError("commit trailer exceeds 4 KiB protocol limit")
    return encoded


def _validate_header(header: Mapping[str, object]) -> None:
    try:
        event_type = str(header["event_type"])
        sequence = int(header["sequence"])
        length = int(header["payload_length"])
        row_id = header["row_id"]
        payload_sha = str(header["payload_sha256"])
        header_sha = str(header["header_sha256"])
    except (KeyError, TypeError, ValueError) as exc:
        raise WALProtocolError("record metadata fields are invalid") from exc
    if header.get("wal_format") != WAL_FORMAT:
        raise WALProtocolError("WAL record format/version mismatch")
    if event_type == IDENTITY:
        if sequence != -1 or row_id is not None or length > WAL_IDENTITY_LIMIT:
            raise WALProtocolError("identity header protocol limits are invalid")
    elif event_type in EVENT_TYPES:
        maximum = (
            WAL_METADATA_LIMIT if event_type == TASK_STARTED
            else ABORT_PAYLOAD_LIMIT if event_type == TASK_ABORTED
            else WAL_PAYLOAD_LIMIT
        )
        if sequence < 0 or not isinstance(row_id, str) or not row_id or length < 0 or length > maximum:
            raise WALProtocolError("event header protocol limits are invalid")
    else:
        raise WALProtocolError("unknown WAL record type")
    expected = sha256_bytes(canonical_json_bytes({
        "event_type": event_type,
        "payload_length": length,
        "payload_sha256": payload_sha,
        "row_id": row_id,
        "sequence": sequence,
        "wal_format": WAL_FORMAT,
    }))
    if header_sha != expected:
        raise WALProtocolError("committed header checksum mismatch")


def scan_wal(path: Path, *, allow_uncommitted_tail: bool = True) -> WALScan:
    """Scan only a valid committed prefix; never resynchronise after damage."""

    target = Path(path)
    identity: Mapping[str, object] | None = None
    records: list[WALRecord] = []
    committed_offset = 0
    trailer_digest: str | None = None
    has_tail = False
    try:
        with target.open("rb") as handle:
            while True:
                frame_offset = handle.tell()
                magic = handle.read(len(PREPARE_MAGIC))
                if not magic:
                    break
                if magic != PREPARE_MAGIC:
                    raise WALProtocolError(f"bad WAL magic at offset {frame_offset}")
                try:
                    raw_length = _read_exact(handle, _LENGTH.size)
                    if raw_length is None:
                        raise EOFError
                    header_length = _LENGTH.unpack(raw_length)[0]
                    if header_length > WAL_METADATA_LIMIT:
                        raise WALProtocolError("record metadata length exceeds protocol limit")
                    raw_header = _read_exact(handle, header_length)
                    if raw_header is None:
                        raise EOFError
                    header = _decode_json(raw_header, label="record metadata", maximum=WAL_METADATA_LIMIT)
                    _validate_header(header)
                    payload_length = int(header["payload_length"])
                    payload = _read_exact(handle, payload_length)
                    if payload is None:
                        raise EOFError
                    if sha256_bytes(payload) != header["payload_sha256"]:
                        # A fully present body can be corrupt even if the
                        # commit trailer was later torn.  It is a committed
                        # corruption if a valid trailer follows, otherwise an
                        # uncommitted tail may safely be discarded by writer.
                        body_ok = False
                    else:
                        body_ok = True
                    commit_magic = _read_exact(handle, len(COMMIT_MAGIC))
                    if commit_magic is None:
                        raise EOFError
                    if commit_magic != COMMIT_MAGIC:
                        raise WALProtocolError("invalid commit trailer magic")
                    raw_commit_length = _read_exact(handle, _LENGTH.size)
                    if raw_commit_length is None:
                        raise EOFError
                    commit_length = _LENGTH.unpack(raw_commit_length)[0]
                    if commit_length > WAL_METADATA_LIMIT:
                        raise WALProtocolError("commit trailer length exceeds protocol limit")
                    raw_commit = _read_exact(handle, commit_length)
                    if raw_commit is None:
                        raise EOFError
                    trailer = _decode_json(raw_commit, label="commit trailer", maximum=WAL_METADATA_LIMIT)
                    if (
                        trailer.get("wal_format") != WAL_FORMAT
                        or trailer.get("sequence") != header["sequence"]
                        or trailer.get("header_sha256") != header["header_sha256"]
                        or trailer.get("payload_sha256") != header["payload_sha256"]
                    ):
                        raise WALProtocolError("commit trailer does not bind record body")
                    if not body_ok:
                        raise WALProtocolError("committed payload checksum mismatch")
                except EOFError:
                    if not allow_uncommitted_tail:
                        raise WALProtocolError("uncommitted WAL tail is not accepted") from None
                    has_tail = True
                    break
                event_type = str(header["event_type"])
                sequence = int(header["sequence"])
                row_id = header["row_id"]
                end_offset = handle.tell()
                record = WALRecord(
                    event_type=event_type,
                    sequence=sequence,
                    row_id=None if row_id is None else str(row_id),
                    payload=payload,
                    offset=frame_offset,
                    end_offset=end_offset,
                    header=header,
                )
                if identity is None:
                    if event_type != IDENTITY:
                        raise WALProtocolError("WAL has event before committed identity header")
                    identity = _decode_json(payload, label="identity payload", maximum=WAL_IDENTITY_LIMIT)
                elif event_type == IDENTITY:
                    raise WALProtocolError("WAL contains a second committed identity header")
                else:
                    records.append(record)
                committed_offset = end_offset
                trailer_digest = sha256_bytes(raw_commit)
    except OSError as exc:
        raise WALProtocolError(f"cannot scan WAL {target}: {exc}") from exc
    return WALScan(identity, tuple(records), committed_offset, has_tail, trailer_digest)


def _validate_event_state(records: Iterable[WALRecord]) -> None:
    starts: dict[tuple[int, str], WALRecord] = {}
    terminal: set[tuple[int, str]] = set()
    prior_started_sequence = -1
    for record in records:
        key = (record.sequence, str(record.row_id))
        if record.event_type == TASK_STARTED:
            if record.sequence <= prior_started_sequence:
                raise WALProtocolError("generation TASK_STARTED sequence is duplicate or decreasing")
            prior_started_sequence = record.sequence
            if key in starts:
                raise WALProtocolError("duplicate TASK_STARTED")
            starts[key] = record
        elif record.event_type in {TASK_RESULT, TASK_ABORTED}:
            if key not in starts:
                raise WALProtocolError("terminal event has no matching TASK_STARTED")
            if key in terminal:
                raise WALProtocolError("TASK_STARTED has multiple terminal events")
            terminal.add(key)
        else:
            raise WALProtocolError("unexpected event in event stream")


def bounded_abort_payload(
    *, reason_code: str, exception: BaseException | None = None,
    actual_bytes: int | None = None, diagnostic: str = "",
) -> bytes:
    if reason_code not in ABORT_REASONS:
        raise WALProtocolError(f"unsupported ABORTED reason code {reason_code!r}")
    exception_type = "" if exception is None else type(exception).__name__.encode("ascii", "ignore")[:128].decode("ascii")
    raw_diagnostic = diagnostic.encode("utf-8", "replace")
    prefix = raw_diagnostic[: 2 * 1024].decode("utf-8", "ignore")
    value = {
        "actual_bytes": None if actual_bytes is None else int(actual_bytes),
        "diagnostic_prefix": prefix,
        "diagnostic_sha256": sha256_bytes(raw_diagnostic),
        "diagnostic_truncated": len(raw_diagnostic) > len(prefix.encode("utf-8")),
        "exception_type": exception_type,
        "reason_code": reason_code,
    }
    encoded = canonical_json_bytes(value)
    if len(encoded) > ABORT_PAYLOAD_LIMIT:
        raise WALProtocolError("bounded ABORTED payload exceeded 4 KiB")
    return encoded


def encode_public_rows(rows: Sequence[Mapping[str, object]], *, header: Sequence[str] | None = None) -> bytes:
    """Use the same CSV serializer for WAL payloads and final materialization."""

    if not rows:
        raise WALProtocolError("TASK_RESULT must contain at least one public row")
    columns = list(header or rows[0].keys())
    if not columns or len(columns) != len(set(columns)):
        raise WALProtocolError("public result header is invalid")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    try:
        writer.writerows({column: row.get(column, "") for column in columns} for row in rows)
    except (TypeError, ValueError, csv.Error) as exc:
        raise WALProtocolError(f"cannot project result rows to CSV: {exc}") from exc
    encoded = stream.getvalue().encode("utf-8")
    if len(encoded) > WAL_PAYLOAD_LIMIT:
        raise WALFrameTooLarge(f"RESULT payload is {len(encoded)} bytes; protocol maximum is {WAL_PAYLOAD_LIMIT}")
    return encoded


def decode_public_rows(payload: bytes) -> tuple[tuple[str, ...], tuple[dict[str, str], ...]]:
    try:
        text = payload.decode("utf-8")
        reader = csv.DictReader(io.StringIO(text, newline=""))
        header = tuple(reader.fieldnames or ())
        rows = tuple(dict(row) for row in reader)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise WALProtocolError("RESULT payload is not valid public CSV") from exc
    if not header or not rows:
        raise WALProtocolError("RESULT payload has no public header or rows")
    return header, rows


class WorkerEventLog:
    """An exclusively owned writer with two-phase event commits."""

    def __init__(self, path: Path, descriptor: int, identity: Mapping[str, object], scan: WALScan):
        self.path = Path(path)
        self._descriptor = descriptor
        self.identity = dict(identity)
        self._scan = scan
        _validate_event_state(scan.records)
        self._next_sequence = (max((record.sequence for record in scan.records), default=-1) + 1)
        self._started = {
            (record.sequence, str(record.row_id)) for record in scan.records
            if record.event_type == TASK_STARTED
        }
        self._terminal = {
            (record.sequence, str(record.row_id)) for record in scan.records
            if record.event_type in {TASK_RESULT, TASK_ABORTED}
        }
        self._closed = False

    @classmethod
    def open_exclusive_and_repair(
        cls,
        path: Path,
        *,
        identity: Mapping[str, object],
        sync: Callable[[int], None] | None = None,
    ) -> "WorkerEventLog":
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(target, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o640)
            created = True
        except FileExistsError:
            descriptor = os.open(target, os.O_RDWR)
            created = False
        try:
            _lock_nonblocking(descriptor, fcntl.LOCK_EX)
            scan = scan_wal(target)
            canonical_identity = dict(identity)
            if canonical_identity.get("wal_format") != WAL_FORMAT:
                raise WALProtocolError("WAL identity must declare worker-event-wal-v1")
            if scan.identity is None:
                if scan.records:
                    raise WALProtocolError("WAL has events but no committed identity")
                if scan.has_uncommitted_tail:
                    os.ftruncate(descriptor, 0)
                    (sync or _sync_fd)(descriptor)
                writer = cls.__new__(cls)
                writer.path = target; writer._descriptor = descriptor; writer.identity = canonical_identity
                writer._scan = WALScan(None, (), 0, False, None); writer._next_sequence = 0; writer._started = set(); writer._terminal = set(); writer._closed = False
                writer._commit(IDENTITY, -1, None, canonical_json_bytes(canonical_identity), sync=sync)
                if created:
                    _fsync_parent(target)
                return writer
            if dict(scan.identity) != canonical_identity:
                raise WALProtocolError("committed WAL identity does not match assignment/contract scope")
            if scan.has_uncommitted_tail:
                os.ftruncate(descriptor, scan.committed_offset)
                (sync or _sync_fd)(descriptor)
                os.lseek(descriptor, scan.committed_offset, os.SEEK_SET)
                scan = scan_wal(target, allow_uncommitted_tail=False)
            else:
                # ``scan_wal`` deliberately uses a separate read fd.  The
                # append fd therefore still points at byte zero unless we
                # seek explicitly; without this, requeues overwrite identity.
                os.lseek(descriptor, 0, os.SEEK_END)
            return cls(target, descriptor, canonical_identity, scan)
        except BaseException:
            os.close(descriptor)
            raise

    @classmethod
    def open_shared(cls, path: Path, *, expected_identity: Mapping[str, object]) -> WALScan:
        target = Path(path)
        descriptor = os.open(target, os.O_RDONLY)
        try:
            _lock_nonblocking(descriptor, fcntl.LOCK_SH)
            scan = scan_wal(target)
            if scan.identity is None:
                # A dead creator can leave an uninitialized inode.  A reader
                # treats it as absent; only a later exclusive writer may fix it.
                if scan.records:
                    raise WALProtocolError("uninitialized WAL has committed event")
                return scan
            if dict(scan.identity) != dict(expected_identity):
                raise WALProtocolError("committed WAL identity mismatch")
            _validate_event_state(scan.records)
            return scan
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def _commit(
        self,
        event_type: str,
        sequence: int,
        row_id: str | None,
        payload: bytes,
        *,
        sync: Callable[[int], None] | None = None,
    ) -> None:
        if self._closed:
            raise WALProtocolError("cannot append to a closed WorkerEventLog")
        raw_header, header = _record_header(
            event_type=event_type, sequence=sequence, row_id=row_id, payload=payload,
        )
        _write_all(self._descriptor, _encode_frame(raw_header, payload))
        (sync or _sync_fd)(self._descriptor)  # body durable boundary
        raw_trailer = _commit_trailer(header)
        _write_all(self._descriptor, COMMIT_MAGIC + _LENGTH.pack(len(raw_trailer)) + raw_trailer)
        (sync or _sync_fd)(self._descriptor)  # commit durable boundary

    def commit_started(self, *, row_id: str, metadata: Mapping[str, object] | None = None) -> int:
        sequence = self._next_sequence
        self._commit(TASK_STARTED, sequence, row_id, canonical_json_bytes(dict(metadata or {})))
        self._started.add((sequence, str(row_id)))
        self._next_sequence += 1
        return sequence

    def commit_result(
        self, *, sequence: int, row_id: str, public_rows: Sequence[Mapping[str, object]],
        header: Sequence[str] | None = None,
    ) -> None:
        key = (int(sequence), str(row_id))
        if key not in self._started:
            raise WALProtocolError("TASK_RESULT sequence has no durable TASK_STARTED")
        if key in self._terminal:
            raise WALProtocolError("TASK_RESULT duplicates an existing terminal event")
        self._commit(TASK_RESULT, sequence, row_id, encode_public_rows(public_rows, header=header))
        self._terminal.add(key)

    def commit_aborted(self, *, sequence: int, row_id: str, payload: bytes) -> None:
        key = (int(sequence), str(row_id))
        if key not in self._started:
            raise WALProtocolError("TASK_ABORTED sequence has no durable TASK_STARTED")
        if key in self._terminal:
            raise WALProtocolError("TASK_ABORTED duplicates an existing terminal event")
        if len(payload) > ABORT_PAYLOAD_LIMIT:
            raise WALProtocolError("TASK_ABORTED payload exceeds 4 KiB")
        self._commit(TASK_ABORTED, sequence, row_id, payload)
        self._terminal.add(key)

    def committed_terminal_row_ids(self) -> set[str]:
        return {
            str(record.row_id)
            for record in self._scan.records
            if record.event_type == TASK_RESULT
        }

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                fcntl.flock(self._descriptor, fcntl.LOCK_UN)
            finally:
                os.close(self._descriptor)

    def __enter__(self) -> "WorkerEventLog":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
