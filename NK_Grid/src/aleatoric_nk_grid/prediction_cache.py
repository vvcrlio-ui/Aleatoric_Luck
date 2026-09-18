"""Durable, lossless prediction shards; only small references cross the queue.

One writer incarnation owns each append file. A successful ``append`` means the
entire checksummed record has been fsynced; it does not grant queue authority.
The queue's accepted lease/result reference remains authoritative. Sealed batch
indexes are disposable and can be rebuilt from the records. Readers never repair.
"""
from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import re
import struct
import time
import uuid
import zlib
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence

import numpy as np

FORMAT = "nk-prediction-cache-v1"
LEGACY_MAGIC = b"NKPRED01"
MAGIC = b"NKPRED02"
RECORD_ENCODING = "NKPRED02"
RECORD_MAGICS = {"NKPRED01": LEGACY_MAGIC, "NKPRED02": MAGIC}
FRAME = struct.Struct(">8sIQ32s")
MAX_HEADER_BYTES = 8 * 1024 * 1024
MAX_STORED_HEADER_BYTES = MAX_HEADER_BYTES + 65536  # zlib worst-case overhead allowance.
MAX_RAW_BYTES = 256 * 1024 * 1024
MAX_RECORD_BYTES = MAX_RAW_BYTES + MAX_HEADER_BYTES + 1024 * 1024
DEFAULT_SHARD_BYTES = 128 * 1024 * 1024
MAX_RECORDS_PER_SHARD = 32768
VALID_STATUSES = frozenset({"ok", "nonconverged", "skipped", "failed"})
INTEGER_PREDICTION_ARRAYS = frozenset({"oof_fold"})
_HEX = re.compile(r"^[0-9a-f]{64}$")


class CacheIntegrityError(ValueError):
    """A cache is corrupt, incomplete, conflicting or incompatible."""


class CacheBusyError(RuntimeError):
    """The writer/repair lease is held by another process."""


class CacheStorageError(OSError):
    """Persistence failed; the writer is blocked and cannot report success."""


def canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, TypeError) as exc:
        raise CacheIntegrityError("identity/metadata must be finite canonical JSON") from exc


def cache_identity(identity: Mapping[str, object]) -> str:
    """Hash the complete frozen identity; never infer identity from a model name."""
    if not isinstance(identity, Mapping) or not identity:
        raise CacheIntegrityError("a nonempty frozen identity is required")
    return hashlib.sha256(canonical_bytes(dict(identity))).hexdigest()


build_cache_identity = cache_identity


def encode_index_json(value):
    return b'ZJ01' + zlib.compress(canonical_bytes(value), 6)


def decode_index_json(value):
    if isinstance(value, bytes) and value.startswith(b'ZJ01'):
        decoder = zlib.decompressobj()
        raw = decoder.decompress(value[4:], MAX_HEADER_BYTES + 1)
        if len(raw) > MAX_HEADER_BYTES or not decoder.eof or decoder.unused_data:
            raise CacheIntegrityError('Compressed reference metadata is oversized or malformed')
        return json.loads(raw)
    return json.loads(value)


def fold_record_identity(parent_identity: str, fold: int) -> dict[str, object]:
    """Small exact fold key; its parent digest covers the full frozen pipeline."""
    if not isinstance(parent_identity, str) or not _HEX.fullmatch(parent_identity):
        raise CacheIntegrityError("fold identity requires a complete parent identity SHA256")
    if isinstance(fold, bool) or not isinstance(fold, int) or fold < -1:
        raise CacheIntegrityError("invalid prediction fold number")
    return {"format": "prediction-fold-v2", "parent_identity": parent_identity, "fold": fold}


def _parent_reference(identity: Mapping[str, object]) -> dict[str, object]:
    if isinstance(identity.get("parent"), Mapping):
        # Older records remain readable and retain their original identity hash.
        return {"parent_identity": cache_identity(identity["parent"]), "fold_identity_version": 1}
    if identity.get("format") == "prediction-fold-v2":
        normalized = fold_record_identity(identity.get("parent_identity"), identity.get("fold"))
        return {"parent_identity": normalized["parent_identity"], "fold_identity_version": 2}
    return {}


def safe_cache_path(root: Path | str, relative: str) -> Path:
    """Reject traversal, drive names and symlink escapes on Linux and Windows."""
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise CacheIntegrityError("cache reference must be a relative POSIX path")
    path = PurePosixPath(relative)
    if path.is_absolute() or any(p in {"", ".", ".."} for p in relative.split("/")):
        raise CacheIntegrityError("unsafe cache reference path")
    base = Path(root).resolve()
    target = base.joinpath(*path.parts).resolve()
    if target == base or not target.is_relative_to(base):
        raise CacheIntegrityError("cache reference escapes its designated root")
    return target


def _sync_directory(path: Path) -> None:
    if os.name == "nt":
        return  # Windows does not expose directory fsync via Python os.open.
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_all(handle, value: bytes) -> None:
    view = memoryview(value)
    while view:
        try:
            count = handle.write(view)
        except InterruptedError:
            continue
        if count is None or count <= 0:
            raise OSError(errno.EIO, "short cache write")
        view = view[count:]


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb", buffering=0) as handle:
            _write_all(handle, canonical_bytes(value))
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _lock(handle) -> None:
    try:
        if os.name == "nt":
            import msvcrt
            position = handle.tell()
            try:
                # A Windows mandatory byte lock must not block live prefix readers.
                handle.seek(2**62)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            finally:
                handle.seek(position)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
            raise CacheBusyError("prediction shard is owned by an active writer") from exc
        raise


def _unlock(handle) -> None:
    if os.name == "nt":
        import msvcrt
        position = handle.tell()
        try:
            handle.seek(2**62)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            handle.seek(position)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def initialize_cache(root: Path | str, manifest: Mapping[str, object]) -> dict[str, object]:
    """Freeze the run contract. A required cache cannot be disabled on resume."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    expected = {"format": FORMAT, "compression": "zlib", "dtype": "float64",
                "record_encoding": RECORD_ENCODING,
                **dict(manifest)}
    if (expected["format"] != FORMAT or expected["compression"] != "zlib" or expected["dtype"] != "float64"
            or expected["record_encoding"] not in RECORD_MAGICS):
        raise CacheIntegrityError("unsupported cache format/compression/precision")
    path = root / "manifest.json"
    # A link publishes without replacing an existing manifest in a startup race.
    temporary = root / f".manifest.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb", buffering=0) as handle:
            _write_all(handle, canonical_bytes(expected))
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
            _sync_directory(root)
        except FileExistsError:
            if json.loads(path.read_text(encoding="utf-8")) != expected:
                raise CacheIntegrityError("frozen prediction cache manifest differs")
    finally:
        temporary.unlink(missing_ok=True)
    for directory in ("shards", "sample-maps", "indexes", "meta-results"):
        (root / directory).mkdir(exist_ok=True)
    return expected


def _encode(identity: Mapping[str, object], arrays: Mapping[str, np.ndarray],
            metadata: Mapping[str, object], *, kind: str,
            record_encoding: str = RECORD_ENCODING) -> tuple[bytes, dict[str, object]]:
    if record_encoding not in RECORD_MAGICS:
        raise CacheIntegrityError("unsupported prediction record encoding")
    identity = dict(identity)
    digest = cache_identity(identity)
    status = str(metadata.get("status", "ok"))
    if status not in VALID_STATUSES:
        raise CacheIntegrityError(f"unsupported cache status: {status}")
    if kind != "sample_map" and status in {"skipped", "failed"}:
        if arrays or not metadata.get("reason"):
            raise CacheIntegrityError("skipped/failed records require reason and no fake arrays")
    chunks, specs, offset = [], {}, 0
    for name, array in sorted(arrays.items()):
        if not isinstance(name, str) or not name or len(name) > 256:
            raise CacheIntegrityError("invalid array name")
        array = np.asarray(array)
        if array.dtype.hasobject or array.dtype.kind not in "biufUS":
            raise CacheIntegrityError("object/pickle/structured arrays are forbidden")
        is_fold = name in INTEGER_PREDICTION_ARRAYS and array.dtype.kind == "i" and array.dtype.itemsize == 8
        if kind != "sample_map" and not is_fold and (array.dtype.kind != "f" or array.dtype.itemsize != 8):
            raise CacheIntegrityError("prediction arrays must already be float64")
        if array.ndim > 4 or array.dtype.itemsize > 65536:
            raise CacheIntegrityError("array shape/dtype exceeds protocol limit")
        if kind != "sample_map" and status in {"ok", "nonconverged"} and not np.isfinite(array).all():
            raise CacheIntegrityError("successful predictions must be finite")
        block = np.ascontiguousarray(array).tobytes()
        offset += len(block)
        if offset > MAX_RAW_BYTES:
            raise CacheIntegrityError("uncompressed arrays exceed record limit")
        specs[name] = {"dtype": array.dtype.str, "shape": list(array.shape),
                       "offset": offset - len(block), "nbytes": len(block)}
        chunks.append(block)
    raw = b"".join(chunks)
    compressed = zlib.compress(raw, level=6)
    header = {"format": FORMAT, "kind": kind, "identity": identity,
              "identity_sha256": digest, "metadata": dict(metadata), "status": status,
              "arrays": specs, "raw_bytes": len(raw), "raw_sha256": hashlib.sha256(raw).hexdigest(),
              "compression": "zlib"}
    header.update(_parent_reference(identity))
    if record_encoding != "NKPRED01":
        header["record_encoding"] = record_encoding
    encoded_header = canonical_bytes(header)
    if len(encoded_header) > MAX_HEADER_BYTES:
        raise CacheIntegrityError("record metadata exceeds header limit")
    stored_header = zlib.compress(encoded_header, level=6) if record_encoding == "NKPRED02" else encoded_header
    body = stored_header + compressed
    checksum = hashlib.sha256(body).digest()
    frame = FRAME.pack(RECORD_MAGICS[record_encoding], len(stored_header), len(compressed), checksum) + body
    if len(frame) > MAX_RECORD_BYTES:
        raise CacheIntegrityError("compressed record exceeds record limit")
    return frame, header


@dataclass(frozen=True)
class CacheRecord:
    identity: Mapping[str, object]
    arrays: Mapping[str, np.ndarray]
    metadata: Mapping[str, object]
    reference: Mapping[str, object]
    kind: str

    @property
    def status(self) -> str:
        return str(self.reference["status"])


def _integer(value: object, label: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise CacheIntegrityError(f"invalid {label}")
    return value


def _decode(frame: bytes, reference: Mapping[str, object]) -> CacheRecord:
    if len(frame) < FRAME.size:
        raise CacheIntegrityError("truncated prediction frame")
    magic, header_bytes, payload_bytes, checksum = FRAME.unpack(frame[:FRAME.size])
    if (magic not in RECORD_MAGICS.values() or header_bytes > MAX_STORED_HEADER_BYTES
            or (magic == LEGACY_MAGIC and header_bytes > MAX_HEADER_BYTES) or payload_bytes > MAX_RECORD_BYTES):
        raise CacheIntegrityError("invalid prediction frame header")
    if FRAME.size + header_bytes + payload_bytes != len(frame) or len(frame) > MAX_RECORD_BYTES:
        raise CacheIntegrityError("invalid prediction frame length")
    body = frame[FRAME.size:]
    if hashlib.sha256(body).digest() != checksum:
        raise CacheIntegrityError("prediction frame checksum mismatch")
    encoded_header = body[:header_bytes]
    if magic == MAGIC:
        decoder = zlib.decompressobj()
        try:
            encoded_header = decoder.decompress(encoded_header, MAX_HEADER_BYTES + 1)
        except zlib.error as exc:
            raise CacheIntegrityError("invalid compressed prediction header") from exc
        if (len(encoded_header) > MAX_HEADER_BYTES or not decoder.eof
                or decoder.unused_data or decoder.unconsumed_tail):
            raise CacheIntegrityError("compressed prediction header exceeds limit or boundary")
    try:
        header = json.loads(encoded_header)
    except (ValueError, UnicodeDecodeError, RecursionError) as exc:
        raise CacheIntegrityError("invalid prediction JSON header") from exc
    if not isinstance(header, dict) or header.get("format") != FORMAT or header.get("compression") != "zlib":
        raise CacheIntegrityError("unsupported prediction record format")
    if header.get("record_encoding", magic.decode("ascii")) != magic.decode("ascii"):
        raise CacheIntegrityError("prediction header encoding differs from frame version")
    if cache_identity(header.get("identity")) != header.get("identity_sha256"):
        raise CacheIntegrityError("prediction identity hash mismatch")
    raw_size = _integer(header.get("raw_bytes"), "uncompressed size", MAX_RAW_BYTES)
    decompressor = zlib.decompressobj()
    try:
        raw = decompressor.decompress(body[header_bytes:], raw_size + 1)
    except zlib.error as exc:
        raise CacheIntegrityError("invalid compressed prediction block") from exc
    if len(raw) != raw_size or not decompressor.eof or decompressor.unused_data or decompressor.unconsumed_tail:
        raise CacheIntegrityError("compressed block length/boundary violation")
    if hashlib.sha256(raw).hexdigest() != header.get("raw_sha256"):
        raise CacheIntegrityError("uncompressed prediction checksum mismatch")
    specs = header.get("arrays")
    if not isinstance(specs, dict) or len(specs) > 1024:
        raise CacheIntegrityError("invalid array table")
    arrays, cursor = {}, 0
    for name, spec in specs.items():
        if not isinstance(spec, dict):
            raise CacheIntegrityError("invalid array descriptor")
        try:
            dtype = np.dtype(spec["dtype"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CacheIntegrityError("invalid array dtype") from exc
        if dtype.hasobject or dtype.kind not in "biufUS" or dtype.itemsize > 65536:
            raise CacheIntegrityError("unsafe array dtype")
        shape = spec.get("shape")
        if not isinstance(shape, list) or len(shape) > 4:
            raise CacheIntegrityError("invalid array shape")
        shape = tuple(_integer(size, "array dimension", MAX_RAW_BYTES) for size in shape)
        nbytes = _integer(spec.get("nbytes"), "array bytes", MAX_RAW_BYTES)
        offset = _integer(spec.get("offset"), "array offset", MAX_RAW_BYTES)
        if offset != cursor or nbytes != math.prod(shape) * dtype.itemsize or offset + nbytes > len(raw):
            raise CacheIntegrityError("array shape/offset/length mismatch")
        arrays[name] = np.frombuffer(raw, dtype=dtype, count=math.prod(shape), offset=offset).reshape(shape).copy()
        cursor += nbytes
    if cursor != len(raw):
        raise CacheIntegrityError("unreferenced prediction payload bytes")
    if header.get("status") not in VALID_STATUSES or not isinstance(header.get("metadata"), dict):
        raise CacheIntegrityError("invalid prediction status/metadata")
    # Apply the same scientific/type constraints to hostile imported records.
    kind = header.get("kind")
    if kind not in {"prediction", "sample_map", "meta_result"}:
        raise CacheIntegrityError("unsupported prediction record kind")
    status = header["status"]
    if kind != "sample_map":
        if status in {"skipped", "failed"} and (arrays or not header["metadata"].get("reason")):
            raise CacheIntegrityError("invalid skipped/failed payload")
        if any((not (name in INTEGER_PREDICTION_ARRAYS and a.dtype.kind == "i" and a.dtype.itemsize == 8)
                and (a.dtype.kind != "f" or a.dtype.itemsize != 8)) or not np.isfinite(a).all()
               for name, a in arrays.items()):
            raise CacheIntegrityError("invalid float64 prediction array")
    actual_ref = {**dict(reference), "identity": header["identity_sha256"], "status": status,
                  "content_sha256": hashlib.sha256(canonical_bytes({
                      "identity": header["identity"], "arrays": header["arrays"],
                      "raw_sha256": header["raw_sha256"], "status": status,
                      "reason": header["metadata"].get("reason")})).hexdigest()}
    parent_reference = _parent_reference(header["identity"])
    if any(key in header and header[key] != value for key, value in parent_reference.items()):
        raise CacheIntegrityError("prediction fold parent reference mismatch")
    actual_ref.update(parent_reference)
    return CacheRecord(header["identity"], arrays, header["metadata"], actual_ref, kind)


def _index_path(root: Path, relative: str) -> Path:
    shard = safe_cache_path(root, relative)
    return root / "indexes" / f"{shard.parent.name}-{shard.name}.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_signature(path: Path) -> tuple[int, int, int]:
    stat = path.stat()
    return stat.st_ino, stat.st_size, stat.st_mtime_ns


@lru_cache(maxsize=8)
def _cached_index(path: str, signature: tuple[int, int, int]) -> tuple[dict, frozenset]:
    index = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = frozenset((row.get("offset"), row.get("length"), row.get("sha256"))
                        for row in index.get("records", []))
    return index, entries


def _load_index(path: Path) -> tuple[dict, frozenset]:
    return _cached_index(str(path.resolve()), _file_signature(path))


@lru_cache(maxsize=4096)
def _cached_file_hash(path: str, signature: tuple[int, int, int]) -> str:
    return _sha256_file(Path(path))


def _sealed_file_hash(path: Path) -> str:
    """Read a sealed file once per stable filesystem identity in this process."""
    return _cached_file_hash(str(path.resolve()), _file_signature(path))


def read_record(root: Path | str, reference: Mapping[str, object], *,
                expected_identity: Mapping[str, object] | str | None = None,
                require_sealed: bool = False) -> CacheRecord:
    root = Path(root)
    original_reference = reference
    from .prediction_layout import resolve_reference
    reference = resolve_reference(root, reference)
    relative = reference.get("path")
    path = safe_cache_path(root, relative)
    offset = _integer(reference.get("offset"), "record offset", 2**63 - 1)
    length = _integer(reference.get("length"), "record length", MAX_RECORD_BYTES)
    digest = reference.get("sha256")
    if not isinstance(digest, str) or not _HEX.fullmatch(digest) or length < FRAME.size:
        raise CacheIntegrityError("invalid prediction reference")
    if require_sealed:
        index_path = _index_path(root, relative)
        if not index_path.is_file():
            raise CacheIntegrityError("consumer requires a sealed prediction shard")
        index, entries = _load_index(index_path)
        if not index.get("sealed") or index.get("path") != relative or (offset, length, digest) not in entries:
            raise CacheIntegrityError("prediction reference is absent from sealed index")
    try:
        with path.open("rb") as handle:
            handle.seek(offset)
            frame = handle.read(length)
    except OSError as exc:
        raise CacheIntegrityError(f"required cache record unavailable: {relative}") from exc
    if len(frame) != length or hashlib.sha256(frame).hexdigest() != digest:
        raise CacheIntegrityError("prediction reference length/checksum mismatch")
    record = _decode(frame, original_reference)
    expected = cache_identity(expected_identity) if isinstance(expected_identity, Mapping) else expected_identity
    if expected is not None and record.reference["identity"] != expected:
        raise CacheIntegrityError("prediction pipeline/sample/data identity mismatch")
    if reference.get("identity") is not None and reference["identity"] != record.reference["identity"]:
        raise CacheIntegrityError("reference identity mismatch")
    if reference.get("status") is not None and reference["status"] != record.status:
        raise CacheIntegrityError("reference status mismatch")
    return record


class PredictionCacheWriter:
    """Exclusive incarnation shards, bounded rotation, fsync before references.

    Call ``close`` before the plan-wide barrier. A fault makes this writer sticky
    failed; create a new incarnation after recovery instead of appending blindly.
    ``writer_id`` is hashed for filesystem safety; incarnation never grants a lease.
    """
    def __init__(self, root: Path | str, *, writer_id: str = "worker", incarnation: str | None = None,
                 shard_target_bytes: int = DEFAULT_SHARD_BYTES, shard_target_mib: float | None = None,
                 record_encoding: str | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        manifest_path = self.root / "manifest.json"
        declared = None
        if manifest_path.is_file():
            declared = json.loads(manifest_path.read_text(encoding="utf-8")).get("record_encoding", "NKPRED01")
        self.record_encoding = record_encoding or declared or RECORD_ENCODING
        if self.record_encoding not in RECORD_MAGICS or (declared is not None and declared != self.record_encoding):
            raise CacheIntegrityError("writer record encoding differs from frozen cache manifest")
        for name in ("shards", "sample-maps", "indexes", "meta-results"):
            (self.root / name).mkdir(exist_ok=True)
        incarnation = incarnation or uuid.uuid4().hex
        self.prefix = hashlib.sha256(writer_id.encode()).hexdigest()[:16] + "." + hashlib.sha256(incarnation.encode()).hexdigest()[:16]
        self.target = int(shard_target_bytes if shard_target_mib is None else shard_target_mib * 1024**2)
        if self.target <= FRAME.size:
            raise ValueError("shard target is too small")
        self._handles, self._refs, self._sequences = {}, {}, {}
        self._sample_refs = {}
        self.failed = False
        self.closed = False
        self.bytes_written = 0
        self.write_seconds = 0.0
        self.sealed_shards = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is not None:
            self.failed = True
        self.close()

    def _open(self, directory: str):
        sequence = self._sequences.get(directory, 0)
        relative = f"{directory}/{self.prefix}.{sequence:06d}.pcshard"
        path = safe_cache_path(self.root, relative)
        try:
            handle = path.open("x+b", buffering=0)
        except FileExistsError as exc:
            raise CacheBusyError("writer incarnation already exists; use a fresh incarnation") from exc
        try:
            _lock(handle)
            _sync_directory(path.parent)
        except BaseException:
            handle.close()
            raise
        self._handles[directory] = (handle, relative)
        self._refs[directory] = []
        return handle, relative

    def _seal(self, directory: str) -> None:
        entry = self._handles.get(directory)
        if entry is None:
            return
        handle, relative = entry
        handle.flush()
        os.fsync(handle.fileno())
        size = os.fstat(handle.fileno()).st_size
        # Read on the owning handle: Windows locks byte 0 against other opens.
        handle.seek(0)
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        _atomic_json(_index_path(self.root, relative), {
            "format": FORMAT, "sealed": True, "path": relative, "bytes": size,
            "sha256": digest.hexdigest(), "record_count": len(self._refs[directory]),
            "records": self._refs[directory]})
        _unlock(handle)
        handle.close()
        del self._handles[directory]
        self._sequences[directory] = self._sequences.get(directory, 0) + 1
        self.sealed_shards.append(relative)

    def append(self, identity: Mapping[str, object], arrays: Mapping[str, np.ndarray],
               metadata: Mapping[str, object] | None = None, *, kind: str = "prediction",
               remaining_bytes: int | None = None) -> dict[str, object]:
        if self.failed or self.closed:
            raise CacheStorageError("prediction writer is failed/closed; cannot acknowledge results")
        if kind not in {"prediction", "sample_map", "meta_result"}:
            raise CacheIntegrityError("unsupported prediction record kind")
        started = time.perf_counter()
        frame, header = _encode(identity, arrays, metadata or {}, kind=kind, record_encoding=self.record_encoding)
        if remaining_bytes is not None and len(frame) > remaining_bytes:
            self.failed = True
            raise CacheStorageError("prediction append exceeds its remaining storage reservation")
        directory = {"prediction": "shards", "sample_map": "sample-maps", "meta_result": "meta-results"}[kind]
        try:
            if directory in self._handles:
                handle, _ = self._handles[directory]
                if handle.tell() and (handle.tell() + len(frame) > self.target
                                      or len(self._refs[directory]) >= MAX_RECORDS_PER_SHARD):
                    self._seal(directory)
            handle, relative = self._handles.get(directory) or self._open(directory)
            offset = handle.tell()
            _write_all(handle, frame)
            os.fsync(handle.fileno())
            reference = {"path": relative, "offset": offset, "length": len(frame),
                         "sha256": hashlib.sha256(frame).hexdigest(),
                         "identity": header["identity_sha256"], "status": header["status"]}
            for key in ("parent_identity", "fold_identity_version"):
                if key in header:
                    reference[key] = header[key]
            self._refs[directory].append(reference)
            self.bytes_written += len(frame)
            return dict(reference)
        except OSError as exc:
            self.failed = True
            raise CacheStorageError(exc.errno, f"prediction persistence failed: {exc}") from exc
        finally:
            self.write_seconds += time.perf_counter() - started

    def append_sample_map(self, metadata: Mapping[str, object], arrays: Mapping[str, np.ndarray] | None = None) -> dict[str, object]:
        """Content-addressed shared maps; repeated maps reuse one record per writer.

        Map metadata contains exact ordered IDs/features; arrays may contain labels
        and fold IDs. Cross-writer duplicates are safe and can be compacted later.
        The root identity embeds content hashes, never an inferred seed ordering.
        """
        arrays = arrays or {}
        contents = {"metadata": dict(metadata), "arrays": {
            key: {"dtype": np.asarray(value).dtype.str, "shape": list(np.asarray(value).shape),
                  "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}
            for key, value in arrays.items()}}
        identity = {"sample_map_content": cache_identity(contents)}
        key = cache_identity(identity)
        if key not in self._sample_refs:
            self._sample_refs[key] = self.append(identity, arrays, metadata, kind="sample_map")
        return dict(self._sample_refs[key])

    def append_frame(self, frame, reference, *, directory):
        """Copy a sealed frame byte-for-byte; no re-encoding of scientific data."""
        if self.closed or self.failed or directory not in ('shards', 'meta-results'):
            raise CacheStorageError('Raw append requires an open prediction writer')
        if len(frame) != reference['length'] or hashlib.sha256(frame).hexdigest() != reference['sha256']:
            raise CacheIntegrityError('Compaction frame checksum differs from sealed reference')
        if directory in self._handles:
            handle, _ = self._handles[directory]
            if handle.tell() and (handle.tell() + len(frame) > self.target or len(self._refs[directory]) >= MAX_RECORDS_PER_SHARD):
                self._seal(directory)
        handle, relative = self._handles.get(directory) or self._open(directory)
        updated = {**reference, 'path': relative, 'offset': handle.tell()}
        _write_all(handle, frame); os.fsync(handle.fileno())
        self._refs[directory].append(updated); self.bytes_written += len(frame)
        return updated

    def close(self) -> None:
        if self.closed:
            return
        try:
            if not self.failed:
                for directory in list(self._handles):
                    self._seal(directory)
        except OSError as exc:
            self.failed = True
            raise CacheStorageError(exc.errno, f"prediction shard sealing failed: {exc}") from exc
        finally:
            for handle, _ in self._handles.values():
                _unlock(handle)
                handle.close()
            self._handles.clear()
            self.closed = True


@dataclass(frozen=True)
class ShardScan:
    references: tuple[Mapping[str, object], ...]
    valid_bytes: int
    file_bytes: int
    tail_error: str | None


def scan_shard(root: Path | str, relative: str) -> ShardScan:
    """Read-only targeted scan; corruption is explicit and never silently skipped."""
    path = safe_cache_path(root, relative)
    with path.open("rb") as handle:
        return _scan_handle(handle, relative)


def _scan_handle(handle, relative: str) -> ShardScan:
    handle.seek(0)
    size = os.fstat(handle.fileno()).st_size
    refs, cursor, error = [], 0, None
    while cursor < size:
        prefix = handle.read(FRAME.size)
        if len(prefix) != FRAME.size:
            error = "incomplete frame prefix"
            break
        magic, header_size, payload_size, _ = FRAME.unpack(prefix)
        length = FRAME.size + header_size + payload_size
        if (magic not in RECORD_MAGICS.values() or header_size > MAX_STORED_HEADER_BYTES
                or (magic == LEGACY_MAGIC and header_size > MAX_HEADER_BYTES) or length > MAX_RECORD_BYTES):
            error = "invalid frame prefix"
            break
        body = handle.read(length - FRAME.size)
        if len(body) != length - FRAME.size:
            error = "incomplete frame body"
            break
        frame = prefix + body
        ref = {"path": relative, "offset": cursor, "length": length,
               "sha256": hashlib.sha256(frame).hexdigest()}
        try:
            record = _decode(frame, ref)
        except CacheIntegrityError as exc:
            error = str(exc)
            break
        refs.append(dict(record.reference))
        cursor += length
    return ShardScan(tuple(refs), cursor, size, error)


def rebuild_index(root: Path | str, relative: str, *, writer_revoked: bool = False,
                  repair_incomplete_tail: bool = False) -> ShardScan:
    """Seal an abandoned shard under an exclusive OS lock.

    The caller must prove the writer lease is revoked. A partial final write can
    be truncated; a complete record with a bad checksum is retained as evidence
    and requires explicit repair/retraining of affected authoritative tasks.
    """
    if not writer_revoked:
        raise CacheBusyError("tail recovery requires explicit proof of revoked writer authority")
    root = Path(root)
    path = safe_cache_path(root, relative)
    with path.open("r+b", buffering=0) as handle:
        _lock(handle)
        try:
            scan = _scan_handle(handle, relative)
            if scan.tail_error:
                if _index_path(root, relative).exists():
                    raise CacheIntegrityError("sealed shard corruption cannot be truncated")
                if not repair_incomplete_tail or not scan.tail_error.startswith("incomplete frame"):
                    raise CacheIntegrityError(f"prediction shard corruption: {scan.tail_error}")
                handle.truncate(scan.valid_bytes)
                os.fsync(handle.fileno())
            handle.seek(0)
            digest = hashlib.sha256()
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
            _atomic_json(_index_path(root, relative), {
                "format": FORMAT, "sealed": True, "path": relative,
                "bytes": scan.valid_bytes, "sha256": digest.hexdigest(),
                "record_count": len(scan.references), "records": list(scan.references),
                "recovered": True, "repaired_tail_bytes": scan.file_bytes - scan.valid_bytes})
            return scan
        finally:
            _unlock(handle)


def verify_coverage(root: Path | str, expected_identities: Iterable[Mapping[str, object] | str],
                    references: Iterable[Mapping[str, object]], *, require_oof: bool = True,
                    require_sealed: bool = True, output_name: str | None = "verified.json",
                    allowed_skip_reasons: Iterable[str] | None = None) -> dict[str, object]:
    """Verify authoritative references, exact coverage, skips and conflicts.

    Pass references from accepted queue results, not arbitrary orphan shards.
    This deliberately does not infer score completeness or lease quiescence; the
    scheduler must check those before its plan-wide barrier may publish success.
    """
    root = Path(root)
    expected = {cache_identity(x) if isinstance(x, Mapping) else x for x in expected_identities}
    seen, conflicts, failures, shard_hashes, sample_maps = {}, [], [], {}, {}
    skips, oof_skips = 0, 0
    legal_reasons = set(allowed_skip_reasons) if allowed_skip_reasons is not None else None
    for reference in references:
        record = read_record(root, reference, require_sealed=require_sealed)
        key = record.reference["identity"]
        content = record.reference["content_sha256"]
        if key in seen and seen[key] != content:
            conflicts.append(key)
            continue
        duplicate = key in seen
        seen[key] = content
        if record.status == "failed":
            failures.append(key)
        elif record.status == "skipped":
            reason = record.metadata.get("reason")
            if legal_reasons is not None and reason not in legal_reasons:
                failures.append(key)
            if not duplicate:
                skips += 1
        else:
            if "holdout_prediction" not in record.arrays:
                failures.append(key)
            oof_skip = record.metadata.get("oof_status") == "skipped" and bool(record.metadata.get("oof_reason"))
            if oof_skip and legal_reasons is not None and record.metadata["oof_reason"] not in legal_reasons:
                failures.append(key)
            if require_oof and "oof_prediction" not in record.arrays:
                if oof_skip and not duplicate:
                    oof_skips += 1
                elif not oof_skip:
                    failures.append(key)
            n_samples = record.identity.get("N", record.identity.get("n_samples"))
            if require_oof and "oof_prediction" in record.arrays and n_samples is not None and (
                    record.arrays["oof_prediction"].ndim < 1 or record.arrays["oof_prediction"].shape[0] != int(n_samples)):
                failures.append(key)
        map_refs = record.metadata.get("sample_map_refs", [])
        if isinstance(map_refs, Mapping):
            map_refs = list(map_refs.values())
        for map_ref in map_refs:
            map_key = str(map_ref.get("sha256"))
            if map_key not in sample_maps:
                mapping = read_record(root, map_ref, require_sealed=require_sealed)
                if mapping.kind != "sample_map":
                    raise CacheIntegrityError("sample map reference points to predictions")
                sample_maps[map_key] = mapping.reference["identity"]
        if require_sealed:
            for ref in [reference, *map_refs]:
                from .prediction_layout import resolve_reference
                relative = str(resolve_reference(root, ref)["path"])
                if relative not in shard_hashes:
                    index, _ = _load_index(_index_path(root, relative))
                    path = safe_cache_path(root, relative)
                    if path.stat().st_size != index["bytes"] or _sealed_file_hash(path) != index["sha256"]:
                        raise CacheIntegrityError("sealed prediction shard checksum mismatch")
                    shard_hashes[relative] = index["sha256"]
    missing, unexpected = sorted(expected - seen.keys()), sorted(seen.keys() - expected)
    complete = not (missing or unexpected or conflicts or failures)
    report = {"format": FORMAT, "prediction_cache_complete": complete,
              "oof_complete": complete if require_oof else None,
              "expected_count": len(expected), "verified_count": len(seen), "skipped_count": skips,
              "oof_skipped_count": oof_skips,
              "missing": missing, "unexpected": unexpected, "conflicts": sorted(set(conflicts)),
              "failed": sorted(set(failures)), "shards": shard_hashes,
              "sample_maps": sample_maps, "index_generation": cache_identity({"records": seen, "shards": shard_hashes}),
              "cache_bytes": sum(safe_cache_path(root, p).stat().st_size for p in shard_hashes)}
    if not complete:
        if output_name:
            _atomic_json(safe_cache_path(root, output_name + ".incomplete"), report)
        raise CacheIntegrityError(f"cache coverage incomplete: missing={len(missing)} unexpected={len(unexpected)} conflicts={len(set(conflicts))} failed={len(set(failures))}")
    if output_name:
        _atomic_json(safe_cache_path(root, output_name), report)
    return report


def estimate_prediction_bytes(*, n_grid: Sequence[int], k_count: int, holdout_samples: int,
                               repeats: int, base_models: int = 8, sl_variants: int = 1) -> dict[str, int]:
    values = [*n_grid, k_count, holdout_samples, repeats, base_models, sl_variants]
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values):
        raise ValueError("capacity parameters must be nonnegative integers")
    base_holdout = repeats * len(n_grid) * k_count * base_models * holdout_samples * 8
    base_oof = repeats * k_count * base_models * sum(n_grid) * 8
    sl_holdout = repeats * len(n_grid) * k_count * sl_variants * holdout_samples * 8
    return {"base_holdout_bytes": base_holdout, "base_oof_bytes": base_oof,
            "sl_holdout_bytes": sl_holdout, "raw_prediction_bytes": base_holdout + base_oof + sl_holdout}


def check_storage_admission(*, used_bytes: int, soft_quota_bytes: int, pending_reserved_bytes: int,
                            temporary_bytes: int, reserve_bytes: int, used_files: int,
                            soft_quota_files: int, pending_files: int, temporary_files: int,
                            reserve_files: int = 1_000_000, filesystem_free_bytes: int | None = None,
                            quota_observed_at: float, max_quota_age_seconds: float = 300,
                            now: float | None = None) -> dict[str, object]:
    """Conservative live soft-quota admission; pending includes all approved runs.

    Callers must refresh site quota and atomically reserve their allocation in a
    shared reservation ledger. This pure calculation is not itself a reservation.
    Compression savings and the hard quota are intentionally excluded.
    """
    now = time.time() if now is None else now
    age = now - quota_observed_at
    if age < -5 or age > max_quota_age_seconds:
        raise CacheStorageError("live quota evidence is missing/stale")
    numbers = [used_bytes, soft_quota_bytes, pending_reserved_bytes, temporary_bytes, reserve_bytes,
               used_files, soft_quota_files, pending_files, temporary_files, reserve_files]
    if any(v < 0 for v in numbers):
        raise ValueError("quota/reservation inputs cannot be negative")
    projected_bytes = used_bytes + pending_reserved_bytes + temporary_bytes + reserve_bytes
    projected_files = used_files + pending_files + temporary_files + reserve_files
    filesystem_ok = filesystem_free_bytes is not None and filesystem_free_bytes >= pending_reserved_bytes + temporary_bytes + reserve_bytes
    admitted = projected_bytes <= soft_quota_bytes and projected_files <= soft_quota_files and filesystem_ok
    report = {"admitted": admitted, "projected_bytes": projected_bytes,
              "projected_files": projected_files, "quota_age_seconds": age,
              "filesystem_free_verified": filesystem_ok}
    if not admitted:
        raise CacheStorageError(f"prediction storage admission rejected: {report}")
    return report


def verify_reference(root: Path | str, reference: Mapping[str, object], *,
                     expected_identity: Mapping[str, object] | str | None = None,
                     require_sealed: bool = False) -> dict[str, object]:
    """Dispatcher adapter. Only use arrays locally, never return them via RPC."""
    record = read_record(root, reference, expected_identity=expected_identity, require_sealed=require_sealed)
    return {"metadata": record.metadata, "arrays": record.arrays, "identity": record.identity,
            "reference": record.reference, "status": record.status, "kind": record.kind}


def seal_stopped_writers(root: Path | str, *, writer_revoked: bool = False,
                         writer_id: str | None = None,
                         references: Iterable[Mapping[str, object]] | None = None) -> list[str]:
    """Target abandoned shards after the controller proves allocations stopped.

    Ordinary restart looks at index names only. Unindexed shards alone require
    a bounded scan; active writer OS locks independently prevent truncation.
    """
    if not writer_revoked:
        raise CacheBusyError("controller must confirm stopped/revoked writer allocations")
    root = Path(root)
    prefix = hashlib.sha256(writer_id.encode()).hexdigest()[:16] + "." if writer_id is not None else ""
    if references is not None:
        paths = set()
        for reference in references:
            from .prediction_layout import resolve_reference
            paths.add(str(resolve_reference(root, reference)["path"]))
            record = read_record(root, reference)
            maps = record.metadata.get("sample_map_refs", [])
            if isinstance(maps, Mapping):
                maps = maps.values()
            for mapping in maps:
                paths.add(str(resolve_reference(root, mapping)["path"]))
    else:
        paths = {path.relative_to(root).as_posix() for directory in ("shards", "sample-maps", "meta-results")
                 for path in (root / directory).glob(prefix + "*.pcshard")}
    recovered = []
    for relative in sorted(paths):
        if not _index_path(root, relative).exists():
            rebuild_index(root, relative, writer_revoked=True, repair_incomplete_tail=True)
            recovered.append(relative)
    return recovered


def verified_index_evidence(root: Path | str, references: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    """Return immutable evidence for exactly these refs, unaffected by later SL."""
    root = Path(root)
    paths = set()
    from .prediction_layout import resolve_reference
    for reference in references:
        record = read_record(root, reference, require_sealed=True)
        paths.add(str(resolve_reference(root, reference)["path"]))
        maps = record.metadata.get("sample_map_refs", [])
        if isinstance(maps, Mapping):
            maps = maps.values()
        for mapping in maps:
            read_record(root, mapping, require_sealed=True)
            paths.add(str(resolve_reference(root, mapping)["path"]))
    evidence = []
    for relative in sorted(paths):
        path = safe_cache_path(root, relative)
        index_path = _index_path(root, relative)
        index, _ = _load_index(index_path)
        digest = _sealed_file_hash(path)
        if digest != index.get("sha256") or path.stat().st_size != index.get("bytes"):
            raise CacheIntegrityError("sealed shard differs from its batch index")
        evidence.extend([{"path": relative, "sha256": digest},
                         {"path": index_path.relative_to(root).as_posix(), "sha256": _sealed_file_hash(index_path)}])
    return evidence


def find_record(root: Path | str, identity: Mapping[str, object] | str, *,
                writer_id: str | None = None,
                authoritative_references: Iterable[Mapping[str, object]] | None = None) -> CacheRecord | None:
    """Find an exact cached training identity using indexes, never model names.

    Use accepted references when recovering a queue. Without those references the
    caller must explicitly trust this run/source cache; conflicting predictions
    cause refusal rather than arbitrary last-writer-wins selection.
    """
    root = Path(root)
    key = cache_identity(identity) if isinstance(identity, Mapping) else identity
    if not isinstance(key, str) or not _HEX.fullmatch(key):
        raise CacheIntegrityError("lookup requires a full identity SHA256")
    if authoritative_references is None:
        prefix = hashlib.sha256(writer_id.encode()).hexdigest()[:16] + "." if writer_id is not None else ""
        references = []
        for directory in ("shards", "meta-results"):
            for index_path in sorted((root / "indexes").glob(directory + "-" + prefix + "*.pcshard.json")):
                index, _ = _load_index(index_path)
                if index.get("sealed"):
                    references.extend(ref for ref in index.get("records", []) if ref.get("identity") == key)
    else:
        references = [ref for ref in authoritative_references if ref.get("identity") == key]
    found = None
    for reference in references:
        record = read_record(root, reference, expected_identity=key, require_sealed=True)
        if found is not None and found.reference["content_sha256"] != record.reference["content_sha256"]:
            raise CacheIntegrityError("identical cache identity has conflicting predictions")
        found = record
    return found


class WriterRecoveryIndex:
    """One startup index pass for a stable worker, then O(1) identity lookup.

    Only small references are retained. Explicit bounds stop the slot before it
    trains work whose recovery metadata cannot be held; callers can start another
    bounded worker slot. Arrays and entire shard data are never retained here.
    """
    def __init__(self, root: Path | str, writer_id: str, *, max_records: int = 65536,
                 max_reference_bytes: int = 64 * 1024 * 1024):
        self.root = Path(root)
        self.max_records = int(max_records)
        self.max_reference_bytes = int(max_reference_bytes)
        self.records = {}
        self.record_count = 0
        self.reference_bytes = 0
        prefix = hashlib.sha256(writer_id.encode()).hexdigest()[:16] + "."
        for directory in ("shards", "meta-results"):
            for path in sorted((self.root / "indexes").glob(directory + "-" + prefix + "*.pcshard.json")):
                index, _ = _load_index(path)
                if index.get("sealed"):
                    for reference in index.get("records", []):
                        self.add(reference, sealed=True)

    def add(self, reference: Mapping[str, object], *, sealed: bool = False) -> None:
        key = reference.get("identity")
        if not isinstance(key, str) or not _HEX.fullmatch(key):
            raise CacheIntegrityError("recovery index requires an identity SHA256")
        reference = dict(reference)
        entries = self.records.setdefault(key, [])
        if any(old[0] == reference for old in entries):
            return
        size = len(canonical_bytes(reference)) + 1024  # Conservative Python object overhead.
        if self.record_count + 1 > self.max_records or self.reference_bytes + size > self.max_reference_bytes:
            raise CacheStorageError("worker recovery index capacity reached; stop and use another bounded slot")
        entries.append((reference, sealed))
        self.record_count += 1
        self.reference_bytes += size

    def find(self, identity: Mapping[str, object] | str) -> CacheRecord | None:
        key = cache_identity(identity) if isinstance(identity, Mapping) else identity
        found = None
        for reference, sealed in self.records.get(key, []):
            record = read_record(self.root, reference, expected_identity=key, require_sealed=sealed)
            if found is not None and found.reference["content_sha256"] != record.reference["content_sha256"]:
                raise CacheIntegrityError("worker recovery identity has conflicting prediction content")
            found = record
        return found

    def remove_path(self, relative: str) -> None:
        """Forget retired private checkpoints, never accepted prediction refs."""
        for key, entries in list(self.records.items()):
            kept = []
            for reference, sealed in entries:
                if reference["path"] == relative:
                    self.record_count -= 1
                    self.reference_bytes -= len(canonical_bytes(reference)) + 1024
                else:
                    kept.append((reference, sealed))
            if kept:
                self.records[key] = kept
            else:
                del self.records[key]


def retire_private_fold_shards(checkpoint_root: Path | str, relatives: Sequence[str], *,
                               completed_parent_identities: Iterable[str],
                               worker_exclusive: bool = False) -> list[str]:
    """Reclaim sealed private fold checkpoints after durable final predictions.

    This API cannot operate on a manifest-bearing accepted prediction cache.
    The logical worker slot must be exclusively held and only that worker's
    referenced shards may be supplied. Active tails and unfinished parents stay.
    """
    if not worker_exclusive:
        raise CacheBusyError("private fold cleanup requires the exclusive logical worker slot")
    root = Path(checkpoint_root)
    if (root / "manifest.json").exists() or root.name != "prediction-training-checkpoints":
        raise CacheIntegrityError("fold cleanup is forbidden for accepted prediction caches")
    completed = set(completed_parent_identities)
    retired = []
    for relative in sorted(set(relatives)):
        path = safe_cache_path(root, relative)
        if path.parent != (root / "shards").resolve() or path.suffix != ".pcshard":
            raise CacheIntegrityError("private fold cleanup path is not a prediction shard")
        index_path = _index_path(root, relative)
        if not index_path.exists():
            continue  # Never rewrite or truncate an active tail.
        index, _ = _load_index(index_path)
        parents = {reference.get("parent_identity") for reference in index.get("records", [])}
        if not parents or None in parents or not parents <= completed:
            continue
        with path.open("r+b", buffering=0) as handle:
            _lock(handle)
            try:
                if os.fstat(handle.fileno()).st_size != index.get("bytes"):
                    raise CacheIntegrityError("private fold shard changed after sealing")
            finally:
                _unlock(handle)
        # No reader can remain: this is a private, fenced logical worker slot.
        path.unlink()
        index_path.unlink()
        _sync_directory(path.parent)
        _sync_directory(index_path.parent)
        retired.append(relative)
    return retired


def compact_shards(root: Path | str, relatives: Sequence[str], *, temporary_byte_limit: int,
                    writer_revoked: bool = False, target_bytes: int = DEFAULT_SHARD_BYTES) -> dict[str, object]:
    """Bounded copy/verify/publish; old files stay valid until readers drain.

    Publication returns an immutable remapping generation. The controller must
    rebase authoritative refs to it before it may call ``retire_compaction``.
    An interrupted copy leaves harmless unreferenced new shards; no old data or
    index is modified, and a generation is published only after all copies verify.
    """
    if not writer_revoked:
        raise CacheBusyError("compaction requires stopped source writers")
    root = Path(root)
    unique = sorted(set(relatives))
    source_refs, source_bytes = [], 0
    for relative in unique:
        index_path = _index_path(root, relative)
        if not index_path.is_file():
            raise CacheIntegrityError("compaction source is unsealed")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        source_bytes += safe_cache_path(root, relative).stat().st_size
        if source_bytes * 2 > temporary_byte_limit:
            # Includes bounded metadata and transient encoder buffers conservatively.
            raise CacheStorageError("bounded compaction temporary budget exceeded")
        source_refs.extend(index["records"])
    verified_index_evidence(root, source_refs)
    generation = uuid.uuid4().hex
    remapping = []
    with PredictionCacheWriter(root, writer_id="compaction", incarnation=generation,
                               shard_target_bytes=target_bytes) as writer:
        for old_ref in source_refs:
            record = read_record(root, old_ref, require_sealed=True)
            new_ref = writer.append(record.identity, record.arrays, record.metadata, kind=record.kind)
            remapping.append({"old": old_ref, "new": new_ref})
    for replacement in remapping:
        old = read_record(root, replacement["old"], require_sealed=True)
        new = read_record(root, replacement["new"], require_sealed=True)
        if old.reference["content_sha256"] != new.reference["content_sha256"]:
            raise CacheIntegrityError("compaction changed numerical content")
    report = {"format": FORMAT, "generation": generation, "sources": unique,
              "source_bytes": source_bytes, "temporary_byte_limit": temporary_byte_limit,
              "replacements": remapping, "source_retirement_pending": True}
    _atomic_json(root / "indexes" / f"compaction-{generation}.json", report)
    return report


def retire_compaction(root: Path | str, generation: str, *, readers_drained: bool = False,
                       authoritative_refs_rebased: bool = False) -> list[str]:
    """Delete only superseded source shards after two explicit controller proofs."""
    if not readers_drained or not authoritative_refs_rebased:
        raise CacheBusyError("old readers must exit and authoritative refs must be rebased before retirement")
    if not re.fullmatch(r"[0-9a-f]{32}", generation):
        raise CacheIntegrityError("invalid compaction generation")
    root = Path(root)
    report_path = root / "indexes" / f"compaction-{generation}.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if any(PurePosixPath(relative).parts[0] == "sample-maps" for relative in report["sources"]):
        raise CacheIntegrityError("sample-map retirement requires rewriting dependent immutable record references")
    verified_index_evidence(root, [item["new"] for item in report["replacements"]])
    retired = []
    for relative in report["sources"]:
        path = safe_cache_path(root, relative)
        if path.parent.name not in {"shards", "sample-maps", "meta-results"} or path.suffix != ".pcshard":
            raise CacheIntegrityError("compaction retirement path is not a cache shard")
        path.unlink(missing_ok=True)
        _index_path(root, relative).unlink(missing_ok=True)
        retired.append(relative)
    _atomic_json(report_path, {**report, "source_retirement_pending": False})
    return retired
