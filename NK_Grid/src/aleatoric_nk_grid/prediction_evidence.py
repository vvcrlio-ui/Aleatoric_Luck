"""Validate sealed cache evidence without treating filesystem times as identity."""
import time
from pathlib import Path
from uuid import uuid4

from .prediction_cache import safe_cache_path
from .shared_queue import QueueError, file_digest


def seal_mtime_barrier(root, mtime_ns, *, timeout_seconds=30.):
    """Make a sealed record's timestamp unusable by any later write.

    Lustre stamps mtime with one-second granularity, so a same-size rewrite
    inside the sealing tick leaves ``st_mtime_ns`` unchanged and would skip the
    content check below. A probe on the same filesystem carries the same clock
    that stamps the records, so waiting until it is strictly newer makes every
    later modification observable without hashing terabytes on each controller
    tick. Sealing happens once per barrier; the wait is bounded by one tick.
    """
    root = Path(root)
    probe = root / ('.mtime-barrier-' + uuid4().hex)
    deadline = time.monotonic() + timeout_seconds
    try:
        while True:
            probe.write_bytes(b'')
            observed = probe.stat().st_mtime_ns
            if observed > mtime_ns:
                return observed
            if time.monotonic() >= deadline:
                raise QueueError('Cache filesystem timestamps did not advance past the sealed '
                                 'records within %.0fs; evidence cannot rely on mtime' % timeout_seconds)
            time.sleep(.05)
    finally:
        probe.unlink(missing_ok=True)


def verify_cache_evidence(root, evidence):
    """Only a changed timestamp requires rereading an immutable data shard.

    Size changes remain fatal. A timestamp disagreement requires the original
    sealed-content checksum, never an updated receipt or relaxed hash.
    Individual prediction records are still verified when consumed.

    The timestamp fast path is sound only because sealing calls
    ``seal_mtime_barrier`` first: without it a same-size rewrite inside the
    sealing tick would keep ``st_mtime_ns`` and pass unread.
    """
    path = safe_cache_path(root, evidence['path'])
    stat = path.stat()
    index = evidence['path'].startswith('indexes/')
    if not index and stat.st_size != evidence.get('bytes'):
        raise QueueError('Verified cache size changed: ' + evidence['path'])
    if index or stat.st_mtime_ns != evidence.get('mtime_ns'):
        if file_digest(path) != evidence.get('sha256'):
            raise QueueError('Verified cache content changed: ' + evidence['path'])
        if path.stat().st_size != stat.st_size:
            raise QueueError('Verified cache changed while reading: ' + evidence['path'])
        return 'index_checked' if index else 'timestamp_content_checked'
    return 'unchanged'
