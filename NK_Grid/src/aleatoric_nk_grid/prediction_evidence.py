"""Validate sealed cache evidence without treating filesystem times as identity."""
from .prediction_cache import safe_cache_path
from .shared_queue import QueueError, file_digest


def verify_cache_evidence(root, evidence):
    """Only a changed timestamp requires rereading an immutable data shard.

    Size changes remain fatal. A timestamp disagreement requires the original
    sealed-content checksum, never an updated receipt or relaxed hash.
    Individual prediction records are still verified when consumed.
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
