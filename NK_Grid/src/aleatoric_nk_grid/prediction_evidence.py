"""Container presence checks; integrity belongs to the records being consumed."""
from .prediction_cache import safe_cache_path
from .shared_queue import QueueError


def verify_cache_evidence(root, evidence):
    """Check presence/size only; content integrity is checked per used record.

    Old receipts with whole-file digests are accepted without rereading the
    entire file. This is not a claim that its unused records were checked.
    """
    path = safe_cache_path(root, evidence['path'])
    stat = path.stat()
    if 'bytes' in evidence and stat.st_size != evidence['bytes']:
        raise QueueError('Verified cache size changed: ' + evidence['path'])
    return 'record_checked_on_read'
