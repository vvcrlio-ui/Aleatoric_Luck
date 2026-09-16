"""Keep primary numerical results unchanged; recover finite-input failures."""
from __future__ import annotations
import logging
import numpy as np
from scipy import linalg


def safe_lstsq(a, b, **kwargs):
    """gelsd -> gelss -> pivoted QR; preserve the caller's rank threshold."""
    try:
        return linalg.lstsq(a, b, **kwargs)
    except np.linalg.LinAlgError:
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            raise
        if kwargs.get("overwrite_a") or kwargs.get("overwrite_b"):
            raise  # The original input may have been destroyed by LAPACK.
        primary = kwargs.get("lapack_driver") or linalg.lstsq.default_lapack_driver
        for driver in ("gelss", "gelsy"):
            if driver == primary:
                continue
            try:
                result = linalg.lstsq(a, b, **dict(kwargs, lapack_driver=driver))
                if not np.isfinite(result[0]).all():
                    raise np.linalg.LinAlgError("Nonfinite least-squares coefficients")
                logging.getLogger(__name__).warning(
                    "Least-squares fallback used: driver=%s shape=%s", driver, a.shape)
                return result
            except np.linalg.LinAlgError:
                if driver == "gelsy":
                    raise
        raise


def safe_svd(matrix, *, primary=np.linalg.svd, **kwargs):
    try:
        return primary(matrix, **kwargs)
    except np.linalg.LinAlgError:
        return _gesvd(matrix, **kwargs)


def _gesvd(matrix, **kwargs):
    if not np.isfinite(matrix).all():
        raise np.linalg.LinAlgError("Cannot recover SVD with nonfinite input")
    factors = linalg.svd(matrix, lapack_driver="gesvd", overwrite_a=False,
                         check_finite=True, **kwargs)
    if not all(np.isfinite(part).all() for part in factors):
        raise np.linalg.LinAlgError("gesvd fallback returned nonfinite factors")
    logging.getLogger(__name__).warning("SVD fallback used: driver=gesvd shape=%s", matrix.shape)
    return factors


def ridge_svd(matrix):
    """Return ((u, s, vt), fallback_used), without altering normal-path results."""
    try:
        return np.linalg.svd(matrix, full_matrices=False), False
    except np.linalg.LinAlgError:
        if not np.isfinite(matrix).all():
            raise
        return _gesvd(matrix, full_matrices=False), True
