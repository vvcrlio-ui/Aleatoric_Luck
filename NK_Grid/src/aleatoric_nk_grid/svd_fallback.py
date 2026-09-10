"""Use the original Ridge SVD first; recover finite-matrix nonconvergence only."""
from __future__ import annotations
import logging
import numpy as np
from scipy import linalg


def ridge_svd(matrix):
    """Return ((u, s, vt), fallback_used), without altering normal-path results."""
    try:
        return np.linalg.svd(matrix, full_matrices=False), False
    except np.linalg.LinAlgError:
        if not np.isfinite(matrix).all():
            raise
        factors = linalg.svd(matrix, full_matrices=False, lapack_driver="gesvd",
                             overwrite_a=False, check_finite=True)
        if not all(np.isfinite(part).all() for part in factors):
            raise np.linalg.LinAlgError("gesvd fallback returned nonfinite factors")
        logging.getLogger(__name__).warning("Ridge SVD fallback used: driver=gesvd shape=%s", matrix.shape)
        return factors, True
