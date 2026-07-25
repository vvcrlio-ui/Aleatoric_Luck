"""Approved FFC encoding strategies."""

from .median_missing_indicator import encode_median_missing_indicator
from .median_mode import encode_median_mode
from .tree_ordinal import encode_tree_ordinal

STRATEGIES = {
    "median_mode": encode_median_mode,
    "median_missing_indicator": encode_median_missing_indicator,
    "tree_ordinal": encode_tree_ordinal,
}

__all__ = ["STRATEGIES"]
