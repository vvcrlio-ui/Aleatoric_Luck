"""Shared strategy helpers."""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from ..common.schema import SourceSpec, numeric_values


def test_mask(frame: pd.DataFrame, id_column: str, test_ids: Iterable[Any]) -> pd.Series:
    return frame[id_column].isin(set(test_ids))


def categorical_state(
    frame: pd.DataFrame, source: SourceSpec
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    numeric, _raw, blank, coded = numeric_values(frame[source.source_column])
    known = numeric.isin(source.levels)
    unknown = numeric.notna() & ~known
    structural_missing = blank | coded
    return numeric, known, unknown, structural_missing


def unknown_counts(
    frame: pd.DataFrame,
    source: SourceSpec,
    *,
    id_column: str,
    test_ids: Iterable[Any],
) -> tuple[int, int]:
    numeric, _known, unknown, _structural = categorical_state(frame, source)
    mask = test_mask(frame, id_column, test_ids)
    denominator = int((mask & numeric.notna()).sum())
    return int((mask & unknown).sum()), denominator
