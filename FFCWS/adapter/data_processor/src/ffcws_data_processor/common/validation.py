"""Input and QA validation helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd


def ensure_unique_ids(frame: pd.DataFrame, id_column: str, label: str) -> None:
    if id_column not in frame:
        raise KeyError(f"{label} is missing ID column: {id_column}")
    duplicated = frame[id_column].duplicated()
    if bool(duplicated.any()):
        example = frame.loc[duplicated, id_column].iloc[0]
        raise ValueError(f"{label} has duplicate {id_column}: {example}")


def unknown_qa_row(
    *, source_column: str, unknown_count: int, denominator: int, threshold: float
) -> dict[str, Any]:
    rate = unknown_count / denominator if denominator else 0.0
    if rate > threshold:
        raise ValueError(
            "Unknown category rate exceeds threshold: "
            f"source={source_column} unknown_count={unknown_count} "
            f"denominator={denominator} rate={rate:.6f} threshold={threshold:.6f}"
        )
    return {
        "source_column": source_column,
        "unknown_count": int(unknown_count),
        "denominator": int(denominator),
        "unknown_rate": float(rate),
        "threshold": float(threshold),
    }
