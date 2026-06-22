"""Official and fallback split construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .io import DataError


def official_holdout_ids(
    truth: pd.DataFrame,
    id_col: str,
    target: str,
) -> set:
    for col in (id_col, target):
        if col not in truth.columns:
            raise DataError(f"Truth file is missing required column: {col}")
    return set(truth.loc[truth[target].notna(), id_col].astype(str))


def random_stratified_holdout_ids(
    outcomes: pd.DataFrame,
    id_col: str,
    target: str,
    fraction: float,
    seed: int,
) -> set:
    usable = outcomes.loc[outcomes[target].notna(), [id_col, target]].copy()
    if usable.empty:
        raise DataError("No labeled rows available for random fallback split.")
    bins = pd.qcut(usable[target].rank(method="first"), q=min(5, len(usable)), duplicates="drop")
    _, holdout = train_test_split(
        usable,
        test_size=fraction,
        random_state=seed,
        stratify=bins if bins.nunique() > 1 else None,
    )
    return set(holdout[id_col].astype(str))
