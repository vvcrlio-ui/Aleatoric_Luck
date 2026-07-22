"""One-hot encoding that leaves categorical missing groups as NaN."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..common.manifests import EncodedResult, feature_row, manifest_frame
from ..common.schema import SharedSchema, numeric_values
from ..common.validation import unknown_qa_row
from ._shared import categorical_state, unknown_counts


STRATEGY = "median_mode"


def encode_median_mode(
    frame: pd.DataFrame,
    schema: SharedSchema,
    *,
    test_ids: Iterable[Any] = (),
    unknown_rate_threshold: float = 0.95,
) -> EncodedResult:
    columns: dict[str, pd.Series] = {schema.id_column: frame[schema.id_column]}
    manifest_rows: list[dict[str, Any]] = []
    unknown_rows: list[dict[str, Any]] = []

    for source in schema.eligible_sources:
        numeric, _raw, _blank, _coded = numeric_values(frame[source.source_column])
        if source.status == "numeric":
            feature = source.observed_features[0]
            columns[feature.feature_name] = numeric.astype(float)
            manifest_rows.append(feature_row(source, feature, strategy=STRATEGY))
            continue

        _numeric, _known, unknown, structural_missing = categorical_state(
            frame, source
        )
        group_missing = structural_missing | unknown
        for feature in source.observed_features:
            if not feature.keep:
                manifest_rows.append(feature_row(source, feature, strategy=STRATEGY))
                continue
            values = (numeric == feature.level).astype(float)
            values.loc[group_missing] = np.nan
            columns[feature.feature_name] = values
            manifest_rows.append(feature_row(source, feature, strategy=STRATEGY))

        unknown_count, denominator = unknown_counts(
            frame, source, id_column=schema.id_column, test_ids=test_ids
        )
        unknown_rows.append(
            unknown_qa_row(
                source_column=source.source_column,
                unknown_count=unknown_count,
                denominator=denominator,
                threshold=unknown_rate_threshold,
            )
        )

    features = pd.DataFrame(columns)
    manifest = manifest_frame(manifest_rows)
    return EncodedResult(
        strategy=STRATEGY,
        features=features,
        feature_manifest=manifest,
        qa={"unknown_categories": unknown_rows},
    )
