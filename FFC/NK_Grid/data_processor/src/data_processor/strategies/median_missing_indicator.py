"""FFC baseline-compatible one-hot and per-code missing indicators."""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from ..common.manifests import EncodedResult, feature_row, manifest_frame
from ..common.schema import SharedSchema, numeric_values
from ..common.validation import unknown_qa_row
from ._shared import categorical_state, unknown_counts


STRATEGY = "median_missing_indicator"


def _missing_values(
    raw: pd.Series, blank: pd.Series, token: str | None
) -> pd.Series:
    if token == "blank":
        return blank.astype("int8")
    return (raw == str(token)).astype("int8")


def encode_median_missing_indicator(
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
        numeric, raw, blank, _coded = numeric_values(frame[source.source_column])
        if source.status == "numeric":
            feature = source.observed_features[0]
            columns[feature.feature_name] = numeric.astype(float)
            manifest_rows.append(feature_row(source, feature, strategy=STRATEGY))
        else:
            _numeric, _known, unknown, _structural = categorical_state(frame, source)
            for feature in source.observed_features:
                values = (numeric == feature.level).astype("int8")
                manifest_rows.append(feature_row(source, feature, strategy=STRATEGY))
                if feature.keep:
                    columns[feature.feature_name] = values
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

        for feature in source.missing_features:
            values = _missing_values(raw, blank, feature.missing_token)
            manifest_rows.append(feature_row(source, feature, strategy=STRATEGY))
            if feature.keep:
                columns[feature.feature_name] = values

    return EncodedResult(
        strategy=STRATEGY,
        features=pd.DataFrame(columns),
        feature_manifest=manifest_frame(manifest_rows),
        qa={"unknown_categories": unknown_rows},
    )
