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
            manifest_rows.append(
                feature_row(
                    source,
                    feature,
                    strategy=STRATEGY,
                    feature_order=0,
                    unit_type="continuous",
                    source_prior=0.0,
                )
            )
            for audit_order, feature in enumerate(source.missing_features, start=1):
                manifest_rows.append(
                    feature_row(
                        source,
                        feature,
                        strategy=STRATEGY,
                        feature_order=audit_order,
                        unit_type="continuous",
                        force_keep=False,
                    )
                )
        else:
            _numeric, _known, unknown, _structural = categorical_state(frame, source)
            features = [*source.observed_features, *source.missing_features]
            reference_level = (
                source.observed_features[0].level
                if source.observed_features
                else features[0].missing_token
            )
            for feature_order, feature in enumerate(features):
                if feature.missing_token is None:
                    values = (numeric == feature.level).astype("int8")
                    level_value = feature.level
                else:
                    values = _missing_values(raw, blank, feature.missing_token)
                    level_value = feature.missing_token
                values = values.astype(float)
                values.loc[unknown] = float("nan")
                columns[feature.feature_name] = values
                manifest_rows.append(
                    feature_row(
                        source,
                        feature,
                        strategy=STRATEGY,
                        feature_order=feature_order,
                        unit_type="onehot_group",
                        is_reference=feature_order == 0,
                        reference_level=reference_level,
                        level_value=level_value,
                        force_keep=True,
                    )
                )
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

    return EncodedResult(
        strategy=STRATEGY,
        features=pd.DataFrame(columns),
        feature_manifest=manifest_frame(manifest_rows),
        qa={"unknown_categories": unknown_rows},
    )
