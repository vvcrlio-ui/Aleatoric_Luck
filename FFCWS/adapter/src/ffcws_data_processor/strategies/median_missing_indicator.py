"""FFC baseline-compatible one-hot and per-code missing indicators."""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from ..common.manifests import EncodedResult, feature_row, manifest_frame
from ..common.schema import FeatureDef, SharedSchema, numeric_values
from ..common.validation import unknown_qa_row
from ._shared import categorical_state, unknown_counts


STRATEGY = "median_missing_indicator"


def _missing_values(
    raw: pd.Series, blank: pd.Series, token: str | None
) -> pd.Series:
    if token == "blank":
        return blank.astype("int8")
    return (raw == str(token)).astype("int8")


def _indicator_feature(feature: FeatureDef) -> FeatureDef:
    name = feature.feature_name
    if name.startswith("C_"):
        name = f"M_{name[2:]}"
    return FeatureDef(
        feature_name=name,
        kind="M",
        missing_token=feature.missing_token,
        prevalence=feature.prevalence,
        keep=feature.keep,
        reason=feature.reason,
    )


def _indicator_source(source_column: str, token: str | None) -> str:
    suffix = "blank" if token == "blank" else f"code_{token}"
    return f"{source_column}__missing__{suffix}"


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
    next_source_order = 0

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
                    source_order=next_source_order,
                    feature_order=0,
                    unit_type="continuous",
                    source_prior=0.0,
                )
            )
            next_source_order += 1
            for feature in source.missing_features:
                indicator = _indicator_feature(feature)
                if indicator.keep:
                    columns[indicator.feature_name] = _missing_values(
                        raw, blank, indicator.missing_token
                    ).astype(float)
                manifest_rows.append(
                    feature_row(
                        source,
                        indicator,
                        strategy=STRATEGY,
                        source_column=_indicator_source(
                            source.source_column, indicator.missing_token
                        ),
                        source_order=next_source_order,
                        feature_order=0,
                        unit_type="continuous",
                        source_prior=0.0,
                    )
                )
                next_source_order += 1
        else:
            _numeric, _known, unknown, structural_missing = categorical_state(
                frame, source
            )
            group_missing = structural_missing | unknown
            reference_level = source.observed_features[0].level
            for feature_order, feature in enumerate(source.observed_features):
                values = (numeric == feature.level).astype(float)
                values.loc[group_missing] = float("nan")
                columns[feature.feature_name] = values
                manifest_rows.append(
                    feature_row(
                        source,
                        feature,
                        strategy=STRATEGY,
                        source_order=next_source_order,
                        feature_order=feature_order,
                        unit_type="onehot_group",
                        is_reference=feature_order == 0,
                        reference_level=reference_level,
                        level_value=feature.level,
                        force_keep=True,
                    )
                )
            next_source_order += 1
            for feature in source.missing_features:
                indicator = _indicator_feature(feature)
                if indicator.keep:
                    columns[indicator.feature_name] = _missing_values(
                        raw, blank, indicator.missing_token
                    ).astype(float)
                manifest_rows.append(
                    feature_row(
                        source,
                        indicator,
                        strategy=STRATEGY,
                        source_column=_indicator_source(
                            source.source_column, indicator.missing_token
                        ),
                        source_order=next_source_order,
                        feature_order=0,
                        unit_type="continuous",
                        source_prior=0.0,
                    )
                )
                next_source_order += 1
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
