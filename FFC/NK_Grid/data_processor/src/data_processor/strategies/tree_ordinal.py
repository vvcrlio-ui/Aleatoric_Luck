"""Deterministic numeric ordinal encoding with NaN passthrough."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..common.io import stable_hash
from ..common.manifests import EncodedResult, feature_row, manifest_frame
from ..common.schema import FeatureDef, SharedSchema, numeric_values
from ..common.validation import unknown_qa_row
from ._shared import categorical_state, unknown_counts


STRATEGY = "tree_ordinal"


def encode_tree_ordinal(
    frame: pd.DataFrame,
    schema: SharedSchema,
    *,
    test_ids: Iterable[Any] = (),
    unknown_rate_threshold: float = 0.95,
) -> EncodedResult:
    columns: dict[str, pd.Series] = {schema.id_column: frame[schema.id_column]}
    manifest_rows: list[dict[str, Any]] = []
    mappings: dict[str, Any] = {}
    unknown_rows: list[dict[str, Any]] = []

    for source in schema.eligible_sources:
        numeric, _raw, _blank, _coded = numeric_values(frame[source.source_column])
        if source.status == "numeric":
            feature = source.observed_features[0]
            columns[feature.feature_name] = numeric.astype("float64")
            manifest_rows.append(feature_row(source, feature, strategy=STRATEGY))
            continue

        mapping = {str(level): index for index, level in enumerate(source.levels)}
        mapping_id = stable_hash(
            {"source_column": source.source_column, "mapping": mapping}
        )
        mappings[source.source_column] = {
            "mapping_id": mapping_id,
            "level_to_code": mapping,
        }
        ordinal = pd.Series(np.nan, index=frame.index, dtype="float64")
        for level, code in zip(source.levels, range(len(source.levels))):
            ordinal.loc[numeric == level] = float(code)
        feature = FeatureDef(
            feature_name=f"C_{source.safe_name}",
            kind="C",
            keep=True,
            reason="kept",
        )
        columns[feature.feature_name] = ordinal
        manifest_rows.append(
            feature_row(source, feature, strategy=STRATEGY, mapping_id=mapping_id)
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
        ordinal_mappings=mappings,
    )
