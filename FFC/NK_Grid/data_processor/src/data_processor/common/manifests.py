"""Manifest builders and cross-strategy validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import pandas as pd

from .schema import FeatureDef, SharedSchema, SourceSpec


@dataclass
class EncodedResult:
    strategy: str
    features: pd.DataFrame
    feature_manifest: pd.DataFrame
    qa: dict[str, Any] = field(default_factory=dict)
    ordinal_mappings: dict[str, Any] = field(default_factory=dict)


def source_manifest_frame(schema: SharedSchema) -> pd.DataFrame:
    rows = []
    for source in schema.sources:
        rows.append(
            {
                "source_column": source.source_column,
                "safe_name": source.safe_name,
                "source_order": source.source_order,
                "status": source.status,
                "eligible": source.eligible,
                "reason": source.reason,
                "distinct": source.distinct,
                "raw_valid_rate": source.raw_valid_rate,
                "numeric_fraction_among_raw_valid": (
                    source.numeric_fraction_among_raw_valid
                ),
                "observed_level_count": len(source.levels),
                "has_blank_missing": source.has_blank_missing,
            }
        )
    return pd.DataFrame(rows)


def feature_row(
    source: SourceSpec,
    feature: FeatureDef,
    *,
    strategy: str,
    mapping_id: str = "",
) -> dict[str, Any]:
    return {
        "source_column": source.source_column,
        "feature_name": feature.feature_name,
        "kind": feature.kind,
        "strategy": strategy,
        "keep": bool(feature.keep),
        "reason": feature.reason,
        "source_order": source.source_order,
        "prevalence": feature.prevalence,
        "observed_variance": None,
        "mapping_id": mapping_id,
    }


def manifest_frame(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "source_column",
        "feature_name",
        "kind",
        "strategy",
        "keep",
        "reason",
        "source_order",
        "prevalence",
        "observed_variance",
        "mapping_id",
    ]
    frame = pd.DataFrame(list(rows), columns=columns)
    return frame.reset_index(drop=True)


def kept_source_order(manifest: pd.DataFrame) -> list[str]:
    kept = manifest[manifest["keep"].astype(bool)]
    return (
        kept.loc[:, ["source_column", "source_order"]]
        .drop_duplicates()
        .sort_values("source_order", kind="stable")["source_column"]
        .astype(str)
        .tolist()
    )


def validate_feature_manifest(
    features: pd.DataFrame, manifest: pd.DataFrame, *, id_column: str
) -> None:
    required = {"source_column", "feature_name", "keep", "source_order"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Feature manifest is missing columns: {sorted(missing)}")
    kept = manifest[manifest["keep"].astype(bool)]
    names = kept["feature_name"].astype(str)
    if names.duplicated().any():
        duplicate = names[names.duplicated()].iloc[0]
        raise ValueError(f"Feature manifest maps predictor more than once: {duplicate}")
    predictors = [column for column in features.columns if column != id_column]
    if predictors != names.tolist():
        raise ValueError("Feature columns do not match kept manifest order")


def validate_cross_strategy_sources(results: Iterable[EncodedResult]) -> list[str]:
    expected: list[str] | None = None
    for result in results:
        current = kept_source_order(result.feature_manifest)
        if expected is None:
            expected = current
        elif current != expected:
            raise ValueError(
                f"Strategy {result.strategy} has a different canonical source set/order"
            )
    return expected or []
