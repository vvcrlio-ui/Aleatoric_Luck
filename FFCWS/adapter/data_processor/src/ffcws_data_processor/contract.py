"""Emit v5 engine schemas and canonical feature-universe definitions."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .common.io import file_sha256, write_json


CLASSIFICATION_OUTCOMES = frozenset({"eviction", "layoff", "jobTraining"})


def _none(value: Any) -> Any:
    if (
        value is None
        or value == ""
        or (isinstance(value, (float, np.floating)) and np.isnan(value))
    ):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() == "true"


def canonical_feature_universe(manifest: pd.DataFrame) -> dict[str, Any]:
    kept = manifest[manifest["keep"].map(_bool)].copy()
    kept["source_order"] = pd.to_numeric(kept["source_order"]).astype(int)
    kept["feature_order"] = pd.to_numeric(kept["feature_order"]).astype(int)
    kept = kept.sort_values(["source_order", "feature_order"], kind="stable")
    sources = []
    for source, rows in kept.groupby("source_column", sort=False):
        unit_type = str(rows["unit_type"].iloc[0])
        ordinal_levels = []
        if unit_type == "ordinal":
            ordinal_levels = json.loads(str(rows["ordinal_levels"].iloc[0]))
        reference_level = _none(rows["reference_level"].iloc[0])
        sources.append(
            {
                "source": str(source),
                "source_order": int(rows["source_order"].iloc[0]),
                "unit_type": unit_type,
                "features": [
                    {
                        "feature": str(row.feature_name),
                        "feature_order": int(row.feature_order),
                        "level_value": (
                            _none(row.level_value)
                            if unit_type == "onehot_group"
                            else None
                        ),
                        "is_reference": _bool(row.is_reference),
                    }
                    for row in rows.itertuples(index=False)
                ],
                "drop_first": _bool(rows["drop_first"].iloc[0]),
                "reference_level": reference_level,
                "ordinal_levels": ordinal_levels,
            }
        )
    result = {
        "sources": sources,
        "predictors": kept["feature_name"].astype(str).tolist(),
    }
    audit = manifest.loc[~manifest["keep"].map(_bool)].copy()
    if not audit.empty:
        audit["source_order"] = pd.to_numeric(audit["source_order"]).astype(int)
        audit["feature_order"] = pd.to_numeric(audit["feature_order"]).astype(int)
        audit = audit.sort_values(
            ["source_order", "feature_order", "feature_name"], kind="stable"
        )
        result["audit_features"] = [
            {
                "source": str(row.source_column),
                "feature": str(row.feature_name),
                "source_order": int(row.source_order),
                "feature_order": int(row.feature_order),
                "unit_type": str(row.unit_type),
                "level_value": _none(row.level_value),
                "ordinal_levels": _none(row.ordinal_levels),
                "reference_level": _none(row.reference_level),
            }
            for row in audit.itertuples(index=False)
        ]
    return result


def enforce_outcome_train_category_coverage(
    outcome_frames: Mapping[tuple[str, str], pd.DataFrame],
    manifest: pd.DataFrame,
    *,
    unknown_rate_threshold: float,
) -> tuple[dict[tuple[str, str], pd.DataFrame], pd.DataFrame]:
    """Mask test category states not observed in that outcome's train rows.

    The shared feature vocabulary is learned from the official training pool,
    but dropping rows with a missing outcome can remove a rare category from an
    outcome-specific training table.  External-test validation is performed
    after that deletion, so the matching test state must be treated as unknown
    for that outcome instead of leaking a vocabulary learned from other rows.
    """

    if not 0.0 <= unknown_rate_threshold <= 1.0:
        raise ValueError("unknown_rate_threshold must be between 0 and 1")

    frames = {key: frame.copy() for key, frame in outcome_frames.items()}
    kept = manifest[manifest["keep"].map(_bool)].copy()
    kept["source_order"] = pd.to_numeric(kept["source_order"]).astype(int)
    kept["feature_order"] = pd.to_numeric(kept["feature_order"]).astype(int)
    kept = kept.sort_values(["source_order", "feature_order"], kind="stable")
    categorical = kept[kept["unit_type"].isin(["ordinal", "onehot_group"])]
    outcomes = sorted({outcome for split, outcome in frames if split == "train"})
    qa_rows: list[dict[str, Any]] = []

    for outcome in outcomes:
        train = frames[("train", outcome)]
        test = frames[("test", outcome)]
        for source, rows in categorical.groupby("source_column", sort=False):
            unit_type = str(rows["unit_type"].iloc[0])
            features = rows["feature_name"].astype(str).tolist()
            if unit_type == "ordinal":
                feature = features[0]
                train_states = set(train[feature].dropna().tolist())
                observed_test = test[feature].notna()
                unseen = observed_test & ~test[feature].isin(train_states)
            else:
                train_values = train.loc[:, features]
                train_complete = train_values.notna().all(axis=1)
                train_states = {
                    tuple(row)
                    for row in train_values.loc[train_complete].to_numpy(dtype=float)
                }
                test_values = test.loc[:, features]
                observed_test = test_values.notna().all(axis=1)
                test_states = pd.Series(
                    [
                        tuple(row) if complete else None
                        for row, complete in zip(
                            test_values.to_numpy(dtype=float),
                            observed_test.to_numpy(dtype=bool),
                        )
                    ],
                    index=test.index,
                )
                unseen = observed_test & ~test_states.isin(train_states)

            denominator = int(observed_test.sum())
            unknown_count = int(unseen.sum())
            unknown_rate = unknown_count / denominator if denominator else 0.0
            if unknown_rate > unknown_rate_threshold:
                raise ValueError(
                    "Outcome-specific unknown category rate exceeds threshold: "
                    f"outcome={outcome} source={source} "
                    f"unknown_count={unknown_count} denominator={denominator} "
                    f"rate={unknown_rate:.6f} "
                    f"threshold={unknown_rate_threshold:.6f}"
                )
            if unknown_count:
                test.loc[unseen, features] = np.nan
            qa_rows.append(
                {
                    "outcome": outcome,
                    "source_column": str(source),
                    "unit_type": unit_type,
                    "unknown_count": unknown_count,
                    "denominator": denominator,
                    "unknown_rate": float(unknown_rate),
                    "threshold": float(unknown_rate_threshold),
                }
            )
        frames[("test", outcome)] = test

    return frames, pd.DataFrame(
        qa_rows,
        columns=[
            "outcome",
            "source_column",
            "unit_type",
            "unknown_count",
            "denominator",
            "unknown_rate",
            "threshold",
        ],
    )


def write_engine_schema(
    *,
    schema_root: Path,
    dataset_dir: Path,
    dataset: str,
    outcome: str,
    table_path: Path,
    test_path: Path,
    manifest_path: Path,
    manifest: pd.DataFrame,
    id_column: str,
    feature_universe_stem: str | None = None,
) -> Path:
    schema_root.mkdir(parents=True, exist_ok=True)
    definition_stem = feature_universe_stem or dataset
    if Path(definition_stem).name != definition_stem:
        raise ValueError("feature_universe_stem must be a plain file stem")
    definition_path = schema_root / f"{definition_stem}.feature_universe.json"
    definition = canonical_feature_universe(manifest)
    write_json(definition_path, definition)
    schema_path = schema_root / f"{dataset}.json"

    def relative(path: Path) -> str:
        return os.path.relpath(path.resolve(), schema_root.resolve())

    schema = {
        "schema_version": 1,
        "feature_manifest_version": 1,
        "dataset": dataset,
        "table": relative(table_path),
        "test_table": relative(test_path),
        "split_mode": "external_test",
        "task": (
            "classification"
            if outcome in CLASSIFICATION_OUTCOMES
            else "regression"
        ),
        "outcome_columns": [outcome],
        "id_column": id_column,
        "predictor_columns": definition["predictors"],
        "predictor_prefix": None,
        "feature_manifest": relative(manifest_path),
        "exchangeable": True,
        "feature_universe": {
            "mode": "train_pool_screened",
            "definition_file": definition_path.name,
            "definition_sha256": file_sha256(definition_path),
        },
        "group_column": None,
        "imputation": {
            "continuous": "median",
            "ordinal": "median_snap",
            "onehot_group": "atomic_mode",
            "model_overrides": {
                "lightgbm": "passthrough",
                "xgboost": "passthrough",
            },
        },
        "max_train_outcome_missing_ratio": 0.5,
        "max_test_outcome_missing_ratio": 0.5,
        "continuous_priors": None,
    }
    write_json(schema_path, schema)
    write_json(
        dataset_dir / "provenance.json",
        {
            "adapter": "ffcws",
            "dataset": dataset,
            "schema_sha256": file_sha256(schema_path),
            "feature_universe_sha256": file_sha256(definition_path),
            "feature_manifest_sha256": file_sha256(manifest_path),
            "data_sha256": file_sha256(table_path),
            "test_sha256": file_sha256(test_path),
        },
    )
    return schema_path
