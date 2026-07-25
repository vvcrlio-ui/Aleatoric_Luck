from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from aleatoric_nk_grid.preprocessing import source_groups
from aleatoric_nk_grid.validate_input import canonical_feature_universe


DEFAULT_IMPUTATION = {
    "continuous": "median",
    "ordinal": "most_frequent",
    "onehot_group": "atomic_mode",
    "model_overrides": {},
}


def write_schema_bundle(
    root: Path,
    train: pd.DataFrame,
    *,
    outcome: str = "y",
    task: str = "regression",
    split_mode: str = "internal_random",
    test: pd.DataFrame | None = None,
    predictors: list[str] | None = None,
    predictor_prefix: list[str] | None = None,
    manifest: pd.DataFrame | None = None,
    id_column: str | None = None,
    imputation: dict[str, Any] | None = None,
    max_train_missing: float = 0.5,
    max_test_missing: float = 0.5,
    continuous_priors: dict[str, float] | None = None,
    schema_overrides: dict[str, Any] | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    train_path = root / "train.csv"
    test_path = root / "test.csv"
    manifest_path = root / "feature_manifest.csv"
    definition_path = root / "feature_universe.json"
    train.to_csv(train_path, index=False)
    if test is not None:
        test.to_csv(test_path, index=False)
    if manifest is not None:
        manifest.to_csv(manifest_path, index=False)
    if predictors is None:
        if predictor_prefix is None:
            predictors = [
                column
                for column in train.columns
                if column not in {outcome, id_column}
            ]
        else:
            predictors = [
                column
                for column in train.columns
                if column.startswith(tuple(predictor_prefix))
            ]
    groups = source_groups(predictors, manifest, continuous_priors)
    definition = canonical_feature_universe(predictors, groups, manifest)
    definition_text = (
        json.dumps(
            definition,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    definition_path.write_text(definition_text, encoding="utf-8")
    definition_hash = hashlib.sha256(definition_path.read_bytes()).hexdigest()
    schema = {
        "schema_version": 1,
        "feature_manifest_version": 1 if manifest is not None else None,
        "dataset": "synthetic",
        "table": "train.csv",
        "test_table": "test.csv" if test is not None else None,
        "split_mode": split_mode,
        "task": task,
        "outcome_columns": [outcome],
        "id_column": id_column,
        "predictor_columns": predictors if predictor_prefix is None else None,
        "predictor_prefix": predictor_prefix,
        "feature_manifest": (
            "feature_manifest.csv" if manifest is not None else None
        ),
        "exchangeable": True,
        "feature_universe": {
            "mode": (
                "fixed_a_priori"
                if split_mode == "internal_random"
                else "train_pool_screened"
            ),
            "definition_file": "feature_universe.json",
            "definition_sha256": definition_hash,
        },
        "group_column": None,
        "imputation": imputation or DEFAULT_IMPUTATION,
        "max_train_outcome_missing_ratio": max_train_missing,
        "max_test_outcome_missing_ratio": max_test_missing,
        "continuous_priors": continuous_priors,
    }
    schema.update(schema_overrides or {})
    schema_path = root / "schema.json"
    schema_path.write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return schema_path
