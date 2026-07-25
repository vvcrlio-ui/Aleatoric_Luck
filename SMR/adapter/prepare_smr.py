"""Prepare the near-identity SMR adapter output and authoritative schema."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_feature_universe(predictors: list[str]) -> dict[str, Any]:
    return {
        "sources": [
            {
                "source": feature,
                "source_order": order,
                "unit_type": "continuous",
                "features": [
                    {
                        "feature": feature,
                        "feature_order": 0,
                        "level_value": None,
                        "is_reference": False,
                    }
                ],
                "drop_first": False,
                "reference_level": None,
                "ordinal_levels": [],
            }
            for order, feature in enumerate(predictors)
        ],
        "predictors": predictors,
    }


def prepare_smr(
    source: Path,
    *,
    article_root: Path,
    dataset: str,
    outcomes: list[str],
    predictor_prefix: list[str],
) -> Path:
    source = Path(source).resolve()
    article_root = Path(article_root).resolve()
    header = pd.read_csv(source, nrows=0).columns.astype(str).tolist()
    missing_outcomes = [outcome for outcome in outcomes if outcome not in header]
    if missing_outcomes:
        raise KeyError(f"Outcome columns not found: {missing_outcomes}")
    predictors = [
        column for column in header if column.startswith(tuple(predictor_prefix))
    ]
    if not predictors:
        raise ValueError("Predictor prefixes resolved to zero columns")
    projected = pd.read_csv(source, usecols=[*outcomes, *predictors])
    ard_dir = article_root / "data" / "ard" / dataset
    table_path = ard_dir / "data.csv"
    ard_dir.mkdir(parents=True, exist_ok=True)
    projected.to_csv(table_path, index=False)

    schema_dir = article_root / "schema"
    definition_path = schema_dir / f"{dataset}.feature_universe.json"
    _write_json(definition_path, canonical_feature_universe(predictors))
    schema_path = schema_dir / f"{dataset}.json"
    relative_table = os.path.relpath(table_path, schema_dir)
    schema = {
        "schema_version": 1,
        "feature_manifest_version": None,
        "dataset": dataset,
        "table": relative_table,
        "test_table": None,
        "split_mode": "internal_random",
        "task": "regression",
        "outcome_columns": outcomes,
        "id_column": None,
        "predictor_columns": None,
        "predictor_prefix": predictor_prefix,
        "feature_manifest": None,
        "exchangeable": True,
        "feature_universe": {
            "mode": "fixed_a_priori",
            "definition_file": definition_path.name,
            "definition_sha256": _sha256(definition_path),
        },
        "group_column": None,
        "imputation": {
            "continuous": "median",
            "ordinal": "most_frequent",
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
    _write_json(schema_path, schema)
    _write_json(
        ard_dir / "provenance.json",
        {
            "adapter": "smr",
            "dataset": dataset,
            "source_sha256": _sha256(source),
            "schema_sha256": _sha256(schema_path),
            "feature_universe_sha256": _sha256(definition_path),
            "ard_sha256": _sha256(table_path),
        },
    )
    return schema_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument(
        "--article-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--dataset", default="asample2_withlag")
    parser.add_argument(
        "--outcome",
        nargs="+",
        default=["Cm_lhourlywage", "Cm_ltotalincome"],
    )
    parser.add_argument(
        "--predictor-prefix", nargs="+", default=["Aset", "Bset"]
    )
    args = parser.parse_args(argv)
    schema = prepare_smr(
        args.source,
        article_root=args.article_root,
        dataset=args.dataset,
        outcomes=args.outcome,
        predictor_prefix=args.predictor_prefix,
    )
    print(schema)


if __name__ == "__main__":
    main()
