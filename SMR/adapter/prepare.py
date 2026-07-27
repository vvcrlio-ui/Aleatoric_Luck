"""Build and validate the SMR adapter artifacts required by the shared engine."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from aleatoric_nk_grid.ingest import canonical_json, load_input
from aleatoric_nk_grid.preprocessing import source_groups
from aleatoric_nk_grid.validate_input import (
    canonical_feature_universe,
    validate_input,
)


ADAPTER_VERSION = "smr-adapter-v1"
DEFAULT_CONTRACT = "asample2_withlag.json"
MANIFEST_COLUMNS = (
    "source_column",
    "feature_name",
    "keep",
    "source_order",
    "feature_order",
    "unit_type",
    "drop_first",
    "is_reference",
    "reference_level",
    "level_value",
    "ordinal_levels",
    "source_prior",
)


@dataclass(frozen=True)
class AdapterResult:
    schema_path: Path
    predictor_count: int
    source_count: int


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


def _load_contract(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        contract = json.load(handle)
    if not isinstance(contract, dict):
        raise ValueError("SMR feature contract must be a JSON object")
    required = {
        "contract_version",
        "dataset",
        "outcome_columns",
        "predictor_columns",
        "onehot_groups",
        "missing_value_codes",
        "exchangeability_justification",
    }
    missing = sorted(required - set(contract))
    if missing:
        raise ValueError(f"SMR feature contract is missing fields: {missing}")
    if contract["contract_version"] != 1:
        raise ValueError("SMR feature contract contract_version must be 1")
    if not str(contract["exchangeability_justification"]).strip():
        raise ValueError("SMR feature contract requires exchangeability justification")
    outcomes = contract["outcome_columns"]
    predictors = contract["predictor_columns"]
    if (
        not isinstance(outcomes, list)
        or not outcomes
        or len(outcomes) != len(set(outcomes))
    ):
        raise ValueError("outcome_columns must be a nonempty unique list")
    if (
        not isinstance(predictors, list)
        or not predictors
        or len(predictors) != len(set(predictors))
    ):
        raise ValueError("predictor_columns must be a nonempty unique list")
    overlap = set(outcomes) & set(predictors)
    if overlap:
        raise ValueError(f"Outcomes cannot be predictors: {sorted(overlap)}")
    if not isinstance(contract["onehot_groups"], dict):
        raise ValueError("onehot_groups must be an object")
    if not isinstance(contract["missing_value_codes"], dict):
        raise ValueError("missing_value_codes must be an object")
    return contract


def _validate_source_header(
    header: Sequence[str],
    *,
    outcomes: Sequence[str],
    predictors: Sequence[str],
) -> None:
    missing = [column for column in [*outcomes, *predictors] if column not in header]
    if missing:
        raise KeyError(f"Contract columns absent from the source table: {missing[:10]}")
    declared = set(predictors)
    undeclared = [
        column
        for column in header
        if column.startswith(("Aset", "Bset")) and column not in declared
    ]
    if undeclared:
        raise ValueError(
            "Source table contains undeclared Aset/Bset predictors: "
            f"{undeclared[:10]}"
        )
    resolved = [column for column in header if column in declared]
    if list(predictors) != resolved:
        raise ValueError(
            "Source predictor order differs from the fixed feature contract"
        )


def _normalize_missing_codes(
    frame: pd.DataFrame, mapping: dict[str, Any]
) -> pd.DataFrame:
    normalized = frame.copy()
    for column, codes in mapping.items():
        if column not in normalized:
            raise KeyError(f"Missing-code rule references absent column: {column}")
        if not isinstance(codes, list):
            raise ValueError(f"Missing-code rule for {column!r} must be a list")
        normalized[column] = normalized[column].replace(codes, pd.NA)
    return normalized


def _manifest_from_contract(
    predictors: Sequence[str], onehot_groups: dict[str, Any]
) -> pd.DataFrame:
    feature_metadata: dict[str, dict[str, Any]] = {}
    for source, definition in onehot_groups.items():
        if not isinstance(definition, dict):
            raise ValueError(f"One-hot definition for {source!r} must be an object")
        features = definition.get("features")
        levels = definition.get("level_values")
        reference = definition.get("reference_level")
        if (
            not isinstance(features, list)
            or len(features) < 2
            or len(features) != len(set(features))
        ):
            raise ValueError(
                f"One-hot source {source!r} requires at least two unique features"
            )
        if not isinstance(levels, list) or len(levels) != len(features):
            raise ValueError(
                f"One-hot source {source!r} requires one level per feature"
            )
        if len({str(value) for value in levels}) != len(levels):
            raise ValueError(f"One-hot source {source!r} has duplicate levels")
        if str(reference) not in {str(value) for value in levels}:
            raise ValueError(
                f"One-hot source {source!r} reference is absent from its levels"
            )
        positions = []
        for feature_order, (feature, level) in enumerate(zip(features, levels)):
            if feature in feature_metadata:
                raise ValueError(f"Feature appears in two one-hot groups: {feature}")
            try:
                positions.append(list(predictors).index(feature))
            except ValueError as exc:
                raise ValueError(
                    f"One-hot feature is absent from predictor_columns: {feature}"
                ) from exc
            feature_metadata[feature] = {
                "source_column": source,
                "feature_order": feature_order,
                "level_value": level,
                "is_reference": str(level) == str(reference),
                "reference_level": reference,
            }
        expected = list(range(min(positions), max(positions) + 1))
        if positions != expected:
            raise ValueError(
                f"One-hot source {source!r} must occupy a contiguous predictor block"
            )

    rows: list[dict[str, Any]] = []
    source_orders: dict[str, int] = {}
    for feature in predictors:
        metadata = feature_metadata.get(feature)
        source = feature if metadata is None else str(metadata["source_column"])
        if source not in source_orders:
            source_orders[source] = len(source_orders)
        if metadata is None:
            rows.append(
                {
                    "source_column": source,
                    "feature_name": feature,
                    "keep": True,
                    "source_order": source_orders[source],
                    "feature_order": 0,
                    "unit_type": "continuous",
                    "drop_first": False,
                    "is_reference": False,
                    "reference_level": None,
                    "level_value": None,
                    "ordinal_levels": None,
                    "source_prior": 0.0,
                }
            )
        else:
            rows.append(
                {
                    "source_column": source,
                    "feature_name": feature,
                    "keep": True,
                    "source_order": source_orders[source],
                    "feature_order": metadata["feature_order"],
                    "unit_type": "onehot_group",
                    "drop_first": False,
                    "is_reference": metadata["is_reference"],
                    "reference_level": metadata["reference_level"],
                    "level_value": metadata["level_value"],
                    "ordinal_levels": None,
                    "source_prior": None,
                }
            )
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def prepare_smr(
    source: Path,
    *,
    article_root: Path,
    contract_path: Path,
    validation_models: Sequence[str] = ("ols",),
    min_n: int = 10,
    test_size: float = 0.3,
    seed: int = 12345,
) -> AdapterResult:
    source = Path(source).resolve()
    article_root = Path(article_root).resolve()
    contract_path = Path(contract_path).resolve()
    contract = _load_contract(contract_path)
    dataset = str(contract["dataset"])
    outcomes = [str(value) for value in contract["outcome_columns"]]
    predictors = [str(value) for value in contract["predictor_columns"]]

    header = pd.read_csv(source, nrows=0).columns.astype(str).tolist()
    _validate_source_header(header, outcomes=outcomes, predictors=predictors)
    projected = pd.read_csv(source, usecols=[*outcomes, *predictors])
    projected = projected.loc[:, [*outcomes, *predictors]]
    projected = _normalize_missing_codes(
        projected, dict(contract["missing_value_codes"])
    )

    # Required generation order: ARD -> manifest -> universe -> schema.
    ard_dir = article_root / "data" / "ard" / dataset
    ard_dir.mkdir(parents=True, exist_ok=True)
    table_path = ard_dir / "data.csv"
    projected.to_csv(table_path, index=False)

    manifest_path = ard_dir / "feature_manifest.csv"
    manifest = _manifest_from_contract(predictors, contract["onehot_groups"])
    manifest.to_csv(manifest_path, index=False)
    persisted_manifest = pd.read_csv(manifest_path)

    groups = source_groups(predictors, persisted_manifest, {})
    schema_dir = article_root / "schema"
    schema_dir.mkdir(parents=True, exist_ok=True)
    definition_path = schema_dir / f"{dataset}.feature_universe.json"
    definition = canonical_feature_universe(
        predictors, groups, persisted_manifest
    )
    definition_path.write_text(canonical_json(definition), encoding="utf-8")

    schema_path = schema_dir / f"{dataset}.json"
    schema = {
        "schema_version": 1,
        "feature_manifest_version": 1,
        "dataset": dataset,
        "table": os.path.relpath(table_path, schema_dir),
        "test_table": None,
        "split_mode": "internal_random",
        "task": "regression",
        "outcome_columns": outcomes,
        "id_column": None,
        "predictor_columns": predictors,
        "predictor_prefix": None,
        "feature_manifest": os.path.relpath(manifest_path, schema_dir),
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
            "adapter_version": ADAPTER_VERSION,
            "dataset": dataset,
            "source_sha256": _sha256(source),
            "contract_sha256": _sha256(contract_path),
            "schema_sha256": _sha256(schema_path),
            "feature_universe_sha256": _sha256(definition_path),
            "feature_manifest_sha256": _sha256(manifest_path),
            "ard_sha256": _sha256(table_path),
        },
    )

    for outcome in outcomes:
        loaded = load_input(schema_path, outcome)
        validate_input(
            loaded,
            outcome,
            models=tuple(validation_models),
            min_n=min_n,
            test_size=test_size,
            seed=seed,
        )
    return AdapterResult(
        schema_path=schema_path,
        predictor_count=len(predictors),
        source_count=len(groups),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--article-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--source", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--validation-model", nargs="+", default=["ols"])
    parser.add_argument("--min-n", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args(argv)

    article_root = args.article_root.resolve()
    contract = (
        args.contract
        if args.contract is not None
        else article_root / "adapter" / "contracts" / DEFAULT_CONTRACT
    )
    contract_document = _load_contract(contract)
    source = (
        args.source
        if args.source is not None
        else article_root
        / "data"
        / "private"
        / f"{contract_document['dataset']}.csv"
    )
    result = prepare_smr(
        source,
        article_root=article_root,
        contract_path=contract,
        validation_models=args.validation_model,
        min_n=args.min_n,
        test_size=args.test_size,
        seed=args.seed,
    )
    print(
        f"OK: schema={result.schema_path} "
        f"predictors={result.predictor_count} sources={result.source_count}"
    )


if __name__ == "__main__":
    main()
