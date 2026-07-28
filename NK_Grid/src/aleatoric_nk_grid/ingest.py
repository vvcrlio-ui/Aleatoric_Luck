"""Schema loading and projection-aware table ingestion."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


SCHEMA_VERSION = 1
FEATURE_MANIFEST_VERSION = 1
SCHEMA_FIELDS = frozenset(
    {
        "schema_version",
        "feature_manifest_version",
        "dataset",
        "table",
        "test_table",
        "split_mode",
        "task",
        "outcome_columns",
        "id_column",
        "predictor_columns",
        "predictor_prefix",
        "feature_manifest",
        "exchangeable",
        "feature_universe",
        "group_column",
        "imputation",
        "max_train_outcome_missing_ratio",
        "max_test_outcome_missing_ratio",
        "continuous_priors",
    }
)
REQUIRED_SCHEMA_FIELDS = SCHEMA_FIELDS - frozenset(
    {
        "max_train_outcome_missing_ratio",
        "max_test_outcome_missing_ratio",
        "continuous_priors",
    }
)


def canonical_json(value: Any) -> str:
    """Serialize identity-bearing JSON deterministically."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _resolve_path(value: str | None, directory: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else (directory / path).resolve()


def _require_string_list(value: Any, field: str, *, nonempty: bool = True) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"schema.{field} must be a list of non-empty strings")
    if nonempty and not value:
        raise ValueError(f"schema.{field} must not be empty")
    if len(value) != len(set(value)):
        raise ValueError(f"schema.{field} must not contain duplicates")
    return tuple(value)


@dataclass(frozen=True)
class InputSchema:
    path: Path
    schema_version: int
    feature_manifest_version: int | None
    dataset: str
    table: Path
    test_table: Path | None
    split_mode: str
    task: str
    outcome_columns: tuple[str, ...]
    id_column: str | None
    predictor_columns: tuple[str, ...] | None
    predictor_prefix: tuple[str, ...] | None
    feature_manifest: Path | None
    exchangeable: bool
    feature_universe: Mapping[str, Any]
    group_column: None
    imputation: Mapping[str, Any]
    max_train_outcome_missing_ratio: float
    max_test_outcome_missing_ratio: float
    continuous_priors: Mapping[str, float]
    semantic_contract: Mapping[str, Any]
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class LoadedInput:
    schema: InputSchema
    train: pd.DataFrame
    test: pd.DataFrame | None
    predictors: tuple[str, ...]
    manifest: pd.DataFrame | None


def load_schema(path: Path) -> InputSchema:
    """Load the single authoritative adapter schema and reject ambiguity."""

    path = Path(path).resolve()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid schema JSON: {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError("schema must be a JSON object")
    missing = sorted(REQUIRED_SCHEMA_FIELDS - set(document))
    unknown = sorted(set(document) - SCHEMA_FIELDS)
    if missing:
        raise ValueError(f"schema is missing required fields: {missing}")
    if unknown:
        raise ValueError(f"schema contains unknown fields: {unknown}")
    if document["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version={document['schema_version']!r}; "
            f"supported value is {SCHEMA_VERSION}"
        )
    manifest_value = document["feature_manifest"]
    manifest_version = document["feature_manifest_version"]
    if manifest_value is None:
        if manifest_version is not None:
            raise ValueError(
                "feature_manifest_version must be null when feature_manifest is null"
            )
    elif manifest_version != FEATURE_MANIFEST_VERSION:
        raise ValueError(
            f"Unsupported feature_manifest_version={manifest_version!r}; "
            f"supported value is {FEATURE_MANIFEST_VERSION}"
        )
    if document["split_mode"] not in {"internal_random", "external_test"}:
        raise ValueError("schema.split_mode must be internal_random or external_test")
    if document["task"] not in {"regression", "classification"}:
        raise ValueError("schema.task must be regression or classification")
    if not isinstance(document["dataset"], str) or not document["dataset"]:
        raise ValueError("schema.dataset must be a non-empty string")
    outcomes = _require_string_list(document["outcome_columns"], "outcome_columns")
    columns_value = document["predictor_columns"]
    prefix_value = document["predictor_prefix"]
    if (columns_value is None) == (prefix_value is None):
        raise ValueError(
            "schema must define exactly one of predictor_columns or predictor_prefix"
        )
    predictor_columns = (
        _require_string_list(columns_value, "predictor_columns")
        if columns_value is not None
        else None
    )
    predictor_prefix = (
        _require_string_list(prefix_value, "predictor_prefix")
        if prefix_value is not None
        else None
    )
    if document["exchangeable"] is not True:
        raise ValueError("schema.exchangeable must be true")
    if document["group_column"] is not None:
        raise ValueError("schema.group_column must be null; grouped splitting is unsupported")
    id_column = document["id_column"]
    if id_column is not None and (not isinstance(id_column, str) or not id_column):
        raise ValueError("schema.id_column must be null or a non-empty string")
    if id_column is not None and id_column in outcomes:
        raise ValueError("schema.id_column must not also be an outcome column")
    if predictor_columns is not None:
        protected = set(outcomes)
        if id_column is not None:
            protected.add(id_column)
        overlap = sorted(set(predictor_columns) & protected)
        if overlap:
            raise ValueError(
                "schema.predictor_columns overlaps protected outcome/ID columns: "
                f"{overlap}"
            )
    test_value = document["test_table"]
    if document["split_mode"] == "external_test":
        if test_value is None or id_column is None:
            raise ValueError(
                "external_test requires both schema.test_table and schema.id_column"
            )
    elif test_value is not None:
        raise ValueError("internal_random requires schema.test_table to be null")
    if not isinstance(document["table"], str) or not document["table"]:
        raise ValueError("schema.table must be a non-empty relative or absolute path")
    universe = document["feature_universe"]
    if not isinstance(universe, dict) or not {"mode", "definition_file"}.issubset(universe):
        raise ValueError(
            "schema.feature_universe must contain exactly mode, definition_file, "
            "and definition_file"
        )
    if universe["mode"] not in {"fixed_a_priori", "train_pool_screened"}:
        raise ValueError("Unknown feature_universe.mode")
    if document["split_mode"] == "internal_random" and universe["mode"] != "fixed_a_priori":
        raise ValueError(
            "internal_random requires feature_universe.mode='fixed_a_priori'"
        )
    if not all(
        isinstance(universe[key], str) and universe[key]
        for key in ("definition_file",)
    ):
        raise ValueError("feature_universe definition_file must be a non-empty string")
    imputation = document["imputation"]
    if not isinstance(imputation, dict):
        raise ValueError("schema.imputation must be an object")
    expected_imputation = {
        "continuous",
        "ordinal",
        "onehot_group",
        "model_overrides",
    }
    if set(imputation) != expected_imputation:
        raise ValueError(
            f"schema.imputation must contain exactly {sorted(expected_imputation)}"
        )
    if imputation["continuous"] not in {"median", "mean"}:
        raise ValueError("imputation.continuous must be median or mean")
    if imputation["ordinal"] not in {"most_frequent", "median_snap"}:
        raise ValueError(
            "imputation.ordinal must be most_frequent or median_snap"
        )
    if imputation["onehot_group"] != "atomic_mode":
        raise ValueError("imputation.onehot_group must be atomic_mode")
    if not isinstance(imputation["model_overrides"], dict) or any(
        value != "passthrough" for value in imputation["model_overrides"].values()
    ):
        raise ValueError("imputation.model_overrides only supports passthrough values")
    for model in imputation["model_overrides"]:
        if model not in {"lightgbm", "xgboost"}:
            raise ValueError(
                f"passthrough is not approved for model {model!r}; "
                "only lightgbm and xgboost are NaN-native"
            )
    thresholds = {}
    for field in (
        "max_train_outcome_missing_ratio",
        "max_test_outcome_missing_ratio",
    ):
        value = float(document.get(field, 0.5))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"schema.{field} must be in [0, 1]")
        thresholds[field] = value
    priors = document.get("continuous_priors") or {}
    if not isinstance(priors, dict):
        raise ValueError("schema.continuous_priors must be an object or null")
    normalized_priors: dict[str, float] = {}
    for source, value in priors.items():
        if not isinstance(source, str) or not source:
            raise ValueError("continuous_priors keys must be non-empty strings")
        try:
            normalized = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"continuous prior for {source!r} is not numeric") from exc
        if not pd.notna(normalized) or normalized in (float("inf"), float("-inf")):
            raise ValueError(f"continuous prior for {source!r} must be finite")
        normalized_priors[source] = normalized

    directory = path.parent
    table = _resolve_path(document["table"], directory)
    assert table is not None
    raw_semantics = {
        key: value
        for key, value in document.items()
        if key not in {"table", "test_table", "feature_manifest"}
    }
    # Behavior-equivalent omission/explicit-default forms have one identity.
    raw_semantics["max_train_outcome_missing_ratio"] = thresholds[
        "max_train_outcome_missing_ratio"
    ]
    raw_semantics["max_test_outcome_missing_ratio"] = thresholds[
        "max_test_outcome_missing_ratio"
    ]
    raw_semantics["continuous_priors"] = normalized_priors
    raw_semantics["feature_universe"] = {
        key: value
        for key, value in document["feature_universe"].items()
        if key != "definition_file"
    }
    return InputSchema(
        path=path,
        schema_version=SCHEMA_VERSION,
        feature_manifest_version=manifest_version,
        dataset=document["dataset"],
        table=table,
        test_table=_resolve_path(test_value, directory),
        split_mode=document["split_mode"],
        task=document["task"],
        outcome_columns=outcomes,
        id_column=id_column,
        predictor_columns=predictor_columns,
        predictor_prefix=predictor_prefix,
        feature_manifest=_resolve_path(manifest_value, directory),
        exchangeable=True,
        feature_universe=dict(universe),
        group_column=None,
        imputation=dict(imputation),
        max_train_outcome_missing_ratio=thresholds[
            "max_train_outcome_missing_ratio"
        ],
        max_test_outcome_missing_ratio=thresholds[
            "max_test_outcome_missing_ratio"
        ],
        continuous_priors=normalized_priors,
        semantic_contract=raw_semantics,
        raw=raw_semantics,
    )


def table_columns(path: Path) -> tuple[str, ...]:
    """Read only CSV headers or Parquet metadata."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return tuple(pd.read_csv(path, nrows=0).columns.astype(str))
    if suffix in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError("Reading Parquet metadata requires pyarrow") from exc
        return tuple(pq.ParquetFile(path).schema_arrow.names)
    raise ValueError(
        f"Unsupported table suffix {suffix!r}; supported suffixes are "
        ".csv, .parquet, and .pq"
    )


def read_table(path: Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Read CSV/Parquet, optionally projecting only resolved columns."""

    path = Path(path)
    suffix = path.suffix.lower()
    selected = list(columns) if columns is not None else None
    if suffix == ".csv":
        return pd.read_csv(path, usecols=selected)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path, columns=selected)
    raise ValueError(
        f"Unsupported table suffix {suffix!r}; supported suffixes are "
        ".csv, .parquet, and .pq"
    )


def read_feature_manifest(schema: InputSchema) -> pd.DataFrame | None:
    if schema.feature_manifest is None:
        return None
    if not schema.feature_manifest.exists():
        raise FileNotFoundError(
            f"Feature manifest not found: {schema.feature_manifest}"
        )
    return pd.read_csv(schema.feature_manifest)


def _manifest_bool(series: pd.Series, field: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise ValueError(f"manifest.{field} must contain only true/false")
    return normalized.eq("true")


def resolve_predictors(
    columns: Sequence[str],
    schema: InputSchema,
    manifest: pd.DataFrame | None = None,
) -> tuple[str, ...]:
    available = tuple(str(column) for column in columns)
    available_set = set(available)
    if schema.predictor_columns is not None:
        predictors = list(schema.predictor_columns)
    else:
        assert schema.predictor_prefix is not None
        predictors = [
            column
            for column in available
            if column.startswith(tuple(schema.predictor_prefix))
        ]
    if not predictors:
        raise ValueError("schema predictor rule resolved to zero columns")
    protected = set(schema.outcome_columns)
    if schema.id_column is not None:
        protected.add(schema.id_column)
    overlap = sorted(set(predictors) & protected)
    if overlap:
        raise ValueError(
            "Resolved predictors overlap protected outcome/ID columns: "
            f"{overlap}"
        )
    missing = [column for column in predictors if column not in available_set]
    if missing:
        raise ValueError(f"Training data is missing predictor columns: {missing}")
    if manifest is not None:
        required = {
            "source_column",
            "feature_name",
            "keep",
            "source_order",
            "feature_order",
        }
        missing_fields = sorted(required - set(manifest))
        if missing_fields:
            raise ValueError(
                f"Feature manifest is missing columns: {missing_fields}"
            )
        kept = manifest.loc[_manifest_bool(manifest["keep"], "keep")].copy()
        kept["source_order"] = pd.to_numeric(
            kept["source_order"], errors="raise"
        ).astype(int)
        kept["feature_order"] = pd.to_numeric(
            kept["feature_order"], errors="raise"
        ).astype(int)
        kept = kept.sort_values(
            ["source_order", "feature_order"], kind="stable"
        )
        manifest_predictors = kept["feature_name"].astype(str).tolist()
        if set(manifest_predictors) != set(predictors):
            absent = sorted(set(predictors) - set(manifest_predictors))
            orphan = sorted(set(manifest_predictors) - set(predictors))
            raise ValueError(
                "Kept manifest rows must exactly cover resolved predictors; "
                f"missing={absent}, orphan={orphan}"
            )
        predictors = manifest_predictors
    return tuple(predictors)


def load_input(schema_path: Path, outcome: str) -> LoadedInput:
    """Resolve headers first, then project only outcome/predictor/ID columns."""

    schema = load_schema(schema_path)
    if outcome not in schema.outcome_columns:
        raise ValueError(
            f"Outcome {outcome!r} is not declared in schema.outcome_columns"
        )
    if not schema.table.exists():
        raise FileNotFoundError(f"Training table not found: {schema.table}")
    manifest = read_feature_manifest(schema)
    train_header = table_columns(schema.table)
    predictors = resolve_predictors(train_header, schema, manifest)
    required = [outcome, *predictors]
    if schema.id_column is not None:
        required.append(schema.id_column)
    missing_required = [column for column in required if column not in train_header]
    if missing_required:
        raise ValueError(
            f"Training data is missing required columns: {missing_required}"
        )
    train = read_table(schema.table, required)
    test = None
    if schema.test_table is not None:
        if not schema.test_table.exists():
            raise FileNotFoundError(f"External test table not found: {schema.test_table}")
        test_header = table_columns(schema.test_table)
        missing_test = [column for column in required if column not in test_header]
        if missing_test:
            raise ValueError(
                f"External test data is missing required columns: {missing_test}"
            )
        test = read_table(schema.test_table, required)
    return LoadedInput(
        schema=schema,
        train=train,
        test=test,
        predictors=predictors,
        manifest=manifest,
    )
