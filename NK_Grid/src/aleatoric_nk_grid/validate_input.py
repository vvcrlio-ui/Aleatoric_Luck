"""Fail-fast validation for adapter schemas and projected ARD tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .ingest import InputSchema, LoadedInput, canonical_json
from .preprocessing import SourceGroup, source_groups, validate_onehot_states


REGRESSION_CV_MIN_N = {
    "ridge": 2,
    "lasso": 2,
    "lightgbm": 5,
    "super_learner": 5,
}


def canonical_feature_universe(
    predictors: Sequence[str],
    groups: Sequence[SourceGroup],
    manifest: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Return the complete identity-bearing feature/source definition."""

    result = {
        "sources": [
            {
                "source": group.name,
                "source_order": group.source_order,
                "unit_type": group.unit_type,
                "features": [
                    {
                        "feature": feature,
                        "feature_order": feature_order,
                        "level_value": (
                            group.level_values[feature_order]
                            if group.unit_type == "onehot_group"
                            else None
                        ),
                        "is_reference": feature == group.reference_feature,
                    }
                    for feature_order, feature in enumerate(group.features)
                ],
                "drop_first": group.drop_first,
                "reference_level": group.reference_level,
                "ordinal_levels": list(group.ordinal_levels),
            }
            for group in groups
        ],
        "predictors": list(predictors),
    }
    if manifest is not None:
        audit = manifest.copy()
        keep = audit["keep"].map(
            lambda value: (
                bool(value)
                if isinstance(value, (bool, np.bool_))
                else str(value).strip().lower() == "true"
            )
        )
        audit = audit.loc[~keep].copy()
        if not audit.empty:
            audit["source_order"] = pd.to_numeric(
                audit["source_order"], errors="raise"
            ).astype(int)
            audit["feature_order"] = pd.to_numeric(
                audit["feature_order"], errors="raise"
            ).astype(int)
            audit = audit.sort_values(
                ["source_order", "feature_order", "feature_name"], kind="stable"
            )

            def clean(value: Any) -> Any:
                if value is None or (
                    isinstance(value, (float, np.floating)) and np.isnan(value)
                ):
                    return None
                if isinstance(value, np.generic):
                    return value.item()
                return value

            result["audit_features"] = [
                {
                    "source": str(row.source_column),
                    "feature": str(row.feature_name),
                    "source_order": int(row.source_order),
                    "feature_order": int(row.feature_order),
                    "unit_type": str(row.unit_type),
                    "level_value": clean(row.level_value),
                    "ordinal_levels": clean(row.ordinal_levels),
                    "reference_level": clean(row.reference_level),
                }
                for row in audit.itertuples(index=False)
            ]
    return result


def _definition_path(schema: InputSchema) -> Path:
    value = Path(str(schema.feature_universe["definition_file"]))
    return value if value.is_absolute() else (schema.path.parent / value).resolve()


def _validate_feature_universe(
    loaded: LoadedInput, groups: Sequence[SourceGroup]
) -> None:
    definition_path = _definition_path(loaded.schema)
    if not definition_path.exists():
        raise FileNotFoundError(
            f"Feature-universe definition not found: {definition_path}"
        )
    try:
        declared = json.loads(definition_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("feature_universe definition_file is invalid JSON") from exc
    actual = canonical_feature_universe(
        loaded.predictors, groups, loaded.manifest
    )
    if canonical_json(declared) != canonical_json(actual):
        raise ValueError(
            "Resolved feature universe does not match the canonical definition"
        )


def _validate_numeric_predictors(
    frame: pd.DataFrame, predictors: Sequence[str], label: str
) -> None:
    for feature in predictors:
        series = frame[feature]
        if (
            pd.api.types.is_object_dtype(series.dtype)
            or pd.api.types.is_string_dtype(series.dtype)
            or pd.api.types.is_complex_dtype(series.dtype)
            or not pd.api.types.is_numeric_dtype(series.dtype)
        ):
            raise ValueError(f"{label} predictor {feature!r} must be finite numeric")
        values = series.to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(values).any():
            raise ValueError(f"{label} predictor {feature!r} contains ±inf")
        if np.isnan(values).all():
            raise ValueError(f"{label} predictor {feature!r} is entirely missing")


def _outcome_observed(
    frame: pd.DataFrame,
    outcome: str,
    *,
    task: str,
    threshold: float,
    label: str,
) -> pd.DataFrame:
    if outcome not in frame:
        raise KeyError(f"Outcome not found in {label}: {outcome}")
    ratio = float(frame[outcome].isna().mean()) if len(frame) else 1.0
    if ratio > threshold:
        raise ValueError(
            f"{label} outcome missing ratio {ratio:.6f} exceeds {threshold:.6f}"
        )
    observed = frame.dropna(subset=[outcome]).copy()
    if observed.empty:
        raise ValueError(f"{label} has no observed outcome rows")
    values = observed[outcome]
    if task == "regression":
        if (
            pd.api.types.is_object_dtype(values.dtype)
            or pd.api.types.is_string_dtype(values.dtype)
            or pd.api.types.is_complex_dtype(values.dtype)
            or not pd.api.types.is_numeric_dtype(values.dtype)
        ):
            raise ValueError(f"{label} regression outcome must be finite numeric")
        numeric = values.to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError(f"{label} regression outcome contains non-finite values")
    else:
        classes = set(values.unique())
        if not classes or not classes.issubset({0, 1, False, True}):
            raise ValueError(
                f"{label} classification outcome must contain only binary {{0,1}}"
            )
    return observed


def _validate_ids(train: pd.DataFrame, test: pd.DataFrame, id_column: str) -> None:
    for label, frame in (("training data", train), ("test data", test)):
        if frame[id_column].isna().any():
            raise ValueError(f"{label} id_column {id_column!r} contains missing IDs")
        if frame[id_column].duplicated().any():
            raise ValueError(f"{label} id_column {id_column!r} contains duplicate IDs")
    overlap = set(train[id_column]) & set(test[id_column])
    if overlap:
        raise ValueError(
            f"External train/test IDs overlap in {id_column!r}; example={next(iter(overlap))!r}"
        )


def _validate_typed_values(
    frame: pd.DataFrame, groups: Sequence[SourceGroup], label: str
) -> None:
    for group in groups:
        if group.unit_type != "ordinal":
            continue
        observed = set(frame[group.features[0]].dropna().unique())
        legal = set(group.ordinal_levels)
        if not observed.issubset(legal):
            invalid = next(iter(observed - legal))
            raise ValueError(
                f"{label} ordinal source {group.name!r} contains illegal "
                f"level {invalid!r}"
            )


def _categorical_states(frame: pd.DataFrame, group: SourceGroup) -> set[Any]:
    if group.unit_type == "ordinal":
        return set(frame[group.features[0]].dropna().unique())
    if group.unit_type != "onehot_group":
        return set()
    values = frame.loc[:, list(group.features)]
    observed = values.loc[~values.isna().all(axis=1)]
    if observed.empty:
        return set()
    states: set[Any] = set()
    for row in observed.to_numpy(dtype=float):
        if group.drop_first and np.sum(row) == 0:
            states.add(("reference", str(group.reference_level)))
        else:
            states.add(("feature", group.features[int(np.argmax(row))]))
    return states


def _validate_external_category_coverage(
    train: pd.DataFrame,
    test: pd.DataFrame,
    groups: Sequence[SourceGroup],
) -> None:
    for group in groups:
        if group.unit_type not in {"ordinal", "onehot_group"}:
            continue
        unseen = _categorical_states(test, group) - _categorical_states(train, group)
        if unseen:
            raise ValueError(
                f"External test source {group.name!r} contains category states "
                f"absent from train: {sorted(map(str, unseen))[:3]}"
            )


def _validate_row_floor(
    train: pd.DataFrame,
    outcome: str,
    *,
    schema: InputSchema,
    models: Sequence[str],
    min_n: int,
    test_size: float,
    seed: int,
) -> None:
    required = max(
        [int(min_n)]
        + [
            REGRESSION_CV_MIN_N.get(model, 1)
            for model in models
            if schema.task == "regression"
        ]
    )
    if schema.split_mode == "internal_random":
        y = train[outcome]
        try:
            train_index, _ = train_test_split(
                np.arange(len(train)),
                test_size=test_size,
                random_state=seed,
                stratify=y if schema.task == "classification" else None,
            )
        except ValueError as exc:
            raise ValueError(f"Internal split is not feasible: {exc}") from exc
        n_train = len(train_index)
        training_outcomes = y.iloc[train_index]
    else:
        n_train = len(train)
        training_outcomes = train[outcome]
    if n_train < required:
        raise ValueError(
            f"Usable training rows {n_train} are below required minimum {required}"
        )
    if schema.task == "classification":
        folds = 5 if "super_learner" in models else 2
        counts = training_outcomes.value_counts()
        if len(counts) != 2 or int(counts.min()) < folds:
            raise ValueError(
                f"Classification requires both classes with at least {folds} rows each"
            )


def _validate_provenance(schema: InputSchema) -> None:
    provenance_path = schema.table.parent / "provenance.json"
    if not provenance_path.exists():
        return
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("provenance.json is invalid JSON") from exc


def validate_input(
    loaded: LoadedInput,
    outcome: str,
    *,
    models: Sequence[str],
    min_n: int,
    test_size: float,
    seed: int,
) -> tuple[LoadedInput, tuple[SourceGroup, ...]]:
    """Validate all global contracts before sampling creates any model cell."""

    schema = loaded.schema
    groups = source_groups(
        loaded.predictors, loaded.manifest, schema.continuous_priors
    )
    _validate_provenance(schema)
    _validate_feature_universe(loaded, groups)
    _validate_numeric_predictors(loaded.train, loaded.predictors, "training data")
    validate_onehot_states(loaded.train, groups, "training data")
    _validate_typed_values(loaded.train, groups, "training data")
    if schema.split_mode == "external_test":
        assert loaded.test is not None and schema.id_column is not None
        # ID integrity is a whole-table contract and must not depend on which
        # rows survive the selected outcome's missing-value deletion.
        _validate_ids(loaded.train, loaded.test, schema.id_column)
    train = _outcome_observed(
        loaded.train,
        outcome,
        task=schema.task,
        threshold=schema.max_train_outcome_missing_ratio,
        label="training data",
    )
    test = loaded.test
    if schema.split_mode == "external_test":
        assert test is not None and schema.id_column is not None
        _validate_numeric_predictors(test, loaded.predictors, "test data")
        validate_onehot_states(test, groups, "test data")
        _validate_typed_values(test, groups, "test data")
        test = _outcome_observed(
            test,
            outcome,
            task=schema.task,
            threshold=schema.max_test_outcome_missing_ratio,
            label="test data",
        )
        _validate_external_category_coverage(train, test, groups)
        if schema.task == "classification":
            train_classes = set(train[outcome].unique())
            test_classes = set(test[outcome].unique())
            if not test_classes.issubset(train_classes):
                raise ValueError("External test outcome contains a class absent from train")
    _validate_row_floor(
        train,
        outcome,
        schema=schema,
        models=models,
        min_n=min_n,
        test_size=test_size,
        seed=seed,
    )
    return (
        LoadedInput(
            schema=schema,
            train=train,
            test=test,
            predictors=loaded.predictors,
            manifest=loaded.manifest,
        ),
        groups,
    )


__all__ = [
    "REGRESSION_CV_MIN_N",
    "canonical_feature_universe",
    "validate_input",
]
