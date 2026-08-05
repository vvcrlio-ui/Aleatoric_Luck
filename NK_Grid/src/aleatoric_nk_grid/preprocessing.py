"""Source-aware, per-cell preprocessing for the shared engine."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


MANIFEST_COLUMNS = frozenset(
    {
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
    }
)


def _as_bool(value: Any, field: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "false"}:
        return normalized == "true"
    raise ValueError(f"{field} must be true or false, got {value!r}")


def _empty(value: Any) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value)) or str(value) == ""


def _canonical_levels(value: Any, source: str) -> tuple[Any, ...]:
    if _empty(value):
        raise ValueError(f"ordinal source {source!r} requires ordinal_levels")
    try:
        levels = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"ordinal_levels for {source!r} must be a canonical JSON array"
        ) from exc
    if (
        not isinstance(levels, list)
        or not levels
        or any(isinstance(level, (list, dict)) or level is None for level in levels)
    ):
        raise ValueError(
            f"ordinal_levels for {source!r} must be a non-empty scalar array"
        )
    if len({json.dumps(level, sort_keys=True) for level in levels}) != len(levels):
        raise ValueError(f"ordinal_levels for {source!r} contains duplicates")
    if any(
        isinstance(level, bool)
        or not isinstance(level, (int, float))
        or not np.isfinite(float(level))
        for level in levels
    ):
        raise ValueError(
            f"ordinal_levels for {source!r} must contain only finite JSON numbers"
        )
    canonical = json.dumps(
        levels, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    if str(value) != canonical:
        raise ValueError(
            f"ordinal_levels for {source!r} is not canonical JSON; expected {canonical}"
        )
    return tuple(levels)


@dataclass(frozen=True)
class SourceGroup:
    name: str
    features: tuple[str, ...]
    source_order: int
    unit_type: str
    drop_first: bool = False
    reference_feature: str | None = None
    reference_level: Any = None
    level_values: tuple[Any, ...] = ()
    ordinal_levels: tuple[Any, ...] = ()
    source_prior: Any = None


def source_groups(
    predictors: Sequence[str],
    manifest: pd.DataFrame | None,
    continuous_priors: Mapping[str, float] | None = None,
) -> tuple[SourceGroup, ...]:
    """Build ordered source units; no manifest means one continuous source/column."""

    continuous_priors = continuous_priors or {}
    if manifest is None:
        return tuple(
            SourceGroup(
                name=feature,
                features=(feature,),
                source_order=index,
                unit_type="continuous",
                source_prior=float(continuous_priors.get(feature, 0.0)),
            )
            for index, feature in enumerate(predictors)
        )
    missing = sorted(MANIFEST_COLUMNS - set(manifest))
    if missing:
        raise ValueError(f"Feature manifest is missing columns: {missing}")
    normalized = manifest.copy()
    keep = normalized["keep"].map(
        lambda value: _as_bool(value, "manifest.keep")
    )
    for field in ("source_order", "feature_order"):
        normalized[field] = pd.to_numeric(
            normalized[field], errors="raise"
        ).astype(int)
        if (normalized[field] < 0).any():
            raise ValueError(f"manifest.{field} must be non-negative")
    kept_sources = set(normalized.loc[keep, "source_column"].astype(str))
    source_order_owners: dict[int, str] = {}
    for source, all_rows in normalized.groupby("source_column", sort=False):
        source = str(source)
        if source not in kept_sources:
            continue
        orders = all_rows["source_order"].unique()
        if len(orders) != 1:
            raise ValueError(f"source {source!r} has inconsistent source_order")
        order = int(orders[0])
        if order in source_order_owners and source_order_owners[order] != source:
            raise ValueError(
                f"source_order {order} is shared by "
                f"{source_order_owners[order]!r} and {source!r}"
            )
        source_order_owners[order] = source
        if len(all_rows["unit_type"].astype(str).unique()) != 1:
            raise ValueError(f"source {source!r} has inconsistent unit_type")
        drop_values = all_rows["drop_first"].map(
            lambda value: _as_bool(value, "manifest.drop_first")
        )
        if len(drop_values.unique()) != 1:
            raise ValueError(f"source {source!r} has inconsistent drop_first")
        feature_orders = sorted(all_rows["feature_order"].tolist())
        if feature_orders != list(range(len(all_rows))):
            raise ValueError(
                f"source {source!r} feature_order must be unique and contiguous "
                "across kept and audit rows"
            )
    kept = normalized.loc[keep].copy()
    if kept.empty:
        raise ValueError("Feature manifest has no keep=true rows")
    if kept["feature_name"].astype(str).duplicated().any():
        duplicate = kept.loc[
            kept["feature_name"].astype(str).duplicated(), "feature_name"
        ].iloc[0]
        raise ValueError(f"Feature manifest maps predictor more than once: {duplicate}")
    if set(kept["feature_name"].astype(str)) != set(predictors):
        raise ValueError("Feature manifest keep=true rows must exactly cover predictors")
    source_orders: dict[str, int] = {}
    groups: list[SourceGroup] = []
    for source, rows in kept.groupby("source_column", sort=False):
        source = str(source)
        rows = rows.sort_values("feature_order", kind="stable")
        orders = rows["source_order"].unique()
        if len(orders) != 1:
            raise ValueError(f"source {source!r} has inconsistent source_order")
        source_order = int(orders[0])
        if source_order in source_orders.values():
            other = next(
                name for name, order in source_orders.items() if order == source_order
            )
            raise ValueError(
                f"source_order {source_order} is shared by {other!r} and {source!r}"
            )
        source_orders[source] = source_order
        feature_order = rows["feature_order"].tolist()
        if len(feature_order) != len(set(feature_order)):
            raise ValueError(f"source {source!r} has duplicate kept feature_order")
        unit_types = rows["unit_type"].astype(str).unique()
        if len(unit_types) != 1 or unit_types[0] not in {
            "continuous",
            "ordinal",
            "onehot_group",
        }:
            raise ValueError(f"source {source!r} has invalid/inconsistent unit_type")
        unit_type = str(unit_types[0])
        drop_values = rows["drop_first"].map(
            lambda value: _as_bool(value, "manifest.drop_first")
        ).unique()
        if len(drop_values) != 1:
            raise ValueError(f"source {source!r} has inconsistent drop_first")
        drop_first = bool(drop_values[0])
        features = tuple(rows["feature_name"].astype(str))
        prior_values = [value for value in rows["source_prior"] if not _empty(value)]
        if prior_values and len({str(value) for value in prior_values}) != 1:
            raise ValueError(f"source {source!r} has inconsistent source_prior")
        source_prior = prior_values[0] if prior_values else None
        reference_feature = None
        reference_level = None
        level_values: tuple[Any, ...] = ()
        ordinal_levels: tuple[Any, ...] = ()

        if unit_type in {"continuous", "ordinal"}:
            if len(features) != 1:
                raise ValueError(
                    f"{unit_type} source {source!r} must have exactly one kept feature"
                )
            if drop_first:
                raise ValueError(f"{unit_type} source {source!r} cannot use drop_first")
        if unit_type == "continuous":
            if source_prior is None:
                source_prior = float(continuous_priors.get(source, 0.0))
            try:
                source_prior = float(source_prior)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"continuous source_prior for {source!r} must be numeric"
                ) from exc
            if not np.isfinite(source_prior):
                raise ValueError(
                    f"continuous source_prior for {source!r} must be finite"
                )
        elif unit_type == "ordinal":
            level_cells = [
                str(value) for value in rows["ordinal_levels"] if not _empty(value)
            ]
            if len(set(level_cells)) != 1:
                raise ValueError(
                    f"ordinal source {source!r} requires one consistent ordinal_levels"
                )
            ordinal_levels = _canonical_levels(level_cells[0], source)
            if source_prior is None:
                source_prior = ordinal_levels[0]
            if source_prior not in ordinal_levels:
                try:
                    numeric_prior = float(source_prior)
                    matches = [
                        level
                        for level in ordinal_levels
                        if isinstance(level, (int, float))
                        and float(level) == numeric_prior
                    ]
                    source_prior = matches[0] if matches else source_prior
                except (TypeError, ValueError):
                    pass
            if source_prior not in ordinal_levels:
                raise ValueError(
                    f"ordinal source_prior for {source!r} must be in ordinal_levels"
                )
        else:
            references = rows["is_reference"].map(
                lambda value: _as_bool(value, "manifest.is_reference")
            )
            if drop_first:
                if references.any():
                    raise ValueError(
                        f"drop-first source {source!r} must not mark a kept reference feature"
                    )
            elif int(references.sum()) != 1:
                raise ValueError(
                    f"non-drop-first source {source!r} requires exactly one reference"
                )
            if references.any():
                reference_feature = str(rows.loc[references, "feature_name"].iloc[0])
            reference_values = [
                value for value in rows["reference_level"] if not _empty(value)
            ]
            if len({str(value) for value in reference_values}) != 1:
                raise ValueError(
                    f"onehot source {source!r} requires one consistent reference_level"
                )
            reference_level = reference_values[0]
            level_values = tuple(rows["level_value"])
            if any(_empty(value) for value in level_values):
                raise ValueError(
                    f"onehot source {source!r} requires level_value for every feature"
                )
            if len({str(value) for value in level_values}) != len(level_values):
                raise ValueError(
                    f"onehot source {source!r} has duplicate level_value entries"
                )
            if (
                reference_feature is not None
                and str(
                    rows.loc[
                        rows["feature_name"].astype(str).eq(reference_feature),
                        "level_value",
                    ].iloc[0]
                )
                != str(reference_level)
            ):
                raise ValueError(
                    f"onehot source {source!r} reference_level does not match "
                    "the is_reference feature"
                )
        groups.append(
            SourceGroup(
                name=source,
                features=features,
                source_order=source_order,
                unit_type=unit_type,
                drop_first=drop_first,
                reference_feature=reference_feature,
                reference_level=reference_level,
                level_values=level_values,
                ordinal_levels=ordinal_levels,
                source_prior=source_prior,
            )
        )
    groups.sort(key=lambda group: group.source_order)
    return tuple(groups)


def _invalid_onehot_rows(frame: pd.DataFrame, group: SourceGroup) -> pd.Series:
    values = frame.loc[:, list(group.features)]
    any_nan = values.isna().any(axis=1)
    all_nan = values.isna().all(axis=1)
    partial_nan = any_nan & ~all_nan
    numeric = values.fillna(0.0)
    binary = numeric.isin([0, 1]).all(axis=1)
    ones = numeric.sum(axis=1)
    legal_count = ones.le(1) if group.drop_first else ones.eq(1)
    return partial_nan | (~all_nan & (~binary | ~legal_count))


def validate_onehot_states(
    frame: pd.DataFrame, groups: Sequence[SourceGroup], label: str
) -> None:
    for group in groups:
        if group.unit_type != "onehot_group":
            continue
        invalid = _invalid_onehot_rows(frame, group)
        if invalid.any():
            sample = frame.index[invalid][0]
            raise ValueError(
                f"{label} contains an invalid one-hot state for source "
                f"{group.name!r} at row {sample!r}"
            )


def _source_observed(frame: pd.DataFrame, group: SourceGroup) -> pd.Series:
    if group.unit_type == "onehot_group":
        return ~frame.loc[:, list(group.features)].isna().all(axis=1)
    return frame[group.features[0]].notna()


def count_unobserved_sources(
    frame: pd.DataFrame, groups: Sequence[SourceGroup]
) -> int:
    """Count selected sources with no observation in one training cell."""

    return int(sum(not _source_observed(frame, group).any() for group in groups))


def _reference_vector(group: SourceGroup) -> np.ndarray:
    vector = np.zeros(len(group.features), dtype=float)
    if not group.drop_first:
        assert group.reference_feature is not None
        vector[group.features.index(group.reference_feature)] = 1.0
    return vector


def _onehot_state(frame: pd.DataFrame, group: SourceGroup) -> pd.Series:
    values = frame.loc[:, list(group.features)]
    missing = values.isna().all(axis=1)
    state = pd.Series(-1, index=frame.index, dtype=int)
    if group.drop_first:
        state.loc[~missing] = values.loc[~missing].to_numpy().argmax(axis=1)
        all_zero = values.loc[~missing].sum(axis=1).eq(0)
        state.loc[all_zero.index[all_zero]] = len(group.features)
    else:
        state.loc[~missing] = values.loc[~missing].to_numpy().argmax(axis=1)
    return state


def _fill_onehot(
    train: pd.DataFrame,
    test: pd.DataFrame,
    group: SourceGroup,
) -> None:
    features = list(group.features)
    train_missing = train.loc[:, features].isna().all(axis=1)
    test_missing = test.loc[:, features].isna().all(axis=1)
    has_train_missing = bool(train_missing.any())
    has_test_missing = bool(test_missing.any())
    if not has_train_missing and not has_test_missing:
        return
    states = _onehot_state(train.loc[~train_missing], group)
    counts = states.value_counts()
    max_count = counts.max()
    candidates = [int(state) for state in counts[counts.eq(max_count)].index]
    # In a drop-first group, the omitted reference is the first category and
    # therefore precedes all kept dummy feature_order values in a tie.
    mode_state = min(
        candidates,
        key=lambda state: -1 if state == len(group.features) else state,
    )
    if mode_state == len(group.features):
        fill = _reference_vector(group)
    else:
        fill = np.zeros(len(group.features), dtype=float)
        fill[mode_state] = 1.0
    if has_train_missing:
        train.loc[train_missing, features] = fill
    if has_test_missing:
        test.loc[test_missing, features] = fill


def _ordinal_statistic(values: pd.Series, group: SourceGroup, strategy: str) -> Any:
    observed = values.dropna()
    if strategy == "most_frequent":
        counts = observed.value_counts()
        top = set(counts[counts.eq(counts.max())].index)
        return next(level for level in group.ordinal_levels if level in top)
    numeric = pd.to_numeric(observed, errors="raise").to_numpy(dtype=float)
    median = float(np.median(numeric))
    observed_values = set(observed.tolist())
    observed_levels = [
        level for level in group.ordinal_levels if level in observed_values
    ]
    return min(
        observed_levels,
        key=lambda level: (abs(float(level) - median), float(level)),
    )


@dataclass(frozen=True)
class CellPreprocessingResult:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    K_unobserved: int
    passthrough: bool


def _preprocess_cell_reference(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    groups: Sequence[SourceGroup],
    imputation: Mapping[str, Any],
    *,
    model_name: str,
) -> CellPreprocessingResult:
    """Original pandas implementation, retained only as the test oracle."""

    train = X_train.copy()
    test = X_test.copy()
    passthrough = imputation["model_overrides"].get(model_name) == "passthrough"
    unobserved = 0
    for group in groups:
        observed = _source_observed(train, group)
        if not observed.any():
            unobserved += 1
            if passthrough:
                train.loc[:, list(group.features)] = np.nan
                test.loc[:, list(group.features)] = np.nan
            elif group.unit_type == "onehot_group":
                prior = _reference_vector(group)
                train.loc[:, list(group.features)] = prior
                test.loc[:, list(group.features)] = prior
            else:
                train.loc[:, group.features[0]] = group.source_prior
                test.loc[:, group.features[0]] = group.source_prior
            continue
        if passthrough:
            continue
        if group.unit_type == "continuous":
            feature = group.features[0]
            values = train[feature]
            # The engine validates numeric dtypes before any cell runs. Keep
            # the legacy direct-call coercion/error behavior for callers that
            # bypass validation without paying its per-cell cost in production.
            statistic_values = (
                values
                if pd.api.types.is_numeric_dtype(values.dtype)
                else pd.to_numeric(values, errors="raise")
            )
            train_missing = not observed.all()
            test_missing = test[feature].hasnans
            if not train_missing and not test_missing:
                continue
            statistic = (
                float(statistic_values.median())
                if imputation["continuous"] == "median"
                else float(statistic_values.mean())
            )
            if train_missing:
                train[feature] = train[feature].fillna(statistic)
            if test_missing:
                test[feature] = test[feature].fillna(statistic)
        elif group.unit_type == "ordinal":
            feature = group.features[0]
            train_missing = not observed.all()
            test_missing = test[feature].hasnans
            if not train_missing and not test_missing:
                continue
            statistic = _ordinal_statistic(
                train[feature], group, imputation["ordinal"]
            )
            if train_missing:
                train[feature] = train[feature].fillna(statistic)
            if test_missing:
                test[feature] = test[feature].fillna(statistic)
        else:
            _fill_onehot(train, test, group)
    return CellPreprocessingResult(
        X_train=train,
        X_test=test,
        K_unobserved=unobserved,
        passthrough=passthrough,
    )


def _validate_disjoint_group_features(groups: Sequence[SourceGroup]) -> None:
    """Reject input for which delayed writes would change legacy semantics."""

    owners: dict[str, str] = {}
    for group in groups:
        for feature in group.features:
            previous = owners.get(feature)
            if previous is not None:
                raise ValueError(
                    f"source groups {previous!r} and {group.name!r} share feature "
                    f"{feature!r}"
                )
            owners[feature] = group.name


def _supports_vectorized_preprocessing(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> bool:
    """Return whether numeric numpy columns can retain legacy dtype semantics."""

    if not X_train.columns.is_unique or not X_test.columns.is_unique:
        return False
    if not len(X_train.columns) or not len(X_test.columns):
        return False
    return all(
        train_dtype == test_dtype
        and pd.api.types.is_numeric_dtype(train_dtype)
        and not pd.api.types.is_extension_array_dtype(train_dtype)
        for train_dtype, test_dtype in zip(X_train.dtypes, X_test.dtypes)
    )


def _homogeneous_float_frame(frame: pd.DataFrame) -> bool:
    dtypes = tuple(frame.dtypes)
    return bool(dtypes) and all(dtype == dtypes[0] for dtype in dtypes) and (
        pd.api.types.is_float_dtype(dtypes[0])
    )


def _vectorized_onehot_fill(
    train_values: np.ndarray,
    test_values: np.ndarray,
    positions: np.ndarray,
    group: SourceGroup,
) -> None:
    """Apply `_fill_onehot`'s exact state and tie-break rules to ndarrays."""

    train_group = train_values[:, positions]
    test_group = test_values[:, positions]
    train_missing = np.isnan(train_group).all(axis=1)
    test_missing = np.isnan(test_group).all(axis=1)
    if not train_missing.any() and not test_missing.any():
        return

    observed = train_group[~train_missing]
    states = observed.argmax(axis=1)
    if group.drop_first:
        states = states.copy()
        states[observed.sum(axis=1) == 0] = len(group.features)
    counts = np.bincount(states, minlength=len(group.features) + 1)
    max_count = counts.max()
    candidates = np.flatnonzero(counts == max_count)
    mode_state = min(
        (int(state) for state in candidates),
        key=lambda state: -1 if state == len(group.features) else state,
    )
    if mode_state == len(group.features):
        fill = _reference_vector(group)
    else:
        fill = np.zeros(len(group.features), dtype=float)
        fill[mode_state] = 1.0
    if train_missing.any():
        train_values[np.ix_(train_missing, positions)] = fill
    if test_missing.any():
        test_values[np.ix_(test_missing, positions)] = fill


def _preprocess_cell_vectorized(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    groups: Sequence[SourceGroup],
    imputation: Mapping[str, Any],
) -> CellPreprocessingResult:
    """Vectorized imputed preprocessing for homogeneous floating-point frames."""

    train_values = X_train.to_numpy(copy=True)
    test_values = X_test.to_numpy(copy=True)
    positions = {column: index for index, column in enumerate(X_train.columns)}
    unobserved = 0
    for group in groups:
        group_positions = np.fromiter(
            (positions[feature] for feature in group.features), dtype=np.intp
        )
        train_group = train_values[:, group_positions]
        missing = np.isnan(train_group).all(axis=1)
        observed = ~missing
        if not observed.any():
            unobserved += 1
            if group.unit_type == "onehot_group":
                prior = _reference_vector(group)
                train_values[:, group_positions] = prior
                test_values[:, group_positions] = prior
            else:
                feature_position = int(group_positions[0])
                train_values[:, feature_position] = group.source_prior
                test_values[:, feature_position] = group.source_prior
            continue
        if group.unit_type == "continuous":
            feature = group.features[0]
            feature_position = int(group_positions[0])
            train_missing = ~observed
            test_missing = np.isnan(test_values[:, feature_position])
            if not train_missing.any() and not test_missing.any():
                continue
            values = X_train[feature]
            statistic = (
                float(values.median())
                if imputation["continuous"] == "median"
                else float(values.mean())
            )
            if train_missing.any():
                train_values[train_missing, feature_position] = statistic
            if test_missing.any():
                test_values[test_missing, feature_position] = statistic
        elif group.unit_type == "ordinal":
            feature = group.features[0]
            feature_position = int(group_positions[0])
            train_missing = ~observed
            test_missing = np.isnan(test_values[:, feature_position])
            if not train_missing.any() and not test_missing.any():
                continue
            statistic = _ordinal_statistic(
                X_train[feature], group, imputation["ordinal"]
            )
            if train_missing.any():
                train_values[train_missing, feature_position] = statistic
            if test_missing.any():
                test_values[test_missing, feature_position] = statistic
        else:
            _vectorized_onehot_fill(
                train_values, test_values, group_positions, group
            )
    return CellPreprocessingResult(
        X_train=pd.DataFrame(train_values, index=X_train.index, columns=X_train.columns),
        X_test=pd.DataFrame(test_values, index=X_test.index, columns=X_test.columns),
        K_unobserved=unobserved,
        passthrough=False,
    )


def _preprocess_cell_vectorized_mixed(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    groups: Sequence[SourceGroup],
    imputation: Mapping[str, Any],
) -> CellPreprocessingResult:
    """Vectorize writes to float columns while preserving integer columns.

    Integer numpy columns cannot contain NaN, so they need no imputation writes.
    Group-level observation decisions are nevertheless made from every feature,
    which preserves mixed one-hot group semantics.
    """

    train = X_train.copy()
    test = X_test.copy()
    float_columns = [
        column for column, dtype in X_train.dtypes.items()
        if pd.api.types.is_float_dtype(dtype)
    ]
    train_values = {
        column: X_train[column].to_numpy(copy=True) for column in float_columns
    }
    test_values = {
        column: X_test[column].to_numpy(copy=True) for column in float_columns
    }

    def values(column: str, *, is_test: bool = False) -> np.ndarray:
        if column in train_values:
            return (test_values if is_test else train_values)[column]
        return (X_test if is_test else X_train)[column].to_numpy(copy=False)

    unobserved = 0
    for group in groups:
        features = list(group.features)
        train_group = np.column_stack([values(feature) for feature in features])
        missing = pd.isna(train_group).all(axis=1)
        observed = ~missing
        if not observed.any():
            unobserved += 1
            if group.unit_type == "onehot_group":
                fill = _reference_vector(group)
                for feature, value in zip(features, fill):
                    train_values[feature][:] = value
                    test_values[feature][:] = value
            else:
                feature = features[0]
                train_values[feature][:] = group.source_prior
                test_values[feature][:] = group.source_prior
            continue
        if group.unit_type == "continuous":
            feature = features[0]
            if feature not in train_values:
                continue
            train_missing = ~observed
            test_missing = np.isnan(test_values[feature])
            if not train_missing.any() and not test_missing.any():
                continue
            source = X_train[feature]
            statistic = (
                float(source.median())
                if imputation["continuous"] == "median"
                else float(source.mean())
            )
            train_values[feature][train_missing] = statistic
            test_values[feature][test_missing] = statistic
        elif group.unit_type == "ordinal":
            feature = features[0]
            if feature not in train_values:
                continue
            train_missing = ~observed
            test_missing = np.isnan(test_values[feature])
            if not train_missing.any() and not test_missing.any():
                continue
            statistic = _ordinal_statistic(X_train[feature], group, imputation["ordinal"])
            train_values[feature][train_missing] = statistic
            test_values[feature][test_missing] = statistic
        else:
            test_group = np.column_stack(
                [values(feature, is_test=True) for feature in features]
            )
            train_missing = pd.isna(train_group).all(axis=1)
            test_missing = pd.isna(test_group).all(axis=1)
            if not train_missing.any() and not test_missing.any():
                continue
            observed_values = train_group[~train_missing]
            states = observed_values.argmax(axis=1)
            if group.drop_first:
                states = states.copy()
                states[observed_values.sum(axis=1) == 0] = len(features)
            counts = np.bincount(states, minlength=len(features) + 1)
            candidates = np.flatnonzero(counts == counts.max())
            mode = min(
                (int(state) for state in candidates),
                key=lambda state: -1 if state == len(features) else state,
            )
            fill = _reference_vector(group) if mode == len(features) else np.eye(1, len(features), mode)[0]
            for feature, value in zip(features, fill):
                train_values[feature][train_missing] = value
                test_values[feature][test_missing] = value
    for column in float_columns:
        train[column] = train_values[column]
        test[column] = test_values[column]
    train.attrs["_preprocess_vectorized"] = True
    test.attrs["_preprocess_vectorized"] = True
    return CellPreprocessingResult(train, test, unobserved, False)


def preprocess_cell(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    groups: Sequence[SourceGroup],
    imputation: Mapping[str, Any],
    *,
    model_name: str,
) -> CellPreprocessingResult:
    """Fit preprocessing only on one cell and apply the learned state to test."""

    _validate_disjoint_group_features(groups)
    passthrough = imputation["model_overrides"].get(model_name) == "passthrough"
    if passthrough or not _supports_vectorized_preprocessing(X_train, X_test):
        result = _preprocess_cell_reference(
            X_train, X_test, groups, imputation, model_name=model_name
        )
        result.X_train.attrs["_preprocess_vectorized"] = False
        result.X_test.attrs["_preprocess_vectorized"] = False
        return result
    if _homogeneous_float_frame(X_train) and _homogeneous_float_frame(X_test):
        result = _preprocess_cell_vectorized(X_train, X_test, groups, imputation)
        result.X_train.attrs["_preprocess_vectorized"] = True
        result.X_test.attrs["_preprocess_vectorized"] = True
        return result
    return _preprocess_cell_vectorized_mixed(X_train, X_test, groups, imputation)
