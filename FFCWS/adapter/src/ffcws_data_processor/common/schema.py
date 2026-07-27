"""Build the canonical, outcome-free source schema shared by all strategies."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


MISSING_STRINGS = {"", "NA", "NaN", "nan", "NAN", "Na", "na", "<NA>"}
FFC_MISSING_CODES = tuple(range(-9, 0))
_FFC_MISSING_RE = re.compile(r"^-(?:[1-9])(?:\.0+)?$")


@dataclass(frozen=True)
class SchemaConfig:
    id_column: str = "challengeID"
    min_valid_rate: float = 0.5
    min_numeric_fraction: float = 0.95
    categorical_max_levels: int = 15
    min_binary_prevalence: float = 0.01

    def validate(self) -> None:
        if not 0.0 <= self.min_valid_rate <= 1.0:
            raise ValueError("min_valid_rate must be between 0 and 1")
        if not 0.0 <= self.min_numeric_fraction <= 1.0:
            raise ValueError("min_numeric_fraction must be between 0 and 1")
        if self.categorical_max_levels < 1:
            raise ValueError("categorical_max_levels must be positive")
        if not 0.0 <= self.min_binary_prevalence <= 0.5:
            raise ValueError("min_binary_prevalence must be between 0 and 0.5")


@dataclass
class FeatureDef:
    feature_name: str
    kind: str
    level: float | None = None
    missing_token: str | None = None
    prevalence: float | None = None
    keep: bool = True
    reason: str = "kept"


@dataclass
class SourceSpec:
    source_column: str
    safe_name: str
    source_order: int
    status: str
    reason: str
    eligible: bool
    raw_valid_rate: float
    numeric_fraction_among_raw_valid: float
    distinct: int
    levels: list[float] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)
    observed_features: list[FeatureDef] = field(default_factory=list)
    missing_features: list[FeatureDef] = field(default_factory=list)
    has_blank_missing: bool = False


@dataclass
class SharedSchema:
    id_column: str
    config: dict[str, Any]
    sources: list[SourceSpec]

    @property
    def eligible_sources(self) -> list[SourceSpec]:
        return [source for source in self.sources if source.eligible]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id_column": self.id_column,
            "config": self.config,
            "sources": [asdict(source) for source in self.sources],
        }

    @property
    def content_hash(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def safe_identifier(value: object) -> str:
    safe = re.sub(r"[^0-9A-Za-z_]+", "_", str(value)).strip("_")
    if not safe:
        safe = "value"
    if safe[0].isdigit():
        safe = f"v_{safe}"
    return safe


def safe_source_name(name: str, seen: set[str]) -> str:
    safe = safe_identifier(name)
    base = safe
    suffix = 2
    while safe in seen:
        safe = f"{base}_{suffix}"
        suffix += 1
    seen.add(safe)
    return safe


def string_series(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("").str.strip()


def missing_masks(values: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    raw = string_series(values)
    blank = raw.isin(MISSING_STRINGS)
    coded = raw.str.match(_FFC_MISSING_RE, na=False)
    return raw, blank, coded


def numeric_values(values: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    raw, blank, coded = missing_masks(values)
    numeric = pd.to_numeric(raw.mask(blank | coded), errors="coerce").astype(float)
    return numeric, raw, blank, coded


def level_token(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else safe_identifier(value)


def negative_token(raw_value: str) -> str:
    return f"neg_{abs(int(float(raw_value)))}"


def label_for_code(labels: Mapping[Any, Any], code: float | int | str) -> str:
    candidates: list[Any] = [code, str(code)]
    try:
        numeric = float(code)
        candidates.extend([numeric, int(numeric), str(int(numeric))])
    except (TypeError, ValueError, OverflowError):
        pass
    for candidate in candidates:
        if candidate in labels:
            return safe_identifier(labels[candidate])
    return ""


def feature_suffix(level: str, label: str = "") -> str:
    return f"{level}__{label}" if label else level


def _binary_feature(
    feature_name: str,
    *,
    kind: str,
    values: pd.Series,
    level: float | None = None,
    missing_token: str | None = None,
    min_prevalence: float,
) -> FeatureDef:
    prevalence = float(values.mean()) if len(values) else 0.0
    keep = min(prevalence, 1.0 - prevalence) >= min_prevalence
    return FeatureDef(
        feature_name=feature_name,
        kind=kind,
        level=level,
        missing_token=missing_token,
        prevalence=prevalence,
        keep=bool(keep),
        reason="kept" if keep else "below_min_binary_prevalence",
    )


def _labels_as_strings(labels: Mapping[Any, Any]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in labels.items():
        try:
            numeric = float(key)
            token = str(int(numeric)) if numeric.is_integer() else str(numeric)
        except (TypeError, ValueError, OverflowError):
            token = str(key)
        normalized[token] = str(value)
    return normalized


def build_shared_schema(
    background: pd.DataFrame,
    train_ids: Iterable[Any],
    *,
    value_labels: Mapping[str, Mapping[Any, Any]] | None = None,
    config: SchemaConfig | None = None,
) -> SharedSchema:
    """Infer one outcome-free schema from the official training pool only."""

    config = config or SchemaConfig()
    config.validate()
    if config.id_column not in background:
        raise KeyError(f"ID column not found: {config.id_column}")
    if background[config.id_column].duplicated().any():
        duplicate = background.loc[
            background[config.id_column].duplicated(), config.id_column
        ].iloc[0]
        raise ValueError(f"background has duplicate {config.id_column}: {duplicate}")

    train_id_set = set(train_ids)
    schema_frame = background[background[config.id_column].isin(train_id_set)]
    if schema_frame.empty:
        raise ValueError("No official train IDs matched the background data")

    value_labels = value_labels or {}
    seen_names: set[str] = set()
    sources: list[SourceSpec] = []
    n_rows = len(schema_frame)

    for source_order, source_column in enumerate(
        column for column in background.columns if column != config.id_column
    ):
        values = schema_frame[source_column]
        numeric, raw, blank, coded = numeric_values(values)
        structural_missing = blank | coded
        raw_valid = ~structural_missing
        raw_valid_count = int(raw_valid.sum())
        raw_valid_rate = raw_valid_count / n_rows if n_rows else 0.0
        numeric_valid_count = int(numeric.notna().sum())
        numeric_fraction = (
            numeric_valid_count / raw_valid_count if raw_valid_count else 0.0
        )
        distinct = int(numeric.nunique(dropna=True))
        safe_name = safe_source_name(source_column, seen_names)
        labels = dict(value_labels.get(source_column, {}))
        observed_features: list[FeatureDef] = []
        missing_features: list[FeatureDef] = []
        status = "dropped"
        reason = "dropped"
        eligible = False

        if raw_valid_rate < config.min_valid_rate:
            reason = "below_min_valid_rate"
        elif numeric_fraction < config.min_numeric_fraction:
            reason = "below_min_numeric_fraction"
        elif distinct <= 1:
            reason = "constant_after_missing"
        else:
            observed = numeric.dropna()
            all_integer = bool(
                len(observed) == 0
                or np.isclose(observed.astype(float) % 1, 0).all()
            )
            is_categorical = distinct <= config.categorical_max_levels and (
                bool(labels) or all_integer
            )
            if is_categorical:
                status = "categorical"
                levels = [float(level) for level in sorted(observed.unique())]
                for level in levels:
                    suffix = feature_suffix(
                        level_token(level), label_for_code(labels, level)
                    )
                    observed_features.append(
                        _binary_feature(
                            f"C_{safe_name}__{suffix}",
                            kind="C",
                            values=(numeric == level).fillna(False).astype(np.int8),
                            level=level,
                            min_prevalence=config.min_binary_prevalence,
                        )
                    )
                for raw_code in sorted(
                    raw[coded].unique(), key=lambda item: int(float(item))
                ):
                    suffix = feature_suffix(
                        negative_token(raw_code),
                        label_for_code(labels, int(float(raw_code))),
                    )
                    missing_features.append(
                        _binary_feature(
                            f"C_{safe_name}__{suffix}",
                            kind="C",
                            values=(raw == raw_code).astype(np.int8),
                            missing_token=str(raw_code),
                            min_prevalence=config.min_binary_prevalence,
                        )
                    )
                if bool(blank.any()):
                    missing_features.append(
                        _binary_feature(
                            f"C_{safe_name}__missing",
                            kind="C",
                            values=blank.astype(np.int8),
                            missing_token="blank",
                            min_prevalence=config.min_binary_prevalence,
                        )
                    )
                eligible = any(feature.keep for feature in observed_features)
                reason = "kept" if eligible else "no_kept_observed_level"
            else:
                status = "numeric"
                levels = []
                feature_name = f"X_{safe_name}"
                observed_features.append(
                    FeatureDef(
                        feature_name=feature_name,
                        kind="X",
                        prevalence=None,
                        keep=True,
                        reason="kept",
                    )
                )
                for raw_code in sorted(
                    raw[coded].unique(), key=lambda item: int(float(item))
                ):
                    suffix = feature_suffix(
                        negative_token(raw_code),
                        label_for_code(labels, int(float(raw_code))),
                    )
                    missing_features.append(
                        _binary_feature(
                            f"M_{safe_name}__{suffix}",
                            kind="M",
                            values=(raw == raw_code).astype(np.int8),
                            missing_token=str(raw_code),
                            min_prevalence=config.min_binary_prevalence,
                        )
                    )
                if bool(blank.any()):
                    missing_features.append(
                        _binary_feature(
                            f"M_{safe_name}__blank",
                            kind="M",
                            values=blank.astype(np.int8),
                            missing_token="blank",
                            min_prevalence=config.min_binary_prevalence,
                        )
                    )
                eligible = True
                reason = "kept"

        sources.append(
            SourceSpec(
                source_column=source_column,
                safe_name=safe_name,
                source_order=source_order,
                status=status,
                reason=reason,
                eligible=eligible,
                raw_valid_rate=float(raw_valid_rate),
                numeric_fraction_among_raw_valid=float(numeric_fraction),
                distinct=distinct,
                levels=(
                    [feature.level for feature in observed_features if feature.level is not None]
                    if status == "categorical"
                    else []
                ),
                labels=_labels_as_strings(labels),
                observed_features=observed_features,
                missing_features=missing_features,
                has_blank_missing=bool(blank.any()),
            )
        )

    return SharedSchema(
        id_column=config.id_column,
        config=asdict(config),
        sources=sources,
    )
