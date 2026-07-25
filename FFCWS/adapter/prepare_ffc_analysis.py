"""Prepare FFC background predictors for NK-grid experiments."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NEGATIVE_SENTINEL_RE = re.compile(r"^-\d+(?:\.0+)?$")
MISSING_STRINGS = {"", "NA", "NaN", "nan", "NAN", "Na", "na", "<NA>"}


def _safe_identifier(value: object) -> str:
    safe = re.sub(r"[^0-9A-Za-z_]+", "_", str(value)).strip("_")
    if not safe:
        safe = "value"
    if safe[0].isdigit():
        safe = f"v_{safe}"
    return safe


def _safe_source_name(name: str, seen: set[str]) -> str:
    safe = _safe_identifier(name)
    base = safe
    suffix = 2
    while safe in seen:
        safe = f"{base}_{suffix}"
        suffix += 1
    seen.add(safe)
    return safe


def _string_series(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("").str.strip()


def _level_token(value: float) -> str:
    if pd.notna(value) and float(value).is_integer():
        return str(int(value))
    return _safe_identifier(value)


def _negative_token(raw_value: str) -> str:
    numeric = int(float(raw_value))
    return f"neg_{abs(numeric)}"


def _label_for_code(labels: dict[Any, Any], code: float | int | str) -> str:
    candidates: list[Any] = [code, str(code)]
    try:
        numeric = float(code)
        candidates.extend([numeric, int(numeric), str(int(numeric))])
    except (TypeError, ValueError, OverflowError):
        pass
    for candidate in candidates:
        if candidate in labels:
            return _safe_identifier(labels[candidate])
    return ""


def _feature_suffix(level: str, label: str = "") -> str:
    return f"{level}__{label}" if label else level


def _metadata_from_stata(
    path: Path, raw_frame: pd.DataFrame
) -> dict[str, dict[Any, str]]:
    """Reconstruct {column: {code: label}} from raw and labeled Stata reads."""

    labeled_frame = pd.read_stata(path, convert_categoricals=True)
    value_labels: dict[str, dict[Any, str]] = {}
    for column in raw_frame.columns:
        labeled_column = labeled_frame.get(column)
        if labeled_column is None or not isinstance(
            labeled_column.dtype, pd.CategoricalDtype
        ):
            continue
        pairs = (
            pd.DataFrame(
                {"code": raw_frame[column], "label": labeled_column.astype(str)}
            )
            .dropna(subset=["code"])
            .drop_duplicates()
        )
        mapping: dict[Any, str] = {}
        for code, label in zip(pairs["code"], pairs["label"]):
            cleaned = re.sub(r"^-?\d+\s+", "", label).strip()
            mapping[code] = cleaned or label
        if mapping:
            value_labels[column] = mapping
    return value_labels


def _labels_for_source(
    source_column: str,
    *,
    value_labels: dict[str, dict[Any, Any]],
) -> dict[Any, Any]:
    return value_labels.get(source_column, {})


def _binary_feature_row(
    *,
    source_column: str,
    feature_name: str,
    kind: str,
    values: pd.Series,
    min_binary_prevalence: float,
) -> tuple[bool, dict[str, Any]]:
    prevalence = float(values.mean()) if len(values) else 0.0
    keep = min(prevalence, 1.0 - prevalence) >= min_binary_prevalence
    return keep, {
        "source_column": source_column,
        "feature_name": feature_name,
        "kind": kind,
        "prevalence": prevalence,
        "observed_variance": np.nan,
        "keep": bool(keep),
        "reason": "kept" if keep else "below_min_binary_prevalence",
    }


def clean_background_frame(
    frame: pd.DataFrame,
    *,
    id_column: str = "challengeID",
    min_valid_rate: float = 0.5,
    min_numeric_fraction: float = 0.95,
    categorical_max_levels: int = 15,
    min_binary_prevalence: float = 0.01,
    value_labels: dict[str, dict[Any, Any]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return v2 FFC predictors, source manifest, feature manifest, and summary."""

    if id_column not in frame:
        raise KeyError(f"ID column not found: {id_column}")
    if not 0.0 <= min_valid_rate <= 1.0:
        raise ValueError("min_valid_rate must be between 0 and 1")
    if not 0.0 <= min_numeric_fraction <= 1.0:
        raise ValueError("min_numeric_fraction must be between 0 and 1")
    if categorical_max_levels < 1:
        raise ValueError("categorical_max_levels must be positive")
    if not 0.0 <= min_binary_prevalence <= 0.5:
        raise ValueError("min_binary_prevalence must be between 0 and 0.5")

    value_labels = dict(value_labels or {})

    n_rows = len(frame)
    feature_columns: dict[str, pd.Series] = {id_column: frame[id_column]}
    manifest_rows: list[dict[str, Any]] = []
    feature_manifest_rows: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    pre_filter_feature_count = 0

    for source_column in frame.columns:
        if source_column == id_column:
            continue

        raw = _string_series(frame[source_column])
        blank_missing = raw.isin(MISSING_STRINGS)
        negative_missing = raw.str.match(NEGATIVE_SENTINEL_RE, na=False)
        structural_missing = blank_missing | negative_missing
        raw_valid = ~structural_missing
        raw_valid_count = int(raw_valid.sum())
        raw_valid_rate = raw_valid_count / n_rows if n_rows else 0.0

        numeric = pd.to_numeric(raw.mask(structural_missing), errors="coerce")
        numeric_valid = numeric.notna()
        numeric_valid_count = int(numeric_valid.sum())
        numeric_fraction = (
            numeric_valid_count / raw_valid_count if raw_valid_count else 0.0
        )
        distinct = int(numeric[numeric_valid].nunique(dropna=True))
        safe_name = _safe_source_name(source_column, seen_names)
        labels = _labels_for_source(
            source_column,
            value_labels=value_labels,
        )
        has_labels = source_column in value_labels
        generated_count = 0
        status = "dropped"

        if raw_valid_rate < min_valid_rate:
            reason = "below_min_valid_rate"
        elif numeric_fraction < min_numeric_fraction:
            reason = "below_min_numeric_fraction"
        elif distinct <= 1:
            reason = "constant_after_missing"
        else:
            valid_numeric = numeric[numeric_valid]
            all_integer = bool(
                len(valid_numeric) == 0
                or np.isclose(valid_numeric.astype(float) % 1, 0).all()
            )
            is_categorical = (
                distinct <= categorical_max_levels and (has_labels or all_integer)
            )
            if is_categorical:
                status = "categorical"
                reason = "kept"
                encoded_levels: list[tuple[str, pd.Series, float | None]] = []
                for level in sorted(valid_numeric.dropna().unique()):
                    token = _level_token(float(level))
                    label = _label_for_code(labels, level)
                    suffix = _feature_suffix(token, label)
                    encoded_levels.append(
                        (
                            f"C_{safe_name}__{suffix}",
                            (numeric == level).fillna(False).astype(np.int8),
                            float(level),
                        )
                    )
                for raw_code in sorted(raw[negative_missing].unique(), key=lambda x: int(float(x))):
                    token = _negative_token(raw_code)
                    label = _label_for_code(labels, int(float(raw_code)))
                    suffix = _feature_suffix(token, label)
                    encoded_levels.append(
                        (
                            f"C_{safe_name}__{suffix}",
                            (raw == raw_code).astype(np.int8),
                            None,
                        )
                    )
                if bool(blank_missing.any()):
                    encoded_levels.append(
                        (
                            f"C_{safe_name}__missing",
                            blank_missing.astype(np.int8),
                            None,
                        )
                    )
                for feature_name, values, _ in encoded_levels:
                    pre_filter_feature_count += 1
                    keep, row = _binary_feature_row(
                        source_column=source_column,
                        feature_name=feature_name,
                        kind="C",
                        values=values,
                        min_binary_prevalence=min_binary_prevalence,
                    )
                    feature_manifest_rows.append(row)
                    if keep:
                        feature_columns[feature_name] = values
                        generated_count += 1
            else:
                values = numeric.astype(float)
                observed = values.dropna()
                if int(observed.nunique(dropna=True)) <= 1:
                    reason = "constant_after_missing"
                else:
                    status = "numeric"
                    reason = "kept"
                    feature_name = f"X_{safe_name}"
                    pre_filter_feature_count += 1
                    feature_columns[feature_name] = values
                    generated_count += 1
                    feature_manifest_rows.append(
                        {
                            "source_column": source_column,
                            "feature_name": feature_name,
                            "kind": "X",
                            "prevalence": np.nan,
                            "observed_variance": float(observed.var(ddof=0)),
                            "keep": True,
                            "reason": "kept",
                        }
                    )
                    for raw_code in sorted(
                        raw[negative_missing].unique(), key=lambda x: int(float(x))
                    ):
                        token = _negative_token(raw_code)
                        label = _label_for_code(labels, int(float(raw_code)))
                        suffix = _feature_suffix(token, label)
                        indicator_name = f"M_{safe_name}__{suffix}"
                        indicator = (raw == raw_code).astype(np.int8)
                        pre_filter_feature_count += 1
                        keep, row = _binary_feature_row(
                            source_column=source_column,
                            feature_name=indicator_name,
                            kind="M",
                            values=indicator,
                            min_binary_prevalence=min_binary_prevalence,
                        )
                        feature_manifest_rows.append(row)
                        if keep:
                            feature_columns[indicator_name] = indicator
                            generated_count += 1
                    if bool(blank_missing.any()):
                        indicator_name = f"M_{safe_name}__blank"
                        indicator = blank_missing.astype(np.int8)
                        pre_filter_feature_count += 1
                        keep, row = _binary_feature_row(
                            source_column=source_column,
                            feature_name=indicator_name,
                            kind="M",
                            values=indicator,
                            min_binary_prevalence=min_binary_prevalence,
                        )
                        feature_manifest_rows.append(row)
                        if keep:
                            feature_columns[indicator_name] = indicator
                            generated_count += 1

        manifest_rows.append(
            {
                "source_column": source_column,
                "status": status,
                "reason": reason,
                "distinct": distinct,
                "raw_valid_rate": raw_valid_rate,
                "generated_feature_count": generated_count,
                "n_rows": n_rows,
                "blank_missing_count": int(blank_missing.sum()),
                "negative_sentinel_missing_count": int(negative_missing.sum()),
                "raw_valid_count": raw_valid_count,
                "numeric_valid_count": numeric_valid_count,
                "numeric_fraction_among_raw_valid": numeric_fraction,
            }
        )

    features = pd.DataFrame(feature_columns)
    manifest = pd.DataFrame(manifest_rows)
    feature_manifest = pd.DataFrame(feature_manifest_rows)
    predictor_columns = [column for column in features.columns if column != id_column]
    summary = {
        "input_rows": n_rows,
        "input_columns": int(frame.shape[1]),
        "source_categorical_columns": int((manifest["status"] == "categorical").sum()),
        "source_numeric_columns": int((manifest["status"] == "numeric").sum()),
        "source_dropped_columns": int((manifest["status"] == "dropped").sum()),
        "kept_X_columns": sum(column.startswith("X_") for column in predictor_columns),
        "kept_C_columns": sum(column.startswith("C_") for column in predictor_columns),
        "kept_M_columns": sum(column.startswith("M_") for column in predictor_columns),
        "pre_filter_feature_count": int(pre_filter_feature_count),
        "final_feature_count": int(len(predictor_columns)),
        "output_columns": int(features.shape[1]),
        "dropped_feature_count": int(
            0
            if feature_manifest.empty
            else (~feature_manifest["keep"].astype(bool)).sum()
        ),
        "parameters": {
            "min_valid_rate": float(min_valid_rate),
            "min_numeric_fraction": float(min_numeric_fraction),
            "categorical_max_levels": int(categorical_max_levels),
            "min_binary_prevalence": float(min_binary_prevalence),
        },
    }
    return features, manifest, feature_manifest, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean FFC background.dta into numeric NK-grid predictors."
    )
    parser.add_argument(
        "--background",
        default=str(ROOT / "data" / "private" / "background.dta"),
        help="Path to FFC background.dta.",
    )
    parser.add_argument(
        "--out-features",
        default=str(
            ROOT / "data" / "intermediate_files" / "ffc_background_clean_features.csv"
        ),
        help="Output CSV with challengeID and X_/C_/M_ feature columns.",
    )
    parser.add_argument(
        "--out-manifest",
        default=str(
            ROOT / "data" / "intermediate_files" / "ffc_background_clean_manifest.csv"
        ),
        help="Output CSV documenting source-column decisions.",
    )
    parser.add_argument(
        "--out-feature-manifest",
        default=str(
            ROOT
            / "data"
            / "intermediate_files"
            / "ffc_background_clean_feature_manifest.csv"
        ),
        help="Output CSV documenting generated feature keep/drop decisions.",
    )
    parser.add_argument(
        "--out-summary",
        default=str(
            ROOT / "data" / "intermediate_files" / "ffc_background_clean_summary.json"
        ),
        help="Output JSON summary.",
    )
    parser.add_argument("--id-column", default="challengeID")
    parser.add_argument("--min-valid-rate", type=float, default=0.5)
    parser.add_argument("--min-numeric-fraction", type=float, default=0.95)
    parser.add_argument("--categorical-max-levels", type=int, default=15)
    parser.add_argument("--min-binary-prevalence", type=float, default=0.01)
    return parser.parse_args()


def _read_background(path: Path) -> tuple[pd.DataFrame, dict[str, dict[Any, Any]]]:
    if path.suffix.lower() == ".dta":
        frame = pd.read_stata(path, convert_categoricals=False)
        value_labels = _metadata_from_stata(path, frame)
    else:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        value_labels = {}
    return frame, value_labels


def main() -> None:
    args = parse_args()
    background_path = Path(args.background)
    features_path = Path(args.out_features)
    manifest_path = Path(args.out_manifest)
    feature_manifest_path = Path(args.out_feature_manifest)
    summary_path = Path(args.out_summary)

    frame, value_labels = _read_background(background_path)
    features, manifest, feature_manifest, summary = clean_background_frame(
        frame,
        id_column=args.id_column,
        min_valid_rate=args.min_valid_rate,
        min_numeric_fraction=args.min_numeric_fraction,
        categorical_max_levels=args.categorical_max_levels,
        min_binary_prevalence=args.min_binary_prevalence,
        value_labels=value_labels,
    )

    features_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    feature_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(features_path, index=False)
    manifest.to_csv(manifest_path, index=False)
    feature_manifest.to_csv(feature_manifest_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
