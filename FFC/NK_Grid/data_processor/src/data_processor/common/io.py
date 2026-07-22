"""Stable readers, writers, and content identity helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from .validation import ensure_unique_ids


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def frame_hash(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update("\x1f".join(map(str, frame.columns)).encode("utf-8"))
    digest.update("\x1f".join(map(str, frame.dtypes)).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return document


def read_stata_with_labels(
    path: Path,
) -> tuple[pd.DataFrame, dict[str, dict[Any, str]]]:
    path = Path(path)
    raw = pd.read_stata(path, convert_categoricals=False)
    labeled = pd.read_stata(path, convert_categoricals=True)
    value_labels: dict[str, dict[Any, str]] = {}
    for column in raw.columns:
        labeled_column = labeled.get(column)
        if labeled_column is None or not isinstance(
            labeled_column.dtype, pd.CategoricalDtype
        ):
            continue
        pairs = (
            pd.DataFrame({"code": raw[column], "label": labeled_column.astype(str)})
            .dropna(subset=["code"])
            .drop_duplicates()
        )
        mapping: dict[Any, str] = {}
        for code, label in zip(pairs["code"], pairs["label"]):
            cleaned = str(label).strip()
            prefix = str(code)
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip()
            mapping[code] = cleaned or str(label)
        if mapping:
            value_labels[column] = mapping
    return raw, value_labels


def write_json(path: Path, value: Mapping[str, Any] | list[Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str)
    path.write_text(payload + "\n", encoding="utf-8")


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame.to_csv(path, index=False)
    elif suffix in {".parquet", ".pq"}:
        frame.to_parquet(path, index=False, engine="pyarrow")
    else:
        raise ValueError(f"Unsupported output table suffix: {suffix}")


def build_metadata(
    *,
    strategy: str,
    schema_hash: str,
    config_hash: str,
    input_paths: Mapping[str, Path],
    rows: int,
    columns: int,
    content_identity: Mapping[str, Any],
) -> dict[str, Any]:
    identity = {
        "strategy": strategy,
        "schema_hash": schema_hash,
        "config_hash": config_hash,
        "inputs": {name: file_sha256(path) for name, path in input_paths.items()},
        "content": content_identity,
    }
    return {
        **identity,
        "content_identity_hash": stable_hash(identity),
        "rows": int(rows),
        "columns": int(columns),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pandas_version": pd.__version__,
    }


def materialize_outcomes(
    features: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    outcomes: list[str],
    id_column: str,
) -> tuple[dict[tuple[str, str], pd.DataFrame], pd.DataFrame]:
    ensure_unique_ids(features, id_column, "features")
    ensure_unique_ids(train, id_column, "train")
    ensure_unique_ids(test, id_column, "test")
    missing = [
        outcome
        for outcome in outcomes
        if outcome not in train.columns or outcome not in test.columns
    ]
    if missing:
        raise KeyError(f"Outcome(s) missing from train or test: {missing}")

    predictors = [column for column in features.columns if column != id_column]
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    summary_rows: list[dict[str, Any]] = []
    for split, outcomes_frame in (("train", train), ("test", test)):
        for outcome in outcomes:
            merged = outcomes_frame.loc[:, [id_column, outcome]].merge(
                features, on=id_column, how="inner", sort=False, validate="one_to_one"
            )
            output = merged.loc[
                merged[outcome].notna(), [id_column, outcome, *predictors]
            ].reset_index(drop=True)
            frames[(split, outcome)] = output
            summary_rows.append(
                {
                    "split": split,
                    "outcome": outcome,
                    "rows_in_outcome_file": int(len(outcomes_frame)),
                    "rows_after_feature_merge": int(len(merged)),
                    "rows_with_observed_outcome": int(len(output)),
                    "predictor_columns": int(len(predictors)),
                }
            )
    return frames, pd.DataFrame(summary_rows)
