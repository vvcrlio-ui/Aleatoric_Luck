"""Build the FFCWS material-hardship inputs: one schema per observation landmark."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Sequence

import numpy as np
import pandas as pd

from aleatoric_nk_grid.ingest import canonical_json, load_input
from aleatoric_nk_grid.input_publication import publish_validated_inputs
from aleatoric_nk_grid.preprocessing import source_groups
from aleatoric_nk_grid.validate_input import canonical_feature_universe, validate_input


ADAPTER_VERSION = "ffcws-icpsr-hardship-adapter-v1"
DEFAULT_CONTRACT = "hardship.json"
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
    schema_paths: dict[int, Path]
    source_counts: dict[int, int]
    sample_size: int


def read_columns(source: Path, columns: Sequence[str]) -> pd.DataFrame:
    """Read lower-case column names from the ICPSR Stata file, codes unconverted."""
    with pd.io.stata.StataReader(source) as reader:
        original = {name.lower(): name for name in reader.variable_labels()}
    missing = [column for column in columns if column not in original]
    if missing:
        raise KeyError(f"Columns absent from the source file: {missing[:10]}")
    frame = pd.read_stata(
        source, columns=[original[column] for column in columns], convert_categoricals=False
    )
    frame.columns = list(columns)
    return frame.apply(pd.to_numeric, errors="coerce")


def numeric_values(series: pd.Series) -> pd.Series:
    """Negative integer codes are FFCWS missing-value codes; other values are kept."""
    integer_negative = (series < 0) & (series == np.round(series))
    return series.mask(integer_negative).astype(float)


def onehot_values(series: pd.Series, levels: Sequence[float]) -> np.ndarray:
    """Dummy columns for labelled categories; any other value makes the row missing."""
    known = series.isin(levels).to_numpy()
    values = (series.to_numpy()[:, None] == np.asarray(levels, dtype=float)[None, :]).astype(float)
    values[~known] = np.nan
    return values


def hardship_scores(frame: pd.DataFrame, outcomes: dict[str, Any]) -> pd.DataFrame:
    """Share of the 11 hardships reported (1 = yes, 2 = no); any other answer leaves it missing."""
    scores = {}
    for name, definition in outcomes.items():
        block = frame[definition["items"]]
        complete = block.isin([1, 2]).all(axis=1)
        scores[name] = block.eq(1).sum(axis=1).div(len(definition["items"])).where(complete)
    return pd.DataFrame(scores, index=frame.index)


def load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(Path(path).read_text(encoding="utf-8"))
    if contract.get("contract_version") != 1:
        raise ValueError("FFCWS ICPSR hardship contract_version must be 1")
    return contract


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _encode_landmark(
    sample: pd.DataFrame, sources: Sequence[dict[str, Any]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for order, source in enumerate(sources):
        name = source["name"]
        levels = source["levels"]
        if levels is None:
            columns[name] = numeric_values(sample[name]).to_numpy()
            rows.append({
                "source_column": name, "feature_name": name, "keep": True,
                "source_order": order, "feature_order": 0, "unit_type": "continuous",
                "drop_first": False, "is_reference": False, "reference_level": None,
                "level_value": None, "ordinal_levels": None, "source_prior": None,
            })
            continue
        dummies = onehot_values(sample[name], levels)
        for position, level in enumerate(levels):
            feature = f"{name}__{position}"
            columns[feature] = dummies[:, position]
            rows.append({
                "source_column": name, "feature_name": feature, "keep": True,
                "source_order": order, "feature_order": position, "unit_type": "onehot_group",
                "drop_first": False, "is_reference": position == 0,
                "reference_level": levels[0], "level_value": level,
                "ordinal_levels": None, "source_prior": None,
            })
    return pd.DataFrame(columns, index=sample.index), pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def build_adapter(
    source: Path,
    *,
    article_root: Path,
    contract_path: Path,
    validation_models: Sequence[str] = ("ols",),
    min_n: int = 10,
    test_size: float = 0.3,
    seed: int = 12345,
) -> AdapterResult:
    """Validate every landmark bundle in staging, then publish them together."""
    article_root = Path(article_root).resolve()
    article_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".adapter-staging-", dir=article_root) as temporary:
        staged = _build_staged(
            Path(source).resolve(), staging=Path(temporary),
            contract_path=Path(contract_path).resolve(),
            validation_models=tuple(validation_models), min_n=min_n,
            test_size=test_size, seed=seed,
        )
        published = publish_validated_inputs(
            list(staged.schema_paths.values()), schema_root=article_root / "schema",
            ard_root=article_root / "data" / "ard",
        )
        return AdapterResult(
            schema_paths={t: published[path] for t, path in staged.schema_paths.items()},
            source_counts=staged.source_counts, sample_size=staged.sample_size,
        )


def _build_staged(
    source: Path,
    *,
    staging: Path,
    contract_path: Path,
    validation_models: Sequence[str],
    min_n: int,
    test_size: float,
    seed: int,
) -> AdapterResult:
    contract = load_contract(contract_path)
    id_column = contract["id_column"]
    outcomes = contract["outcomes"]
    sources = contract["sources"]
    items = [item for definition in outcomes.values() for item in definition["items"]]
    needed = list(dict.fromkeys([id_column, *items, *(s["name"] for s in sources)]))
    frame = read_columns(source, needed)
    if frame[id_column].isna().any() or not frame[id_column].is_unique:
        raise ValueError(f"{id_column} must be non-missing and unique")

    scores = hardship_scores(frame, outcomes)
    in_sample = scores.notna().all(axis=1)
    if int(in_sample.sum()) != contract["sample_size"]:
        raise ValueError(
            f"Families with every hardship score: {int(in_sample.sum())}; "
            f"the contract was built for {contract['sample_size']}"
        )
    sample = frame.loc[in_sample].reset_index(drop=True)
    scores = scores.loc[in_sample].reset_index(drop=True)

    schema_dir = staging / "schema"
    schema_paths: dict[int, Path] = {}
    source_counts: dict[int, int] = {}
    for landmark in sorted(int(value) for value in contract["landmarks"]):
        dataset = f"{contract['dataset_prefix']}_lm{landmark}"
        landmark_sources = [s for s in sources if s["landmark"] <= landmark]
        landmark_outcomes = [name for name, d in outcomes.items() if d["age"] > landmark]
        predictors, manifest = _encode_landmark(sample, landmark_sources)
        ard = pd.concat(
            [sample[[id_column]].astype("int64"), scores[landmark_outcomes], predictors], axis=1
        )
        ard_dir = staging / "data" / "ard" / dataset
        ard_dir.mkdir(parents=True)
        table_path = ard_dir / "data.parquet"
        ard.to_parquet(table_path, index=False)
        manifest_path = ard_dir / "feature_manifest.csv"
        manifest.to_csv(manifest_path, index=False)
        persisted = pd.read_csv(manifest_path)
        predictor_columns = list(predictors.columns)
        groups = source_groups(predictor_columns, persisted, {})
        definition_path = schema_dir / f"{dataset}.feature_universe.json"
        definition_path.parent.mkdir(parents=True, exist_ok=True)
        definition_path.write_text(
            canonical_json(canonical_feature_universe(predictor_columns, groups, persisted)),
            encoding="utf-8",
        )
        schema_path = schema_dir / f"{dataset}.json"
        _write_json(schema_path, {
            "schema_version": 1,
            "feature_manifest_version": 1,
            "dataset": dataset,
            "table": os.path.relpath(table_path, schema_dir),
            "test_table": None,
            "split_mode": "internal_random",
            "task": "regression",
            "outcome_columns": landmark_outcomes,
            "id_column": id_column,
            "predictor_columns": predictor_columns,
            "predictor_prefix": None,
            "feature_manifest": os.path.relpath(manifest_path, schema_dir),
            "exchangeable": True,
            "feature_universe": {"mode": "fixed_a_priori", "definition_file": definition_path.name},
            "group_column": None,
            "imputation": {
                "continuous": "median",
                "ordinal": "most_frequent",
                "onehot_group": "atomic_mode",
                "model_overrides": {"lightgbm": "passthrough", "xgboost": "passthrough"},
            },
            "max_train_outcome_missing_ratio": 0.5,
            "max_test_outcome_missing_ratio": 0.5,
            "continuous_priors": None,
        })
        _write_json(ard_dir / "provenance.json", {
            "adapter": "ffcws_icpsr_hardship",
            "adapter_version": ADAPTER_VERSION,
            "dataset": dataset,
            "landmark": landmark,
            "sample_size": len(sample),
        })
        for outcome in landmark_outcomes:
            validate_input(
                load_input(schema_path, outcome), outcome, models=tuple(validation_models),
                min_n=min_n, test_size=test_size, seed=seed, require_id=True,
            )
        schema_paths[landmark] = schema_path
        source_counts[landmark] = len(groups)
    return AdapterResult(schema_paths=schema_paths, source_counts=source_counts, sample_size=len(sample))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--article-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path, help="Write prepared schemas and data under this directory")
    parser.add_argument("--source", type=Path, required=True, help="ICPSR 31622 DS0001 Stata file")
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--validation-model", nargs="+", default=["ols"])
    parser.add_argument("--min-n", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args(argv)
    article_root = args.article_root.resolve()
    result = build_adapter(
        args.source,
        article_root=args.output_root or article_root,
        contract_path=args.contract or article_root / "adapter" / "config" / DEFAULT_CONTRACT,
        validation_models=args.validation_model,
        min_n=args.min_n,
        test_size=args.test_size,
        seed=args.seed,
    )
    for landmark, path in result.schema_paths.items():
        print(f"OK: lm{landmark} schema={path} sources={result.source_counts[landmark]}")
    print(f"families={result.sample_size}")


if __name__ == "__main__":
    main()
