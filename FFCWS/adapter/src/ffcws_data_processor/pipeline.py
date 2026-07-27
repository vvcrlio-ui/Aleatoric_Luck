"""Orchestrate shared schema construction and strategy-specific outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from aleatoric_nk_grid.ingest import load_input
from aleatoric_nk_grid.validate_input import validate_input

from .common.io import (
    build_metadata,
    frame_hash,
    load_yaml,
    materialize_outcomes,
    read_stata_with_labels,
    stable_hash,
    write_frame,
    write_json,
)
from .common.manifests import (
    kept_source_order,
    source_manifest_frame,
    validate_feature_manifest,
)
from .common.schema import FFC_MISSING_CODES, SchemaConfig, build_shared_schema
from .common.validation import ensure_disjoint_ids, ensure_unique_ids
from .contract import enforce_outcome_train_category_coverage, write_engine_schema
from .strategies import STRATEGIES


DEFAULT_OUTCOMES = (
    "gpa",
    "grit",
    "materialHardship",
    "eviction",
    "layoff",
    "jobTraining",
)


def _resolve_path(value: str | Path, config_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_dir / path).resolve()


def _required_mapping(document: dict[str, Any], key: str) -> dict[str, Any]:
    value = document.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration requires a '{key}' mapping")
    return value


def run_pipeline(
    config_path: Path,
    *,
    strategies: Iterable[str] | None = None,
    validation_models: Iterable[str] = ("ols",),
    min_n: int = 10,
    test_size: float = 0.3,
    seed: int = 12345,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    document = load_yaml(config_path)
    if document.get("contract_version") != "ffcws-adapter-v1":
        raise ValueError("Unsupported FFCWS contract_version")
    if document.get("split_mode") != "external_test":
        raise ValueError("FFCWS contract requires split_mode=external_test")
    if document.get("feature_universe_mode") != "train_pool_screened":
        raise ValueError(
            "FFCWS contract requires feature_universe_mode=train_pool_screened"
        )
    if tuple(document.get("missing_value_codes", ())) != FFC_MISSING_CODES:
        raise ValueError(
            f"FFCWS contract missing_value_codes must be {list(FFC_MISSING_CODES)}"
        )
    justification = str(document.get("exchangeability_justification", "")).strip()
    if not justification:
        raise ValueError("FFCWS contract requires an exchangeability justification")
    paths = _required_mapping(document, "paths")
    schema_document = dict(document.get("schema") or {})
    id_column = str(document.get("id_column", "challengeID"))
    outcomes = [str(item) for item in document.get("outcomes", DEFAULT_OUTCOMES)]
    unknown_threshold = float(document.get("unknown_rate_threshold", 0.95))
    if not 0.0 <= unknown_threshold <= 1.0:
        raise ValueError("unknown_rate_threshold must be between 0 and 1")

    config_dir = config_path.parent
    background_path = _resolve_path(paths["background"], config_dir)
    train_path = _resolve_path(paths["train"], config_dir)
    test_path = _resolve_path(paths["test"], config_dir)
    output_root = _resolve_path(paths["output_root"], config_dir)
    ard_root = _resolve_path(paths.get("ard_root", output_root / "ard"), config_dir)
    schema_root = _resolve_path(
        paths.get("schema_root", output_root / "schemas"), config_dir
    )
    selected = list(strategies or document.get("strategies") or STRATEGIES)
    selected_validation_models = tuple(validation_models)
    if not selected_validation_models:
        raise ValueError("At least one validation model is required")
    unknown = [name for name in selected if name not in STRATEGIES]
    if unknown:
        raise ValueError(f"Unknown preprocessing strategy: {', '.join(unknown)}")

    background, value_labels = read_stata_with_labels(background_path)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    ensure_unique_ids(background, id_column, "background")
    ensure_unique_ids(train, id_column, "train")
    ensure_unique_ids(test, id_column, "test")
    ensure_disjoint_ids(train, test, id_column=id_column)

    schema_config = SchemaConfig(id_column=id_column, **schema_document)
    schema = build_shared_schema(
        background,
        train[id_column],
        value_labels=value_labels,
        config=schema_config,
    )
    source_manifest = source_manifest_frame(schema)
    output_root.mkdir(parents=True, exist_ok=True)
    write_frame(output_root / "source_manifest.csv", source_manifest)
    write_json(output_root / "schema.json", schema.to_dict())

    results = []
    for strategy in selected:
        result = STRATEGIES[strategy](
            background,
            schema,
            test_ids=test[id_column],
            unknown_rate_threshold=unknown_threshold,
        )
        validate_feature_manifest(
            result.features, result.feature_manifest, id_column=id_column
        )
        results.append(result)
    input_paths = {
        "background": background_path,
        "train": train_path,
        "test": test_path,
    }
    config_hash = stable_hash(document)
    run_summary: dict[str, Any] = {
        "schema_hash": schema.content_hash,
        "eligible_raw_source_count": len(schema.eligible_sources),
        "strategies": {},
    }

    for result in results:
        strategy_dir = output_root / result.strategy
        suffix = ".parquet"
        features_path = strategy_dir / f"features{suffix}"
        stale_features_path = strategy_dir / "features.csv"
        if stale_features_path.exists():
            stale_features_path.unlink()
        manifest_path = strategy_dir / "feature_manifest.csv"
        qa_path = strategy_dir / "qa_summary.json"
        write_frame(features_path, result.features)
        write_frame(manifest_path, result.feature_manifest)
        persisted_manifest = pd.read_csv(manifest_path)
        if result.ordinal_mappings:
            write_json(strategy_dir / "ordinal_mappings.json", result.ordinal_mappings)

        outcome_frames, outcome_summary = materialize_outcomes(
            result.features,
            train,
            test,
            outcomes=outcomes,
            id_column=id_column,
        )
        outcome_frames, outcome_category_coverage = (
            enforce_outcome_train_category_coverage(
                outcome_frames,
                persisted_manifest,
                unknown_rate_threshold=unknown_threshold,
            )
        )
        write_json(
            qa_path,
            {
                **result.qa,
                "outcome_category_coverage": (
                    outcome_category_coverage.to_dict(orient="records")
                ),
            },
        )
        write_frame(
            strategy_dir / "outcome_category_coverage.csv",
            outcome_category_coverage,
        )
        write_frame(strategy_dir / "outcome_summary.csv", outcome_summary)

        engine_schemas: dict[str, str] = {}
        for outcome in outcomes:
            dataset = f"ffc_{result.strategy}_{outcome}"
            dataset_dir = ard_root / dataset
            train_ard = dataset_dir / f"data{suffix}"
            test_ard = dataset_dir / f"test{suffix}"
            for stale_name in ("data.csv", "test.csv"):
                stale_path = dataset_dir / stale_name
                if stale_path.exists():
                    stale_path.unlink()
            manifest_ard = dataset_dir / "feature_manifest.csv"
            write_frame(train_ard, outcome_frames[("train", outcome)])
            write_frame(test_ard, outcome_frames[("test", outcome)])
            write_frame(manifest_ard, persisted_manifest)
            schema_path = write_engine_schema(
                schema_root=schema_root,
                dataset_dir=dataset_dir,
                dataset=dataset,
                outcome=outcome,
                table_path=train_ard,
                test_path=test_ard,
                manifest_path=manifest_ard,
                manifest=persisted_manifest,
                id_column=id_column,
                feature_universe_stem=f"ffc_{result.strategy}",
            )
            loaded = load_input(schema_path, outcome)
            validate_input(
                loaded,
                outcome,
                models=selected_validation_models,
                min_n=min_n,
                test_size=test_size,
                seed=seed,
            )
            engine_schemas[outcome] = str(schema_path)

        metadata = build_metadata(
            strategy=result.strategy,
            schema_hash=schema.content_hash,
            config_hash=config_hash,
            input_paths=input_paths,
            rows=len(result.features),
            columns=result.features.shape[1],
            content_identity={
                "features": frame_hash(result.features),
                "feature_manifest": frame_hash(persisted_manifest),
                "outcome_summary": frame_hash(outcome_summary),
                "outcome_category_coverage": frame_hash(
                    outcome_category_coverage
                ),
                "ordinal_mappings": stable_hash(result.ordinal_mappings),
            },
        )
        write_json(strategy_dir / "metadata.json", metadata)
        run_summary["strategies"][result.strategy] = {
            "features": str(features_path),
            "feature_manifest": str(manifest_path),
            "content_identity_hash": metadata["content_identity_hash"],
            "predictor_count": result.features.shape[1] - 1,
            "source_count": len(kept_source_order(persisted_manifest)),
            "engine_schemas": engine_schemas,
        }

    write_json(output_root / "run_summary.json", run_summary)
    return run_summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build leak-free FFC encoding strategies for NK Grid."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--strategy",
        nargs="+",
        choices=tuple(STRATEGIES),
        default=None,
        help="One or more strategies; defaults to the config list.",
    )
    parser.add_argument("--validation-model", nargs="+", default=["ols"])
    parser.add_argument("--min-n", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args(argv)
    summary = run_pipeline(
        args.config,
        strategies=args.strategy,
        validation_models=args.validation_model,
        min_n=args.min_n,
        test_size=args.test_size,
        seed=args.seed,
    )
    print(summary)


if __name__ == "__main__":
    main(sys.argv[1:])
