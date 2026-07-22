"""Orchestrate shared schema construction and strategy-specific outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

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
    source_manifest_frame,
    validate_cross_strategy_sources,
    validate_feature_manifest,
)
from .common.schema import SchemaConfig, build_shared_schema
from .common.validation import ensure_unique_ids
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
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    document = load_yaml(config_path)
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
    selected = list(strategies or document.get("strategies") or STRATEGIES)
    unknown = [name for name in selected if name not in STRATEGIES]
    if unknown:
        raise ValueError(f"Unknown preprocessing strategy: {', '.join(unknown)}")

    background, value_labels = read_stata_with_labels(background_path)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    ensure_unique_ids(background, id_column, "background")
    ensure_unique_ids(train, id_column, "train")
    ensure_unique_ids(test, id_column, "test")

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
    canonical_sources = validate_cross_strategy_sources(results)

    input_paths = {
        "background": background_path,
        "train": train_path,
        "test": test_path,
    }
    config_hash = stable_hash(document)
    run_summary: dict[str, Any] = {
        "schema_hash": schema.content_hash,
        "canonical_source_count": len(canonical_sources),
        "strategies": {},
    }

    for result in results:
        strategy_dir = output_root / result.strategy
        suffix = ".parquet" if result.strategy == "tree_ordinal" else ".csv"
        features_path = strategy_dir / f"features{suffix}"
        manifest_path = strategy_dir / "feature_manifest.csv"
        qa_path = strategy_dir / "qa_summary.json"
        write_frame(features_path, result.features)
        write_frame(manifest_path, result.feature_manifest)
        write_json(qa_path, result.qa)
        if result.ordinal_mappings:
            write_json(strategy_dir / "ordinal_mappings.json", result.ordinal_mappings)

        outcome_frames, outcome_summary = materialize_outcomes(
            result.features,
            train,
            test,
            outcomes=outcomes,
            id_column=id_column,
        )
        nk_dir = strategy_dir / "nk_inputs"
        for (split, outcome), outcome_frame in outcome_frames.items():
            write_frame(nk_dir / f"ffc_{split}_{outcome}{suffix}", outcome_frame)
        write_frame(strategy_dir / "outcome_summary.csv", outcome_summary)

        metadata = build_metadata(
            strategy=result.strategy,
            schema_hash=schema.content_hash,
            config_hash=config_hash,
            input_paths=input_paths,
            rows=len(result.features),
            columns=result.features.shape[1],
            content_identity={
                "features": frame_hash(result.features),
                "feature_manifest": frame_hash(result.feature_manifest),
                "outcome_summary": frame_hash(outcome_summary),
                "ordinal_mappings": stable_hash(result.ordinal_mappings),
            },
        )
        write_json(strategy_dir / "metadata.json", metadata)
        run_summary["strategies"][result.strategy] = {
            "features": str(features_path),
            "feature_manifest": str(manifest_path),
            "content_identity_hash": metadata["content_identity_hash"],
            "predictor_count": result.features.shape[1] - 1,
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
    args = parser.parse_args(argv)
    summary = run_pipeline(args.config, strategies=args.strategy)
    print(summary)


if __name__ == "__main__":
    main(sys.argv[1:])
