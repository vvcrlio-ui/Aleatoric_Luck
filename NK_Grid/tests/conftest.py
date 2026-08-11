from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from aleatoric_nk_grid.preprocessing import source_groups
from aleatoric_nk_grid.validate_input import canonical_feature_universe


DEFAULT_IMPUTATION = {
    "continuous": "median",
    "ordinal": "most_frequent",
    "onehot_group": "atomic_mode",
    "model_overrides": {},
}


def write_schema_bundle(
    root: Path,
    train: pd.DataFrame,
    *,
    outcome: str = "y",
    task: str = "regression",
    split_mode: str = "internal_random",
    test: pd.DataFrame | None = None,
    predictors: list[str] | None = None,
    predictor_prefix: list[str] | None = None,
    manifest: pd.DataFrame | None = None,
    id_column: str | None = None,
    imputation: dict[str, Any] | None = None,
    max_train_missing: float = 0.5,
    max_test_missing: float = 0.5,
    continuous_priors: dict[str, float] | None = None,
    schema_overrides: dict[str, Any] | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    train_path = root / "train.csv"
    test_path = root / "test.csv"
    manifest_path = root / "feature_manifest.csv"
    definition_path = root / "feature_universe.json"
    train.to_csv(train_path, index=False)
    if test is not None:
        test.to_csv(test_path, index=False)
    if manifest is not None:
        manifest.to_csv(manifest_path, index=False)
    if predictors is None:
        if predictor_prefix is None:
            predictors = [
                column
                for column in train.columns
                if column not in {outcome, id_column}
            ]
        else:
            predictors = [
                column
                for column in train.columns
                if column.startswith(tuple(predictor_prefix))
            ]
    groups = source_groups(predictors, manifest, continuous_priors)
    definition = canonical_feature_universe(predictors, groups, manifest)
    definition_text = (
        json.dumps(
            definition,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    definition_path.write_text(definition_text, encoding="utf-8")
    schema = {
        "schema_version": 1,
        "feature_manifest_version": 1 if manifest is not None else None,
        "dataset": "synthetic",
        "table": "train.csv",
        "test_table": "test.csv" if test is not None else None,
        "split_mode": split_mode,
        "task": task,
        "outcome_columns": [outcome],
        "id_column": id_column,
        "predictor_columns": predictors if predictor_prefix is None else None,
        "predictor_prefix": predictor_prefix,
        "feature_manifest": (
            "feature_manifest.csv" if manifest is not None else None
        ),
        "exchangeable": True,
        "feature_universe": {
            "mode": (
                "fixed_a_priori"
                if split_mode == "internal_random"
                else "train_pool_screened"
            ),
            "definition_file": "feature_universe.json",
        },
        "group_column": None,
        "imputation": imputation or DEFAULT_IMPUTATION,
        "max_train_outcome_missing_ratio": max_train_missing,
        "max_test_outcome_missing_ratio": max_test_missing,
        "continuous_priors": continuous_priors,
    }
    schema.update(schema_overrides or {})
    schema_path = root / "schema.json"
    schema_path.write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return schema_path


def write_repo_schema_bundle(_temporary_root: Path, train: pd.DataFrame, **kwargs) -> Path:
    """Create a disposable schema below the actual repository root.

    Dynamic contracts intentionally reject host-absolute external locators.
    These fixtures live in the ignored pytest cache so the production planner
    can exercise the same canonical repo-relative codec as a real run.
    """

    repo_root = Path(__file__).resolve().parents[2]
    root = repo_root / ".pytest_cache" / "nk-grid-inputs" / uuid.uuid4().hex
    return write_schema_bundle(root, train, **kwargs)


def write_legacy_dynamic_fixture(
    path: Path,
    *,
    table_path: Path,
    panel: str,
    config: Any,
    output_dir: Path,
    workers: int,
    preparation_tmp_dir: Path | str | None = None,
    verification_tmp_dir: Path | str | None = None,
    finalization_tmp_dir: Path | str | None = None,
    **_: object,
) -> Path:
    """Build an old CSV snapshot solely for tests of the retired adapter.

    Production ``write_work_snapshot`` no longer has a compatibility switch;
    keeping this encoder under ``tests/`` prevents a serialized flag from
    re-enabling the retired per-task state machine.
    """

    from aleatoric_nk_grid import flat_task_table as ft

    if workers < 1:
        raise ValueError("workers must be positive")
    table = Path(table_path).resolve()
    try:
        source = pq.ParquetFile(table, memory_map=True)
        ft._validate_task_table_columns(source.schema_arrow.names)
        if source.metadata.num_rows < 1:
            raise ValueError("task table must contain at least one row")
    except ValueError:
        raise
    except (OSError, pa.ArrowInvalid) as exc:
        raise ValueError(f"cannot read task table metadata {table}: {exc}") from exc
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "format_version": ft.TABLE_FORMAT_VERSION,
        "panel": str(panel),
        "task_table": str(table),
        "config": ft._config_to_json(config),
        "output_dir": str(Path(output_dir).resolve()),
        "workers": int(workers),
        "result_store_format": "retired-test-csv-adapter-v1",
    }
    for phase, temporary_directory in (
        ("preparation", preparation_tmp_dir),
        ("verification", verification_tmp_dir),
        ("finalization", finalization_tmp_dir),
    ):
        if temporary_directory is not None:
            payload[phase] = {
                "tmp_dir": str(Path(temporary_directory).expanduser().resolve())
            }
    ft.write_json_atomic(target, payload)
    os.chmod(target, 0o444)
    return target
