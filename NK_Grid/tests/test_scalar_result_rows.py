from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aleatoric_nk_grid.experiment import (
    checkpoint_parts,
    checkpoint_parts_dir,
    write_checkpoint_part as real_write_checkpoint_part,
)
from aleatoric_nk_grid.ingest import load_schema
from aleatoric_nk_grid.nk_grid import (
    ROW_METADATA_FIELDS,
    NKGridConfig,
    run_nk_grid,
)
from aleatoric_nk_grid.preprocessing import source_groups
from aleatoric_nk_grid.validate_input import canonical_feature_universe

from conftest import write_schema_bundle


MODEL_PARAMS = Path(__file__).resolve().parents[1] / "model_params.yaml"
DEFINITION_MARKER = "scalar_result_rows_contract_marker"
ARTIFACT_METADATA_FIELDS = {
    *ROW_METADATA_FIELDS,
    "identity",
    "semantic_contract",
}
NON_SCALAR_TYPES = (Mapping, list, tuple, set)


def _frame(rows: int = 40) -> pd.DataFrame:
    x = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "y": 1.5 * x + (x % 3),
            "X_a": x,
            "X_b": x % 5,
        }
    )


def _audit_manifest(audit_rows: int) -> pd.DataFrame:
    records = [
        {
            "source_column": "source_a",
            "feature_name": "X_a",
            "keep": True,
            "source_order": 0,
            "feature_order": 0,
            "unit_type": "continuous",
            "drop_first": False,
            "is_reference": False,
            "reference_level": np.nan,
            "level_value": np.nan,
            "ordinal_levels": np.nan,
            "source_prior": 0.0,
        },
        {
            "source_column": "source_b",
            "feature_name": "X_b",
            "keep": True,
            "source_order": 1,
            "feature_order": 0,
            "unit_type": "continuous",
            "drop_first": False,
            "is_reference": False,
            "reference_level": np.nan,
            "level_value": np.nan,
            "ordinal_levels": np.nan,
            "source_prior": 0.0,
        },
    ]
    for index in range(audit_rows):
        feature = (
            DEFINITION_MARKER
            if index == 0
            else f"audit_feature_{index:05d}_{'x' * 48}"
        )
        records.append(
            {
                "source_column": f"audit_source_{index:05d}",
                "feature_name": feature,
                "keep": False,
                "source_order": index + 2,
                "feature_order": 0,
                "unit_type": "continuous",
                "drop_first": False,
                "is_reference": False,
                "reference_level": np.nan,
                "level_value": np.nan,
                "ordinal_levels": np.nan,
                "source_prior": 0.0,
            }
        )
    return pd.DataFrame.from_records(records)


def _config(schema: Path, out: Path, *, n_draws: int, batch_size: int) -> NKGridConfig:
    return NKGridConfig(
        schema=schema,
        out=out,
        outcome="y",
        models=("ols",),
        seed=123,
        test_size=0.3,
        n_seeds=1,
        n_draws=n_draws,
        n_sizes_n=1,
        n_sizes_k=1,
        min_n=10,
        max_n=0,
        max_k=0,
        batch_size=batch_size,
        n_jobs=1,
        model_params=MODEL_PARAMS,
    )


def _header(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return next(csv.reader(handle))


def test_large_contract_stays_in_manifest_not_result_rows(tmp_path, monkeypatch):
    bundle = tmp_path / "input"
    schema = write_schema_bundle(
        bundle,
        _frame(),
        predictors=["X_a", "X_b"],
        manifest=_audit_manifest(1_100),
    )
    definition_path = bundle / "feature_universe.json"
    assert definition_path.stat().st_size >= 200 * 1024

    out = tmp_path / "result.csv"
    captured_rows: list[dict] = []

    def capture_rows(rows, out_path):
        materialized = list(rows)
        captured_rows.extend(materialized)
        return real_write_checkpoint_part(materialized, out_path)

    monkeypatch.setattr(
        "aleatoric_nk_grid.nk_grid.write_checkpoint_part",
        capture_rows,
    )
    config = _config(schema, out, n_draws=20, batch_size=20)
    run_nk_grid(
        config,
        stop_after_batch=lambda: True,
        defer_materialization_on_stop=True,
    )

    parts = checkpoint_parts(out)
    assert len(parts) == 1
    part = parts[0]
    part_header = _header(part)
    assert "semantic_contract" not in part_header
    assert "identity" not in part_header
    assert part.read_text(encoding="utf-8").count(DEFINITION_MARKER) == 0
    assert len(captured_rows) == 20
    for row in captured_rows:
        assert set(row) & ARTIFACT_METADATA_FIELDS == set(
            ROW_METADATA_FIELDS
        )
        assert not any(
            isinstance(value, NON_SCALAR_TYPES) for value in row.values()
        )

    partial_manifest = json.loads(
        out.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    expected_identity = partial_manifest["identity"]
    expected_contract = partial_manifest["semantic_contract"]
    schema_contract = load_schema(schema).semantic_contract
    assert expected_contract["feature_universe"] == schema_contract["feature_universe"]

    run_nk_grid(config)

    final_header = _header(out)
    assert "semantic_contract" not in final_header
    assert "identity" not in final_header
    assert {
        "experiment_id",
        "model",
        "seed",
        "draw",
        "N",
        "K",
    }.issubset(final_header)
    assert out.read_text(encoding="utf-8").count(DEFINITION_MARKER) == 0
    final_manifest = json.loads(
        out.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    assert final_manifest["identity"] == expected_identity
    assert final_manifest["semantic_contract"] == expected_contract
    assert not checkpoint_parts_dir(out).exists()


def test_definition_change_with_same_identity_is_rejected_on_resume(
    tmp_path,
):
    bundle = tmp_path / "input"
    manifest = _audit_manifest(1)
    schema = write_schema_bundle(
        bundle,
        _frame(),
        predictors=["X_a", "X_b"],
        manifest=manifest,
    )
    out = tmp_path / "result.csv"
    config = _config(schema, out, n_draws=2, batch_size=1)
    run_nk_grid(
        config,
        stop_after_batch=lambda: True,
        defer_materialization_on_stop=True,
    )
    prior = json.loads(
        out.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )

    manifest.loc[~manifest["keep"], "feature_name"] = "changed_audit_feature"
    manifest.to_csv(bundle / "feature_manifest.csv", index=False)
    definition = canonical_feature_universe(
        ["X_a", "X_b"],
        source_groups(["X_a", "X_b"], manifest),
        manifest,
    )
    (bundle / "feature_universe.json").write_text(
        json.dumps(
            definition,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"semantic_contract\.feature_universe\.definition\.audit_features",
    ):
        run_nk_grid(config)
    current = json.loads(
        out.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    assert current["identity"] == prior["identity"]
