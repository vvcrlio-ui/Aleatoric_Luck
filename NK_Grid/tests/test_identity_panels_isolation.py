from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from aleatoric_nk_grid.ingest import load_schema
from aleatoric_nk_grid.nk_grid import NKGridConfig, run_nk_grid
from aleatoric_nk_grid import run_panels
from aleatoric_nk_grid.run_panels import resolve_panel

from conftest import write_schema_bundle


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_SRC = REPO_ROOT / "NK_Grid" / "src"
MODEL_PARAMS = REPO_ROOT / "NK_Grid" / "model_params.yaml"


def _frame() -> pd.DataFrame:
    x = np.arange(30, dtype=float)
    return pd.DataFrame({"y": 1.0 + 2.0 * x, "X_a": x, "X_b": x % 5})


def _config(schema: Path, out: Path) -> NKGridConfig:
    return NKGridConfig(
        schema=schema,
        out=out,
        outcome="y",
        models=("ols",),
        seed=123,
        test_size=0.3,
        n_seeds=1,
        n_draws=1,
        n_sizes_n=1,
        n_sizes_k=1,
        min_n=10,
        max_n=0,
        max_k=0,
        batch_size=10,
        n_jobs=1,
        model_params=MODEL_PARAMS,
    )


def test_provenance_change_does_not_change_identity_or_duplicate_cells(tmp_path):
    bundle = tmp_path / "bundle"
    schema = write_schema_bundle(
        bundle, _frame(), predictors=["X_a", "X_b"]
    )
    provenance = bundle / "provenance.json"
    provenance.write_text(
        json.dumps({"schema_sha256": "obsolete-adapter-field", "note": "first"}),
        encoding="utf-8",
    )
    out = tmp_path / "result.csv"
    run_nk_grid(_config(schema, out))
    first = pd.read_csv(out)
    provenance.write_text(
        json.dumps({"schema_sha256": "changed-but-ignored", "note": "changed only"}),
        encoding="utf-8",
    )
    run_nk_grid(_config(schema, out))
    second = pd.read_csv(out)
    assert len(second) == 1
    assert second.loc[0, "experiment_id"] == first.loc[0, "experiment_id"]


def test_imputation_semantic_change_rejects_resume_with_field_path(tmp_path):
    schema = write_schema_bundle(
        tmp_path / "bundle", _frame(), predictors=["X_a", "X_b"]
    )
    first_out = tmp_path / "first.csv"
    run_nk_grid(_config(schema, first_out))
    document = json.loads(schema.read_text())
    document["imputation"]["continuous"] = "mean"
    schema.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="semantic_contract.imputation.continuous"):
        run_nk_grid(_config(schema, first_out))


def test_runtime_timeout_change_does_not_change_explicit_identity(tmp_path):
    schema = write_schema_bundle(
        tmp_path / "bundle", _frame(), predictors=["X_a", "X_b"]
    )
    first_config = replace(
        _config(schema, tmp_path / "first.csv"),
        native_process_timeout_seconds=60.0,
    )
    second_config = replace(
        _config(schema, tmp_path / "second.csv"),
        native_process_timeout_seconds=120.0,
    )

    first = run_nk_grid(first_config)
    second = run_nk_grid(second_config)

    assert pd.read_csv(first).loc[0, "experiment_id"] == pd.read_csv(second).loc[0, "experiment_id"]


def test_schema_semantic_contract_excludes_physical_paths(tmp_path):
    first = write_schema_bundle(
        tmp_path / "one", _frame(), predictors=["X_a", "X_b"]
    )
    second = write_schema_bundle(
        tmp_path / "two", _frame(), predictors=["X_a", "X_b"]
    )
    assert load_schema(first).semantic_contract == load_schema(second).semantic_contract


def test_schema_semantic_contract_normalizes_optional_defaults(tmp_path):
    schema = write_schema_bundle(
        tmp_path / "bundle", _frame(), predictors=["X_a", "X_b"]
    )
    explicit_null = load_schema(schema).semantic_contract
    document = json.loads(schema.read_text(encoding="utf-8"))
    document.pop("max_train_outcome_missing_ratio")
    document.pop("max_test_outcome_missing_ratio")
    document.pop("continuous_priors")
    schema.write_text(json.dumps(document), encoding="utf-8")
    omitted = load_schema(schema).semantic_contract
    document["max_train_outcome_missing_ratio"] = 0.5
    document["max_test_outcome_missing_ratio"] = 0.5
    document["continuous_priors"] = {}
    schema.write_text(json.dumps(document), encoding="utf-8")
    explicit_empty = load_schema(schema).semantic_contract
    assert explicit_null == omitted == explicit_empty


def test_explicit_identity_versions_prevent_checkpoint_aliasing(tmp_path):
    first = replace(_config(tmp_path / "schema.json", tmp_path / "one.csv"), data_version="data-v1")
    second = replace(first, data_version="data-v2")
    assert first.data_version != second.data_version


def test_panel_rejects_schema_owned_fields_and_external_test_size(tmp_path):
    train = _frame()
    train.insert(0, "id", np.arange(len(train)))
    test = _frame().iloc[:10].copy()
    test.insert(0, "id", 100 + np.arange(len(test)))
    schema = write_schema_bundle(
        tmp_path / "bundle",
        train,
        test=test,
        split_mode="external_test",
        predictors=["X_a", "X_b"],
        id_column="id",
    )
    base = {
        "name": "bad",
        "schema": str(schema),
        "outcome": "y",
        "models": ["ols"],
        "out": "out.csv",
    }
    with pytest.raises(ValueError, match="schema-owned"):
        resolve_panel({**base, "task": "regression"}, tmp_path)
    with pytest.raises(ValueError, match="must not set test_size"):
        resolve_panel({**base, "test_size": 0.2}, tmp_path)
    _, resolved = resolve_panel(
        {**base, "rerun_completed": False},
        tmp_path,
    )
    assert resolved.rerun_completed is False


def test_panel_allow_large_run_survives_the_cli_default(tmp_path, monkeypatch):
    """An absent --allow-large-run must not silently veto a panel that set it."""

    schema = write_schema_bundle(
        tmp_path / "bundle", _frame(), predictors=["X_a", "X_b"]
    )
    manifest = tmp_path / "panels.yaml"
    manifest.write_text(
        "panels:\n"
        "  - name: big\n"
        f"    schema: {schema}\n"
        "    outcome: y\n"
        "    preset: production\n"
        "    allow_large_run: true\n"
        "    models: [ols]\n"
        "    out: outputs/big.csv\n",
        encoding="utf-8",
    )

    _, config = resolve_panel(
        {
            "name": "big",
            "schema": str(schema),
            "outcome": "y",
            "preset": "production",
            "allow_large_run": True,
            "models": ["ols"],
            "out": "outputs/big.csv",
        },
        tmp_path,
    )
    assert config.allow_large_run is True

    seen: dict[str, object] = {}

    def fake_run(config, **kwargs):
        seen["allow_large_run"] = kwargs.get("allow_large_run")
        return config.out

    monkeypatch.setattr(run_panels, "run_nk_grid", fake_run)
    run_panels.main(["--manifest", str(manifest)])
    # None lets run_nk_grid fall back to the panel's own authorization.
    assert seen["allow_large_run"] is None


def test_unique_package_import_works_from_unrelated_cwd(tmp_path):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(PACKAGE_SRC)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import aleatoric_nk_grid; print(aleatoric_nk_grid.__file__)",
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert str(PACKAGE_SRC) in completed.stdout
