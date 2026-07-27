from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aleatoric_nk_grid.ingest import load_input
from aleatoric_nk_grid.validate_input import validate_input
from SMR.adapter.prepare import prepare_smr


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source() -> pd.DataFrame:
    rows = 30
    color = np.arange(rows) % 2
    frame = pd.DataFrame(
        {
            "Cm_lhourlywage": np.linspace(1.0, 2.0, rows),
            "Cm_ltotalincome": np.linspace(2.0, 3.0, rows),
            "Aset_age": np.arange(rows, dtype=float),
            "Aset_color_0": (color == 0).astype(int),
            "Aset_color_1": (color == 1).astype(int),
        }
    )
    frame.loc[0, "Cm_ltotalincome"] = np.nan
    return frame


def _write_contract(path: Path) -> Path:
    contract = {
        "contract_version": 1,
        "dataset": "synthetic_smr",
        "outcome_columns": ["Cm_lhourlywage", "Cm_ltotalincome"],
        "predictor_columns": ["Aset_age", "Aset_color_0", "Aset_color_1"],
        "onehot_groups": {
            "Aset_color": {
                "features": ["Aset_color_0", "Aset_color_1"],
                "level_values": [0, 1],
                "reference_level": 0,
            }
        },
        "missing_value_codes": {},
        "exchangeability_justification": "Synthetic contract test.",
    }
    path.write_text(json.dumps(contract), encoding="utf-8")
    return path


def test_adapter_emits_typed_validated_deterministic_bundle(tmp_path):
    source = tmp_path / "raw.csv"
    _source().to_csv(source, index=False)
    contract = _write_contract(tmp_path / "contract.json")
    article = tmp_path / "article"

    first = prepare_smr(source, article_root=article, contract_path=contract)
    bundle = article / "data" / "ard" / "synthetic_smr"
    tracked = [
        first.schema_path,
        article / "schema" / "synthetic_smr.feature_universe.json",
        bundle / "data.csv",
        bundle / "feature_manifest.csv",
        bundle / "provenance.json",
    ]
    first_hashes = [_hash(path) for path in tracked]
    second = prepare_smr(source, article_root=article, contract_path=contract)
    assert first_hashes == [_hash(path) for path in tracked]
    assert second.predictor_count == 3
    assert second.source_count == 2

    loaded = load_input(first.schema_path, "Cm_ltotalincome")
    assert len(loaded.train) == len(_source())
    assert loaded.train["Cm_ltotalincome"].isna().sum() == 1
    validated, groups = validate_input(
        loaded,
        "Cm_ltotalincome",
        models=("ols",),
        min_n=10,
        test_size=0.3,
        seed=1,
    )
    assert len(validated.train) == len(_source()) - 1
    assert [(group.name, group.unit_type) for group in groups] == [
        ("Aset_age", "continuous"),
        ("Aset_color", "onehot_group"),
    ]


def test_row_perturbation_does_not_change_fixed_universe(tmp_path):
    contract = _write_contract(tmp_path / "contract.json")
    first_source = tmp_path / "first.csv"
    second_source = tmp_path / "second.csv"
    first = _source()
    second = first.copy()
    second["Aset_age"] = second["Aset_age"] * 100 + 7
    first.to_csv(first_source, index=False)
    second.to_csv(second_source, index=False)

    first_result = prepare_smr(
        first_source,
        article_root=tmp_path / "first_article",
        contract_path=contract,
    )
    second_result = prepare_smr(
        second_source,
        article_root=tmp_path / "second_article",
        contract_path=contract,
    )
    first_universe = first_result.schema_path.with_name(
        "synthetic_smr.feature_universe.json"
    )
    second_universe = second_result.schema_path.with_name(
        "synthetic_smr.feature_universe.json"
    )
    assert first_universe.read_bytes() == second_universe.read_bytes()


def test_adapter_rejects_header_drift(tmp_path):
    source = tmp_path / "raw.csv"
    frame = _source()
    frame["Aset_undeclared"] = 1
    frame.to_csv(source, index=False)
    contract = _write_contract(tmp_path / "contract.json")
    with pytest.raises(ValueError, match="undeclared Aset/Bset"):
        prepare_smr(
            source,
            article_root=tmp_path / "article",
            contract_path=contract,
        )
