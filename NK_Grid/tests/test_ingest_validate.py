from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from aleatoric_nk_grid.ingest import load_input, load_schema
from aleatoric_nk_grid.validate_input import validate_input

from conftest import write_schema_bundle


def _frame(rows: int = 20) -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "id": np.arange(rows),
            "y": 1.0 + 2.0 * index,
            "X_a": index,
            "X_b": index % 3,
            "large_unused_object": ["not projected"] * rows,
        }
    )


def _validated(schema_path, **kwargs):
    loaded = load_input(schema_path, "y")
    return validate_input(
        loaded,
        "y",
        models=kwargs.get("models", ("ols",)),
        min_n=kwargs.get("min_n", 10),
        test_size=kwargs.get("test_size", 0.3),
        seed=123,
    )


def test_projection_excludes_unrequested_columns(tmp_path):
    schema_path = write_schema_bundle(
        tmp_path,
        _frame(),
        predictors=["X_a", "X_b"],
    )
    loaded, groups = _validated(schema_path)
    assert loaded.train.columns.tolist() == ["y", "X_a", "X_b"]
    assert [group.name for group in groups] == ["X_a", "X_b"]


def test_predictor_columns_and_prefix_are_mutually_exclusive(tmp_path):
    schema_path = write_schema_bundle(
        tmp_path,
        _frame(),
        predictors=["X_a", "X_b"],
        schema_overrides={"predictor_prefix": ["X_"]},
    )
    with pytest.raises(ValueError, match="exactly one"):
        load_schema(schema_path)


def test_predictor_prefix_cannot_capture_current_outcome(tmp_path):
    frame = _frame().rename(columns={"y": "f_outcome", "X_a": "f_a"})
    schema_path = write_schema_bundle(
        tmp_path,
        frame,
        outcome="f_outcome",
        predictor_prefix=["f_"],
    )
    with pytest.raises(ValueError, match="protected outcome/ID"):
        load_input(schema_path, "f_outcome")


def test_predictor_prefix_cannot_capture_another_declared_outcome(tmp_path):
    frame = _frame().rename(columns={"X_a": "f_a"})
    frame["f_future_outcome"] = frame["y"] + 1.0
    schema_path = write_schema_bundle(
        tmp_path,
        frame,
        predictor_prefix=["f_"],
        schema_overrides={"outcome_columns": ["y", "f_future_outcome"]},
    )
    with pytest.raises(ValueError, match="protected outcome/ID"):
        load_input(schema_path, "y")


def test_explicit_predictors_cannot_include_any_declared_outcome(tmp_path):
    frame = _frame()
    frame["future_outcome"] = frame["y"] + 1.0
    schema_path = write_schema_bundle(
        tmp_path,
        frame,
        predictors=["X_a", "future_outcome"],
        schema_overrides={"outcome_columns": ["y", "future_outcome"]},
    )
    with pytest.raises(ValueError, match="protected outcome/ID"):
        load_input(schema_path, "y")


def test_explicit_predictors_cannot_include_current_outcome(tmp_path):
    schema_path = write_schema_bundle(
        tmp_path,
        _frame(),
        predictors=["y", "X_a"],
    )
    with pytest.raises(ValueError, match="protected outcome/ID"):
        load_input(schema_path, "y")


@pytest.mark.parametrize("rule", ["explicit", "prefix"])
def test_id_column_cannot_be_a_predictor(tmp_path, rule):
    frame = _frame()
    kwargs = (
        {"predictors": ["id", "X_a"]}
        if rule == "explicit"
        else {"predictor_prefix": ["id"]}
    )
    schema_path = write_schema_bundle(
        tmp_path,
        frame,
        id_column="id",
        **kwargs,
    )
    with pytest.raises(ValueError, match="protected outcome/ID"):
        load_input(schema_path, "y")


@pytest.mark.parametrize("version_field,value", [("schema_version", 2), ("feature_manifest_version", 2)])
def test_unknown_versions_fail(tmp_path, version_field, value):
    manifest = None
    if version_field == "feature_manifest_version":
        manifest = pd.DataFrame(
            {
                "source_column": ["a", "b"],
                "feature_name": ["X_a", "X_b"],
                "keep": [True, True],
                "source_order": [0, 1],
                "feature_order": [0, 0],
                "unit_type": ["continuous", "continuous"],
                "drop_first": [False, False],
                "is_reference": [False, False],
                "reference_level": [np.nan, np.nan],
                "level_value": [np.nan, np.nan],
                "ordinal_levels": [np.nan, np.nan],
                "source_prior": [0.0, 0.0],
            }
        )
    schema_path = write_schema_bundle(
        tmp_path,
        _frame(),
        predictors=["X_a", "X_b"],
        manifest=manifest,
        schema_overrides={version_field: value},
    )
    with pytest.raises(ValueError, match=f"Unsupported {version_field}"):
        load_schema(schema_path)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda frame: frame.assign(X_a="bad"), "predictor 'X_a'.*numeric"),
        (lambda frame: frame.assign(X_a=np.inf), "contains ±inf"),
        (lambda frame: frame.assign(X_a=np.nan), "entirely missing"),
        (lambda frame: frame.assign(y=np.inf), "non-finite"),
        (lambda frame: frame.assign(y="bad"), "outcome must be finite numeric"),
    ],
)
def test_numeric_contract_failures_happen_before_sampling(tmp_path, mutation, match):
    schema_path = write_schema_bundle(
        tmp_path, mutation(_frame()), predictors=["X_a", "X_b"]
    )
    with pytest.raises(ValueError, match=match):
        _validated(schema_path)


def test_binary_only_contract(tmp_path):
    frame = _frame()
    frame["y"] = np.arange(len(frame)) % 3
    schema_path = write_schema_bundle(
        tmp_path,
        frame,
        predictors=["X_a", "X_b"],
        task="classification",
    )
    with pytest.raises(ValueError, match=r"binary \{0,1\}"):
        _validated(schema_path)


def test_outcome_missing_threshold_and_uniform_deletion(tmp_path):
    frame = _frame()
    frame.loc[:10, "y"] = np.nan
    schema_path = write_schema_bundle(
        tmp_path,
        frame,
        predictors=["X_a", "X_b"],
        max_train_missing=0.5,
    )
    with pytest.raises(ValueError, match="missing ratio"):
        _validated(schema_path)

    schema_path = write_schema_bundle(
        tmp_path / "allowed",
        frame,
        predictors=["X_a", "X_b"],
        max_train_missing=0.6,
    )
    loaded, _ = _validated(schema_path, min_n=5)
    assert loaded.train["y"].notna().all()
    assert len(loaded.train) == 9


@pytest.mark.parametrize(
    "case,match",
    [
        ("overlap", "overlap"),
        ("duplicate", "duplicate"),
        ("missing", "missing IDs"),
    ],
)
def test_external_id_contract(tmp_path, case, match):
    train = _frame(20)
    test = _frame(10)
    test["id"] += 100
    if case == "overlap":
        test.loc[0, "id"] = train.loc[0, "id"]
    elif case == "duplicate":
        test.loc[1, "id"] = test.loc[0, "id"]
    else:
        test.loc[0, "id"] = np.nan
    schema_path = write_schema_bundle(
        tmp_path,
        train,
        test=test,
        split_mode="external_test",
        predictors=["X_a", "X_b"],
        id_column="id",
    )
    with pytest.raises(ValueError, match=match):
        _validated(schema_path)


@pytest.mark.parametrize(
    "case,match",
    [
        ("train_duplicate", "duplicate"),
        ("test_duplicate", "duplicate"),
        ("test_missing", "missing IDs"),
        ("overlap", "overlap"),
    ],
)
def test_external_id_contract_checks_rows_dropped_for_missing_outcome(
    tmp_path, case, match
):
    train = _frame(20)
    test = _frame(10)
    test["id"] += 100
    if case == "train_duplicate":
        train.loc[1, "id"] = train.loc[0, "id"]
        train.loc[1, "y"] = np.nan
    elif case == "test_duplicate":
        test.loc[1, "id"] = test.loc[0, "id"]
        test.loc[1, "y"] = np.nan
    elif case == "test_missing":
        test.loc[0, ["id", "y"]] = np.nan
    else:
        test.loc[0, "id"] = train.loc[0, "id"]
        test.loc[0, "y"] = np.nan
    schema_path = write_schema_bundle(
        tmp_path,
        train,
        test=test,
        split_mode="external_test",
        predictors=["X_a", "X_b"],
        id_column="id",
    )
    with pytest.raises(ValueError, match=match):
        _validated(schema_path)


def test_feature_universe_hash_and_semantics_are_both_checked(tmp_path):
    schema_path = write_schema_bundle(
        tmp_path, _frame(), predictors=["X_a", "X_b"]
    )
    definition_path = tmp_path / "feature_universe.json"
    document = json.loads(definition_path.read_text())
    document["predictors"].reverse()
    definition_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="definition_sha256"):
        _validated(schema_path)

    schema = json.loads(schema_path.read_text())
    import hashlib

    schema["feature_universe"]["definition_sha256"] = hashlib.sha256(
        definition_path.read_bytes()
    ).hexdigest()
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical definition"):
        _validated(schema_path)


@pytest.mark.parametrize(
    "override,match",
    [
        ({"group_column": "family"}, "group_column must be null"),
        (
            {
                "feature_universe": {
                    "mode": "train_pool_screened",
                    "definition_file": "feature_universe.json",
                    "definition_sha256": "placeholder",
                }
            },
            "internal_random requires",
        ),
    ],
)
def test_group_and_internal_universe_contracts(tmp_path, override, match):
    schema_path = write_schema_bundle(
        tmp_path,
        _frame(),
        predictors=["X_a", "X_b"],
        schema_overrides=override,
    )
    with pytest.raises(ValueError, match=match):
        load_schema(schema_path)


def test_external_test_outcome_threshold_and_required_columns(tmp_path):
    train = _frame(20)
    test = _frame(10)
    train["id"] = np.arange(20)
    test["id"] = 100 + np.arange(10)
    test.loc[:5, "y"] = np.nan
    schema_path = write_schema_bundle(
        tmp_path / "threshold",
        train,
        test=test,
        split_mode="external_test",
        predictors=["X_a", "X_b"],
        id_column="id",
        max_test_missing=0.5,
    )
    with pytest.raises(ValueError, match="test data outcome missing ratio"):
        _validated(schema_path)

    missing = test.drop(columns=["X_b"])
    schema_path = write_schema_bundle(
        tmp_path / "missing",
        train,
        test=missing,
        split_mode="external_test",
        predictors=["X_a", "X_b"],
        id_column="id",
    )
    with pytest.raises(ValueError, match="External test data is missing"):
        load_input(schema_path, "y")


def test_external_train_row_floor_is_validated_before_sampling(tmp_path):
    train = _frame(8)
    test = _frame(5)
    train["id"] = np.arange(8)
    test["id"] = 100 + np.arange(5)
    schema_path = write_schema_bundle(
        tmp_path,
        train,
        test=test,
        split_mode="external_test",
        predictors=["X_a", "X_b"],
        id_column="id",
    )
    with pytest.raises(ValueError, match="below required minimum"):
        _validated(schema_path, min_n=10)


def test_provenance_schema_hash_is_cross_checked(tmp_path):
    schema_path = write_schema_bundle(
        tmp_path, _frame(), predictors=["X_a", "X_b"]
    )
    (tmp_path / "provenance.json").write_text(
        json.dumps({"schema_sha256": "incorrect"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="provenance schema_sha256"):
        _validated(schema_path)


def test_external_ordinal_category_must_be_covered_by_train(tmp_path):
    train = _frame(20)[["id", "y", "X_a"]].rename(columns={"X_a": "ordinal"})
    train["ordinal"] = np.where(np.arange(20) % 2, 1.0, 2.0)
    test = train.iloc[:8].copy()
    test["id"] += 100
    test.loc[0, "ordinal"] = 3.0
    manifest = pd.DataFrame(
        {
            "source_column": ["ordinal_source"],
            "feature_name": ["ordinal"],
            "keep": [True],
            "source_order": [0],
            "feature_order": [0],
            "unit_type": ["ordinal"],
            "drop_first": [False],
            "is_reference": [False],
            "reference_level": [np.nan],
            "level_value": [np.nan],
            "ordinal_levels": ["[1,2,3]"],
            "source_prior": [1],
        }
    )
    schema_path = write_schema_bundle(
        tmp_path,
        train,
        test=test,
        split_mode="external_test",
        predictors=["ordinal"],
        manifest=manifest,
        id_column="id",
    )
    with pytest.raises(ValueError, match="category states absent from train"):
        _validated(schema_path)
