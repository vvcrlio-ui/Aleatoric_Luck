from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import aleatoric_nk_grid.preprocessing as preprocessing
from aleatoric_nk_grid.preprocessing import (
    preprocess_cell,
    source_groups,
    validate_onehot_states,
)


def _manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_column": ["continuous", "ordinal", "category", "category"],
            "feature_name": ["x", "o", "c0", "c1"],
            "keep": [True, True, True, True],
            "source_order": [0, 1, 2, 2],
            "feature_order": [0, 0, 0, 1],
            "unit_type": [
                "continuous",
                "ordinal",
                "onehot_group",
                "onehot_group",
            ],
            "drop_first": [False, False, True, True],
            "is_reference": [False, False, False, False],
            "reference_level": [np.nan, np.nan, "ref", "ref"],
            "level_value": [np.nan, np.nan, "a", "b"],
            "ordinal_levels": [
                np.nan,
                json.dumps([1, 2, 3], separators=(",", ":")),
                np.nan,
                np.nan,
            ],
            "source_prior": [10.0, 1, np.nan, np.nan],
        }
    )


IMPUTATION = {
    "continuous": "median",
    "ordinal": "median_snap",
    "onehot_group": "atomic_mode",
    "model_overrides": {"xgboost": "passthrough"},
}


def test_typed_imputation_and_tie_breaks():
    groups = source_groups(["x", "o", "c0", "c1"], _manifest())
    train = pd.DataFrame(
        {
            "x": [1.0, 3.0, np.nan],
            "o": [1.0, 2.0, np.nan],
            "c0": [1.0, 0.0, np.nan],
            "c1": [0.0, 1.0, np.nan],
        }
    )
    test = pd.DataFrame(
        {
            "x": [np.nan],
            "o": [np.nan],
            "c0": [np.nan],
            "c1": [np.nan],
        }
    )
    result = preprocess_cell(train, test, groups, IMPUTATION, model_name="ols")
    assert result.K_unobserved == 0
    assert result.X_train.loc[2, "x"] == 2.0
    assert result.X_train.loc[2, "o"] == 1
    assert result.X_train.loc[2, ["c0", "c1"]].tolist() == [1.0, 0.0]
    assert result.X_test.loc[0].tolist() == [2.0, 1.0, 1.0, 0.0]


def test_no_missing_fast_paths_skip_imputation_statistics(monkeypatch):
    groups = source_groups(["x", "o", "c0", "c1"], _manifest())
    train = pd.DataFrame(
        {
            "x": [1.0, 2.0],
            "o": [1.0, 2.0],
            "c0": [1.0, 0.0],
            "c1": [0.0, 1.0],
        }
    )
    test = train.iloc[[0]].copy()

    def unexpected_call(*args, **kwargs):
        raise AssertionError("imputation statistic must not run without missing values")

    monkeypatch.setattr(pd.Series, "median", unexpected_call)
    monkeypatch.setattr(preprocessing, "_ordinal_statistic", unexpected_call)
    monkeypatch.setattr(preprocessing, "_onehot_state", unexpected_call)

    result = preprocess_cell(train, test, groups, IMPUTATION, model_name="ols")
    pd.testing.assert_frame_equal(result.X_train, train)
    pd.testing.assert_frame_equal(result.X_test, test)


def test_continuous_nullable_int64_noninteger_median_keeps_fill_error():
    groups = source_groups(["x"], _manifest().iloc[[0]].copy())
    train = pd.DataFrame({"x": pd.Series([1, 2, pd.NA], dtype="Int64")})
    test = pd.DataFrame({"x": pd.Series([1, 2], dtype="Int64")})
    with pytest.raises(TypeError, match=r"Invalid value '1\.5' for dtype 'Int64'"):
        preprocess_cell(train, test, groups, IMPUTATION, model_name="ols")


def test_continuous_nullable_int64_without_missing_preserves_dtype():
    groups = source_groups(["x"], _manifest().iloc[[0]].copy())
    train = pd.DataFrame({"x": pd.Series([1, 2], dtype="Int64")})
    test = pd.DataFrame({"x": pd.Series([2, 1], dtype="Int64")})
    result = preprocess_cell(train, test, groups, IMPUTATION, model_name="ols")
    assert result.X_train["x"].dtype == pd.Int64Dtype()
    assert result.X_test["x"].dtype == pd.Int64Dtype()
    assert result.X_train["x"].tolist() == [1, 2]
    assert result.X_test["x"].tolist() == [2, 1]


def test_continuous_direct_call_retains_legacy_numeric_string_coercion():
    groups = source_groups(["x"], _manifest().iloc[[0]].copy())
    train = pd.DataFrame({"x": ["1", "3", None]})
    test = pd.DataFrame({"x": [None]})
    result = preprocess_cell(train, test, groups, IMPUTATION, model_name="ols")
    assert result.X_train["x"].tolist() == ["1", "3", 2.0]
    assert result.X_test["x"].tolist() == [2.0]


def test_continuous_direct_call_still_rejects_nonnumeric_without_missing():
    groups = source_groups(["x"], _manifest().iloc[[0]].copy())
    train = pd.DataFrame({"x": ["not-a-number", "still-not-a-number"]})
    test = pd.DataFrame({"x": ["not-a-number"]})

    with pytest.raises(ValueError, match="Unable to parse string"):
        preprocess_cell(train, test, groups, IMPUTATION, model_name="ols")


def test_fully_unobserved_source_forces_same_prior_on_train_and_test():
    groups = source_groups(["x"], _manifest().iloc[[0]].copy())
    train = pd.DataFrame({"x": [np.nan, np.nan]})
    test = pd.DataFrame({"x": [999.0, np.nan]})
    result = preprocess_cell(train, test, groups, IMPUTATION, model_name="ols")
    assert result.K_unobserved == 1
    assert result.X_train["x"].tolist() == [10.0, 10.0]
    assert result.X_test["x"].tolist() == [10.0, 10.0]


def test_passthrough_forces_both_sides_to_nan_for_unobserved_source():
    groups = source_groups(["x"], _manifest().iloc[[0]].copy())
    train = pd.DataFrame({"x": [np.nan, np.nan]})
    test = pd.DataFrame({"x": [999.0, np.nan]})
    result = preprocess_cell(
        train, test, groups, IMPUTATION, model_name="xgboost"
    )
    assert result.K_unobserved == 1
    assert result.X_train["x"].isna().all()
    assert result.X_test["x"].isna().all()


@pytest.mark.parametrize(
    "bad",
    [
        pd.DataFrame({"c0": [1.0], "c1": [np.nan]}),
        pd.DataFrame({"c0": [1.0], "c1": [1.0]}),
        pd.DataFrame({"c0": [0.5], "c1": [0.5]}),
    ],
)
def test_invalid_onehot_states_fail_fast(bad):
    groups = source_groups(["c0", "c1"], _manifest().iloc[[2, 3]].copy())
    with pytest.raises(ValueError, match="invalid one-hot"):
        validate_onehot_states(bad, groups, "training data")


def test_drop_first_all_zero_is_valid():
    groups = source_groups(["c0", "c1"], _manifest().iloc[[2, 3]].copy())
    validate_onehot_states(
        pd.DataFrame({"c0": [0.0], "c1": [0.0]}), groups, "training data"
    )


def test_drop_first_reference_wins_mode_tie_against_dummy():
    groups = source_groups(["c0", "c1"], _manifest().iloc[[2, 3]].copy())
    train = pd.DataFrame(
        {
            "c0": [0.0, 1.0, np.nan],
            "c1": [0.0, 0.0, np.nan],
        }
    )
    test = pd.DataFrame({"c0": [np.nan], "c1": [np.nan]})
    result = preprocess_cell(train, test, groups, IMPUTATION, model_name="ols")
    assert result.X_train.loc[2, ["c0", "c1"]].tolist() == [0.0, 0.0]
    assert result.X_test.loc[0, ["c0", "c1"]].tolist() == [0.0, 0.0]


def test_ordinal_levels_must_be_canonical_json():
    manifest = _manifest()
    manifest.loc[1, "ordinal_levels"] = "[1, 2, 3]"
    with pytest.raises(ValueError, match="not canonical JSON"):
        source_groups(["x", "o", "c0", "c1"], manifest)


@pytest.mark.parametrize(
    "levels",
    [
        ["low", "high"],
        [1, "2"],
        [False, True],
        [1, float("inf")],
    ],
)
def test_ordinal_levels_must_be_finite_json_numbers(levels):
    manifest = _manifest()
    manifest.loc[1, "ordinal_levels"] = json.dumps(
        levels, separators=(",", ":")
    )
    with pytest.raises(ValueError, match="finite JSON numbers"):
        source_groups(["x", "o", "c0", "c1"], manifest)


def test_manifest_source_order_and_type_must_be_consistent():
    manifest = _manifest()
    manifest.loc[3, "source_order"] = 9
    with pytest.raises(ValueError, match="inconsistent source_order"):
        source_groups(["x", "o", "c0", "c1"], manifest)
    manifest = _manifest()
    manifest.loc[3, "unit_type"] = "continuous"
    with pytest.raises(ValueError, match="inconsistent unit_type"):
        source_groups(["x", "o", "c0", "c1"], manifest)
