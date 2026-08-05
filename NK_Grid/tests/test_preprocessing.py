from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import aleatoric_nk_grid.preprocessing as preprocessing
from aleatoric_nk_grid.preprocessing import (
    SourceGroup,
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


def _equivalence_groups(k: int) -> tuple[SourceGroup, ...]:
    """Build disjoint groups covering each imputation unit type."""

    assert k >= 3
    groups = [
        SourceGroup(
            name="continuous",
            features=("continuous",),
            source_order=0,
            unit_type="continuous",
            source_prior=11.0,
        ),
        SourceGroup(
            name="ordinal",
            features=("ordinal",),
            source_order=1,
            unit_type="ordinal",
            ordinal_levels=(1, 2, 3),
            source_prior=1,
        ),
        SourceGroup(
            name="onehot",
            features=("onehot_0", "onehot_1"),
            source_order=2,
            unit_type="onehot_group",
            drop_first=True,
            reference_level=0,
            level_values=(1, 2),
        ),
    ]
    groups.extend(
        SourceGroup(
            name=f"extra_{index}",
            features=(f"extra_{index}",),
            source_order=index,
            unit_type="continuous",
            source_prior=0.0,
        )
        for index in range(3, k)
    )
    return tuple(groups)


def _equivalence_frames(
    n: int,
    k: int,
    *,
    homogeneous: bool = False,
    unobserved_onehot: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create exact, mixed-dtype inputs with opposing train/test missingness."""

    continuous_dtype = float if homogeneous else np.float32
    test_n = max(2, n // 4)
    train_columns: dict[str, np.ndarray] = {
        "continuous": np.arange(n, dtype=continuous_dtype),
        "ordinal": np.resize(np.array([1.0, 2.0, 3.0]), n),
        "onehot_0": np.resize(np.array([0.0, 1.0, 0.0]), n),
        "onehot_1": np.resize(np.array([0.0, 0.0, 1.0]), n),
    }
    test_columns: dict[str, np.ndarray] = {
        "continuous": np.arange(test_n, dtype=continuous_dtype),
        "ordinal": np.resize(np.array([1.0, 3.0]), test_n),
        "onehot_0": np.resize(np.array([0.0, 1.0]), test_n),
        "onehot_1": np.resize(np.array([0.0, 0.0]), test_n),
    }
    for index in range(3, k):
        name = f"extra_{index}"
        train_values = np.arange(n, dtype=continuous_dtype)
        test_values = np.arange(test_n, dtype=continuous_dtype)
        if index % 11 == 0:
            train_values[:] = np.nan  # fully unobserved source
        elif index % 2 == 0:
            train_values[0] = np.nan
        else:
            test_values[0] = np.nan
        train_columns[name] = train_values
        test_columns[name] = test_values
    train = pd.DataFrame(train_columns)
    test = pd.DataFrame(test_columns)
    train.loc[0, ["continuous", "ordinal", "onehot_0", "onehot_1"]] = np.nan
    test.loc[1, ["continuous", "ordinal", "onehot_0", "onehot_1"]] = np.nan
    if unobserved_onehot:
        train.loc[:, ["onehot_0", "onehot_1"]] = np.nan
    return train, test


@pytest.mark.slow
@pytest.mark.parametrize("k", [10, 100, 1000, 3125])
@pytest.mark.parametrize("n", [10, 1000, 4242])
@pytest.mark.parametrize("model_name", ["ols", "xgboost"])
def test_preprocess_cell_matches_reference_across_scale_matrix(k, n, model_name):
    """The fast implementation is exactly equal to the retained reference."""

    groups = _equivalence_groups(k)
    train, test = _equivalence_frames(n, k)
    expected = preprocessing._preprocess_cell_reference(
        train, test, groups, IMPUTATION, model_name=model_name
    )
    actual = preprocess_cell(train, test, groups, IMPUTATION, model_name=model_name)

    pd.testing.assert_frame_equal(actual.X_train, expected.X_train, check_exact=True)
    pd.testing.assert_frame_equal(actual.X_test, expected.X_test, check_exact=True)
    assert actual.K_unobserved == expected.K_unobserved
    assert actual.passthrough == expected.passthrough


def test_preprocess_cell_does_not_mutate_mixed_dtype_inputs():
    groups = _equivalence_groups(10)
    train, test = _equivalence_frames(20, 10)
    train_before = train.copy(deep=True)
    test_before = test.copy(deep=True)

    preprocess_cell(train, test, groups, IMPUTATION, model_name="ols")

    pd.testing.assert_frame_equal(train, train_before, check_exact=True)
    pd.testing.assert_frame_equal(test, test_before, check_exact=True)


@pytest.mark.parametrize("n,k", [(10, 10), (1000, 100)])
@pytest.mark.parametrize("unobserved_onehot", [False, True])
def test_vectorized_preprocess_cell_matches_reference_exactly(
    n, k, unobserved_onehot
):
    groups = _equivalence_groups(k)
    train, test = _equivalence_frames(
        n, k, homogeneous=True, unobserved_onehot=unobserved_onehot
    )
    assert preprocessing._supports_vectorized_preprocessing(train, test)

    expected = preprocessing._preprocess_cell_reference(
        train, test, groups, IMPUTATION, model_name="ols"
    )
    actual = preprocess_cell(train, test, groups, IMPUTATION, model_name="ols")

    pd.testing.assert_frame_equal(actual.X_train, expected.X_train, check_exact=True)
    pd.testing.assert_frame_equal(actual.X_test, expected.X_test, check_exact=True)
    assert actual.K_unobserved == expected.K_unobserved


def test_source_groups_have_disjoint_feature_sets():
    groups = source_groups(["x", "o", "c0", "c1"], _manifest())
    seen: set[str] = set()
    for group in groups:
        overlap = seen.intersection(group.features)
        assert not overlap
        seen.update(group.features)


def test_preprocess_cell_rejects_overlapping_groups():
    groups = (
        SourceGroup("left", ("x",), 0, "continuous", source_prior=0.0),
        SourceGroup("right", ("x",), 1, "continuous", source_prior=0.0),
    )
    frame = pd.DataFrame({"x": [1.0, np.nan]})

    with pytest.raises(ValueError, match="share feature"):
        preprocess_cell(frame, frame, groups, IMPUTATION, model_name="ols")


@pytest.mark.parametrize(
    "dtypes",
    [
        ("int8", "int16", "int32", "int64"),
        ("float32", "float64", "float32", "float64"),
        ("int8", "int16", "int32", "float64"),
        ("float32", "float64", "int32", "int64"),
        ("int8", "int16", "int32", "int64", "float32", "float64"),
    ],
)
@pytest.mark.parametrize("passthrough", [False, True])
@pytest.mark.parametrize("missing_in", ["train", "test"])
def test_vectorized_mixed_numeric_dtypes_match_reference_exactly(
    dtypes, passthrough, missing_in
):
    columns = [f"x{index}" for index in range(len(dtypes))]
    train = pd.DataFrame({column: np.array([1, 2, 3], dtype=dtype) for column, dtype in zip(columns, dtypes)})
    test = pd.DataFrame({column: np.array([3, 2], dtype=dtype) for column, dtype in zip(columns, dtypes)})
    float_column = next(
        (column for column, dtype in zip(columns, dtypes) if dtype.startswith("float")),
        None,
    )
    if float_column is not None:
        (train if missing_in == "train" else test).loc[0, float_column] = np.nan
    groups = tuple(
        SourceGroup(column, (column,), index, "continuous", source_prior=0.0)
        for index, column in enumerate(columns)
    )
    model_name = "xgboost" if passthrough else "ols"
    expected = preprocessing._preprocess_cell_reference(
        train, test, groups, IMPUTATION, model_name=model_name
    )
    actual = preprocess_cell(train, test, groups, IMPUTATION, model_name=model_name)
    pd.testing.assert_frame_equal(actual.X_train, expected.X_train, check_exact=True)
    pd.testing.assert_frame_equal(actual.X_test, expected.X_test, check_exact=True)
    assert actual.K_unobserved == expected.K_unobserved
    assert actual.X_train.attrs["_preprocess_vectorized"] is (not passthrough)


def test_vectorized_mixed_onehot_group_keeps_group_observation_semantics():
    groups = (
        SourceGroup("mixed", ("float_dummy", "int_dummy"), 0, "onehot_group", drop_first=True),
    )
    train = pd.DataFrame({"float_dummy": np.array([np.nan, 1.0, 0.0], dtype="float32"), "int_dummy": np.array([0, 0, 1], dtype="int16")})
    test = pd.DataFrame({"float_dummy": np.array([np.nan, 0.0], dtype="float32"), "int_dummy": np.array([0, 1], dtype="int16")})
    expected = preprocessing._preprocess_cell_reference(train, test, groups, IMPUTATION, model_name="ols")
    actual = preprocess_cell(train, test, groups, IMPUTATION, model_name="ols")
    pd.testing.assert_frame_equal(actual.X_train, expected.X_train, check_exact=True)
    pd.testing.assert_frame_equal(actual.X_test, expected.X_test, check_exact=True)
    assert actual.X_train.attrs["_preprocess_vectorized"] is True
