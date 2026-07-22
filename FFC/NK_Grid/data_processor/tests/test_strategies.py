from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


NK_ROOT = Path(__file__).resolve().parents[2]
PROCESSOR_SRC = NK_ROOT / "data_processor" / "src"
NK_SRC = NK_ROOT / "src"
sys.path.insert(0, str(PROCESSOR_SRC))
sys.path.insert(0, str(NK_SRC))

from data_processor.common.io import materialize_outcomes
from data_processor.common.manifests import (
    kept_source_order,
    validate_cross_strategy_sources,
    validate_feature_manifest,
)
from data_processor.common.schema import SchemaConfig, build_shared_schema
from data_processor.strategies.median_missing_indicator import (
    encode_median_missing_indicator,
)
from data_processor.strategies.median_mode import encode_median_mode
from data_processor.strategies.tree_ordinal import encode_tree_ordinal
from prepare_ffc_analysis import clean_background_frame
from prepare_ffc_nk_inputs import build_outcome_frames


def synthetic_background(*, include_test_unknown: bool = True) -> pd.DataFrame:
    n = 120
    category = [str(index % 2) for index in range(n)]
    if include_test_unknown:
        category[-2:] = ["2", "2"]
    category[0] = "-1"
    category[1] = "-2"
    continuous = [str(index / 10) for index in range(n)]
    continuous[0] = "-6"
    continuous[1] = "-9"
    continuous[2] = ""
    return pd.DataFrame(
        {
            "challengeID": range(1000, 1000 + n),
            "continuous": continuous,
            "category": category,
            "small_integer_cat": [str(index % 3) for index in range(n)],
            "constant": ["5"] * n,
        }
    )


VALUE_LABELS = {
    "category": {-1: "not asked", -2: "refused", 0: "no", 1: "yes"},
    "small_integer_cat": {0: "a", 1: "b", 2: "c"},
    "continuous": {-6: "not asked", -9: "missing"},
}


class StrategyContractTest(unittest.TestCase):
    def setUp(self):
        self.background = synthetic_background()
        self.train_ids = self.background.loc[:99, "challengeID"]
        self.test_ids = self.background.loc[100:, "challengeID"]
        self.schema = build_shared_schema(
            self.background,
            self.train_ids,
            value_labels=VALUE_LABELS,
            config=SchemaConfig(min_binary_prevalence=0.01),
        )

    def test_three_strategies_share_source_order_and_preserve_nan(self):
        results = [
            encode_median_mode(
                self.background, self.schema, test_ids=self.test_ids
            ),
            encode_median_missing_indicator(
                self.background, self.schema, test_ids=self.test_ids
            ),
            encode_tree_ordinal(
                self.background, self.schema, test_ids=self.test_ids
            ),
        ]
        expected = validate_cross_strategy_sources(results)
        self.assertEqual(expected, ["continuous", "category", "small_integer_cat"])
        for result in results:
            validate_feature_manifest(
                result.features,
                result.feature_manifest,
                id_column="challengeID",
            )
            self.assertEqual(kept_source_order(result.feature_manifest), expected)
        self.assertTrue(pd.isna(results[0].features.loc[0, "X_continuous"]))
        self.assertFalse(
            any(column.startswith("M_") for column in results[0].features)
        )

    def test_median_mode_marks_whole_dummy_group_nan(self):
        result = encode_median_mode(
            self.background, self.schema, test_ids=self.test_ids
        )
        category_columns = [
            column for column in result.features if column.startswith("C_category__")
        ]
        self.assertTrue(category_columns)
        self.assertTrue(result.features.loc[0, category_columns].isna().all())
        self.assertTrue(result.features.loc[119, category_columns].isna().all())

    def test_missing_indicator_keeps_distinct_ffc_codes(self):
        result = encode_median_missing_indicator(
            self.background, self.schema, test_ids=self.test_ids
        )
        self.assertIn("M_continuous__neg_6__not_asked", result.features)
        self.assertIn("M_continuous__neg_9__missing", result.features)
        self.assertIn("C_category__neg_1__not_asked", result.features)
        self.assertIn("C_category__neg_2__refused", result.features)
        self.assertEqual(result.features["M_continuous__neg_6__not_asked"].sum(), 1)

    def test_tree_ordinal_is_float_deterministic_and_unknown_is_nan(self):
        first = encode_tree_ordinal(
            self.background, self.schema, test_ids=self.test_ids
        )
        second = encode_tree_ordinal(
            self.background, self.schema, test_ids=self.test_ids
        )
        self.assertEqual(first.features["C_category"].dtype, np.dtype("float64"))
        self.assertTrue(np.isnan(first.features.loc[0, "C_category"]))
        self.assertTrue(np.isnan(first.features.loc[119, "C_category"]))
        self.assertEqual(first.ordinal_mappings, second.ordinal_mappings)
        assert_frame_equal(first.features, second.features)

    def test_unknown_ceiling_fails_with_diagnostics(self):
        frame = self.background.copy()
        frame.loc[frame["challengeID"].isin(self.test_ids), "category"] = "99"
        with self.assertRaisesRegex(ValueError, "source=category"):
            encode_tree_ordinal(
                frame,
                self.schema,
                test_ids=self.test_ids,
                unknown_rate_threshold=0.95,
            )


class BaselineParityTest(unittest.TestCase):
    def test_median_missing_indicator_matches_existing_encoder(self):
        background = synthetic_background(include_test_unknown=False)
        schema = build_shared_schema(
            background,
            background["challengeID"],
            value_labels=VALUE_LABELS,
        )
        result = encode_median_missing_indicator(background, schema)
        baseline, baseline_sources, baseline_features, _ = clean_background_frame(
            background,
            value_labels=VALUE_LABELS,
        )
        self.assertEqual(result.features.columns.tolist(), baseline.columns.tolist())
        assert_frame_equal(result.features, baseline, check_dtype=False)
        new_kept = result.feature_manifest[result.feature_manifest["keep"]][
            "feature_name"
        ].tolist()
        old_kept = baseline_features[baseline_features["keep"]][
            "feature_name"
        ].tolist()
        self.assertEqual(new_kept, old_kept)
        old_sources = baseline_sources.loc[
            baseline_sources["status"].isin(["numeric", "categorical"]),
            "source_column",
        ].tolist()
        self.assertEqual(kept_source_order(result.feature_manifest), old_sources)

    def test_outcome_materialization_matches_existing_helper(self):
        background = synthetic_background(include_test_unknown=False)
        schema = build_shared_schema(
            background, background["challengeID"], value_labels=VALUE_LABELS
        )
        features = encode_median_missing_indicator(background, schema).features
        train = pd.DataFrame(
            {"challengeID": range(1000, 1060), "gpa": np.linspace(2.0, 4.0, 60)}
        )
        test = pd.DataFrame(
            {"challengeID": range(1060, 1120), "gpa": np.linspace(2.1, 3.9, 60)}
        )
        new_frames, _ = materialize_outcomes(
            features, train, test, outcomes=["gpa"], id_column="challengeID"
        )
        old_frames, _ = build_outcome_frames(
            features, train, test, outcomes=["gpa"]
        )
        for key in old_frames:
            assert_frame_equal(new_frames[key], old_frames[key])


if __name__ == "__main__":
    unittest.main()
