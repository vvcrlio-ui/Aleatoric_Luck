from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


ADAPTER_ROOT = Path(__file__).resolve().parents[1]
PROCESSOR_SRC = ADAPTER_ROOT / "src"
sys.path.insert(0, str(PROCESSOR_SRC))

from ffcws_data_processor.common.io import materialize_outcomes
from ffcws_data_processor.common.manifests import (
    kept_source_order,
    validate_feature_manifest,
)
from ffcws_data_processor.common.schema import SchemaConfig, build_shared_schema
from ffcws_data_processor.contract import enforce_outcome_train_category_coverage
from ffcws_data_processor.strategies.median_missing_indicator import (
    encode_median_missing_indicator,
)
from ffcws_data_processor.strategies.median_mode import encode_median_mode
from ffcws_data_processor.strategies.tree_ordinal import encode_tree_ordinal


def synthetic_background(*, include_test_unknown: bool = True) -> pd.DataFrame:
    n = 120
    category = [str(index % 2) for index in range(n)]
    if include_test_unknown:
        category[-2:] = ["2", "2"]
    category[0] = "-1"
    category[1] = "-2"
    category[2] = ""
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
        for result in results:
            validate_feature_manifest(
                result.features,
                result.feature_manifest,
                id_column="challengeID",
            )
        expected = ["continuous", "category", "small_integer_cat"]
        self.assertEqual(kept_source_order(results[0].feature_manifest), expected)
        self.assertEqual(kept_source_order(results[2].feature_manifest), expected)
        indicator_sources = kept_source_order(results[1].feature_manifest)
        self.assertTrue(set(expected).issubset(indicator_sources))
        self.assertTrue(
            any(source.startswith("continuous__missing__") for source in indicator_sources)
        )
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

    def test_missing_indicator_uses_separate_sources_and_keeps_groups_missing(self):
        result = encode_median_missing_indicator(
            self.background, self.schema, test_ids=self.test_ids
        )
        numeric_indicators = result.feature_manifest[
            result.feature_manifest["feature_name"].isin(
                [
                    "M_continuous__neg_6__not_asked",
                    "M_continuous__neg_9__missing",
                ]
            )
        ]
        self.assertEqual(len(numeric_indicators), 2)
        self.assertTrue(numeric_indicators["keep"].astype(bool).all())
        self.assertTrue(numeric_indicators["unit_type"].eq("continuous").all())
        self.assertEqual(numeric_indicators["source_column"].nunique(), 2)
        self.assertIn("M_category__neg_1__not_asked", result.features)
        self.assertIn("M_category__neg_2__refused", result.features)
        self.assertEqual(result.features["M_category__neg_1__not_asked"].sum(), 1)
        category_columns = [
            column for column in result.features if column.startswith("C_category__")
        ]
        self.assertTrue(result.features.loc[0, category_columns].isna().all())

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

    def test_test_rows_cannot_change_training_pool_schema(self):
        perturbed = self.background.copy()
        test_mask = perturbed["challengeID"].isin(self.test_ids)
        perturbed.loc[test_mask, "continuous"] = "999999"
        perturbed.loc[test_mask, "category"] = "777"
        second = build_shared_schema(
            perturbed,
            self.train_ids,
            value_labels=VALUE_LABELS,
            config=SchemaConfig(min_binary_prevalence=0.01),
        )
        self.assertEqual(self.schema.content_hash, second.content_hash)


class ContractMigrationTest(unittest.TestCase):
    def test_median_missing_indicator_emits_typed_manifest(self):
        background = synthetic_background(include_test_unknown=False)
        schema = build_shared_schema(
            background,
            background["challengeID"],
            value_labels=VALUE_LABELS,
        )
        result = encode_median_missing_indicator(background, schema)
        required = {
            "feature_order",
            "unit_type",
            "drop_first",
            "is_reference",
            "reference_level",
            "level_value",
            "ordinal_levels",
            "source_prior",
        }
        self.assertTrue(required.issubset(result.feature_manifest.columns))
        validate_feature_manifest(
            result.features, result.feature_manifest, id_column="challengeID"
        )
        continuous = result.feature_manifest[
            result.feature_manifest["source_column"].eq("continuous")
            & result.feature_manifest["keep"].astype(bool)
        ]
        self.assertEqual(continuous["unit_type"].tolist(), ["continuous"])
        category = result.feature_manifest[
            result.feature_manifest["source_column"].eq("category")
            & result.feature_manifest["keep"].astype(bool)
        ]
        self.assertTrue(category["unit_type"].eq("onehot_group").all())
        self.assertEqual(int(category["is_reference"].astype(bool).sum()), 1)

    def test_outcome_materialization_preserves_missing_outcomes(self):
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
        train.loc[0, "gpa"] = np.nan
        test.loc[0, "gpa"] = np.nan
        frames, summary = materialize_outcomes(
            features, train, test, outcomes=["gpa"], id_column="challengeID"
        )
        self.assertEqual(len(frames[("train", "gpa")]), 60)
        self.assertEqual(len(frames[("test", "gpa")]), 60)
        self.assertEqual(int(frames[("train", "gpa")]["gpa"].isna().sum()), 1)
        self.assertEqual(int(frames[("test", "gpa")]["gpa"].isna().sum()), 1)
        self.assertEqual(summary["rows_written"].tolist(), [60, 60])

    def test_outcome_specific_test_only_categories_are_masked(self):
        background = synthetic_background(include_test_unknown=False)
        background.loc[99, "category"] = "2"
        background.loc[100, "category"] = "2"
        train_ids = background.loc[:99, "challengeID"]
        test_ids = background.loc[100:, "challengeID"]
        schema = build_shared_schema(
            background,
            train_ids,
            value_labels=VALUE_LABELS,
            config=SchemaConfig(min_binary_prevalence=0.01),
        )
        train = pd.DataFrame(
            {
                "challengeID": train_ids,
                "gpa": np.linspace(2.0, 4.0, 100),
            }
        )
        train.loc[train["challengeID"].eq(1099), "gpa"] = np.nan
        test = pd.DataFrame(
            {
                "challengeID": test_ids,
                "gpa": np.linspace(2.1, 3.9, 20),
            }
        )

        for encoder in (encode_median_mode, encode_tree_ordinal):
            encoded = encoder(background, schema, test_ids=test_ids)
            frames, _ = materialize_outcomes(
                encoded.features,
                train,
                test,
                outcomes=["gpa"],
                id_column="challengeID",
            )
            masked, qa = enforce_outcome_train_category_coverage(
                frames,
                encoded.feature_manifest,
                unknown_rate_threshold=0.95,
            )
            category_features = (
                encoded.feature_manifest.loc[
                    encoded.feature_manifest["source_column"].eq("category")
                    & encoded.feature_manifest["keep"].astype(bool),
                    "feature_name",
                ]
                .astype(str)
                .tolist()
            )
            row = masked[("test", "gpa")].loc[
                lambda frame: frame["challengeID"].eq(1100), category_features
            ]
            self.assertTrue(row.isna().all(axis=None))
            category_qa = qa.loc[
                qa["source_column"].eq("category"), "unknown_count"
            ]
            self.assertEqual(category_qa.tolist(), [1])


if __name__ == "__main__":
    unittest.main()
