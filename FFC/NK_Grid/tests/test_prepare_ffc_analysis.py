import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prepare_ffc_analysis import _metadata_from_stata, clean_background_frame


class CleanBackgroundFrameTest(unittest.TestCase):
    def test_v2_cleaning_rules_and_feature_manifest(self):
        n = 120
        frame = pd.DataFrame(
            {
                "challengeID": range(1000, 1000 + n),
                "income": [str(i) for i in range(n)],
                "low_labeled": [str((i % 3) + 1) for i in range(n)],
                "continuous_missing": [
                    "-6" if i in (0, 1) else "-9" if i in (2, 3) else str(i / 10)
                    for i in range(n)
                ],
                "categorical_negative": [
                    "-6" if i in (0, 1) else str(i % 2) for i in range(n)
                ],
                "text_var": ["alpha" if i % 2 else "beta" for i in range(n)],
                "sparse_var": ["1" if i < 40 else "" for i in range(n)],
                "rare_cat": ["1" if i == 0 else "0" for i in range(n)],
                "rare_missing": ["-6" if i == 0 else str(i) for i in range(n)],
                "small_variance_numeric": [str((i % 11) / 100) for i in range(n)],
                "constant_numeric": ["5" for _ in range(n)],
            }
        )

        features, manifest, feature_manifest, summary = clean_background_frame(
            frame,
            min_valid_rate=0.5,
            min_numeric_fraction=0.95,
            categorical_max_levels=15,
            min_binary_prevalence=0.01,
            value_labels={
                "income": {1: "one dollar"},
                "low_labeled": {1: "low", 2: "middle", 3: "high"},
                "continuous_missing": {-6: "not asked", -9: "unknown"},
                "categorical_negative": {-6: "not asked", -9: "unknown"},
                "rare_missing": {-6: "not asked", -9: "unknown"},
            },
        )

        self.assertIn("X_income", features.columns)
        self.assertNotIn("C_income__0", features.columns)
        self.assertEqual(
            manifest.loc[manifest["source_column"] == "income", "status"].item(),
            "numeric",
        )

        self.assertIn("C_low_labeled__1__low", features.columns)
        self.assertIn("C_low_labeled__2__middle", features.columns)
        self.assertIn("C_low_labeled__3__high", features.columns)
        self.assertEqual(
            manifest.loc[manifest["source_column"] == "low_labeled", "status"].item(),
            "categorical",
        )

        self.assertTrue(np.isnan(features.loc[0, "X_continuous_missing"]))
        self.assertTrue(np.isnan(features.loc[2, "X_continuous_missing"]))
        self.assertIn(
            "M_continuous_missing__neg_6__not_asked",
            features.columns,
        )
        self.assertIn("M_continuous_missing__neg_9__unknown", features.columns)
        self.assertEqual(
            features["M_continuous_missing__neg_6__not_asked"].tolist().count(1),
            2,
        )

        self.assertIn("C_categorical_negative__neg_6__not_asked", features.columns)
        self.assertNotIn("M_categorical_negative__neg_6__not_asked", features.columns)

        reasons = dict(zip(manifest["source_column"], manifest["reason"]))
        self.assertEqual(reasons["text_var"], "below_min_numeric_fraction")
        self.assertEqual(reasons["sparse_var"], "below_min_valid_rate")
        self.assertEqual(reasons["constant_numeric"], "constant_after_missing")

        self.assertNotIn("C_rare_cat__1", features.columns)
        self.assertNotIn("M_rare_missing__neg_6__not_asked", features.columns)
        self.assertIn("X_small_variance_numeric", features.columns)

        feature_rows = set(feature_manifest["feature_name"])
        for column in features.columns:
            if column != "challengeID":
                self.assertIn(column, feature_rows)
        self.assertTrue(
            {
                "source_column",
                "feature_name",
                "kind",
                "prevalence",
                "observed_variance",
                "keep",
                "reason",
            }.issubset(feature_manifest.columns)
        )
        self.assertEqual(summary["kept_X_columns"], 4)
        self.assertEqual(summary["parameters"]["categorical_max_levels"], 15)

    def test_metadata_from_stata_reconstructs_labels_from_raw_and_categorical_reads(self):
        raw = pd.DataFrame(
            {
                "challengeID": [1, 2, 3],
                "relationship": [-9, 1, 2],
                "plain": [10, 11, 12],
            }
        )
        labeled = pd.DataFrame(
            {
                "challengeID": [1, 2, 3],
                "relationship": pd.Categorical(
                    ["-9 Not in wave", "1 married", "2 cohab"]
                ),
                "plain": [10, 11, 12],
            }
        )
        with patch(
            "prepare_ffc_analysis.pd.read_stata",
            return_value=labeled,
        ) as read_stata:
            labels = _metadata_from_stata(Path("background.dta"), raw)

        read_stata.assert_called_once_with(
            Path("background.dta"), convert_categoricals=True
        )
        self.assertEqual(
            labels,
            {"relationship": {-9: "Not in wave", 1: "married", 2: "cohab"}},
        )

    def test_feature_names_use_variable_keyed_stata_labels_when_available(self):
        n = 120
        frame = pd.DataFrame(
            {
                "challengeID": range(2000, 2000 + n),
                "labeled_cat": ["-2" if i < 2 else str(i % 2) for i in range(n)],
                "unlabeled_cat": ["-2" if i < 2 else str(i % 2) for i in range(n)],
                "labeled_num": ["-9" if i < 2 else str(i) for i in range(n)],
                "unlabeled_num": ["-9" if i < 2 else str(i) for i in range(n)],
            }
        )
        features, _, _, _ = clean_background_frame(
            frame,
            min_valid_rate=0.5,
            min_numeric_fraction=0.95,
            categorical_max_levels=15,
            min_binary_prevalence=0.01,
            value_labels={
                "labeled_cat": {-2: "enforced skip"},
                "labeled_num": {-9: "missing"},
            },
        )

        self.assertIn("C_labeled_cat__neg_2__enforced_skip", features.columns)
        self.assertIn("C_unlabeled_cat__neg_2", features.columns)
        self.assertIn("M_labeled_num__neg_9__missing", features.columns)
        self.assertIn("M_unlabeled_num__neg_9", features.columns)


if __name__ == "__main__":
    unittest.main()
