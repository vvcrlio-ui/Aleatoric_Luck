import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prepare_ffc_nk_inputs import build_outcome_frames


class BuildOutcomeFramesTest(unittest.TestCase):
    def test_builds_split_specific_outcome_frames_without_imputation(self):
        features = pd.DataFrame(
            {
                "challengeID": [1, 2, 3, 4],
                "X_income": [10.0, np.nan, 30.0, 40.0],
                "C_grade__1": [1, 0, 1, 0],
                "M_income__missing": [0, 1, 0, 0],
            }
        )
        train = pd.DataFrame({"challengeID": [1, 2, 3], "gpa": [3.1, 2.9, 2.8]})
        test = pd.DataFrame({"challengeID": [3, 4], "gpa": [np.nan, 3.6]})

        frames, summary = build_outcome_frames(
            features,
            train,
            test,
            outcomes=["gpa"],
        )

        train_frame = frames[("train", "gpa")]
        test_frame = frames[("test", "gpa")]
        self.assertEqual(
            train_frame.columns.tolist(),
            ["challengeID", "gpa", "X_income", "C_grade__1", "M_income__missing"],
        )
        self.assertEqual(train_frame["challengeID"].tolist(), [1, 2, 3])
        self.assertEqual(test_frame["challengeID"].tolist(), [4])
        self.assertFalse(np.isnan(train_frame.loc[0, "X_income"]))
        self.assertTrue(np.isnan(train_frame.loc[1, "X_income"]))
        self.assertEqual(train_frame["M_income__missing"].tolist(), [0, 1, 0])
        self.assertEqual(train_frame["C_grade__1"].tolist(), [1, 0, 1])
        summary_counts = {
            (row.split, row.outcome): row.rows_with_observed_outcome
            for row in summary.itertuples(index=False)
        }
        self.assertEqual(summary_counts[("train", "gpa")], 3)
        self.assertEqual(summary_counts[("test", "gpa")], 1)
        predictor_counts = {
            (row.split, row.outcome): row.categorical_columns
            for row in summary.itertuples(index=False)
        }
        self.assertEqual(predictor_counts[("train", "gpa")], 1)


if __name__ == "__main__":
    unittest.main()
