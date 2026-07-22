import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model_registry import make_model
from nk_grid import (
    NKGridConfig,
    _feature_groups,
    _predictor_columns,
    read_table,
    run_nk_grid,
)


class ReadTableTest(unittest.TestCase):
    def test_csv_and_parquet_round_trip(self):
        frame = pd.DataFrame(
            {
                "challengeID": [1, 2, 3],
                "X_value": [1.5, np.nan, 3.5],
                "C_group": pd.Series([0.0, 1.0, np.nan], dtype="float64"),
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "input.csv"
            parquet_path = root / "input.parquet"
            pq_path = root / "input.pq"
            frame.to_csv(csv_path, index=False)
            frame.to_parquet(parquet_path, index=False)
            frame.to_parquet(pq_path, index=False)

            assert_frame_equal(read_table(csv_path), frame, check_dtype=False)
            assert_frame_equal(read_table(parquet_path), frame)
            assert_frame_equal(read_table(pq_path), frame)
            self.assertEqual(read_table(parquet_path).columns.tolist(), frame.columns.tolist())
            self.assertEqual(read_table(parquet_path)["C_group"].dtype, np.dtype("float64"))

    def test_unknown_suffix_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "supported suffixes"):
            read_table(Path("input.feather"))

    def test_parquet_external_smoke_run_respects_max_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_rows = np.arange(24, dtype=float)
            test_rows = np.arange(8, dtype=float)
            train = pd.DataFrame(
                {
                    "challengeID": np.arange(24),
                    "y": 1.5 * train_rows + 2.0,
                    "X_value": train_rows,
                    "C_group": np.where(train_rows % 5 == 0, np.nan, train_rows % 3),
                }
            )
            test = pd.DataFrame(
                {
                    "challengeID": np.arange(100, 108),
                    "y": 1.5 * (test_rows + 24.0) + 2.0,
                    "X_value": test_rows + 24.0,
                    "C_group": np.where(test_rows % 4 == 0, np.nan, test_rows % 3),
                }
            )
            train_path = root / "train.parquet"
            test_path = root / "test.parquet"
            manifest_path = root / "feature_manifest.csv"
            out_path = root / "results.csv"
            train.to_parquet(train_path, index=False)
            test.to_parquet(test_path, index=False)
            pd.DataFrame(
                {
                    "source_column": ["value", "group"],
                    "feature_name": ["X_value", "C_group"],
                    "keep": [True, True],
                }
            ).to_csv(manifest_path, index=False)

            run_nk_grid(
                NKGridConfig(
                    data=train_path,
                    test_data=test_path,
                    out=out_path,
                    dataset="synthetic_tree_ordinal",
                    outcome="y",
                    models=("ols", "ridge"),
                    seed=123,
                    test_size=0.3,
                    n_seeds=1,
                    n_draws=1,
                    n_sizes_n=1,
                    n_sizes_k=1,
                    max_n=24,
                    max_k=2,
                    batch_size=10,
                    n_jobs=1,
                    predictor_prefix=("X_", "C_"),
                    feature_manifest=manifest_path,
                ),
                max_jobs=1,
            )

            result = pd.read_csv(out_path)
            self.assertEqual(len(result), 1)
            self.assertEqual(result.loc[0, "status"], "ok")
            self.assertEqual(result.loc[0, "n_features_total"], 2)


class ExistingContractsTest(unittest.TestCase):
    def test_predictor_prefixes_and_source_groups_need_no_new_logic(self):
        frame = pd.DataFrame(
            {
                "challengeID": [1],
                "outcome": [0.0],
                "X_income": [1.0],
                "C_group": [0.0],
                "M_income__neg_9": [0],
                "metadata": ["x"],
            }
        )
        predictors = _predictor_columns(frame, ("X_", "C_", "M_"))
        self.assertEqual(
            predictors, ["X_income", "C_group", "M_income__neg_9"]
        )
        manifest = pd.DataFrame(
            {
                "source_column": ["income", "group", "income"],
                "feature_name": predictors,
                "keep": [True, True, True],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            manifest.to_csv(path, index=False)
            units, groups = _feature_groups(predictors, path)
        self.assertEqual(units, ["income", "group"])
        self.assertEqual(groups["income"], ["X_income", "M_income__neg_9"])
        self.assertEqual(groups["group"], ["C_group"])

    def test_all_nan_predictor_keeps_shape_in_existing_pipeline(self):
        X_train = pd.DataFrame(
            {"all_missing": [np.nan] * 4, "observed": [1.0, 2.0, 3.0, 4.0]}
        )
        X_test = pd.DataFrame(
            {"all_missing": [np.nan, np.nan], "observed": [2.5, 3.5]}
        )
        y_train = np.array([1.0, 2.0, 3.0, 4.0])
        model = make_model("ols", seed=123, task="regression")
        model.fit(X_train, y_train)
        transformed_train = model[:-1].transform(X_train)
        transformed_test = model[:-1].transform(X_test)
        self.assertEqual(transformed_train.shape[1], X_train.shape[1])
        self.assertEqual(transformed_test.shape[1], X_train.shape[1])
        self.assertEqual(len(model.predict(X_test)), len(X_test))


if __name__ == "__main__":
    unittest.main()
