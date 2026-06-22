from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from ffcws_predict.pipeline import audit_features, learning_curve, prepare, summarize, train
from ffcws_predict.powerlaw import fit_curve
from ffcws_predict.io import read_table
from ffcws_predict.preprocessing import Preprocessor


def write_config(root: Path, models=None, n_draws=3, boot=20, imputation_strategy=None) -> Path:
    models = models or ["mean_baseline", "ridge"]
    imputation_strategy = imputation_strategy or ["median"]
    config = {
        "paths": {
            "features_path": str(root / "features.csv"),
            "train_outcomes_path": str(root / "train.csv"),
            "truth_csv_path": str(root / "truth.csv"),
            "truth_url": "",
            "feature_dictionary_path": str(root / "feature_dictionary.csv"),
            "output_dir": str(root / "out"),
        },
        "target": {
            "id_column": "challengeID",
            "name": "materialHardship",
            "split_column": "split",
        },
        "experiment": {"seed": 123, "n_jobs": 1},
        "preprocessing": {
            "missing_codes": [-9, -8, -7, -6, -5, -4, -3, -2, -1],
            "missing_threshold": 0.8,
            "low_variance_threshold": 1e-12,
            "imputation_strategy": imputation_strategy,
        },
        "models": models,
        "learning_curve": {
            "sample_sizes": [18, 30],
            "n_draws": n_draws,
            "bootstrap_iterations": boot,
        },
    }
    path = root / "config.yaml"
    path.write_text(json.dumps(config))
    return path


def write_synthetic_ds15(root: Path, n: int = 90, n_holdout: int = 30) -> None:
    rng = np.random.default_rng(42)
    ids = np.arange(1, n + 1)
    income = rng.normal(0, 1, n)
    employed = rng.integers(0, 2, n)
    moves = rng.poisson(1, n).astype(float)
    support = rng.normal(0, 1, n)
    category = np.where(income > 0, "high", "low")
    moves[::11] = -9
    y = 0.2 - 0.05 * income - 0.03 * employed + 0.04 * moves.clip(min=0) - 0.02 * support
    y = np.clip(y + rng.normal(0, 0.02, n), 0, 0.9)

    pd.DataFrame(
        {
            "challengeID": ids,
            "income_anchor": income,
            "employed_anchor": employed,
            "moves_anchor": moves,
            "support_anchor": support,
            "category_anchor": category,
            "constant_anchor": 1,
            "mostly_missing_anchor": [np.nan] * (n - 2) + [1, 2],
        }
    ).to_csv(root / "features.csv", index=False)

    holdout_ids = set(ids[:n_holdout])
    train = pd.DataFrame({"challengeID": ids, "materialHardship": y})
    train.loc[train["challengeID"].isin(holdout_ids), "materialHardship"] = np.nan
    train.to_csv(root / "train.csv", index=False)

    truth = pd.DataFrame(
        {
            "challengeID": ids,
            "gpa": np.nan,
            "grit": np.nan,
            "materialHardship": np.nan,
            "eviction": np.nan,
            "layoff": np.nan,
            "jobTraining": np.nan,
        }
    )
    truth.loc[truth["challengeID"].isin(holdout_ids), "materialHardship"] = y[:n_holdout]
    truth.to_csv(root / "truth.csv", index=False)

    pd.DataFrame(
        [
            ["family_economics", "income", "baseline", "mother", "income_anchor", "synthetic income", True],
            ["employment", "employed", "baseline", "mother", "employed_anchor", "synthetic employment", True],
            ["housing", "moves", "year9", "pcg", "moves_anchor", "synthetic moves", True],
            ["social_support", "support", "year5", "mother", "support_anchor", "synthetic support", True],
            ["family_structure", "category", "year1", "mother", "category_anchor", "synthetic category", True],
            ["family_economics", "missing exact", "year1", "mother", "not_in_data", "missing", True],
            ["family_economics", "constant", "year1", "mother", "constant_anchor", "constant", True],
            ["family_economics", "mostly missing", "year1", "mother", "mostly_missing_anchor", "mostly missing", True],
        ],
        columns=[
            "domain",
            "concept",
            "wave",
            "respondent",
            "exact_column_name",
            "source_label",
            "include",
        ],
    ).to_csv(root / "feature_dictionary.csv", index=False)


class FFCWSPredictTests(unittest.TestCase):
    def test_full_pipeline_with_synthetic_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_synthetic_ds15(root)
            config = write_config(root)

            audit_features(config)
            prepare(config)
            train(config)
            learning_curve(config)
            summarize(config)

            out = root / "out"
            self.assertTrue((out / "feature_dictionary_resolved.csv").exists())
            self.assertTrue((out / "domain_feature_report.csv").exists())
            self.assertTrue((out / "model_metrics.csv").exists())
            self.assertTrue((out / "holdout_predictions.csv").exists())
            self.assertTrue((out / "prediction_range_diagnostics.csv").exists())
            self.assertTrue((out / "learning_curve.csv").exists())
            self.assertTrue((out / "power_law_fit.csv").exists())
            self.assertTrue((out / "summary_report.md").exists())

            manifest = json.loads((out / "run_manifest.json").read_text())
            self.assertEqual(manifest["n_holdout_ids"], 30)
            self.assertEqual(manifest["split_type"], "official_challenge")

            resolved = pd.read_csv(out / "feature_dictionary_resolved.csv")
            self.assertIn("missing_from_data", set(resolved["resolved_status"]))
            prepared_path = out / "prepared_material_hardship.csv"
            self.assertTrue(prepared_path.exists())
            prepared = pd.read_csv(prepared_path)
            train_ids = set(prepared.loc[prepared["split"].eq("train"), "challengeID"])
            holdout_ids = set(prepared.loc[prepared["split"].eq("holdout"), "challengeID"])
            self.assertTrue(train_ids.isdisjoint(holdout_ids))

            pre = json.loads((out / "preprocessing_report.json").read_text())
            ridge_report = pre["ridge__median"]["preprocessing"]
            self.assertIn("moves_anchor", ridge_report["missing_indicator_columns"])
            self.assertIn("constant_anchor", ridge_report["dropped"]["low_variance"])
            self.assertIn("mostly_missing_anchor", ridge_report["dropped"]["high_missing"])

    def test_missing_feature_dictionary_blocks_prepare(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_synthetic_ds15(root)
            (root / "feature_dictionary.csv").unlink()
            config = write_config(root)
            with self.assertRaises(Exception):
                prepare(config)

    def test_power_law_recovers_known_parameters(self):
        n = np.array([100, 200, 400, 800, 1600], dtype=float)
        c, alpha, epsilon = 2.0, 0.6, 0.03
        mse = c * n ** (-alpha) + epsilon
        params, _ = fit_curve(n, mse)
        self.assertAlmostEqual(params[0], c, places=3)
        self.assertAlmostEqual(params[1], alpha, places=3)
        self.assertAlmostEqual(params[2], epsilon, places=3)

    def test_truth_nonmissing_ids_define_holdout_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_synthetic_ds15(root, n=620, n_holdout=530)
            config = write_config(root, models=["mean_baseline"], n_draws=1, boot=5)
            prepare(config)
            manifest = json.loads((root / "out" / "run_manifest.json").read_text())
            self.assertEqual(manifest["n_holdout_ids"], 530)

    def test_read_table_supports_zip_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv = root / "inner.csv"
            csv.write_text("challengeID,x\n1,2\n")
            zip_path = root / "tables.zip"
            with ZipFile(zip_path, "w") as archive:
                archive.write(csv, "inner.csv")
            df = read_table(f"zip://{zip_path}!inner.csv")
            self.assertEqual(df.shape, (1, 2))
            self.assertEqual(int(df.loc[0, "x"]), 2)

    def test_native_nan_preprocessor_preserves_numeric_nan(self):
        X = pd.DataFrame(
            {
                "num": [1.0, -9, 3.0, np.nan],
                "cat": ["a", "b", None, "a"],
                "constant": [1, 1, 1, 1],
            }
        )
        pre = Preprocessor(imputation_strategy="native_nan")
        Xt = pre.fit_transform(X)
        self.assertIn("num", Xt.columns)
        self.assertTrue(Xt["num"].isna().any())
        self.assertFalse(any(col.endswith("__missing") for col in Xt.columns))
        self.assertEqual(pre.missing_indicator_columns_, [])

    def test_imputation_strategies_keep_same_row_count(self):
        X = pd.DataFrame(
            {
                "num": [1.0, -9, 3.0, np.nan, 5.0],
                "cat": ["a", "b", None, "a", "c"],
                "other": [2.0, 2.5, np.nan, 4.0, -8],
            }
        )
        median = Preprocessor(imputation_strategy="median")
        native = Preprocessor(imputation_strategy="native_nan")
        X_median = median.fit_transform(X)
        X_native = native.fit_transform(X)
        self.assertEqual(X_median.shape[0], X_native.shape[0])
        self.assertGreater(X_median.shape[1], X_native.shape[1])
        self.assertFalse(pd.isna(X_median).to_numpy().any())
        self.assertTrue(pd.isna(X_native).to_numpy().any())


if __name__ == "__main__":
    unittest.main()
