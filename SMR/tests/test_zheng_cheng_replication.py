from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from Zheng_Cheng_Replication.src.SHAP_experiment import evaluate_feature_count
from Zheng_Cheng_Replication.src.feature_sets import main as feature_sets_main
from Zheng_Cheng_Replication.src.sample_size import fit_power_law


class ZhengChengReplicationTests(unittest.TestCase):
    def test_shared_module_copies_are_byte_identical(self):
        root = Path(__file__).parents[1]
        for name in ("model_registry.py", "evaluation.py", "experiment.py"):
            nk_path = root / "NK_Grid" / "src" / name
            replication_path = root / "Zheng_Cheng_Replication" / "src" / name
            self.assertEqual(
                hashlib.sha256(nk_path.read_bytes()).hexdigest(),
                hashlib.sha256(replication_path.read_bytes()).hexdigest(),
                name,
            )

    def test_feature_cli_does_not_resume_across_outcomes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_path = root / "data.csv"
            output = root / "feature_results.csv"
            rng = np.random.default_rng(19)
            frame = pd.DataFrame(
                rng.normal(size=(60, 4)),
                columns=["Aset1_a", "Aset2_b", "Bset1_c", "Bset2_d"],
            )
            frame["outcome_a"] = frame["Aset1_a"] + rng.normal(size=len(frame))
            frame["outcome_b"] = frame["Bset1_c"] + rng.normal(size=len(frame))
            frame.to_csv(data_path, index=False)

            for outcome in ("outcome_a", "outcome_b"):
                argv = [
                    "feature_sets.py",
                    "--data",
                    str(data_path),
                    "--outcome",
                    outcome,
                    "--out",
                    str(output),
                    "--models",
                    "ols",
                    "--n-sizes",
                    "2",
                    "--n-draws",
                    "1",
                    "--batch-size",
                    "2",
                    "--n-jobs",
                    "1",
                ]
                with patch.object(sys, "argv", argv):
                    feature_sets_main()

            saved = pd.read_csv(output)
            self.assertEqual(saved["experiment_id"].nunique(), 2)
            self.assertEqual(set(saved["outcome"]), {"outcome_a", "outcome_b"})

    def test_shap_ordering_returns_both_directions(self):
        rng = np.random.default_rng(11)
        X = pd.DataFrame(rng.normal(size=(60, 3)), columns=["f1", "f2", "f3"])
        y = pd.Series(X["f1"] + rng.normal(0, 0.1, len(X)))
        rows = evaluate_feature_count(
            1,
            ["f1", "f2", "f3"],
            X.iloc[:45],
            X.iloc[45:],
            y.iloc[:45],
            y.iloc[45:],
            seed=12345,
        )
        self.assertEqual({row["direction"] for row in rows}, {"high_to_low", "low_to_high"})
        self.assertEqual(len(rows), 2)

    def test_power_law_fit_is_guarded_per_model(self):
        rows = []
        for n in [100, 200, 400, 800, 1600]:
            for draw in range(6):
                rows.append(
                    {
                        "model": "known",
                        "n_samples": n,
                        "mse": 1.5 * n ** -0.6 + 0.03 + (draw - 2.5) * 1e-5,
                        "status": "ok",
                    }
                )
        for n in [100, 200, 400]:
            rows.append(
                {"model": "too_few", "n_samples": n, "mse": 0.1, "status": "ok"}
            )
        fits = fit_power_law(pd.DataFrame(rows), bootstrap_iterations=20, seed=7)
        known = fits[fits["model"].eq("known")].iloc[0]
        failed = fits[fits["model"].eq("too_few")].iloc[0]
        self.assertAlmostEqual(float(known["epsilon"]), 0.03, places=3)
        self.assertIn(known["status"], {"stable", "unstable"})
        self.assertEqual(failed["status"], "fit_failed")

    def test_colab_notebook_is_valid_json_without_saved_outputs(self):
        path = Path(__file__).parents[1] / "Zheng_Cheng_Replication" / "colab_run.ipynb"
        notebook = json.loads(path.read_text())
        self.assertEqual(notebook["nbformat"], 4)
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertTrue(code_cells)
        self.assertTrue(all(cell.get("outputs") == [] for cell in code_cells))


if __name__ == "__main__":
    unittest.main()
