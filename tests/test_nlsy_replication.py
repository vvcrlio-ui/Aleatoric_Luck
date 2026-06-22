from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from nlsy_replication.src.SHAP_experiment import evaluate_feature_count
from nlsy_replication.src.model_registry import MODEL_NAMES, make_model
from nlsy_replication.src.sample_size import fit_power_law


class NLSYReplicationTests(unittest.TestCase):
    def test_linear_registry_model_fits(self):
        rng = np.random.default_rng(7)
        X = pd.DataFrame(rng.normal(size=(60, 4)), columns=list("abcd"))
        y = 0.5 * X["a"] - 0.2 * X["b"] + rng.normal(0, 0.05, len(X))
        model = make_model("ridge", seed=12345, n_jobs=1)
        model.fit(X, y)
        self.assertEqual(model.predict(X).shape, (60,))
        self.assertIn("lightgbm", MODEL_NAMES)
        self.assertIn("bart", MODEL_NAMES)

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
        path = Path(__file__).parents[1] / "nlsy_replication" / "colab_run.ipynb"
        notebook = json.loads(path.read_text())
        self.assertEqual(notebook["nbformat"], 4)
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertTrue(code_cells)
        self.assertTrue(all(cell.get("outputs") == [] for cell in code_cells))


if __name__ == "__main__":
    unittest.main()
