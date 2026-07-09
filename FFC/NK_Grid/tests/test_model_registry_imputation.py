import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model_registry import make_model


class ModelRegistryImputationTest(unittest.TestCase):
    def test_regression_models_keep_all_missing_feature_columns(self):
        X_train = pd.DataFrame({"X_all_missing": [np.nan]})
        y_train = pd.Series([1.25])
        X_test = pd.DataFrame({"X_all_missing": [np.nan, np.nan]})

        for model_name in ["ols", "random_forest"]:
            with self.subTest(model=model_name):
                model = make_model(model_name, seed=123, n_jobs=1, task="regression")
                model.fit(X_train, y_train)

                predictions = model.predict(X_test)

                self.assertEqual(len(predictions), 2)
                self.assertTrue(np.isfinite(predictions).all())


if __name__ == "__main__":
    unittest.main()
