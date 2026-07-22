import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model_registry import make_model


class TreeOrdinalModelSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(123)
        cls.X = pd.DataFrame(
            {
                "X_value": [np.nan if index % 7 == 0 else index / 10 for index in range(30)],
                "C_group": [np.nan if index % 11 == 0 else float(index % 3) for index in range(30)],
            }
        )
        cls.y_reg = rng.normal(size=30)
        cls.y_clf = np.array([index % 2 for index in range(30)])

    def test_xgboost_regression_and_classification_accept_numeric_nan(self):
        reg = make_model(
            "xgboost",
            seed=1,
            n_jobs=1,
            task="regression",
            params={
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "max_depth": 2,
                "eta": 0.3,
                "max_rounds": 2,
                "cv_folds": 2,
            },
        )
        reg.fit(self.X, self.y_reg)
        self.assertEqual(len(reg.predict(self.X.iloc[:4])), 4)

        clf = make_model(
            "xgboost",
            seed=1,
            n_jobs=1,
            task="classification",
            params={
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "max_depth": 2,
                "learning_rate": 0.3,
                "n_estimators": 2,
            },
        )
        clf.fit(self.X, self.y_clf)
        self.assertEqual(clf.predict_proba(self.X.iloc[:4]).shape, (4, 2))

    def test_lightgbm_regression_and_classification_accept_numeric_nan(self):
        reg = make_model(
            "lightgbm",
            seed=1,
            n_jobs=1,
            task="regression",
            params={
                "objective": "regression",
                "metric": "rmse",
                "learning_rate": 0.05,
                "num_leaves": 4,
                "min_data_in_leaf": 2,
                "verbosity": -1,
                "max_rounds": 2,
                "cv_folds": 2,
                "early_stopping_rounds": 1,
            },
        )
        reg.fit(self.X, self.y_reg)
        self.assertEqual(len(reg.predict(self.X.iloc[:4])), 4)

        clf = make_model(
            "lightgbm",
            seed=1,
            n_jobs=1,
            task="classification",
            params={
                "objective": "binary",
                "learning_rate": 0.05,
                "num_leaves": 4,
                "min_data_in_leaf": 2,
                "n_estimators": 2,
                "verbosity": -1,
            },
        )
        clf.fit(self.X, self.y_clf)
        self.assertEqual(clf.predict_proba(self.X.iloc[:4]).shape, (4, 2))


if __name__ == "__main__":
    unittest.main()
