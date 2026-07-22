import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_panels import resolved_panels


class PreprocessingPanelsTest(unittest.TestCase):
    def test_three_strategies_cover_six_outcomes_with_isolated_paths(self):
        panels = resolved_panels(ROOT / "panels.yaml")
        strategy_panels = [
            (name, config)
            for name, config in panels
            if name.startswith(
                (
                    "ffc_median_mode_",
                    "ffc_median_missing_indicator_",
                    "ffc_tree_ordinal_",
                )
            )
        ]
        self.assertEqual(len(strategy_panels), 18)
        self.assertEqual(len({name for name, _ in strategy_panels}), 18)
        self.assertEqual(len({str(config.out) for _, config in strategy_panels}), 18)

        tree_panels = [
            config for name, config in strategy_panels if name.startswith("ffc_tree_ordinal_")
        ]
        self.assertEqual(len(tree_panels), 6)
        for config in tree_panels:
            self.assertEqual(config.models, ("xgboost", "lightgbm"))
            self.assertEqual(config.data.suffix, ".parquet")
            self.assertEqual(config.test_data.suffix, ".parquet")
            self.assertEqual(config.predictor_prefix, ["X_", "C_"])
            self.assertIn("tree_ordinal", config.dataset)

        for name, config in strategy_panels:
            self.assertIn("preprocessing", str(config.data))
            self.assertIn("preprocessing", str(config.feature_manifest))
            self.assertTrue(config.dataset.startswith("ffc_"), name)


if __name__ == "__main__":
    unittest.main()
