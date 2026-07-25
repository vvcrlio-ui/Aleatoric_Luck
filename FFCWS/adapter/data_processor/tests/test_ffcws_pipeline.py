from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from aleatoric_nk_grid.ingest import load_input
from aleatoric_nk_grid.validate_input import validate_input

NK_ROOT = Path(__file__).resolve().parents[2]
PROCESSOR_SRC = NK_ROOT / "data_processor" / "src"
sys.path.insert(0, str(PROCESSOR_SRC))

from ffcws_data_processor.pipeline import DEFAULT_OUTCOMES, run_pipeline


class PipelineTest(unittest.TestCase):
    def test_end_to_end_outputs_and_content_identity_are_deterministic(self):
        n = 120
        ids = list(range(2000, 2000 + n))
        background = pd.DataFrame(
            {
                "challengeID": ids,
                "continuous": [
                    -6.0 if index == 0 else -9.0 if index == 1 else index / 10
                    for index in range(n)
                ],
                "category": [index % 3 for index in range(n)],
            }
        )
        train = pd.DataFrame({"challengeID": ids[:100]})
        test = pd.DataFrame({"challengeID": ids[100:]})
        for offset, outcome in enumerate(DEFAULT_OUTCOMES):
            if outcome in {"eviction", "layoff", "jobTraining"}:
                train[outcome] = [(index + offset) % 2 for index in range(100)]
                test[outcome] = [(index + offset) % 2 for index in range(20)]
            else:
                train[outcome] = np.linspace(0.0 + offset, 1.0 + offset, 100)
                test[outcome] = np.linspace(0.1 + offset, 0.9 + offset, 20)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            background_path = root / "background.dta"
            train_path = root / "train.csv"
            test_path = root / "test.csv"
            output_path = root / "output"
            config_path = root / "config.yaml"
            background.to_stata(
                background_path,
                write_index=False,
                value_labels={
                    "category": {0: "zero", 1: "one", 2: "two"},
                    "continuous": {-6: "not asked", -9: "missing"},
                },
            )
            train.to_csv(train_path, index=False)
            test.to_csv(test_path, index=False)
            config = {
                "paths": {
                    "background": str(background_path),
                    "train": str(train_path),
                    "test": str(test_path),
                    "output_root": str(output_path),
                },
                "id_column": "challengeID",
                "outcomes": list(DEFAULT_OUTCOMES),
                "strategies": [
                    "median_mode",
                    "median_missing_indicator",
                    "tree_ordinal",
                ],
                "schema": {
                    "min_valid_rate": 0.5,
                    "min_numeric_fraction": 0.95,
                    "categorical_max_levels": 15,
                    "min_binary_prevalence": 0.01,
                },
                "unknown_rate_threshold": 0.95,
            }
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            first = run_pipeline(config_path)
            second = run_pipeline(config_path)

            self.assertEqual(first, second)
            self.assertEqual(first["canonical_source_count"], 2)
            for strategy in config["strategies"]:
                strategy_root = output_path / strategy
                self.assertTrue((strategy_root / "feature_manifest.csv").exists())
                suffix = ".parquet" if strategy == "tree_ordinal" else ".csv"
                self.assertTrue((strategy_root / f"features{suffix}").exists())
                for split in ("train", "test"):
                    for outcome in DEFAULT_OUTCOMES:
                        self.assertTrue(
                            (
                                strategy_root
                                / "nk_inputs"
                                / f"ffc_{split}_{outcome}{suffix}"
                            ).exists()
                        )
                engine_schemas = first["strategies"][strategy]["engine_schemas"]
                definition_files = {
                    json.loads(Path(path).read_text(encoding="utf-8"))[
                        "feature_universe"
                    ]["definition_file"]
                    for path in engine_schemas.values()
                }
                self.assertEqual(
                    definition_files, {f"ffc_{strategy}.feature_universe.json"}
                )
                schema_path = Path(engine_schemas["gpa"])
                self.assertTrue(schema_path.exists())
                loaded = load_input(schema_path, "gpa")
                validated, groups = validate_input(
                    loaded,
                    "gpa",
                    models=("ols",),
                    min_n=10,
                    test_size=0.3,
                    seed=123,
                )
                self.assertEqual(len(groups), 2)
                self.assertIn("gpa", validated.train)
                self.assertIn("challengeID", validated.train)
                provenance_path = validated.schema.table.parent / "provenance.json"
                provenance_text = provenance_path.read_text(encoding="utf-8")
                self.assertNotIn(str(root), provenance_text)
                self.assertNotIn("challengeID", provenance_text)


if __name__ == "__main__":
    unittest.main()
