from __future__ import annotations

import json
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from NK_Grid.src.evaluation import (
    r2_against_training_mean,
    training_mean_null_mse,
)
from NK_Grid.src.experiment import (
    add_metadata,
    build_experiment_metadata,
    load_checkpoint,
    parallel_preference,
    write_checkpoint,
)
from NK_Grid.src.model_registry import MODEL_NAMES, make_model
from NK_Grid.src.nk_grid import (
    CLASSIFICATION_METRIC_COLUMNS,
    METRIC_COLUMNS,
    NKGridConfig,
    compute_classification_metrics,
    compute_regression_metrics,
    draw_orders,
    log2_size_grid,
    split_frame,
    run_nk_grid,
)
from NK_Grid.src.run_panels import main as run_panels_main


class NKGridTests(unittest.TestCase):
    def test_linear_registry_model_fits(self):
        rng = np.random.default_rng(7)
        X = pd.DataFrame(rng.normal(size=(60, 4)), columns=list("abcd"))
        y = 0.5 * X["a"] - 0.2 * X["b"] + rng.normal(0, 0.05, len(X))
        model = make_model("ridge", seed=12345, n_jobs=1)
        model.fit(X, y)
        self.assertEqual(model.predict(X).shape, (60,))
        self.assertIn("lightgbm", MODEL_NAMES)
        self.assertIn("bart", MODEL_NAMES)

    def test_classification_registry_model_predicts_probabilities(self):
        rng = np.random.default_rng(17)
        X = pd.DataFrame(rng.normal(size=(80, 4)), columns=list("abcd"))
        y = (X["a"] - 0.5 * X["b"] > 0).astype(int)
        for model_name in ("ols", "ridge", "lasso", "elastic_net", "random_forest"):
            model = make_model(model_name, seed=12345, n_jobs=1, task="classification")
            model.fit(X, y)
            probabilities = model.predict_proba(X)
            self.assertEqual(probabilities.shape, (80, 2), model_name)
            self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)), model_name)

    def test_classification_lasso_and_elastic_net_use_distinct_penalties(self):
        lasso = make_model("lasso", seed=12345, n_jobs=1, task="classification")
        elastic_net = make_model("elastic_net", seed=12345, n_jobs=1, task="classification")
        self.assertEqual(lasso.named_steps["logisticregression"].penalty, "l1")
        self.assertEqual(elastic_net.named_steps["logisticregression"].penalty, "elasticnet")
        self.assertNotEqual(
            lasso.named_steps["logisticregression"].penalty,
            elastic_net.named_steps["logisticregression"].penalty,
        )

    def test_bart_defaults_match_paper_and_use_process_parallelism(self):
        model = make_model("bart", seed=12345, n_jobs=1)
        self.assertEqual(model.n_trees, 200)
        self.assertEqual(model.n_samples, 1000)
        self.assertEqual(model.n_burn, 100)
        self.assertEqual(model.thin, 1.0)
        self.assertEqual(parallel_preference(["ridge", "bart"]), "processes")
        self.assertEqual(parallel_preference(["ridge"]), "threads")

    def test_training_mean_metric_uses_supplied_subset(self):
        y_test = np.array([0.0, 10.0])
        y_subset = np.array([0.0, 0.0])
        self.assertEqual(training_mean_null_mse(y_test, y_subset), 50.0)
        self.assertEqual(r2_against_training_mean(25.0, y_test, y_subset), 0.5)

    def test_experiment_identity_and_checkpoint_are_scoped(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data = root / "data.csv"
            output = root / "results.csv"
            data.write_text("Aset1_x,outcome_a,outcome_b\n1,2,3\n")
            common = {
                "kind": "feature_sets",
                "data_path": data,
                "test_size": 0.3,
                "split_seed": 12345,
                "extra": {"n_sizes": 2},
            }
            first = build_experiment_metadata(outcome="outcome_a", **common)
            second = build_experiment_metadata(outcome="outcome_b", **common)
            self.assertNotEqual(first["experiment_id"], second["experiment_id"])

            row = {"model": "ridge", "k": 1, "seed": 12345, "status": "ok"}
            write_checkpoint(
                pd.DataFrame(),
                [add_metadata(row, first), add_metadata(row, second)],
                output,
                key_columns=["model", "k", "seed"],
                sort_columns=["model", "k", "seed"],
            )
            saved = load_checkpoint(output)
            self.assertEqual(len(saved), 2)
            self.assertEqual(set(saved["outcome"]), {"outcome_a", "outcome_b"})

    def test_legacy_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "legacy.csv"
            pd.DataFrame([{"model": "ridge", "k": 1}]).to_csv(output, index=False)
            with self.assertRaisesRegex(ValueError, "lacks experiment metadata"):
                load_checkpoint(output)

    def test_nk_log2_size_grid_is_deduplicated_and_clipped(self):
        grid = log2_size_grid(total=1_000, n_sizes=8, max_size=100)
        self.assertEqual(grid[0], 1)
        self.assertEqual(grid[-1], 100)
        self.assertEqual(len(grid), len(set(grid)))
        self.assertTrue(np.all(np.diff(np.log2(grid)) > 0))
        self.assertTrue(np.all(grid <= 100))
        self.assertTrue(np.array_equal(log2_size_grid(10, 1), np.array([10])))

    def test_nk_regression_metrics_known_values(self):
        metrics = compute_regression_metrics(
            y_test=np.array([1.0, 2.0, 3.0]),
            y_pred=np.array([1.0, 2.0, 4.0]),
            y_train=np.array([1.0, 2.0, 3.0]),
        )
        self.assertEqual(len(metrics), 30)
        self.assertEqual(len(METRIC_COLUMNS), 30)
        self.assertAlmostEqual(metrics["r2_test"], 0.5)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(1.0 / 3.0))
        self.assertAlmostEqual(metrics["spearman_rho"], 1.0)
        self.assertAlmostEqual(metrics["pinball_q10"], 0.3)
        self.assertAlmostEqual(metrics["pinball_q50"], 0.5 * metrics["mae"])
        self.assertAlmostEqual(metrics["pearson_r2"], metrics["pearson_r"] ** 2)
        self.assertAlmostEqual(metrics["rsr"], metrics["rmse"] / np.std([1.0, 2.0, 3.0]))
        self.assertAlmostEqual(metrics["top_decile_hit_rate"], 1.0)
        self.assertAlmostEqual(metrics["bottom_decile_hit_rate"], 1.0)
        self.assertAlmostEqual(metrics["wasserstein_distance"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["ks_statistic"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["mase"], 0.5)

    def test_nk_regression_metrics_degenerate_inputs_return_nan(self):
        metrics = compute_regression_metrics(
            y_test=np.array([0.0, 0.0, 0.0]),
            y_pred=np.array([0.0, 0.0, 0.0]),
            y_train=np.array([0.0, 0.0, 0.0]),
        )
        for key in ("spearman_rho", "pearson_r", "pearson_r2", "kendall_tau", "rsr", "cv_rmse", "mase"):
            self.assertTrue(np.isnan(metrics[key]), key)

    def test_nk_classification_metrics_known_values(self):
        y_test = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.4, 0.35, 0.8])
        y_train = np.array([0, 1, 0, 1])
        metrics = compute_classification_metrics(y_test, probabilities, y_train)
        model_loglik = np.log(0.9) + np.log(0.6) + np.log(0.35) + np.log(0.8)
        null_loglik = 4 * np.log(0.5)
        self.assertEqual(len(metrics), 8)
        self.assertEqual(len(CLASSIFICATION_METRIC_COLUMNS), 8)
        self.assertAlmostEqual(metrics["roc_auc"], 0.75)
        self.assertAlmostEqual(metrics["brier"], 0.158125)
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["mcfadden_pseudo_r2"], 1.0 - model_loglik / null_loglik)

    def test_nk_classification_metrics_single_class_probability_metrics_nan(self):
        metrics = compute_classification_metrics(
            y_test=np.array([1, 1, 1]),
            y_score=np.array([0.2, 0.8, 0.9]),
            y_train=np.array([0, 1, 1, 0]),
        )
        for key in ("roc_auc", "pr_auc", "log_loss"):
            self.assertTrue(np.isnan(metrics[key]), key)

    def test_nk_seed_changes_split_and_draw_changes_subsample(self):
        frame = pd.DataFrame(
            {
                "Aset1_a": np.arange(30),
                "Bset1_b": np.arange(30) * 2,
                "outcome": np.arange(30, dtype=float),
            }
        )
        predictors = ["Aset1_a", "Bset1_b"]
        first = split_frame(frame, predictors, "outcome", test_size=0.3, seed=1)
        second = split_frame(frame, predictors, "outcome", test_size=0.3, seed=2)
        self.assertNotEqual(list(first.X_train.index), list(second.X_train.index))

        order_a = draw_orders(first.X_train.index, predictors, seed=1, draw=0)
        order_b = draw_orders(first.X_train.index, predictors, seed=1, draw=1)
        self.assertNotEqual(list(order_a.row_index), list(order_b.row_index))

    def test_nk_classification_split_is_stratified(self):
        frame = pd.DataFrame(
            {
                "Aset1_a": np.arange(100),
                "Bset1_b": np.arange(100) * 2,
                "employed": np.r_[np.zeros(70, dtype=int), np.ones(30, dtype=int)],
            }
        )
        split = split_frame(
            frame,
            ["Aset1_a", "Bset1_b"],
            "employed",
            test_size=0.3,
            seed=4,
            task="classification",
        )
        self.assertAlmostEqual(split.y_train.mean(), 0.3, delta=0.05)
        self.assertAlmostEqual(split.y_test.mean(), 0.3, delta=0.05)

    def test_nk_grid_end_to_end_schema_and_row_count(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_path = root / "synthetic.csv"
            out_path = root / "nk_grid.csv"
            self._write_nk_synthetic_data(data_path)

            config = NKGridConfig(
                data=data_path,
                out=out_path,
                dataset="synthetic",
                outcome="outcome",
                models=("ols",),
                seed=10,
                test_size=0.3,
                n_seeds=2,
                n_draws=2,
                n_sizes_n=2,
                n_sizes_k=2,
                max_n=20,
                max_k=3,
                batch_size=4,
                n_jobs=1,
            )
            run_nk_grid(config)
            saved = pd.read_csv(out_path)
            self.assertEqual(len(saved), 16)
            expected_columns = {
                "experiment_id",
                "dataset",
                "outcome",
                "model",
                "seed",
                "draw",
                "N",
                "K",
                "split_random_state",
                "n_train_total",
                "n_features_total",
                "r2_test",
                "skill_score_pct",
                "rmse",
                "mae",
                "medae",
                "max_error",
                "nrmse",
                "spearman_rho",
                "pearson_r",
                "kendall_tau",
                "ccc",
                "explained_variance",
                "mean_bias",
                "median_bias",
                "pinball_q10",
                "pinball_q90",
                "d2_absolute_error",
                "pinball_q05",
                "pinball_q25",
                "pinball_q50",
                "pinball_q75",
                "pinball_q95",
                "ks_statistic",
                "wasserstein_distance",
                "top_decile_hit_rate",
                "bottom_decile_hit_rate",
                "rsr",
                "cv_rmse",
                "mase",
                "pearson_r2",
                "status",
                "error",
                "experiment_kind",
                "data_sha256",
                "data_path",
                "test_size",
                "split_seed",
            }
            self.assertTrue(expected_columns.issubset(saved.columns))
            self.assertEqual(set(saved["status"]), {"ok"})
            self.assertTrue((saved["N"] <= saved["n_train_total"]).all())
            self.assertTrue((saved["K"] <= saved["n_features_total"]).all())

    def test_nk_grid_predictor_prefix_is_configurable_for_other_datasets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_path = root / "other_paper.csv"
            out_path = root / "nk_grid.csv"
            rng = np.random.default_rng(41)
            frame = pd.DataFrame(
                rng.normal(size=(80, 4)),
                columns=["Feat_income", "Feat_age", "Cov_region", "Cov_edu"],
            )
            frame["target"] = (
                0.6 * frame["Feat_income"] - 0.3 * frame["Cov_region"]
                + rng.normal(0, 0.1, len(frame))
            )
            frame.to_csv(data_path, index=False)

            default_prefix_config = NKGridConfig(
                data=data_path,
                out=out_path,
                dataset="other_paper",
                outcome="target",
                models=("ols",),
                seed=11,
                test_size=0.3,
                n_seeds=1,
                n_draws=1,
                n_sizes_n=1,
                n_sizes_k=1,
                max_n=20,
                max_k=2,
                batch_size=1,
                n_jobs=1,
            )
            with self.assertRaisesRegex(ValueError, "No predictor columns found"):
                run_nk_grid(default_prefix_config)

            custom_prefix_config = NKGridConfig(
                data=data_path,
                out=out_path,
                dataset="other_paper",
                outcome="target",
                models=("ols",),
                seed=11,
                test_size=0.3,
                n_seeds=1,
                n_draws=1,
                n_sizes_n=1,
                n_sizes_k=1,
                max_n=20,
                max_k=2,
                batch_size=1,
                n_jobs=1,
                predictor_prefix=("Feat", "Cov"),
            )
            run_nk_grid(custom_prefix_config)
            saved = pd.read_csv(out_path)
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved.loc[0, "status"], "ok")
            self.assertEqual(saved.loc[0, "n_features_total"], 4)

    def test_nk_grid_checkpoint_resume_completes_without_duplicates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_path = root / "synthetic.csv"
            out_path = root / "nk_grid.csv"
            self._write_nk_synthetic_data(data_path)
            config = NKGridConfig(
                data=data_path,
                out=out_path,
                dataset="synthetic",
                outcome="outcome",
                models=("ols",),
                seed=20,
                test_size=0.3,
                n_seeds=1,
                n_draws=2,
                n_sizes_n=2,
                n_sizes_k=2,
                max_n=20,
                max_k=3,
                batch_size=2,
                n_jobs=1,
            )
            run_nk_grid(config, max_jobs=4)
            partial = pd.read_csv(out_path)
            self.assertEqual(len(partial), 4)
            run_nk_grid(config)
            saved = pd.read_csv(out_path)
            self.assertEqual(len(saved), 8)
            self.assertEqual(
                len(saved[["model", "seed", "draw", "N", "K"]].drop_duplicates()),
                8,
            )

    def test_nk_grid_skips_bart_below_minimum_without_fitting(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_path = root / "synthetic.csv"
            out_path = root / "nk_grid.csv"
            self._write_nk_synthetic_data(data_path)
            config = NKGridConfig(
                data=data_path,
                out=out_path,
                dataset="synthetic",
                outcome="outcome",
                models=("bart",),
                seed=22,
                test_size=0.3,
                n_seeds=1,
                n_draws=1,
                n_sizes_n=1,
                n_sizes_k=1,
                max_n=5,
                max_k=1,
                batch_size=1,
                n_jobs=1,
                bart_min_n=10,
                bart_min_k=2,
            )
            with patch("NK_Grid.src.nk_grid.make_model") as make_model_mock:
                run_nk_grid(config)
            make_model_mock.assert_not_called()
            saved = pd.read_csv(out_path)
            self.assertEqual(saved.loc[0, "status"], "skipped")
            self.assertEqual(saved.loc[0, "error"], "below BART minimum N/K floor")
            self.assertTrue(saved.loc[0, list(METRIC_COLUMNS)].isna().all())

    def test_nk_grid_logs_progress_to_stderr(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_path = root / "synthetic.csv"
            out_path = root / "nk_grid.csv"
            self._write_nk_synthetic_data(data_path)
            config = NKGridConfig(
                data=data_path,
                out=out_path,
                dataset="synthetic",
                outcome="outcome",
                models=("ols",),
                seed=25,
                test_size=0.3,
                n_seeds=1,
                n_draws=1,
                n_sizes_n=1,
                n_sizes_k=1,
                max_n=20,
                max_k=3,
                batch_size=1,
                n_jobs=1,
            )
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                run_nk_grid(config)
            logs = stderr.getvalue()
            self.assertIn("[nk_grid]", logs)
            self.assertIn("loaded data", logs)
            self.assertIn("jobs total=1", logs)
            self.assertIn("batch 1/1", logs)
            self.assertIn("wrote checkpoint", logs)

    def test_nk_grid_failed_rows_include_expanded_metrics_as_nan(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_path = root / "synthetic.csv"
            out_path = root / "nk_grid.csv"
            self._write_nk_synthetic_data(data_path)
            config = NKGridConfig(
                data=data_path,
                out=out_path,
                dataset="synthetic",
                outcome="outcome",
                models=("ridge",),
                seed=30,
                test_size=0.3,
                n_seeds=1,
                n_draws=1,
                n_sizes_n=1,
                n_sizes_k=1,
                max_n=1,
                max_k=1,
                batch_size=1,
                n_jobs=1,
            )
            run_nk_grid(config)
            saved = pd.read_csv(out_path)
            self.assertEqual(saved.loc[0, "status"], "failed")
            expanded_metrics = [
                "pinball_q05",
                "pinball_q25",
                "pinball_q50",
                "pinball_q75",
                "pinball_q95",
                "ks_statistic",
                "wasserstein_distance",
                "top_decile_hit_rate",
                "bottom_decile_hit_rate",
                "rsr",
                "cv_rmse",
                "mase",
                "pearson_r2",
            ]
            self.assertTrue(saved.loc[0, expanded_metrics].isna().all())

    def test_nk_classification_grid_end_to_end_schema_and_row_count(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_path = root / "synthetic.csv"
            out_path = root / "nk_grid_clf.csv"
            self._write_nk_synthetic_data(data_path)
            config = NKGridConfig(
                data=data_path,
                out=out_path,
                dataset="synthetic",
                outcome="employed",
                models=("ols",),
                seed=40,
                test_size=0.3,
                n_seeds=2,
                n_draws=2,
                n_sizes_n=1,
                n_sizes_k=2,
                max_n=30,
                max_k=3,
                batch_size=4,
                n_jobs=1,
                task="classification",
            )
            run_nk_grid(config)
            saved = pd.read_csv(out_path)
            self.assertEqual(len(saved), 8)
            expected_columns = {
                "experiment_id",
                "dataset",
                "outcome",
                "task",
                "model",
                "seed",
                "draw",
                "N",
                "K",
                "split_random_state",
                "n_train_total",
                "n_features_total",
                "roc_auc",
                "pr_auc",
                "brier",
                "log_loss",
                "balanced_accuracy",
                "f1",
                "accuracy",
                "mcfadden_pseudo_r2",
                "status",
                "error",
            }
            self.assertTrue(expected_columns.issubset(saved.columns))
            self.assertEqual(set(saved["task"]), {"classification"})
            self.assertEqual(set(saved["status"]), {"ok"})

    def test_nk_classification_checkpoint_resume_completes_without_duplicates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_path = root / "synthetic.csv"
            out_path = root / "nk_grid_clf.csv"
            self._write_nk_synthetic_data(data_path)
            config = NKGridConfig(
                data=data_path,
                out=out_path,
                dataset="synthetic",
                outcome="employed",
                models=("ols",),
                seed=50,
                test_size=0.3,
                n_seeds=1,
                n_draws=2,
                n_sizes_n=1,
                n_sizes_k=2,
                max_n=30,
                max_k=3,
                batch_size=2,
                n_jobs=1,
                task="classification",
            )
            run_nk_grid(config, max_jobs=2)
            partial = pd.read_csv(out_path)
            self.assertEqual(len(partial), 2)
            run_nk_grid(config)
            saved = pd.read_csv(out_path)
            self.assertEqual(len(saved), 4)
            self.assertEqual(
                len(saved[["model", "seed", "draw", "N", "K"]].drop_duplicates()),
                4,
            )

    def test_helpers_logging_is_shared_by_nk_grid_and_run_panels(self):
        from NK_Grid.src import helpers_logging
        from NK_Grid.src import nk_grid as nk_grid_module
        from NK_Grid.src import run_panels as run_panels_module

        self.assertIs(nk_grid_module.log_progress, helpers_logging.log_progress)
        self.assertIs(run_panels_module.log_progress, helpers_logging.log_progress)

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            helpers_logging.log_progress("x")
        self.assertRegex(stderr.getvalue(), r"^\[nk_grid\] \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} x\n$")

    def test_dev_preset_uses_five_n_and_k_grid_points(self):
        from NK_Grid.src.run_panels import PRESETS

        self.assertEqual(PRESETS["dev"]["n_sizes_n"], 5)
        self.assertEqual(PRESETS["dev"]["n_sizes_k"], 5)

    def test_run_panels_dry_run_prints_resolved_config_without_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = self._write_panel_manifest(root)
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                run_panels_main(["--manifest", str(manifest), "--dry-run"])
            payload = json.loads(stdout.getvalue())
            self.assertEqual([panel["name"] for panel in payload["panels"]], ["reg_panel", "clf_panel"])
            first_config = payload["panels"][0]["config"]
            self.assertEqual(first_config["n_seeds"], 2)
            self.assertEqual(first_config["n_draws"], 2)
            self.assertEqual(first_config["n_sizes_n"], 2)
            self.assertEqual(first_config["n_sizes_k"], 2)
            self.assertFalse((root / "outputs" / "reg.csv").exists())
            self.assertFalse((root / "outputs" / "clf.csv").exists())

    def test_run_panels_yaml_manifest_without_grid_overrides_uses_dev_five_points(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_path = root / "synthetic.csv"
            self._write_nk_synthetic_data(data_path)
            manifest = root / "panels.yaml"
            manifest.write_text(
                "\n".join(
                    [
                        "panels:",
                        "  - name: dev_panel",
                        f"    data: {data_path}",
                        "    dataset: synthetic",
                        "    outcome: outcome",
                        "    task: regression",
                        "    models:",
                        "      - ols",
                        "    preset: dev",
                        f"    out: {root / 'outputs' / 'dev.csv'}",
                        "    max_n: 20",
                        "    max_k: 3",
                        "    batch_size: 3",
                        "",
                    ]
                )
            )
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                run_panels_main(["--manifest", str(manifest), "--dry-run"])
            payload = json.loads(stdout.getvalue())
            config = payload["panels"][0]["config"]
            self.assertEqual(config["n_sizes_n"], 5)
            self.assertEqual(config["n_sizes_k"], 5)

    def test_run_panels_yaml_manifest_missing_panels_has_clear_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "panels.yaml"
            manifest.write_text("not_panels: []\n")
            with self.assertRaisesRegex(ValueError, "YAML object with a 'panels' list"):
                run_panels_main(["--manifest", str(manifest), "--dry-run"])

    def test_run_panels_only_runs_selected_panel(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = self._write_panel_manifest(root)
            run_panels_main(["--manifest", str(manifest), "--only", "reg_panel"])
            self.assertTrue((root / "outputs" / "reg.csv").exists())
            self.assertFalse((root / "outputs" / "clf.csv").exists())

    def test_run_panels_runs_two_panels_with_independent_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = self._write_panel_manifest(root)
            run_panels_main(["--manifest", str(manifest)])
            reg = pd.read_csv(root / "outputs" / "reg.csv")
            clf = pd.read_csv(root / "outputs" / "clf.csv")
            self.assertEqual(len(reg), 16)
            self.assertEqual(len(clf), 16)
            self.assertIn("r2_test", reg.columns)
            self.assertIn("roc_auc", clf.columns)
            self.assertEqual(set(clf["task"]), {"classification"})
            self.assertEqual(set(reg["status"]), {"ok"})
            self.assertEqual(set(clf["status"]), {"ok"})

    def test_run_panels_resume_does_not_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = self._write_panel_manifest(root)
            run_panels_main(["--manifest", str(manifest), "--only", "reg_panel", "--max-jobs", "3"])
            partial = pd.read_csv(root / "outputs" / "reg.csv")
            self.assertEqual(len(partial), 3)
            run_panels_main(["--manifest", str(manifest), "--only", "reg_panel"])
            saved = pd.read_csv(root / "outputs" / "reg.csv")
            self.assertEqual(len(saved), 16)
            self.assertEqual(
                len(saved[["model", "seed", "draw", "N", "K"]].drop_duplicates()),
                16,
            )

    def _write_nk_synthetic_data(self, data_path: Path) -> None:
        rng = np.random.default_rng(33)
        frame = pd.DataFrame(
            rng.normal(size=(80, 5)),
            columns=["Aset1_a", "Aset2_b", "Aset3_c", "Bset1_d", "Bset2_e"],
        )
        frame["outcome"] = (
            0.7 * frame["Aset1_a"] - 0.4 * frame["Bset1_d"] + rng.normal(0, 0.1, len(frame))
        )
        frame["employed"] = (
            frame["Aset1_a"] - 0.5 * frame["Bset1_d"] + rng.normal(0, 0.2, len(frame)) > 0
        ).astype(int)
        frame.to_csv(data_path, index=False)

    def _write_panel_manifest(self, root: Path) -> Path:
        data_path = root / "synthetic.csv"
        self._write_nk_synthetic_data(data_path)
        manifest = root / "panels.yaml"
        manifest.write_text(
            "\n".join(
                [
                    "panels:",
                    "  - name: reg_panel",
                    f"    data: {data_path}",
                    "    dataset: synthetic",
                    "    outcome: outcome",
                    "    task: regression",
                    "    models:",
                    "      - ols",
                    "    preset: dev",
                    f"    out: {root / 'outputs' / 'reg.csv'}",
                    "    n_sizes_n: 2",
                    "    n_sizes_k: 2",
                    "    max_n: 20",
                    "    max_k: 3",
                    "    batch_size: 3",
                    "  - name: clf_panel",
                    f"    data: {data_path}",
                    "    dataset: synthetic",
                    "    outcome: employed",
                    "    task: classification",
                    "    models:",
                    "      - ols",
                    "    preset: dev",
                    f"    out: {root / 'outputs' / 'clf.csv'}",
                    "    n_sizes_n: 1",
                    "    n_sizes_k: 4",
                    "    max_n: 30",
                    "    max_k: 5",
                    "    batch_size: 4",
                    "",
                ]
            )
        )
        return manifest


if __name__ == "__main__":
    unittest.main()
