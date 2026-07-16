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
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import r2_score

from NK_Grid.src import model_registry as model_registry_module
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
from NK_Grid.src.model_registry import MODEL_NAMES, SUPPORTED_MODEL_NAMES, make_model
from NK_Grid.src.nk_grid import (
    CLASSIFICATION_METRIC_COLUMNS,
    METRIC_COLUMNS,
    NKGridConfig,
    REGRESSION_CV_MIN_N,
    compute_classification_metrics,
    compute_regression_metrics,
    _constant_prediction,
    _model_converged,
    draw_orders,
    external_test_split,
    log2_size_grid,
    split_frame,
    run_nk_grid,
)
from NK_Grid.src.run_panels import main as run_panels_main
from NK_Grid.src.run_panels import resolve_panel
from NK_Grid.src.tune_anchors import anchor_candidates


class DummyRegressor:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=float)


class DummyClassifier:
    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        probabilities = np.zeros((len(X), 2), dtype=float)
        probabilities[:, 0] = 0.6
        probabilities[:, 1] = 0.4
        return probabilities


class FailingRegressor:
    def fit(self, X, y):
        raise ValueError("synthetic fit failure")


class NKGridTests(unittest.TestCase):
    @staticmethod
    def _fast_new_model_params(task: str, model_name: str) -> dict:
        params = model_registry_module.load_model_params(
            model_registry_module.DEFAULT_MODEL_PARAMS_PATH,
            task=task,
            models=[model_name],
        )[model_name]
        if model_name == "extra_trees":
            params["n_estimators"] = 20
        elif model_name == "shallow_neural_network":
            params["max_iter"] = 50
        elif model_name == "super_learner":
            params["n_estimators"] = 20
            params["lgbm_n_estimators"] = 20
            params["max_iter"] = 50
        return params

    def test_linear_registry_model_fits(self):
        rng = np.random.default_rng(7)
        X = pd.DataFrame(rng.normal(size=(60, 4)), columns=list("abcd"))
        y = 0.5 * X["a"] - 0.2 * X["b"] + rng.normal(0, 0.05, len(X))
        model = make_model("ridge", seed=12345, n_jobs=1)
        model.fit(X, y)
        self.assertEqual(model.predict(X).shape, (60,))
        self.assertIn("lightgbm", MODEL_NAMES)
        self.assertIn("bart", SUPPORTED_MODEL_NAMES)

    def test_expanded_model_space_matches_specification(self):
        self.assertEqual(
            MODEL_NAMES,
            (
                "ols",
                "ridge",
                "lasso",
                "elastic_net",
                "random_forest",
                "xgboost",
                "lightgbm",
                "shallow_neural_network",
                "extra_trees",
                "super_learner",
            ),
        )

    def test_new_models_fit_regression_and_classification(self):
        rng = np.random.default_rng(29)
        X = pd.DataFrame(
            rng.normal(size=(40, 8)), columns=[f"x{index}" for index in range(8)]
        )
        y = X["x0"] - 0.4 * X["x1"] + rng.normal(0, 0.1, len(X))
        y_binary = (y > np.median(y)).astype(int)

        for model_name in (
            "shallow_neural_network",
            "extra_trees",
            "super_learner",
        ):
            regressor = make_model(
                model_name,
                seed=12345,
                n_jobs=1,
                params=self._fast_new_model_params("regression", model_name),
            )
            regressor.fit(X, y)
            self.assertEqual(regressor.predict(X).shape, (40,), model_name)

            classifier = make_model(
                model_name,
                seed=12345,
                n_jobs=1,
                task="classification",
                params=self._fast_new_model_params("classification", model_name),
            )
            classifier.fit(X, y_binary)
            probabilities = classifier.predict_proba(X)
            self.assertEqual(probabilities.shape, (40, 2), model_name)
            self.assertTrue(
                np.all((probabilities >= 0) & (probabilities <= 1)), model_name
            )

    def test_super_learner_has_four_plain_base_families(self):
        rng = np.random.default_rng(31)
        X = pd.DataFrame(rng.normal(size=(40, 8)))
        y = X[0] - 0.25 * X[1] + rng.normal(0, 0.1, len(X))
        y_binary = (y > np.median(y)).astype(int)

        regressor = make_model(
            "super_learner",
            seed=12345,
            n_jobs=1,
            params=self._fast_new_model_params("regression", "super_learner"),
        ).fit(X, y)
        regression_bases = dict(regressor.model_.estimators)
        self.assertEqual(
            set(regression_bases),
            {"ridge", "extra_trees", "lightgbm", "shallow_nn"},
        )
        self.assertIsInstance(
            regression_bases["ridge"][-1], model_registry_module.RidgeCV
        )
        self.assertIsInstance(
            regression_bases["shallow_nn"][-1], TransformedTargetRegressor
        )
        self.assertNotIsInstance(
            regression_bases["lightgbm"][-1],
            model_registry_module.LightGBMCVRegressor,
        )

        classifier = make_model(
            "super_learner",
            seed=12345,
            n_jobs=1,
            task="classification",
            params=self._fast_new_model_params("classification", "super_learner"),
        ).fit(X, y_binary)
        classification_bases = dict(classifier.model_.estimators)
        self.assertEqual(
            set(classification_bases),
            {"logistic", "extra_trees", "lightgbm", "shallow_nn"},
        )
        self.assertIsInstance(
            classifier.model_.final_estimator_, model_registry_module.LogisticRegression
        )
        self.assertNotIsInstance(
            classification_bases["lightgbm"][-1],
            model_registry_module.LightGBMCVRegressor,
        )

    def test_shallow_nn_regression_standardizes_large_scale_target(self):
        rng = np.random.default_rng(37)
        X = pd.DataFrame(rng.normal(size=(40, 8)))
        y = 10.0 + X[0] - 0.5 * X[1] + rng.normal(0, 0.1, len(X))
        model = make_model(
            "shallow_neural_network",
            seed=12345,
            n_jobs=1,
            params=self._fast_new_model_params(
                "regression", "shallow_neural_network"
            ),
        )

        self.assertIsInstance(model[-1], TransformedTargetRegressor)
        predictions = model.fit(X, y).predict(X)
        self.assertTrue(np.isfinite(r2_score(y, predictions)))

    def test_super_learner_passthrough_rejects_nan_features(self):
        X = pd.DataFrame({"x1": [0.0, 1.0, np.nan, 3.0], "x2": [1, 0, 1, 0]})
        outcomes = {
            "regression": pd.Series([0.0, 1.0, 2.0, 3.0]),
            "classification": pd.Series([0, 0, 1, 1]),
        }
        for task, y in outcomes.items():
            with self.subTest(task=task):
                params = self._fast_new_model_params(task, "super_learner")
                params["passthrough"] = True
                model = make_model(
                    "super_learner",
                    seed=12345,
                    n_jobs=1,
                    task=task,
                    params=params,
                )
                with self.assertRaisesRegex(ValueError, "passthrough=True.*NaN"):
                    model.fit(X, y)

        regression_params = self._fast_new_model_params(
            "regression", "super_learner"
        )
        make_model(
            "super_learner",
            seed=12345,
            n_jobs=1,
            params=regression_params,
        ).fit(X, outcomes["regression"])
        regression_params["passthrough"] = True
        make_model(
            "super_learner",
            seed=12345,
            n_jobs=1,
            params=regression_params,
        ).fit(X.fillna(2.0), outcomes["regression"])

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

    def test_constant_prediction_uses_exact_correlation_semantics(self):
        self.assertTrue(_constant_prediction([2.0, 2.0, 2.0]))
        self.assertTrue(_constant_prediction([np.nan, np.inf]))
        self.assertFalse(_constant_prediction([2.0, 2.0 + 1e-12]))

    def test_convergence_is_read_from_estimator_attributes(self):
        class IterativeEstimator:
            def __init__(self, n_iter: int, max_iter: int):
                self.n_iter_ = n_iter
                self.max_iter = max_iter

        self.assertTrue(_model_converged(IterativeEstimator(4, 5)))
        self.assertFalse(_model_converged(IterativeEstimator(5, 5)))
        self.assertTrue(_model_converged(DummyRegressor()))

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

    def test_n_grid_can_start_at_ten_while_k_grid_still_starts_at_one(self):
        n_grid = log2_size_grid(100, 8, min_size=10)
        k_grid = log2_size_grid(100, 8)

        self.assertEqual(n_grid[0], 10)
        self.assertEqual(k_grid[0], 1)

    def test_large_run_requires_explicit_noninteractive_authorization(self):
        config = NKGridConfig(
            data=Path("missing.csv"),
            out=Path("missing-output.csv"),
            dataset="synthetic",
            outcome="outcome",
            models=("ols",),
            seed=1,
            test_size=0.3,
            n_seeds=100,
            n_draws=50,
            n_sizes_n=20,
            n_sizes_k=20,
            max_n=0,
            max_k=0,
            batch_size=50,
            n_jobs=1,
        )
        with self.assertRaisesRegex(ValueError, "--allow-large-run"):
            run_nk_grid(config)

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            estimate = run_nk_grid(config, dry_run=True)
        self.assertEqual(estimate["top_level_model_cells"], 2_000_000)
        self.assertEqual(json.loads(stdout.getvalue()), estimate)

    def test_anchor_candidate_grid_matches_preregistered_search(self):
        self.assertEqual(len(anchor_candidates("random_forest")), 6)
        self.assertEqual(len(anchor_candidates("extra_trees")), 6)
        self.assertEqual(len(anchor_candidates("lightgbm")), 3)
        self.assertEqual(len(anchor_candidates("shallow_neural_network")), 8)
        self.assertEqual(
            {row["min_data_in_leaf"] for row in anchor_candidates("lightgbm")},
            {5, 10, 20},
        )

    def test_model_params_yaml_covers_active_models_for_both_tasks(self):
        models = {
            "ols",
            "ridge",
            "lasso",
            "elastic_net",
            "random_forest",
            "xgboost",
            "lightgbm",
            "shallow_neural_network",
            "extra_trees",
            "super_learner",
        }
        path = Path(__file__).resolve().parents[1] / "NK_Grid" / "model_params.yaml"

        regression = model_registry_module.load_model_params(
            path, task="regression", models=models
        )
        classification = model_registry_module.load_model_params(
            path, task="classification", models=models
        )

        self.assertEqual(set(regression), models)
        self.assertEqual(set(classification), models)
        self.assertEqual(regression["random_forest"]["n_estimators"], 500)
        self.assertEqual(regression["xgboost"]["max_rounds"], 90)
        self.assertEqual(classification["elastic_net"]["l1_ratio"], 0.5)
        expected_lgbm_keys = {
            "lgbm_n_estimators",
            "lgbm_learning_rate",
            "lgbm_num_leaves",
            "lgbm_min_data_in_leaf",
        }
        self.assertTrue(expected_lgbm_keys.issubset(regression["super_learner"]))
        self.assertTrue(expected_lgbm_keys.issubset(classification["super_learner"]))
        self.assertEqual(regression["super_learner"]["cv"], 5)
        self.assertEqual(classification["super_learner"]["cv"], 5)

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

    def test_external_test_split_drops_missing_outcomes_independently(self):
        train = pd.DataFrame(
            {
                "Aset1_a": [1, 2, 3],
                "Bset1_b": [4, 5, 6],
                "outcome": [10.0, np.nan, 30.0],
            },
            index=[10, 11, 12],
        )
        test = pd.DataFrame(
            {
                "Aset1_a": [7, 8, 9],
                "Bset1_b": [10, 11, 12],
                "outcome": [np.nan, 80.0, 90.0],
            },
            index=[20, 21, 22],
        )
        split = external_test_split(train, test, ["Aset1_a", "Bset1_b"], "outcome")

        self.assertEqual(list(split.X_train.index), [10, 12])
        self.assertEqual(list(split.X_test.index), [21, 22])
        self.assertEqual(list(split.y_train), [10.0, 30.0])
        self.assertEqual(list(split.y_test), [80.0, 90.0])

    def test_external_test_split_validates_required_columns(self):
        train = pd.DataFrame({"Aset1_a": [1], "outcome": [2]})
        test = pd.DataFrame({"outcome": [3]})
        with self.assertRaisesRegex(KeyError, "Predictor not found in test data: Aset1_a"):
            external_test_split(train, test, ["Aset1_a"], "outcome")

    def test_external_test_split_aligns_test_columns_to_training_order(self):
        train = pd.DataFrame(
            {"Aset1_a": [1, 2], "Bset1_b": [3, 4], "outcome": [5, 6]}
        )
        test = pd.DataFrame(
            {
                "extra": [99, 98],
                "Bset1_b": [7, 8],
                "Aset1_a": [9, 10],
                "outcome": [11, 12],
            }
        )
        split = external_test_split(train, test, ["Aset1_a", "Bset1_b"], "outcome")
        self.assertEqual(list(split.X_test.columns), ["Aset1_a", "Bset1_b"])
        self.assertEqual(split.X_test.to_dict("list"), {"Aset1_a": [9, 10], "Bset1_b": [7, 8]})

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
            self.assertTrue(saved["N"].ge(10).all())
            self.assertTrue((saved["N"] <= saved["n_train_total"]).all())
            self.assertTrue((saved["K"] <= saved["n_features_total"]).all())
            self.assertTrue(
                {"K_varying", "constant_prediction", "underdetermined", "converged"}
                .issubset(saved.columns)
            )
            self.assertNotIn("_fit_seconds", saved.columns)
            self.assertNotIn("_best_rounds", saved.columns)
            parts = sorted((root / "nk_grid.parts").glob("part-*.csv"))
            self.assertEqual(len(parts), 4)
            manifest = json.loads((root / "nk_grid.manifest.json").read_text())
            self.assertEqual(manifest["algorithm_version"], "nk-grid-v2")
            self.assertEqual(manifest["completion"]["expected_rows"], 16)
            self.assertEqual(manifest["completion"]["materialized_rows"], 16)
            self.assertEqual(manifest["completion"]["status"], "complete")
            self.assertEqual(
                manifest["model_parameters"]["resolved"]["ols"],
                {"fit_intercept": True},
            )

    def test_nk_grid_external_test_file_uses_fixed_split_for_all_seeds(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_path = root / "train.csv"
            test_path = root / "test.csv"
            out_path = root / "nk_grid.csv"
            self._write_nk_synthetic_data(train_path)
            test_frame = pd.read_csv(train_path).iloc[:20].copy()
            test_frame["outcome"] = test_frame["outcome"] + 1.0
            test_frame.to_csv(test_path, index=False)

            config = NKGridConfig(
                data=train_path,
                test_data=test_path,
                out=out_path,
                dataset="synthetic",
                outcome="outcome",
                models=("ols",),
                seed=70,
                test_size=0.9,
                n_seeds=2,
                n_draws=1,
                n_sizes_n=1,
                n_sizes_k=1,
                max_n=30,
                max_k=3,
                batch_size=2,
                n_jobs=1,
            )
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                run_nk_grid(config)
            saved = pd.read_csv(out_path)

            self.assertEqual(len(saved), 2)
            self.assertEqual(set(saved["split_mode"]), {"external_test"})
            self.assertEqual(saved["test_data_sha256"].nunique(), 1)
            self.assertTrue(saved["test_data_sha256"].str.fullmatch(r"[0-9a-f]{64}").all())
            self.assertEqual(set(saved["n_train_total"]), {80})
            self.assertEqual(set(saved["n_test_total"]), {20})
            self.assertEqual(set(saved["n_features_total"]), {5})
            self.assertIn("ignoring --test-size", stderr.getvalue())

    def test_nk_grid_external_test_file_validates_predictor_alignment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_path = root / "train.csv"
            test_path = root / "test.csv"
            out_path = root / "nk_grid.csv"
            self._write_nk_synthetic_data(train_path)
            test_frame = pd.read_csv(train_path).drop(columns=["Bset2_e"])
            test_frame.to_csv(test_path, index=False)
            config = NKGridConfig(
                data=train_path,
                test_data=test_path,
                out=out_path,
                dataset="synthetic",
                outcome="outcome",
                models=("ols",),
                seed=71,
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
            with self.assertRaisesRegex(ValueError, "missing predictor columns"):
                run_nk_grid(config)

    def test_nk_grid_external_test_file_handles_classification_and_isolates_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_path = root / "train.csv"
            test_path = root / "test.csv"
            out_path = root / "nk_grid.csv"
            self._write_nk_synthetic_data(train_path)
            test_frame = pd.read_csv(train_path).iloc[:16].copy()
            test_frame["employed"] = [0, 1] * 8
            test_frame.to_csv(test_path, index=False)
            common = dict(
                data=train_path,
                out=out_path,
                dataset="synthetic",
                outcome="employed",
                models=("ols",),
                seed=72,
                test_size=0.3,
                n_seeds=1,
                n_draws=1,
                n_sizes_n=1,
                n_sizes_k=1,
                max_n=30,
                max_k=3,
                batch_size=1,
                n_jobs=1,
                task="classification",
            )

            run_nk_grid(NKGridConfig(**common))
            run_nk_grid(NKGridConfig(**common, test_data=test_path))
            saved = pd.read_csv(out_path)

            self.assertEqual(set(saved["split_mode"]), {"internal_random", "external_test"})
            self.assertEqual(saved["experiment_id"].nunique(), 2)
            external = saved[saved["split_mode"].eq("external_test")].iloc[0]
            internal = saved[saved["split_mode"].eq("internal_random")].iloc[0]
            self.assertEqual(external["n_test_total"], 16)
            self.assertRegex(external["test_data_sha256"], r"^[0-9a-f]{64}$")
            self.assertNotEqual(external["test_data_sha256"], internal["test_data_sha256"])
            self.assertEqual(external["status"], "ok")

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
                min_n=1,
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

    def test_regression_cv_min_n_constants_match_internal_cv_requirements(self):
        self.assertEqual(
            REGRESSION_CV_MIN_N,
            {
                "ridge": 2,
                "lasso": 2,
                "elastic_net": 2,
                "lightgbm": 5,
                "super_learner": 5,
            },
        )

    def test_nk_grid_skips_regression_cv_models_below_minimum_without_fitting(self):
        for model_name, min_n in REGRESSION_CV_MIN_N.items():
            with (
                self.subTest(model=model_name),
                tempfile.TemporaryDirectory() as temp_dir,
            ):
                root = Path(temp_dir)
                data_path = root / "synthetic.csv"
                out_path = root / "nk_grid.csv"
                self._write_nk_synthetic_data(data_path)
                config = NKGridConfig(
                    data=data_path,
                    out=out_path,
                    dataset="synthetic",
                    outcome="outcome",
                    models=(model_name,),
                    seed=30,
                    test_size=0.3,
                    n_seeds=1,
                    n_draws=1,
                    n_sizes_n=1,
                    n_sizes_k=1,
                    min_n=1,
                    max_n=min_n - 1,
                    max_k=2,
                    batch_size=1,
                    n_jobs=1,
                )
                with patch("NK_Grid.src.nk_grid.make_model") as make_model_mock:
                    run_nk_grid(config)
                make_model_mock.assert_not_called()
                saved = pd.read_csv(out_path)
                self.assertEqual(saved.loc[0, "status"], "skipped")
                self.assertEqual(
                    saved.loc[0, "error"],
                    (
                        f"below minimum N for {model_name}'s internal CV "
                        f"(requires N>={min_n})"
                    ),
                )
                self.assertTrue(saved.loc[0, list(METRIC_COLUMNS)].isna().all())

    def test_nk_grid_attempts_regression_cv_models_at_minimum_n(self):
        for model_name, min_n in REGRESSION_CV_MIN_N.items():
            with (
                self.subTest(model=model_name),
                tempfile.TemporaryDirectory() as temp_dir,
            ):
                root = Path(temp_dir)
                data_path = root / "synthetic.csv"
                out_path = root / "nk_grid.csv"
                self._write_nk_synthetic_data(data_path)
                config = NKGridConfig(
                    data=data_path,
                    out=out_path,
                    dataset="synthetic",
                    outcome="outcome",
                    models=(model_name,),
                    seed=31,
                    test_size=0.3,
                    n_seeds=1,
                    n_draws=1,
                    n_sizes_n=1,
                    n_sizes_k=1,
                    min_n=1,
                    max_n=min_n,
                    max_k=2,
                    batch_size=1,
                    n_jobs=1,
                )
                with patch(
                    "NK_Grid.src.nk_grid.make_model",
                    return_value=DummyRegressor(),
                ) as make_model_mock:
                    run_nk_grid(config)
                make_model_mock.assert_called_once()
                saved = pd.read_csv(out_path)
                self.assertEqual(saved.loc[0, "N"], min_n)
                self.assertEqual(saved.loc[0, "status"], "ok")

    def test_nk_grid_does_not_apply_regression_cv_floor_to_classification(self):
        for model_name, min_n in REGRESSION_CV_MIN_N.items():
            with (
                self.subTest(model=model_name),
                tempfile.TemporaryDirectory() as temp_dir,
            ):
                root = Path(temp_dir)
                data_path = root / "synthetic.csv"
                out_path = root / "nk_grid.csv"
                self._write_nk_synthetic_data(data_path)
                config = NKGridConfig(
                    data=data_path,
                    out=out_path,
                    dataset="synthetic",
                    outcome="employed",
                    models=(model_name,),
                    seed=30,
                    test_size=0.3,
                    n_seeds=1,
                    n_draws=1,
                    n_sizes_n=1,
                    n_sizes_k=1,
                    min_n=1,
                    max_n=max(4, min_n - 1),
                    max_k=2,
                    batch_size=1,
                    n_jobs=1,
                    task="classification",
                )
                with patch(
                    "NK_Grid.src.nk_grid.make_model",
                    return_value=DummyClassifier(),
                ) as make_model_mock:
                    run_nk_grid(config)
                make_model_mock.assert_called_once()
                saved = pd.read_csv(out_path)
                self.assertEqual(saved.loc[0, "status"], "ok")
                self.assertEqual(saved.loc[0, "task"], "classification")

    def test_nk_grid_skips_single_class_classification_sample_without_fitting(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_path = root / "synthetic.csv"
            out_path = root / "nk_grid_clf.csv"
            frame = pd.DataFrame(
                {
                    "Aset1_a": np.arange(20),
                    "Bset1_b": np.arange(20) * 2,
                    "employed": [0] * 10 + [1] * 10,
                }
            )
            frame.to_csv(data_path, index=False)
            config = NKGridConfig(
                data=data_path,
                out=out_path,
                dataset="synthetic",
                outcome="employed",
                models=("ols",),
                seed=32,
                test_size=0.3,
                n_seeds=1,
                n_draws=1,
                n_sizes_n=1,
                n_sizes_k=1,
                min_n=1,
                max_n=1,
                max_k=1,
                batch_size=1,
                n_jobs=1,
                task="classification",
            )
            with patch("NK_Grid.src.nk_grid.make_model") as make_model_mock:
                run_nk_grid(config)
            make_model_mock.assert_not_called()
            saved = pd.read_csv(out_path)
            self.assertEqual(saved.loc[0, "status"], "skipped")
            self.assertEqual(
                saved.loc[0, "error"],
                "single-class training sample for classification",
            )
            self.assertTrue(
                saved.loc[0, list(CLASSIFICATION_METRIC_COLUMNS)].isna().all()
            )

    def test_nk_grid_does_not_apply_regression_cv_floor_to_other_models(self):
        for model_name in ("ols", "random_forest", "xgboost", "bart"):
            with (
                self.subTest(model=model_name),
                tempfile.TemporaryDirectory() as temp_dir,
            ):
                root = Path(temp_dir)
                data_path = root / "synthetic.csv"
                out_path = root / "nk_grid.csv"
                self._write_nk_synthetic_data(data_path)
                config = NKGridConfig(
                    data=data_path,
                    out=out_path,
                    dataset="synthetic",
                    outcome="outcome",
                    models=(model_name,),
                    seed=33,
                    test_size=0.3,
                    n_seeds=1,
                    n_draws=1,
                    n_sizes_n=1,
                    n_sizes_k=1,
                    min_n=1,
                    max_n=1,
                    max_k=1,
                    batch_size=1,
                    n_jobs=1,
                    bart_min_n=0,
                    bart_min_k=0,
                )
                with patch(
                    "NK_Grid.src.nk_grid.make_model",
                    return_value=DummyRegressor(),
                ) as make_model_mock:
                    run_nk_grid(config)
                make_model_mock.assert_called_once()
                saved = pd.read_csv(out_path)
                self.assertEqual(saved.loc[0, "status"], "ok")

    def test_nk_grid_end_to_end_marks_tiny_regression_cv_cells_skipped(self):
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
                models=tuple(REGRESSION_CV_MIN_N),
                seed=34,
                test_size=0.3,
                n_seeds=1,
                n_draws=1,
                n_sizes_n=3,
                n_sizes_k=1,
                min_n=1,
                max_n=10,
                max_k=2,
                batch_size=12,
                n_jobs=1,
            )
            with patch(
                "NK_Grid.src.nk_grid.make_model",
                return_value=DummyRegressor(),
            ):
                run_nk_grid(config)
            saved = pd.read_csv(out_path)
            self.assertEqual(set(saved["N"]), {1, 3, 10})
            expected_skipped = {
                ("ridge", 1, 2),
                ("lasso", 1, 2),
                ("elastic_net", 1, 2),
                ("lightgbm", 1, 5),
                ("lightgbm", 3, 5),
                ("super_learner", 1, 5),
                ("super_learner", 3, 5),
            }
            for model_name, n_samples, min_n in expected_skipped:
                row = saved[
                    (saved["model"] == model_name) & (saved["N"] == n_samples)
                ].iloc[0]
                self.assertEqual(row["status"], "skipped")
                self.assertIn(model_name, row["error"])
                self.assertIn(f"N>={min_n}", row["error"])
            for _, row in saved.iterrows():
                row_key = (
                    row["model"],
                    row["N"],
                    REGRESSION_CV_MIN_N[row["model"]],
                )
                if row_key in expected_skipped:
                    continue
                self.assertEqual(row["status"], "ok")

    def test_nk_grid_marks_tiny_super_learner_class_skipped(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_path = root / "train.csv"
            test_path = root / "test.csv"
            out_path = root / "nk_grid.csv"
            pd.DataFrame(
                {
                    "outcome": [0, 0, 0, 0, 1],
                    "X_feature": [0.0, 0.1, 0.2, 0.3, 1.0],
                }
            ).to_csv(data_path, index=False)
            pd.DataFrame(
                {
                    "outcome": [0, 0, 1, 1],
                    "X_feature": [0.05, 0.25, 0.8, 1.1],
                }
            ).to_csv(test_path, index=False)
            config = NKGridConfig(
                data=data_path,
                test_data=test_path,
                out=out_path,
                dataset="tiny_classification",
                outcome="outcome",
                models=("super_learner",),
                seed=41,
                test_size=0.3,
                n_seeds=1,
                n_draws=1,
                n_sizes_n=1,
                n_sizes_k=1,
                min_n=1,
                max_n=5,
                max_k=1,
                batch_size=1,
                n_jobs=1,
                task="classification",
                predictor_prefix=("X_",),
            )

            run_nk_grid(config)

            saved = pd.read_csv(out_path)
            self.assertEqual(saved.loc[0, "status"], "skipped")
            self.assertEqual(
                saved.loc[0, "error"],
                "below minimum per-class count for super_learner CV",
            )
            self.assertTrue(
                saved.loc[0, list(CLASSIFICATION_METRIC_COLUMNS)].isna().all()
            )

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
                models=("ols",),
                seed=30,
                test_size=0.3,
                n_seeds=1,
                n_draws=1,
                n_sizes_n=1,
                n_sizes_k=1,
                min_n=1,
                max_n=1,
                max_k=1,
                batch_size=1,
                n_jobs=1,
            )
            with patch(
                "NK_Grid.src.nk_grid.make_model",
                return_value=FailingRegressor(),
            ):
                run_nk_grid(config)
            saved = pd.read_csv(out_path)
            self.assertEqual(saved.loc[0, "status"], "failed")
            self.assertIn("synthetic fit failure", saved.loc[0, "error"])
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

    def test_nk_grid_preset_creates_timestamped_output_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_path = root / "synthetic.csv"
            out_template = root / "outputs" / "nk_grid.csv"
            self._write_nk_synthetic_data(data_path)
            config = NKGridConfig(
                data=data_path,
                out=out_template,
                dataset="synthetic",
                outcome="outcome",
                models=("ols",),
                seed=60,
                test_size=0.3,
                n_seeds=1,
                n_draws=1,
                n_sizes_n=1,
                n_sizes_k=1,
                max_n=20,
                max_k=3,
                batch_size=1,
                n_jobs=1,
                preset="dev",
            )
            run_nk_grid(config)
            self.assertFalse(out_template.exists())
            output = self._single_output(root / "outputs", "nk_grid_dev_*.csv")
            self.assertRegex(output.name, r"^nk_grid_dev_\d{8}-\d{6}\.csv$")
            saved = pd.read_csv(output)
            self.assertEqual(len(saved), 1)

    def test_nk_grid_preset_resumes_incomplete_timestamped_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_path = root / "synthetic.csv"
            out_template = root / "outputs" / "nk_grid.csv"
            self._write_nk_synthetic_data(data_path)
            config = NKGridConfig(
                data=data_path,
                out=out_template,
                dataset="synthetic",
                outcome="outcome",
                models=("ols",),
                seed=61,
                test_size=0.3,
                n_seeds=1,
                n_draws=2,
                n_sizes_n=1,
                n_sizes_k=2,
                max_n=20,
                max_k=3,
                batch_size=2,
                n_jobs=1,
                preset="dev",
            )
            run_nk_grid(config, max_jobs=2)
            output = self._single_output(root / "outputs", "nk_grid_dev_*.csv")
            partial = pd.read_csv(output)
            self.assertEqual(len(partial), 2)
            run_nk_grid(config)
            outputs = sorted((root / "outputs").glob("nk_grid_dev_*.csv"))
            self.assertEqual(outputs, [output])
            saved = pd.read_csv(output)
            self.assertEqual(len(saved), 4)
            self.assertEqual(
                len(saved[["model", "seed", "draw", "N", "K"]].drop_duplicates()),
                4,
            )

    def test_nk_grid_preset_complete_rerun_creates_new_output_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_path = root / "synthetic.csv"
            out_template = root / "outputs" / "nk_grid.csv"
            self._write_nk_synthetic_data(data_path)
            config = NKGridConfig(
                data=data_path,
                out=out_template,
                dataset="synthetic",
                outcome="outcome",
                models=("ols",),
                seed=62,
                test_size=0.3,
                n_seeds=1,
                n_draws=1,
                n_sizes_n=1,
                n_sizes_k=2,
                max_n=20,
                max_k=3,
                batch_size=2,
                n_jobs=1,
                preset="dev",
            )
            run_nk_grid(config)
            first = self._single_output(root / "outputs", "nk_grid_dev_*.csv")
            run_nk_grid(config)
            outputs = sorted((root / "outputs").glob("nk_grid_dev_*.csv"))
            self.assertEqual(len(outputs), 2)
            self.assertNotEqual(outputs[0], outputs[1])
            for output in outputs:
                saved = pd.read_csv(output)
                self.assertEqual(len(saved), 2)
                self.assertEqual(saved["experiment_id"].nunique(), 1)
            self.assertIn(first, outputs)

    def test_nk_grid_preset_config_change_creates_separate_output_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_path = root / "synthetic.csv"
            out_template = root / "outputs" / "nk_grid.csv"
            self._write_nk_synthetic_data(data_path)
            common = dict(
                data=data_path,
                out=out_template,
                dataset="synthetic",
                outcome="outcome",
                models=("ols",),
                seed=63,
                test_size=0.3,
                n_seeds=1,
                n_draws=1,
                max_n=20,
                max_k=3,
                batch_size=2,
                n_jobs=1,
                preset="dev",
            )
            run_nk_grid(NKGridConfig(**common, n_sizes_n=1, n_sizes_k=1))
            run_nk_grid(NKGridConfig(**common, n_sizes_n=2, n_sizes_k=1))
            outputs = sorted((root / "outputs").glob("nk_grid_dev_*.csv"))
            self.assertEqual(len(outputs), 2)
            row_counts = sorted(len(pd.read_csv(output)) for output in outputs)
            self.assertEqual(row_counts, [1, 2])
            for output in outputs:
                saved = pd.read_csv(output)
                self.assertEqual(saved["experiment_id"].nunique(), 1)
                self.assertEqual(
                    len(saved[["model", "seed", "draw", "N", "K"]]),
                    len(saved[["model", "seed", "draw", "N", "K"]].drop_duplicates()),
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

    def test_dev_preset_keeps_canonical_grid_points(self):
        from NK_Grid.src.run_panels import PRESETS

        self.assertEqual(PRESETS["dev"]["n_seeds"], 5)
        self.assertEqual(PRESETS["dev"]["n_draws"], 5)
        self.assertEqual(PRESETS["dev"]["n_sizes_n"], 8)
        self.assertEqual(PRESETS["dev"]["n_sizes_k"], 8)
        self.assertEqual(PRESETS["dev"]["min_n"], 10)
        self.assertEqual(PRESETS["production"]["min_n"], 10)

    def test_run_panels_resolves_model_params_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _, config = resolve_panel(
                {
                    "name": "params_panel",
                    "data": "synthetic.csv",
                    "dataset": "synthetic",
                    "outcome": "outcome",
                    "models": ["ols"],
                    "model_params": "params.yaml",
                    "out": "outputs/results.csv",
                },
                root,
            )

            self.assertEqual(config.model_params, root / "params.yaml")

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
            self.assertEqual(first_config["n_seeds"], 5)
            self.assertEqual(first_config["n_draws"], 5)
            self.assertEqual(first_config["n_sizes_n"], 2)
            self.assertEqual(first_config["n_sizes_k"], 2)
            self.assertEqual(first_config["preset"], "dev")
            self.assertFalse((root / "outputs" / "reg.csv").exists())
            self.assertFalse((root / "outputs" / "clf.csv").exists())
            self.assertEqual(list((root / "outputs").glob("*.csv")), [])

    def test_run_panels_resolve_panel_sets_explicit_and_default_preset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_path = root / "synthetic.csv"
            self._write_nk_synthetic_data(data_path)
            explicit_name, explicit_config = resolve_panel(
                {
                    "name": "medium_panel",
                    "data": str(data_path),
                    "dataset": "synthetic",
                    "outcome": "outcome",
                    "models": ["ols"],
                    "preset": "medium",
                    "out": str(root / "outputs" / "medium.csv"),
                },
                root,
            )
            default_name, default_config = resolve_panel(
                {
                    "name": "default_panel",
                    "data": str(data_path),
                    "dataset": "synthetic",
                    "outcome": "outcome",
                    "models": ["ols"],
                    "out": str(root / "outputs" / "default.csv"),
                },
                root,
            )
            self.assertEqual(explicit_name, "medium_panel")
            self.assertEqual(explicit_config.preset, "medium")
            self.assertEqual(default_name, "default_panel")
            self.assertEqual(default_config.preset, "dev")

    def test_run_panels_resolve_panel_maps_test_to_test_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_path = root / "synthetic.csv"
            test_path = root / "external" / "test.csv"
            test_path.parent.mkdir()
            self._write_nk_synthetic_data(data_path)
            self._write_nk_synthetic_data(test_path)
            _, config = resolve_panel(
                {
                    "name": "external_panel",
                    "data": "synthetic.csv",
                    "test": "external/test.csv",
                    "dataset": "synthetic",
                    "outcome": "outcome",
                    "models": ["ols"],
                    "out": "outputs/external.csv",
                },
                root,
            )
            self.assertEqual(config.test_data, test_path)

    def test_run_panels_resolve_panel_rejects_test_and_test_data_together(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_path = root / "synthetic.csv"
            self._write_nk_synthetic_data(data_path)
            with self.assertRaisesRegex(ValueError, "test and test_data"):
                resolve_panel(
                    {
                        "name": "external_panel",
                        "data": str(data_path),
                        "test": "test.csv",
                        "test_data": "test2.csv",
                        "dataset": "synthetic",
                        "outcome": "outcome",
                        "models": ["ols"],
                        "out": str(root / "outputs" / "external.csv"),
                    },
                    root,
                )

    def test_run_panels_yaml_manifest_without_grid_overrides_uses_canonical_dev_points(self):
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
            self.assertEqual(config["n_sizes_n"], 8)
            self.assertEqual(config["n_sizes_k"], 8)

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
            reg_outputs = sorted((root / "outputs").glob("reg_dev_*.csv"))
            clf_outputs = sorted((root / "outputs").glob("clf_dev_*.csv"))
            self.assertEqual(len(reg_outputs), 1)
            self.assertEqual(clf_outputs, [])

    def test_run_panels_runs_two_panels_with_independent_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = self._write_panel_manifest(root)
            run_panels_main(["--manifest", str(manifest)])
            reg = pd.read_csv(self._single_output(root / "outputs", "reg_dev_*.csv"))
            clf = pd.read_csv(self._single_output(root / "outputs", "clf_dev_*.csv"))
            self.assertEqual(len(reg), 100)
            self.assertEqual(len(clf), 100)
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
            output = self._single_output(root / "outputs", "reg_dev_*.csv")
            partial = pd.read_csv(output)
            self.assertEqual(len(partial), 3)
            run_panels_main(["--manifest", str(manifest), "--only", "reg_panel"])
            self.assertEqual(sorted((root / "outputs").glob("reg_dev_*.csv")), [output])
            saved = pd.read_csv(output)
            self.assertEqual(len(saved), 100)
            self.assertEqual(
                len(saved[["model", "seed", "draw", "N", "K"]].drop_duplicates()),
                100,
            )

    def _single_output(self, directory: Path, pattern: str) -> Path:
        outputs = sorted(directory.glob(pattern))
        self.assertEqual(len(outputs), 1)
        return outputs[0]

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
