"""Select one fixed regression configuration using training-only anchor-cell CV."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment import core_environment, file_sha256, git_state, utc_now, write_json_atomic
from model_registry import (
    DEFAULT_MODEL_PARAMS_PATH,
    load_algorithm_version,
    load_model_params,
    make_model,
)
from nk_grid import _model_seed, _predictor_columns, draw_orders, split_frame


TUNABLE_MODELS = (
    "random_forest",
    "extra_trees",
    "lightgbm",
    "shallow_neural_network",
)
TUNING_FIT_THRESHOLD = 1_000


def anchor_candidates(model_name: str) -> list[dict[str, Any]]:
    if model_name in {"random_forest", "extra_trees"}:
        return [
            {"min_samples_leaf": leaf, "max_features": max_features}
            for leaf, max_features in itertools.product((1, 2, 5), ("sqrt", 0.1))
        ]
    if model_name == "lightgbm":
        return [{"min_data_in_leaf": leaf} for leaf in (5, 10, 20)]
    if model_name == "shallow_neural_network":
        return [
            {
                "hidden_layer_sizes": [width],
                "alpha": alpha,
                "early_stopping": early_stopping,
                "validation_fraction": 0.2,
                "n_iter_no_change": 20,
            }
            for width, alpha, early_stopping in itertools.product(
                (16, 32), (0.001, 0.01), (False, True)
            )
        ]
    raise ValueError(f"No anchor candidate grid declared for {model_name}")


def _realized_anchors(values: Iterable[int], full: int) -> list[int]:
    return sorted({full if int(value) <= 0 else min(int(value), full) for value in values})


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def tune(args: argparse.Namespace) -> dict[str, Any]:
    data_path = Path(args.data)
    frame = pd.read_csv(data_path)
    predictors = _predictor_columns(frame, args.predictor_prefix)
    if not predictors:
        raise ValueError("No predictor columns matched --predictor-prefix")
    for outcome in args.outcomes:
        if outcome not in frame:
            raise KeyError(f"Outcome not found: {outcome}")

    base_params = load_model_params(
        Path(args.model_params), task="regression", models=args.models
    )
    evaluations: list[dict[str, Any]] = []
    selected: dict[str, Any] = {}

    for model_name in args.models:
        candidate_results: list[dict[str, Any]] = []
        for candidate_index, override in enumerate(anchor_candidates(model_name), start=1):
            params = {**base_params[model_name], **override}
            cell_scores: list[dict[str, Any]] = []
            failed = False
            for outcome in args.outcomes:
                for tuning_seed_offset in range(args.tuning_seeds):
                    split_seed = args.seed + tuning_seed_offset
                    split = split_frame(
                        frame,
                        predictors,
                        outcome,
                        test_size=args.test_size,
                        seed=split_seed,
                        task="regression",
                    )
                    n_anchors = _realized_anchors(args.anchor_n, len(split.X_train))
                    k_anchors = _realized_anchors(args.anchor_k, len(predictors))
                    for draw in range(args.tuning_draws):
                        orders = draw_orders(
                            split.X_train.index,
                            predictors,
                            seed=split_seed,
                            draw=draw,
                        )
                        for n_samples, k_features in itertools.product(
                            n_anchors, k_anchors
                        ):
                            rows = orders.row_index[:n_samples]
                            columns = orders.feature_names[:k_features]
                            X_sub = split.X_train.loc[rows, columns]
                            y_sub = split.y_train.loc[rows]
                            scale = float(np.std(y_sub, ddof=1))
                            if not np.isfinite(scale) or scale <= 0:
                                failed = True
                                cell_scores.append(
                                    {
                                        "outcome": outcome,
                                        "seed": split_seed,
                                        "draw": draw,
                                        "N": n_samples,
                                        "K": k_features,
                                        "normalized_rmse": None,
                                        "error": "zero or non-finite outcome scale",
                                    }
                                )
                                continue
                            estimator = make_model(
                                model_name,
                                seed=_model_seed(split_seed, draw, n_samples, k_features),
                                n_jobs=1,
                                task="regression",
                                params=params,
                            )
                            cv = KFold(
                                n_splits=min(args.cv, len(y_sub)),
                                shuffle=True,
                                random_state=split_seed,
                            )
                            try:
                                rmse = -cross_val_score(
                                    estimator,
                                    X_sub,
                                    y_sub,
                                    scoring="neg_root_mean_squared_error",
                                    cv=cv,
                                    n_jobs=args.n_jobs,
                                    error_score="raise",
                                )
                                normalized = float(np.mean(rmse) / scale)
                                error = ""
                            except Exception as exc:
                                failed = True
                                normalized = np.nan
                                error = f"{type(exc).__name__}: {exc}"
                            cell_scores.append(
                                {
                                    "outcome": outcome,
                                    "seed": split_seed,
                                    "draw": draw,
                                    "N": n_samples,
                                    "K": k_features,
                                    "normalized_rmse": (
                                        normalized if np.isfinite(normalized) else None
                                    ),
                                    "error": error,
                                }
                            )
            finite_scores = [
                row["normalized_rmse"]
                for row in cell_scores
                if row["normalized_rmse"] is not None
            ]
            result = {
                "model": model_name,
                "candidate_index": candidate_index,
                "override": override,
                "resolved_parameters": params,
                "mean_normalized_rmse": (
                    float(np.mean(finite_scores)) if finite_scores and not failed else None
                ),
                "all_cells_valid": not failed,
                "cells": cell_scores,
            }
            candidate_results.append(result)
            evaluations.append(result)
        valid = [
            result
            for result in candidate_results
            if result["mean_normalized_rmse"] is not None
        ]
        if not valid:
            raise RuntimeError(f"Every anchor candidate failed for {model_name}")
        winner = min(valid, key=lambda result: result["mean_normalized_rmse"])
        selected[model_name] = winner["resolved_parameters"]

    payload = {
        "schema_version": "1",
        "created_at": utc_now(),
        "algorithm_version": load_algorithm_version(Path(args.model_params)),
        "git": git_state(ROOT),
        "environment": core_environment(),
        "method": "training-pool anchor-cell cross-validation",
        "holdout_used_for_selection": False,
        "data": {
            "path": os.path.relpath(data_path.resolve(), ROOT),
            "sha256": file_sha256(data_path),
            "outcomes": args.outcomes,
            "predictor_prefix": args.predictor_prefix,
        },
        "design": {
            "seed": args.seed,
            "tuning_seeds": args.tuning_seeds,
            "tuning_draws": args.tuning_draws,
            "test_size": args.test_size,
            "cv": args.cv,
            "anchor_n": args.anchor_n,
            "anchor_k": args.anchor_k,
            "selection_metric": "mean normalized RMSE across all declared cells",
            "n10_nonconstant_is_selection_criterion": False,
        },
        "selected_model_parameters": selected,
        "evaluations": evaluations,
    }
    return json.loads(json.dumps(payload, default=_jsonable))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Tune one fixed regression configuration on training-only anchor cells."
    )
    parser.add_argument("--data", default=str(ROOT / "data" / "asample2_withlag.csv"))
    parser.add_argument("--outcomes", nargs="+", required=True)
    parser.add_argument("--models", nargs="+", choices=TUNABLE_MODELS, default=list(TUNABLE_MODELS))
    parser.add_argument("--model-params", default=str(DEFAULT_MODEL_PARAMS_PATH))
    parser.add_argument("--out", default=str(ROOT / "outputs" / "anchor_tuning.json"))
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--tuning-seeds", type=int, default=1)
    parser.add_argument("--tuning-draws", type=int, default=1)
    parser.add_argument("--anchor-n", nargs="+", type=int, default=[50, 100, 500, 0])
    parser.add_argument("--anchor-k", nargs="+", type=int, default=[50, 100, 1000, 0])
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--predictor-prefix", nargs="+", default=["Aset", "Bset"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-large-run", action="store_true")
    args = parser.parse_args(argv)

    candidate_count = sum(len(anchor_candidates(model)) for model in args.models)
    declared_cells = (
        len(args.outcomes)
        * args.tuning_seeds
        * args.tuning_draws
        * len(args.anchor_n)
        * len(args.anchor_k)
    )
    estimate = {
        "models": args.models,
        "candidate_configurations": candidate_count,
        "declared_anchor_cells_per_candidate": declared_cells,
        "upper_bound_cv_fits": candidate_count * declared_cells * args.cv,
    }
    if args.dry_run:
        print(json.dumps(estimate, indent=2, sort_keys=True))
        return
    if estimate["upper_bound_cv_fits"] > TUNING_FIT_THRESHOLD and not args.allow_large_run:
        raise ValueError(
            "Large anchor tuning run requires --allow-large-run: "
            f"{estimate['upper_bound_cv_fits']:,} estimated CV fits exceeds the "
            f"{TUNING_FIT_THRESHOLD:,} safety threshold."
        )

    payload = tune(args)
    payload["estimate"] = estimate
    write_json_atomic(Path(args.out), payload)
    print(json.dumps({"output": str(args.out), "selected": payload["selected_model_parameters"]}, indent=2))


if __name__ == "__main__":
    main()
