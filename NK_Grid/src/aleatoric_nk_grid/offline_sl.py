"""Cache-only SL recombination. This module has no base-model training path."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression


class IncompletePredictionCache(ValueError):
    pass


def recombine_predictions(*, oof_predictions, holdout_predictions, y_train,
                          model_names, selected_models, task="regression",
                          combiner_config=None, variant_id, statuses=None):
    """Learn from OOF and training labels only; holdout labels are not accepted.

    Nonnegative regression coefficients include a fitted intercept and are not
    normalized. Classification uses the declared logistic regression rule on
    positive-class probabilities. Required missing columns are never dropped.
    """
    names, selected = tuple(model_names), tuple(selected_models)
    if not variant_id or not selected or len(set(selected)) != len(selected) or len(set(names)) != len(names):
        raise ValueError("variant ID and unique nonempty model lists are required")
    missing = set(selected) - set(names)
    if missing:
        raise IncompletePredictionCache(f"required model columns missing: {sorted(missing)}")
    if statuses is not None:
        unavailable = {name: statuses.get(name, "missing") for name in selected if statuses.get(name) != "ok"}
        if unavailable:
            raise IncompletePredictionCache(f"required model columns unavailable: {unavailable}")
    Z, P, y = np.asarray(oof_predictions), np.asarray(holdout_predictions), np.asarray(y_train, dtype=np.float64)
    if Z.dtype != np.float64 or P.dtype != np.float64:
        raise ValueError("prediction arrays must be float64")
    if Z.ndim != 2 or P.ndim != 2 or Z.shape[1] != len(names) or P.shape[1] != len(names) or y.shape != (Z.shape[0],):
        raise ValueError("OOF/holdout/model/sample dimensions do not align")
    columns = [names.index(name) for name in selected]
    Z, P = Z[:, columns], P[:, columns]
    if not np.isfinite(Z).all() or not np.isfinite(P).all() or not np.isfinite(y).all():
        raise IncompletePredictionCache("nonfinite required predictions/labels")
    config = dict(combiner_config or {})
    started = time.perf_counter()
    convergence = []
    if task == "regression":
        if config not in ({}, {"rule": "nnls-intercept-v1"}):
            raise ValueError("regression combiner must explicitly use nnls-intercept-v1")
        center = Z.mean(axis=0)
        weights, _ = nnls(Z - center, y - y.mean())
        intercept = float(y.mean() - center @ weights)
        prediction = P @ weights + intercept
        objective = float(np.mean((Z @ weights + intercept - y) ** 2))
        config = {"rule": "nnls-intercept-v1"}
        classes = []
    elif task == "classification":
        unknown = set(config) - {"rule", "C", "max_iter", "random_state"}
        if unknown or config.get("rule", "logistic-v1") != "logistic-v1":
            raise ValueError("classification combiner must use frozen logistic-v1 rule")
        if not np.array_equal(np.unique(y), [0, 1]):
            raise IncompletePredictionCache("classification combiner requires both classes [0, 1]")
        if np.any((Z < 0) | (Z > 1)) or np.any((P < 0) | (P > 1)):
            raise ValueError("classification cache must contain probabilities")
        config = {"rule": "logistic-v1", "C": float(config.get("C", 1.)),
                  "max_iter": int(config.get("max_iter", 100)),
                  "random_state": int(config.get("random_state", 0))}
        estimator = LogisticRegression(**{key: value for key, value in config.items() if key != "rule"})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            estimator.fit(Z, y)
        convergence = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
        weights = estimator.coef_[0]
        intercept = float(estimator.intercept_[0])
        prediction = estimator.predict_proba(P)[:, 1]
        objective = None
        classes = estimator.classes_.tolist()
    else:
        raise ValueError("task must be regression or classification")
    return {"holdout_prediction": np.asarray(prediction, dtype=np.float64),
            "coefficients": np.asarray(weights, dtype=np.float64), "intercept": intercept,
            "metadata": {"variant_id": variant_id, "selected_models": list(selected),
                         "task": task, "combiner": config, "classes": classes,
                         "base_fit_count": 0, "combiner_fit_count": 1,
                         "combiner_seconds": time.perf_counter() - started,
                         "oof_objective_mse": objective,
                         "converged": not convergence, "convergence_warnings": convergence}}


def recombine_cache_records(records, *, y_train, selected_models, variant_id,
                            task="regression", combiner_config=None):
    """Assemble verified same-cell records; callers read/check storage first."""
    records = list(records)
    by_name = {}
    common = None
    fold_ids = None
    for record in records:
        metadata = record.metadata if hasattr(record, "metadata") else record["metadata"]
        arrays = record.arrays if hasattr(record, "arrays") else record["arrays"]
        identity = record.identity if hasattr(record, "identity") else record["identity"]
        name = metadata["model_name"]
        if name in by_name:
            raise IncompletePredictionCache(f"duplicate model column {name}")
        # Pipeline/model fields legitimately differ; all cell/data order fields
        # must come from the exact same immutable common identity.
        cell_identity = identity.get("cell_identity")
        if cell_identity is None:
            raise IncompletePredictionCache("cache record lacks common cell_identity")
        encoded = json.dumps(cell_identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if common is None:
            common = encoded
        elif common != encoded:
            raise IncompletePredictionCache("base records belong to different cell/sample identities")
        if metadata.get("status") != "ok":
            if name in selected_models:
                raise IncompletePredictionCache(f"required model {name}: {metadata.get('status')}: {metadata.get('reason')}")
            continue
        if "oof_prediction" not in arrays or "holdout_prediction" not in arrays or "oof_fold" not in arrays:
            raise IncompletePredictionCache(f"model {name} lacks required OOF/full arrays")
        if fold_ids is None:
            fold_ids = arrays["oof_fold"]
        elif not np.array_equal(fold_ids, arrays["oof_fold"]):
            raise IncompletePredictionCache("base model fold assignments differ")
        by_name[name] = arrays
    missing = set(selected_models) - set(by_name)
    if missing:
        raise IncompletePredictionCache(f"required model columns missing: {sorted(missing)}")
    names = list(selected_models)
    return recombine_predictions(oof_predictions=np.column_stack([by_name[n]["oof_prediction"] for n in names]),
        holdout_predictions=np.column_stack([by_name[n]["holdout_prediction"] for n in names]),
        y_train=y_train, model_names=names, selected_models=names, variant_id=variant_id,
        task=task, combiner_config=combiner_config)


def audit_legacy_npz(path):
    """Recompute saved SL7/SL8 under their original scipy centered-NNLS rule."""
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path, allow_pickle=False) as cache:
        names = cache["models"].tolist()
        results = {}
        for count, selected in ((8, names), (7, [name for name in names if name != "ols"])):
            if len(selected) != count:
                raise ValueError("legacy SL8/SL7 cache has unexpected model names")
            answer = recombine_predictions(oof_predictions=cache["oof"], holdout_predictions=cache["full_predictions"],
                y_train=cache["y_train"], model_names=names, selected_models=selected,
                variant_id=f"legacy-sl{count}-recomputed", task="regression")
            expected = cache[f"sl{count}_prediction"]
            np.testing.assert_allclose(answer["holdout_prediction"], expected, rtol=1e-10, atol=1e-12)
            results[f"sl{count}"] = {**answer["metadata"],
                "prediction_max_abs_difference": float(np.max(np.abs(answer["holdout_prediction"] - expected))),
                "coefficient_max_abs_difference": float(np.max(np.abs(answer["coefficients"] - cache[f"sl{count}_weights"]))),
                "mse": float(np.mean((answer["holdout_prediction"] - cache["y_valid"]) ** 2))}
    return {"source": str(path), "source_sha256": digest, "passed": True, "variants": results}


def recombine_verified_cell(*, plan_path, panel_id, seed, draw, n, k, pipeline_ids,
                            variant_id, output, combiner_config=None, store_prediction=False):
    """Create one independently identified sensitivity variant from a sealed run."""
    from dataclasses import asdict
    from .prediction_cache import read_record
    from .prediction_workflow import PredictionTask, base_record_for_task, verify_base_receipt
    from .shared_queue import digest
    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    receipt = verify_base_receipt(plan)
    contract = plan["prediction_workflow"]
    panel = next((p for p in contract["panels"] if p["panel_id"] == panel_id), None)
    if panel is None:
        raise ValueError("panel is outside the verified source plan")
    if not pipeline_ids or len(set(pipeline_ids)) != len(pipeline_ids):
        raise ValueError("selected pipeline IDs must be unique and nonempty")
    records = []
    for pipeline_id in pipeline_ids:
        recipe = next((p for p in panel["pipelines"] if p["pipeline_id"] == pipeline_id), None)
        if recipe is None:
            raise ValueError(f"pipeline {pipeline_id} is outside the frozen source library")
        task = PredictionTask(seed=int(seed), draw=int(draw), N=int(n), K=int(k), model=recipe["model"],
            phase="base", panel_id=panel_id, pipeline_id=pipeline_id, variant_id="",
            base_library_id=contract["base_library_id"])
        reference = base_record_for_task(contract, task)["reference"]
        record = read_record(contract["cache_root"], reference, require_sealed=True)
        if (record.metadata.get("task") != asdict(task)
                or record.metadata.get("pipeline_sha256") != digest(recipe)
                or record.metadata.get("input_spec_sha256") != digest(panel["cell_spec"])):
            raise IncompletePredictionCache("source cache identity differs from frozen plan")
        records.append(record)
    training = read_record(contract["cache_root"], records[0].metadata["sample_map_refs"]["training"], require_sealed=True)
    if panel["task_kind"] == "classification" and combiner_config is None:
        raise ValueError("classification recombination requires explicit frozen combiner config")
    names = [r.metadata["model_name"] for r in records]
    answer = recombine_cache_records(records, y_train=training.arrays["y_train"], selected_models=names,
        variant_id=variant_id, task=panel["task_kind"], combiner_config=combiner_config)
    # Evaluation is intentionally loaded only after all fitted parameters exist.
    evaluation = read_record(contract["cache_root"], records[0].metadata["sample_map_refs"]["evaluation"], require_sealed=True)
    from .nk_grid import compute_regression_metrics, compute_classification_metrics
    metric = compute_classification_metrics if panel["task_kind"] == "classification" else compute_regression_metrics
    from .execution_contract import runtime_environment
    manifest = {**answer["metadata"], "source_plan": str(plan_path.resolve()),
        "combiner_implementation": {"module": __name__, "dtype": "float64",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "runtime_environment": runtime_environment()},
        "source_plan_sha256": digest(plan), "base_verification_sha256": digest(receipt),
        "panel_id": panel_id, "seed": seed, "draw": draw, "N": n, "K": k,
        "pipeline_ids": list(pipeline_ids), "source_records": [dict(r.reference) for r in records],
        "coefficients": answer["coefficients"].tolist(), "intercept": answer["intercept"],
        "scores": metric(evaluation.arrays["y_holdout"], answer["holdout_prediction"], training.arrays["y_train"]),
        "prediction_stored": bool(store_prediction)}
    from .single_model_worker import json_result
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "variant.json").write_text(json.dumps(json_result(manifest), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if store_prediction:
        np.savez_compressed(output / "holdout.npz", holdout_prediction=answer["holdout_prediction"])
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--legacy-npz", nargs="+", type=Path,
                        help="saved SL7/SL8 NPZ paths or directories containing them")
    source.add_argument("--plan", type=Path, help="trusted frozen plan with sealed base-verified.json")
    parser.add_argument("--panel")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--draw", type=int, default=0)
    parser.add_argument("--n", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--pipelines", nargs="+")
    parser.add_argument("--variant-id")
    parser.add_argument("--combiner-config", type=Path, help="JSON configuration of the new combiner")
    parser.add_argument("--store-prediction", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.plan is not None:
        if any(value is None for value in (args.panel, args.seed, args.n, args.k, args.pipelines, args.variant_id)):
            parser.error("--plan needs --panel, --seed, --n, --k, --pipelines and --variant-id")
        combo = json.loads(args.combiner_config.read_text(encoding="utf-8")) if args.combiner_config else None
        manifest = recombine_verified_cell(plan_path=args.plan, panel_id=args.panel, seed=args.seed,
            draw=args.draw, n=args.n, k=args.k, pipeline_ids=args.pipelines, variant_id=args.variant_id,
            output=args.output, combiner_config=combo, store_prediction=args.store_prediction)
        print(json.dumps({"variant_id": manifest["variant_id"], "base_fit_count": 0, "output": str(args.output)}))
        return
    paths = sorted({p for item in args.legacy_npz for p in
                    (item.rglob("predictions.npz") if item.is_dir() else [item])})
    if not paths:
        raise ValueError("no prediction caches found")
    results = [audit_legacy_npz(path) for path in paths]
    payload = {"passed": True, "cases": len(results), "base_fit_count": 0, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("passed", "cases", "base_fit_count")}))


if __name__ == "__main__":
    main()
