"""One run of the N x K design: a frozen plan, workers that each compute whole cells, and one published table.

A cell is one (seed, draw, N, K). Its worker fits the eight base models, combines
SL7 from their out-of-fold predictions in memory, and appends the cell's rows and
predictions to the worker's own two files. Cells are claimed from a shared counter
in the order of an attempt's ``order.npy``. A later attempt computes only the
cells that no earlier attempt finished. ``finalize`` publishes ``final.csv`` once
every cell is present exactly once and no fit failed.

Usage: python -m aleatoric_nk_grid.cell_pipeline {attempt,work,finalize} --plan PLAN [--attempt DIR]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
import uuid
from pathlib import Path

import numpy as np

from .execution_contract import CellExecutionSpec
from .offline_sl import MissingBasePredictions, recombine_predictions
from .prediction_training import BASE_LIBRARY

FORMAT = "per-cell-v1"
SL7_VARIANT = "standalone8-sl7-v1"
# OLS keeps its own column but is not combined: underdetermined OLS predicts far
# outside the label range (docs/sl7-no-ols-60nodes-20260918).
SL7_MODELS = ("ridge", "lasso", "random_forest", "extra_trees", "xgboost", "lightgbm", "shallow_neural_network")
PREFIX_COLUMNS = ("panel_id", "phase", "pipeline_id", "variant_id", "base_library_id")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def json_result(value):
    """Finite numbers as they are; JSON null for unavailable values."""
    if isinstance(value, dict):
        return {str(k): json_result(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_result(v) for v in value]
    if hasattr(value, "item"):
        return json_result(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def cells(plan):
    """Every (seed, draw, N, K) of the frozen design, in publication order."""
    spec = plan["cell_spec"]
    return [(int(seed), int(draw), int(n), int(k)) for seed, draw in spec["resolved_repeat_plan"]
            for n in spec["resolved_n_grid"] for k in spec["resolved_k_grid"]]


def prepare(config, launch, repo):
    """Freeze the plan and each seed's test IDs; an existing plan must be identical."""
    from . import nk_grid as nk
    root = Path(launch["output"])
    with nk.NKGridExecutionSession.open_from_config(config) as session:
        if "super_learner" in config.models and not set(SL7_MODELS) <= set(session.base_models):
            raise ValueError("SL7 requires all seven of its base models: " + ", ".join(SL7_MODELS))
        spec = CellExecutionSpec.from_config(config, repo_root=repo, panel_id=launch["panel"],
            resolved_n_grid=session.n_grid, resolved_k_grid=session.k_grid,
            resolved_repeat_plan=session.repeat_pairs, model_n_jobs=1,
            git_commit=launch["source"]["commit"], algorithm_version=session.algorithm_version,
            resolved_model_params=nk.resolved_model_params(session.selected_model_params),
            environment_overrides=nk.model_run_settings(config.models),
            execution_groups=[{"k_features": int(k), "groups": [{"group": g, "models": list(ms)}
                for g, ms in nk.execution_groups_for_models(config.models)]} for k in session.k_grid],
            input_provenance=nk._frozen_input_provenance_for_schema(session.schema), require_clean_worktree=True)
        seeds = sorted({seed for seed, _ in session.repeat_pairs})
        test_ids = {str(seed): session.test_ids(seed) for seed in seeds}
        task = session.task
    plan = {"format": FORMAT, "cell_spec": spec.to_payload(), "launch": launch, "task_kind": task,
            "public_columns": [*PREFIX_COLUMNS, *nk.public_result_columns(task)]}
    plan["public_columns"] = list(dict.fromkeys(plan["public_columns"]))
    path = root / "plan.json"
    if path.exists():
        if read(path) != json.loads(json.dumps(plan)):
            raise ValueError("Prepared plan changed")
    else:
        np.savez(root / "test-ids.npz", **test_ids)
        atomic_json(path, plan)
    return path


# One cell -------------------------------------------------------------------

def _sl7_row(task, rows, arrays, y_train, y_test, y_train_full):
    """SL7 from the seven models' OOF predictions, laid out like their rows."""
    from .nk_grid import (CLASSIFICATION_METRIC_COLUMNS, METRIC_COLUMNS,
                          compute_classification_metrics, compute_regression_metrics)
    by_model = {row["model"]: row for row in rows}
    row = dict(by_model[SL7_MODELS[0]])
    row.update(model="super_learner", phase="sl", pipeline_id="", variant_id=SL7_VARIANT)
    combined = {}
    try:
        folds = None
        for name in SL7_MODELS:
            source = by_model[name]
            if source["status"] != "ok":
                raise MissingBasePredictions(f"required model {name}: {source['status']}: {source['error']}")
            if "oof_prediction" not in arrays[name]:
                raise MissingBasePredictions(f"model {name} lacks required OOF/full arrays")
            if folds is None:
                folds = arrays[name]["oof_fold"]
            elif not np.array_equal(folds, arrays[name]["oof_fold"]):
                raise MissingBasePredictions("base model fold assignments differ")
        answer = recombine_predictions(
            oof_predictions=np.column_stack([arrays[name]["oof_prediction"] for name in SL7_MODELS]),
            holdout_predictions=np.column_stack([arrays[name]["holdout_prediction"] for name in SL7_MODELS]),
            y_train=y_train, model_names=SL7_MODELS, selected_models=SL7_MODELS,
            variant_id=SL7_VARIANT, task=task)
    except MissingBasePredictions as exc:
        row.update(status="skipped", error=str(exc))
        for name in (*METRIC_COLUMNS, *CLASSIFICATION_METRIC_COLUMNS):
            if name in row:
                row[name] = None
        combiner_converged = None
    else:
        if task == "classification":
            row.update(compute_classification_metrics(y_test, answer["holdout_prediction"], y_train, y_train_full))
        else:
            row.update(compute_regression_metrics(y_test, answer["holdout_prediction"], y_train, y_train_full))
        row.update(status="ok", error="")
        combined = {"holdout_prediction": answer["holdout_prediction"], "coefficients": answer["coefficients"],
                    "intercept": np.asarray([answer["intercept"]], dtype=np.float64)}
        combiner_converged = answer["metadata"]["converged"]
    # Any nonconverged base fit, or the combiner, makes SL7 nonconverged; all
    # converged is True; otherwise unknown.
    base = [by_model[name]["converged"] for name in SL7_MODELS if by_model[name]["status"] == "ok"]
    if not all(base) or combiner_converged is False:
        row["converged"] = False
    elif combiner_converged is True:
        row["converged"] = True
    else:
        row["converged"] = None
    return row, combined


def compute_cell(session, panel_id, seed, draw, n, k):
    """All rows and predictions of one cell, ready to be written."""
    cell = session.run_cell(seed=seed, draw=draw, n_samples=n, k_features=k)
    rows, model_arrays = cell["rows"], cell["arrays"]
    for row in rows:
        row.update(panel_id=panel_id, phase="base", pipeline_id=f"{BASE_LIBRARY}/{row['model']}",
                   variant_id="", base_library_id=BASE_LIBRARY)
    arrays = {"train_ids": cell["train_ids"]}
    for name, values in model_arrays.items():
        if "oof_fold" in values and "oof_fold" not in arrays:
            arrays["oof_fold"] = values["oof_fold"]
        for key, label in (("holdout_prediction", "holdout"), ("oof_prediction", "oof")):
            if key in values:
                arrays[f"{name}/{label}"] = values[key]
    if "super_learner" in session.config.models:
        row, combined = _sl7_row(session.task, rows, model_arrays, cell["y_train"], cell["y_test"],
                                 cell["y_train_full"])
        rows.append(row)
        for key, values in combined.items():
            arrays["super_learner/" + key.replace("_prediction", "")] = values
    return rows, arrays


# Files of one worker ----------------------------------------------------------

class CellWriter:
    """Append-only: a cell's arrays are durable before the line that refers to them."""

    def __init__(self, directory, rank):
        directory = Path(directory)
        self.name = f"predictions-r{rank:05d}.bin"
        self.arrays = (directory / self.name).open("ab")
        self.results = (directory / f"results-r{rank:05d}.jsonl").open("a", encoding="utf-8")

    def write(self, cell, rows, arrays, seconds):
        references = {}
        for name, values in arrays.items():
            values = np.ascontiguousarray(values)
            references[name] = [self.arrays.tell(), values.dtype.str, list(values.shape)]
            self.arrays.write(values.tobytes())
        self.arrays.flush()
        os.fsync(self.arrays.fileno())
        line = json.dumps(json_result({"cell": list(cell), "rows": rows, "file": self.name,
                                       "arrays": references, "seconds": seconds}), allow_nan=False)
        self.results.write(line + "\n")
        self.results.flush()
        os.fsync(self.results.fileno())

    def close(self):
        self.arrays.close()
        self.results.close()


def results(path):
    """(byte offset, record) of each complete line; a torn last line is skipped."""
    with Path(path).open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                return
            if not line.endswith(b"\n"):
                if handle.read(1):
                    raise ValueError(f"{path}: incomplete line before the end of the file")
                return
            yield offset, json.loads(line)


def finished(root):
    """Cell -> (results file, line offset) for every cell written by any attempt."""
    found = {}
    for path in sorted(Path(root).glob("attempt-*/results-r*.jsonl")):
        for offset, record in results(path):
            cell = tuple(record["cell"])
            if cell in found:
                raise ValueError(f"cell {cell} was written twice: {found[cell][0]} and {path}")
            found[cell] = (path, offset)
    return found


def read_arrays(directory, record):
    """The arrays of one results line, read back from its worker's file."""
    out = {}
    with (Path(directory) / record["file"]).open("rb") as handle:
        for name, (offset, dtype, shape) in record["arrays"].items():
            dtype = np.dtype(dtype)
            handle.seek(offset)
            out[name] = np.frombuffer(handle.read(dtype.itemsize * math.prod(shape)), dtype=dtype).reshape(shape)
    return out


# Attempts, workers and publication ---------------------------------------------

def start_attempt(plan_path, directory):
    """Order the cells no attempt has finished, largest N x K first."""
    plan = read(plan_path)
    done = finished(Path(plan_path).parent)
    design = cells(plan)
    order = sorted((index for index, cell in enumerate(design) if cell not in done),
                   key=lambda index: (-design[index][2] * design[index][3], index))
    directory = Path(directory)
    directory.mkdir()
    np.save(directory / "order.npy", np.asarray(order, dtype=np.int64))
    (directory / "claims").write_bytes(bytes(8))
    print(json.dumps({"attempt": str(directory), "cells": len(design), "finished": len(done),
                      "to_compute": len(order)}), flush=True)


def claim(path):
    """Next position of the shared counter, under a POSIX record lock (cross-node on Lustre and GPFS)."""
    import fcntl
    with Path(path).open("r+b") as handle:
        fcntl.lockf(handle.fileno(), fcntl.LOCK_EX)
        try:
            value = int.from_bytes(handle.read(8), "little")
            handle.seek(0)
            handle.write((value + 1).to_bytes(8, "little"))
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.lockf(handle.fileno(), fcntl.LOCK_UN)
    return value


def work(plan_path, directory, rank):
    from .nk_grid import NKGridExecutionSession
    plan = read(plan_path)
    directory = Path(directory)
    order = np.load(directory / "order.npy")
    design = cells(plan)
    spec = CellExecutionSpec.from_payload(plan["cell_spec"])
    writer = CellWriter(directory, rank)
    try:
        with NKGridExecutionSession.open(spec) as session:
            while (position := claim(directory / "claims")) < len(order):
                cell = design[int(order[position])]
                started = time.perf_counter()
                rows, arrays = compute_cell(session, plan["launch"]["panel"], *cell)
                writer.write(cell, rows, arrays, time.perf_counter() - started)
    finally:
        writer.close()


def finalize(plan_path):
    """Publish final.csv, predictions.jsonl and verified.json once every cell is present and no fit failed."""
    plan = read(plan_path)
    root = Path(plan_path).parent
    if (root / "verified.json").exists():
        return read(root / "verified.json")
    design = cells(plan)
    found = finished(root)
    missing = [cell for cell in design if cell not in found]
    if missing or len(found) != len(design):
        raise SystemExit(f"{len(missing)} of {len(design)} cells are not finished; resume the run")
    columns = plan["public_columns"]
    counts, failed = {}, []
    final_tmp, index_tmp = root / f".final.csv.tmp-{uuid.uuid4().hex}", root / f".predictions.jsonl.tmp-{uuid.uuid4().hex}"
    handles = {}
    try:
        with final_tmp.open("x", newline="", encoding="utf-8") as out, index_tmp.open("x", encoding="utf-8") as index:
            writer = csv.DictWriter(out, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            for cell in design:
                path, offset = found[cell]
                handle = handles.get(path) or handles.setdefault(path, path.open("rb"))
                handle.seek(offset)
                record = json.loads(handle.readline())
                for row in record["rows"]:
                    if row["status"] == "failed":
                        failed.append((cell, row["model"], row["error"]))
                    counts[row["status"]] = counts.get(row["status"], 0) + 1
                    writer.writerow({column: row.get(column, "") for column in columns})
                index.write(json.dumps({"cell": record["cell"],
                                        "file": str(path.parent.relative_to(root) / record["file"]),
                                        "arrays": record["arrays"]}) + "\n")
            out.flush()
            os.fsync(out.fileno())
            index.flush()
            os.fsync(index.fileno())
        if failed:
            raise SystemExit(f"{len(failed)} fits failed; first: {failed[0]}. Nothing was published.")
        os.replace(index_tmp, root / "predictions.jsonl")
        os.replace(final_tmp, root / "final.csv")
    finally:
        for handle in handles.values():
            handle.close()
        final_tmp.unlink(missing_ok=True)
        index_tmp.unlink(missing_ok=True)
    receipt = {"format": FORMAT, "complete": True, "cells": len(design), "rows": sum(counts.values()),
               "statuses": counts, "commit": plan["cell_spec"]["git_commit"],
               "attempts": sorted(path.name for path in root.glob("attempt-*"))}
    atomic_json(root / "verified.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("attempt", "work", "finalize"))
    parser.add_argument("--plan", required=True)
    parser.add_argument("--attempt")
    args = parser.parse_args(argv)
    if args.action == "attempt":
        start_attempt(args.plan, args.attempt)
    elif args.action == "work":
        work(args.plan, args.attempt, int(os.environ["SLURM_PROCID"]))
    else:
        print(json.dumps(finalize(args.plan)), flush=True)


if __name__ == "__main__":
    main()
