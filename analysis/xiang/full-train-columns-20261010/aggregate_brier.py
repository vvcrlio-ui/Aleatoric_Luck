"""Per-cell mean, P5, min and max of the FFC R-squared of a classification run, from its existing brier column.

brier_full_train_r2 = 1 - brier / null_brier, where null_brier is the squared error of predicting the
seed's training pool positive rate on its test labels, computed as classification_denominators does.
Writes the same aggregate as aggregate_panels.py, so the plots read either.
Usage: python aggregate_brier.py OUT_DIR TARGETS.npz RUN_DIR
"""
from array import array
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

out, targets_path, run = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
plan = json.loads((run / "plan.json").read_text())
assert plan["task_kind"] == "classification"
targets = np.load(targets_path)
null_brier = {}
for key in targets.files:
    seed = int(key.split("/")[0])
    if seed not in null_brier:
        y_true = np.asarray(targets[f"{seed}/test_y"], dtype=int)
        full = np.asarray(targets[f"{seed}/pool_y"], dtype=int)
        null_brier[seed] = float(np.mean((y_true - np.mean(full)) ** 2))


def quantile(ordered, q):
    """Linear interpolation between closest ranks, as numpy.quantile's default."""
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


values, statuses, meta = {}, {}, {}
csv.field_size_limit(1 << 24)
with (run / "final.csv").open(newline="") as handle:
    for row in csv.DictReader(handle):
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
        if not meta:
            meta = {k: row[k] for k in ("panel_id", "dataset", "outcome", "split_mode", "test_size", "n_train_total", "n_test_total")}
        if row["status"] == "ok" and row["brier"] != "":
            r2 = 1.0 - float(row["brier"]) / null_brier[int(row["seed"])]
            values.setdefault((row["model"], int(row["N"]), int(row["K"])), array("d")).append(r2)
cells = []
for (model, n, k), v in sorted(values.items()):
    ordered = sorted(v)
    cells.append(dict(model=model, N=n, K=k, count=len(ordered), mean=math.fsum(ordered) / len(ordered),
                      p05=quantile(ordered, 0.05), min=ordered[0], max=ordered[-1]))
spec = plan["cell_spec"]
payload = dict(run=str(run), table="final.csv (brier column)", task="classification", metric="brier_full_train_r2",
               statuses=statuses, **meta, seeds=len({s for s, _ in spec["resolved_repeat_plan"]}),
               draws=len({d for _, d in spec["resolved_repeat_plan"]}),
               null_brier=sorted(set(null_brier.values())), evaluation_commit="78c77df", cells=cells)
(out / f"aggregate-{meta['panel_id']}.json").write_text(json.dumps(payload, allow_nan=False) + "\n")
print(json.dumps({"panel": meta["panel_id"], "cells": len(cells), "statuses": statuses,
                  "null_brier": sorted(set(null_brier.values()))}), flush=True)
