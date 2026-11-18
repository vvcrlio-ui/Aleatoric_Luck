"""Per-cell mean, P5, min and max of the full-training-pool score of published runs, for plotting.

Regression uses r2_full_train_mean; classification the Fragile Families Challenge R-squared
brier_full_train_r2 where the table has it, else mcfadden_full_train_pseudo_r2; ok rows only.
Usage: python aggregate_panels.py OUT_DIR TABLE RUN_DIR...   (TABLE: final.csv or final.pending.csv)
"""
from array import array
import csv
import json
import math
from pathlib import Path
import sys

out = Path(sys.argv[1])
csv.field_size_limit(1 << 24)


def quantile(ordered, q):
    """Linear interpolation between closest ranks, as numpy.quantile's default."""
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


table = sys.argv[2]
for run in map(Path, sys.argv[3:]):
    plan = json.loads((run / "plan.json").read_text())
    # Runs from before the columns existed record where they were added; later runs computed them.
    added = run / "added-columns.json"
    commit = json.loads(added.read_text())["evaluation_commit"] if added.exists() else plan["cell_spec"]["git_commit"]
    task = plan["task_kind"]
    values, statuses, meta = {}, {}, {}
    with (run / table).open(newline="") as handle:
        reader = csv.DictReader(handle)
        metric = ("r2_full_train_mean" if task == "regression" else
                  "brier_full_train_r2" if "brier_full_train_r2" in reader.fieldnames else "mcfadden_full_train_pseudo_r2")
        for row in reader:
            statuses[row["status"]] = statuses.get(row["status"], 0) + 1
            if not meta:
                meta = {k: row[k] for k in ("panel_id", "dataset", "outcome", "split_mode", "test_size", "n_train_total", "n_test_total")}
            if row["status"] == "ok" and row[metric] != "":
                values.setdefault((row["model"], int(row["N"]), int(row["K"])), array("d")).append(float(row[metric]))
    cells = []
    for (model, n, k), v in sorted(values.items()):
        ordered = sorted(v)
        cells.append(dict(model=model, N=n, K=k, count=len(ordered), mean=math.fsum(ordered) / len(ordered),
                          p05=quantile(ordered, 0.05), min=ordered[0], max=ordered[-1]))
    spec = plan["cell_spec"]
    payload = dict(run=str(run), table=table, task=task, metric=metric, statuses=statuses, **meta,
                   seeds=len({s for s, _ in spec["resolved_repeat_plan"]}), draws=len({d for _, d in spec["resolved_repeat_plan"]}),
                   evaluation_commit=commit, cells=cells)
    (out / f"aggregate-{meta['panel_id']}.json").write_text(json.dumps(payload, allow_nan=False) + "\n")
    print(json.dumps({"panel": meta["panel_id"], "task": task, "cells": len(cells), "statuses": statuses}), flush=True)
