"""Export a published run's full-training-pool score as an array: model x N x K x seed x draw.

Regression uses r2_full_train_mean, classification brier_full_train_r2; rows that are not ok are NaN.
Usage: python export_matrix.py OUT_DIR RUN_DIR...
"""
import csv
import json
from pathlib import Path
import sys

import numpy as np

MODELS = ["ols", "ridge", "lasso", "random_forest", "shallow_neural_network", "extra_trees", "xgboost", "lightgbm",
          "super_learner"]
out = Path(sys.argv[1])
csv.field_size_limit(1 << 24)
for run in map(Path, sys.argv[2:]):
    plan = json.loads((run / "plan.json").read_text())
    task = plan["task_kind"]
    metric = "r2_full_train_mean" if task == "regression" else "brier_full_train_r2"
    seeds = sorted({s for s, _ in plan["cell_spec"]["resolved_repeat_plan"]})
    draws = sorted({d for _, d in plan["cell_spec"]["resolved_repeat_plan"]})
    rows = []
    with (run / "final.csv").open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        at = {name: header.index(name) for name in ("panel_id", "model", "seed", "draw", "N", "K", "status", metric,
                                                     "n_train_total", "n_features_total")}
        for row in reader:
            rows.append((row[at["model"]], int(row[at["N"]]), int(row[at["K"]]), int(row[at["seed"]]),
                         int(row[at["draw"]]), float(row[at[metric]]) if row[at["status"]] == "ok" else np.nan))
            meta = row
    n_grid = sorted({r[1] for r in rows})
    k_grid = sorted({r[2] for r in rows})
    index = [{v: i for i, v in enumerate(axis)} for axis in (MODELS, n_grid, k_grid, seeds, draws)]
    values = np.full([len(axis) for axis in index], np.nan, dtype=np.float32)
    filled = np.zeros(values.shape, dtype=bool)
    for r in rows:
        position = tuple(axis[v] for axis, v in zip(index, r[:5]))
        assert not filled[position], r
        filled[position] = True
        values[position] = r[5]
    assert filled.all(), int((~filled).sum())
    panel = meta[at["panel_id"]]
    np.savez_compressed(out / f"matrix-{panel}.npz", values=values, models=np.array(MODELS), N=np.array(n_grid),
                        K=np.array(k_grid), seeds=np.array(seeds), draws=np.array(draws))
    (out / f"matrix-{panel}.json").write_text(json.dumps(dict(
        panel=panel, run=str(run), task=task, metric=metric, rows=len(rows), nan=int(np.isnan(values).sum()),
        n_train_total=int(meta[at["n_train_total"]]), n_features_total=int(meta[at["n_features_total"]]))) + "\n")
    print(json.dumps({"panel": panel, "rows": len(rows), "nan": int(np.isnan(values).sum()),
                      "N_max": n_grid[-1], "K_max": k_grid[-1], "n_train_total": meta[at["n_train_total"]],
                      "n_features_total": meta[at["n_features_total"]]}), flush=True)
