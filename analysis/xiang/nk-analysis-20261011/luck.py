"""Within-seed and between-seed variance of the score in every complete cell.

A draw reorders both the training rows and the variables, so the two cannot be separated inside the grid. On its
edges they can: at K_max every draw uses all variables, so the within-seed spread along N is sample luck plus
algorithmic noise; at N_max every draw uses the whole training pool, so the spread along K is variable luck plus
algorithmic noise; at (N_max, K_max) only algorithmic noise (model seed, CV folds) is left. Between-seed variance
is the test-split luck of internally split panels and should be near zero for a fixed external test set.
Usage: python luck.py MATRIX_DIR OUT_DIR
"""
import csv
import json
from pathlib import Path
import sys

import numpy as np

out = Path(sys.argv[2])
out.mkdir(exist_ok=True)
summary, edges, rows = {}, {}, []
for path in sorted(Path(sys.argv[1]).glob("matrix-*.npz")):
    z = np.load(path)
    name = path.stem.removeprefix("matrix-")
    meta = json.loads(path.with_suffix(".json").read_text())
    values, models, n_grid, k_grid = z["values"].astype(np.float64), list(z["models"]), z["N"], z["K"]
    draws = values.shape[-1]
    complete = ~np.isnan(values).any(axis=(3, 4))
    within = values.var(-1, ddof=1).mean(-1)
    # One-way random-effects ANOVA: Var(seed means) = between + within / draws.
    between = np.maximum(values.mean(-1).var(-1, ddof=1) - within / draws, 0)
    within[~complete], between[~complete] = np.nan, np.nan
    share = between / (between + within)
    np.savez_compressed(out / f"variance-{name}.npz", within=within, between=between, models=z["models"],
                        N=n_grid, K=k_grid)
    edges[name] = {model: dict(N=n_grid.tolist(), K=k_grid.tolist(),
                               sample_sd=np.sqrt(within[m, :, -1]).tolist(),
                               variable_sd=np.sqrt(within[m, -1, :]).tolist(),
                               algorithmic_sd=float(np.sqrt(within[m, -1, -1])))
                   for m, model in enumerate(models)}
    for m, model in enumerate(models):
        for i, n in enumerate(n_grid):
            for j, k in enumerate(k_grid):
                if complete[m, i, j]:
                    rows.append(dict(panel=name, model=model, N=int(n), K=int(k), within_sd=np.sqrt(within[m, i, j]),
                                     between_sd=np.sqrt(between[m, i, j]), between_share=share[m, i, j]))
    summary[name] = dict(task=meta["task"], metric=meta["metric"],
                         complete_cells=int(complete.sum()), cells=int(complete.size),
                         between_share_median=float(np.nanmedian(share)),
                         between_share_p95=float(np.nanpercentile(share, 95)),
                         N_max=int(n_grid[-1]), n_train_total=meta["n_train_total"],
                         K_max=int(k_grid[-1]), n_features_total=meta["n_features_total"])
    assert n_grid[-1] == meta["n_train_total"] and k_grid[-1] == meta["n_features_total"], name
    print(json.dumps({name: summary[name]}), flush=True)
with (out / "variance-cells.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(out / "luck-edges.json").write_text(json.dumps(edges))
(out / "luck-summary.json").write_text(json.dumps(summary, indent=2))
