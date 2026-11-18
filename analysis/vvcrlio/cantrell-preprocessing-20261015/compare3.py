"""Compare cantrell_metadata, median_mode and tree_ordinal production results for GPA and MH.

Usage: python compare3.py RESULTS_DIR OUT_DIR
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
RUNS = {
    "gpa": {"cantrell": "ffc_cantrell_metadata_gpa-3cd0749879ae",
            "median_mode": "ffc_median_mode_gpa-9928d0cf2a23",
            "tree_ordinal": "ffc_tree_ordinal_gpa-e2d2d3089950"},
    "materialHardship": {"cantrell": "ffc_cantrell_metadata_materialHardship-2b19d40ec5e2",
                         "median_mode": "ffc_median_mode_materialHardship-4918ff834e52",
                         "tree_ordinal": "ffc_tree_ordinal_materialHardship-8fdccdbe0c6a"},
    "grit": {"cantrell": "ffc_cantrell_metadata_grit-4dbc85334733",
             "median_mode": "ffc_median_mode_grit-9ba5df2fbef5",
             "tree_ordinal": "ffc_tree_ordinal_grit-35424ef4643b"},
}
ONLY = sys.argv[3:] or list(RUNS)
ENCODINGS = ["cantrell", "median_mode", "tree_ordinal"]
COLS = ["model", "seed", "draw", "N", "K", "K_expanded", "status", "error", "r2_full_train_mean"]
MATCH_K = [10, 108, 442, 1127, 3401]
pd.set_option("display.width", 200)

for outcome, runs in RUNS.items():
    if outcome not in ONLY:
        continue
    cells, best_rows = [], []
    for encoding in ENCODINGS:
        frame = pd.read_csv(ROOT / runs[encoding] / "final.csv", usecols=COLS, low_memory=False)
        skipped = frame[frame.status == "skipped"]
        if len(skipped):
            print(f"{outcome} {encoding}: {len(skipped)} skipped rows, {len(skipped) // 9} cells; "
                  f"N {skipped.N.min()}-{skipped.N.max()}, K {sorted(skipped.K.unique())}")
            print(skipped.error.value_counts().to_string())
        frame = frame[frame.status == "ok"]
        best_rows.append(frame.loc[frame.groupby("model").r2_full_train_mean.idxmax()].assign(encoding=encoding))
        cell = (frame.groupby(["model", "N", "K"])
                .agg(r2_mean=("r2_full_train_mean", "mean"), r2_sd=("r2_full_train_mean", "std"),
                     reps=("r2_full_train_mean", "size"), k_expanded=("K_expanded", "mean"))
                .reset_index().assign(encoding=encoding))
        cells.append(cell)
    cells = pd.concat(cells, ignore_index=True)
    best_rows = pd.concat(best_rows, ignore_index=True)
    cells.to_csv(OUT / f"cell_means_{outcome}.csv", index=False)
    best_rows.to_csv(OUT / f"best_rows_{outcome}.csv", index=False)

    print(f"\n===== {outcome}")
    print("Best single row (any seed, draw, N, K) per model:")
    print(best_rows.pivot(index="model", columns="encoding", values="r2_full_train_mean")[ENCODINGS]
          .sort_values("cantrell", ascending=False).round(4).to_string())
    top = best_rows.loc[best_rows.groupby("encoding").r2_full_train_mean.idxmax()]
    print(top[["encoding", "model", "seed", "draw", "N", "K", "K_expanded", "r2_full_train_mean"]]
          .round(5).to_string(index=False))

    nmax = cells.N.max()
    at_n = cells[cells.N == nmax]
    full = at_n.loc[at_n.groupby(["encoding", "model"]).K.idxmax()]
    print(f"\nMean over 500 seed/draw at N = {nmax} with all variables of each pool "
          f"(max cell-mean SE {float((full.r2_sd / np.sqrt(full.reps)).max()):.4f}):")
    print(full.pivot(index="model", columns="encoding", values="r2_mean")[ENCODINGS].round(4).to_string())
    best_mean = cells.loc[cells.groupby(["encoding", "model"]).r2_mean.idxmax()]
    print("\nBest (N, K) cell mean per model:")
    print(best_mean.pivot(index="model", columns="encoding", values="r2_mean")[ENCODINGS].round(4).to_string())

    print(f"\nMean at N = {nmax} and the same K (cantrell interpolated on log K):")
    rows = []
    for model in sorted(at_n.model.unique()):
        for encoding in ENCODINGS:
            curve = at_n[(at_n.model == model) & (at_n.encoding == encoding)].sort_values("K")
            values = np.interp(np.log(MATCH_K), np.log(curve.K), curve.r2_mean)
            rows.append({"model": model, "encoding": encoding, **{f"K{k}": v for k, v in zip(MATCH_K, values)}})
    matched = pd.DataFrame(rows).set_index(["model", "encoding"])
    print(matched.round(4).to_string())
    matched.to_csv(OUT / f"matched_k_{outcome}.csv")
