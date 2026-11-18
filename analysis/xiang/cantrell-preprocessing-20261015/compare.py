"""Compare cantrell_metadata and median_mode production results for GPA and MH."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
RUNS = {
    ("gpa", "cantrell"): "ffc_cantrell_metadata_gpa-3cd0749879ae",
    ("gpa", "median_mode"): "ffc_median_mode_gpa-9928d0cf2a23",
    ("materialHardship", "cantrell"): "ffc_cantrell_metadata_materialHardship-2b19d40ec5e2",
    ("materialHardship", "median_mode"): "ffc_median_mode_materialHardship-4918ff834e52",
}
COLS = ["model", "seed", "draw", "N", "K", "K_expanded", "K_unobserved", "status", "error", "r2_full_train_mean"]

cells = []
for (outcome, encoding), run in RUNS.items():
    frame = pd.read_csv(ROOT / run / "final.csv", usecols=COLS, low_memory=False)
    print(outcome, encoding, len(frame), frame.status.value_counts().to_dict())
    skipped = frame[frame.status == "skipped"]
    if len(skipped):
        print("  skipped by reason:")
        print(skipped.groupby("error").size().to_string())
        print("  skipped N range", skipped.N.min(), skipped.N.max(), "K values", sorted(skipped.K.unique()))
        print("  K_unobserved among skipped:", skipped.K_unobserved.describe()[["min", "max"]].to_dict())
        skipped.groupby(["N", "K", "model"]).size().rename("rows").to_csv(OUT / f"skipped_{outcome}_{encoding}.csv")
    ok = frame[frame.status == "ok"]
    cell = (ok.groupby(["model", "N", "K"])
              .agg(r2_mean=("r2_full_train_mean", "mean"), r2_sd=("r2_full_train_mean", "std"),
                   reps=("r2_full_train_mean", "size"), k_expanded=("K_expanded", "mean"))
              .reset_index())
    cell["outcome"], cell["encoding"] = outcome, encoding
    cells.append(cell)

cells = pd.concat(cells, ignore_index=True)
cells.to_csv(OUT / "cell_means.csv", index=False)

for outcome in ["gpa", "materialHardship"]:
    c = cells[cells.outcome == outcome]
    print(f"\n===== {outcome}")
    nmax = c.N.max()
    top = c[c.N == nmax]
    full = top.loc[top.groupby(["encoding", "model"]).K.idxmax()]
    best = c.loc[c.groupby(["encoding", "model"]).r2_mean.idxmax()]
    table = (full.pivot(index="model", columns="encoding", values="r2_mean")
             .add_prefix("allK_")
             .join(best.pivot(index="model", columns="encoding", values="r2_mean").add_prefix("best_")))
    table["allK_diff"] = table.allK_cantrell - table.allK_median_mode
    table["best_diff"] = table.best_cantrell - table.best_median_mode
    print(f"N = {nmax}; allK = all variables of each pool; best = best (N, K) cell mean")
    print(table.round(4).to_string())
    print("best cells:")
    print(best[["encoding", "model", "N", "K", "r2_mean", "reps"]].round(4).to_string(index=False))
    shared = sorted(set(top[top.encoding == "cantrell"].K) & set(top[top.encoding == "median_mode"].K))
    sl = top[(top.model == "super_learner") & top.K.isin(shared)].pivot(index="K", columns="encoding", values="r2_mean")
    sl["diff"] = sl.cantrell - sl.median_mode
    print(f"SL7 at N = {nmax} on K values in both grids:")
    print(sl.round(4).to_string())
