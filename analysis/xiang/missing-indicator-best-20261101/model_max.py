"""Highest single result per model for one final.csv: r2_full_train_mean (regression) or brier_full_train_r2 (binary).

Usage: python model_max.py FINAL_CSV OUT_CSV
"""
import sys
import pandas as pd

head = pd.read_csv(sys.argv[1], nrows=0).columns
metric = "brier_full_train_r2" if "brier_full_train_r2" in head else "r2_full_train_mean"
d = pd.read_csv(sys.argv[1], usecols=["dataset", "model", "N", "K", "seed", "draw", metric, "status"])
d = d[d.status == "ok"]
best = d.loc[d.groupby("model")[metric].idxmax()].rename(columns={metric: "best"})
best["metric"] = metric
best.drop(columns="status").to_csv(sys.argv[2], index=False)
print(best.drop(columns="status").to_string(index=False))
