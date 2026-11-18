"""Compare 50 seeds x 10 draws with 100 seeds x 5 draws on a finished 100 x 50 run.

For every pipeline (phase, model, variant) and cell (N, K) with all 100 x 50
results, a balanced nested random-effects ANOVA on r2_test gives the
between-seed variance s2_seed and the within-seed (between-draw) variance
s2_draw. The variance of a cell mean under S seeds x D draws is
s2_seed / S + s2_draw / (S * D).

Empirical check: the actual sub-designs (first 50 seeds x first 10 draws, all
100 seeds x first 5 draws) are nested in the full design, so under the model
E[(sub mean - full mean)^2] = V(sub) - V(full). Their observed squared
deviations, summed over cells, are compared with that prediction.

Usage: analyze.py FINAL_CSV LABEL OUTPUT_JSON
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

COLUMNS = ["phase", "model", "variant_id", "seed", "draw", "N", "K", "r2_test", "status"]
DESIGNS = {"50x10": (50, 10), "100x5": (100, 5), "100x50": (100, 50)}


def variance(s2_seed, s2_draw, seeds, draws):
    return s2_seed / seeds + s2_draw / (seeds * draws)


def main(path, label, out):
    parts = []
    for chunk in pd.read_csv(path, usecols=COLUMNS, chunksize=2_000_000,
                             dtype={"phase": str, "model": str, "variant_id": str, "status": str}):
        chunk = chunk[(chunk["status"] == "ok") & np.isfinite(chunk["r2_test"])]
        parts.append(chunk.drop(columns="status"))
    frame = pd.concat(parts, ignore_index=True)
    frame["variant_id"] = frame["variant_id"].fillna("")
    seeds = np.sort(frame["seed"].unique())
    draws = np.sort(frame["draw"].unique())
    assert len(seeds) == 100 and len(draws) == 50, (len(seeds), len(draws))
    frame["s"] = np.searchsorted(seeds, frame["seed"])
    frame["d"] = np.searchsorted(draws, frame["draw"])
    cells, skipped = [], 0
    for key, group in frame.groupby(["phase", "model", "variant_id", "N", "K"], sort=True):
        if len(group) != 5000 or group[["s", "d"]].duplicated().any():
            skipped += 1
            continue
        y = np.full((100, 50), np.nan)
        y[group["s"].to_numpy(), group["d"].to_numpy()] = group["r2_test"].to_numpy()
        seed_means = y.mean(axis=1)
        ms_within = y.var(axis=1, ddof=1).mean()
        ms_between = 50 * seed_means.var(ddof=1)
        s2_draw = ms_within
        s2_seed = (ms_between - ms_within) / 50  # unbiased, may be negative
        full = y.mean()
        sub = {"50x10": y[:50, :10].mean(), "100x5": y[:, :5].mean()}
        cells.append(dict(phase=key[0], model=key[1], variant_id=key[2], N=int(key[3]), K=int(key[4]),
                          s2_seed=float(s2_seed), s2_draw=float(s2_draw),
                          dev_50x10=float((sub["50x10"] - full) ** 2), dev_100x5=float((sub["100x5"] - full) ** 2)))
    table = pd.DataFrame(cells)
    table["pipeline"] = table["phase"] + "/" + table["model"] + table["variant_id"].map(lambda v: f"/{v}" if v else "")
    for name, (s, d) in DESIGNS.items():
        table[f"V_{name}"] = variance(table["s2_seed"], table["s2_draw"], s, d)
    table["ratio"] = table["V_50x10"] / table["V_100x5"]
    table["icc"] = table["s2_seed"] / (table["s2_seed"] + table["s2_draw"])

    def summary(part):
        expected = {n: float((part[f"V_{n}"] - part["V_100x50"]).sum()) for n in ("50x10", "100x5")}
        observed = {n: float(part[f"dev_{n}"].sum()) for n in ("50x10", "100x5")}
        return {"cells": int(len(part)),
                "pooled_variance_ratio_50x10_over_100x5": float(part["V_50x10"].sum() / part["V_100x5"].sum()),
                "ratio_quantiles": {q: float(part["ratio"].quantile(q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
                "icc_median": float(part["icc"].median()),
                "icc_quantiles": {q: float(part["icc"].quantile(q)) for q in (0.05, 0.5, 0.95)},
                "share_negative_s2_seed": float((part["s2_seed"] < 0).mean()),
                "check_expected_sum_sq_dev": expected, "check_observed_sum_sq_dev": observed,
                "observed_dev_ratio_50x10_over_100x5": observed["50x10"] / observed["100x5"]}

    n_values = np.sort(table["N"].unique())
    band = pd.cut(table["N"], [0, n_values[6], n_values[13], np.inf], labels=["small", "middle", "large"])
    result = {"label": label, "path": path, "rows_ok": int(len(frame)), "cells_skipped_incomplete": skipped,
              "N_bands": {"small": [int(n_values[0]), int(n_values[6])], "middle": [int(n_values[7]), int(n_values[13])],
                          "large": [int(n_values[14]), int(n_values[-1])]},
              "all": summary(table),
              "by_pipeline": {p: summary(g) for p, g in table.groupby("pipeline")},
              "by_N_band": {str(b): summary(g) for b, g in table.groupby(band, observed=True)}}
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=1)
    table.to_csv(out.replace(".json", ".cells.csv"), index=False)
    print(json.dumps(result["all"], indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
