"""Fit L = L_inf + b * (x / x_0)^(-c) to the cell-mean loss ratio L = 1 - R-squared, along N at every K and
along K at every N, for every model except OLS. Along N at K_max, a seed bootstrap gives 95% intervals; the
fits along K are kept as point estimates only (the K curves are not power laws). Only cells with all seed x draw
scores are used.
Usage: python power_law.py MATRIX_DIR OUT_DIR
"""
import csv
import json
from multiprocessing import Pool
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares

BOOTSTRAP = 1000
FIELDS = ["panel", "model", "direction", "fixed", "points", "x_min", "x_max", "L_inf", "b", "c", "rmse",
          "R2_inf", "R2_observed_at_x_max", "x_90", "R2_inf_lo", "R2_inf_hi", "c_lo", "c_hi", "x_90_lo", "x_90_hi"]


def fit(x, loss, se):
    """Weighted least squares; returns L_inf, b, c on x scaled by its first value, and the weighted RMSE."""
    t = x / x[0]
    start = [max(loss.min() * 0.95, 0.0), max(loss[0] - loss.min(), 1e-6), 0.5]
    result = least_squares(lambda p: (p[0] + p[1] * t ** -p[2] - loss) / se, start,
                           bounds=([0, 0, 0.01], [2, np.inf, 5]))
    return (*result.x, float(np.sqrt(np.mean(result.fun ** 2))))


def x_90(x0, l_inf, b, c):
    """Where R-squared reaches 90% of its ceiling: b (x / x_0)^(-c) = 0.1 (1 - L_inf)."""
    return x0 * (b / (0.1 * (1 - l_inf))) ** (1 / c) if l_inf < 1 and b > 0 else np.nan


def curves(seed_means, axis):
    """Cell means and their seed standard errors along ``axis`` (0 = N, 1 = K) of a model's seed-mean array."""
    mean = seed_means.mean(-1)
    se = seed_means.std(-1, ddof=1) / np.sqrt(seed_means.shape[-1])
    return (mean, se) if axis == 0 else (mean.T, se.T)


def panel(path):
    z = np.load(path)
    name = path.stem.removeprefix("matrix-")
    values, models, grids = z["values"].astype(np.float64), list(z["models"]), (z["N"], z["K"])
    complete = ~np.isnan(values).any(axis=(3, 4))
    seed_means = values.mean(-1)  # model x N x K x seed
    rng = np.random.default_rng(20261011)
    resamples = rng.integers(0, seed_means.shape[-1], size=(BOOTSTRAP, seed_means.shape[-1]))
    rows, fitted = [], {}
    for m, model in enumerate(models):
        if model == "ols":
            continue
        for axis, direction in ((0, "N"), (1, "K")):
            x_all, fixed_grid = grids[axis].astype(float), grids[1 - axis]
            mean, se = curves(1 - seed_means[m], axis)
            ok = complete[m] if axis == 0 else complete[m].T
            for j, fixed in enumerate(fixed_grid):
                use = ok[:, j]
                if use.sum() < 5:
                    continue
                x, loss, s = x_all[use], mean[use, j], np.maximum(se[use, j], 1e-6)
                l_inf, b, c, rmse = fit(x, loss, s)
                row = dict(panel=name, model=model, direction=direction, fixed=int(fixed), points=int(use.sum()),
                           x_min=int(x[0]), x_max=int(x[-1]), L_inf=l_inf, b=b, c=c, rmse=rmse, R2_inf=1 - l_inf,
                           R2_observed_at_x_max=1 - loss[-1], x_90=x_90(x[0], l_inf, b, c))
                if direction == "N" and j == len(fixed_grid) - 1:
                    draws = []
                    for idx in resamples:
                        boot = seed_means[m][..., idx]
                        bm, bs = curves(1 - boot, axis)
                        bl, bb, bc, _ = fit(x, bm[use, j], np.maximum(bs[use, j], 1e-6))
                        draws.append((1 - bl, bc, x_90(x[0], bl, bb, bc)))
                    draws = np.array(draws)
                    lo, hi = np.nanpercentile(draws, [2.5, 97.5], axis=0)
                    row.update(R2_inf_lo=lo[0], R2_inf_hi=hi[0], c_lo=lo[1], c_hi=hi[1], x_90_lo=lo[2], x_90_hi=hi[2])
                    fitted[f"{model}/{direction}"] = dict(x=x.tolist(), R2=(1 - loss).tolist(),
                                                          se=s.tolist(), L_inf=l_inf, b=b, c=c, x0=x[0])
                rows.append(row)
    print(json.dumps({"panel": name, "fits": len(rows)}), flush=True)
    return name, rows, fitted


if __name__ == "__main__":
    out = Path(sys.argv[2])
    out.mkdir(exist_ok=True)
    with Pool(14) as pool:
        results = pool.map(panel, sorted(Path(sys.argv[1]).glob("matrix-*.npz")))
    with (out / "power-law-fits.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDS)
        writer.writeheader()
        for _, rows, _ in results:
            writer.writerows(rows)
    (out / "power-law-curves.json").write_text(json.dumps({name: fitted for name, _, fitted in results}))
