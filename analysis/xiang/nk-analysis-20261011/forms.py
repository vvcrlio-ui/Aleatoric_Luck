"""Compare learning-curve forms along N at K_max (Viering & Loog 2022, Table I) on the loss ratio L = 1 - R-squared:
POW3 a + b t^-c, EXP3 a + b exp(-c t), MMF4 (a b + c t^d) / (b + t^d), WBL4 c + b exp(-a t^d), with t = N / N_min,
fitted on the whole curve (asymptote per form), and on N <= N_max / 4 to predict the larger N (held-out
extrapolation, with LOG2 a + b log t and the last fitted value as baselines). POW3 is also refitted without the
five smallest N. Every model except OLS; cells with skipped repeats are left out.
Usage: python forms.py MATRIX_DIR OUT_DIR
"""
import csv
from multiprocessing import Pool
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares

# name: (curve of parameters p and t, asymptote, lower bounds, upper bounds, starting points from the loss)
FORMS = {
    "POW3": (lambda p, t: p[0] + p[1] * t ** -p[2], lambda p: p[0], [0, 0, 0.01], [2, np.inf, 5],
             lambda lo, gap, first: [[lo, gap, c] for c in (0.5, 0.1, 1.5)]),
    "EXP3": (lambda p, t: p[0] + p[1] * np.exp(-p[2] * t), lambda p: p[0], [0, 0, 1e-6], [2, np.inf, 10],
             lambda lo, gap, first: [[lo, gap, c] for c in (0.1, 0.01, 0.001)]),
    "MMF4": (lambda p, t: (p[0] * p[1] + p[2] * t ** p[3]) / (p[1] + t ** p[3]), lambda p: p[2],
             [0, 1e-9, 0, 0.01], [10, np.inf, 2, 5],
             lambda lo, gap, first: [[first, b, lo, d] for b, d in ((1, 0.5), (10, 1), (100, 0.3))]),
    "WBL4": (lambda p, t: p[2] + p[1] * np.exp(-p[0] * t ** p[3]), lambda p: p[2], [0, 0, 0, 0.01],
             [10, np.inf, 2, 5], lambda lo, gap, first: [[a, gap, lo, d] for a, d in ((0.1, 0.5), (0.01, 1), (1, 0.2))]),
}


def fit(form, t, loss, se):
    """Best of three starts; returns the parameters, or None when every start fails."""
    curve, _, lower, upper, starts = FORMS[form]
    lower, upper = np.array(lower, dtype=float), np.array(upper, dtype=float)
    best = None
    for start in starts(max(loss.min() * 0.95, 0), max(loss[0] - loss.min(), 1e-6), loss[0]):
        p0 = np.clip(start, lower + 1e-9, upper - 1e-9)
        result = least_squares(lambda p: (curve(p, t) - loss) / se, p0, bounds=(lower, upper))
        if np.isfinite(result.cost) and (best is None or result.cost < best.cost):
            best = result
    return None if best is None else best.x


def panel(path):
    z = np.load(path)
    name = path.stem.removeprefix("matrix-")
    values, models, n_grid = z["values"].astype(np.float64), list(z["models"]), z["N"].astype(float)
    seed_means = values[:, :, -1].mean(-1)  # model x N x seed, at K_max
    complete = ~np.isnan(values[:, :, -1]).any(axis=(2, 3))
    rows = []
    for m, model in enumerate(models):
        if model == "ols":
            continue
        use = complete[m]
        x, loss = n_grid[use], 1 - seed_means[m][use].mean(-1)
        se = np.maximum(seed_means[m][use].std(-1, ddof=1) / np.sqrt(seed_means.shape[-1]), 1e-6)
        t = x / x[0]
        train = x <= n_grid[-1] / 4
        test = ~train
        row = dict(panel=name, model=model, points=int(use.sum()), train_points=int(train.sum()),
                   test_points=int(test.sum()), R2_observed_at_N_max=1 - loss[-1])
        for form, (curve, asymptote, *_) in FORMS.items():
            p = fit(form, t, loss, se)
            if p is not None:
                row[f"{form}_R2_inf"] = 1 - asymptote(p)
                row[f"{form}_rmse"] = float(np.sqrt(np.mean(((curve(p, t) - loss) / se) ** 2)))
                row[f"{form}_at_bound"] = bool(asymptote(p) < 1e-6 or asymptote(p) > 2 - 1e-6)
            p = fit(form, t[train], loss[train], se[train]) if train.sum() > len(FORMS[form][2]) else None
            if p is not None:
                row[f"{form}_holdout_mae"] = float(np.mean(np.abs(curve(p, t[test]) - loss[test])))
        log2 = np.polyfit(np.log(t[train]), loss[train], 1, w=1 / se[train])
        row["LOG2_holdout_mae"] = float(np.mean(np.abs(np.polyval(log2, np.log(t[test])) - loss[test])))
        row["LAST_holdout_mae"] = float(np.mean(np.abs(loss[train][-1] - loss[test])))
        late = x >= n_grid[5]
        p = fit("POW3", x[late] / x[late][0], loss[late], se[late])
        row["POW3_without_small_N_R2_inf"] = 1 - p[0]
        rows.append(row)
    return rows


if __name__ == "__main__":
    out = Path(sys.argv[2])
    with Pool(14) as pool:
        rows = [r for part in pool.map(panel, sorted(Path(sys.argv[1]).glob("matrix-*.npz"))) for r in part]
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (out / "forms.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)
    print(len(rows), "rows")
