"""Plot the power-law fits from power_law.py: learning curves along N with all variables, extrapolated to
10 x N_max, and a table of the fitted ceilings with their seed-bootstrap 95% intervals. A ceiling counts as
identified when the lower 95% bound of the decay rate c is at least 0.1; below that the curve cannot be told apart
from a log-linear trend inside the data, and the asymptote is set by the parameter bounds, not the data.
Usage: python plot_power_law.py RESULTS_DIR
"""
import csv
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
os.environ["MPLCONFIGDIR"] = str(HERE / ".mplconfig")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from panels import MODELS, MUTED, ROWS, TITLE

results = Path(sys.argv[1])
curves = json.loads((results / "power-law-curves.json").read_text())
fits = {(r["panel"], r["model"]): r for r in csv.DictReader((results / "power-law-fits.csv").open())
        if r["R2_inf_lo"] != ""}
identified = {key: float(r["c_lo"]) >= 0.1 for key, r in fits.items()}
forms = {(r["panel"], r["model"]): r for r in csv.DictReader((results / "forms.csv").open())}
RANGE_FORMS, FORM = ("POW3", "MMF4", "WBL4"), "#7d5a2b"
fig, axes = plt.subplots(len(ROWS), 2, figsize=(15, 4 + 3.3 * len(ROWS)), facecolor="white")
fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.07, wspace=0.18, hspace=0.5)
for r, row in enumerate(ROWS):
    for p, (panel, label) in enumerate(row):
        ax = axes[r, p]
        low, high = 0.0, 0.0
        for model, name, color in MODELS:
            c = curves[panel].get(f"{model}/N")
            if c is None:
                continue
            x = np.array(c["x"])
            grid = np.geomspace(x[0], 10 * x[-1], 200)
            fitted = 1 - (c["L_inf"] + c["b"] * (grid / c["x0"]) ** -c["c"])
            inside = grid <= x[-1]
            width = 1.8 if model == "super_learner" else 1.1
            ax.plot(grid[inside], fitted[inside], color=color, linewidth=width, label=name)
            if identified[(panel, model)]:
                ax.plot(grid[~inside], fitted[~inside], color=color, linewidth=width, linestyle=(0, (3, 2)))
            ax.scatter(x, c["R2"], s=9, color=color, zorder=3)
            low = min(low, min(c["R2"][len(c["R2"]) // 3:]))
            high = max(high, max(c["R2"]), fitted[-1] if identified[(panel, model)] else 0)
        ax.axvline(curves[panel][f"{MODELS[0][0]}/N"]["x"][-1], color="#9aa3aa", linewidth=0.8, linestyle=":")
        ax.axhline(0, color="#9aa3aa", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_ylim(max(low - 0.02, -0.3), high + 0.03)
        ax.grid(color="#e6e8ea", linewidth=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_title(label, fontsize=10.5, color=TITLE, fontweight="bold", loc="left")
        ax.set_xlabel("Sample size $N$ (all variables)", fontsize=9)
        if p == 0:
            ax.set_ylabel("Mean test $R^2$", fontsize=9)
fig.text(0.07, 0.965, "Learning curves and where they level off", fontsize=19, fontweight="bold", color=TITLE)
fig.text(0.07, 0.952, "Dots: mean test R² over 500 repeats at each N with all variables. Lines: weighted fit of "
         "1 − R² = L∞ + b·N^(−c); solid within the data, dashed out to 10 × the training pool (dotted line)\n"
         "where the ceiling is identified (lower 95% bound of c ≥ 0.1; seed bootstrap, 1,000 resamples).",
         fontsize=10.5, color=MUTED)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.02))
fig.text(0.07, 0.008, "OLS omitted (interpolation peak near N = K). Classification: cells with skipped repeats are "
         "left out of the fit. Y axis cut at −0.3; the smallest N may fall below it.", fontsize=8.5, color=MUTED)
for suffix in ("png", "pdf"):
    fig.savefig(results / f"power-law-curves.{suffix}", dpi=150)
plt.close(fig)

# Table: fitted ceiling and interval per panel and model, next to the best observed mean at N_max.
COLUMNS = [("Panel", 3.0), ("Best mean\nat N_max", 1.05)] + [(name.replace(" ", "\n", 1), 1.2) for _, name, _ in MODELS]
X = np.cumsum([0] + [w for _, w in COLUMNS[:-1]])
TOTAL, ROW_H, HEAD_H = sum(w for _, w in COLUMNS), 0.66, 0.75
panels = [cell for row in ROWS for cell in row]
height = HEAD_H + ROW_H * len(panels)
fig = plt.figure(figsize=(TOTAL + 0.6, height + 2.6), facecolor="white")
ax = fig.add_axes([0.3 / (TOTAL + 0.6), 1.5 / (height + 2.6), TOTAL / (TOTAL + 0.6), height / (height + 2.6)])
ax.set_xlim(0, TOTAL)
ax.set_ylim(height, 0)
ax.axis("off")
for i, (name, width) in enumerate(COLUMNS):
    ax.text(X[i] + (0.08 if i == 0 else width / 2), HEAD_H / 2, name, ha="left" if i == 0 else "center", va="center",
            fontsize=9, fontweight="bold", linespacing=1.15)
for y in (0, HEAD_H, height):
    ax.plot([0, TOTAL], [y, y], color="#222222", linewidth=1.0)
for r, (panel, label) in enumerate(panels):
    y = HEAD_H + r * ROW_H
    if r % 2 == 0:
        ax.add_patch(plt.Rectangle((0, y), TOTAL, ROW_H, color="#f4f5f6", zorder=0, linewidth=0))
    ax.text(X[0] + 0.08, y + ROW_H / 2, label, va="center", fontsize=9)
    observed = max(float(fits[(panel, m)]["R2_observed_at_x_max"]) for m, _, _ in MODELS)
    ax.text(X[1] + COLUMNS[1][1] / 2, y + ROW_H / 2, f"{observed:.3f}", ha="center", va="center", fontsize=9,
            color=MUTED)
    for j, (model, _, _) in enumerate(MODELS):
        f, cx = fits[(panel, model)], X[2 + j] + COLUMNS[2 + j][1] / 2
        if identified[(panel, model)]:
            ax.text(cx, y + ROW_H * 0.27, f"{float(f['R2_inf']):.3f}", ha="center", va="center", fontsize=9.5)
            ax.text(cx, y + ROW_H * 0.57, f"[{float(f['R2_inf_lo']):.3f}, {float(f['R2_inf_hi']):.3f}]",
                    ha="center", va="center", fontsize=7, color=MUTED)
            r2 = [float(forms[(panel, model)][f"{form}_R2_inf"]) for form in RANGE_FORMS
                  if forms[(panel, model)][f"{form}_at_bound"] == "False"]
            ax.text(cx, y + ROW_H * 0.82, f"forms {min(r2):.3f}–{max(r2):.3f}".replace("-", "−"), ha="center", va="center",
                    fontsize=7, color=FORM)
        else:
            ax.text(cx, y + ROW_H / 2, "not identified", ha="center", va="center", fontsize=7.5, color="#9aa3aa",
                    style="italic")
fig.text(0.3 / (TOTAL + 0.6), 1 - 0.25 / (height + 2.6), "Fitted ceiling of mean test R² as N grows, all variables",
         fontsize=15, fontweight="bold", color=TITLE, va="top")
fig.text(0.3 / (TOTAL + 0.6), 1 - 0.62 / (height + 2.6),
         "R²∞ = 1 − L∞ from 1 − R² = L∞ + b·N^(−c), weighted by the seed standard error of each cell mean; "
         "95% interval from 1,000 seed resamples.\nThird line: range of the ceiling across the POW3, MMF4 and WBL4 "
         "forms (forms.py).", fontsize=10, color=MUTED, va="top", linespacing=1.5)
fig.text(0.3 / (TOTAL + 0.6), 1.25 / (height + 2.6),
         "Not identified: lower 95% bound of the decay rate c below 0.1, so inside the data the curve is close to "
         "log-linear and the ceiling follows the parameter bounds rather than the data.\nForms whose ceiling sits on a bound "
         "are left out of the range; EXP3 is left out because it extrapolates held-out N about three times worse "
         "(extrapolation-check).\nCeilings extrapolate beyond the "
         "training pool and assume the curve keeps its fitted shape. OLS omitted. Classification: FFC Brier R²; cells "
         "with skipped repeats are left out of the fit.", fontsize=8.3, color=MUTED, va="top", linespacing=1.5)
for suffix in ("png", "pdf"):
    fig.savefig(results / f"power-law-ceilings.{suffix}", dpi=200)
print(results / "power-law-ceilings.png")
