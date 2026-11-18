"""Plot a run's P5-to-max test-score bars as flat small multiples: across N at five K, and across K at five N."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
os.environ["MPLCONFIGDIR"] = str(HERE / ".mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np

import range_cells

MODELS = [
    ("ols", "OLS"),
    ("ridge", "Ridge"),
    ("lasso", "Lasso"),
    ("random_forest", "Random forest"),
    ("shallow_neural_network", "Shallow neural\nnetwork"),
    ("extra_trees", "Extra trees"),
    ("xgboost", "XGBoost"),
    ("lightgbm", "LightGBM"),
    ("super_learner", "Super learner"),
]
# Usage: plot_range_bars_2d.py {sampled,full,AGGREGATE.json} -- see range_cells.py.
cells, meta = range_cells.load(sys.argv[1])
OUT = meta["out"]
assert {c["model"] for c in cells} == {name for name, _ in MODELS}

FLOOR, TOP, COLOR_TOP = -0.50, meta["top"], meta["color_top"]
# Upper ticks only where they clear the tick below by 0.1.
Y_TICKS = [FLOOR, -0.25, 0]
for _tick in (COLOR_TOP, TOP):
    if _tick - Y_TICKS[-1] >= 0.1:
        Y_TICKS.append(_tick)


def tick_label(value):
    return "0" if value == 0 else f"{value:.2f}".replace("-", "−")


assert max(c["max_r2"] for c in cells) <= TOP
n_values = sorted({c["N"] for c in cells})
k_values = sorted({c["K"] for c in cells})
# The tick values of the 3D figure: the 1st, 6th, 11th, 16th and 20th grid values of each axis.
K_COLUMNS = [k_values[i] for i in (0, 5, 10, 15, 19)]
N_COLUMNS = [n_values[i] for i in (0, 5, 10, 15, 19)]
by_key = {(c["model"], c["K"], c["N"]): c for c in cells}
norm = TwoSlopeNorm(vmin=FLOOR, vcenter=0, vmax=COLOR_TOP)
# Pink #E7CACC and blue #B6CAE7, deepened towards both ends, around a light grey midpoint at R-squared = 0.
cmap = LinearSegmentedColormap.from_list(
    "pink_grey_blue",
    list(zip([0, 0.2, 0.4, 0.5, 0.6, 0.8, 1], ["#7d3c45", "#b9767d", "#e7cacc", "#ebeae8", "#b6cae7", "#6f92c8", "#2f5596"])),
)
INK = "#222222"
CLIP = "#9b4f58"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "savefig.facecolor": "white"})


def fmt_r2(value):
    return f"{value:.3f}".replace("-", "−")


def bars(ax, row, ring):
    """One panel: a P5-to-max bar per cell of ``row``, the mean as a line; ring the bar at index ``ring``."""
    x = np.arange(len(row))
    p05 = np.array([c["p05_r2"] for c in row])
    high = np.array([c["max_r2"] for c in row])
    mean = np.array([c["r2_test"] for c in row])
    low_d, high_d, mean_d = np.maximum(p05, FLOOR), np.maximum(high, FLOOR), np.maximum(mean, FLOOR)
    ax.axhspan(FLOOR, 0, color="#f5f4f2", zorder=0)
    ax.axhline(0, color="#8a949c", linewidth=0.8, zorder=1)
    ax.bar(x, high_d - low_d, bottom=low_d, width=0.72, color=cmap(norm(mean_d)), edgecolor="white",
           linewidth=0.4, zorder=2)
    clipped = p05 < FLOOR
    ax.scatter(x[clipped], np.full(clipped.sum(), FLOOR + 0.014), marker="v", s=13, color=CLIP, zorder=4,
               clip_on=False)
    ax.plot(x, mean_d, color=INK, linewidth=0.9, zorder=3)
    ax.scatter(x, mean_d, s=10, color=INK, zorder=4)
    if ring is not None:
        ax.scatter([ring], [row[ring]["max_r2"]], s=46, facecolors="white", edgecolors=INK, linewidths=1.3,
                   zorder=5)
    ax.set_ylim(FLOOR, TOP)
    ax.set_yticks(Y_TICKS, [tick_label(t) for t in Y_TICKS])
    ax.set_xlim(-0.7, len(row) - 0.3)
    ax.tick_params(labelsize=8.5, length=2.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#77818a")
    return int(clipped.sum())


fig = plt.figure(figsize=(36, 25), facecolor="white")
LEFT, RIGHT, GAP = 0.075, 0.855, 0.035
half = (RIGHT - LEFT - GAP) / 2
grid = dict(top=0.915, bottom=0.04, wspace=0.07, hspace=0.16)
by_n = fig.add_gridspec(len(MODELS), len(K_COLUMNS), left=LEFT, right=LEFT + half, **grid)
by_k = fig.add_gridspec(len(MODELS), len(N_COLUMNS), left=RIGHT - half, right=RIGHT, **grid)
fig.text(0.02, 0.985, "Spread of prediction performance by sample size, model and feature count",
         fontsize=26, fontweight="bold", color="#203b4c", va="top")
fig.text(0.02, 0.967, meta["subtitle"], fontsize=13, color="#5c6971", va="top")
for left, label in ((LEFT, "Across sample size N, at five feature counts K"),
                    (RIGHT - half, "Across feature count K, at five sample sizes N")):
    fig.text(left + half / 2, 0.946, label, ha="center", fontsize=17, fontweight="bold", color="#203b4c")

ticks = [0, 5, 10, 15, 19]
last = len(MODELS) - 1
stats = {}
for r, (model, title) in enumerate(MODELS):
    top = max((c for c in cells if c["model"] == model), key=lambda c: c["max_r2"])
    below = 0
    for col, k in enumerate(K_COLUMNS):
        ax = fig.add_subplot(by_n[r, col])
        below += bars(ax, [by_key[(model, k, n)] for n in n_values],
                      n_values.index(top["N"]) if top["K"] == k else None)
        ax.set_xticks(ticks, [str(n_values[i]) for i in ticks] if r == last else [])
        if r == 0:
            ax.set_title(f"K = {k:,}", fontsize=14, fontweight="bold", color="#203b4c", pad=10)
        if r == last:
            ax.set_xlabel("Sample size, $N$", fontsize=10)
        if col == 0:
            ax.set_ylabel(meta["metric"][0].upper() + meta["metric"][1:], fontsize=9.5)
            ax.text(-0.34, 0.68, title, transform=ax.transAxes, ha="right", va="center", fontsize=13,
                    fontweight="bold", color="#203b4c")
            shown = "ringed" if top["K"] in K_COLUMNS or top["N"] in N_COLUMNS else "not shown"
            ax.text(-0.34, 0.2, f"Highest bar top ({shown})\nN = {top['N']:,}, K = {top['K']:,}\n"
                    f"max = {fmt_r2(top['max_r2'])}", transform=ax.transAxes, ha="right", va="center",
                    fontsize=8.5, color="#39464d", linespacing=1.4)
        else:
            ax.set_yticklabels([])
    for col, n in enumerate(N_COLUMNS):
        ax = fig.add_subplot(by_k[r, col])
        below += bars(ax, [by_key[(model, k, n)] for k in k_values],
                      k_values.index(top["K"]) if top["N"] == n else None)
        ax.set_xticks(ticks, [f"{k_values[i]:,}" for i in ticks] if r == last else [])
        if r == 0:
            ax.set_title(f"N = {n:,}", fontsize=14, fontweight="bold", color="#203b4c", pad=10)
        if r == last:
            ax.set_xlabel("Feature count, $K$", fontsize=10)
        if col:
            ax.set_yticklabels([])
    stats[model] = {
        "p05_below_display_floor_in_shown_panels": below,
        "highest_bar_top": {key: top[key] for key in ("N", "K", "r2_test", "p05_r2", "max_r2")},
    }

# Key: one annotated bar.
kx = fig.add_axes([0.875, 0.76, 0.02, 0.12])
kx.bar(0, 0.55, bottom=-0.35, width=0.6, color=cmap(norm(0.05)), edgecolor="white")
kx.scatter([0], [0.05], s=30, color=INK, zorder=3)
for y, label in ((0.2, "best repeat (max)"), (0.05, "mean"), (-0.35, "5th percentile")):
    kx.annotate(label, (0.32, y), xytext=(0.9, y), va="center", fontsize=9.5, color="#39464d",
                annotation_clip=False, arrowprops={"arrowstyle": "-", "color": "#77818a", "linewidth": 0.8})
kx.set_xlim(-0.5, 0.5)
kx.set_ylim(-0.4, 0.25)
kx.axis("off")
fig.text(0.875, 0.915, "How to read a bar", fontsize=14, fontweight="bold", color="#203b4c", va="top")
fig.text(
    0.875,
    0.74,
    f"One bar per (N, K) cell, over\n{meta['repeats'].replace(chr(10), ' ')}.\n"
    f"Line and dots: mean {meta['metric']}.\n"
    "Bar colour = mean.\n\n"
    + meta["formula"] + "\n"
    + meta["null_text"] + "\n"
    "Grey band: worse than that baseline.\n\n"
    "Display floor: −0.50\n"
    "▼ P5 below −0.50; the bar is cut\nat the floor.\n\n"
    "Left: 5 of 20 feature counts, N on\nthe x axis. Right: 5 of 20 sample\nsizes, K on the x axis. Both are\n"
    "the tick values of the 3D figure.\nN and K use logarithmic spacing.",
    fontsize=10,
    linespacing=1.6,
    va="top",
    color="#39464d",
)
cax = fig.add_axes([0.875, 0.33, 0.09, 0.008])
colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal",
                        ticks=[FLOOR, 0, COLOR_TOP])
colorbar.ax.set_xticklabels(["≤−0.50", "0", f"≥{COLOR_TOP:.2f}"])
colorbar.ax.tick_params(labelsize=8.5, length=2)
colorbar.set_label(f"Bar colour: mean {meta['metric']}", fontsize=9, labelpad=6)
colorbar.outline.set_linewidth(0.45)
fig.text(0.02, 0.012, meta["footer"], fontsize=9, color="#74808a")

stem = meta["prefix"] + "-2d-small-multiples" + meta["suffix"]
fig.savefig(OUT / f"{stem}.png", dpi=160)
fig.savefig(OUT / f"{stem}.pdf")
plt.close(fig)
(OUT / f"plot-audit-{stem}.json").write_text(json.dumps(dict(**meta["audit"],
                                                        k_columns=K_COLUMNS, n_columns=N_COLUMNS, display_floor=FLOOR, y_top=TOP,
                                                        models=stats), indent=2), encoding="utf-8")
print(OUT / f"{stem}.png")
