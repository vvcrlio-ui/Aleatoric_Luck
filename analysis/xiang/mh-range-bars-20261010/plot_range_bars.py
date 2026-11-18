"""Plot a run's test score as one P5-to-max bar per (N, K) cell and model, with the mean as a dot."""

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
from matplotlib.colors import LightSource, LinearSegmentedColormap, TwoSlopeNorm
from mpl_toolkits.mplot3d import proj3d
import numpy as np

import range_cells

MODELS = [
    ("ols", "OLS"),
    ("ridge", "Ridge"),
    ("lasso", "Lasso"),
    ("random_forest", "Random forest"),
    ("shallow_neural_network", "Shallow neural network"),
    ("extra_trees", "Extra trees"),
    ("xgboost", "XGBoost"),
    ("lightgbm", "LightGBM"),
    ("super_learner", "Super learner"),
]
# Usage: plot_range_bars.py {sampled,full,AGGREGATE.json} -- see range_cells.py.
cells, meta = range_cells.load(sys.argv[1])
OUT = meta["out"]
assert {c["model"] for c in cells} == {name for name, _ in MODELS}

FLOOR, TOP, COLOR_TOP = -0.50, meta["top"], meta["color_top"]
# Upper ticks only where they clear the tick below by 0.1.
Z_TICKS = [FLOOR, 0]
for _tick in (COLOR_TOP, TOP):
    if _tick - Z_TICKS[-1] >= 0.1:
        Z_TICKS.append(_tick)


def tick_label(value):
    return "0" if value == 0 else f"{value:.2f}".replace("-", "−")


assert max(c["max_r2"] for c in cells) <= TOP
n_values = sorted({c["N"] for c in cells})
k_values = sorted({c["K"] for c in cells})
assert len(n_values) == len(k_values) == 20
n_max, k_max = n_values[-1], k_values[-1]
x_max, y_max = np.log10(n_max), np.log10(k_max)
# Bars take 45% of the log spacing between neighbouring grid values.
dx = 0.45 * (np.log10(n_values[-1]) - np.log10(n_values[0])) / 19
dy = 0.45 * (np.log10(k_values[-1]) - np.log10(k_values[0])) / 19
norm = TwoSlopeNorm(vmin=FLOOR, vcenter=0, vmax=COLOR_TOP)
# Pink #E7CACC and blue #B6CAE7, deepened towards both ends, around a light grey midpoint at R-squared = 0.
cmap = LinearSegmentedColormap.from_list(
    "pink_grey_blue",
    list(zip([0, 0.2, 0.4, 0.5, 0.6, 0.8, 1], ["#7d3c45", "#b9767d", "#e7cacc", "#ebeae8", "#b6cae7", "#6f92c8", "#2f5596"])),
)
INK = "#222222"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "savefig.facecolor": "white",
    }
)
stats: dict[str, dict[str, float | int]] = {}


def fmt_r2(value):
    return f"{value:.3f}".replace("-", "−")


def bars(ax, name: str, title: str) -> None:
    ax.computed_zorder = False
    # Far cells first for the azim=-135 view.
    model_cells = sorted((c for c in cells if c["model"] == name), key=lambda c: (-c["N"], c["K"]))
    low = np.array([max(c["p05_r2"], FLOOR) for c in model_cells])
    high = np.array([max(c["max_r2"], FLOOR) for c in model_cells])
    mean = np.array([c["r2_test"] for c in model_cells])
    x = np.log10([c["N"] for c in model_cells])
    y = np.log10([c["K"] for c in model_cells])
    ax.bar3d(
        x - dx / 2,
        y - dy / 2,
        low,
        dx,
        dy,
        np.maximum(high - low, 1e-4),
        color=cmap(norm(np.maximum(mean, FLOOR))),
        shade=True,
        lightsource=LightSource(azdeg=300, altdeg=55),
        edgecolor=(1, 1, 1, 0.35),
        linewidth=0.2,
    )
    ax.scatter(x, y, np.maximum(mean, FLOOR), s=4, color=INK, depthshade=False)

    ax.set_xlim(np.log10(n_values[0]) - dx, x_max + dx)
    ax.set_ylim(y_max + dy, np.log10(k_values[0]) - dy)
    ax.set_zlim(FLOOR, TOP)
    ax.view_init(elev=25, azim=-135)
    ax.set_box_aspect((1, 1, 0.82))
    n_ticks = [n_values[i] for i in (0, 5, 10, 15, 19)]
    k_ticks = [k_values[i] for i in (0, 5, 10, 15, 19)]
    ax.set_xticks(np.log10(n_ticks), [f"  {n_ticks[0]}", *map(str, n_ticks[1:])])
    ax.set_yticks(np.log10(k_ticks), [*map(str, k_ticks[:-1]), f"{k_ticks[-1]}  "])
    ax.get_xticklabels()[0].set_horizontalalignment("left")
    ax.get_yticklabels()[-1].set_horizontalalignment("right")
    ax.set_zticks(Z_TICKS, [tick_label(t) for t in Z_TICKS])
    ax.set_xlabel("Sample size, $N$", labelpad=3)
    ax.set_ylabel("Feature count, $K$", labelpad=3)
    ax.set_zlabel(meta["metric"][0].upper() + meta["metric"][1:], labelpad=2)
    ax.tick_params(labelsize=8, pad=1)
    ax.set_title(title, pad=1)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("#e6e9ed")
        axis._axinfo["grid"].update(color="#e3e6ea", linewidth=0.65)
        axis.line.set_color("#77818a")

    p05_below = sum(c["p05_r2"] < FLOOR for c in model_cells)
    if p05_below:
        noun = "cell" if p05_below == 1 else "cells"
        ax.text2D(0.015, 0.97, f"{p05_below} {noun}: P5 < −0.50", transform=ax.transAxes,
                  color="#9b4f58", fontsize=8)

    top = max(model_cells, key=lambda c: c["max_r2"])
    top_xyz = (np.log10(top["N"]), np.log10(top["K"]), top["max_r2"])
    ax.scatter(*top_xyz, s=34, facecolors="white", edgecolors=INK, linewidths=1.3, depthshade=False)
    x_projected, y_projected, _ = proj3d.proj_transform(*top_xyz, ax.get_proj())
    ax.annotate(
        f"Highest bar top: N = {top['N']:,}, K = {top['K']:,}\n"
        f"max = {fmt_r2(top['max_r2'])} · P5 = {fmt_r2(top['p05_r2'])}\n"
        f"$\\overline{{R^2}}$ = {fmt_r2(top['r2_test'])}",
        xy=(x_projected, y_projected),
        xycoords="data",
        xytext=(0.62, 0.9),
        textcoords="axes fraction",
        ha="left",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        color=INK,
        bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": "#d4dce2", "alpha": 0.94},
        arrowprops={"arrowstyle": "-", "color": INK, "linewidth": 1.15, "shrinkA": 1, "shrinkB": 3,
                    "connectionstyle": "arc3,rad=0.05"},
        annotation_clip=False,
        zorder=20,
    )
    stats[name] = {
        "cells": len(model_cells),
        "p05_below_display_floor": p05_below,
        "minimum_p05_r2": float(min(c["p05_r2"] for c in model_cells)),
        "maximum_max_r2": float(max(c["max_r2"] for c in model_cells)),
        "highest_bar_top": {k: top[k] for k in ("N", "K", "r2_test", "p05_r2", "max_r2")},
    }


fig = plt.figure(figsize=(22, 16.4), facecolor="white")
fig.text(0.04, 0.974, "Spread of prediction performance across sample size and feature count",
         fontsize=24, fontweight="bold", color="#203b4c", va="top")
fig.text(0.04, 0.946, meta["subtitle"],
         fontsize=12, color="#5c6971", va="top")
for i, (name, title) in enumerate(MODELS):
    row, col = divmod(i, 3)
    ax = fig.add_axes([0.012 + col * 0.263, 0.635 - row * 0.303, 0.266, 0.286], projection="3d")
    bars(ax, name, title)

fig.text(0.825, 0.82, "How to read the bars", fontsize=14, fontweight="bold", color="#203b4c")
note = (
    f"One bar per (N, K) cell, over {meta['repeats']}.\n"
    f"Bar bottom: 5th percentile of {meta['metric']}.\n"
    "Bar top: best repeat (maximum).\n"
    f"Dot: mean {meta['metric']}; bar colour = mean.\n"
    "Ring: the highest bar top in the grid.\n\n"
    + meta["formula"] + "\n"
    + meta["null_text"] + "\n"
    "Below 0: worse than that baseline.\n\n"
    "Display floor: −0.50\n"
    "Bars with P5 below −0.50 are cut at the\n"
    "floor; red notes count those cells.\n\n"
    "N and K axes use logarithmic spacing.\n"
    "K counts original feature groups.\n"
    "No smoothing; clip only after aggregating."
)
fig.text(0.825, 0.791, note, fontsize=10.3, linespacing=1.65, va="top", color="#39464d")

cax = fig.add_axes([0.825, 0.405, 0.145, 0.014])
colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal",
                        ticks=[FLOOR, 0, COLOR_TOP])
colorbar.ax.set_xticklabels(["≤−0.50", "0", f"≥{COLOR_TOP:.2f}"])
colorbar.ax.tick_params(labelsize=9, length=2)
colorbar.set_label(f"Bar colour: mean {meta['metric']}", fontsize=9, labelpad=8)
colorbar.outline.set_linewidth(0.45)

fig.text(0.825, 0.34, "Verified production run", fontsize=13, fontweight="bold", color="#203b4c")
fig.text(
    0.825,
    0.317,
    f"{meta['run_lines'][0]}\n"
    + meta["design_line"] + "\n"
    "Super learner: seven base models, excluding OLS\n\n"
    + "\n".join(meta["run_lines"][1:-1]) + "\n"
    f"Maximum N = {n_max:,} · K = {k_max:,}\n"
    + meta["run_lines"][-1],
    fontsize=10.3,
    linespacing=1.7,
    va="top",
    color="#39464d",
)
fig.text(0.04, 0.013, meta["footer"], fontsize=9, color="#74808a")

stem = meta["prefix"] + "-nine-models" + meta["suffix"]
fig.savefig(OUT / f"{stem}.png", dpi=180)
fig.savefig(OUT / f"{stem}.pdf")
plt.close(fig)

with (OUT / meta["csv_name"]).open("w", encoding="utf-8", newline="") as fh:
    fh.write("model,N,K,count,mean_r2,p05_r2,min_r2,max_r2\n")
    for c in cells:
        fh.write(f"{c['model']},{c['N']},{c['K']},{c['count']},{c['r2_test']!r},{c['p05_r2']!r},"
                 f"{c['min_r2']!r},{c['max_r2']!r}\n")
(OUT / f"plot-audit-{stem}.json").write_text(json.dumps(dict(**meta["audit"],
                                                     display_floor=FLOOR, z_top=TOP, models=stats), indent=2),
                                     encoding="utf-8")
print(json.dumps(stats, indent=2))
print(OUT / f"{stem}.png")
