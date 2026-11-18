"""Plot FFC tree-ordinal material hardship mean R-squared across all 100 seeds and 50 draws in the reference style."""

from __future__ import annotations

import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "visualizations"
OUT.mkdir(exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(HERE / ".mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d import proj3d
import numpy as np


MODELS = [
    ("ols", "OLS"),
    ("ridge", "Ridge"),
    ("lasso", "Lasso"),
    ("random_forest", "Random forest"),
    ("shallow_neural_network", "Shallow neural network"),
    ("extra_trees", "Extra trees"),
    ("super_learner", "Super learner (SL7)"),
    ("xgboost", "XGBoost"),
    ("lightgbm", "LightGBM"),
]
SOURCE = HERE / "aggregate.json"
agg = json.loads(SOURCE.read_text(encoding="utf-8-sig"))
cells = agg["cells"]
assert len(cells) == 3600
min_count = min(c["count"] for c in cells)
skipped = agg["status_counts"].get("skipped", 0)
assert sum(c["count"] for c in cells) + skipped == 18000000
assert set(agg["skip_reasons"]) <= {"all_selected_sources_unobserved"}
assert agg["rows"] == agg["receipt"]["rows"] == 18000000
assert agg["metadata"]["dataset"] == ["ffc_tree_ordinal_materialHardship"]
assert agg["metadata"]["outcome"] == ["materialHardship"]
assert agg["metadata"]["split_mode"] == ["external_test"]
assert agg["receipt"]["complete"]

FLOOR = -0.50
all_r2 = np.array([c["r2_test"] for c in cells])
TOP = max(0.20, float(np.ceil(all_r2.max() * 20) / 20))
TOP_TICK = TOP
ALGORITHM_VERSION = agg["metadata"]["algorithm_version"][0]
test_rows = int(agg["metadata"]["n_test_total"][0])


def fmt_r2(value):
    return f"{value:.3f}".replace("-", "−")


n_values = sorted({c["N"] for c in cells})
k_values = sorted({c["K"] for c in cells})
assert len(n_values) == len(k_values) == 20
assert {c["model"] for c in cells} == {name for name, _ in MODELS}
n_index = {n: i for i, n in enumerate(n_values)}
k_index = {k: i for i, k in enumerate(k_values)}
grids: dict[str, np.ndarray] = {name: np.full((20, 20), np.nan) for name, _ in MODELS}
for c in cells:
    grids[c["model"]][k_index[c["K"]], n_index[c["N"]]] = c["r2_test"]

x_grid, y_grid = np.meshgrid(np.log10(n_values), np.log10(k_values))
n_max, k_max = int(max(n_values)), int(max(k_values))
x_max, y_max = np.log10(n_max), np.log10(k_max)
norm = TwoSlopeNorm(vmin=FLOOR, vcenter=0, vmax=TOP)
cmap = plt.get_cmap("RdYlBu")
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


def surface(ax, name: str, title: str) -> None:
    z = grids[name]
    assert z.shape == (20, 20) and np.isfinite(z).all()
    clipped = np.maximum(z, FLOOR)
    ax.plot_surface(
        x_grid,
        y_grid,
        clipped,
        cmap=cmap,
        norm=norm,
        rcount=20,
        ccount=20,
        linewidth=0.16,
        edgecolor=(1, 1, 1, 0.23),
        antialiased=True,
        shade=False,
    )
    ax.set_xlim(np.log10(n_values[0]), np.log10(n_values[-1]))
    ax.set_ylim(np.log10(k_values[-1]), np.log10(k_values[0]))
    ax.set_zlim(FLOOR, TOP)
    ax.view_init(elev=25, azim=-135)
    ax.set_box_aspect((1, 1, 0.82))

    n_ticks = [n_values[i] for i in (0, 5, 10, 15, 19)]
    k_ticks = [k_values[i] for i in (0, 5, 10, 15, 19)]
    ax.set_xticks(np.log10(n_ticks), [f"  {n_ticks[0]}", *map(str, n_ticks[1:])])
    ax.set_yticks(np.log10(k_ticks), [*map(str, k_ticks[:-1]), f"{k_ticks[-1]}  "])
    ax.get_xticklabels()[0].set_horizontalalignment("left")
    ax.get_yticklabels()[-1].set_horizontalalignment("right")
    ax.set_zticks([FLOOR, 0, TOP_TICK], ["−0.50", "0", f"{TOP_TICK:.2f}"])
    ax.set_xlabel("Sample size, $N$", labelpad=3)
    ax.set_ylabel("Feature count, $K$", labelpad=3)
    ax.set_zlabel("Mean test $R^2$", labelpad=2)
    ax.tick_params(labelsize=8, pad=1)
    ax.set_title(title, pad=1)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("#e6e9ed")
        axis._axinfo["grid"].update(color="#e3e6ea", linewidth=0.65)
        axis.line.set_color("#77818a")

    below_floor = int((z < FLOOR).sum())
    if below_floor:
        noun = "cell" if below_floor == 1 else "cells"
        ax.text2D(
            0.015,
            0.97,
            f"{below_floor} {noun} < −0.50",
            transform=ax.transAxes,
            color="#a9443d",
            fontsize=8,
        )

    endpoint_raw = float(z[-1, -1])
    endpoint_display = max(endpoint_raw, FLOOR)
    endpoint_color = "#173b59"
    ax.scatter(
        [x_max],
        [y_max],
        [endpoint_display],
        s=32,
        facecolors="white",
        edgecolors=endpoint_color,
        linewidths=1.4,
        depthshade=False,
        zorder=10,
    )
    ax.plot(
        [x_max, x_max],
        [y_max, y_max],
        [FLOOR, endpoint_display],
        color="#a17b34",
        linewidth=1.5,
        alpha=0.85,
        zorder=9,
    )
    x_projected, y_projected, _ = proj3d.proj_transform(
        x_max, y_max, endpoint_display, ax.get_proj()
    )
    clipped_note = "\n(displayed at floor)" if endpoint_raw < FLOOR else ""
    ax.annotate(
        f"$(N_{{max}}, K_{{max}})$\n$\\overline{{R^2}}$ = {fmt_r2(endpoint_raw)}{clipped_note}",
        xy=(x_projected, y_projected),
        xycoords="data",
        xytext=(0.66, 0.88),
        textcoords="axes fraction",
        ha="left",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        color=endpoint_color,
        bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": "#d4dce2", "alpha": 0.94},
        arrowprops={
            "arrowstyle": "-",
            "color": endpoint_color,
            "linewidth": 1.15,
            "shrinkA": 1,
            "shrinkB": 3,
            "connectionstyle": "arc3,rad=0.05",
        },
        annotation_clip=False,
        zorder=20,
    )
    stats[name] = {
        "cells": 400,
        "below_display_floor": below_floor,
        "minimum_raw_r2": float(z.min()),
        "maximum_raw_r2": float(z.max()),
        "r2_at_max_N_K": endpoint_raw,
    }


fig = plt.figure(figsize=(22, 16.4), facecolor="white")
fig.text(
    0.04,
    0.974,
    "Prediction performance across sample size and feature count",
    fontsize=24,
    fontweight="bold",
    color="#203b4c",
    va="top",
)
fig.text(
    0.04,
    0.946,
    "FFC tree-ordinal material hardship · 100 seeds × 50 draws · completed 23 Sep 2026",
    fontsize=12,
    color="#5c6971",
    va="top",
)
for i, (name, title) in enumerate(MODELS):
    row, col = divmod(i, 3)
    ax = fig.add_axes(
        [0.012 + col * 0.263, 0.635 - row * 0.303, 0.266, 0.286],
        projection="3d",
    )
    surface(ax, name, title)

fig.text(0.825, 0.82, "How to read the surfaces", fontsize=14, fontweight="bold", color="#203b4c")
note = (
    "Each height is mean test $R^2$.\n"
    "100 seeds × 50 draws planned per cell\n"
    "20 sample sizes × 20 feature counts\n\n"
    "$R^2 = 1 - \\mathrm{MSE}/\\mathrm{MSE}_{null}$\n"
    "The null predicts the training mean.\n\n"
    "$R^2 = 0$: training-mean baseline\n"
    "$R^2 > 0$: better than the baseline\n"
    "$R^2 < 0$: worse than the baseline\n\n"
    "Display floor: −0.50\n"
    "Values below the floor stay in the data;\n"
    "red annotations count affected cells.\n\n"
    "N and K axes use logarithmic spacing.\n"
    "K counts original feature groups.\n"
    "Arithmetic mean across valid repeats.\nNo smoothing; clip only after averaging."
)
fig.text(0.825, 0.791, note, fontsize=10.3, linespacing=1.65, va="top", color="#39464d")

cax = fig.add_axes([0.825, 0.417, 0.145, 0.014])
colorbar = fig.colorbar(
    ScalarMappable(norm=norm, cmap=cmap),
    cax=cax,
    orientation="horizontal",
    ticks=[FLOOR, 0, TOP_TICK],
)
colorbar.ax.set_xticklabels(["≤−0.50", "0", f"{TOP_TICK:.2f}"])
colorbar.ax.tick_params(labelsize=9, length=2)
colorbar.set_label("Mean test $R^2$ · training-mean baseline", fontsize=9, labelpad=8)
colorbar.outline.set_linewidth(0.45)

fig.text(0.825, 0.34, "Verified FFC production run", fontsize=13, fontweight="bold", color="#203b4c")
fig.text(
    0.825,
    0.317,
    "100 seeds × 50 draws · completed\n"
    "Tree-ordinal encoding · held-out FFC test set\n"
    "SL7: seven base models, excluding OLS\n\n"
    f"{agg['status_counts']['ok']:,} valid results\n"
    + (f"{min_count:,}–5,000 valid repeats per cell\n"
       f"{skipped} draws skipped: selected feature unobserved\n"
       if skipped else "5,000 valid repeats per cell\n")
    + f"Maximum N = {n_max:,} · K = {k_max:,}\n"
    + f"Test rows = {test_rows:,}",
    fontsize=10.3,
    linespacing=1.7,
    va="top",
    color="#39464d",
)
fig.text(
    0.04,
    0.013,
    "Source: verified FFC tree-ordinal material hardship production results · "
    f"{ALGORITHM_VERSION} · prod-mh-6h3g-b · 23 Sep 2026",
    fontsize=9,
    color="#74808a",
)

stem = "mh-20260923-100seeds-50draws-nine-models"
fig.savefig(OUT / f"{stem}.png", dpi=180)
fig.savefig(OUT / f"{stem}.pdf")
fig.savefig(OUT / f"{stem}-preview.png", dpi=90)
plt.close(fig)

with (OUT / "ffc-tree-ordinal-materialhardship-mean-r2-3600-cells.csv").open("w", encoding="utf-8", newline="") as fh:
    fh.write("model,N,K,count,r2_test,min_r2,max_r2,repeats_below_floor\n")
    for c in cells:
        fh.write(
            f"{c['model']},{c['N']},{c['K']},{c['count']},{c['r2_test']!r},"
            f"{c['min_r2']!r},{c['max_r2']!r},{c['repeats_below_floor']}\n"
        )

audit = dict(
    source=agg["source"],
    receipt=agg["receipt"],
    metadata=agg["metadata"],
    validation=agg["validation"],
    status_counts=agg["status_counts"],
    skip_reasons=agg["skip_reasons"],
    aggregation=(
        "Arithmetic mean of valid r2_test over 100 seeds x 50 draws; explicitly skipped "
        "records excluded; no smoothing; clipping after averaging"
    ),
    display_floor=FLOOR,
    color_max=TOP,
    models=stats,
)
(OUT / "plot-audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps(stats, indent=2))
print(OUT / f"{stem}.png")
