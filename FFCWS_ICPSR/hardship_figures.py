"""Figures of test R² for FFCWS material hardship by information landmark and outcome age.

Usage: python hardship_figures.py FINALS_DIR OUT_DIR [--model super_learner]
FINALS_DIR holds one <run>/final.csv per panel (hardship_t<t>_T<T>). Each value is the
panel's model trained on every training family, averaged over draws within a seed and
then over seeds, with the standard error across seeds. Writes triangle.png,
unpredictable.png and horizon.png.
"""

import argparse
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({"font.family": "DejaVu Sans", "savefig.facecolor": "white"})
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

# Palette and type of FFCWS/domain_contribution/summarize.py (heatmap_paper.png).
INK, NAVY, TEXT, SUBTITLE, FOOT, AXIS = "#0b0b0b", "#203b4c", "#39464d", "#5c6971", "#74808a", "#77818a"
GRID, REST = "#e4e3df", "#e7e6e2"
CMAP = LinearSegmentedColormap.from_list(
    "diverging", list(zip([0, 0.2, 0.4, 0.5, 0.6, 0.8, 1],
                          ["#7d3c45", "#b9767d", "#e7cacc", "#ebeae8", "#b6cae7", "#6f92c8", "#2f5596"])))
NORM = TwoSlopeNorm(vmin=-0.05, vcenter=0, vmax=0.30)
LANDMARKS = (0, 1, 3, 5, 9, 15)
AGES = (22, 15, 9, 5, 1)
NAMES = {0: "Birth", 1: "Age 1", 3: "Age 3", 5: "Age 5", 9: "Age 9", 15: "Age 15"}
WAVE_COLORS = dict(zip(LANDMARKS, LinearSegmentedColormap.from_list(
    "waves", ["#1b2f57", "#2f5596", "#4a6fae", "#6f92c8", "#9fbbe0", "#c9d8ee"])(np.linspace(0, 1, len(LANDMARKS)))))
AGE_COLORS = {1: "#b9767d", 5: "#8fb0dd", 9: "#4a6fae", 15: "#203b4c", 22: "#9b4f58"}
MODELS = ("super_learner", "lightgbm", "ridge", "lasso", "random_forest", "extra_trees", "xgboost",
          "shallow_neural_network", "ols")


def cells(finals, model):
    frame = pd.concat(pd.read_csv(path) for path in glob.glob(str(Path(finals) / "*" / "final.csv")))
    frame = frame[(frame.model == model) & (frame.status == "ok")]
    frame[["t", "T"]] = frame.panel_id.str.extract(r"hardship_t(\d+)_T(\d+)").astype(int)
    frame = frame[frame.N == frame.groupby("panel_id").N.transform("max")]
    frame["k_kind"] = np.where(frame.K == frame.groupby("panel_id").K.transform("max"), "full", "matched")
    by_seed = frame.groupby(["t", "T", "K", "k_kind", "seed"]).r2_test.mean().reset_index()
    stats = by_seed.groupby(["t", "T", "K", "k_kind"]).r2_test.agg(
        mean="mean", se=lambda x: x.std(ddof=1) / np.sqrt(len(x))).reset_index()
    # The birth landmark has 660 variables, so its one K is both the full and the matched value.
    return pd.concat([stats, stats[stats.t == 0].assign(k_kind="matched")])


def style(ax, grid):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=TEXT, labelsize=10)
    ax.grid(axis=grid, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def title(fig, text):
    fig.text(0.02, 0.97, text, fontsize=21, fontweight="bold", color=NAVY, va="top")


def triangle_panel(ax, stats, kind):
    rows = stats[stats.k_kind == kind]
    grid = np.full((len(AGES), len(LANDMARKS)), np.nan)
    for row in rows.itertuples():
        grid[AGES.index(row.T), LANDMARKS.index(row.t)] = row.mean
    ax.imshow(np.ma.masked_invalid(grid), cmap=CMAP, norm=NORM, aspect="auto")
    for row in rows.itertuples():
        r, c = AGES.index(row.T), LANDMARKS.index(row.t)
        ink = "white" if abs(NORM(row.mean) - 0.5) > 0.32 else INK
        ax.text(c, r - 0.1, f"{round(row.mean, 3) + 0:.3f}", ha="center", va="center", fontsize=12,
                fontweight="bold", color=ink)
        ax.text(c, r + 0.24, f"± {row.se:.3f}", ha="center", va="center", fontsize=8, color=ink)
        ax.text(c + 0.44, r - 0.4, f"{row.T - row.t} yr", ha="right", va="top", fontsize=7.5,
                color=ink if ink == "white" else FOOT)
    ax.set_xticks(np.arange(len(LANDMARKS) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(AGES) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=3)
    ax.tick_params(which="both", length=0)
    ax.set_xticks(range(len(LANDMARKS)), [NAMES[t] for t in LANDMARKS], fontsize=11, color=TEXT)
    ax.set_yticks(range(len(AGES)), [f"Age {T}" for T in AGES], fontsize=11.5, fontweight="bold", color=NAVY)
    ax.set_xlabel("Information observed up to (landmark t)", fontsize=10.5, color=TEXT, labelpad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)


def triangle(stats, out):
    """Two triangles of cells: every variable observed by t, and 660 of them."""
    fig = plt.figure(figsize=(17, 8.4))
    axes = [fig.add_axes([0.075, 0.1, 0.31, 0.68]), fig.add_axes([0.445, 0.1, 0.31, 0.68])]
    full_k = stats[stats.k_kind == "full"].K
    matched_k = stats[stats.k_kind == "matched"].K.min()
    notes = (("Everything observed so far", f"all survey variables up to t ({full_k.min():,}–{full_k.max():,})"),
             ("Same number of variables", f"{matched_k:,} variables drawn from those observed up to t"))
    for ax, kind, (head, note) in zip(axes, ("full", "matched"), notes):
        triangle_panel(ax, stats, kind)
        ax.set_title(head, loc="left", fontsize=14, fontweight="bold", color=NAVY, pad=28)
        ax.text(0, 1.03, note, transform=ax.transAxes, fontsize=10, color=SUBTITLE)
    axes[0].set_ylabel("Hardship measured at (outcome age T)", fontsize=10.5, color=TEXT, labelpad=8)
    title(fig, "How predictable is material hardship from earlier childhood?")
    read = ("Each cell predicts hardship at age T\n(row) from survey information collected\nup to age t (column), "
            "for the same\nfamilies in every cell.\n\nBold number: test R², the share of the\n"
            "variance of hardship at age T that\nthe prediction explains; ± standard\nerror across seeds.\n\n"
            "Corner: years between t and T. Cells\nwith equal gaps compare the same\nforecast horizon at "
            "different ages:\n1→5 and 5→9 (4 yr); 3→9 and 9→15\n(6 yr).\n\n"
            f"Right panel: every cell uses {matched_k:,}\nvariables, so a difference reflects\nwhich information, "
            "not how much.\n\nRed: worse than predicting the mean.")
    fig.text(0.785, 0.8, "How to read a cell", fontsize=13, fontweight="bold", color=NAVY, va="top")
    fig.text(0.785, 0.76, read, fontsize=9.5, color=TEXT, va="top", linespacing=1.45)
    cax = fig.add_axes([0.785, 0.12, 0.17, 0.022])
    bar = fig.colorbar(ScalarMappable(norm=NORM, cmap=CMAP), cax=cax, orientation="horizontal",
                       ticks=[NORM.vmin, 0, 0.1, 0.2, NORM.vmax])
    bar.ax.set_xticklabels([f"≤−{-NORM.vmin:.2f}", "0", "0.10", "0.20", f"≥{NORM.vmax:.2f}"])
    bar.ax.tick_params(labelsize=8.5, length=2)
    bar.set_label("Cell colour: test R²", fontsize=9, labelpad=6)
    bar.outline.set_linewidth(0.45)
    fig.savefig(out / "triangle.png", dpi=200)
    plt.close(fig)


def unpredictable(stats, out):
    """Stacked share of variance explained wave by wave, against what remains unexplained."""
    f = stats[stats.k_kind == "full"].set_index(["T", "t"])
    fig = plt.figure(figsize=(15, 7.2))
    ax = fig.add_axes([0.1, 0.17, 0.86, 0.6])
    for y, T in enumerate(AGES):
        level = 0.0
        for t in [w for w in LANDMARKS if w < T]:
            value = f.loc[(T, t), "mean"]
            step = value - level
            ax.barh(y, step, left=level, color=WAVE_COLORS[t], height=0.62, edgecolor="white", linewidth=1.2)
            if step >= 0.025:
                ax.text(level + step / 2, y, f"{step:.2f}", ha="center", va="center", fontsize=8.5,
                        color="white" if t <= 3 else NAVY)
            level = value
        ax.barh(y, 1 - level, left=level, color=REST, height=0.62, edgecolor="white", linewidth=1.2)
        ax.text(level + 0.006, y, f"{level:.0%}", ha="left", va="center", fontsize=12, fontweight="bold",
                color=NAVY)
    ax.set_yticks(range(len(AGES)), [f"Hardship at {T}" for T in AGES], fontsize=12, fontweight="bold",
                  color=NAVY)
    ax.set_xlim(0, 0.4)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel("Share of the variance of hardship at that age (test R²); the axis stops at 40%", fontsize=10.5,
                  color=TEXT)
    style(ax, "x")
    ax.tick_params(axis="y", length=0)
    for k, t in enumerate(LANDMARKS):
        fig.patches.append(plt.Rectangle((0.1 + k * 0.105, 0.83), 0.018, 0.022, transform=fig.transFigure,
                                         color=WAVE_COLORS[t]))
        fig.text(0.122 + k * 0.105, 0.841, NAMES[t] if t == 0 else f"+ {NAMES[t]}", fontsize=10, color=TEXT,
                 va="center")
    title(fig, "Most of a family's future hardship cannot be predicted")
    fig.text(0.02, 0.915, "Each bar adds the survey waves in order up to the last one before the outcome; grey is "
             "what all of them together leave unexplained", fontsize=11, color=SUBTITLE, va="top")
    fig.savefig(out / "unpredictable.png", dpi=200)
    plt.close(fig)


def horizon(stats, out):
    """Points with 95% intervals grouped by outcome age, joined from the earliest to the latest landmark."""
    f = stats[stats.k_kind == "full"]
    fig = plt.figure(figsize=(16, 8.4))
    ax = fig.add_axes([0.06, 0.12, 0.92, 0.68])
    x, ticks, labels = 0.0, [], []
    for T in sorted(AGES):
        rows = f[f["T"] == T].sort_values("t")
        color = AGE_COLORS[T]
        xs = x + np.arange(len(rows))
        ax.axvspan(xs[0] - 0.5, xs[-1] + 0.5, color=color, alpha=0.06, linewidth=0, zorder=0)
        ax.plot(xs, rows["mean"], color=color, linewidth=2, zorder=2)
        ax.errorbar(xs, rows["mean"], yerr=1.96 * rows.se, fmt="o", color=color, ecolor=color, elinewidth=1.6,
                    capsize=3, markersize=7, zorder=3)
        ax.text(xs.mean(), 0.345, f"hardship\nat {T}", ha="center", va="bottom", fontsize=12, fontweight="bold",
                color=color, clip_on=False, linespacing=1.1)
        ticks += list(xs)
        labels += [NAMES[t] for t in rows.t]
        x += len(rows) + 0.4
    ax.set_xticks(ticks, labels, fontsize=9.5, color=TEXT)
    ax.set_xlim(-0.6, x - 0.8)
    ax.set_ylim(0, 0.34)
    ax.set_ylabel("Test R² with 95% interval across seeds", fontsize=10.5, color=TEXT)
    ax.set_xlabel("Information up to", fontsize=10.5, color=TEXT, labelpad=8)
    style(ax, "y")
    ax.tick_params(axis="x", length=0)
    title(fig, "The shorter the horizon, the better the prediction")
    fig.savefig(out / "horizon.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("finals", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--model", default="super_learner", choices=MODELS)
    args = parser.parse_args()
    stats = cells(args.finals, args.model)
    args.out.mkdir(parents=True, exist_ok=True)
    for draw in (triangle, unpredictable, horizon):
        draw(stats, args.out)


if __name__ == "__main__":
    main()
