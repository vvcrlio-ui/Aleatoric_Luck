"""Domain contributions from the fit.py results, averaged over seeds.

For domain d and each subset S of the other D-1 domains, the gain is v(S+d) - v(S).
"paper" averages the 2^(D-1) gains with equal weight (Yan 2026, AJE 195:681);
"shapley" weights them by |S|!(D-1-|S|)!/D!. "alone" is v({d}) - v(empty) and
"last" is v(all domains) - v(all domains but d). Standard errors are across seeds.
"""

import argparse
from math import factorial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({"font.family": "DejaVu Sans", "savefig.facecolor": "white"})
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from fit import ALL_SOURCES, BINARY, DOMAINS

FFCWS = Path(__file__).resolve().parents[1]
EVERY_DOMAIN = (1 << len(DOMAINS)) - 1
ESTIMATORS = ("paper", "shapley", "alone", "last")
LABELS = {"demographics": "Demographics", "economy_employment": "Economy & employment",
          "housing_neighborhood": "Housing & neighborhood", "family_social_ties": "Family & social ties",
          "parenting_childcare": "Parenting & childcare", "child_development_education": "Child development & education",
          "health": "Health", "attitudes_expectations": "Attitudes & expectations", "systems": "Systems",
          "parent_education_traits": "Parent education & traits"}
METRIC_LABELS = {"r2": "R²", "pr_auc": "PR-AUC", "brier_r2": "Brier R²"}
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
# Palette and type of the project's range-bar figures (runs/mh-range-bars-20261010/plot_range_bars_2d.py).
NAVY, TEXT, SUBTITLE, FOOT, AXIS, GRID = "#203b4c", "#39464d", "#5c6971", "#74808a", "#77818a", "#c9ced3"
FILL, NEGATIVE = "#4a6fae", "#9b4f58"
CMAP = LinearSegmentedColormap.from_list(
    "diverging", list(zip([0, 0.2, 0.4, 0.5, 0.6, 0.8, 1],
                          ["#7d3c45", "#b9767d", "#e7cacc", "#ebeae8", "#b6cae7", "#6f92c8", "#2f5596"])))
NORM = TwoSlopeNorm(vmin=-0.02, vcenter=0, vmax=0.10)
OUTCOMES = ("materialHardship", "eviction", "layoff", "gpa", "grit", "jobTraining")
NAMES = {"materialHardship": "Material hardship", "eviction": "Eviction", "layoff": "Layoff", "gpa": "GPA",
         "grit": "Grit", "jobTraining": "Job training"}
GROUPS = (("Household", 0, 2), ("Child", 3, 4), ("Caregiver", 5, 5))
RADAR_ORDER = ("family_social_ties", "parenting_childcare", "child_development_education", "parent_education_traits",
               "health", "demographics", "attitudes_expectations", "systems", "housing_neighborhood",
               "economy_employment")
READ_CELL = ("Each cell is the test-score gain from adding\none domain, averaged over every subset of\n"
             "the other nine domains (Yan 2026, AJE).\n\nBold number: mean over seeds.\n"
             "± : standard error across seeds; the seeds\nshare families, so it understates the\n"
             "uncertainty.\n\nR² for continuous outcomes, PR-AUC for\nbinary outcomes: compare cells within a\n"
             "column, not across columns.\n\nRed: adding the domain lowers the score.")
FOOTER = ("Source: FFC domain contribution, LightGBM on the tree_ordinal inputs; domains from FFCWS metadata "
          "topics (FFMetadata v20)")


def contributions(v):
    """v: values indexed by mask 0..EVERY_DOMAIN for one outcome, seed and metric."""
    others = len(DOMAINS) - 1
    out = []
    for i, domain in enumerate(DOMAINS):
        bit = 1 << i
        background = [s for s in range(EVERY_DOMAIN + 1) if not s & bit]
        gains = np.array([v[s | bit] - v[s] for s in background])
        weights = np.array([factorial(bin(s).count("1")) * factorial(others - bin(s).count("1"))
                            for s in background]) / factorial(len(DOMAINS))
        out += [(domain, "paper", gains.mean()), (domain, "shapley", gains @ weights),
                (domain, "alone", v[bit] - v[0]), (domain, "last", v[EVERY_DOMAIN] - v[EVERY_DOMAIN & ~bit])]
    return out


def summarize(results):
    rows, ends = [], []
    for (outcome, seed), group in results.groupby(["outcome", "seed"]):
        values = group.set_index("mask")
        for metric in ("pr_auc", "brier_r2") if outcome in BINARY else ("r2",):
            v = values[metric]
            rows += [(outcome, seed, metric, d, e, x) for d, e, x in contributions(v)]
            ends += [(outcome, seed, metric, "empty", v[0]), (outcome, seed, metric, "all_domains", v[EVERY_DOMAIN]),
                     (outcome, seed, metric, "all_sources", v[ALL_SOURCES])]
    stats = lambda frame, keys: frame.groupby(keys).value.agg(
        mean="mean", se=lambda x: x.std(ddof=1) / np.sqrt(len(x)), seeds="size").reset_index()
    summary = stats(pd.DataFrame(rows, columns=["outcome", "seed", "metric", "domain", "estimator", "value"]),
                    ["outcome", "metric", "domain", "estimator"])
    endpoints = stats(pd.DataFrame(ends, columns=["outcome", "seed", "metric", "model", "value"]),
                      ["outcome", "metric", "model"])
    return summary, endpoints


def bar_pair(ax, frame, first, second, names):
    order = frame[frame.estimator == first].sort_values("mean").domain
    y = np.arange(len(order))
    for offset, estimator, color, name in ((0.2, first, BLUE, names[0]), (-0.2, second, ORANGE, names[1])):
        part = frame[frame.estimator == estimator].set_index("domain").loc[order]
        ax.barh(y + offset, part["mean"], height=0.36, color=color, label=name,
                xerr=part.se.fillna(0), error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 2})
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_yticks(y, [LABELS[d] for d in order], color=INK)
    ax.grid(axis="x", color="#e4e3df", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def figures(summary, out):
    out.mkdir(parents=True, exist_ok=True)
    primary = summary[summary.metric.isin(["r2", "pr_auc"])]
    for (outcome, metric), frame in summary.groupby(["outcome", "metric"]):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)
        bar_pair(axes[0], frame, "paper", "shapley", ("Equal-weight mean (paper)", "Shapley"))
        bar_pair(axes[1], frame, "alone", "last", ("Alone: v({d}) − v(∅)", "Last: v(G) − v(G∖d)"))
        seeds = int(frame.seeds.max())
        for ax, title in zip(axes, ("Average gain over subsets", "Endpoints")):
            ax.set_title(title, loc="left", color=INK)
            ax.set_xlabel(f"Gain in test {METRIC_LABELS[metric]}", color=MUTED)
            ax.legend(frameon=False, loc="lower right")
        fig.suptitle(f"{outcome}: domain contribution to {METRIC_LABELS[metric]} (LightGBM, {seeds} seed"
                     f"{'s' if seeds > 1 else ''}; bars ±1 SE)", x=0.01, ha="left", color=INK)
        fig.tight_layout()
        fig.savefig(out / f"{outcome}_{metric}.png", dpi=150)
        plt.close(fig)

    heatmap(primary[primary.estimator == "paper"], out)
    radars(primary[primary.estimator == "paper"], out)


def heatmap(paper, out):
    mean = paper.pivot(index="domain", columns="outcome", values="mean").loc[list(RADAR_ORDER), list(OUTCOMES)]
    se = paper.pivot(index="domain", columns="outcome", values="se").loc[list(RADAR_ORDER), list(OUTCOMES)]
    seeds = int(paper.seeds.max())
    fig = plt.figure(figsize=(16, 9.6), facecolor="white")
    ax = fig.add_axes([0.2, 0.1, 0.52, 0.66])
    ax.imshow(mean.to_numpy(), cmap=CMAP, norm=NORM, aspect="auto")
    for (r, c), value in np.ndenumerate(mean.to_numpy()):
        ink = "white" if abs(NORM(value) - 0.5) > 0.32 else INK
        ax.text(c, r - 0.1, f"{round(value, 3) + 0:.3f}", ha="center", va="center", fontsize=11.5,
                fontweight="bold", color=ink)
        ax.text(c, r + 0.24, f"± {se.iat[r, c]:.3f}", ha="center", va="center", fontsize=8, color=ink)
    ax.set_xticks(np.arange(len(OUTCOMES) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(RADAR_ORDER) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=3)
    ax.tick_params(which="both", length=0)
    ax.set_xticks(range(len(OUTCOMES)), [f"{NAMES[o]}\n{'PR-AUC' if o in BINARY else 'R²'}" for o in OUTCOMES],
                  fontsize=11, color=TEXT)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(RADAR_ORDER)), [LABELS[d] for d in RADAR_ORDER], fontsize=11.5, fontweight="bold",
                  color=NAVY)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.canvas.draw()
    to_figure = fig.transFigure.inverted()
    for label, first, last in GROUPS:
        left = to_figure.transform(ax.transData.transform((first - 0.45, 0)))[0]
        right = to_figure.transform(ax.transData.transform((last + 0.45, 0)))[0]
        fig.text((left + right) / 2, 0.84, label, ha="center", fontsize=13, fontweight="bold", color=NAVY)
        fig.add_artist(plt.Line2D([left, right], [0.832, 0.832], color=AXIS, linewidth=0.8))
    fig.text(0.02, 0.975, "Domain contribution to predicting FFC outcomes", fontsize=22, fontweight="bold",
             color=NAVY, va="top")
    fig.text(0.02, 0.928, "Mean gain in test score from adding a domain, over all 512 subsets of the other nine "
             f"domains · LightGBM · {seeds} seeds × random 70/30 splits", fontsize=11.5, color=SUBTITLE, va="top")
    fig.text(0.76, 0.76, "How to read a cell", fontsize=13, fontweight="bold", color=NAVY, va="top")
    fig.text(0.76, 0.72, READ_CELL, fontsize=9.5, color=TEXT, va="top", linespacing=1.5)
    colorbar_axis(fig, [0.76, 0.17, 0.17, 0.022], "Cell colour: mean gain")
    fig.text(0.02, 0.02, FOOTER, fontsize=9, color=FOOT)
    fig.savefig(out / "heatmap_paper.png", dpi=200)
    plt.close(fig)


def radars(paper, out):
    angles = np.pi / 2 - np.arange(len(RADAR_ORDER)) * 2 * np.pi / len(RADAR_ORDER)
    closed = np.append(angles, angles[0])
    seeds = int(paper.seeds.max())
    fig = plt.figure(figsize=(24, 17), facecolor="white")
    width, height = 0.16, 0.16 * 24 / 17
    for k, outcome in enumerate(OUTCOMES):
        values = paper[paper.outcome == outcome].set_index("domain")["mean"].loc[list(RADAR_ORDER)].to_numpy()
        centre, bottom = (0.18, 0.5, 0.82)[k % 3], (0.55, 0.12)[k // 3]
        ax = fig.add_axes([centre - width / 2, bottom, width, height], polar=True)
        rings = MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]).tick_values(0, values.max())
        rings = rings[rings > 0]
        outer = rings[-1]
        shown = np.clip(values, 0, None)
        circle = np.linspace(0, 2 * np.pi, 200)
        ax.fill(closed, np.append(shown, shown[0]), color=FILL, alpha=0.85, zorder=2)
        ax.plot(closed, np.append(shown, shown[0]), color=INK, linewidth=1.3, zorder=3)
        ax.scatter(angles, shown, s=18, color=INK, zorder=4)
        for ring in rings:
            ax.plot(circle, np.full(200, ring), color=GRID, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
            ax.text(np.deg2rad(72), ring, f"{ring:g}", fontsize=8.5, color=FOOT, ha="left", va="bottom", zorder=5)
        for angle in angles:
            ax.plot([angle, angle], [0, outer], color=GRID, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
        ax.set_ylim(0, outer)
        ax.grid(False)
        ax.spines["polar"].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for angle, domain, value in zip(angles, RADAR_ORDER, values):
            x = np.cos(angle)
            ha = "center" if abs(x) < 0.3 else ("left" if x > 0 else "right")
            shown = round(value, 3) + 0
            ax.text(angle, outer * 1.2, f"{LABELS[domain].replace(' & ', ' &' + chr(10))}\n{shown:.3f}",
                    ha=ha, va="center", fontsize=11, linespacing=1.25,
                    color=NEGATIVE if shown < 0 else TEXT)
        top = bottom + height + 0.075
        fig.text(centre, top, NAMES[outcome], ha="center", fontsize=19, fontweight="bold", color=NAVY)
        fig.text(centre, top - 0.02, f"gain in test {'PR-AUC' if outcome in BINARY else 'R²'}", ha="center",
                 fontsize=11, color=SUBTITLE)
    fig.text(0.02, 0.975, "Domain contribution by outcome", fontsize=26, fontweight="bold", color=NAVY, va="top")
    fig.text(0.02, 0.945, "Mean gain from adding a domain, over all 512 subsets of the other nine domains · "
             f"LightGBM · {seeds} seeds · each panel has its own radial scale · negative means (lowest "
             f"{paper['mean'].min():.3f}) are drawn at the centre and labelled in red", fontsize=13, color=SUBTITLE, va="top")
    fig.text(0.02, 0.012, FOOTER, fontsize=10, color=FOOT)
    fig.savefig(out / "radar_paper.png", dpi=200)
    plt.close(fig)


def colorbar_axis(fig, box, label):
    cax = fig.add_axes(box)
    bar = fig.colorbar(ScalarMappable(norm=NORM, cmap=CMAP), cax=cax, orientation="horizontal",
                       ticks=[NORM.vmin, 0, 0.05, NORM.vmax])
    bar.ax.set_xticklabels([f"≤−{-NORM.vmin:.2f}", "0", "0.05", f"≥{NORM.vmax:.2f}"])
    bar.ax.tick_params(labelsize=8.5, length=2)
    bar.set_label(label, fontsize=9, labelpad=6)
    bar.outline.set_linewidth(0.45)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=FFCWS / "outputs" / "domain_contribution")
    args = parser.parse_args()
    results = pd.concat(pd.read_csv(path) for path in sorted(args.results.glob("*/seed*.csv")))
    summary, endpoints = summarize(results)
    summary.to_csv(args.results / "summary.csv", index=False)
    endpoints.to_csv(args.results / "endpoints.csv", index=False)
    figures(summary, args.results / "figures")
    print(endpoints.to_string(index=False))
    print(summary[summary.estimator == "paper"].sort_values(["outcome", "metric", "mean"], ascending=[True, True, False])
          .to_string(index=False))


if __name__ == "__main__":
    main()
