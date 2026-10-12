"""Domain contributions from the fit.py results, averaged over seeds.

For domain d and each subset S of the other eight domains, the gain is v(S+d) - v(S).
"paper" averages the 256 gains with equal weight (Yan 2026, AJE 195:681);
"shapley" weights them by |S|!(8-|S|)!/9!. "alone" is v({d}) - v(empty) and
"last" is v(all nine) - v(all nine but d). Standard errors are across seeds.
"""

import argparse
from math import factorial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fit import BINARY, DOMAINS

FFCWS = Path(__file__).resolve().parents[1]
NINE = (1 << len(DOMAINS)) - 1
ESTIMATORS = ("paper", "shapley", "alone", "last")
LABELS = {"demographics": "Demographics", "economy_employment": "Economy & employment",
          "housing_neighborhood": "Housing & neighborhood", "family_social_ties": "Family & social ties",
          "parenting_childcare": "Parenting & childcare", "child_development_education": "Child development & education",
          "health": "Health", "attitudes_expectations": "Attitudes & expectations", "systems": "Systems"}
METRIC_LABELS = {"r2": "R²", "pr_auc": "PR-AUC", "brier_r2": "Brier R²"}
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"


def contributions(v):
    """v: values indexed by mask 0..NINE for one outcome, seed and metric."""
    others = len(DOMAINS) - 1
    out = []
    for i, domain in enumerate(DOMAINS):
        bit = 1 << i
        background = [s for s in range(NINE + 1) if not s & bit]
        gains = np.array([v[s | bit] - v[s] for s in background])
        weights = np.array([factorial(bin(s).count("1")) * factorial(others - bin(s).count("1"))
                            for s in background]) / factorial(len(DOMAINS))
        out += [(domain, "paper", gains.mean()), (domain, "shapley", gains @ weights),
                (domain, "alone", v[bit] - v[0]), (domain, "last", v[NINE] - v[NINE & ~bit])]
    return out


def summarize(results):
    rows, ends = [], []
    for (outcome, seed), group in results.groupby(["outcome", "seed"]):
        values = group.set_index("mask")
        for metric in ("pr_auc", "brier_r2") if outcome in BINARY else ("r2",):
            v = values[metric]
            rows += [(outcome, seed, metric, d, e, x) for d, e, x in contributions(v)]
            ends += [(outcome, seed, metric, "empty", v[0]), (outcome, seed, metric, "nine_domains", v[NINE]),
                     (outcome, seed, metric, "all_sources", v[1023])]
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

    table = primary[primary.estimator == "paper"].pivot(index="domain", columns="outcome", values="mean")
    table = table.loc[[d for d in DOMAINS if d in table.index]]
    limit = np.nanmax(np.abs(table.to_numpy()))
    fig, ax = plt.subplots(figsize=(1.6 * table.shape[1] + 5, 5.5))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("diverging", ["#d03b3b", "#f0efec", BLUE])
    image = ax.imshow(table.to_numpy(), cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
    for (r, c), value in np.ndenumerate(table.to_numpy()):
        ax.text(c, r, f"{round(value, 3) + 0:.3f}", ha="center", va="center", color=INK, fontsize=9)
    ax.set_xticks(range(table.shape[1]), [f"{o}\n({'PR-AUC' if o in BINARY else 'R²'})" for o in table.columns])
    ax.set_yticks(range(table.shape[0]), [LABELS[d] for d in table.index])
    ax.set_title("Mean gain over subsets (paper estimator)", loc="left", color=INK)
    fig.colorbar(image, ax=ax, shrink=0.8, label="Gain")
    fig.tight_layout()
    fig.savefig(out / "heatmap_paper.png", dpi=150)
    plt.close(fig)


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
