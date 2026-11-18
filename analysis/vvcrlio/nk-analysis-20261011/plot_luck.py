"""Plot the luck decomposition from luck.py: sample and variable luck on the grid edges for every panel, and the
test-split share of the variance for the internally split SMR panels.
Usage: python plot_luck.py RESULTS_DIR
"""
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

results = Path(sys.argv[1])
from panels import MODELS, MUTED, ROWS, TITLE

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "savefig.facecolor": "white"})
edges = json.loads((results / "luck-edges.json").read_text())

fig, axes = plt.subplots(len(ROWS), 4, figsize=(19, 4 + 2.75 * len(ROWS)), facecolor="white")
fig.subplots_adjust(left=0.05, right=0.985, top=0.9, bottom=0.075, wspace=0.28, hspace=0.55)
for r, row in enumerate(ROWS):
    for p, (panel, label) in enumerate(row):
        for side, (key, axis, xlabel, kind) in enumerate((("sample_sd", "N", "Sample size $N$ (all variables)", "Sample luck"),
                                                        ("variable_sd", "K", "Feature count $K$ (whole training pool)",
                                                         "Variable luck"))):
            ax = axes[r, 2 * p + side]
            for model, name, color in MODELS:
                d = edges[panel][model]
                ax.plot(d[axis], d[key], color=color, linewidth=1.8 if model == "super_learner" else 1.1,
                        marker="o", markersize=2.2, label=name)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(5e-4, 1)
            ax.grid(color="#e6e8ea", linewidth=0.6)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.set_title(f"{label}\n{kind}", fontsize=9.5, color=TITLE, fontweight="bold", loc="left")
            ax.set_xlabel(xlabel, fontsize=8.5)
            if p == 0 and side == 0:
                ax.set_ylabel("Within-seed SD of test $R^2$", fontsize=8.5)
fig.text(0.05, 0.965, "How much of the spread is luck in the sample, luck in the variables, or the algorithm",
         fontsize=19, fontweight="bold", color=TITLE)
fig.text(0.05, 0.948, "Standard deviation of test R² across the 5 draws within a seed, pooled over 100 seeds. A draw "
         "reorders both rows and variables, so the two are separable only on the grid edges.",
         fontsize=11, color=MUTED)
fig.text(0.05, 0.936, "Left of each pair: all variables, N varies (sample luck + algorithm). Right: the whole training "
         "pool, K varies (variable luck + algorithm). Both curves end in the same cell (N_max, K_max), where only the "
         "algorithm's noise is left.", fontsize=11, color=MUTED)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=len(MODELS), frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.028))
fig.text(0.05, 0.008, "OLS omitted (its spread near N = K is orders of magnitude larger). Classification panels: cells "
         "where any repeat was skipped (too few of the rarer class) are left out. Y axis clipped at 1.\nSamples are drawn "
         "from a finite training pool, so as N nears N_max the draws overlap and sample luck falls towards zero by "
         "construction; it understates sampling from the population there.",
         fontsize=8.5, color=MUTED)
for suffix in ("png", "pdf"):
    fig.savefig(results / f"luck-edges.{suffix}", dpi=150)
plt.close(fig)

# Test-split share: SMR draws a new random 30% test set for each seed.
SMR = [("smr_hourlywage", "Log hourly wage"), ("smr_totalincome", "Log total income")]
ALL = [("ols", "OLS")] + [(m, n) for m, n, _ in MODELS]
fig, axes = plt.subplots(len(SMR), len(ALL), figsize=(24, 7.4), facecolor="white", sharex=True, sharey=True)
fig.subplots_adjust(left=0.06, right=0.93, top=0.78, bottom=0.12, wspace=0.08, hspace=0.25)
for r, (panel, label) in enumerate(SMR):
    z = np.load(results / f"variance-{panel}.npz")
    models = list(z["models"])
    with np.errstate(invalid="ignore"):
        share = z["between"] / (z["between"] + z["within"])
    for c, (model, name) in enumerate(ALL):
        ax = axes[r, c]
        image = ax.imshow(share[models.index(model)].T, origin="lower", cmap="Blues", vmin=0, vmax=1, aspect="auto")
        if r == 0:
            ax.set_title(name, fontsize=10.5, color=TITLE, fontweight="bold")
        ticks = [0, 5, 10, 15, 19]
        ax.set_xticks(ticks, [f"{z['N'][i]:,}" for i in ticks], fontsize=7.5, rotation=45)
        ax.set_yticks(ticks, [f"{z['K'][i]:,}" for i in ticks], fontsize=7.5)
        if c == 0:
            ax.set_ylabel(f"{label}\nFeature count $K$", fontsize=9.5, color=TITLE)
        if r == len(SMR) - 1:
            ax.set_xlabel("Sample size $N$", fontsize=9)
cax = fig.add_axes([0.945, 0.2, 0.01, 0.5])
fig.colorbar(image, cax=cax).set_label("Test-split share of variance", fontsize=9)
fig.text(0.06, 0.93, "SMR: how much of the spread comes from which 30% of people land in the test set",
         fontsize=17, fontweight="bold", color=TITLE)
fig.text(0.06, 0.885, "Between-seed variance / (between + within), one-way random-effects ANOVA over 100 seeds × 5 draws. "
         "Each seed draws its own test set; draws within a seed share it. FFC uses one fixed test set (share ≈ 0, a check).\n"
         "Near (N_max, K_max) the within-seed spread shrinks towards the algorithm's noise (the draws share almost all "
         "rows and variables), which pushes the share towards 1.",
         fontsize=10.5, color=MUTED)
for suffix in ("png", "pdf"):
    fig.savefig(results / f"smr-test-split-share.{suffix}", dpi=150)
print(results / "luck-edges.png")
