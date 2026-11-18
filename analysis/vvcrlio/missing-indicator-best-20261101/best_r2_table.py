"""Draw the best-test-R² table for the six FFC outcomes under four predictor preprocessings.

Values: compare3/best_rows_<outcome>.csv (cantrell, tree_ordinal, median_mode; regression outcomes),
best_<outcome>.csv in this directory (median + missing indicator), and manual_values.csv
(outcome, encoding, model, best; numbers copied from the earlier table). Missing cells print as "—".

Usage: python best_r2_table.py COMPARE3_DIR OUT_PNG
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
HERE = Path(__file__).parent
MODELS = [("ols", "OLS"), ("ridge", "Ridge"), ("lasso", "Lasso"), ("random_forest", "Random\nforest"),
          ("shallow_neural_network", "Shallow\nneural net"), ("extra_trees", "Extra\ntrees"),
          ("xgboost", "XGBoost"), ("lightgbm", "LightGBM"), ("super_learner", "Super\nlearner")]
ENCODINGS = [("cantrell", "Emily's code"), ("tree_ordinal", "Tree-ordinal"), ("median_mode", "Median/mode"),
             ("missing_indicator", "Median + missing\nindicator")]
# outcome, title, FFC best submission, FFC benchmark (SI Fig. S6A of Salganik et al. 2020)
OUTCOMES = [("materialHardship", "Material hardship", 0.23, 0.18), ("gpa", "GPA", 0.19, 0.11),
            ("grit", "Grit", 0.06, 0.01), ("eviction", "Eviction", 0.05, 0.01),
            ("layoff", "Layoff", 0.03, 0.01), ("jobTraining", "Job training", 0.05, 0.05)]

frames = []
for outcome, *_ in OUTCOMES:
    path = SRC / f"best_rows_{outcome}.csv"
    if path.exists():
        frames.append(pd.read_csv(path).rename(columns={"r2_full_train_mean": "best"})
                      .assign(outcome=outcome)[["outcome", "encoding", "model", "best"]])
    frames.append(pd.read_csv(HERE / f"best_{outcome}.csv")
                  .assign(outcome=outcome, encoding="missing_indicator")[["outcome", "encoding", "model", "best"]])
if (HERE / "manual_values.csv").exists():
    frames.append(pd.read_csv(HERE / "manual_values.csv"))
values = pd.concat(frames).set_index(["outcome", "encoding", "model"])["best"]
present = {o: [e for e in ENCODINGS if e[0] in values.loc[o].index.get_level_values(0)] for o, *_ in OUTCOMES}

INK, MUTED, LINE, BAND, BLUE = "#1f3a4d", "#5f6b73", "#c9ced2", "#f3f5f7", "#2f5597"
row_h, head_h = 0.62, 1.15
n_rows = sum(len(present[o]) for o, *_ in OUTCOMES)
height = 1.9 + 0.9 + head_h + n_rows * row_h + 1.3
fig = plt.figure(figsize=(20.6, height))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 20.6)
ax.set_ylim(height, 0)
ax.axis("off")

x_outcome, x_prep = 0.6, 3.1
x_models = [6.15 + 1.27 * i for i in range(len(MODELS))]
x_div = x_models[-1] + 0.6
x_ffc = [x_div + 0.85, x_div + 2.45]
x_end = 20.15

ax.text(0.47, 0.6, "Best test R² of each model across all sample sizes, feature counts and repeats",
        fontsize=23, weight="bold", color=INK, va="center")
ax.text(0.47, 1.15, "Highest single result over 20 N × 20 K × 500 repeats (100 seeds × 5 draws) · "
        "null: whole training pool · four predictor preprocessings of the same FFC data",
        fontsize=14.5, color=MUTED, va="center")
ax.text(0.47, 1.95, "Fragile Families Challenge outcomes", fontsize=17, weight="bold", color=INK, va="center")

top = 2.35
ax.plot([0.47, x_end], [top, top], color="#222", lw=1.6)
y_head = top + head_h / 2
ax.text(x_outcome, y_head, "Outcome", fontsize=13.5, weight="bold", va="center", color="#222")
ax.text(x_prep, y_head, "Preprocessing", fontsize=13.5, weight="bold", va="center", color="#222")
for x, (_, label) in zip(x_models, MODELS):
    ax.text(x, y_head, label, fontsize=13.5, weight="bold", ha="center", va="center", color="#222", linespacing=0.95)
for x, label in zip(x_ffc, ["FFC best\nsubmission", "FFC\nbenchmark"]):
    ax.text(x, y_head, label, fontsize=13.5, weight="bold", ha="center", va="center", color="#222", linespacing=0.95)
y = top + head_h
ax.plot([0.47, x_end], [y, y], color="#222", lw=1.6)

for index, (outcome, title, ffc_best, ffc_bench) in enumerate(OUTCOMES):
    block = len(present[outcome]) * row_h
    outcome_best = values.loc[outcome].round(3).max()
    if index % 2 == 0:
        ax.add_patch(plt.Rectangle((0.47, y), x_end - 0.47, block, color=BAND, lw=0))
    ax.text(x_outcome, y + block / 2, title, fontsize=15, weight="bold", va="center", color="#222")
    for row, (encoding, label) in enumerate(present[outcome]):
        yc = y + (row + 0.5) * row_h
        ax.text(x_prep, yc, label, fontsize=14 if "\n" not in label else 12.5, color=MUTED, va="center",
                linespacing=0.95)
        for x, (model, _) in zip(x_models, MODELS):
            value = values.get((outcome, encoding, model))
            if value is None:
                ax.text(x, yc, "—", fontsize=14.5, ha="center", va="center", color=MUTED)
                continue
            top_value = round(value, 3) == outcome_best
            ax.text(x, yc, f"{value:.3f}", fontsize=14.5, ha="center", va="center",
                    color=BLUE if top_value else "#222", weight="bold" if top_value else "normal")
        for x, value in zip(x_ffc, [ffc_best, ffc_bench]):
            ax.text(x, yc, "—" if value is None else f"{value:.2f}", fontsize=14.5, ha="center", va="center",
                    color=MUTED)
    y += block
    ax.plot([0.47, x_end], [y, y], color="#222" if index == len(OUTCOMES) - 1 else LINE,
            lw=1.6 if index == len(OUTCOMES) - 1 else 0.9)
ax.plot([x_div, x_div], [top, y], color=LINE, lw=1.2)

notes = [
    "Bold: highest value for the outcome over all preprocessings and models. Continuous outcomes: R² = 1 − MSE/MSE_null; "
    "binary outcomes: Brier R² = 1 − Brier/Brier_null; the null predicts the mean of the whole training pool.",
    "Emily's code: predictor preprocessing of Cantrell et al. (predicting-life-outcomes-ffcws), 7,343 variables; "
    "the other three share our screening, 3,401 variables.",
    "FFC columns: SI Appendix Fig. S6A of Salganik et al. (2020), best of 160 submissions on the FFC holdout set and a "
    "four-variable benchmark, two decimals.",
    "Our maximum is taken over 200,000 results per model and panel, so it is more optimistic than one selected fit.",
]
for i, note in enumerate(notes):
    ax.text(0.47, y + 0.55 + i * 0.3, note, fontsize=12, color=MUTED, va="center")

fig.savefig(OUT, dpi=110, facecolor="white")
print(OUT)
