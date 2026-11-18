"""Draw each model's best single test R-squared per panel: FFC outcomes (A) and SMR outcomes (B) in one figure.

Best = the highest single result over every (N, K) cell, seed and draw. R-squared is against the whole
training pool; FFC binary outcomes use the FFC Brier form. References: FFC best submission and benchmark
from SI Appendix Fig. S6A of Salganik et al. (2020); for SMR, the best model of the fullest specification
(+ B2) in the NLSY79 comparison table (Table 2).
Usage: python plot_best_table.py   (reads final/aggregate-*.json, writes final/best-r2-table.{png,pdf,csv})
"""
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ["MPLCONFIGDIR"] = str(HERE / ".mplconfig")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = [("ols", "OLS"), ("ridge", "Ridge"), ("lasso", "Lasso"), ("random_forest", "Random\nforest"),
          ("shallow_neural_network", "Shallow\nneural net"), ("extra_trees", "Extra\ntrees"), ("xgboost", "XGBoost"),
          ("lightgbm", "LightGBM"), ("super_learner", "Super\nlearner")]
FFC_OUTCOMES = [("materialHardship", "Material hardship"), ("gpa", "GPA"), ("grit", "Grit"), ("eviction", "Eviction"),
                ("jobTraining", "Job training"), ("layoff", "Layoff")]
FFC = {"materialHardship": (0.23, 0.18), "gpa": (0.19, 0.11), "grit": (0.06, 0.01), "eviction": (0.05, 0.01),
       "jobTraining": (0.05, 0.05), "layoff": (0.03, 0.01)}
# NLSY79 Table 2, + B2 row: the largest value across its six methods.
SMR = [("smr_hourlywage", "Log hourly wage", (0.53,)), ("smr_totalincome", "Log total income", (0.33,))]
INK, MUTED, RULE, BAND, BEST = "#222222", "#5c6971", "#c9ced3", "#f4f5f6", "#2f5596"
plt.rcParams["font.family"] = "DejaVu Sans"

best = {}
for path in (HERE / "final").glob("aggregate-*.json"):
    agg = json.loads(path.read_text(encoding="utf-8"))
    best[agg["panel_id"]] = {m: max(c["max"] for c in agg["cells"] if c["model"] == m) for m, _ in MODELS}


LEAD = [("Outcome", 1.75), ("Preprocessing", 1.15)]
REFS = 2
WIDTHS = [w for _, w in LEAD] + [0.82] * len(MODELS) + [1.05] * REFS
X = [0.0]
for _w in WIDTHS[:-1]:
    X.append(X[-1] + _w)
TOTAL, ROW_H, HEAD_H, PANEL_H = sum(WIDTHS), 0.42, 0.8, 0.45


def draw(ax, top, label, rows, references):
    """One panel from height ``top``: label, header and rows (outcome, preprocessing, best by model, references)."""
    ax.text(0, top + PANEL_H / 2, label, va="center", fontsize=11.5, fontweight="bold", color="#203b4c")
    top += PANEL_H
    names = [n for n, _ in LEAD] + [n for _, n in MODELS] + references + [""] * (REFS - len(references))
    first_ref = len(LEAD) + len(MODELS)
    for i, name in enumerate(names):
        left = i < len(LEAD)
        ax.text(X[i] + (0.08 if left else WIDTHS[i] / 2), top + HEAD_H / 2, name, ha="left" if left else "center",
                va="center", fontsize=9, fontweight="bold", color=INK, linespacing=1.15)
    bottom = top + HEAD_H + ROW_H * len(rows)
    for y in (top, top + HEAD_H, bottom):
        ax.plot([0, TOTAL], [y, y], color=INK, linewidth=1.0)
    ax.plot([X[first_ref] - 0.03] * 2, [top, bottom], color=RULE, linewidth=0.8)
    grouped = rows[0][1] is not None
    step = 2 if grouped else 1
    for r, (outcome, encoding, values, refs) in enumerate(rows):
        y0 = top + HEAD_H + r * ROW_H
        yc = y0 + ROW_H / 2
        if (r // step) % 2 == 0:
            ax.add_patch(plt.Rectangle((0, y0), TOTAL, ROW_H, color=BAND, zorder=0, linewidth=0))
        if r % step == 0:
            ax.text(X[0] + 0.08, yc + (ROW_H / 2 if grouped else 0), outcome, va="center", fontsize=9.5, color=INK,
                    fontweight="bold")
        ax.text(X[1] + 0.08, yc, encoding or "—", va="center", fontsize=9, color=MUTED)
        best_value = max(values.values())
        for j, (model, _) in enumerate(MODELS):
            column = len(LEAD) + j
            is_top = values[model] == best_value
            ax.text(X[column] + WIDTHS[column] / 2, yc, f"{values[model]:.3f}".replace("-", "−"), ha="center",
                    va="center", fontsize=9.5, fontweight="bold" if is_top else "normal", color=BEST if is_top else INK)
        for j, value in enumerate(refs):
            column = first_ref + j
            ax.text(X[column] + WIDTHS[column] / 2, yc, f"{value:.2f}", ha="center", va="center", fontsize=9.5,
                    color=MUTED)
        if r % step == step - 1 and r < len(rows) - 1:
            ax.plot([0, TOTAL], [y0 + ROW_H] * 2, color=RULE, linewidth=0.6)
    return bottom


ffc_rows = [(label, encoding, best[prefix + outcome], FFC[outcome]) for outcome, label in FFC_OUTCOMES
            for prefix, encoding in (("ffc_tree_ordinal_", "Tree-ordinal"), ("ffc_median_mode_", "Median/mode"))]
smr_rows = [(label, None, best[panel], refs) for panel, label, refs in SMR]
ffc_refs, smr_refs = ["FFC best\nsubmission", "FFC\nbenchmark"], ["NLSY79 table\nbest (+ B2)"]
height = 2 * (PANEL_H + HEAD_H) + ROW_H * (len(ffc_rows) + len(smr_rows)) + 0.35
fig_w, fig_h = TOTAL + 0.6, height + 2.6
fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
ax = fig.add_axes([0.3 / fig_w, 1.55 / fig_h, TOTAL / fig_w, height / fig_h])
ax.set_xlim(0, TOTAL)
ax.set_ylim(height, 0)
ax.axis("off")
fig.text(0.3 / fig_w, 1 - 0.28 / fig_h, "Best test R² of each model across all sample sizes, feature counts and repeats",
         fontsize=15, fontweight="bold", color="#203b4c", va="top")
fig.text(0.3 / fig_w, 1 - 0.68 / fig_h,
         "Highest single result over 20 N × 20 K × 500 repeats (100 seeds × 5 draws) · null: whole training pool · "
         "FFC binary outcomes scored as in the FFC (Brier)", fontsize=10, color=MUTED, va="top")
end = draw(ax, 0, "A  Fragile Families Challenge outcomes", ffc_rows, ffc_refs)
draw(ax, end + 0.35, "B  SMR outcomes", smr_rows, smr_refs)
fig.text(0.3 / fig_w, 1.25 / fig_h,
         "Bold: highest of the nine models in the row. R² = 1 − MSE/MSE_null; for FFC binary outcomes MSE is the Brier score "
         "of the predicted probability.\nFFC columns: SI Appendix Fig. S6A of Salganik et al. (2020), best of 160 submissions "
         "on the FFC holdout set and a four-variable benchmark, two decimals.\nNLSY79 column: the largest test R² across OLS, "
         "Ridge, Lasso, random forest, gradient boosting and BART in the fullest specification (+ B2) of the NLSY79 comparison "
         "(Table 2).\nOur maximum is taken over 200,000 results per model and panel (SMR: each with its own random 30% test "
         "split), so it is more optimistic than one selected fit.", fontsize=8.3, color=MUTED, va="top", linespacing=1.5)
for suffix in ("png", "pdf"):
    fig.savefig(HERE / "final" / f"best-r2-table.{suffix}", dpi=200)
with (HERE / "final" / "best-r2-table.csv").open("w", encoding="utf-8", newline="") as handle:
    handle.write("panel,outcome,preprocessing," + ",".join(m for m, _ in MODELS) + ",reference_1,reference_2\n")
    for panel, rows, refs in (("FFC", ffc_rows, ffc_refs), ("SMR", smr_rows, smr_refs)):
        for outcome, encoding, values, ref_values in rows:
            cells = [str(v) for v in ref_values] + [""] * (REFS - len(ref_values))
            handle.write(f"{panel},{outcome},{encoding or ''}," + ",".join(repr(values[m]) for m, _ in MODELS) + ","
                         + ",".join(cells) + "\n")
print(HERE / "final" / "best-r2-table.png")
