"""Table of the held-out extrapolation check from forms.py: fit on N <= N_max / 4, predict the larger N.
Usage: python plot_forms.py RESULTS_DIR
"""
import csv
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
os.environ["MPLCONFIGDIR"] = str(HERE / ".mplconfig")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from panels import MUTED, TITLE

results = Path(sys.argv[1])
rows = list(csv.DictReader((results / "forms.csv").open()))
FORMS = [("POW3", "a + b·N^(−c)"), ("MMF4", "(ab + c·N^d) / (b + N^d)"), ("WBL4", "c + b·exp(−a·N^d)"),
         ("EXP3", "a + b·exp(−c·N)"), ("LOG2", "a + b·log N  (no ceiling)"), ("LAST", "last fitted value, flat")]
REGRESSION = ("gpa", "grit", "materialHardship")
GROUPS = [("FFC continuous", lambda r: r["panel"].startswith("ffc") and r["panel"].endswith(REGRESSION)),
          ("FFC binary", lambda r: r["panel"].startswith("ffc") and not r["panel"].endswith(REGRESSION)),
          ("SMR", lambda r: r["panel"].startswith("smr"))]

table = {}
for group, keep in GROUPS:
    sub = [r for r in rows if keep(r)]
    wins = dict.fromkeys((f for f, _ in FORMS), 0)
    for r in sub:
        wins[min((float(r[f"{f}_holdout_mae"]), f) for f, _ in FORMS if r[f"{f}_holdout_mae"])[1]] += 1
    for form, _ in FORMS:
        errors = [float(r[f"{form}_holdout_mae"]) for r in sub if r[f"{form}_holdout_mae"]]
        table[(group, form)] = (np.median(errors), wins[form], len(errors))
    table[group] = len(sub)

fig = plt.figure(figsize=(12, 5.4), facecolor="white")
ax = fig.add_axes([0.03, 0.2, 0.94, 0.58])
ax.set_xlim(0, 12)
ax.set_ylim(len(FORMS) + 1.4, 0)
ax.axis("off")
X = [0.1, 1.1, 4.6, 7.1, 9.6]
heads = ["Form", "1 − R² =", *(f"{g}\n({table[g]} curves)" for g, _ in GROUPS)]
for x, head in zip(X, heads):
    ax.text(x, 0.6, head, fontsize=10, fontweight="bold", va="center")
for y in (0, 1.2, len(FORMS) + 1.2):
    ax.plot([0, 12], [y, y], color="#222222", linewidth=1.0)
for i, (form, formula) in enumerate(FORMS):
    y = 1.7 + i
    ax.text(X[0], y, form, fontsize=10, va="center", fontweight="bold")
    ax.text(X[1], y, formula, fontsize=9.5, va="center", color=MUTED)
    for x, (group, _) in zip(X[2:], GROUPS):
        error, wins, fitted = table[(group, form)]
        best = error == min(table[(group, f)][0] for f, _ in FORMS)
        ax.text(x, y, f"{error:.4f}", fontsize=10, va="center", fontweight="bold" if best else "normal",
                color="#2f5596" if best else "#222222")
        ax.text(x + 0.85, y, f"best in {wins}" + (f" · fitted {fitted}" if fitted < table[group] else ""),
                fontsize=8.5, va="center", color=MUTED)
fig.text(0.03, 0.93, "Which curve form extrapolates: fit on N ≤ N_max / 4, predict the larger N",
         fontsize=14, fontweight="bold", color=TITLE)
fig.text(0.03, 0.89, "Median absolute error of the predicted mean test R² at the held-out sample sizes, with all "
         "variables; each curve is one panel × model (OLS omitted).\nBold: lowest median in the column.",
         fontsize=9.5, color=MUTED, va="top", linespacing=1.5)
fig.text(0.03, 0.13, "Forms from Viering & Loog (2022), Table I, fitted by weighted nonlinear least squares on 1 − R². "
         "The held-out range spans about a factor of four in N;\nthe ceilings in the power-law table extrapolate to "
         "infinite N, much further than this check reaches. Four-parameter forms need five points and are not fitted "
         "where fewer\nare complete (classification panels with skipped repeats at small N).",
         fontsize=8.3, color=MUTED, va="top", linespacing=1.5)
for suffix in ("png", "pdf"):
    fig.savefig(results / f"extrapolation-check.{suffix}", dpi=200)
print(results / "extrapolation-check.png")
