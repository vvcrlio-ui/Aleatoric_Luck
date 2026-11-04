"""Per-cell mean, P5 and maximum of a run's test score, and the text the range-bar figures need.

Sources:
sampled: MH run prod-mh-6h3g-b, R-squared against the mean of the N training rows of each repeat
         (``r2_test``), aggregated in this directory's aggregate.json.
full:    the same run against the mean of the whole training pool, one constant c for every cell.
         R-squared = 1 - MSE / c, so its mean, P5 and maximum follow from the mean, P95 and minimum
         of MSE in ../mh-mse-baseline-20261010/aggregate.json.
PATH:    an aggregate-<panel>.json written by ../full-train-columns-20261010/aggregate_panels.py,
         a production run scored against its whole training pool (r2_full_train_mean or
         mcfadden_full_train_pseudo_r2).
"""

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTCOMES = {"materialHardship": "material hardship", "gpa": "GPA", "grit": "grit",
            "eviction": "eviction", "layoff": "layoff", "jobTraining": "job training"}
FAMILIES = {"ffc_tree_ordinal_": ("FFC tree-ordinal", "Tree-ordinal"), "ffc_median_mode_": ("FFC median/mode", "Median/mode")}
SMR = {"smr_hourlywage": "SMR log hourly wage", "smr_totalincome": "SMR total income"}


def _panel_text(agg):
    """Panel label and the line describing its data and test set."""
    panel = agg["panel_id"]
    if panel in SMR:
        return SMR[panel], f"SMR · random {float(agg['test_size']):.0%} test split per seed"
    prefix = next(p for p in FAMILIES if panel.startswith(p))
    label, encoding = FAMILIES[prefix]
    return f"{label} {OUTCOMES[panel[len(prefix):]]}", f"{encoding} encoding · held-out FFC test set"


FORMULA = {"r2_full_train_mean": "$R^2 = 1 - \\mathrm{MSE}/\\mathrm{MSE}_{null}$",
           "brier_full_train_r2": "$R^2 = 1 - \\mathrm{Brier}/\\mathrm{Brier}_{null}$ (FFC)",
           "mcfadden_full_train_pseudo_r2": "$R^2_{McF} = 1 - \\log L/\\log L_{null}$"}


def _mh(baseline):
    if baseline == "sampled":
        agg = json.loads((HERE / "aggregate.json").read_text(encoding="utf-8"))
        cells = agg["cells"]
        null_text = "The null predicts the mean of the\nN training rows of each repeat."
    else:
        agg = json.loads((HERE.parent / "mh-mse-baseline-20261010" / "aggregate.json").read_text(encoding="utf-8"))
        (c,) = map(float, agg["full_train_null_mse"])
        cells = [dict(model=x["model"], N=x["N"], K=x["K"], count=x["count"], r2_test=1 - x["mean_mse"] / c,
                      p05_r2=1 - x["p95_mse"] / c, min_r2=1 - x["max_mse"] / c, max_r2=1 - x["min_mse"] / c)
                 for x in agg["cells"]]
        null_text = "The null predicts the mean of all\n1,459 training rows: one constant\nfor every cell and repeat."
    assert len(cells) == 3600 and {x["count"] for x in cells} == {5000}
    assert agg["rows"] == agg["receipt"]["rows"] == 18000000
    suffix = "" if baseline == "sampled" else "-full-train-baseline"
    return cells, dict(
        out=HERE / "visualizations", prefix="mh-20260923-p5-max-bars", suffix=suffix,
        csv_name=f"ffc-tree-ordinal-materialhardship-p5-max-3600-cells{suffix}.csv",
        top=0.45 if baseline == "sampled" else 0.30, color_top=0.25, task="regression", metric="test $R^2$",
        formula=FORMULA["r2_full_train_mean"],
        design_line="Tree-ordinal encoding · held-out FFC test set",
        subtitle="FFC tree-ordinal material hardship · 100 seeds × 50 draws per bar · completed 23 Sep 2026",
        repeats="5,000 repeats\n(100 seeds × 50 draws)", null_text=null_text,
        run_lines=["100 seeds × 50 draws · completed", f"{agg['rows']:,} valid results", "5,000 valid repeats per cell",
                   "Test rows = 1,099"],
        footer="Source: verified FFC tree-ordinal material hardship production results · prod-mh-6h3g-b · "
               "P5 and maximum aggregated 10 Oct 2026",
        audit=dict(source=agg["source"], aggregation=agg["aggregation"]))


def _panel(path):
    agg = json.loads(Path(path).read_text(encoding="utf-8"))
    task = agg["task"]
    label, design_line = _panel_text(agg)
    cells = [dict(model=x["model"], N=x["N"], K=x["K"], count=x["count"], r2_test=x["mean"], p05_r2=x["p05"],
                  min_r2=x["min"], max_r2=x["max"]) for x in agg["cells"]]
    repeats = agg["seeds"] * agg["draws"]
    counts = sorted({x["count"] for x in cells})
    count_text = f"{counts[0]:,}" if len(counts) == 1 else f"{counts[0]:,}–{counts[-1]:,}"
    design = f"{agg['seeds']} seeds × {agg['draws']} draws"
    # Leave headroom above the highest bar top; colour saturates at the highest cell mean.
    top = math.ceil(max(x["max_r2"] for x in cells) * 20) / 20 + 0.05
    color_top = max(0.05, math.ceil(max(x["r2_test"] for x in cells) * 20) / 20)
    null_what = "mean" if task == "regression" else "positive rate"
    statuses = " · ".join(f"{v:,} {k}" for k, v in sorted(agg["statuses"].items()))
    return cells, dict(
        out=Path(path).parent / "visualizations", prefix=f"{agg['panel_id']}-p5-max-bars", suffix="",
        csv_name=f"{agg['panel_id']}-p5-max-cells.csv", top=top, color_top=color_top, task=task,
        metric="McFadden $R^2$" if agg["metric"] == "mcfadden_full_train_pseudo_r2" else "test $R^2$",
        formula=FORMULA[agg["metric"]],
        subtitle=f"{label} · {design} per bar · null: whole training pool",
        repeats=f"{repeats:,} repeats\n({design})",
        null_text=(f"The null predicts the {null_what} of all\n{int(agg['n_train_total']):,} training rows: one constant\n"
                   "for every cell and repeat." if agg["split_mode"] == "external_test" else
                   f"The null predicts the {null_what} of all\n{int(agg['n_train_total']):,} training rows of the seed's\n"
                   "split: constant within a seed."),
        design_line=design_line,
        run_lines=[f"{design} · completed", statuses, f"{count_text} valid repeats per cell",
                   f"Test rows = {int(agg['n_test_total']):,}"],
        footer=f"Source: {label} production run {Path(agg['run']).name} · full-training-pool score from "
               f"evaluation commit {agg['evaluation_commit'][:7]}",
        audit=dict(source=agg["run"], metric=agg["metric"], evaluation_commit=agg["evaluation_commit"]))


def load(source):
    cells, meta = _mh(source) if source in ("sampled", "full") else _panel(source)
    meta["out"].mkdir(exist_ok=True)
    return cells, meta
