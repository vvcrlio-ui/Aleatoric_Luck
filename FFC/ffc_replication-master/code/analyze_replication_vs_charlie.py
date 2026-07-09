from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

AUTHOR_R2 = ROOT / "data" / "derived" / "r_squared_all.csv"
PYTHON_R2 = ROOT / "outputs" / "tables" / "r_squared_all_python.csv"
COMPARE_R2 = ROOT / "outputs" / "tables" / "r_squared_all_compare.csv"
CHARLIE_FIX = ROOT / "output" / "seed" / "seed_analysis_100_logit_fix.csv"

OUTCOMES = [
    "materialHardship",
    "gpa",
    "grit",
    "eviction",
    "jobTraining",
    "layoff",
]

BINARY = {"eviction", "jobTraining", "layoff"}


def pct_rank(value, values):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if np.isnan(value) or len(values) == 0:
        return np.nan
    return 100 * np.mean(values <= value)


def summarize_method(charlie, author, method):
    rows = []
    method_df = charlie[charlie["account"].eq(method)].copy()
    for outcome in OUTCOMES:
        vals = method_df.loc[method_df["outcome"].eq(outcome), "r2_holdout"].dropna()
        team_vals = author.loc[author["outcome"].eq(outcome), "author_r2"].dropna()
        if vals.empty:
            rows.append({
                "outcome": outcome,
                "method": method,
                "n_seed_nonmissing": 0,
                "mean_r2": np.nan,
                "min_r2": np.nan,
                "max_r2": np.nan,
                "seed_range": np.nan,
                "percentile_vs_teams": np.nan,
                "gap_to_best_team": np.nan,
            })
            continue
        mean = vals.mean()
        rows.append({
            "outcome": outcome,
            "method": method,
            "n_seed_nonmissing": int(vals.notna().sum()),
            "mean_r2": mean,
            "min_r2": vals.min(),
            "max_r2": vals.max(),
            "seed_range": vals.max() - vals.min(),
            "percentile_vs_teams": pct_rank(mean, team_vals),
            "gap_to_best_team": team_vals.max() - mean,
        })
    return pd.DataFrame(rows)


def main():
    author = pd.read_csv(AUTHOR_R2).rename(columns={"r.squared": "author_r2"})
    python = pd.read_csv(PYTHON_R2).rename(columns={"r.squared": "python_r2"})
    compare = pd.read_csv(COMPARE_R2)
    charlie = pd.read_csv(CHARLIE_FIX)

    team_summary = (
        author
        .groupby("outcome", as_index=False)
        .agg(
            n_teams=("team", "nunique"),
            team_median=("author_r2", "median"),
            team_p25=("author_r2", lambda x: np.nanpercentile(x, 25)),
            team_p75=("author_r2", lambda x: np.nanpercentile(x, 75)),
            best_team_r2=("author_r2", "max"),
            worst_team_r2=("author_r2", "min"),
            share_beating_baseline=("beatingBaseline", "mean"),
        )
    )

    best_team = (
        author
        .sort_values(["outcome", "author_r2"], ascending=[True, False])
        .groupby("outcome", as_index=False)
        .first()[["outcome", "team", "author_r2"]]
        .rename(columns={"team": "best_team", "author_r2": "best_team_r2_check"})
    )

    method_summary = pd.concat(
        [
            summarize_method(charlie, author, "ols"),
            summarize_method(charlie, author, "logit"),
        ],
        ignore_index=True,
    )

    wide = method_summary.pivot(index="outcome", columns="method")
    wide.columns = [f"{method}_{metric}" for metric, method in wide.columns]
    wide = wide.reset_index()

    out = (
        team_summary
        .merge(best_team, on="outcome", how="left")
        .merge(wide, on="outcome", how="left")
    )
    out["outcome"] = pd.Categorical(out["outcome"], OUTCOMES, ordered=True)
    out = out.sort_values("outcome")

    out.to_csv(TABLE_DIR / "experimental_results_vs_charlie.csv", index=False)

    replication = {
        "n_cells": len(compare),
        "max_abs_diff": compare["abs_diff"].max(),
        "mean_abs_diff": compare["abs_diff"].mean(),
        "python_author_exact_cells_1e_10": int((compare["abs_diff"] < 1e-10).sum()),
    }

    lines = []
    lines.append("# FFC replication and Charlie benchmark analysis\n")
    lines.append("## Core result\n")
    lines.append(
        "The Python reproduction matches the author-provided R2 table essentially exactly "
        f"across {replication['n_cells']} team-outcome cells "
        f"(max absolute difference = {replication['max_abs_diff']:.2e}). "
        "The empirical conclusion therefore comes from the paper/results themselves, "
        "not from a Python implementation discrepancy.\n"
    )
    lines.append("## Per-outcome comparison\n")
    for row in out.to_dict("records"):
        outcome = row["outcome"]
        lines.append(f"### {outcome}\n")
        lines.append(
            f"- Original/author result: best team R2 = {row['best_team_r2']:.3f} "
            f"({row['best_team']}), team median = {row['team_median']:.3f}, "
            f"IQR = [{row['team_p25']:.3f}, {row['team_p75']:.3f}].\n"
        )
        lines.append(
            f"- Charlie OLS 100 seeds: mean R2 = {row['ols_mean_r2']:.3f}, "
            f"range = [{row['ols_min_r2']:.3f}, {row['ols_max_r2']:.3f}], "
            f"seed range = {row['ols_seed_range']:.3f}, "
            f"percentile vs teams = {row['ols_percentile_vs_teams']:.1f}.\n"
        )
        if outcome in BINARY:
            lines.append(
                f"- Charlie logit 100 seeds: mean R2 = {row['logit_mean_r2']:.3f}, "
                f"range = [{row['logit_min_r2']:.3f}, {row['logit_max_r2']:.3f}], "
                f"seed range = {row['logit_seed_range']:.3f}, "
                f"percentile vs teams = {row['logit_percentile_vs_teams']:.1f}.\n"
            )
        else:
            lines.append("- Charlie logit: not applicable because this outcome is not binary.\n")
        lines.append("\n")

    lines.append("## Interpretation\n")
    lines.append(
        "- The paper's qualitative result is reproduced: even the best submissions have modest R2, "
        "with material hardship and GPA highest and the other outcomes much lower.\n"
    )
    lines.append(
        "- Charlie's benchmark is not a fragile single-seed artifact: the 100-seed ranges are small "
        "relative to the spread of team performance for most outcomes.\n"
    )
    lines.append(
        "- Charlie OLS is competitive with many teams and especially strong for material hardship, "
        "GPA, and job training, supporting the paper's statement that simple benchmarks were only "
        "somewhat worse than more complex submissions.\n"
    )
    lines.append(
        "- The forced-logit fix produces valid logit results only for eviction, job training, and layoff; "
        "the continuous outcomes should remain OLS-only.\n"
    )

    (TABLE_DIR / "experimental_results_vs_charlie.md").write_text("".join(lines))


if __name__ == "__main__":
    main()
