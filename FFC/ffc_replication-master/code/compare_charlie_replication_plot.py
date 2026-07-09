from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "outputs" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

AUTHOR_R2 = ROOT / "data" / "derived" / "r_squared_all.csv"
PYTHON_R2 = ROOT / "outputs" / "tables" / "r_squared_all_python.csv"
COMPARE_R2 = ROOT / "outputs" / "tables" / "r_squared_all_compare.csv"
CHARLIE_SEEDS = ROOT / "output" / "seed" / "seed_analysis_100_logit_fix.csv"

OUTCOMES = [
    "materialHardship",
    "gpa",
    "grit",
    "eviction",
    "jobTraining",
    "layoff",
]

OUTCOME_LABELS = {
    "materialHardship": "Material\nhardship",
    "gpa": "GPA",
    "grit": "Grit",
    "eviction": "Eviction",
    "jobTraining": "Job\ntraining",
    "layoff": "Layoff",
}


def save_pub_py(fig, filename, dpi=600):
    fig.savefig(f"{filename}.svg", bbox_inches="tight")
    fig.savefig(f"{filename}.pdf", bbox_inches="tight")
    fig.savefig(f"{filename}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{filename}.tiff", dpi=dpi, bbox_inches="tight")


def jitter_positions(n, center, width=0.28):
    if n == 1:
        return np.array([center])
    offsets = np.linspace(-width, width, n)
    return center + offsets


def main():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    })

    author = pd.read_csv(AUTHOR_R2)
    python = pd.read_csv(PYTHON_R2)
    compare = pd.read_csv(COMPARE_R2)
    charlie = pd.read_csv(CHARLIE_SEEDS)

    author = author.rename(columns={"r.squared": "author_r2"})
    python = python.rename(columns={"r.squared": "python_r2"})

    merged = author[["outcome", "team", "author_r2"]].merge(
        python[["outcome", "team", "python_r2"]],
        on=["outcome", "team"],
        how="inner",
    )

    charlie_ols = charlie[charlie["account"].eq("ols")].copy()
    charlie_logit = charlie[charlie["account"].eq("logit")].copy()

    ols_summary = (
        charlie_ols
        .groupby("outcome", as_index=False)
        .agg(
            charlie_ols_mean=("r2_holdout", "mean"),
            charlie_ols_min=("r2_holdout", "min"),
            charlie_ols_max=("r2_holdout", "max"),
            charlie_ols_nonmissing=("r2_holdout", lambda x: x.notna().sum()),
        )
    )
    logit_summary = (
        charlie_logit
        .groupby("outcome", as_index=False)
        .agg(
            charlie_logit_mean=("r2_holdout", "mean"),
            charlie_logit_min=("r2_holdout", "min"),
            charlie_logit_max=("r2_holdout", "max"),
            charlie_logit_nonmissing=("r2_holdout", lambda x: x.notna().sum()),
        )
    )
    team_summary = (
        author
        .groupby("outcome", as_index=False)
        .agg(
            team_median=("author_r2", "median"),
            team_p25=("author_r2", lambda x: np.nanpercentile(x, 25)),
            team_p75=("author_r2", lambda x: np.nanpercentile(x, 75)),
            team_max=("author_r2", "max"),
        )
    )
    summary = (
        team_summary
        .merge(ols_summary, on="outcome", how="left")
        .merge(logit_summary, on="outcome", how="left")
    )
    summary["outcome"] = pd.Categorical(summary["outcome"], OUTCOMES, ordered=True)
    summary = summary.sort_values("outcome")
    summary.to_csv(ROOT / "outputs" / "tables" / "charlie_vs_replication_summary.csv", index=False)

    fig = plt.figure(figsize=(9.2, 4.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.95, 1.55])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    neutral = "#5f6673"
    light = "#d7dce2"
    accent = "#c9443b"
    accent_dark = "#8f2e2b"
    blue = "#4b79a8"
    green = "#3b8a5b"

    # Panel A: exact reproduction check.
    ax0.scatter(
        merged["author_r2"],
        merged["python_r2"],
        s=7,
        color=neutral,
        alpha=0.42,
        linewidths=0,
    )
    lim_low = -0.16
    lim_high = 0.25
    ax0.plot([lim_low, lim_high], [lim_low, lim_high], color=accent, lw=0.9)
    ax0.set_xlim(lim_low, lim_high)
    ax0.set_ylim(lim_low, lim_high)
    ax0.set_box_aspect(1)
    ax0.set_xticks([-0.15, -0.05, 0.05, 0.15, 0.25])
    ax0.set_yticks([-0.15, -0.05, 0.05, 0.15, 0.25])
    ax0.set_xlabel("Author R2")
    ax0.set_ylabel("Python reproduced R2")
    ax0.set_title("A. Replication check", loc="left", fontweight="bold")
    max_abs = compare["abs_diff"].max()
    ax0.text(
        0.02,
        0.98,
        f"n = {len(compare):,}\nmax |diff| = {max_abs:.2e}",
        ha="left",
        va="top",
        transform=ax0.transAxes,
        color="#23272f",
    )

    # Panel B: author/team score distribution with Charlie OLS overlay.
    rng = np.random.default_rng(8544)
    for i, outcome in enumerate(OUTCOMES):
        vals = author.loc[author["outcome"].eq(outcome), "author_r2"].dropna().to_numpy()
        vals_clip = np.clip(vals, -0.15, 0.25)
        x = i + rng.uniform(-0.22, 0.22, size=len(vals_clip))
        ax1.scatter(x, vals_clip, s=5, color=light, alpha=0.75, linewidths=0)

        q25, med, q75 = np.nanpercentile(vals, [25, 50, 75])
        ax1.vlines(i, max(q25, -0.15), min(q75, 0.25), color=neutral, lw=2.0)
        ax1.scatter(i, np.clip(med, -0.15, 0.25), s=18, color=neutral, zorder=3)

        row = ols_summary[ols_summary["outcome"].eq(outcome)].iloc[0]
        ax1.vlines(
            i + 0.31,
            row["charlie_ols_min"],
            row["charlie_ols_max"],
            color=accent,
            lw=1.6,
            zorder=4,
        )
        ax1.scatter(
            i + 0.31,
            row["charlie_ols_mean"],
            s=24,
            marker="D",
            color=accent_dark,
            edgecolor="white",
            linewidth=0.5,
            zorder=5,
        )
        logit_row = logit_summary[logit_summary["outcome"].eq(outcome)]
        if not logit_row.empty and logit_row.iloc[0]["charlie_logit_nonmissing"] > 0:
            logit_row = logit_row.iloc[0]
            ax1.vlines(
                i + 0.43,
                logit_row["charlie_logit_min"],
                logit_row["charlie_logit_max"],
                color=green,
                lw=1.4,
                zorder=4,
            )
            ax1.scatter(
                i + 0.43,
                logit_row["charlie_logit_mean"],
                s=24,
                marker="s",
                color=green,
                edgecolor="white",
                linewidth=0.5,
                zorder=5,
            )

    ax1.axhline(0, color="#2f333b", lw=0.7, ls=(0, (2, 2)))
    ax1.set_xticks(range(len(OUTCOMES)))
    ax1.set_xticklabels([OUTCOME_LABELS[o] for o in OUTCOMES])
    ax1.tick_params(axis="x", labelsize=6)
    ax1.set_ylabel("Holdout R2")
    ax1.set_title("B. Charlie OLS vs FFC teams", loc="left", fontweight="bold")
    ax1.set_ylim(-0.15, 0.25)
    ax1.text(
        0.01,
        0.02,
        "Grey: FFC teams (extreme negatives clipped)\nRed diamond: OLS; green square: logit, across 100 seeds",
        transform=ax1.transAxes,
        ha="left",
        va="bottom",
        color="#23272f",
    )

    # Panel C: output availability, not another numeric scale.
    status = summary.set_index("outcome").loc[OUTCOMES].reset_index()
    ax2.set_title("C. Charlie output availability", loc="left", fontweight="bold")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    x_outcome, x_ols, x_logit = 0.02, 0.60, 0.86
    ax2.text(x_outcome, 0.91, "Outcome", fontweight="bold", ha="left", va="center", transform=ax2.transAxes)
    ax2.text(x_ols, 0.91, "OLS R2", fontweight="bold", ha="center", va="center", transform=ax2.transAxes)
    ax2.text(x_logit, 0.91, "logit R2", fontweight="bold", ha="center", va="center", transform=ax2.transAxes)
    for yi, row in status.iterrows():
        y_pos = 0.79 - yi * 0.115
        ax2.text(
            x_outcome,
            y_pos,
            OUTCOME_LABELS[row["outcome"]].replace("\n", " "),
            ha="left",
            va="center",
            transform=ax2.transAxes,
        )
        ax2.text(
            x_ols,
            y_pos,
            f"{int(row['charlie_ols_nonmissing'])}/100",
            ha="center",
            va="center",
            color=blue,
            fontweight="bold",
            transform=ax2.transAxes,
        )
        logit_n = int(row["charlie_logit_nonmissing"]) if not np.isnan(row["charlie_logit_nonmissing"]) else 0
        is_binary = row["outcome"] in {"eviction", "jobTraining", "layoff"}
        label = f"{logit_n}/100" if is_binary else "not binary"
        ax2.text(
            x_logit,
            y_pos,
            label,
            ha="center",
            va="center",
            color=green if logit_n > 0 else neutral,
            fontweight="bold" if logit_n > 0 else "normal",
            transform=ax2.transAxes,
        )
    ax2.text(
        x_outcome,
        0.06,
        "Forced-logit fix produces logit R2 for the three binary outcomes only.",
        ha="left",
        va="top",
        color="#23272f",
        transform=ax2.transAxes,
    )

    fig.suptitle(
        "FFC replication scores align with author results; Charlie benchmarks now include binary-outcome logit",
        x=0.01,
        ha="left",
        fontsize=9,
        fontweight="bold",
    )

    save_pub_py(fig, OUTDIR / "charlie_vs_replication")


if __name__ == "__main__":
    main()
