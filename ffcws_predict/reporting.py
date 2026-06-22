"""Markdown reports and plots."""

from __future__ import annotations

import json
from pathlib import Path
import os

import pandas as pd


def _table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    shown = df.head(max_rows).copy()
    headers = [str(col) for col in shown.columns]
    rows = [[str(value) for value in row] for row in shown.to_numpy()]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]
    header = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _load_prepared(output_dir: Path) -> pd.DataFrame:
    parquet = output_dir / "prepared_material_hardship.parquet"
    csv = output_dir / "prepared_material_hardship.csv"
    if parquet.exists():
        try:
            return pd.read_parquet(parquet)
        except Exception:
            pass
    if csv.exists():
        return pd.read_csv(csv)
    return pd.DataFrame()


def _target_distribution(output_dir: Path, manifest: dict) -> pd.DataFrame:
    prepared = _load_prepared(output_dir)
    target = manifest.get("target", "materialHardship")
    split_col = manifest.get("split_column", "split")
    if prepared.empty or target not in prepared or split_col not in prepared:
        return pd.DataFrame()
    return (
        prepared.groupby(split_col)[target]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .reset_index()
        .rename(columns={split_col: "split"})
    )


def _main_epsilon(metrics: pd.DataFrame, power: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty or power.empty:
        return pd.DataFrame()
    if "status" not in power or "model" not in power or "model" not in metrics:
        return pd.DataFrame()
    stable = power[power["status"].eq("stable")].copy()
    if stable.empty:
        return pd.DataFrame()
    candidates = metrics.merge(stable, on="model", how="inner")
    if candidates.empty or "mse" not in candidates:
        return pd.DataFrame()
    cols = [
        "model",
        "mse",
        "epsilon",
        "epsilon_ci_low",
        "epsilon_ci_high",
        "alpha",
        "status",
    ]
    return candidates.sort_values("mse")[cols].head(1)


def _model_notes(metrics: pd.DataFrame) -> list[str]:
    if metrics.empty or "model" not in metrics or "mse" not in metrics:
        return []
    notes: list[str] = []
    linear = metrics[metrics["model"].isin(["ridge", "lasso", "elastic_net"])]
    if not linear.empty:
        best_linear = linear.sort_values("mse").iloc[0]
        tree = metrics[metrics["model"].isin(["random_forest", "xgboost", "lightgbm"])]
        weaker_tree = tree[tree["mse"] > float(best_linear["mse"])]
        if not weaker_tree.empty:
            names = ", ".join(
                dict.fromkeys(weaker_tree.sort_values("mse")["model"].tolist())
            )
            notes.append(
                f"- `{names}` underperform the best regularized linear model "
                f"(`{best_linear['model']}`) on this holdout. With n around 1,459, "
                "187 hand-selected predictors, and weak signal, boosted/tree "
                "models can select slightly too-complex fits even with internal CV; "
                "this is expected behavior rather than a pipeline error."
            )
    if "ols" in set(metrics["model"]) and not linear.empty:
        ols = metrics[metrics["model"].eq("ols")].iloc[0]
        best_linear = linear.sort_values("mse").iloc[0]
        if float(ols["mse"]) > float(best_linear["mse"]):
            notes.append(
                f"- OLS is retained as an unregularized baseline, but its MSE "
                f"({float(ols['mse']):.6g}) is worse than `{best_linear['model']}` "
                f"({float(best_linear['mse']):.6g}), supporting the use of "
                "regularization for this feature set."
            )
    return notes


def _imputation_robustness(metrics: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if (
        metrics.empty
        or "imputation_strategy" not in metrics.columns
        or "model" not in metrics.columns
    ):
        return pd.DataFrame(), []
    subset = metrics[
        metrics["model"].isin(["xgboost", "lightgbm"])
        & metrics["imputation_strategy"].isin(["median", "native_nan"])
    ].copy()
    if subset.empty:
        return pd.DataFrame(), []
    cols = ["model", "imputation_strategy", "mse", "rmse", "mae", "r2"]
    table = subset[cols].sort_values(["model", "mse"]).reset_index(drop=True)
    notes: list[str] = []
    for model, group in subset.groupby("model"):
        strategies = set(group["imputation_strategy"])
        if {"median", "native_nan"}.issubset(strategies):
            median = group[group["imputation_strategy"].eq("median")].iloc[0]
            native = group[group["imputation_strategy"].eq("native_nan")].iloc[0]
            delta = float(native["mse"]) - float(median["mse"])
            winner = "native_nan" if delta < 0 else "median"
            notes.append(
                f"- `{model}`: `{winner}` has lower MSE; "
                f"native_nan - median MSE difference = {delta:.6g}."
            )
    return table, notes


def _imputation_matrix_table(output_dir: Path) -> pd.DataFrame:
    path = output_dir / "imputation_comparison.json"
    if not path.exists():
        return pd.DataFrame()
    data = _load_manifest(path)
    rows = data.get("rows", [])
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows)
    cols = [
        "model",
        "imputation_strategy",
        "train_rows",
        "holdout_rows",
        "train_columns",
        "holdout_columns",
        "train_contains_nan",
        "holdout_contains_nan",
        "missing_indicator_columns",
    ]
    return table[[col for col in cols if col in table.columns]]


def _imputation_prediction_difference_table(output_dir: Path) -> pd.DataFrame:
    path = output_dir / "imputation_prediction_differences.csv"
    if not path.exists():
        return pd.DataFrame()
    diffs = pd.read_csv(path)
    if diffs.empty:
        return pd.DataFrame()
    rows = []
    for model, group in diffs.groupby("model"):
        delta = group["native_minus_median"].astype(float)
        rows.append(
            {
                "model": model,
                "count": int(delta.count()),
                "mean_diff": float(delta.mean()),
                "std_diff": float(delta.std()),
                "min_diff": float(delta.min()),
                "median_diff": float(delta.median()),
                "max_diff": float(delta.max()),
                "mean_abs_diff": float(delta.abs().mean()),
                "prediction_corr": float(group[["median", "native_nan"]].corr().iloc[0, 1]),
            }
        )
    return pd.DataFrame(rows)


def write_summary_report(output_dir: Path) -> Path:
    manifest = output_dir / "run_manifest.json"
    manifest_data = _load_manifest(manifest)
    preprocessing = output_dir / "preprocessing_report.json"
    metrics = pd.read_csv(output_dir / "model_metrics.csv") if (output_dir / "model_metrics.csv").exists() else pd.DataFrame()
    domain = pd.read_csv(output_dir / "domain_feature_report.csv") if (output_dir / "domain_feature_report.csv").exists() else pd.DataFrame()
    model_domain = pd.read_csv(output_dir / "model_domain_feature_report.csv") if (output_dir / "model_domain_feature_report.csv").exists() else pd.DataFrame()
    power = pd.read_csv(output_dir / "power_law_fit.csv") if (output_dir / "power_law_fit.csv").exists() else pd.DataFrame()
    learning_curve_report = _load_manifest(output_dir / "learning_curve_report.json")
    imputation_table, imputation_notes = _imputation_robustness(metrics)
    imputation_matrix = _imputation_matrix_table(output_dir)
    imputation_prediction_diffs = _imputation_prediction_difference_table(output_dir)
    target_distribution = _target_distribution(output_dir, manifest_data)
    main_epsilon = _main_epsilon(metrics, power)
    model_notes = _model_notes(metrics)
    fixed_hyperparams = learning_curve_report.get("fixed_hyperparams", {})
    fixed_hyperparam_lines = [
        f"- `{model}`: fixed hyperparameters from "
        f"`{params.get('source', 'not recorded')}` "
        f"(alpha={params.get('alpha')}, l1_ratio={params.get('l1_ratio')})"
        for model, params in fixed_hyperparams.items()
    ]

    lines = [
        "# FFCWS Material Hardship Prediction Summary",
        "",
        "## Data and Split",
        f"- Data version: {manifest_data.get('data_version', 'not recorded')}",
        f"- Target: `{manifest_data.get('target', 'materialHardship')}`",
        f"- Split type: `{manifest_data.get('split_type', 'not prepared')}`",
        f"- Truth source: `{manifest_data.get('truth_source', 'not recorded')}`",
        f"- Holdout IDs: {manifest_data.get('n_holdout_ids', 'not recorded')}",
        f"- Train rows: {manifest_data.get('n_train_rows', 'not recorded')}",
        f"- Holdout rows: {manifest_data.get('n_holdout_rows', 'not recorded')}",
        f"- Feature columns after dictionary de-duplication: {manifest_data.get('n_feature_columns', 'not recorded')}",
        f"- Models: {', '.join(manifest_data.get('models', [])) if manifest_data.get('models') else 'not recorded'}",
        f"- Learning-curve models: {', '.join(manifest_data.get('learning_curve_models', [])) if manifest_data.get('learning_curve_models') else 'not recorded'}",
        f"- Prepared dataset: `{Path(str(manifest_data.get('prepared_dataset_path', 'missing'))).name}`",
        f"- Manifest: `{manifest.name}`" if manifest.exists() else "- Manifest: missing",
        f"- Preprocessing report: `{preprocessing.name}`" if preprocessing.exists() else "- Preprocessing report: missing",
        "",
        "## Target Distribution",
        _table(target_distribution),
        "",
        "## Domain Feature Coverage",
        _table(domain),
        "",
        "## Model-Domain Feature Coverage",
        _table(model_domain),
        "",
        "## Model Holdout Metrics",
        _table(metrics.sort_values("mse") if "mse" in metrics else metrics),
        "",
        "## Imputation Robustness Check",
        _table(imputation_table),
        "\n".join(imputation_notes) if imputation_notes else "_No native-NaN comparison rows._",
        "",
        "## Imputation Matrix Diagnostics",
        _table(imputation_matrix),
        "",
        "## Imputation Prediction Differences",
        _table(imputation_prediction_diffs),
        "",
        "## Main Epsilon Estimate",
        _table(main_epsilon),
        "",
        "## Learning Curve Settings",
        f"- Draws per sample size: {learning_curve_report.get('n_draws', 'not recorded')}",
        f"- Bootstrap iterations: {learning_curve_report.get('bootstrap_iterations', 'not recorded')}",
        f"- Sample sizes: {learning_curve_report.get('sample_sizes', 'not recorded')}",
        "\n".join(fixed_hyperparam_lines) if fixed_hyperparam_lines else "- Fixed hyperparameters: none",
        "",
        "## Power-law Error Floor",
        _table(power),
        "",
        "## Model Notes",
        "\n".join(model_notes) if model_notes else "_No model notes._",
        "",
        "## Methodological Limits",
        "- `epsilon` is conditional on the chosen feature dictionary, model family, split, and finite-sample extrapolation.",
        "- The power-law form may not be identified well if the observed learning curve has not reached a stable regime.",
        "- FFCWS missingness and attrition are not assumed to be random; missingness indicators are retained as predictors, but this does not solve selection bias.",
        "- Official Challenge comparability depends on using the DS15 Challenge features, labels, and reference holdout IDs.",
    ]
    path = output_dir / "summary_report.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def write_plots(output_dir: Path) -> None:
    matplotlib_dir = output_dir / ".matplotlib"
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_dir)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    metrics_path = output_dir / "model_metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        if not metrics.empty and "mse" in metrics:
            fig, ax = plt.subplots(figsize=(8, 4))
            metrics.sort_values("mse").plot.bar(x="model", y="mse", ax=ax, legend=False)
            ax.set_ylabel("Holdout MSE")
            ax.set_title("Model MSE comparison")
            fig.tight_layout()
            fig.savefig(output_dir / "model_mse_comparison.png", dpi=160)
            plt.close(fig)

    curve_path = output_dir / "learning_curve.csv"
    if curve_path.exists():
        curve = pd.read_csv(curve_path)
        if not curve.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            for model, group in curve.groupby("model"):
                means = group.groupby("n_samples", as_index=False)["mse"].mean()
                ax.plot(means["n_samples"], means["mse"], marker="o", label=model)
            ax.set_xlabel("Training samples")
            ax.set_ylabel("Holdout MSE")
            ax.set_title("Learning curves")
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / "learning_curves.png", dpi=160)
            plt.close(fig)

            power_path = output_dir / "power_law_fit.csv"
            if power_path.exists():
                import numpy as np

                from .powerlaw import power_law

                power = pd.read_csv(power_path)
                fig, ax = plt.subplots(figsize=(8, 4))
                for model, group in curve.groupby("model"):
                    means = group.groupby("n_samples", as_index=False)["mse"].mean()
                    ax.scatter(means["n_samples"], means["mse"], label=f"{model} observed")
                    fit = power[power["model"].eq(model)]
                    if not fit.empty and fit[["c", "alpha", "epsilon"]].notna().all(axis=None):
                        row = fit.iloc[0]
                        xs = np.linspace(means["n_samples"].min(), means["n_samples"].max(), 100)
                        ys = power_law(xs, row["c"], row["alpha"], row["epsilon"])
                        ax.plot(xs, ys, linestyle="--", label=f"{model} fitted")
                ax.set_xlabel("Training samples")
                ax.set_ylabel("Holdout MSE")
                ax.set_title("Power-law fits")
                ax.legend(fontsize=8)
                fig.tight_layout()
                fig.savefig(output_dir / "power_law_fits.png", dpi=160)
                plt.close(fig)

    power_path = output_dir / "power_law_fit.csv"
    if power_path.exists():
        power = pd.read_csv(power_path)
        if not power.empty and "epsilon" in power:
            fig, ax = plt.subplots(figsize=(8, 4))
            power = power.sort_values("epsilon")
            yerr = [
                (power["epsilon"] - power["epsilon_ci_low"]).clip(lower=0),
                (power["epsilon_ci_high"] - power["epsilon"]).clip(lower=0),
            ]
            ax.errorbar(power["model"], power["epsilon"], yerr=yerr, fmt="o")
            ax.set_ylabel("epsilon")
            ax.set_title("Power-law epsilon estimates")
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            fig.savefig(output_dir / "epsilon_ci.png", dpi=160)
            plt.close(fig)
