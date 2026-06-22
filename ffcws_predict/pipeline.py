"""Command implementations for the FFCWS prediction CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import os

import numpy as np
import pandas as pd

from .config import (
    PUBLIC_TRUTH_URL,
    ConfigError,
    get_optional,
    get_required,
    id_column,
    load_config,
    output_dir,
    resolve_path,
    split_column,
    target_name,
)
from .feature_dictionary import (
    audit_feature_dictionary,
    load_feature_dictionary,
    parse_codebook_labels,
)
from .io import DataError, read_table, read_text_resource, write_json, write_table
from .metrics import null_mse, range_diagnostics, regression_metrics
from .models import MODEL_NAMES, fit_model
from .powerlaw import fit_all_power_laws
from .preprocessing import Preprocessor
from .reporting import write_plots, write_summary_report
from .splits import official_holdout_ids, random_stratified_holdout_ids


OUTCOME_COLUMNS = {
    "gpa",
    "grit",
    "materialhardship",
    "material_hardship",
    "eviction",
    "layoff",
    "jobtraining",
    "job_training",
}


def _configure_local_caches(out_dir: Path) -> None:
    matplotlib_dir = out_dir / ".matplotlib"
    cache_dir = out_dir / ".cache"
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_dir)
    os.environ["XDG_CACHE_HOME"] = str(cache_dir)


def _path(cfg: dict[str, Any], key: str, required: bool = True):
    value = get_required(cfg, key) if required else get_optional(cfg, key)
    resolved = resolve_path(cfg, value)
    if resolved is None and required:
        raise ConfigError(f"Missing required path: {key}")
    return resolved


def _read_truth(cfg: dict[str, Any]) -> tuple[pd.DataFrame | None, str]:
    truth_path = _path(cfg, "paths.truth_csv_path", required=False)
    if truth_path:
        return read_table(truth_path), str(truth_path)
    truth_url = get_optional(cfg, "paths.truth_url", PUBLIC_TRUTH_URL)
    if truth_url:
        return read_table(truth_url), str(truth_url)
    return None, ""


def _read_codebook_labels(cfg: dict[str, Any]) -> tuple[dict[str, str], str]:
    codebook_path = _path(cfg, "paths.codebook_path", required=False)
    if not codebook_path:
        return {}, ""
    text = read_text_resource(codebook_path)
    return parse_codebook_labels(text), str(codebook_path)


def _feature_paths(cfg: dict[str, Any]) -> tuple[Path | str, Path | str, Path]:
    features = _path(cfg, "paths.features_path")
    outcomes = _path(cfg, "paths.train_outcomes_path")
    dictionary = _path(cfg, "paths.feature_dictionary_path")
    assert features is not None and outcomes is not None and isinstance(dictionary, Path)
    return features, outcomes, dictionary


def _validate_no_leakage(columns: list[str], id_col: str, target: str) -> None:
    blocked = {id_col.lower(), target.lower(), *OUTCOME_COLUMNS}
    leaking = [col for col in columns if col.lower() in blocked]
    if leaking:
        raise DataError(
            "Feature dictionary includes outcome/ID columns that would leak labels: "
            + ", ".join(leaking)
        )


def audit_features(config_path: str | Path) -> None:
    cfg = load_config(config_path)
    out_dir = output_dir(cfg)
    _configure_local_caches(out_dir)
    features_path, _, dictionary_path = _feature_paths(cfg)
    features = read_table(features_path)
    dictionary = load_feature_dictionary(dictionary_path)
    codebook_labels, codebook_source = _read_codebook_labels(cfg)
    selected, resolved, domain = audit_feature_dictionary(
        dictionary,
        features.columns.tolist(),
        out_dir,
        codebook_labels=codebook_labels,
    )
    write_json(
        {
            "n_available_columns": len(features.columns),
            "n_dictionary_rows": len(dictionary),
            "n_present_dictionary_rows": int(resolved["resolved_status"].eq("present").sum()),
            "n_selected_present_columns": len(selected),
            "n_duplicate_selected_rows": int(
                resolved.loc[
                    resolved["resolved_status"].eq("present"),
                    "exact_column_name",
                ].duplicated().sum()
            ),
            "n_cross_domain_columns": int(
                resolved.loc[
                    resolved["resolved_status"].eq("present")
                    & resolved["model_domain"].eq("cross_domain"),
                    "exact_column_name",
                ].nunique()
            ),
            "selected_columns": selected,
            "codebook_source": codebook_source,
            "n_codebook_labels": len(codebook_labels),
        },
        out_dir / "feature_audit_report.json",
    )


def prepare(config_path: str | Path) -> None:
    cfg = load_config(config_path)
    out_dir = output_dir(cfg)
    _configure_local_caches(out_dir)
    id_col = id_column(cfg)
    target = target_name(cfg)
    split_col = split_column(cfg)
    seed = int(get_optional(cfg, "experiment.seed", 333))

    features_path, outcomes_path, dictionary_path = _feature_paths(cfg)
    features = read_table(features_path)
    outcomes = read_table(outcomes_path)
    truth, truth_source = _read_truth(cfg)
    codebook_labels, codebook_source = _read_codebook_labels(cfg)

    for name, df in {"features": features, "train_outcomes": outcomes}.items():
        if id_col not in df.columns:
            raise DataError(f"{name} is missing ID column: {id_col}")
    if target not in outcomes.columns:
        raise DataError(
            f"Training outcome file is missing target column '{target}'. Provide "
            "`paths.train_outcomes_path`; this framework does not reconstruct "
            "Year 15 labels from questionnaire items."
        )

    features[id_col] = features[id_col].astype(str)
    outcomes[id_col] = outcomes[id_col].astype(str)
    outcomes[target] = pd.to_numeric(outcomes[target], errors="coerce")

    split_type = "official_challenge"
    if truth is not None:
        truth[id_col] = truth[id_col].astype(str)
        truth[target] = pd.to_numeric(truth[target], errors="coerce")
        holdout_ids = official_holdout_ids(truth, id_col, target)
    else:
        if not bool(get_optional(cfg, "split.allow_random_fallback", False)):
            raise DataError(
                "Could not read official truth file and random fallback is disabled."
            )
        holdout_ids = random_stratified_holdout_ids(
            outcomes,
            id_col,
            target,
            fraction=float(get_optional(cfg, "split.random_holdout_fraction", 0.2)),
            seed=seed,
        )
        split_type = "random_stratified_non_official"

    dictionary = load_feature_dictionary(dictionary_path)
    feature_cols, _, domain = audit_feature_dictionary(
        dictionary,
        features.columns.tolist(),
        out_dir,
        codebook_labels=codebook_labels,
    )
    if not feature_cols:
        raise DataError(
            "Feature dictionary resolved zero present included columns. Run "
            "`audit-features` and fill exact_column_name values before training."
        )
    _validate_no_leakage(feature_cols, id_col, target)

    train_labels = outcomes.loc[
        outcomes[target].notna() & ~outcomes[id_col].isin(holdout_ids),
        [id_col, target],
    ].copy()
    if train_labels.empty:
        raise DataError("No non-holdout training labels remain after applying split.")

    if truth is not None:
        holdout_labels = truth.loc[
            truth[id_col].isin(holdout_ids) & truth[target].notna(), [id_col, target]
        ].copy()
    else:
        holdout_labels = outcomes.loc[
            outcomes[id_col].isin(holdout_ids) & outcomes[target].notna(), [id_col, target]
        ].copy()
    if holdout_labels.empty:
        raise DataError("No holdout labels available for target.")

    selected = features[[id_col, *feature_cols]].copy()
    train = train_labels.merge(selected, on=id_col, how="inner")
    holdout = holdout_labels.merge(selected, on=id_col, how="inner")
    if train.empty or holdout.empty:
        raise DataError("Feature/label merge produced empty train or holdout data.")
    train[split_col] = "train"
    holdout[split_col] = "holdout"
    prepared = pd.concat([train, holdout], ignore_index=True, sort=False)
    prepared = prepared[[id_col, split_col, target, *feature_cols]]

    written_path = write_table(
        prepared,
        out_dir / "prepared_material_hardship.parquet",
        prefer_parquet=True,
    )
    write_json(
        {
            "target": target,
            "id_column": id_col,
            "split_column": split_col,
            "data_version": str(get_optional(cfg, "data.version", "")),
            "split_type": split_type,
            "truth_source": truth_source,
            "n_holdout_ids": len(holdout_ids),
            "n_train_rows": len(train),
            "n_holdout_rows": len(holdout),
            "n_feature_columns": len(feature_cols),
            "prepared_dataset_path": str(written_path),
            "feature_dictionary_path": str(dictionary_path),
            "codebook_source": codebook_source,
            "n_codebook_labels": len(codebook_labels),
            "models": _models(cfg),
            "learning_curve_models": _models(cfg, "learning_curve.models"),
            "imputation_strategies": _configured_imputation_strategies(cfg),
            "seed": seed,
        },
        out_dir / "run_manifest.json",
    )


def _prepared_path(out_dir: Path) -> Path:
    parquet = out_dir / "prepared_material_hardship.parquet"
    csv = out_dir / "prepared_material_hardship.csv"
    if parquet.exists():
        return parquet
    if csv.exists():
        return csv
    raise DataError("Prepared dataset not found. Run `prepare` first.")


def _load_prepared(cfg: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    out_dir = output_dir(cfg)
    df = read_table(_prepared_path(out_dir))
    id_col = id_column(cfg)
    target = target_name(cfg)
    split_col = split_column(cfg)
    feature_cols = [c for c in df.columns if c not in {id_col, target, split_col}]
    return df, feature_cols


TREE_NATIVE_NAN_MODELS = {"xgboost", "lightgbm"}


def _configured_imputation_strategies(cfg: dict[str, Any]) -> list[str]:
    raw = get_optional(cfg, "preprocessing.imputation_strategy", ["median"])
    if isinstance(raw, str):
        strategies = [raw]
    else:
        strategies = list(raw)
    invalid = [value for value in strategies if value not in {"median", "native_nan"}]
    if invalid:
        raise ConfigError(
            "Invalid preprocessing.imputation_strategy values: "
            + ", ".join(str(value) for value in invalid)
        )
    return strategies or ["median"]


def _model_imputation_strategies(model_name: str, cfg: dict[str, Any]) -> list[str]:
    configured = _configured_imputation_strategies(cfg)
    if model_name in TREE_NATIVE_NAN_MODELS and "native_nan" in configured:
        return ["median", "native_nan"] if "median" in configured else ["native_nan"]
    return ["median"]


def _model_run_key(model_name: str, imputation_strategy: str) -> str:
    return f"{model_name}__{imputation_strategy}"


def _preprocessor_from_config(
    cfg: dict[str, Any],
    standardize: bool,
    imputation_strategy: str = "median",
) -> Preprocessor:
    return Preprocessor(
        missing_codes=list(get_optional(cfg, "preprocessing.missing_codes", [-9, -8, -7, -6, -5, -4, -3, -2, -1])),
        missing_threshold=float(get_optional(cfg, "preprocessing.missing_threshold", 0.8)),
        low_variance_threshold=float(get_optional(cfg, "preprocessing.low_variance_threshold", 1e-12)),
        standardize=standardize,
        imputation_strategy=imputation_strategy,
    )


def _models(cfg: dict[str, Any], key: str = "models") -> list[str]:
    fallback = get_optional(
        cfg,
        "models",
        ["mean_baseline", "ridge", "random_forest"],
    )
    requested = list(get_optional(cfg, key, fallback))
    invalid = [m for m in requested if m not in MODEL_NAMES]
    if invalid:
        raise ConfigError(f"Unknown models: {', '.join(invalid)}")
    return requested


def train(config_path: str | Path) -> None:
    cfg = load_config(config_path)
    out_dir = output_dir(cfg)
    _configure_local_caches(out_dir)
    df, feature_cols = _load_prepared(cfg)
    target = target_name(cfg)
    split_col = split_column(cfg)
    seed = int(get_optional(cfg, "experiment.seed", 333))
    n_jobs = int(get_optional(cfg, "experiment.n_jobs", 1))

    train_df = df[df[split_col].eq("train")].copy()
    holdout_df = df[df[split_col].eq("holdout")].copy()
    y_train = train_df[target].astype(float)
    y_holdout = holdout_df[target].astype(float)
    train_mean, base_mse = null_mse(y_train, y_holdout)

    predictions = []
    metrics = []
    reports = {}
    imputation_comparison = []
    for model_name in _models(cfg):
        for imputation_strategy in _model_imputation_strategies(model_name, cfg):
            standardize = model_name in {"ols", "ridge", "lasso", "elastic_net"}
            pre = _preprocessor_from_config(
                cfg,
                standardize=standardize,
                imputation_strategy=imputation_strategy,
            )
            X_train = pre.fit_transform(train_df[feature_cols])
            X_holdout = pre.transform(holdout_df[feature_cols])
            fitted = fit_model(model_name, X_train, y_train, seed=seed, n_jobs=n_jobs)
            pred = fitted.predict(X_holdout)
            row = {
                "model": model_name,
                "imputation_strategy": imputation_strategy,
                "train_mean_baseline": train_mean,
                "null_mse": base_mse,
                "cv_used": fitted.cv_used,
                **regression_metrics(y_holdout, pred, null_mse=base_mse),
            }
            metrics.append(row)
            run_key = _model_run_key(model_name, imputation_strategy)
            reports[run_key] = {
                "model": model_name,
                "imputation_strategy": imputation_strategy,
                "preprocessing": pre.report(),
                "model_params": fitted.params,
            }
            imputation_comparison.append(
                {
                    "model": model_name,
                    "imputation_strategy": imputation_strategy,
                    "train_rows": int(X_train.shape[0]),
                    "holdout_rows": int(X_holdout.shape[0]),
                    "train_columns": int(X_train.shape[1]),
                    "holdout_columns": int(X_holdout.shape[1]),
                    "train_contains_nan": bool(pd.isna(X_train).to_numpy().any()),
                    "holdout_contains_nan": bool(pd.isna(X_holdout).to_numpy().any()),
                    "missing_indicator_columns": len(pre.missing_indicator_columns_),
                }
            )
            predictions.append(
                pd.DataFrame(
                    {
                        "challengeID": holdout_df[id_column(cfg)].values,
                        "model": model_name,
                        "imputation_strategy": imputation_strategy,
                        "actual": y_holdout.values,
                        "prediction": pred,
                    }
                )
            )

    metrics_df = pd.DataFrame(metrics)
    preds_df = pd.concat(predictions, ignore_index=True)
    metrics_df.to_csv(out_dir / "model_metrics.csv", index=False)
    preds_df.to_csv(out_dir / "holdout_predictions.csv", index=False)
    diff_rows = []
    for model_name in sorted(TREE_NATIVE_NAN_MODELS):
        sub = preds_df[preds_df["model"].eq(model_name)]
        if set(sub["imputation_strategy"]) >= {"median", "native_nan"}:
            wide = sub.pivot(
                index="challengeID",
                columns="imputation_strategy",
                values="prediction",
            )
            actual = sub.drop_duplicates("challengeID").set_index("challengeID")["actual"]
            wide["actual"] = actual
            wide["model"] = model_name
            wide["native_minus_median"] = wide["native_nan"] - wide["median"]
            diff_rows.append(wide.reset_index())
    if diff_rows:
        pd.concat(diff_rows, ignore_index=True).to_csv(
            out_dir / "imputation_prediction_differences.csv",
            index=False,
        )
    range_diagnostics(preds_df, y_train, y_holdout).to_csv(
        out_dir / "prediction_range_diagnostics.csv", index=False
    )
    write_json(reports, out_dir / "preprocessing_report.json")
    write_json(
        {
            "requested_strategies": _configured_imputation_strategies(cfg),
            "rows": imputation_comparison,
        },
        out_dir / "imputation_comparison.json",
    )
    write_plots(out_dir)


def _sample_sizes(n_train: int, cfg: dict[str, Any]) -> np.ndarray:
    explicit = get_optional(cfg, "learning_curve.sample_sizes")
    if explicit:
        values = np.array([int(v) for v in explicit], dtype=int)
    else:
        n_sizes = int(get_optional(cfg, "learning_curve.n_sizes", 10))
        min_fraction = float(get_optional(cfg, "learning_curve.min_fraction", 0.1))
        max_fraction = float(get_optional(cfg, "learning_curve.max_fraction", 1.0))
        values = np.unique(
            np.clip(
                np.round(np.linspace(min_fraction, max_fraction, n_sizes) * n_train),
                2,
                n_train,
            ).astype(int)
        )
    return np.unique(np.clip(values, 2, n_train))


def learning_curve(config_path: str | Path) -> None:
    cfg = load_config(config_path)
    out_dir = output_dir(cfg)
    _configure_local_caches(out_dir)
    df, feature_cols = _load_prepared(cfg)
    target = target_name(cfg)
    split_col = split_column(cfg)
    seed = int(get_optional(cfg, "experiment.seed", 333))
    n_jobs = int(get_optional(cfg, "experiment.n_jobs", 1))
    n_draws = int(get_optional(cfg, "learning_curve.n_draws", 50))
    boot = int(get_optional(cfg, "learning_curve.bootstrap_iterations", 2000))
    reset_models = set(get_optional(cfg, "learning_curve.reset_models", []) or [])
    fixed_hyperparams = get_optional(cfg, "learning_curve.fixed_hyperparams", {}) or {}

    train_df = df[df[split_col].eq("train")].copy()
    holdout_df = df[df[split_col].eq("holdout")].copy()
    y_train_full = train_df[target].astype(float)
    y_holdout = holdout_df[target].astype(float)
    _, base_mse = null_mse(y_train_full, y_holdout)
    sizes = _sample_sizes(len(train_df), cfg)

    rng = np.random.default_rng(seed)
    curve_path = out_dir / "learning_curve.csv"
    if curve_path.exists():
        existing = pd.read_csv(curve_path)
        if reset_models:
            existing = existing[~existing["model"].isin(reset_models)].copy()
        existing_keys = {
            (str(row.model), int(row.n_samples), int(row.draw))
            for row in existing.itertuples(index=False)
        }
    else:
        existing = pd.DataFrame()
        existing_keys = set()
    rows = []

    def write_curve_checkpoint() -> None:
        frames = [frame for frame in (existing, pd.DataFrame(rows)) if not frame.empty]
        if not frames:
            return
        checkpoint = pd.concat(frames, ignore_index=True)
        checkpoint = checkpoint.drop_duplicates(
            subset=["model", "n_samples", "draw"],
            keep="last",
        ).sort_values(["model", "n_samples", "draw"])
        checkpoint.to_csv(curve_path, index=False)

    for model_name in _models(cfg, "learning_curve.models"):
        standardize = model_name in {"ols", "ridge", "lasso", "elastic_net"}
        for n_samples in sizes:
            for draw in range(n_draws):
                draw_seed = int(rng.integers(0, 2**31 - 1))
                key = (model_name, int(n_samples), int(draw))
                if key in existing_keys:
                    continue
                if n_samples >= len(train_df):
                    sub = train_df
                else:
                    sub = train_df.sample(n=n_samples, replace=False, random_state=draw_seed)
                y_sub = sub[target].astype(float)
                pre = _preprocessor_from_config(cfg, standardize=standardize)
                X_sub = pre.fit_transform(sub[feature_cols])
                X_holdout = pre.transform(holdout_df[feature_cols])
                model_fixed_params = fixed_hyperparams.get(model_name)
                if model_fixed_params:
                    if model_name != "elastic_net":
                        raise ConfigError(
                            "learning_curve.fixed_hyperparams currently supports "
                            "elastic_net only."
                        )
                    from sklearn.linear_model import ElasticNet

                    estimator = ElasticNet(
                        alpha=float(model_fixed_params["alpha"]),
                        l1_ratio=float(model_fixed_params["l1_ratio"]),
                        max_iter=int(model_fixed_params.get("max_iter", 20000)),
                        random_state=draw_seed,
                    )
                    estimator.fit(X_sub, y_sub)
                    pred = np.asarray(estimator.predict(X_holdout), dtype=float)
                else:
                    fitted = fit_model(model_name, X_sub, y_sub, seed=draw_seed, n_jobs=n_jobs)
                    pred = fitted.predict(X_holdout)
                metric = regression_metrics(y_holdout, pred, null_mse=base_mse)
                rows.append(
                    {
                        "model": model_name,
                        "n_samples": int(n_samples),
                        "draw": int(draw),
                        "seed": draw_seed,
                        "train_target_mean": float(y_sub.mean()),
                        "train_target_variance": float(y_sub.var()),
                        "null_mse": base_mse,
                        **metric,
                    }
                )
            write_curve_checkpoint()

    write_curve_checkpoint()
    results = pd.read_csv(curve_path)
    write_json(
        {
            "models": _models(cfg, "learning_curve.models"),
            "sample_sizes": [int(value) for value in sizes],
            "n_draws": n_draws,
            "bootstrap_iterations": boot,
            "reset_models": sorted(reset_models),
            "fixed_hyperparams": fixed_hyperparams,
        },
        out_dir / "learning_curve_report.json",
    )
    power = fit_all_power_laws(
        results,
        null_mse=base_mse,
        seed=seed,
        bootstrap_iterations=boot,
    )
    power.to_csv(out_dir / "power_law_fit.csv", index=False)
    write_plots(out_dir)


def summarize(config_path: str | Path) -> None:
    cfg = load_config(config_path)
    out_dir = output_dir(cfg)
    _configure_local_caches(out_dir)
    write_plots(out_dir)
    write_summary_report(out_dir)
