"""Predictor preprocessing of Cantrell et al. applied to the FFC background.

Follows the predictor rules of jobfile_monkeys_full_run_2026-01-26.toml in
emilycantrell/predicting-life-outcomes-ffcws (commit 995c845). The variable
universe comes from the FFCWS v09 metadata, not from the shared schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from ..cantrell_scales import StataLabels, compute_scales, get_var_list
from ..common.manifests import EncodedResult, manifest_frame
from ..common.schema import level_token, safe_identifier


STRATEGY = "cantrell_metadata"
RESOURCES = Path(__file__).resolve().parents[1] / "cantrell_scales" / "rules"
WAVES = ("Baseline", "Year 1", "Year 3", "Year 5", "Year 9")
TYPES = ("Continuous", "Binary", "Ordered Categorical", "Unordered Categorical")
MIN_CITIES = 18
WARNINGS = {
    "Misordered Categorical (outcomes do not have a contstant scale)": 1,
    "Misordered Categorical (outcomes do not have a constant scale)": 1,
    "Variable has outcomes which override a continuous answer set": 2,
    "A unique outcome is coded as a negative value": 4,
    "Missing data is coded as something other than the default": 5,
}
REMOVED_CONCEPTS = (
    "Standardized_Units",
    "Unstandardized_Units",
    "Combined_Units",
    "Uncombined_Units",
    "Calendar_Features",
)
MIN_NONMODAL = 10
# Every model gets mean imputation; the job file imputes before all pipelines.
IMPUTATION = {
    "continuous": "mean",
    "ordinal": "median_snap",
    "onehot_group": "atomic_mode",
    "model_overrides": {},
}


def load_metadata() -> pd.DataFrame:
    meta = pd.read_csv(RESOURCES / "FFMetadata_v09_rules.csv", dtype=str)
    meta = meta.set_index("new_name")
    edits = pd.read_csv(RESOURCES / "Metadata_Edit_Tracker.csv", dtype=str)
    edits.columns = edits.columns.str.strip()
    for row in edits.itertuples(index=False):
        if str(row[edits.columns.get_loc("Monkeys Auto-Fix")]).upper() != "TRUE":
            continue
        field = str(row[edits.columns.get_loc("Field")]).strip()
        value = row[edits.columns.get_loc("New value")]
        for name in str(row[edits.columns.get_loc("Variable(s)")]).split(","):
            if name.strip() in meta.index and field in meta.columns:
                meta.at[name.strip(), field] = value
    return meta


def ffcws_names(columns: Iterable[str], meta: pd.DataFrame) -> dict[str, str]:
    """Map FFC Challenge column names to FFCWS v09 names.

    The Challenge file keeps most v09 names, uses v09 old_name for others,
    and names the Year 3 to Year 5 home-visit variables hv*, which are p* or
    ch* in v09. Columns matching none of these have no metadata.
    """
    old = meta["old_name"].dropna()
    old = pd.Series(old.index, index=old.values)
    mapping = {}
    for column in columns:
        if column in meta.index:
            mapping[column] = column
        elif column in old.index:
            mapping[column] = old[column]
        elif column.startswith("hv") and "p" + column[2:] in meta.index:
            mapping[column] = "p" + column[2:]
        elif column.startswith("hv") and "ch" + column[2:] in meta.index:
            mapping[column] = "ch" + column[2:]
    return mapping


def scale_labels(
    value_labels: Mapping[str, Mapping[Any, str]],
    mapping: Mapping[str, str],
) -> StataLabels:
    """Value labels as pyreadstat gives them for FFCWS: "<code> <label>".

    The Challenge file lacks some value labels; v09 metadata fills them.
    """
    labels: dict[str, dict[float, str]] = {}
    fallback = pd.read_csv(RESOURCES / "FFMetadata_v09_scale_input_labels.csv")
    for row in fallback.itertuples(index=False):
        labels.setdefault(row.variable, {})[float(row.value)] = str(row.label)
    for column, codes in value_labels.items():
        if column in mapping:
            target = labels.setdefault(mapping[column], {})
            for code, label in codes.items():
                target[float(code)] = f"{int(float(code))} {label}"
    return StataLabels(labels)


def _few_nonmodal(values: pd.Series) -> bool:
    """check_nonmodal_observations: values within 1e-6 count as one."""
    counts = values.value_counts().sort_index()
    index = list(counts.index)
    tallies = list(counts.to_numpy())
    i = 0
    while i < len(index) - 1:
        if index[i + 1] - index[i] < 1e-6:
            tallies[i + 1] += tallies[i]
            del index[i], tallies[i]
            i -= 1
        i += 1
    tallies.sort()
    return sum(tallies[:-1]) < MIN_NONMODAL


def encode_cantrell_metadata(
    frame: pd.DataFrame,
    value_labels: Mapping[str, Mapping[Any, str]],
    *,
    id_column: str,
    train_ids: Iterable[Any],
) -> EncodedResult:
    meta = load_metadata()
    mapping = ffcws_names([c for c in frame.columns if c != id_column], meta)
    X = frame[list(mapping)].rename(columns=mapping).apply(
        pd.to_numeric, errors="coerce"
    )
    X = X.copy()
    pool = frame[id_column].isin(set(train_ids)).to_numpy()
    steps: list[dict[str, Any]] = [
        {"step": "no v09 metadata", "dropped": frame.shape[1] - 1 - len(mapping)}
    ]

    instructions = pd.read_csv(
        RESOURCES / "parsable_scales_instructions.csv", comment="#"
    )
    intermediate = instructions.metadata_new_name.str.contains("intermediate")
    rows = {row.metadata_new_name: row for row in instructions.itertuples()}

    def inputs_present(name: str) -> bool:
        return all(
            inputs_present(var) if var in rows and var != name else var in X.columns
            for var in get_var_list(rows[name].varstr)
        )

    candidates = instructions[
        ~intermediate & instructions.metadata_wave.isin(WAVES)
    ].metadata_new_name
    scales = [name for name in candidates if inputs_present(name)]
    computed = compute_scales(
        X, instructions, scales, scale_labels(value_labels, mapping)
    )
    X = X.drop(columns=[name for name in computed if "intermediate" in name])
    scale_meta = instructions.loc[~intermediate, instructions.columns.str.startswith("metadata")]
    scale_meta.columns = scale_meta.columns.str.replace("metadata_", "")
    meta = pd.concat([meta, scale_meta.set_index("new_name").astype(str)])
    meta = meta[~meta.index.duplicated(keep="last")]

    def drop(mask: pd.Series, step: str) -> None:
        nonlocal X
        columns = mask[mask].index.intersection(X.columns)
        X = X.drop(columns=columns)
        steps.append({"step": step, "dropped": len(columns)})

    m = meta.loc[X.columns]
    drop(~m.wave.isin(WAVES), "wave")
    m = meta.loc[X.columns]
    drop(~m.type.isin(TYPES), "type")
    m = meta.loc[X.columns]
    drop(pd.to_numeric(m.n_cities_asked, errors="coerce") < MIN_CITIES, "cities")
    m = meta.loc[X.columns]
    subtopics = m.subtopics.fillna("")
    drop(subtopics.str.contains("survey weights", case=False), "weights")
    drop(subtopics.loc[X.columns].str.contains("paradata", case=False), "paradata")
    m = meta.loc[X.columns]
    drop(m.warning.map(WARNINGS).isin([1, 2, 4, 5]), "warnings 1/2/4/5")
    concepts = {
        item["concept"]: set(item["varnames"])
        for item in json.loads((RESOURCES / "VariablesToRemove.json").read_text())
    }
    for concept in REMOVED_CONCEPTS:
        drop(pd.Series(X.columns.isin(concepts[concept]), index=X.columns), concept)
    X = X.mask((X <= -1) & (X >= -9))
    drop(X[pool].isna().all(), "all missing in pool")
    variance = X[pool].var()
    drop((variance <= 0) | variance.isna(), "zero variance in pool")

    unordered = meta.loc[X.columns].index[meta.loc[X.columns].type == "Unordered Categorical"]
    values: dict[str, pd.Series] = {}
    dummies: dict[str, list[tuple[str, pd.Series]]] = {}
    for column in X.columns:
        if column not in unordered:
            values[column] = X[column]
            continue
        levels = sorted(X.loc[pool, column].dropna().unique())
        group = [(level_token(level), (X[column] == level).astype(float)) for level in levels]
        group.append(("nan", X[column].isna().astype(float)))
        dummies[column] = [
            (token, series)
            for token, series in group
            if series[pool].var() > 0
        ]

    nonmodal = 0
    for column in list(values):
        if _few_nonmodal(values[column][pool]):
            del values[column]
            nonmodal += 1
    for column in list(dummies):
        kept = [(t, s) for t, s in dummies[column] if not _few_nonmodal(s[pool])]
        nonmodal += len(dummies[column]) - len(kept)
        if kept:
            dummies[column] = kept
        else:
            del dummies[column]
    steps.append({"step": f"fewer than {MIN_NONMODAL} non-modal in pool", "dropped": nonmodal})

    columns: dict[str, pd.Series] = {id_column: frame[id_column]}
    manifest_rows: list[dict[str, Any]] = []
    source_order = 0

    def add(feature: str, series: pd.Series, source: str, sampling: str) -> None:
        nonlocal source_order
        columns[feature] = series.astype(float)
        manifest_rows.append(
            {
                "source_column": source,
                "sampling_source": sampling,
                "feature_name": feature,
                "kind": "X",
                "strategy": STRATEGY,
                "keep": True,
                "reason": "kept",
                "source_order": source_order,
                "feature_order": 0,
                "unit_type": "continuous",
                "drop_first": False,
                "is_reference": False,
                "reference_level": None,
                "level_value": None,
                "ordinal_levels": "",
                "source_prior": 0.0,
                "prevalence": None,
                "observed_variance": None,
                "mapping_id": "",
            }
        )
        source_order += 1

    masks = 0
    for column in X.columns:
        safe = safe_identifier(column)
        if column in values:
            series = values[column]
            add(f"X_{safe}", series, column, column)
            missing = series[pool].isna()
            if missing.any() and not missing.all():
                add(f"MASK_{safe}", series.isna(), f"{column}__mask", column)
                masks += 1
        elif column in dummies:
            for index, (token, series) in enumerate(dummies[column]):
                source = column if index == 0 else f"{column}__{token}"
                add(f"D_{safe}__{token}", series, source, column)

    features = pd.DataFrame(columns)
    return EncodedResult(
        strategy=STRATEGY,
        features=features,
        feature_manifest=manifest_frame(manifest_rows),
        qa={
            "steps": steps,
            "renamed": {k: v for k, v in mapping.items() if k != v},
            "scales_computed": sorted(set(scales)),
            "scales_missing_inputs": sorted(set(candidates) - set(scales)),
            "sources": len(values) + len(dummies),
            "value_columns": len(values),
            "dummy_columns": sum(len(group) for group in dummies.values()),
            "mask_columns": masks,
        },
        imputation=IMPUTATION,
    )
