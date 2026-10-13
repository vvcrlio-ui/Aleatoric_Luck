"""Author the fixed hardship feature contract and the panel catalog from ICPSR 31622.

Run from the repository root: python -m FFCWS_ICPSR.adapter.build_contract ...

Run once when the source release changes. Variable selection, types and category
levels come from the FFCWS metadata and the Stata value labels, not from row
values. The one exception is recorded in the contract: a variable with no
observed value among the analysis families is dropped, because the engine
rejects predictor columns that are entirely missing.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aleatoric_nk_grid.run_panels import DEFAULTS

from FFCWS_ICPSR.adapter.adapter import hardship_scores, numeric_values, onehot_values, read_columns


DATASET_PREFIX = "ffcws_icpsr_hardship"
ID_COLUMN = "idnum"
# Observation landmark (child age) -> the last wave whose variables it may use.
LANDMARK_WAVES = {
    0: "Baseline",
    1: "Year 1",
    3: "Year 3",
    5: "Year 5",
    9: "Year 9",
    15: "Year 15",
}
# The 11 Challenge hardship items in Year 15 order: free food, hungry, rent,
# eviction, utility bill, utilities shut off, borrowed money, moved in with
# others, shelter, could not see a doctor, phone disconnected. Years 1-9 are the
# mother's report, Years 15 and 22 the primary caregiver's. Year 3 asks only
# eight of them and is not an outcome age.
OUTCOMES = {
    "hardship_1": {"age": 1, "items": [
        "m2h19a", "m2h19c", "m2h19d", "m2h19e", "m2h19f", "m2h19g",
        "m2h19i", "m2h19j", "m2h19k", "m2h19l", "m2h19h"]},
    "hardship_5": {"age": 5, "items": [
        "m4i23a", "m4i23c", "m4i23d", "m4i23e", "m4i23f", "m4i23g",
        "m4i23h", "m4i23i", "m4i23j", "m4i23k", "m4i23n"]},
    "hardship_9": {"age": 9, "items": [f"m5f23{letter}" for letter in "abcdefghijk"]},
    "hardship_15": {"age": 15, "items": [f"p6j{number}" for number in range(37, 48)]},
    "hardship_22": {"age": 22, "items": [f"p7d{number}" for number in range(10, 21)]},
}
NUMERIC_TYPES = ("Continuous", "Binary", "Ordered Categorical")
N_SIZES = 6
# Ten seeds (each a new 70/30 split) with five draws each.
SEEDS = list(range(12345, 12355))
DRAWS = list(range(5))
MIN_N = 50
MODELS = [
    "ols", "ridge", "lasso", "random_forest", "xgboost", "lightgbm",
    "shallow_neural_network", "extra_trees", "super_learner",
]
EXCHANGEABILITY_JUSTIFICATION = (
    "K samples FFCWS survey variables observed by the landmark age, selected by "
    "wave, type and topic in the study metadata. They are treated as exchangeable "
    "units of observed life history; the dummy columns of one categorical variable "
    "stay together and count once."
)


def _levels(labels: dict[Any, str]) -> list[int]:
    """Labelled category codes; -1 to -9 are FFCWS missing-value codes."""
    return sorted(int(code) for code in labels if code >= 0 or code <= -10)


def build_contract(source: Path, metadata_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = pd.read_csv(metadata_path, encoding="latin-1", low_memory=False)
    metadata["key"] = metadata["new_name"].astype("string").str.strip().str.lower()
    metadata = metadata.drop_duplicates("key").set_index("key")
    with pd.io.stata.StataReader(source) as reader:
        original = {name.lower(): name for name in reader.variable_labels()}
        value_labels = reader.value_labels()
    columns = [name for name in original if name != ID_COLUMN]
    info = metadata.reindex(columns)
    unmatched = info.index[info["wave"].isna()].tolist()

    wave_landmark = {wave: landmark for landmark, wave in LANDMARK_WAVES.items()}
    in_window = info.loc[info["wave"].isin(wave_landmark)]
    excluded: dict[str, list[str]] = {
        "string_or_id": sorted(in_window.index[in_window["type"].isin(["String", "ID Number"])]),
        "paradata_and_weights": sorted(
            in_window.index[in_window["topics"].fillna("").str.contains("Paradata and weights")]
        ),
        "unordered_without_value_labels": [],
        "no_observation_among_analysis_families": [],
    }
    dropped = set(excluded["string_or_id"]) | set(excluded["paradata_and_weights"])
    candidates = []
    for name, row in in_window.iterrows():
        if name in dropped:
            continue
        levels = None
        if row["type"] == "Unordered Categorical":
            levels = _levels(value_labels.get(original[name], {}))
            if not levels:
                excluded["unordered_without_value_labels"].append(name)
                continue
        elif row["type"] not in NUMERIC_TYPES:
            raise ValueError(f"Unexpected metadata type for {name}: {row['type']!r}")
        elif row["type"] != "Continuous" and max(value_labels.get(original[name], {0: ""})) >= 100:
            # Codes of 100 and above answer on a second scale (e.g. 0-7 days a week
            # or "201 almost every day"), so the variable has no single numeric order.
            levels = _levels(value_labels[original[name]])
        candidates.append({"name": name, "landmark": wave_landmark[row["wave"]], "levels": levels})

    items = [item for definition in OUTCOMES.values() for item in definition["items"]]
    frame = read_columns(source, list(dict.fromkeys([ID_COLUMN, *items, *(c["name"] for c in candidates)])))
    scores = hardship_scores(frame, OUTCOMES)
    sample = frame.loc[scores.notna().all(axis=1)]

    sources = []
    for candidate in candidates:
        values = sample[candidate["name"]]
        if candidate["levels"] is None:
            observed = numeric_values(values).notna().any()
        else:
            observed = not np.isnan(onehot_values(values, candidate["levels"])).all()
        if observed:
            sources.append(candidate)
        else:
            excluded["no_observation_among_analysis_families"].append(candidate["name"])
    position = {name: index for index, name in enumerate(columns)}
    sources.sort(key=lambda source: (source["landmark"], position[source["name"]]))

    contract = {
        "contract_version": 1,
        "dataset_prefix": DATASET_PREFIX,
        "id_column": ID_COLUMN,
        "landmarks": {str(landmark): wave for landmark, wave in LANDMARK_WAVES.items()},
        "outcomes": OUTCOMES,
        "sample_size": int(len(sample)),
        "sources": sources,
        "excluded": {key: sorted(value) for key, value in excluded.items()},
        "metadata_unmatched": sorted(unmatched),
        "exchangeability_justification": EXCHANGEABILITY_JUSTIFICATION,
    }
    summary = {
        "families": int(len(sample)),
        "sources_by_landmark": {
            landmark: sum(1 for source in sources if source["landmark"] <= landmark)
            for landmark in LANDMARK_WAVES
        },
        "excluded": {key: len(value) for key, value in excluded.items()},
    }
    return contract, summary


def panel_catalog(summary: dict[str, Any]) -> str:
    """One panel per (landmark t, outcome age T) cell with t < T, on a 6 x 2 N-K grid."""
    families = summary["families"]
    capacity = families - math.ceil(DEFAULTS["test_size"] * families)
    n_grid = [int(value) for value in np.round(np.geomspace(MIN_N, capacity, N_SIZES))]
    k_match = summary["sources_by_landmark"][min(LANDMARK_WAVES)]
    lines = [
        "# One panel per cell of the landmark x outcome-age triangle (t < T).",
        "# Generated by adapter/build_contract.py; regenerate rather than edit.",
        f"# N: {N_SIZES} sizes up to the {capacity} training families of each split.",
        f"# K: {k_match} variables (the birth landmark's count, matched across",
        "# landmarks) and every variable observed by the landmark.",
        "model_spec_version: nkgrid-models-v6-balanced-batch-1",
        "preset: dev",
        "experiment_id: ffcws-icpsr-hardship-v1",
        "data_version: ffcws-icpsr-hardship-ard-v2",
        "repeat_plan:",
        f"  - seeds: [{', '.join(map(str, SEEDS))}]",
        f"    draws: [{', '.join(map(str, DRAWS))}]",
        "",
        "preparation:",
        "  adapter: adapter/adapter.py",
        "  contract: adapter/config/hardship.json",
        "  inputs:",
        "    source: 31622-0001-Data.dta",
        "  panel_pattern: 'hardship_t(?P<t>\\d+)_T(?P<T>\\d+)'",
        f"  schema: 'schema/{DATASET_PREFIX}_lm{{t}}.json'",
        "",
        "panels:",
    ]
    for name, definition in OUTCOMES.items():
        for landmark in LANDMARK_WAVES:
            if landmark >= definition["age"]:
                continue
            k_full = summary["sources_by_landmark"][landmark]
            panel = f"hardship_t{landmark}_T{definition['age']}"
            lines += [
                f"  - name: {panel}",
                f"    schema: schema/{DATASET_PREFIX}_lm{landmark}.json",
                f"    outcome: {name}",
                f"    models: [{', '.join(MODELS)}]",
                f"    n_grid: [{', '.join(map(str, n_grid))}]",
                f"    k_grid: [{', '.join(map(str, sorted({k_match, k_full})))}]",
                f"    out: outputs/nk_grid_{panel}.csv",
                "",
            ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    article_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--source", type=Path, required=True, help="ICPSR 31622 DS0001 Stata file")
    parser.add_argument("--metadata", type=Path, required=True, help="FFMetadata_v20_f.csv")
    parser.add_argument("--out", type=Path, default=article_root / "adapter" / "config" / "hardship.json")
    parser.add_argument("--panels", type=Path, default=article_root / "panels.yaml")
    args = parser.parse_args(argv)
    contract, summary = build_contract(args.source, args.metadata)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(contract, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    args.panels.write_text(panel_catalog(summary), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
