"""Author the fixed SMR feature contract from a known analysis-matrix header."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


ONEHOT_SOURCES = (
    "Aset2_College_major2",
    "Bset1_occ1",
    "Bset1_occmf1",
    "Bset1_occ2",
    "Bset1_occmf2",
    "Bset1_occ3",
    "Bset1_occmf3",
    "Bset1_occ4",
    "Bset1_occmf4",
    "Bset1_occ5",
    "Bset1_occmf5",
    "Bset1_occ6",
    "Bset1_occmf6",
    "Bset1_occ7",
    "Bset1_occmf7",
    "Bset1_occ8",
    "Bset1_occmf8",
    "Aset1_Fatherwfp2_1979",
    "Aset1_Motherwfp2_1979",
    "Aset1_Fatherwfp2_1980",
    "Aset1_Motherwfp2_1980",
    "Aset1_mfFatherwfp2_1979",
    "Aset1_mfMotherwfp2_1979",
    "Aset1_mfFatherwfp2_1980",
    "Aset1_mfMotherwfp2_1980",
    "Aset1_Fatherocc2_1979",
    "Aset1_Motherocc2_1979",
    "Aset1_mfFatherocc2_1979",
    "Aset1_mfMotherocc2_1979",
)
OUTCOMES = ("Cm_lhourlywage", "Cm_ltotalincome")
EXCHANGEABILITY_JUSTIFICATION = (
    "The N×K estimand samples from the paper's predeclared Aset/Bset predictor "
    "sources. These sources are treated as exchangeable units for the feature-"
    "availability experiment; dummy columns from one raw categorical variable "
    "remain atomic and are sampled together."
)


def _level_value(token: str) -> Any:
    if re.fullmatch(r"-?\d+", token):
        return int(token)
    if re.fullmatch(r"-?\d+\.\d+", token):
        return float(token)
    return token


def build_contract(source: Path) -> dict[str, Any]:
    header = pd.read_csv(source, nrows=0).columns.astype(str).tolist()
    predictors = [
        column for column in header if column.startswith(("Aset", "Bset"))
    ]
    if not predictors:
        raise ValueError("No Aset/Bset predictors found")
    missing_outcomes = [outcome for outcome in OUTCOMES if outcome not in header]
    if missing_outcomes:
        raise KeyError(f"Outcome columns not found: {missing_outcomes}")

    onehot_groups: dict[str, Any] = {}
    assigned: set[str] = set()
    for source_name in ONEHOT_SOURCES:
        prefix = f"{source_name}_"
        features = [feature for feature in predictors if feature.startswith(prefix)]
        if len(features) < 2:
            raise ValueError(
                f"Fixed one-hot source {source_name!r} resolved to fewer than two columns"
            )
        overlap = assigned & set(features)
        if overlap:
            raise ValueError(
                f"Features assigned to multiple one-hot sources: {sorted(overlap)[:5]}"
            )
        assigned.update(features)
        levels = [_level_value(feature[len(prefix) :]) for feature in features]
        onehot_groups[source_name] = {
            "features": features,
            "level_values": levels,
            "reference_level": levels[0],
        }

    return {
        "contract_version": 1,
        "dataset": "asample2_withlag",
        "outcome_columns": list(OUTCOMES),
        "predictor_columns": predictors,
        "onehot_groups": onehot_groups,
        "missing_value_codes": {},
        "scalar_unit_type": "continuous",
        "source_matrix_semantics": (
            "The provider-supplied analysis matrix already contains numeric "
            "scalar encodings and explicit missingness features. The adapter "
            "does not fit, impute, split, screen, or derive vocabularies from "
            "row values."
        ),
        "exchangeability_justification": EXCHANGEABILITY_JUSTIFICATION,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    article_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--source",
        type=Path,
        default=article_root / "data" / "private" / "asample2_withlag.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=article_root / "adapter" / "contracts" / "asample2_withlag.json",
    )
    args = parser.parse_args(argv)
    contract = build_contract(args.source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            contract,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.out)


if __name__ == "__main__":
    main()
