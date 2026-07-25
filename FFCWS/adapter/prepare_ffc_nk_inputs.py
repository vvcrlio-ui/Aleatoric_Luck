"""Build outcome-specific NK-grid input files for the FFC data."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTCOMES = (
    "gpa",
    "grit",
    "materialHardship",
    "eviction",
    "layoff",
    "jobTraining",
)


def _safe_file_part(value: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z_]+", "_", str(value)).strip("_")
    return safe or "outcome"


def _ensure_unique_ids(frame: pd.DataFrame, id_column: str, label: str) -> None:
    if id_column not in frame:
        raise KeyError(f"{label} is missing ID column: {id_column}")
    duplicated = frame[id_column].duplicated()
    if bool(duplicated.any()):
        example = frame.loc[duplicated, id_column].iloc[0]
        raise ValueError(f"{label} has duplicate {id_column}: {example}")


def build_outcome_frames(
    features: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    outcomes: Iterable[str] | None = None,
    id_column: str = "challengeID",
) -> tuple[dict[tuple[str, str], pd.DataFrame], pd.DataFrame]:
    """Merge cleaned predictors with train/test outcomes, without imputing."""

    selected_outcomes = tuple(outcomes or DEFAULT_OUTCOMES)
    _ensure_unique_ids(features, id_column, "features")
    _ensure_unique_ids(train, id_column, "train")
    _ensure_unique_ids(test, id_column, "test")

    missing_outcomes = [
        outcome
        for outcome in selected_outcomes
        if outcome not in train.columns or outcome not in test.columns
    ]
    if missing_outcomes:
        raise KeyError(f"Outcome(s) missing from train or test: {missing_outcomes}")

    predictor_columns = [column for column in features.columns if column != id_column]
    value_predictor_columns = [
        column for column in predictor_columns if column.startswith("X_")
    ]
    categorical_columns = [
        column for column in predictor_columns if column.startswith("C_")
    ]
    missing_indicator_columns = [
        column for column in predictor_columns if column.startswith("M_")
    ]

    frames: dict[tuple[str, str], pd.DataFrame] = {}
    summary_rows: list[dict[str, int | str]] = []
    for split, outcome_frame in (("train", train), ("test", test)):
        for outcome in selected_outcomes:
            merged = outcome_frame.loc[:, [id_column, outcome]].merge(
                features,
                on=id_column,
                how="inner",
                sort=False,
            )
            observed = merged[outcome].notna()
            output = merged.loc[
                observed,
                [id_column, outcome, *predictor_columns],
            ].reset_index(drop=True)
            frames[(split, outcome)] = output
            summary_rows.append(
                {
                    "split": split,
                    "outcome": outcome,
                    "rows_in_outcome_file": int(len(outcome_frame)),
                    "rows_after_feature_merge": int(len(merged)),
                    "rows_with_observed_outcome": int(len(output)),
                    "predictor_columns": int(len(predictor_columns)),
                    "value_predictor_columns": int(len(value_predictor_columns)),
                    "categorical_columns": int(len(categorical_columns)),
                    "missing_indicator_columns": int(len(missing_indicator_columns)),
                }
            )

    return frames, pd.DataFrame(summary_rows)


def write_outcome_frames(
    frames: dict[tuple[str, str], pd.DataFrame],
    *,
    out_dir: Path,
    summary: pd.DataFrame,
    summary_name: str = "ffc_nk_input_summary.csv",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for (split, outcome), frame in frames.items():
        path = out_dir / f"ffc_{split}_{_safe_file_part(outcome)}.csv"
        frame.to_csv(path, index=False)
    summary.to_csv(out_dir / summary_name, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge cleaned FFC predictors with train/test outcomes for NK-grid."
    )
    parser.add_argument(
        "--features",
        default=str(
            ROOT
            / "data"
            / "intermediate_files"
            / "ffc_background_clean_features.csv"
        ),
        help="Cleaned feature CSV produced by prepare_ffc_analysis.py.",
    )
    parser.add_argument(
        "--train",
        default=str(ROOT / "data" / "private" / "train.csv"),
        help="FFC train outcome CSV.",
    )
    parser.add_argument(
        "--test",
        default=str(ROOT / "data" / "private" / "test.csv"),
        help="FFC test outcome CSV.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "data" / "intermediate_files" / "nk_inputs"),
        help="Directory for per-outcome train/test NK input CSVs.",
    )
    parser.add_argument("--id-column", default="challengeID")
    parser.add_argument("--outcomes", nargs="+", default=list(DEFAULT_OUTCOMES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = pd.read_csv(args.features)
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    frames, summary = build_outcome_frames(
        features,
        train,
        test,
        outcomes=args.outcomes,
        id_column=args.id_column,
    )
    write_outcome_frames(frames, out_dir=Path(args.out_dir), summary=summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
