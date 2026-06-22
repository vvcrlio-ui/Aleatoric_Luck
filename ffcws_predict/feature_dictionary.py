"""Feature dictionary auditing and selection."""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from .io import DataError


REQUIRED_COLUMNS = [
    "domain",
    "concept",
    "wave",
    "respondent",
    "exact_column_name",
    "source_label",
    "include",
]


def _truthy(value) -> bool:
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "include"}


def load_feature_dictionary(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise DataError(
            "Feature dictionary is missing required columns: " + ", ".join(missing)
        )
    return df


def parse_codebook_labels(text: str) -> dict[str, str]:
    """Parse Stata-style codebook variable headings into labels."""
    lines = text.splitlines()
    sep = re.compile(r"^-{20,}$")
    heading = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s+(.+?)\s*$")
    labels: dict[str, str] = {}
    for index, line in enumerate(lines):
        if index == 0 or index + 1 >= len(lines):
            continue
        if not sep.match(lines[index - 1].strip()):
            continue
        if not sep.match(lines[index + 1].strip()):
            continue
        match = heading.match(line)
        if match:
            labels[match.group(1)] = re.sub(r"\s+", " ", match.group(2)).strip()
    return labels


def audit_feature_dictionary(
    dictionary: pd.DataFrame,
    available_columns: list[str],
    output_dir: Path,
    codebook_labels: dict[str, str] | None = None,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    available = set(available_columns)
    codebook_labels = codebook_labels or {}
    rows = []
    selected: list[str] = []
    for _, row in dictionary.iterrows():
        include = _truthy(row["include"])
        col = "" if pd.isna(row["exact_column_name"]) else str(row["exact_column_name"]).strip()
        if not include:
            status = "excluded"
        elif not col:
            status = "missing_exact_column_name"
        elif col in available:
            status = "present"
            selected.append(col)
        else:
            status = "missing_from_data"
        out = row.to_dict()
        out["resolved_status"] = status
        out["codebook_label"] = codebook_labels.get(col, "")
        source_label = "" if pd.isna(row["source_label"]) else str(row["source_label"]).strip()
        out["source_label_matches_codebook"] = (
            bool(col)
            and col in codebook_labels
            and source_label == codebook_labels[col]
        )
        rows.append(out)

    resolved = pd.DataFrame(rows)
    present = resolved["resolved_status"].eq("present")
    duplicate_present = resolved.loc[present, "exact_column_name"].duplicated(keep=False)
    cross_domain_columns = set(
        resolved.loc[present, ["exact_column_name", "domain"]]
        .drop_duplicates()
        .groupby("exact_column_name")["domain"]
        .nunique()
        .pipe(lambda s: s[s > 1])
        .index
    )
    resolved["model_domain"] = resolved["domain"]
    resolved.loc[
        resolved["exact_column_name"].isin(cross_domain_columns)
        & resolved["resolved_status"].eq("present"),
        "model_domain",
    ] = "cross_domain"
    resolved["is_duplicate_model_column"] = False
    resolved.loc[present, "is_duplicate_model_column"] = duplicate_present.to_numpy()
    domain = (
        resolved.assign(is_present=resolved["resolved_status"].eq("present"))
        .groupby("domain", dropna=False)
        .agg(
            requested_features=("include", lambda s: int(sum(_truthy(v) for v in s))),
            present_features=("is_present", "sum"),
            total_rows=("domain", "size"),
        )
        .reset_index()
    )
    model_domain_rows = (
        resolved.loc[resolved["resolved_status"].eq("present")]
        .drop_duplicates("exact_column_name")
        .groupby("model_domain", dropna=False)
        .agg(unique_model_features=("exact_column_name", "size"))
        .reset_index()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved.to_csv(output_dir / "feature_dictionary_resolved.csv", index=False)
    domain.to_csv(output_dir / "domain_feature_report.csv", index=False)
    model_domain_rows.to_csv(output_dir / "model_domain_feature_report.csv", index=False)
    selected = list(dict.fromkeys(selected))
    return selected, resolved, domain
