"""Audit local FFCWS wave comparability; write aggregates and metadata only.

Does not modify source data, create family-level exports, fit models, or split data.
Run with --help for input paths. Requires pandas and scipy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd
from scipy.stats import hypergeom


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def counts(series):
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).sort_index().items()}


def describe(series, binary=False):
    valid = series.dropna()
    result = {"valid_n": len(valid), "missing_n": int(series.isna().sum()),
              "mean": float(valid.mean()), "sd": float(valid.std())}
    if binary:
        result["events"] = int(valid.sum())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--ffc-root", type=Path,
                        help="Optional local Challenge directory to compare with DS0015 by challengeID")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.data_root
    paths = {"all_years": root / "DS0001/31622-0001-Data.dta",
             "year15": root / "DS0009/31622-0009-Data.dta",
             "year22": root / "DS0012/31622-0012-Data.dta",
             "challenge": root / "DS0015/31622-0015-Zipped_package.zip",
             "metadata": args.metadata}
    items = {15: [f"P6J{i}" for i in range(37, 48)],
             22: [f"P7D{i}" for i in range(10, 21)]}
    admin = ["CP6PINT", "CP7PINT", "CP6PCGREL", "CP7PCGREL",
             "CP7INTYST", "CP7MODE", "P7F1"]
    columns = ["IDNUM", "P6J51"] + items[15] + items[22] + admin
    data = pd.read_stata(paths["all_years"], columns=columns, convert_categoricals=False)
    assert data.IDNUM.notna().all() and data.IDNUM.is_unique
    report = {"sources": {k: fingerprint(p) for k, p in paths.items()},
              "rows": len(data), "unique_family_ids": data.IDNUM.nunique(),
              "rules": {"yes": 1, "no": 2, "hardship_denominator": 11,
                        "hardship_complete_items_required": 11,
                        "all_other_item_codes": "missing; never zero",
                        "weights": "unweighted descriptive audit",
                        "exports": "aggregate statistics and variable metadata only"},
              "standalone_checks": {}, "waves": {}, "paired": {}}
    value_labels = {}
    for year, key in [(15, "year15"), (22, "year22")]:
        read_cols = ["IDNUM"] + items[year]
        standalone = pd.read_stata(paths[key], columns=read_cols, convert_categoricals=False)
        assert standalone.IDNUM.is_unique and standalone.IDNUM.notna().all()
        pd.testing.assert_frame_equal(
            data[read_cols].sort_values("IDNUM").reset_index(drop=True),
            standalone.sort_values("IDNUM").reset_index(drop=True), check_dtype=False)
        report["standalone_checks"][str(year)] = "IDs and all 11 items exactly agree with DS0001"
        with pd.io.stata.StataReader(paths[key]) as reader:
            labels = reader.value_labels()
            # These files attach same-named value-label sets to the selected fields.
            selected = items[year] + (["P6J51", "CP6PCGREL"] if year == 15 else ["CP7PCGREL", "P7F1"])
            value_labels.update({c: {str(k): str(v) for k, v in labels[c].items()}
                                 for c in selected if c in labels})
        block = data[items[year]]
        complete = block.isin([1, 2]).all(axis=1)
        data[f"hardship_{year}"] = block.eq(1).sum(axis=1).div(11).where(complete)
        eviction = "P6J40" if year == 15 else "P7D13"
        data[f"eviction_{year}"] = data[eviction].map({1: 1, 2: 0})
        assert data.loc[complete, f"hardship_{year}"].between(0, 1).all()
        # Alternate scales each require every one of their own ten items.
        alternatives = {}
        for label, drop in [("without_eviction", eviction), ("without_free_food", items[year][0])]:
            alt = block.drop(columns=drop)
            score = alt.eq(1).sum(axis=1).div(10).where(alt.isin([1, 2]).all(axis=1))
            alternatives[label] = describe(score)
        flag = data[f"CP{6 if year == 15 else 7}PINT"].eq(1)
        assert data.loc[~flag, [f"hardship_{year}", f"eviction_{year}"]].isna().all().all()
        report["waves"][str(year)] = {
            "pcg_interviews": int(flag.sum()),
            "hardship": describe(data[f"hardship_{year}"]),
            "eviction": describe(data[f"eviction_{year}"], True),
            "items": {c: counts(data[c]) for c in items[year]},
            "valid_item_count_among_interviewed": counts(block.loc[flag].isin([1, 2]).sum(axis=1)),
            "alternative_scales": alternatives}
    for outcome in ["hardship", "eviction"]:
        cols = [f"{outcome}_15", f"{outcome}_22"]
        sub = data.loc[data[cols].notna().all(axis=1)]
        pair = {"n": len(sub), "year15": describe(sub[cols[0]], outcome == "eviction"),
                "year22": describe(sub[cols[1]], outcome == "eviction"),
                "mean_difference_22_minus_15": float((sub[cols[1]] - sub[cols[0]]).mean())}
        # Exact biological-parent roles are a narrower continuity sensitivity;
        # matching non-parental relationship categories cannot establish identity.
        parent = sub.CP6PCGREL.isin([1, 2]) & sub.CP6PCGREL.eq(sub.CP7PCGREL)
        pair["same_biological_parent_role_n"] = int(parent.sum())
        pair["same_biological_parent_role_year22"] = describe(sub.loc[parent, cols[1]], outcome == "eviction")
        if outcome == "eviction":
            pair["transitions"] = [
                {"year15": a, "year22": b, "n": int((sub[cols[0]].eq(a) & sub[cols[1]].eq(b)).sum())}
                for a in [0, 1] for b in [0, 1]]
            total, events = len(sub), int(sub[cols[1]].sum())
            pair["simple_random_sample_feasibility"] = [
                {"n": n, "expected_events": n * events / total,
                 "probability_zero_events": float(hypergeom.pmf(0, total, events, n)),
                 "probability_fewer_than_10_events": float(hypergeom.cdf(9, total, events, n))}
                for n in [100, 256, 512, 1024, 1536, 2101] if n <= total]
        report["paired"][outcome] = pair
    report["year22_admin"] = {c: counts(data.loc[data.CP7PINT.eq(1), c])
                              for c in ["CP7INTYST", "CP7MODE", "P7F1"]}
    ever_yes = data.P6J40.eq(1) | data.P6J51.eq(1)
    ever_no = data.P6J40.eq(2) & data.P6J51.eq(2)
    longer = pd.Series(float("nan"), index=data.index)
    longer.loc[ever_yes] = 1
    longer.loc[ever_no] = 0
    report["year15_longer_window_candidate"] = {
        "definition": "1 if P6J40=1 or P6J51=1; 0 if both =2; otherwise missing",
        "not_verified_as_challenge_definition": True, **describe(longer, True)}
    with zipfile.ZipFile(paths["challenge"]) as archive:
        report["challenge_archive_members"] = archive.namelist()
        report["challenge_outcomes"] = {}
        for name in ["train.csv", "test.csv", "leaderboardUnfilled.csv"]:
            with archive.open(name) as stream:
                challenge = pd.read_csv(stream)
            assert challenge.challengeID.is_unique
            report["challenge_outcomes"][name] = {
                "rows": len(challenge), "columns": challenge.columns.tolist(),
                "eviction": describe(challenge.eviction, True),
                "materialHardship": describe(challenge.materialHardship)}
        with archive.open("background.dta") as stream:
            with pd.io.stata.StataReader(stream) as reader:
                names = reader.variable_labels()
                report["challenge_has_idnum"] = "idnum" in {n.lower() for n in names}
                report["challenge_has_challengeid"] = "challengeid" in {n.lower() for n in names}
        if args.ffc_root is not None:
            comparisons = {}
            for name in ["train.csv", "test.csv"]:
                local_path = args.ffc_root / name
                local = pd.read_csv(local_path).set_index("challengeID").sort_index()
                with archive.open(name) as stream:
                    archived = pd.read_csv(stream).set_index("challengeID").sort_index()
                assert local.index.is_unique and archived.index.is_unique
                assert local.index.notna().all() and archived.index.notna().all()
                assert local.index.equals(archived.index), "Local and archived family IDs differ"
                rows = {}
                for column in ["materialHardship", "eviction"]:
                    left, right = local[column], archived[column]
                    equal = left.eq(right) | (left.isna() & right.isna())
                    rows[column] = {
                        "matched_family_rows": len(equal),
                        "equal_including_missing": int(equal.sum()),
                        "mismatches": int((~equal).sum()),
                        "both_observed": int((left.notna() & right.notna()).sum()),
                        "both_missing": int((left.isna() & right.isna()).sum())}
                comparisons[name] = {"source": fingerprint(local_path), "outcomes": rows}
            with archive.open("background.dta") as stream:
                archived_hash = hashlib.file_digest(stream, "sha256").hexdigest()
            local_background = fingerprint(args.ffc_root / "background.dta")
            report["local_ffc_vs_archive"] = {
                "scope": "By-family comparison within Challenge ID namespace; NOT raw Year15-to-Challenge validation",
                "files": comparisons,
                "background_source": local_background,
                "background_byte_identical": local_background["sha256"] == archived_hash}
    report["raw_year15_vs_archived_labels"] = {
        "status": "not_completed_missing_formal_id_crosswalk",
        "verified_family_pairs": 0,
        "explanation": "IDNUM and challengeID are separate namespaces; local/archive agreement does not validate raw reconstructed scores.",
        "official_source": "https://www.icpsr.umich.edu/web/DSDR/studies/31622/versions/V5",
        "official_note": "Data Collection Notes item 3 explicitly excludes DS15 from IDNUM merging."}
    # Latin-1 is byte preserving: source fails strict UTF-8 and CP1252 decoding.
    # Retain raw text, and do not silently discard malformed bytes.
    metadata = pd.read_csv(args.metadata, encoding="latin1", low_memory=False)
    lookup = metadata.set_index(metadata.new_name.astype(str).str.upper())
    assert lookup.index.is_unique
    crosswalk = []
    for c15, c22 in zip(items[15], items[22]):
        row = {"year15": c15, "year22": c22}
        for suffix, column in [("15", c15), ("22", c22)]:
            entry = lookup.loc[column]
            assert entry.respondent == "Primary Caregiver"
            assert entry.wave == f"Year {15 if suffix == '15' else 22}"
            row[f"metadata_{suffix}"] = {
                k: None if pd.isna(entry[k]) else str(entry[k])
                for k in ["old_name", "varlab", "qtext", "warning", "wave", "respondent", "type", "in_FFC_file"]}
        row["comparability_note"] = (
            "Y22 adds 'that you otherwise could not afford'; include omit-free-food sensitivity"
            if c22 == "P7D10" else "Same substantive item and 12-month window; minor wording/format differences")
        crosswalk.append(row)
    report["metadata_encoding"] = "latin1 byte-preserving fallback; selected question text inspected"
    report["item_crosswalk"] = crosswalk
    report["value_labels"] = value_labels
    report["cautions"] = [
        "No ID crosswalk found in supplied Challenge archive; do not equate challengeID and IDNUM.",
        "Original Challenge eviction differs from current annual item; exact original construction remains unverified.",
        "CP6PCGREL and CP7PCGREL use different codes for brother/other adult; equal codes do not establish same person.",
        "P7F1 has structural skips; inspect questionnaire routing before deriving PCG-YA coresidence.",
        "Year22 value-label sets include -10; do not limit missing-code handling to -9 through -1.",
        "Observed PCG interview start years in these files are 2020-2022; wave-wide fieldwork dates are not PCG-specific dates.",
        "Predictor Kmax is unresolved until a training-only feature eligibility audit; original Challenge Kmax is not portable."]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "aggregate_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str((args.output_dir / "aggregate_summary.json").resolve()),
                      "standalone_checks": report["standalone_checks"],
                      "waves": {k: {x: v[x] for x in ["pcg_interviews", "hardship", "eviction"]}
                                for k, v in report["waves"].items()},
                      "paired": report["paired"]}, indent=2))


if __name__ == "__main__":
    main()
