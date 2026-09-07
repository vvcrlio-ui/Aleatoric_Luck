"""Read-only CSV audit. Never relabel or deduplicate historical results."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import tempfile
from pathlib import Path


def audit(path: Path) -> dict:
    report = {"path": str(path.resolve()), "source_commit": None, "rows": 0,
              "over_capacity": [], "duplicate_keys": [], "invalid_rows": [], "missing_evidence": []}
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    report["sha256"] = digest.hexdigest()
    manifest = path.with_suffix(".manifest.json")
    if manifest.exists():
        report["manifest"] = json.loads(manifest.read_text(encoding="utf-8"))
        report["source_commit"] = report["manifest"].get("git", {}).get("commit")
    if not report["source_commit"]:
        report["missing_evidence"].append("source commit unavailable; inspect original manifest")
    grids = {"N": set(), "K": set()}
    with tempfile.TemporaryDirectory(prefix="nk-history-audit-") as temporary:
        connection = sqlite3.connect(str(Path(temporary) / "keys.sqlite"))
        try:
            connection.execute("CREATE TABLE keys (key TEXT PRIMARY KEY)")
            with path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                for field in ("n_train_total", "n_features_total"):
                    if field not in (reader.fieldnames or []):
                        report["missing_evidence"].append(field)
                for line, row in enumerate(reader, 2):
                    report["rows"] += 1
                    try:
                        n, k = int(row["N"]), int(row["K"])
                        if n < 1 or k < 1:
                            raise ValueError("nonpositive N/K")
                        grids["N"].add(n); grids["K"].add(k)
                        key = json.dumps([row.get("experiment_id"), row["model"], int(row["seed"]), int(row["draw"]), n, k])
                        try:
                            connection.execute("INSERT INTO keys VALUES (?)", (key,))
                        except sqlite3.IntegrityError:
                            report["duplicate_keys"].append({"line": line, "key": key})
                        for value, field in ((n, "n_train_total"), (k, "n_features_total")):
                            if row.get(field) and value > int(row[field]):
                                report["over_capacity"].append({"line": line, "field": field, "declared": value, "capacity": int(row[field])})
                    except (KeyError, ValueError, TypeError) as exc:
                        report["invalid_rows"].append({"line": line, "error": str(exc)})
        finally:
            connection.close()
    report["grid"] = {name: sorted(values) for name, values in grids.items()}
    report["exclude_from_analysis"] = bool(report["over_capacity"] or report["duplicate_keys"] or report["invalid_rows"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps([audit(path) for path in args.csv], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
