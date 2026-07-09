#!/usr/bin/env python3
import csv
import math
import zipfile
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SUBMISSIONS_ZIP = ROOT / "data" / "derived" / "submissions.csv.zip"
AUTHOR_R2 = ROOT / "data" / "derived" / "r_squared_all.csv"
OUT_DIR = ROOT / "outputs" / "tables"
PY_R2 = OUT_DIR / "r_squared_all_python.csv"
COMPARE = OUT_DIR / "r_squared_all_compare.csv"


def as_float(value):
    if value in ("", "NA", "NaN", "nan"):
        return None
    return float(value)


def recompute_r2():
    sums = defaultdict(lambda: [0.0, 0.0, 0])
    with zipfile.ZipFile(SUBMISSIONS_ZIP) as zf, zf.open("submissions.csv") as raw:
        rows = csv.DictReader((line.decode("utf-8") for line in raw))
        for row in rows:
            pred = as_float(row["prediction"])
            truth = as_float(row["truth"])
            ybar = as_float(row["ybar_train"])
            if pred is None or truth is None or ybar is None:
                continue
            key = (row["outcome"], row["account"])
            sums[key][0] += (truth - pred) ** 2
            sums[key][1] += (truth - ybar) ** 2
            sums[key][2] += 1

    out = []
    for (outcome, team), (model_sse, base_sse, n) in sorted(sums.items()):
        r2 = float("nan") if base_sse == 0 else 1 - model_sse / base_sse
        out.append(
            {
                "outcome": outcome,
                "team": team,
                "r.squared": repr(r2),
                "beatingBaseline": "TRUE" if r2 > 0 else "FALSE",
                "n": str(n),
            }
        )
    return out


def read_author_r2():
    with AUTHOR_R2.open(newline="") as f:
        return {
            (row["outcome"], row["team"]): row
            for row in csv.DictReader(f)
        }


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    py_rows = recompute_r2()
    write_csv(PY_R2, py_rows, ["outcome", "team", "r.squared", "beatingBaseline", "n"])

    author = read_author_r2()
    compare = []
    max_abs_diff = 0.0
    missing = 0
    for row in py_rows:
        key = (row["outcome"], row["team"])
        author_row = author.get(key)
        if author_row is None:
            missing += 1
            continue
        py_val = float(row["r.squared"])
        author_val = float(author_row["r.squared"])
        diff = py_val - author_val
        if math.isfinite(diff):
            max_abs_diff = max(max_abs_diff, abs(diff))
        compare.append(
            {
                "outcome": key[0],
                "team": key[1],
                "python_r2": row["r.squared"],
                "author_r2": author_row["r.squared"],
                "abs_diff": repr(abs(diff)),
            }
        )

    write_csv(COMPARE, compare, ["outcome", "team", "python_r2", "author_r2", "abs_diff"])
    print(f"wrote {PY_R2.relative_to(ROOT)}")
    print(f"wrote {COMPARE.relative_to(ROOT)}")
    print(f"rows={len(py_rows)} compared={len(compare)} missing_in_author={missing}")
    print(f"max_abs_diff={max_abs_diff:.3g}")
    assert missing == 0
    assert max_abs_diff < 1e-10


if __name__ == "__main__":
    main()
