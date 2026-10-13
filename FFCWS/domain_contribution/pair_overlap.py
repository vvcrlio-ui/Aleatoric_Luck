"""How much prediction information the two official topics inside each merged domain share.

For a merged domain d = A + B and the other nine domains R, this script fits v(A), v(B),
v(R+A) and v(R+B) on the same splits fit.py used; fit.py already scored v(A+B) (d alone),
v(R) and v(R+A+B) (every domain). With g(X) the gain of X over v(empty), the overlap of A
and B is g(A) + g(B) - g(A+B) when they are alone, and the same with gains over v(R) after
the other domains. Zero means A and B carry separate information; a positive overlap is
information both carry. For comparison, the same two overlaps are computed from the fit.py
results for every pair of the ten domains, which needs no new fits.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

import fit
from fit import BINARY, DOMAINS, FFCWS

PAIRS = {
    "economy_employment": ("Finances", "Employment"),
    "family_social_ties": ("Romantic relationships", "Family and social ties"),
    "parenting_childcare": ("Parenting", "Childcare"),
    "child_development_education": ("Cognitive and behavioral development", "Education and school"),
}
PARTS = ("A", "B", "rest+A", "rest+B")
EVERY_DOMAIN = (1 << len(DOMAINS)) - 1

data = None


def selected(domain, part):
    topic = PAIRS[domain][part[-1] == "B"]
    own = [c for c in fit.state["columns"][domain] if fit.state["topic"][c] == topic]
    rest = fit.columns(EVERY_DOMAIN & ~(1 << DOMAINS.index(domain))) if part.startswith("rest") else []
    return rest + own


def start(data_dir, outcome):
    global data
    data = data_dir
    fit.state.update(outcome=outcome)


def task(job):
    seed, domain, part = job
    if fit.state.get("seed") != seed:
        fit.load(data, fit.state["outcome"], seed)
    row = fit.evaluate(selected(domain, part))
    return {"outcome": row.pop("outcome"), "seed": row.pop("seed"), "domain": domain, "part": part,
            "topic": PAIRS[domain][part[-1] == "B"], **row}


def overlaps(v, a, b, rest):
    """Overlap of a and b: alone (over v(empty)) and after rest (over v(rest)); keys index v."""
    return (v[a] + v[b] - v[a | b] - v[0],
            v[rest | a] + v[rest | b] - v[rest | a | b] - v[rest])


def summarize(results, pairs):
    rows = []
    for (outcome, seed), group in results.groupby(["outcome", "seed"]):
        metrics = ("pr_auc", "brier_r2") if outcome in BINARY else ("r2",)
        new = pairs[(pairs.outcome == outcome) & (pairs.seed == seed)].set_index(["domain", "part"])
        for metric in metrics:
            v = group.set_index("mask")[metric].to_dict()
            for domain in PAIRS:
                bit = 1 << DOMAINS.index(domain)
                rest = EVERY_DOMAIN & ~bit
                w = {0: v[0], "A": new.loc[(domain, "A"), metric], "B": new.loc[(domain, "B"), metric], "AB": v[bit],
                     "R": v[rest], "RA": new.loc[(domain, "rest+A"), metric], "RB": new.loc[(domain, "rest+B"), metric],
                     "RAB": v[EVERY_DOMAIN]}
                alone = w["A"] + w["B"] - w["AB"] - w[0]
                after = w["RA"] + w["RB"] - w["RAB"] - w["R"]
                rows.append((outcome, seed, metric, "merged", " + ".join(PAIRS[domain]), alone, after,
                             w["AB"] - w[0], w["RAB"] - w["R"]))
            for one, two in combinations(range(len(DOMAINS)), 2):
                a, b = 1 << one, 1 << two
                rest = EVERY_DOMAIN & ~a & ~b
                alone, after = overlaps(v, a, b, rest)
                rows.append((outcome, seed, metric, "domains", f"{DOMAINS[one]} + {DOMAINS[two]}", alone, after,
                             v[a | b] - v[0], v[EVERY_DOMAIN] - v[rest]))
    frame = pd.DataFrame(rows, columns=["outcome", "seed", "metric", "kind", "pair", "overlap_alone",
                                        "overlap_after_rest", "joint_gain_alone", "joint_gain_after_rest"])
    return frame.groupby(["outcome", "metric", "kind", "pair"]).agg(
        overlap_alone=("overlap_alone", "mean"), overlap_alone_se=("overlap_alone", "sem"),
        overlap_after_rest=("overlap_after_rest", "mean"), overlap_after_rest_se=("overlap_after_rest", "sem"),
        joint_gain_alone=("joint_gain_alone", "mean"), joint_gain_after_rest=("joint_gain_after_rest", "mean"),
        seeds=("seed", "size")).reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--data", type=Path, default=FFCWS / "data" / "remediation_20260907")
    parser.add_argument("--results", type=Path, default=FFCWS / "outputs" / "domain_contribution_10",
                        help="fit.py results of the ten-domain run")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    for outcome in args.outcomes:
        path = args.results / "pairs" / f"{outcome}.csv"
        if path.exists():
            continue
        fit.load(args.data, outcome, args.seeds[0])
        size = {(d, p): len(selected(d, p)) for d in PAIRS for p in PARTS}
        jobs = sorted(((s, d, p) for s in args.seeds for d in PAIRS for p in PARTS),
                      key=lambda job: size[job[1:]], reverse=True)
        with Pool(args.workers, initializer=start, initargs=(args.data, outcome)) as pool:
            rows = list(pool.imap_unordered(task, jobs))
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).sort_values(["seed", "domain", "part"]).to_csv(path.with_suffix(".tmp"), index=False)
        path.with_suffix(".tmp").replace(path)
        print(f"{outcome}: {len(rows)} fits", flush=True)

    results = pd.concat(pd.read_csv(p) for p in sorted(args.results.glob("*/seed*.csv")))
    pairs = pd.concat(pd.read_csv(p) for p in sorted((args.results / "pairs").glob("*.csv")))
    summary = summarize(results[results.outcome.isin(pairs.outcome.unique())], pairs)
    summary.to_csv(args.results / "pair_overlap.csv", index=False)
    print(summary[summary.kind == "merged"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
