"""Fit LightGBM on every subset of the ten FFC domains, for one outcome and seed at a time.

Families with an observed outcome are pooled from train.csv and test.csv and split
70/30 by the seed (stratified for binary outcomes); every subset of one outcome and
seed uses that split. Mask bit i selects DOMAINS[i]. Mask 0 is the constant
training-mean prediction, and mask 2047 adds the sources outside every domain, so it
holds all 3,400 sources.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer

FFCWS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FFCWS.parent / "NK_Grid" / "src"))

from aleatoric_nk_grid.evaluation import classification_denominators, regression_denominators
from aleatoric_nk_grid.model_registry import load_model_params, make_model

DOMAINS = ("demographics", "economy_employment", "housing_neighborhood", "family_social_ties",
           "parenting_childcare", "child_development_education", "health",
           "attitudes_expectations", "systems", "parent_education_traits")
ALL_SOURCES = 2047
BINARY = {"eviction", "layoff", "jobTraining"}

state = {}


def load(data, outcome, seed):
    features = pd.read_parquet(data / "adapter_work" / "tree_ordinal" / "features.parquet").set_index("challengeID")
    y = pd.concat([pd.read_csv(data / "train.csv"), pd.read_csv(data / "test.csv")]).set_index("challengeID")[outcome].dropna()
    manifest = pd.read_csv(data / "adapter_work" / "tree_ordinal" / "feature_manifest.csv")
    domain = pd.read_csv(FFCWS / "domain_contribution" / "domains.csv").set_index("source").domain
    column_domain = manifest.set_index("feature_name").source_column.map(domain)
    binary = outcome in BINARY
    train, test = train_test_split(y.index, test_size=0.3, random_state=seed, stratify=y if binary else None)
    state.update(
        outcome=outcome, seed=seed, binary=binary,
        X_train=features.loc[train], X_test=features.loc[test],
        y_train=y[train].to_numpy(int if binary else float), y_test=y[test].to_numpy(int if binary else float),
        columns={d: [c for c in features.columns if column_domain[c] == d] for d in DOMAINS},
        params=load_model_params(FFCWS / "model_params.yaml", task="classification" if binary else "regression",
                                 models=["lightgbm"])["lightgbm"])


def columns(mask):
    if mask == ALL_SOURCES:
        return list(state["X_train"].columns)
    return [c for i, d in enumerate(DOMAINS) if mask >> i & 1 for c in state["columns"][d]]


def scores(prediction):
    y_test, y_train = state["y_test"], state["y_train"]
    if state["binary"]:
        metrics = classification_denominators(y_test, prediction, y_train, y_train)
        return {"pr_auc": average_precision_score(y_test, prediction), "roc_auc": roc_auc_score(y_test, prediction),
                "brier": metrics["brier"], "brier_r2": metrics["brier_full_train_r2"]}
    metrics = regression_denominators(y_test, prediction, y_train, y_train)
    return {"r2": metrics["skill_train_mean"], "mse": metrics["mse"]}


def fit(mask):
    selected = columns(mask)
    start = time.perf_counter()
    if mask == 0:
        rounds, prediction = 0, np.full(len(state["y_test"]), state["y_train"].mean())
    else:
        model = make_model("lightgbm", state["seed"], 1, "classification" if state["binary"] else "regression",
                           state["params"], preprocessor=FunctionTransformer())
        model.fit(state["X_train"][selected], state["y_train"])
        rounds = model.best_rounds_
        X_test = state["X_test"][selected]
        prediction = model.predict_proba(X_test)[:, 1] if state["binary"] else model.predict(X_test)
    return {"outcome": state["outcome"], "seed": state["seed"], "mask": mask,
            "domains": "all_sources" if mask == ALL_SOURCES else "+".join(d for i, d in enumerate(DOMAINS) if mask >> i & 1),
            "n_sources": len(selected), "n_train": len(state["y_train"]), "n_test": len(state["y_test"]),
            "test_positives": int(state["y_test"].sum()) if state["binary"] else "",
            "rounds": rounds, **scores(prediction), "fit_seconds": time.perf_counter() - start}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--data", type=Path, default=FFCWS / "data" / "remediation_20260907",
                        help="directory with train.csv, test.csv and adapter_work/tree_ordinal/")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--out", type=Path, default=FFCWS / "outputs" / "domain_contribution")
    args = parser.parse_args()

    masks = [*range(1 << len(DOMAINS)), ALL_SOURCES]
    for outcome in args.outcomes:
        for seed in args.seeds:
            path = args.out / outcome / f"seed{seed}.csv"
            if path.exists():
                continue
            load(args.data, outcome, seed)
            size = {mask: len(columns(mask)) for mask in masks}
            start = time.perf_counter()
            with Pool(args.workers, initializer=load, initargs=(args.data, outcome, seed)) as pool:
                rows = list(pool.imap_unordered(fit, sorted(masks, key=size.get, reverse=True)))
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).sort_values("mask").to_csv(path.with_suffix(".tmp"), index=False)
            path.with_suffix(".tmp").replace(path)
            print(f"{outcome} seed {seed}: {len(rows)} fits in {time.perf_counter() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
