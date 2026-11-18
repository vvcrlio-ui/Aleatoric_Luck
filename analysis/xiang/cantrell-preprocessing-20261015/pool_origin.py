"""Which Cantrell-pool variables carry the signal for each outcome?

Fits LassoCV on the whole official training pool with the cantrell_metadata
features (mean imputation, as in the encoding), then refits it on the subset of
sources that our shared screening also keeps. Scores both on the official test
set against the training-pool mean. One fit per outcome, not the N x K grid.

Usage: python pool_origin.py PREPARED_WORK_DIR DATA_DIR NAME_MAP_CSV
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "FFCWS/adapter/src"))
sys.path.insert(0, str(ROOT / "NK_Grid/src"))
from ffcws_data_processor.common.io import read_stata_with_labels  # noqa: E402
from ffcws_data_processor.common.schema import SchemaConfig, build_shared_schema  # noqa: E402

WORK, DATA, NAME_MAP = (Path(p) for p in sys.argv[1:4])
features = pd.read_parquet(WORK / "features.parquet")
manifest = pd.read_csv(WORK / "feature_manifest.csv")
source_of = dict(zip(manifest.feature_name, manifest.sampling_source))
train, test = pd.read_csv(DATA / "train.csv"), pd.read_csv(DATA / "test.csv")

background, labels = read_stata_with_labels(DATA / "background.dta")
ours = build_shared_schema(background, train["challengeID"], value_labels=labels,
                           config=SchemaConfig(min_valid_rate=0.5, categorical_max_levels=15,
                                               min_binary_prevalence=0.01))
to_ffcws = pd.read_csv(NAME_MAP).dropna(subset=["target"]).set_index("ffc").target.to_dict()
our_pool = {to_ffcws.get(s.source_column, s.source_column) for s in ours.eligible_sources}
scales = set(pd.read_csv(ROOT / "FFCWS/adapter/src/ffcws_data_processor/cantrell_scales/data/"
                         "parsable_scales_instructions.csv", comment="#").metadata_new_name)
pool_rows = features.challengeID.isin(train.challengeID)
missing = {source_of[c]: float(features.loc[pool_rows, c].isna().mean())
           for c in features.columns if c.startswith("X_")}


def origin(source):
    if source in scales:
        return "constructed scale"
    if source in our_pool:
        return "also in our pool"
    return "only Cantrell, >50% missing" if missing.get(source, 0) > 0.5 else "only Cantrell, other"


def fit(columns, y_train, X_train, X_test, y_test, null):
    model = LassoCV(n_alphas=50, cv=KFold(5, shuffle=True, random_state=0), max_iter=20000, n_jobs=1)
    model.fit(X_train[columns], y_train)
    mse = float(np.mean((model.predict(X_test[columns]) - y_test) ** 2))
    return model, 1 - mse / null


predictors = [c for c in features.columns if c != "challengeID"]
for outcome in ["gpa", "materialHardship", "grit"]:
    tr = train[["challengeID", outcome]].dropna().merge(features, on="challengeID")
    te = test[["challengeID", outcome]].dropna().merge(features, on="challengeID")
    means = tr[predictors].mean()
    keep = [c for c in predictors if np.isfinite(means[c])]
    X_train = tr[keep].fillna(means[keep])
    X_test = te[keep].fillna(means[keep])
    sd = X_train.std().replace(0, 1)
    X_train, X_test = (X_train - X_train.mean()) / sd, (X_test - X_train.mean()) / sd
    y_train, y_test = tr[outcome].to_numpy(), te[outcome].to_numpy()
    null = float(np.mean((y_test - y_train.mean()) ** 2))

    full, r2_full = fit(keep, y_train, X_train, X_test, y_test, null)
    shared = [c for c in keep if origin(source_of[c]) == "also in our pool"]
    _, r2_shared = fit(shared, y_train, X_train, X_test, y_test, null)

    coef = pd.Series(full.coef_, index=keep)
    chosen = coef[coef != 0]
    table = pd.DataFrame({"weight": chosen.abs(), "source": [source_of[c] for c in chosen.index]})
    table["origin"] = table.source.map(origin)
    summary = table.groupby("origin").agg(columns=("weight", "size"), sources=("source", "nunique"),
                                          weight_share=("weight", "sum"))
    summary["weight_share"] = (summary.weight_share / summary.weight_share.sum()).round(3)
    print(f"\n===== {outcome}: train {len(tr)}, test {len(te)}")
    print(f"test R2, all Cantrell columns: {r2_full:.4f}; only sources also in our pool: {r2_shared:.4f}")
    print(f"selected columns {len(chosen)} from {table.source.nunique()} sources")
    print(summary.to_string())
    top = table.groupby(["source", "origin"]).weight.sum().sort_values(ascending=False).head(12)
    print("largest sources:")
    for (source, kind), weight in top.items():
        print(f"  {source:28s} {kind:28s} missing {missing.get(source, 0):.2f}  |coef| {weight:.4f}")
