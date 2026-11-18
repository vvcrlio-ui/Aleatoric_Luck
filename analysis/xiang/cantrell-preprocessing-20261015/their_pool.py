# Apply Cantrell et al. predictor rules (jobfile_monkeys_full_run_2026-01-26) to the FFC Challenge background
# on the official training pool, and count variables and model columns.
import sys, json
from collections import Counter
import numpy as np, pandas as pd
EXT, ROOT = sys.argv[1], "D:/Aleatoric_power_law/aleatoric_luck"
sys.path.insert(0, ROOT + "/FFCWS/adapter/src"); sys.path.insert(0, ROOT + "/NK_Grid/src")
from ffcws_data_processor.common.io import read_stata_with_labels
from ffcws_data_processor.common.schema import build_shared_schema, SchemaConfig

D = ROOT + "/FFCWS/data/remediation_20260907"
bg, labels = read_stata_with_labels(D + "/background.dta")
train = pd.read_csv(D + "/train.csv")
X = bg[bg.challengeID.isin(set(train.challengeID))].set_index("challengeID")
X = X.apply(pd.to_numeric, errors="coerce")
print("pool rows", len(X), "background vars", X.shape[1])

F = EXT + "/data/dataset/fragile-families/"
meta = pd.read_csv(F + "FFMetadata_v09.csv", encoding="utf-8-sig", low_memory=False).set_index("new_name")
ed = pd.read_csv(F + "Metadata_Edit_Tracker.csv"); ed.columns = ed.columns.str.strip()
for _, r in ed.iterrows():
    if str(r["Monkeys Auto-Fix"]).upper() == "TRUE":
        for v in r["Variable(s)"].split(","):
            if v.strip() in meta.index:
                meta.at[v.strip(), r["Field"]] = r["New value"]
inmeta = X.columns.intersection(meta.index)
print("vars matched in v09 metadata", len(inmeta), "unmatched", X.shape[1] - len(inmeta))
X = X[inmeta]; m = meta.loc[inmeta]
steps = []
def drop(mask, name):
    global X, m
    cols = X.columns[mask.reindex(X.columns).fillna(False).to_numpy(bool)]
    X = X.drop(columns=cols); m = m.loc[X.columns]; steps.append((name, len(cols), X.shape[1]))
drop(~m.wave.isin(["Baseline", "Year 1", "Year 3", "Year 5", "Year 9"]), "wave not Baseline-Y9")
drop(~m.type.isin(["Continuous", "Binary", "Ordered Categorical", "Unordered Categorical"]), "type")
drop(pd.to_numeric(m.n_cities_asked, errors="coerce") < 18, "n_cities < 18")
drop(m.subtopics.fillna("").str.contains("survey weights", case=False), "weights")
drop(m.subtopics.fillna("").str.contains("paradata", case=False), "paradata")
W = {"Misordered Categorical (outcomes do not have a contstant scale)": 1, "Misordered Categorical (outcomes do not have a constant scale)": 1,
     "Variable has outcomes which override a continuous answer set": 2, "A unique outcome is coded as a negative value": 4,
     "Missing data is coded as something other than the default": 5}
drop(m.warning.map(W).isin([1, 2, 4, 5]), "warning 1/2/4/5")
cj = {d["concept"]: set(d["varnames"]) for d in json.load(open(F + "VariablesToRemove.json"))}
for c in ["Standardized_Units", "Unstandardized_Units", "Combined_Units", "Uncombined_Units", "Calendar_Features"]:
    drop(pd.Series(X.columns.isin(cj[c]), index=X.columns), "concept " + c)
X = X.mask((X <= -1) & (X >= -9))           # standard missing codes -> NaN
drop(X.isna().all(), "all missing")
drop((X.var() <= 0) | X.var().isna(), "zero variance")
units = X.shape[1]
print("\n".join(f"  {a:32s} -{b:5d} -> {c}" for a, b, c in steps))
print("variables after their variable rules", units, Counter(m.type))
uc = m.index[m.type == "Unordered Categorical"]
dum = pd.get_dummies(X[uc].astype("float"), columns=list(uc), prefix=list(uc), dummy_na=True, dtype=float)
dum = dum.loc[:, ~(dum.columns.str.endswith("_nan") & (dum.sum() == 0))]
dum = dum.loc[:, dum.var() > 0]
Z = X.drop(columns=uc).join(dum)
def few_nonmodal(x):  # their rule, values within 1e-6 merged; ints here so plain counts suffice
    vc = x.value_counts()
    return vc.sum() - vc.max() < 10 if len(vc) else True
fn = Z.columns[Z.apply(few_nonmodal)]
Z = Z.drop(columns=fn)
mask = Z.columns[Z.isna().any() & ~Z.isna().all()]
print("unordered vars", len(uc), "dummy cols", dum.shape[1])
print("cols after encoding", X.shape[1] - len(uc) + dum.shape[1], "dropped <10 non-modal", len(fn),
      "value cols", Z.shape[1], "mask_ cols", len(mask), "total model cols", Z.shape[1] + len(mask))
surv = set(c if c in X.columns else c.rsplit("_", 1)[0] for c in Z.columns)
print("source variables with >=1 surviving column", len(surv))

ours = build_shared_schema(bg, train["challengeID"], value_labels=labels,
    config=SchemaConfig(min_valid_rate=0.5, categorical_max_levels=15, min_binary_prevalence=0.01,
                        variable_types={"m4f2d2": "categorical", "m5a5d03": "categorical"}))
o = {s.source_column for s in ours.eligible_sources}
print("ours eligible", len(o), "overlap with theirs", len(o & surv), "only ours", len(o - surv), "only theirs", len(surv - o))
