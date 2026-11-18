import sys, re, pandas as pd
from collections import Counter
EXT=sys.argv[1]
with pd.io.stata.StataReader("FFCWS/data/remediation_20260907/background.dta") as r:
    vlab=r.variable_labels()
m=pd.read_csv(EXT+"/data/dataset/fragile-families/FFMetadata_v09.csv",encoding="utf-8-sig",low_memory=False)
meta=m.set_index("new_name"); old=m.dropna(subset=["old_name"]).set_index("old_name")
def norm(s): return re.sub(r"[^a-z0-9]","",str(s).lower())
rows=[]
for c,lab in vlab.items():
    if c=="challengeID": continue
    if c in meta.index: how,t="new_name",c
    elif c in old.index: how,t="old_name",old.at[c,"new_name"]
    elif c.startswith("hv") and "p"+c[2:] in meta.index: how,t="hv->p","p"+c[2:]
    elif c.startswith("hv") and "ch"+c[2:] in meta.index: how,t="hv->ch","ch"+c[2:]
    else: how,t="none",None
    a=norm(lab); b=norm(meta.at[t,"varlab"]) if t else ""
    same = bool(a) and (a==b or a in b or b in a)
    rows.append(dict(ffc=c,target=t,how=how,ffc_label=lab,meta_label=meta.at[t,"varlab"] if t else None,label_match=same))
r=pd.DataFrame(rows); r.to_csv(sys.argv[2],index=False)
print(pd.crosstab(r.how,r.label_match))
for h in ["new_name","old_name","hv->p","hv->ch"]:
    x=r[(r.how==h)&~r.label_match].head(4)
    for _,y in x.iterrows(): print(h,"|",y.ffc,"->",y.target,"|",y.ffc_label,"||",y.meta_label)
