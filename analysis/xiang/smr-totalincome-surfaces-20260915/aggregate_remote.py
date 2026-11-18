import json, sys, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

source = Path('/valhalla/projects/ehpc-dev-2026d08-299/aleatoric_luck-smr-production-20260914/SMR/outputs/smr_totalincome-production-720-24h-20260914/recovered-linear-fallback-20260915/final.csv')
receipt = json.loads((source.parent / 'verified.json').read_text())
assert receipt['complete'] and receipt['rows'] == 18000000
assert receipt['statuses'] == {'ok':18000000}
assert receipt['final_csv_sha256'] == 'dd93e2784bb8c527283a1485ed387c7f2c9675c885e0bb4d58e47f354bef83b6'
cols = ['model','seed','draw','N','K','r2_test','mse','null_mse_train_mean','status','error','dataset','algorithm_version','outcome','split_mode','n_test_total']
stats, seen, metadata = {}, {}, {}
total = 0
status_counts = {}
skip_reasons = {}
version_counts = {}
for chunk in pd.read_csv(source, usecols=cols, chunksize=250000, low_memory=False, dtype={"error": str}):
    assert (chunk.status == 'ok').all(), chunk.status.value_counts().to_dict()
    for k,v in chunk.status.value_counts().items(): status_counts[k]=status_counts.get(k,0)+int(v)
    for k,v in chunk.loc[chunk.status=='skipped','error'].fillna('NO REASON').value_counts().items(): skip_reasons[k]=skip_reasons.get(k,0)+int(v)
    for k,v in chunk.algorithm_version.value_counts().items(): version_counts[k]=version_counts.get(k,0)+int(v)
    valid=chunk[chunk.status=='ok']
    assert np.isfinite(valid.r2_test).all()
    np.testing.assert_allclose(valid.r2_test, 1-valid.mse/valid.null_mse_train_mean, rtol=1e-10, atol=1e-10)
    assert chunk.seed.between(12345,12444).all() and chunk.draw.between(0,49).all()
    for col in ['status','dataset','algorithm_version','outcome','split_mode','n_test_total']:
        metadata.setdefault(col, set()).update(chunk[col].unique().tolist())
    for key, g in chunk.groupby(['model','N','K'], sort=False):
        ids = ((g.seed.to_numpy()-12345)*50+g.draw.to_numpy()).astype(int)
        bits = seen.setdefault(key, np.zeros(5000, dtype=bool))
        assert not bits[ids].any() and len(np.unique(ids)) == len(ids), 'duplicate repeat'
        bits[ids] = True
        v = g.loc[g.status=='ok','r2_test'].to_numpy()
        if not len(v): continue
        s = stats.setdefault(key, [0,0.,float('inf'),float('-inf'),0])
        s[0] += len(v); s[1] += v.sum(); s[2] = min(s[2],float(v.min())); s[3] = max(s[3],float(v.max())); s[4] += int((v < -.5).sum())
    total += len(chunk)
    if total % 2000000 == 0: print(f'Aggregated {total:,} / 18,000,000', file=sys.stderr, flush=True)
assert version_counts == receipt['algorithm_versions']
assert total == 18000000 and len(stats) == 3600
assert all(b.all() for b in seen.values())
records = [dict(model=k[0],N=int(k[1]),K=int(k[2]),count=s[0],r2_test=float(s[1]/s[0]),min_r2=s[2],max_r2=s[3],repeats_below_floor=s[4]) for k,s in sorted(stats.items())]
print(json.dumps(dict(source=str(source),receipt=receipt,status_counts=status_counts,skip_reasons=skip_reasons,rows=total,unique_repeats_verified=True,metadata={k:sorted(v) for k,v in metadata.items()},cells=records)))

