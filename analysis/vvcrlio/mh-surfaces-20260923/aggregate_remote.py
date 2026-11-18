"""Aggregate the already-verified MH CSV for plotting; no rehash or revalidation."""
import csv
import json
from pathlib import Path
import sys
import time

RUN = Path('/valhalla/projects/ehpc-dev-2026d08-299/aleatoric_luck-mh-8ba797e/runs/prod-mh-6h3g-b')
OUT = Path(sys.argv[1])
SOURCE = RUN / 'final.csv'
started = time.monotonic()
receipt = json.loads((RUN / 'verified.json').read_text())
assert receipt['complete']
stats, metadata, status_counts, skip_reasons = {}, {}, {}, {}
total = 0
csv.field_size_limit(1 << 24)
with SOURCE.open(newline='') as fh:
    reader = csv.reader(fh)
    header = next(reader)
    idx = {name: i for i, name in enumerate(header)}
    print('Columns:', header, flush=True)
    meta_idx = [(name, idx[name]) for name in ('dataset', 'outcome', 'split_mode',
        'algorithm_version', 'n_test_total', 'panel_id', 'variant_id', 'base_library_id') if name in idx]
    for name, _ in meta_idx:
        metadata[name] = set()
    im, inn, ik, ir, ist = [idx[k] for k in ('model','N','K','r2_test','status')]
    for row in reader:
        total += 1
        status = row[ist]
        status_counts[status] = status_counts.get(status, 0) + 1
        key = (row[im], int(row[inn]), int(row[ik]))
        if key not in stats:
            stats[key] = [0, 0.0, 0.0, float('inf'), float('-inf'), 0]
        s = stats[key]
        if s[0] == 0:
            for name, i in meta_idx:
                metadata[name].add(row[i])
        if status == 'ok':
            r2 = float(row[ir])
            adjusted = r2 - s[2]
            new_sum = s[1] + adjusted
            s[2] = (new_sum - s[1]) - adjusted
            s[1] = new_sum
            s[0] += 1
            s[3] = min(s[3], r2)
            s[4] = max(s[4], r2)
            s[5] += r2 < -0.5
        elif status == 'skipped':
            reason = row[idx['error']]
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
        if total % 2000000 == 0:
            print('Aggregated {:,} rows, {:.1f}s'.format(total,time.monotonic()-started),flush=True)

cells = [dict(model=k[0],N=k[1],K=k[2],count=s[0],r2_test=s[1]/s[0],
    min_r2=s[3],max_r2=s[4],repeats_below_floor=s[5]) for k,s in sorted(stats.items()) if s[0]]
payload = dict(source=str(SOURCE),receipt=receipt,status_counts=status_counts,
    skip_reasons=skip_reasons,rows=total,metadata={k:sorted(v) for k,v in metadata.items()},
    cells=cells,validation='Existing production verified.json reused; no rehash or duplicate-key rescan',
    aggregation='Arithmetic mean using compensated summation; no clipping before averaging',
    elapsed_seconds=time.monotonic()-started)
OUT.parent.mkdir(parents=True,exist_ok=True)
OUT.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
print('Wrote {} cells to {}'.format(len(cells),OUT),flush=True)
