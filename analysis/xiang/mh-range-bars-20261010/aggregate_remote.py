"""Aggregate per-cell mean, P5, min and max (P100) of test R-squared from the verified MH final.csv."""
from array import array
import csv
import json
import math
from pathlib import Path
import sys
import time

RUN = Path('/valhalla/projects/ehpc-dev-2026d08-299/aleatoric_luck-mh-8ba797e/runs/prod-mh-6h3g-b')
OUT = Path(sys.argv[1])
SOURCE = RUN / 'final.csv'
started = time.monotonic()
receipt = json.loads((RUN / 'verified.json').read_text())
assert receipt['complete']
values = {}
total = 0
csv.field_size_limit(1 << 24)
with SOURCE.open(newline='') as fh:
    reader = csv.reader(fh)
    header = next(reader)
    idx = {name: i for i, name in enumerate(header)}
    im, inn, ik, ir, ist = [idx[k] for k in ('model', 'N', 'K', 'r2_test', 'status')]
    for row in reader:
        total += 1
        assert row[ist] == 'ok', row[ist]
        key = (row[im], int(row[inn]), int(row[ik]))
        if key not in values:
            values[key] = array('d')
        values[key].append(float(row[ir]))
        if total % 2000000 == 0:
            print('Read {:,} rows, {:.1f}s'.format(total, time.monotonic() - started), flush=True)


def quantile(ordered, q):
    """Linear interpolation between closest ranks, as numpy.quantile's default."""
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


cells = []
for (model, n, k), v in sorted(values.items()):
    ordered = sorted(v)
    cells.append(dict(model=model, N=n, K=k, count=len(ordered), r2_test=math.fsum(ordered) / len(ordered),
                      p05_r2=quantile(ordered, 0.05), min_r2=ordered[0], max_r2=ordered[-1]))
payload = dict(source=str(SOURCE), receipt=receipt, rows=total, cells=cells,
               aggregation='Per model/N/K cell over all ok repeats: exact mean (math.fsum); '
                           'P5 by linear interpolation (numpy default); min; max (P100)',
               elapsed_seconds=time.monotonic() - started)
OUT.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
print('Wrote {} cells to {} in {:.1f}s'.format(len(cells), OUT, time.monotonic() - started), flush=True)
