"""Aggregate per-cell mean, P95, min and max of test MSE from the verified MH final.csv.

With a fixed null MSE c, R-squared = 1 - MSE / c is a decreasing linear map of MSE, so the mean,
P5 and maximum of R-squared follow from the mean, P95 and minimum of MSE.
"""
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
full_train_null, test_variance = set(), set()
total = 0
csv.field_size_limit(1 << 24)
with SOURCE.open(newline='') as fh:
    reader = csv.reader(fh)
    header = next(reader)
    idx = {name: i for i, name in enumerate(header)}
    im, inn, ik, imse, ist, intr, inull, ivar = [idx[k] for k in (
        'model', 'N', 'K', 'mse', 'status', 'n_train_total', 'null_mse_train_mean', 'test_target_variance')]
    for row in reader:
        total += 1
        assert row[ist] == 'ok', row[ist]
        key = (row[im], int(row[inn]), int(row[ik]))
        if key not in values:
            values[key] = array('d')
        values[key].append(float(row[imse]))
        test_variance.add(row[ivar])
        # At N equal to the whole training set, the null is the full-training-mean null.
        if row[inn] == row[intr]:
            full_train_null.add(row[inull])
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
    cells.append(dict(model=model, N=n, K=k, count=len(ordered), mean_mse=math.fsum(ordered) / len(ordered),
                      p95_mse=quantile(ordered, 0.95), min_mse=ordered[0], max_mse=ordered[-1]))
payload = dict(source=str(SOURCE), receipt=receipt, rows=total, cells=cells,
               full_train_null_mse=sorted(full_train_null), test_target_variance=sorted(test_variance),
               aggregation='Per model/N/K cell over all ok repeats: exact mean of mse (math.fsum); '
                           'P95 by linear interpolation (numpy default); min; max',
               elapsed_seconds=time.monotonic() - started)
OUT.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
print('Wrote {} cells to {} in {:.1f}s'.format(len(cells), OUT, time.monotonic() - started), flush=True)
