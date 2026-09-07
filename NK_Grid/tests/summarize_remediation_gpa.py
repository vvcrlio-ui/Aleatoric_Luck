"""Summarize every prespecified GPA cell, explicitly retaining incomplete runs."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input', type=Path, nargs='+', required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
inputs = []
cases = []
expected = set()
observed = set()
for path in args.input:
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode().splitlines()]
    identity = rows[0]['identity']
    method = identity['method']
    inputs.append({'path':str(path), 'sha256':hashlib.sha256(raw).hexdigest(), 'identity':identity})
    for seed in identity['seeds']:
        for draw in identity['draws']:
            for k in identity['k']:
                for n in identity['n']:
                    for batch in ('auto', 'full'):
                        key = (method, seed, draw, k, n, batch)
                        if key in expected:
                            raise ValueError(f'overlapping requested design: {key}')
                        expected.add(key)
    for row in rows[1:]:
        key = (method, row['seed'], row['draw'], row['K'], row['N'], row['batch'])
        if key not in expected or key in observed:
            raise ValueError(f'duplicate or out-of-design result: {key}')
        observed.add(key)
        cases.append({'method':method, **row})
groups = defaultdict(list)
pairs = defaultdict(dict)
for row in cases:
    groups[(row['method'], row['K'], row['N'], row['batch'])].append(row)
    pairs[(row['method'], row['seed'], row['draw'], row['K'], row['N'])][row['batch']] = row
aggregates = []
for key, rows in sorted(groups.items()):
    values = [row['validation_mse'] for row in rows if row['status'] == 'ok']
    aggregates.append({'method':key[0], 'K':key[1], 'N':key[2], 'batch':key[3],
        'observed':len(rows), 'failed':len(rows)-len(values),
        'mse_values':values, 'mean':statistics.mean(values) if values else None,
        'min':min(values) if values else None, 'max':max(values) if values else None})
paired = []
for key, policies in sorted(pairs.items()):
    if len(policies) != 2:
        continue
    a, b = policies['auto'], policies['full']
    for field in ('row_order_sha256','sources','model_seed','K_expanded'):
        if a[field] != b[field]:
            raise ValueError(f'unpaired design: {key} {field}')
    paired.append({'method':key[0], 'seed':key[1], 'draw':key[2], 'K':key[3], 'N':key[4],
        'status':'ok' if a['status'] == b['status'] == 'ok' else 'failed',
        'full_minus_auto_mse':b['validation_mse'] - a['validation_mse']
            if a['status'] == b['status'] == 'ok' else None})
report = {'inputs':inputs, 'expected_cells':len(expected), 'observed_cells':len(observed),
    'complete':observed == expected, 'missing_cells':sorted(expected-observed),
    'failed_cells':sum(c['status'] != 'ok' for c in cases),
    'interpretation':'training-pool validation only; auto remains default; no historical causal claim; diagnostic replay adds computation',
    'aggregates':aggregates, 'paired':paired, 'cases':cases}
args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
print(json.dumps({k:report[k] for k in ('expected_cells','observed_cells','complete','failed_cells')}))
