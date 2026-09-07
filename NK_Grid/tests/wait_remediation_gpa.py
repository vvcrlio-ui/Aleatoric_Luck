"""Finish the ledger after already-running bounded GPA drivers finish.

Does not launch fits, submit jobs, change code, or commit. Incomplete/failed
drivers stay explicitly unaccepted. This is a completion step, not a monitor
for future experiments.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input', type=Path, nargs='+', required=True)
parser.add_argument('--timeout-hours', type=float, default=8)
args = parser.parse_args()
started = time.monotonic()
evidence = ROOT / 'docs/remediation_evidence'
status_path = evidence / 'GPA-completion-status.json'

def status(state, counts):
    status_path.write_text(json.dumps({'state':state, 'observed_and_expected':counts,
        'time_utc':datetime.now(timezone.utc).isoformat(),
        'note':'Only waits for the existing prespecified runs; does not launch new computation.'},indent=2)+'\n',encoding='utf-8')

status('waiting', [])
while True:
    counts = []
    for path in args.input:
        try:
            rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            identity = rows[0]['identity']
            expected = 2 * len(identity['seeds']) * len(identity['draws']) * len(identity['k']) * len(identity['n'])
            counts.append({'path':str(path), 'observed':len(rows)-1, 'expected':expected})
        except (OSError, ValueError, KeyError, IndexError):
            counts.append({'path':str(path), 'observed':None, 'expected':None})
    complete = all(c['observed'] is not None and c['observed'] == c['expected'] for c in counts)
    expired = time.monotonic() - started > args.timeout_hours * 3600
    if complete or expired:
        break
    time.sleep(30)

status('all_cells_recorded' if complete else 'incomplete_timeout', counts)
subprocess.run([sys.executable, str(ROOT/'NK_Grid/tests/summarize_remediation_gpa.py'),
    '--input', *map(str,args.input), '--output', str(evidence/'GPA-paired-summary.json')],cwd=ROOT,check=True)
subprocess.run([sys.executable, str(ROOT/'docs/write_remediation_acceptance.py')],cwd=ROOT,check=True)
raise SystemExit(0 if complete else 1)
