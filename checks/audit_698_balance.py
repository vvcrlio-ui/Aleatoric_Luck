"""Read existing production audit snapshots; no new training or cluster writes."""
from pathlib import Path
from datetime import datetime
from collections import Counter
import json
import math
import statistics

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL=ROOT.parents[2]
AUDITS=ORIGINAL/'runs/ffc-gpa-production-20260909'
OUT=ROOT.parent/'balance-698'


def quantile(values,p):
    ordered=sorted(values);at=(len(ordered)-1)*p
    low=math.floor(at);high=math.ceil(at)
    return ordered[low]*(high-at)+ordered[high]*(at-low) if high!=low else ordered[low]


def main():
    newer=AUDITS/'full-audit-20260910T210700Z';older=AUDITS/'full-audit-20260910T200630Z'
    new_summary=json.loads((newer/'summary.json').read_bytes())
    old_summary=json.loads((older/'summary.json').read_bytes())
    interval=(datetime.fromisoformat(new_summary['frontier_captured_to_utc'])-
              datetime.fromisoformat(old_summary['frontier_captured_to_utc'])).total_seconds()
    new={r['worker']:r for r in map(json.loads,(newer/'workers.jsonl').read_bytes().splitlines())}
    old={r['worker']:r for r in map(json.loads,(older/'workers.jsonl').read_bytes().splitlines())}
    groups={}
    for category in ('passthrough2','imputed7'):
        rows=[r for r in new.values() if ('xgboost' in r['by_model'])==(category=='passthrough2')]
        rates=[];details=[]
        for row in rows:
            previous=old[row['worker']]
            delta=row['completed_tasks']-previous['completed_tasks']
            # Require unfinished assignments at BOTH snapshots. A worker that
            # finishes early has less than one interval of active work.
            if delta>0 and row['completed_tasks']<row['assigned_tasks'] and previous['completed_tasks']<previous['assigned_tasks']:
                rates.append(interval/delta)
                details.append({'worker':row['worker'],'completed_group_delta':delta,
                    'seconds_per_group_over_interval':interval/delta,
                    'last_completed_cell':row['last_completed_cell']})
        groups[category]={'workers':len(rows),'assignments_finished':sum(r['completed_tasks']==r['assigned_tasks'] for r in rows),
            'assignments_unfinished':sum(r['completed_tasks']<r['assigned_tasks'] for r in rows),
            'groups_remaining':sum(r['assigned_tasks']-r['completed_tasks'] for r in rows),
            'steady_worker_samples':len(rates),'seconds_per_group_p10':quantile(rates,.1),
            'seconds_per_group_median':statistics.median(rates),'seconds_per_group_p90':quantile(rates,.9),
            'rate_details':details}
    total_models=18000000
    report={'snapshot_utc':new_summary['frontier_captured_to_utc'],'prior_snapshot_utc':old_summary['frontier_captured_to_utc'],
        'interval_seconds':interval,'workers':698,'groups':groups,'valid_results':new_summary['totals']['valid_results'],
        'missing_or_failed_model_results':total_models-new_summary['totals']['valid_results'],
        'source_files':[str(newer/'summary.json'),str(newer/'workers.jsonl'),str(older/'summary.json'),str(older/'workers.jsonl')],
        'caveat':'Rates are interval-averaged completed legacy groups, not individual model runtimes; future N/K mix changes. No production runtime extrapolation.'}
    OUT.mkdir(exist_ok=True)
    (OUT/'snapshot-analysis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({**report,'groups':{k:{a:b for a,b in v.items() if a!='rate_details'} for k,v in groups.items()}},ensure_ascii=False))


if __name__=='__main__':main()
