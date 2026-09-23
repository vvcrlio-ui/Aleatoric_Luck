"""Adopt exact historical workflow timing and conservative RSS evidence."""
from collections import defaultdict
import argparse
import json
from pathlib import Path

from .cost_profile import CostProfileBuilder, observation_kind
from .prediction_workflow import cost_identity, cost_training_context, PredictionTask
from .shared_queue import QueueError, digest, atomic_json


def build(plan, sources):
    contract = plan['prediction_workflow']
    panels = {p['panel_id']: p for p in contract['panels']}
    builder = CostProfileBuilder(); evidence = []; memory = defaultdict(list)
    identities = {}
    contexts = {key: cost_training_context(panel['cell_spec']) for key, panel in panels.items()}
    for source in sources:
        rows = 0
        with Path(source).open('rb') as handle:
            for line in handle:
                if not line.endswith(b'\n'):
                    raise QueueError('Timing adoption requires a stopped complete journal')
                row = json.loads(line)['result']; rows += 1
                task = PredictionTask(**{k: row[k] for k in PredictionTask.__dataclass_fields__})
                old = row.get('_cost_identity')
                key = (task.phase, task.panel_id, task.pipeline_id, task.variant_id)
                if key not in identities:
                    expected = cost_identity(task, contract)
                    new = dict(expected); new.pop('cell_spec_sha256', None)
                    new['training_context_sha256'] = contexts[task.panel_id]
                    identities[key] = (expected, new, digest(new))
                expected, new, timing_identity = identities[key]
                if old != expected:
                    raise QueueError('Historical timing identity differs from its source plan')
                row = {**row, '_cost_identity': new}; builder.add(row)
                rss = row.get('_peak_rss_bytes')
                if observation_kind(row, new) == 'cold' and type(rss) is int and rss > 0:
                    memory[(timing_identity, task.N, task.K)].append(rss)
        evidence.append({'path': str(Path(source).resolve()), 'rows': rows})
    profile = builder.profile(evidence={'source_plan_sha256': digest(plan), 'sources': evidence,
        'compatibility': 'exact numerical/input/environment contract; new run and storage fields excluded',
        'stride': 1, 'limit': None})
    # Process high-water RSS is conservative but not an independently measured
    # per-task peak. Never silently use it to lower production memory requests.
    profile['memory_evidence'] = [{'training_identity': identity, 'N': n, 'K': k,
        'observations': len(values), 'max_process_rss_bytes': max(values),
        'eligible_for_lowering': False, 'reason': 'process high-water mark; no isolated peak/headroom validation'}
        for (identity, n, k), values in sorted(memory.items())]
    profile['evidence']['batch_eligible_groups'] = sum(s['observations'] >= 100 for s in profile['samples'])
    profile['evidence']['quantile_note'] = 'Reported quantiles are empirical order statistics; groups below 100 never authorize p99 batching or economic drain.'
    return profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--journal', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(json.loads(args.plan.read_bytes()), args.journal)
    atomic_json(args.output, result)
    print(json.dumps(result['evidence']))


if __name__ == '__main__':
    main()
