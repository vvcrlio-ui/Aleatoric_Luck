"""Fresh panel run using the deployed direct-success single-model dispatcher.

The runtime is supplied explicitly and hash-bound, never downloaded implicitly.
No legacy result is imported across an algorithm change.
"""
import argparse
import csv
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import subprocess

from aleatoric_nk_grid.shared_queue import atomic_json, digest, file_digest, QueueError


def prepare(args):
    import numpy as np
    from aleatoric_nk_grid import nk_grid as nk
    from aleatoric_nk_grid.execution_contract import CellExecutionSpec
    from aleatoric_nk_grid.pending_resume import Design
    from aleatoric_nk_grid.run_panels import resolved_panels
    from aleatoric_nk_grid.scheduler_cost import CostEstimator

    if args.root.exists():
        raise QueueError('Fresh output directory already exists')
    commit = subprocess.check_output(['git', '-C', str(args.repo), 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', '-C', str(args.repo), 'status', '--porcelain'], text=True).strip():
        raise QueueError('Clean committed checkout required')
    _, config = resolved_panels(args.repo/'SMR/panels.yaml', {args.panel}, preset='timing_full')[0]
    config = replace(config, schema=args.schema, out=args.root/'unused.csv', n_jobs=1)
    if args.probe:
        config = replace(config, n_grid=(10, 100), k_grid=(10, 50), n_sizes_n=2, n_sizes_k=2)
    with nk.NKGridExecutionSession.open_from_config(config) as session:
        spec = CellExecutionSpec.from_config(config, repo_root=args.repo, panel_id=args.panel,
            resolved_n_grid=session.n_grid, resolved_k_grid=session.k_grid,
            resolved_repeat_plan=session.repeat_pairs, model_n_jobs=1, git_commit=commit,
            algorithm_version=session.algorithm_version,
            resolved_model_params=nk.resolved_model_params(session.selected_model_params),
            environment_overrides=nk.model_run_settings(config.models),
            execution_groups=[{'k_features': int(k), 'groups': [{'group': g, 'models': list(ms)}
                for g, ms in nk.execution_groups_for_models(config.models)]} for k in session.k_grid],
            input_provenance=nk._frozen_input_provenance_for_schema(session.schema), require_clean_worktree=True)
    payload = spec.to_payload()
    design = Design(payload)
    if not args.probe and design.count != 3600:
        raise QueueError('Unexpected timing_full grid; expected 3600 single-model tasks')
    estimator = CostEstimator()
    tasks = [(k, n, seed, draw, model) for k in design.ks for n in design.ns
        for seed, draw in design.repeats for model in design.models]
    costs = [estimator.estimate(m, n, k) for k, n, s, d, m in tasks]
    # Expensive tasks start first; every worker can take any model.
    order = np.argsort(-np.asarray(costs), kind='stable').astype('<u4')
    args.root.mkdir(parents=True)
    order.tofile(args.root/'remaining.u32')
    manifest = {'format': 'direct-success-bitmap-v1', 'identity': {'cell_spec': payload,
        'runtime_sha256': file_digest(args.runtime)}, 'count': len(tasks),
        'remaining_sha256': file_digest(args.root/'remaining.u32'),
        'lease_seconds': 300., 'max_attempts': 3, 'fresh': True, 'probe': args.probe}
    atomic_json(args.root/'manifest.json', manifest)
    atomic_json(args.root/'queue-id.json', {'queue_id': digest(manifest)})
    atomic_json(args.root/'prepared.json', {'tasks': len(tasks), 'source_commit': commit,
        'N': payload['resolved_n_grid'], 'K': payload['resolved_k_grid'], 'probe': args.probe})


def finalize(root):
    from aleatoric_nk_grid import nk_grid as nk
    from aleatoric_nk_grid.pending_resume import Design
    from aleatoric_nk_grid.result_migration import validate_scientific_result

    manifest = json.loads((root/'manifest.json').read_text())
    spec = manifest['identity']['cell_spec']
    design = Design(spec)
    rows = {}
    with (root/'results.jsonl').open() as handle:
        for line in handle:
            entry = json.loads(line)
            row = entry['result']
            if entry['origin']['queue_id'] != digest(manifest):
                raise QueueError('Result queue identity mismatch')
            if row['algorithm_version'] != spec['algorithm_version']:
                raise QueueError('Result algorithm mismatch')
            ordinal = design.ordinal(row)
            if not validate_scientific_result(row, task_kind='regression'):
                raise QueueError('Failed or invalid scientific result')
            if ordinal in rows:
                raise QueueError('Duplicate result key')
            rows[ordinal] = row
    if len(rows) != design.count:
        raise QueueError('Incomplete design')
    header = nk.public_result_columns('regression')
    temporary = root/'final.csv.tmp'
    with temporary.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for ordinal in sorted(rows):
            writer.writerow(nk.project_public_result(rows[ordinal], header=header))
    temporary.replace(root/'final.csv')
    atomic_json(root/'verified.json', {'complete': True, 'rows': len(rows),
        'algorithm_version': spec['algorithm_version'], 'queue_id': digest(manifest),
        'final_csv_sha256': file_digest(root/'final.csv'),
        'results_sha256': file_digest(root/'results.jsonl'),
        'nonconverged_results': sum(row.get('converged') is False for row in rows.values())})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'run', 'finalize'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--schema', type=Path)
    p.add_argument('--panel', choices=['smr_hourlywage', 'smr_totalincome'])
    p.add_argument('--workers', type=int, default=63)
    p.add_argument('--probe', action='store_true')
    args = p.parse_args()
    if args.command == 'prepare':
        prepare(args)
    elif args.command == 'finalize':
        finalize(args.root)
    else:
        manifest = json.loads((args.root/'manifest.json').read_text())
        if file_digest(args.runtime) != manifest['identity']['runtime_sha256']:
            raise QueueError('Deployed runtime changed')
        module_spec = importlib.util.spec_from_file_location('direct_flat', args.runtime)
        runtime = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(runtime)
        runtime.run(args.root, args.repo, args.repo, args.workers, validate_only=True)
        finalize(args.root)


if __name__ == '__main__':
    main()
