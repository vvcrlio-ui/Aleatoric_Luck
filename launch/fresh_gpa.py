"""Fresh FFC GPA design on the recoverable direct-success dispatcher."""
import argparse
import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import uuid

from aleatoric_nk_grid import direct_success_queue as runtime
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
    _, config = resolved_panels(args.repo/'FFCWS/panels.yaml', {args.panel}, preset=args.preset)[0]
    config = replace(config, schema=args.schema, out=args.root/'final.csv', n_jobs=1,
                     allow_large_run=args.allow_large_run)
    if args.preset == 'production' and not args.allow_large_run:
        raise QueueError('Production requires --allow-large-run')
    if args.probe:
        config = replace(config, n_grid=(122,), k_grid=(47,), n_sizes_n=1, n_sizes_k=1,
                         n_seeds=1, n_draws=1, repeat_plan=None)
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
    expected = 9 if args.probe else (18000000 if args.preset == 'production' else 3600)
    if design.count != expected:
        raise QueueError(f'Unexpected design: {design.count}; expected {expected}')
    if design.count >= 2**32:
        raise QueueError('Design exceeds uint32 ordinal range')
    estimator = CostEstimator()
    # Sort the small K/N/model table, then stream each group of repeat ordinals.
    # Avoid constructing 18 million Python tuples or a full float cost vector.
    groups = [(estimator.estimate(m,n,k),ki,ni,mi) for ki,k in enumerate(design.ks)
              for ni,n in enumerate(design.ns) for mi,m in enumerate(design.models)]
    groups.sort(key=lambda item:-item[0])
    args.root.mkdir(parents=True, exist_ok=False)
    with (args.root/'remaining.u32').open('xb') as handle:
        for _,ki,ni,mi in groups:
            base = (ki*len(design.ns)+ni)*len(design.repeats)*len(design.models)+mi
            order = base + np.arange(len(design.repeats),dtype='<u4')*len(design.models)
            handle.write(order.astype('<u4',copy=False).tobytes())
        handle.flush(); os.fsync(handle.fileno())
    manifest = {'format':'direct-success-bitmap-v1', 'identity':{'cell_spec':payload,
        'runtime_sha256':file_digest(Path(runtime.__file__))}, 'count':design.count,
        'remaining_sha256':file_digest(args.root/'remaining.u32'),
        'lease_seconds':300., 'max_attempts':5, 'fresh':True, 'probe':args.probe,
        'panel':args.panel, 'preset':args.preset}
    atomic_json(args.root/'manifest.json',manifest)
    atomic_json(args.root/'queue-id.json',{'queue_id':digest(manifest)})
    atomic_json(args.root/'prepared.json',{'tasks':design.count,'source_commit':commit,
        'N':payload['resolved_n_grid'],'K':payload['resolved_k_grid'],
        'repeats':len(design.repeats),'models':list(design.models),'probe':args.probe,
        'root':str(args.root),'final_csv':str(args.root/'final.csv')})
    print((args.root/'prepared.json').read_text(),flush=True)


def finalize(root):
    from aleatoric_nk_grid import nk_grid as nk
    from aleatoric_nk_grid.pending_resume import Design
    from aleatoric_nk_grid.result_migration import validate_scientific_result
    manifest = json.loads((root/'manifest.json').read_bytes())
    spec = manifest['identity']['cell_spec']; qid = digest(manifest)
    design = Design(spec); count = nonconverged = 0
    final = root/'final.csv'
    if final.exists(): raise QueueError('Final CSV already exists')
    temporary = root/'final.incomplete.csv'
    header = nk.public_result_columns('regression')
    # Full coverage/duplicate checking takes one bit per result; no 18M-row dict.
    with temporary.open('x',newline='',encoding='utf-8') as out:
        writer=csv.DictWriter(out,fieldnames=header,lineterminator='\n'); writer.writeheader()
        with (root/'results.jsonl').open('rb') as source:
            for line in source:
                entry=json.loads(line); row=entry['result']; ordinal=design.ordinal(row)
                if entry['origin']['queue_id'] != qid or entry['task_id'] != runtime.task_at(design,ordinal).id:
                    raise QueueError('Result provenance/key mismatch')
                if row['algorithm_version'] != spec['algorithm_version'] or not validate_scientific_result(row,task_kind='regression'):
                    raise QueueError('Invalid result or scientific identity')
                if design.contains(ordinal): raise QueueError('Duplicate result key')
                design.mark(ordinal); count += 1
                nonconverged += row.get('converged') is False
                writer.writerow(nk.project_public_result(row,header=header))
        if count != design.count: raise QueueError('Incomplete design')
        out.flush(); os.fsync(out.fileno())
    os.replace(temporary,final)
    atomic_json(root/'verified.json',{'complete':True,'rows':count,'panel':manifest['panel'],
        'algorithm_version':spec['algorithm_version'],'queue_id':qid,
        'final_csv_sha256':file_digest(final),'results_sha256':file_digest(root/'results.jsonl'),
        'nonconverged_results':nonconverged})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','run','finalize'])
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--root',type=Path)
    p.add_argument('--schema',type=Path)
    p.add_argument('--panel',choices=['ffc_median_mode_gpa','ffc_median_missing_indicator_gpa'],
                   default='ffc_median_missing_indicator_gpa')
    p.add_argument('--preset',choices=['production','timing_full'],default='production')
    p.add_argument('--allow-large-run',action='store_true')
    p.add_argument('--probe',action='store_true')
    p.add_argument('--workers',type=int,default=21)
    args=p.parse_args(); args.repo=args.repo.expanduser().resolve()
    if args.root is None:
        if args.command != 'prepare': p.error('run/finalize require the prepared --root')
        args.root=args.repo/'FFCWS'/'outputs'/(args.panel+'-'+uuid.uuid4().hex[:12])
    args.root=args.root.expanduser().resolve()
    if args.command == 'prepare':
        if args.schema is None: p.error('prepare requires --schema')
        args.schema=args.schema.expanduser().resolve(); prepare(args)
    elif args.command == 'finalize': finalize(args.root)
    else:
        manifest=json.loads((args.root/'manifest.json').read_bytes())
        if file_digest(Path(runtime.__file__)) != manifest['identity']['runtime_sha256']:
            raise QueueError('Deployed dispatcher changed')
        runtime.run(args.root,args.repo,args.repo,args.workers,validate_only=True)
        import gc
        gc.collect()
        finalize(args.root)


if __name__=='__main__': main()
