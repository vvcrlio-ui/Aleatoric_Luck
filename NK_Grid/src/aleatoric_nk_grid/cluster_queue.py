"""Panel-independent preparation and verified publication for the flat dispatcher.

Each round has its own immutable queue and append-only receipts. Only stopped
rounds are scanned; successful keys are excluded from every subsequent round.
"""
from array import array
from contextlib import ExitStack
import csv
import json
import os
from pathlib import Path
import sys
import uuid

from . import direct_success_queue as runtime
from .pending_resume import Design
from .scheduler_cost import CostEstimator
from .shared_queue import (QueueError, atomic_json, digest, file_digest, file_lock,
                           transport_manifest)
from .result_migration import validate_scientific_result


def read(path):
    return json.loads(Path(path).read_bytes())


def prepare(config, launch, repo):
    from .prediction_contract import normalize_prediction_options
    options, execution = normalize_prediction_options(getattr(config, 'prediction_cache', None),
                                                      getattr(config, 'execution', None))
    if options and execution.get('workflow') != 'base_then_sl':
        raise QueueError('Cluster prediction cache requires base_then_sl; capture is a numerical-session API only')
    from . import nk_grid as nk
    from .execution_contract import CellExecutionSpec
    root = Path(launch['output'])
    with nk.NKGridExecutionSession.open_from_config(config) as session:
        task_kind = session.task
        prediction_dimensions = None
        if (getattr(config, 'execution', None) or {}).get('workflow') == 'base_then_sl':
            frames = [session.frame] + ([session.external_frame] if session.external_frame is not None else [])
            max_id_width = max((len(str(value)) * 4 for frame in frames
                for value in frame[session.schema.id_column]), default=4)
            prediction_dimensions = {
                'n_test_max': max(len(session.split_manager.for_seed(seed).test_index)
                                  for seed in {seed for seed, _ in session.repeat_pairs}),
                'feature_map_bytes': sum(len(str(name)) * 4 + 16 for name in
                                         (*session.predictors, *session.feature_units)),
                'max_id_width': max_id_width,
                'sample_index_width': max((len(str(value)) * 4 for frame in frames for value in frame.index), default=4)}
        spec = CellExecutionSpec.from_config(config, repo_root=repo, panel_id=launch['panel'],
            resolved_n_grid=session.n_grid, resolved_k_grid=session.k_grid,
            resolved_repeat_plan=session.repeat_pairs, model_n_jobs=1,
            git_commit=launch['source']['commit'], algorithm_version=session.algorithm_version,
            resolved_model_params=nk.resolved_model_params(session.selected_model_params),
            environment_overrides=nk.model_run_settings(config.models),
            execution_groups=[{'k_features': int(k), 'groups': [{'group': g, 'models': list(ms)}
                for g, ms in nk.execution_groups_for_models(config.models)]} for k in session.k_grid],
            input_provenance=nk._frozen_input_provenance_for_schema(session.schema), require_clean_worktree=True)
    payload = spec.to_payload()
    if Design(payload).count >= 2**32:
        raise QueueError('Design exceeds uint32 task ordinal range')
    environment = read(root / 'cluster-environment.json')
    plan = {'format': 'single-model-slurm-v1', 'cell_spec': payload,
            'runtime_sha256': file_digest(Path(runtime.__file__)),
            'launch': {**launch, 'worker_environment': environment},
            'submission': launch['cluster'], 'task_kind': task_kind, 'public_columns': nk.public_result_columns(task_kind),
            'checkpoint_retention': config.checkpoint_retention}
    from .prediction_workflow import contract_from_config
    if prediction_dimensions is not None:
        plan['prediction_dimensions'] = prediction_dimensions
    workflow = contract_from_config(config, plan)
    if workflow is not None:
        plan['prediction_workflow'] = workflow
        from .prediction_cache import initialize_cache
        initialize_cache(workflow['cache_root'], {'workflow_sha256': digest(workflow),
            'plan_sha256': digest(plan), 'contract': workflow})
        from .prediction_maps import prepare_maps
        prepare_maps(plan, repo_root=repo)
    path = root / 'plan.json'
    if path.exists():
        if read(path) != plan:
            raise QueueError('Prepared plan changed')
    else:
        atomic_json(path, plan)
    return path


def prepare_joint(plans, launch, *, phase_round_limits, sl_resources):
    """Freeze one multi-panel submission and one barrier, without submitting.

    Inputs are already-frozen single-panel plans, not running queues. The new
    output identity is independent and never mutates their code, caches or jobs.
    """
    from .prediction_workflow import joint_contract, validate_round_time_limits
    from .prediction_cache import initialize_cache
    plans = [read(p) if isinstance(p, (str, Path)) else p for p in plans]
    if not plans or any(p.get('format') != 'single-model-slurm-v1' or not p.get('prediction_workflow') for p in plans):
        raise QueueError('Joint prediction submission needs frozen two-phase panel plans')
    first = plans[0]
    if any(p['runtime_sha256'] != first['runtime_sha256'] or
           p['cell_spec'].get('git_commit') != first['cell_spec'].get('git_commit') for p in plans):
        raise QueueError('Joint panels must use the same frozen runtime/code')
    root = Path(launch['output']).resolve()
    contract = joint_contract([p['prediction_workflow'] for p in plans], output_root=root,
                             phase_round_limits=phase_round_limits, sl_resources=sl_resources)
    validate_round_time_limits(contract, global_time_limit=launch['cluster']['time_limit'])
    dimensions = [p.get('prediction_dimensions', {}) for p in plans]
    if any(not d for d in dimensions): raise QueueError('Every joint panel requires storage dimensions')
    merged = {}
    for original in plans:
        dims = original['prediction_dimensions']
        for panel in original['prediction_workflow']['panels']:
            merged[panel['panel_id']] = dims.get(panel['panel_id'], dims)
    plan = {'format': 'single-model-slurm-v1', 'cell_spec': first['cell_spec'],
        'runtime_sha256': first['runtime_sha256'], 'launch': launch, 'submission': launch['cluster'],
        'task_kind': first['task_kind'], 'checkpoint_retention': 'keep',
        'public_columns': list(dict.fromkeys(column for p in plans for column in p['public_columns'])),
        'prediction_workflow': contract, 'prediction_dimensions': merged,
        'source_panel_plan_sha256': [digest(p) for p in plans]}
    root.mkdir(parents=True, exist_ok=True); path = root / 'plan.json'
    if path.exists() and read(path) != plan: raise QueueError('Joint frozen plan changed')
    initialize_cache(contract['cache_root'], {'workflow_sha256': digest(contract),
        'plan_sha256': digest(plan), 'contract': contract})
    from .prediction_maps import prepare_maps
    prepare_maps(plan, repo_root=launch.get('source', {}).get('root', Path.cwd()))
    if not path.exists(): atomic_json(path, plan)
    return path


def scan(plan, rounds, accept=lambda row: None):
    """Validate durable receipts with bounded key memory, including partial tails.

    The caller also checks Slurm termination. OS locks fence numerical owners
    and protect against a concurrent manually started allocation.
    """
    from .prediction_workflow import reject_unphased_cache
    reject_unphased_cache(plan['cell_spec'])
    design = Design(plan['cell_spec'])
    sources = []
    with ExitStack() as locks:
        for directory in map(Path, rounds):
            locks.enter_context(file_lock(directory / 'dispatcher.lock'))
            manifest = read(directory / 'manifest.json')
            qid = digest(manifest)
            if (read(directory / 'queue-id.json')['queue_id'] != qid
                    or manifest['identity']['cell_spec'] != plan['cell_spec']
                    or manifest['identity']['plan_sha256'] != digest(plan)
                    or manifest.get('sources') != sources):
                raise QueueError('Round identity changed')
            if file_digest(directory / 'remaining.u32') != manifest['remaining_sha256']:
                raise QueueError('Round task list changed')
            order = array('I'); order.frombytes((directory / 'remaining.u32').read_bytes())
            if sys.byteorder != 'little': order.byteswap()
            if len(order) != manifest['count']:
                raise QueueError('Round task count changed')
            membership = bytearray(len(design.bits))
            for ordinal in order:
                if ordinal >= design.count or design.contains(ordinal):
                    raise QueueError('Round overlaps already successful keys')
                byte, shift = divmod(ordinal, 8)
                if membership[byte] & (1 << shift): raise QueueError('Duplicate round task')
                membership[byte] |= 1 << shift
            del order
            source = directory / 'results.jsonl'
            if not source.exists():
                sources.append({'root': str(directory), 'queue_id': qid, 'results_sha256': None})
                continue  # Allocation failed before dispatcher creation.
            size = source.stat().st_size
            with source.open('rb') as handle:
                while line := handle.readline(2 * 1024 * 1024 + 1):
                    if len(line) > 2 * 1024 * 1024: raise QueueError('Oversized result')
                    if not line.endswith(b'\n'): break  # Unacknowledged interrupted write.
                    entry = json.loads(line); row = entry['result']; ordinal = design.ordinal(row)
                    if (entry['origin']['queue_id'] != qid
                            or entry['task_id'] != runtime.task_at(design, ordinal).id
                            or not membership[ordinal // 8] & (1 << (ordinal % 8))):
                        raise QueueError('Result provenance/key mismatch')
                    if not validate_scientific_result(row, task_kind=plan.get('task_kind', 'regression')): continue
                    if row.get('algorithm_version') != plan['cell_spec']['algorithm_version']:
                        raise QueueError('Result algorithm changed')
                    if design.contains(ordinal): raise QueueError('Duplicate successful result')
                    design.mark(ordinal); accept(row)
            if source.stat().st_size != size: raise QueueError('Stopped results changed')
            sources.append({'root': str(directory), 'queue_id': qid, 'results_sha256': file_digest(source)})
    return design, sources


def prepare_round(plan, rounds, directory, *, cost_profile=None):
    directory = Path(directory)
    design, sources = scan(plan, rounds)
    done = sum(b.bit_count() for b in design.bits)
    if done == design.count:
        return {'done': done, 'remaining': 0}
    if directory.exists():
        saved = read(directory / 'prepared.json')
        manifest = read(directory / 'manifest.json')
        if (manifest['identity']['plan_sha256'] != digest(plan)
                or manifest['sources'] != sources or saved['done'] != done
                or saved['queue_id'] != digest(manifest)
                or read(directory / 'queue-id.json')['queue_id'] != digest(manifest)
                or file_digest(directory / 'remaining.u32') != manifest['remaining_sha256']):
            raise QueueError('Prepared round changed')
        # Preparation can precede live admission by hours (or a controller
        # restart). Reprice the frozen ordinal list using this round's current
        # operational profile; never reorder it or rewrite its queue identity.
        work = None; coverage = None
        if cost_profile is not None:
            estimator = CostEstimator(profile=cost_profile)
            order = array('I'); order.frombytes((directory / 'remaining.u32').read_bytes())
            if sys.byteorder != 'little': order.byteswap()
            counts = {}
            for ordinal in order:
                if ordinal >= design.count:
                    raise QueueError('Prepared task ordinal exceeds design')
                group, mi = divmod(ordinal, len(design.models))
                group //= len(design.repeats)
                ki, ni = divmod(group, len(design.ns))
                key = (design.models[mi], design.ns[ni], design.ks[ki])
                counts[key] = counts.get(key, 0) + 1
            means = [(estimator.mean_seconds(*key), count) for key, count in counts.items()]
            if all(mean is not None for mean, _ in means):
                work = sum(mean * count for mean, count in means)
            coverage = estimator.coverage(counts)
        return {**saved, 'work_seconds': work,
                'cost_profile_coverage': coverage,
                'cost_profile_sha256': digest(cost_profile) if cost_profile is not None else None}
    directory.parent.mkdir(parents=True, exist_ok=True)
    stage = directory.with_name('.' + directory.name + '-' + uuid.uuid4().hex)
    stage.mkdir()
    estimator = CostEstimator(profile=cost_profile)
    fallback = CostEstimator()
    groups = []
    for ki, k in enumerate(design.ks):
        for ni, n in enumerate(design.ns):
            for mi, model in enumerate(design.models):
                mean = estimator.mean_seconds(model, n, k)
                # Relative fallback weights only order unknown groups. They are
                # never added to a seconds total or treated as a batch price.
                ordering = mean if mean is not None else fallback.estimate(model, n, k)
                groups.append((ordering, ki, ni, mi, mean))
    groups.sort(key=lambda item: -item[0])
    count = 0; work = 0.; remaining_groups = []; fully_priced = cost_profile is not None
    with (stage / 'remaining.u32').open('xb') as handle:
        for _, ki, ni, mi, mean in groups:
            base = (ki * len(design.ns) + ni) * len(design.repeats) * len(design.models) + mi
            chunk = array('I', (base + ri * len(design.models) for ri in range(len(design.repeats))
                               if not design.contains(base + ri * len(design.models))))
            count += len(chunk)
            if chunk:
                if mean is None: fully_priced = False
                else: work += mean * len(chunk)
                remaining_groups.append((design.models[mi], design.ns[ni], design.ks[ki]))
            if sys.byteorder != 'little': chunk.byteswap()
            handle.write(chunk.tobytes())
        handle.flush(); os.fsync(handle.fileno())
    if count + done != design.count: raise QueueError('Task complement mismatch')
    manifest = {'format': 'direct-success-bitmap-v1',
        'identity': {'cell_spec': plan['cell_spec'], 'plan_sha256': digest(plan),
                     'runtime_sha256': plan['runtime_sha256'], 'task_kind': plan.get('task_kind', 'regression')},
        'count': count, **transport_manifest(), 'max_attempts': 5, 'sources': sources,
        'remaining_sha256': file_digest(stage / 'remaining.u32'), 'fresh': True}
    # Only a measured profile carries seconds; the analytic estimator is a
    # relative order, so reporting its total as work would invite sizing an
    # allocation from a number that means nothing.
    saved = {'done': done, 'remaining': count, 'queue_id': digest(manifest),
             'work_seconds': work if fully_priced else None,
             'cost_profile_coverage': estimator.coverage(remaining_groups) if cost_profile is not None else None,
             'cost_profile_sha256': digest(cost_profile) if cost_profile is not None else None}
    atomic_json(stage / 'manifest.json', manifest)
    atomic_json(stage / 'queue-id.json', {'queue_id': digest(manifest)})
    atomic_json(stage / 'prepared.json', saved)
    stage.replace(directory)
    return saved


def finalize(plan, rounds):
    from .prediction_workflow import reject_unphased_cache
    reject_unphased_cache(plan['cell_spec'])
    from .nk_grid import project_public_result
    root = Path(plan['launch']['output'])
    final, receipt = root / 'final.csv', root / 'verified.json'
    if receipt.exists():
        saved = read(receipt)
        if saved['plan_sha256'] != digest(plan) or saved['final_csv_sha256'] != file_digest(final):
            raise QueueError('Published result changed')
        cleanup(plan, saved)
        return saved
    temporary = root / ('final-' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x', newline='', encoding='utf-8') as out:
        writer = csv.DictWriter(out, fieldnames=plan['public_columns'], lineterminator='\n')
        writer.writeheader()
        design, sources = scan(plan, rounds, lambda row: writer.writerow(
            project_public_result(row, header=plan['public_columns'])))
        count = sum(b.bit_count() for b in design.bits)
        if count != design.count: raise QueueError('Incomplete design; final CSV not published')
        out.flush(); os.fsync(out.fileno())
    os.replace(temporary, final)
    saved = {'complete': True, 'rows': count, 'plan_sha256': digest(plan), 'sources': sources,
             'final_csv_sha256': file_digest(final), 'panel': plan['launch']['panel']}
    atomic_json(receipt, saved)
    cleanup(plan, saved)
    return saved


def cleanup(plan, receipt):
    """Delete only this run's round checkpoints, after verified publication."""
    if plan.get('checkpoint_retention', 'default') != 'delete': return
    import shutil
    root = Path(plan['launch']['output']).resolve()
    target = root / 'rounds'
    if file_digest(root / 'final.csv') != receipt['final_csv_sha256']:
        raise QueueError('Final CSV changed before checkpoint cleanup')
    if target.is_symlink() or target.resolve() != root / 'rounds':
        raise QueueError('Checkpoint directory escapes run root')
    if target.exists(): shutil.rmtree(target)
    atomic_json(root / 'checkpoint-archive.json', {'policy': 'delete', 'complete': True,
        'plan_sha256': digest(plan), 'final_csv_sha256': receipt['final_csv_sha256']})
