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
    from . import nk_grid as nk
    from .execution_contract import CellExecutionSpec
    root = Path(launch['output'])
    with nk.NKGridExecutionSession.open_from_config(config) as session:
        task_kind = session.task
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
    path = root / 'plan.json'
    if path.exists():
        if read(path) != plan:
            raise QueueError('Prepared plan changed')
    else:
        atomic_json(path, plan)
    return path


def scan(plan, rounds, accept=lambda row: None):
    """Validate durable receipts with bounded key memory, including partial tails.

    The caller also checks Slurm termination. OS locks fence numerical owners
    and protect against a concurrent manually started allocation.
    """
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
        return saved
    directory.parent.mkdir(parents=True, exist_ok=True)
    stage = directory.with_name('.' + directory.name + '-' + uuid.uuid4().hex)
    stage.mkdir()
    estimator = CostEstimator(profile=cost_profile)
    groups = [(estimator.estimate(m, n, k), ki, ni, mi)
        for ki, k in enumerate(design.ks) for ni, n in enumerate(design.ns)
        for mi, m in enumerate(design.models)]
    groups.sort(key=lambda item: -item[0])
    count = 0; work = 0.
    with (stage / 'remaining.u32').open('xb') as handle:
        for cost, ki, ni, mi in groups:
            base = (ki * len(design.ns) + ni) * len(design.repeats) * len(design.models) + mi
            chunk = array('I', (base + ri * len(design.models) for ri in range(len(design.repeats))
                               if not design.contains(base + ri * len(design.models))))
            count += len(chunk); work += cost * len(chunk)
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
             'work_seconds': work if cost_profile else None}
    atomic_json(stage / 'manifest.json', manifest)
    atomic_json(stage / 'queue-id.json', {'queue_id': digest(manifest)})
    atomic_json(stage / 'prepared.json', saved)
    stage.replace(directory)
    return saved


def finalize(plan, rounds):
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
