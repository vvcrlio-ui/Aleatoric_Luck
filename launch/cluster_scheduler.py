"""Shared single-model Slurm submission and bounded per-round continuation.

Standard-library imports suffice for login-node preview/resume. Every sbatch is
journaled before submission. A successor controller is armed before a worker
allocation is submitted; no future worker allocation is submitted in advance.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import uuid

import experiment as common
import inspect
from discoverer_continuation import Journal, Slurm, TERMINAL, controller_lock as _lock, read

FORMAT = 'single-model-slurm-v1'
DONE = {'complete', 'round_budget_exhausted', 'no_progress', 'control_budget_exhausted'}


def load(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = read(plan_path)
    if plan.get('format') != FORMAT:
        raise ValueError('Legacy grouped plan cannot be submitted by the new scheduler. '
                         'Use its frozen checkout for recovery or explicitly migrate sealed results.')
    if Path(plan['launch']['output']).resolve() != plan_path.parent:
        raise ValueError('Plan output directory changed')
    common.validate_source(plan['launch'])
    return plan


def batch_args(spec, mode, plan_path, *, dependency=None, allocation=None):
    cluster, root = spec['cluster'], Path(spec['output'])
    args = ['--account=' + cluster['account'], '--partition=' + cluster['partition'],
            '--cpus-per-task=1', '--ntasks-per-core=1', '--export=ALL', '--no-requeue',
            '--chdir=' + str(root), '--output=' + str(root / ('logs/' + mode + '-%j.out')),
            '--error=' + str(root / ('logs/' + mode + '-%j.err'))]
    qos = allocation['qos'] if allocation else cluster.get('qos')
    if qos: args.append('--qos=' + qos)
    if cluster.get('constraint') not in (None, 'none'):
        args.append('--constraint=' + cluster['constraint'])
    if dependency: args.append('--dependency=afterany:' + str(dependency))
    if allocation:
        args += ['--nodes=' + str(allocation['nodes']), '--ntasks=' + str(allocation['workers'] + 1),
                 '--ntasks-per-node=' + str(allocation['tasks_per_node']),
                 '--mem=' + str(allocation['memory_mb_per_node']) + 'M',
                 '--time=' + allocation['time_limit']]
    else:
        args += ['--nodes=1', '--ntasks=1', '--mem=' + spec['plan_memory'], '--time=' + spec['plan_time']]
    environment_path = root / 'cluster-environment.json'
    environment = spec.get('worker_environment') or (read(environment_path) if environment_path.exists() else {
        'python': str(Path(sys.executable).absolute()), 'python_module': os.environ.get('PYTHON_MODULE', '')}
    )
    return args + [str(common.ROOT / 'launch/cluster_queue.sbatch'),
                   environment['python'], str(common.ROOT / 'launch/cluster_scheduler.py'),
                   mode, str(Path(plan_path).resolve()), environment['python_module']]


def state_for(plan_path, plan):
    path = Path(plan_path).parent / 'cluster-state.json'
    if path.exists():
        state = read(path)
        if state['plan_sha256'] != common.sha256(plan_path): raise ValueError('Frozen plan changed')
    else:
        rounds = plan['launch']['cluster']['rounds']
        state = {'format': FORMAT, 'run_id': uuid.uuid4().hex, 'plan_sha256': common.sha256(plan_path),
                 'jobs': {}, 'rounds': [], 'status': 'ready', 'completed': 0, 'no_progress': 0,
                 'limits': {'max_control_jobs': 4 * rounds + 8}}
    return path, state


def scheduler(spec):
    return Slurm(spec['cluster']['account'], spec['cluster'].get('qos'))


def terminal(journal, label):
    job = journal.submit(label, [])  # Recover an accepted-but-unacknowledged intent.
    return all(s in TERMINAL for s in journal.slurm.states(job)), job


def start(plan_path, *, slurm=None):
    plan = load(plan_path); root = Path(plan_path).parent
    with _lock(root / '.cluster-state.lock'):
        path, state = state_for(plan_path, plan)
        if state['status'] in DONE: raise ValueError('Continuation is terminal: ' + state['status'])
        journal = Journal(path, state, slurm or scheduler(plan['launch']))
        journal.save()
        for label in list(state['jobs']):
            ended, job = terminal(journal, label)
            if not ended:
                print('Continuation already active: ' + job)
                return job
        label = 'C' + str(sum(k.startswith('C') for k in state['jobs']))
        job = journal.submit(label, batch_args(plan['launch'], 'control', plan_path,
                                               dependency=os.environ.get('SLURM_JOB_ID')))
        print('Single-model controller: ' + job)
        return job


def advance(plan_path, *, slurm=None, backend=None, resource_resolver=None):
    plan = load(plan_path); spec = plan['launch']; root = Path(plan_path).parent
    current = os.environ.get('SLURM_JOB_ID')
    if not current: raise ValueError('Controller requires a Slurm allocation')
    if backend is None:
        from aleatoric_nk_grid import cluster_queue as backend
    if resource_resolver is None:
        from cluster_resources import resolve as resource_resolver
    with _lock(root / '.cluster-state.lock'):
        path, state = state_for(plan_path, plan)
        if state['status'] in DONE: return state
        journal = Journal(path, state, slurm or scheduler(spec)); journal.save()
        controls = [journal.submit(label, []) for label in list(state['jobs']) if label.startswith(('C', 'G'))]
        if current not in controls:
            raise ValueError('Controller allocation is not in the submission journal')
        # Resolve ALL intents before interpreting any missing job receipt.
        active = []
        for label in list(state['jobs']):
            if label.startswith('W'):
                ended, job = terminal(journal, label)
                if not ended: active.append(job)
        if active:
            if len(active) != 1: raise ValueError('Overlapping worker allocations')
            label = 'Gwait-' + active[0]
            journal.submit(label, batch_args(spec, 'control', plan_path, dependency=active[0]))
            return state
        controls = sum(k.startswith(('C', 'G')) for k in state['jobs'])
        if controls >= state['limits']['max_control_jobs']:
            state['status'] = 'control_budget_exhausted'; journal.save(); return state
        # This guard covers failures during scanning, preparation and sbatch.
        journal.submit('Gafter-' + current, batch_args(spec, 'control', plan_path, dependency=current))
        rounds = state['rounds']
        # A prepared round without a W intent is safe to reuse after controller loss.
        previous = [r['root'] for r in rounds if r['label'] in state['jobs']]
        if (root / 'verified.json').exists():
            backend.finalize(plan, previous)
            state['status'] = 'complete'; journal.save(); return state
        index = len(previous)
        queue_root = root / 'rounds' / ('round-' + str(index))
        # Operational, not frozen: a profile beside the plan changes how many
        # workers a round asks for, never what the round computes.
        profile_path = root / 'cost-profile.json'
        profile = read(profile_path) if profile_path.exists() else None
        report = backend.prepare_round(plan, previous, queue_root, cost_profile=profile)
        state['completed'] = report['done']
        if report['remaining'] == 0:
            backend.finalize(plan, previous)
            state['status'] = 'complete'; journal.save(); return state
        if index >= spec['cluster']['rounds']:
            state['status'] = 'round_budget_exhausted'; journal.save(); return state
        if index:
            # Derive from round history; replaying a controller cannot add a strike.
            stalled = 0
            for old in reversed(rounds[:index]):
                if old['done_before'] != report['done']: break
                stalled += 1
            if stalled >= spec.get('continuation', {}).get('max_no_progress_rounds', 2):
                state['status'] = 'no_progress'; journal.save(); return state
        extra = ({'work_seconds': report.get('work_seconds')}
                 if 'work_seconds' in inspect.signature(resource_resolver).parameters else {})
        allocation = resource_resolver(spec, report['remaining'], **extra)
        item = {'index': index, 'root': str(queue_root), 'label': 'W' + str(index),
                'done_before': report['done'], 'allocation': allocation}
        if len(rounds) == index: rounds.append(item)
        else: rounds[index] = item
        state['status'] = 'running'; journal.save()
        job = journal.submit(item['label'], batch_args(spec, 'work', plan_path,
            dependency=current, allocation=allocation))
        item['job_id'] = job; journal.save()
        return state


def work(plan_path):
    from aleatoric_nk_grid import direct_success_queue as runtime
    from aleatoric_nk_grid.shared_queue import file_digest
    plan = load(plan_path); root = Path(plan_path).parent
    if file_digest(Path(runtime.__file__)) != plan['runtime_sha256']:
        raise ValueError('Frozen dispatcher changed')
    with _lock(root / '.cluster-state.lock'):
        path, state = state_for(plan_path, plan)
        journal = Journal(path, state, scheduler(plan['launch']))
        matches = [r for r in state['rounds'] if journal.submit(r['label'], []) == os.environ.get('SLURM_JOB_ID')]
        if len(matches) != 1: raise ValueError('Allocation does not match one recorded round')
        item = matches[0]
        journal.submit('Gwait-' + os.environ['SLURM_JOB_ID'],
            batch_args(plan['launch'], 'control', plan_path, dependency=os.environ['SLURM_JOB_ID']))
    from discoverer_resources import duration
    runtime.run(Path(item['root']), common.ROOT, common.ROOT, item['allocation']['workers'],
                validate_only=True, max_seconds=duration(item['allocation']['time_limit']))


def resume(args):
    path = common.path_from_repo(args.resume)
    # Identity and resource checks are also performed during a dry-run.
    plan = read(path)
    if plan.get('format') != FORMAT:
        return common.resume_legacy(args, path, plan)
    spec = plan['launch']
    if (path.parent / 'verified.json').exists() or (path.parent / 'checkpoint-archive.json').exists():
        raise ValueError('Run already completed; final CSV is retained; do not resume training')
    for field in ('account', 'qos', 'constraint'):
        value = getattr(args, field)
        if value is not None and value != spec['cluster'].get(field):
            raise ValueError('resume cannot override frozen ' + field)
    if args.profile != spec['profile']: raise ValueError('resume cannot change cluster profile')
    frozen_retention = plan.get('checkpoint_retention', 'default')
    if args.checkpoints and args.checkpoints != ('keep' if frozen_retention == 'default' else frozen_retention):
        raise ValueError('resume cannot override frozen checkpoint policy')
    if args.refresh_env: raise ValueError('resume cannot refresh the frozen environment')
    environment = spec.get('worker_environment') or read(path.parent / 'cluster-environment.json')
    python = Path(environment['python'])
    if args.venv and common.path_from_repo(args.venv) != python.parent.parent:
        raise ValueError('resume cannot change frozen environment')
    if args.dry_run:
        print(json.dumps({'scheduler': FORMAT, 'plan': str(path), 'python': str(python),
                          'launch': {**spec, 'checkpoint_retention': frozen_retention},
                          'actions': ['verify stopped rounds', 'resume missing single-model tasks']}, indent=2))
        return
    load(path)
    if not python.is_file(): raise ValueError('Frozen Python is unavailable')
    common.command([python, common.ROOT / 'launch/cluster_scheduler.py', 'start', path])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['start', 'control', 'work', 'preview'])
    p.add_argument('plan', type=Path)
    args = p.parse_args()
    if args.command == 'preview':
        plan = read(args.plan)
        if plan.get('format') != FORMAT: raise ValueError('Legacy grouped plan; use its frozen checkout')
        print(json.dumps({'scheduler': FORMAT, 'submission': plan['submission']}, indent=2))
    else:
        {'start': start, 'control': advance, 'work': work}[args.command](args.plan.resolve())


if __name__ == '__main__': main()
