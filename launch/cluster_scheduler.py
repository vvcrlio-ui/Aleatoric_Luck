"""Shared single-model Slurm submission and bounded per-round continuation.

Standard-library imports suffice for login-node preview/resume. Every sbatch is
journaled before submission. A successor controller is armed before a worker
allocation is submitted; no future worker allocation is submitted in advance.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import uuid

import experiment as common
import inspect
from discoverer_continuation import Journal, Slurm, TERMINAL, controller_lock as _lock, read

FORMAT = 'single-model-slurm-v1'
DONE = {'complete', 'round_budget_exhausted', 'no_progress', 'control_budget_exhausted',
        'cpu_budget_exhausted', 'protocol_blocked', 'repair_required', 'storage_blocked'}


def operational_policy(root):
    from aleatoric_nk_grid.scheduler_policy import validate_policy
    path = Path(root) / 'scheduler-policy.json'
    return validate_policy(read(path) if path.exists() else None)


def refresh_cost_profile(plan, root, previous):
    """Price the next round from every timing this run is allowed to reuse.

    Operational only: a profile decides how tasks are batched and whether the
    economic drain can be costed, never what a round computes. Sources
    accumulate, so a calibration run imported once keeps contributing its
    breadth after later rounds add their own depth.

    Pricing must never stall the controller. A source that has gone missing is
    dropped; one whose timing identity does not belong to this plan makes the
    whole import untrusted, so the run falls back to its own rounds and, if
    those cannot be priced either, keeps whatever profile it already had.
    Unpriced work still runs, one task per claim.
    """
    from aleatoric_nk_grid import prediction_profile
    from aleatoric_nk_grid.shared_queue import atomic_json
    root = Path(root)
    profile_path = root / 'cost-profile.json'
    imported = []
    if profile_path.exists():
        try:
            for item in read(profile_path).get('evidence', {}).get('sources', ()):
                imported.append(Path(item['path']))
        except (ValueError, OSError):
            imported = []
    own = [Path(directory) / 'results.jsonl' for directory in previous]

    def usable(paths):
        result, seen = [], set()
        for path in paths:
            try: resolved = path.resolve()
            except OSError: continue
            if resolved in seen or not resolved.is_file(): continue
            seen.add(resolved); result.append(resolved)
        return result

    for candidate in (usable(imported + own), usable(own)):
        if not candidate: continue
        try:
            profile = prediction_profile.build(plan, candidate)
        except Exception:
            continue
        atomic_json(profile_path, profile)
        return profile
    return None


def operational_inputs(root, directory, item=None):
    """A submitted round reads its immutable snapshot, never a mutable profile."""
    from aleatoric_nk_grid.scheduler_cost import CostEstimator
    from aleatoric_nk_grid.scheduler_policy import validate_policy
    from aleatoric_nk_grid.shared_queue import digest
    path = Path(directory) / 'operational.json'
    if path.exists():
        saved = read(path)
        if item is not None and common.sha256(path) != item['operational_sha256']:
            raise ValueError('Round operational snapshot changed')
        if (saved['policy_sha256'] != digest(saved['policy'])
                or saved['cost_profile_sha256'] != (digest(saved['cost_profile'])
                    if saved['cost_profile'] is not None else None)):
            raise ValueError('Round operational snapshot hashes changed')
        validate_policy(saved['policy'])
        if saved['cost_profile'] is not None: CostEstimator(profile=saved['cost_profile'])
        return saved
    if item is not None and item.get('operational_sha256'):
        raise ValueError('Round operational snapshot missing')
    policy = operational_policy(root)
    profile_path = Path(root) / 'cost-profile.json'
    profile = read(profile_path) if profile_path.exists() else None
    if profile is not None: CostEstimator(profile=profile)
    return {'format': 'scheduler-operational-v1', 'policy': policy,
            'policy_sha256': digest(policy), 'cost_profile': profile,
            'cost_profile_sha256': digest(profile) if profile is not None else None}


def worker_cpu_hours(state):
    """Charge stopped rounds once; missing receipt means the full reservation."""
    from discoverer_resources import duration
    total = 0.
    for item in state['rounds']:
        if item['label'] not in state['jobs']: continue
        allocation = item['allocation']
        cpu = allocation.get('allocated_cpu_bound')
        if cpu is None:
            raise ValueError('CPU budget requires a recorded allocated_cpu_bound')
        seconds = duration(allocation['time_limit'])
        evidence = {'source': 'full_reservation', 'cpu': cpu, 'seconds': seconds}
        receipt_path = Path(item['root']) / 'control/round-result.json'
        if receipt_path.exists():
            receipt = read(receipt_path)
            latest_path = receipt_path.with_name('latest.json')
            latest = read(latest_path) if latest_path.exists() else {}
            queue_id = read(Path(item['root']) / 'queue-id.json')['queue_id']
            elapsed = receipt.get('elapsed_seconds')
            job_id = state['jobs'][item['label']].get('job_id')
            receipt_cpu = receipt.get('allocation_cpu', cpu)
            if (str(receipt.get('job_id')) == str(job_id)
                    and receipt.get('queue_id') == queue_id
                    and receipt.get('elapsed_source') == 'slurm_job_start'
                    and isinstance(receipt.get('generation'), str) and bool(receipt['generation'])
                    and latest.get('generation') == receipt['generation']
                    and latest.get('queue_id') == queue_id and str(latest.get('job_id')) == str(job_id)
                    and receipt.get('state') in ('complete', 'drained', 'incomplete')
                    and isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool)
                    and math.isfinite(elapsed) and elapsed >= 0
                    and isinstance(receipt_cpu, (int, float)) and math.isfinite(receipt_cpu)
                    and receipt_cpu > 0):
                evidence = {'source': 'round_receipt', 'cpu': max(cpu, receipt_cpu),
                            'seconds': elapsed, 'receipt_sha256': common.sha256(receipt_path)}
        previous = item.get('cpu_charge')
        if previous is not None and previous.get('source') == 'round_receipt' and previous != evidence:
            raise ValueError('Stopped round accounting receipt changed')
        item['cpu_charge'] = evidence
        total += evidence['cpu'] * evidence['seconds'] / 3600
    return total


def cpu_budget(state, policy):
    """Operational edits can tighten an established run budget, never reset it."""
    limit = policy.get('max_cpu_hours')
    old = state.get('cpu_budget', {}).get('limit')
    if old is not None: limit = min(old, limit) if limit is not None else old
    if limit is None: return None
    workers = worker_cpu_hours(state)
    state['cpu_budget'] = {'limit': limit, 'worker_cpu_hours': workers}
    return max(0., limit - workers)


def release_prediction_storage(plan, state, journal):
    """Science is verified already; a ledger failure only leaves release pending."""
    from aleatoric_nk_grid.prediction_admission import release_plan_storage
    try:
        state['storage_release'] = release_plan_storage(plan)
    except (ValueError, OSError) as exc:
        state.update(status='storage_release_pending', blocked_reason=str(exc))
        journal.save()
        return False
    return True


def load(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = read(plan_path)
    if plan.get('format') != FORMAT:
        raise ValueError('Legacy grouped plan cannot be submitted by the new scheduler. '
                         'Use its frozen checkout for recovery or explicitly migrate sealed results.')
    if Path(plan['launch']['output']).resolve() != plan_path.parent:
        raise ValueError('Plan output directory changed')
    common.validate_source(plan['launch'])
    if plan.get('prediction_workflow') is not None:
        from aleatoric_nk_grid.prediction_workflow import validate_contract, validate_round_time_limits
        validate_contract(plan['prediction_workflow'])
        validate_round_time_limits(plan['prediction_workflow'],
                                   global_time_limit=plan['launch']['cluster']['time_limit'])
    else:
        from aleatoric_nk_grid.prediction_workflow import reject_unphased_cache
        reject_unphased_cache(plan.get('cell_spec') or {})
    return plan


def batch_args(spec, mode, plan_path, *, dependency=None, allocation=None, policy=None):
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
    excluded = (policy or {}).get('exclude_nodes', [])
    if excluded: args.append('--exclude=' + ','.join(excluded))
    if allocation:
        if allocation['nodes'] + allocation.get('control_node_reserve', 0) > (policy or {}).get('max_nodes', 60):
            raise ValueError('Allocation exceeds the explicit total node hard limit')
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
        workflow = plan.get('prediction_workflow')
        rounds = (sum(workflow['phase_round_limits'].values()) if workflow
                  else plan['launch']['cluster']['rounds'])
        state = {'format': FORMAT, 'run_id': uuid.uuid4().hex, 'plan_sha256': common.sha256(plan_path),
                 'jobs': {}, 'rounds': [], 'status': 'ready', 'completed': 0, 'no_progress': 0,
                 'limits': {'max_control_jobs': 4 * rounds + 8}}
        if workflow:
            state.update(workflow_state='PLANNED', phase='base', phase_completed={'base': 0, 'sl': 0},
                         phase_round_limits=workflow['phase_round_limits'])
    return path, state


def scheduler(spec):
    return Slurm(spec['cluster']['account'], spec['cluster'].get('qos'))


def terminal(journal, label):
    job = journal.submit(label, [])  # Recover an accepted-but-unacknowledged intent.
    return all(s in TERMINAL for s in journal.slurm.states(job)), job


def start(plan_path, *, slurm=None):
    plan = load(plan_path); root = Path(plan_path).parent
    policy = operational_policy(root)
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
                                               dependency=os.environ.get('SLURM_JOB_ID'), policy=policy))
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
        policy = operational_policy(root)
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
            journal.submit(label, batch_args(spec, 'control', plan_path, dependency=active[0], policy=policy))
            return state
        for item in state['rounds']:
            if item['label'] not in state['jobs']: continue
            receipt_path = Path(item['root']) / 'control/round-result.json'
            if not receipt_path.exists(): continue
            receipt = read(receipt_path)
            reason = receipt.get('drain_reason', '')
            if (str(receipt.get('job_id')) == str(state['jobs'][item['label']].get('job_id'))
                    and receipt.get('queue_id') == read(Path(item['root']) / 'queue-id.json')['queue_id']
                    and isinstance(reason, str) and reason.startswith('worker_fault:')):
                state.update(status='protocol_blocked', blocked_reason=reason)
                journal.save(); return state
        controls = sum(k.startswith(('C', 'G')) for k in state['jobs'])
        if controls >= state['limits']['max_control_jobs']:
            state['status'] = 'control_budget_exhausted'; journal.save(); return state
        remaining_budget = cpu_budget(state, policy)
        # A restart or policy-file removal cannot raise an established node cap.
        state['max_nodes'] = min(state.get('max_nodes', policy['max_nodes']), policy['max_nodes'])
        prior_control_bounds = [r['allocation'].get('control_cpu_bound',
                               r['allocation'].get('allocated_cpu_bound')) for r in state['rounds']
                               if r['label'] in state['jobs']]
        prior_control_bounds = [value for value in prior_control_bounds if value is not None]
        guard_allowed = True
        if remaining_budget is not None and prior_control_bounds:
            from discoverer_resources import duration
            guard_allowed = ((controls + 1) * max(prior_control_bounds)
                             * duration(spec['plan_time']) / 3600 <= remaining_budget)
        # This guard covers failures during scanning, preparation and sbatch.
        if guard_allowed:
            journal.submit('Gafter-' + current, batch_args(spec, 'control', plan_path,
                                                         dependency=current, policy=policy))
        rounds = state['rounds']
        # A prepared round without a W intent is safe to reuse after controller loss.
        all_previous = [r['root'] for r in rounds if r['label'] in state['jobs']]
        workflow = plan.get('prediction_workflow')
        if workflow:
            from aleatoric_nk_grid import prediction_workflow as phases
            from aleatoric_nk_grid.prediction_cache import CacheBusyError
            phase = state['phase']
            base_rounds = [r['root'] for r in rounds if r['label'] in state['jobs'] and r['phase'] == 'base']
            sl_rounds = [r['root'] for r in rounds if r['label'] in state['jobs'] and r['phase'] == 'sl']
            # A receipt can be published just before controller loss. Recover the
            # transition without starting another base allocation or resetting any budgets.
            if phase == 'sl' or (root / 'base-verified.json').exists():
                # Reread the sealed bytes on the transition and after controller
                # loss, not on every advance: the index is tens of gigabytes at
                # production repeat counts and cannot change under our own lock.
                reread = phase != 'sl' or not state.get('base_receipt_verified')
                try:
                    phases.verify_base_receipt(plan, verify_files=reread)
                    if reread: state['base_receipt_verified'] = True
                except (ValueError, OSError, CacheBusyError) as exc:
                    state.update(status='repair_required', workflow_state='REPAIR_REQUIRED',
                                 blocked_reason=str(exc), derived_results_valid=False)
                    journal.save(); return state
                phase = 'sl'; state.update(phase='sl', workflow_state='SL_READY')
                journal.save()
            previous = base_rounds if phase == 'base' else sl_rounds
        else:
            phase = None; previous = all_previous
        if (root / 'verified.json').exists():
            if workflow:
                try:
                    phases.finalize(plan, base_rounds, sl_rounds)
                except (ValueError, OSError, CacheBusyError) as exc:
                    state.update(status='repair_required', workflow_state='REPAIR_REQUIRED',
                                 blocked_reason=str(exc), derived_results_valid=False)
                    journal.save(); return state
                state['workflow_state'] = 'COMPLETE'
                if not release_prediction_storage(plan, state, journal): return state
            else:
                backend.finalize(plan, previous)
            state['status'] = 'complete'; journal.save(); return state
        index = len(all_previous)
        queue_root = root / 'rounds' / ('round-' + str(index))
        # Operational, not frozen: a profile beside the plan changes how many
        # workers a round asks for, never what the round computes.
        prepared_item = rounds[index] if len(rounds) > index else None
        if workflow and prepared_item is None:
            refresh_cost_profile(plan, root, previous)
        operational = operational_inputs(root, queue_root, prepared_item)
        policy, profile = operational['policy'], operational['cost_profile']
        try:
            report = (phases.prepare_round(plan, phase, previous, queue_root, cost_profile=profile) if workflow
                      else backend.prepare_round(plan, previous, queue_root, cost_profile=profile))
        except (ValueError, OSError, RuntimeError) as exc:
            if not workflow: raise
            state.update(status='repair_required', workflow_state='REPAIR_REQUIRED',
                         blocked_reason=str(exc), derived_results_valid=False)
            journal.save(); return state
        state['completed'] = report['done']
        if workflow:
            state['phase_completed'][phase] = report['done']
            state['completed'] = sum(state['phase_completed'].values())
        if report['remaining'] == 0:
            if workflow and phase == 'base':
                state.update(workflow_state='BASE_VERIFYING'); journal.save()
                try:
                    phases.seal_base(plan, previous)
                except (ValueError, OSError, CacheBusyError) as exc:
                    state.update(status='repair_required', workflow_state='REPAIR_REQUIRED',
                                 blocked_reason=str(exc), derived_results_valid=False)
                    journal.save(); return state
                state.update(phase='sl', workflow_state='SL_READY'); journal.save()
                phase = 'sl'; previous = sl_rounds
                report = phases.prepare_round(plan, phase, previous, queue_root, cost_profile=profile)
                state['phase_completed']['sl'] = report['done']
                state['completed'] = sum(state['phase_completed'].values())
            if report['remaining'] == 0:
                if workflow:
                    state['workflow_state'] = 'FINAL_VERIFYING'; journal.save()
                    try:
                        phases.finalize(plan, base_rounds, sl_rounds)
                    except (ValueError, OSError, CacheBusyError) as exc:
                        state.update(status='repair_required', workflow_state='REPAIR_REQUIRED',
                                     blocked_reason=str(exc), derived_results_valid=False)
                        journal.save(); return state
                    state['workflow_state'] = 'COMPLETE'
                    if not release_prediction_storage(plan, state, journal): return state
                else:
                    backend.finalize(plan, previous)
                state['status'] = 'complete'; journal.save(); return state
        if not guard_allowed:
            state['status'] = 'cpu_budget_exhausted'; journal.save(); return state
        phase_index = len(previous)
        round_limit = workflow['phase_round_limits'][phase] if workflow else spec['cluster']['rounds']
        if phase_index >= round_limit:
            state['status'] = 'round_budget_exhausted'; journal.save(); return state
        stalled = 0
        max_no_progress = spec.get('continuation', {}).get('max_no_progress_rounds', 2)
        if index:
            # Derive from round history; replaying a controller cannot add a strike.
            for old in reversed([r for r in rounds[:index] if not workflow or r['phase'] == phase]):
                if old['done_before'] != report['done']: break
                stalled += 1
            if stalled >= max_no_progress:
                state['status'] = 'no_progress'; journal.save(); return state
        remaining_budget = cpu_budget(state, policy)
        control_count = sum(k.startswith(('C', 'G')) for k in state['jobs'])
        candidates = {'work_seconds': report.get('work_seconds'),
                      'max_nodes': max(0, state['max_nodes'] - 2),
                      'target_round_seconds': policy.get('target_round_seconds'),
                      'cpu_hours_remaining': remaining_budget,
                      'control_jobs_reserved': control_count + 2,
                      # Geometry is operational, like the cost profile beside it.
                      'worker_memory': policy.get('worker_memory'),
                      'worker_cap': policy.get('worker_cap')}
        parameters = inspect.signature(resource_resolver).parameters
        extra = {key: value for key, value in candidates.items() if key in parameters}
        from cluster_resources import CpuBudgetExhausted
        try:
            from cluster_resources import phase_spec
            request_spec = phase_spec(spec, workflow, phase, round_index=phase_index) if workflow else spec
            allocation = dict(resource_resolver(request_spec, report['remaining'], **extra))
            allocation['control_node_reserve'] = 2
            allocation['total_node_bound'] = allocation['nodes'] + 2
            if allocation['total_node_bound'] > state['max_nodes']:
                raise ValueError('Resolver exceeded the cumulative node hard cap')
        except CpuBudgetExhausted:
            state['status'] = 'cpu_budget_exhausted'; journal.save(); return state
        allocation['work_seconds'] = report.get('work_seconds')
        allocation['prior_no_progress_rounds'] = stalled
        allocation['max_no_progress_rounds'] = max_no_progress
        if remaining_budget is not None:
            from discoverer_resources import duration
            cpu = allocation.get('allocated_cpu_bound')
            control_cpu = allocation.get('control_cpu_bound')
            if cpu is None or control_cpu is None:
                raise ValueError('Budgeted allocation requires worker and controller CPU bounds')
            # Past controllers use the largest observed bound, including old
            # rounds whose allocation predates operational-policy support.
            past_control_cpu = max([control_cpu] + [
                r['allocation'].get('control_cpu_bound', r['allocation'].get('allocated_cpu_bound', control_cpu))
                for r in rounds if r['label'] in state['jobs']])
            controls_hours = (control_count * past_control_cpu + 2 * control_cpu) * duration(spec['plan_time']) / 3600
            next_hours = cpu * duration(allocation['time_limit']) / 3600
            state['cpu_budget'].update(control_cpu_hours_reserved=controls_hours,
                                       next_allocation_cpu_hours_reserved=next_hours)
            if next_hours + controls_hours > remaining_budget + 1e-9:
                state['status'] = 'cpu_budget_exhausted'; journal.save(); return state
            allocation['cpu_hours_remaining'] = remaining_budget - controls_hours
        operational.update(queue_id=report['queue_id'], work_seconds=report.get('work_seconds'),
            cost_profile_coverage=report.get('cost_profile_coverage'),
            software={'commit': spec.get('source', {}).get('commit'), 'runtime_sha256': plan['runtime_sha256']})
        snapshot_path = queue_root / 'operational.json'
        if snapshot_path.exists():
            if read(snapshot_path) != operational:
                raise ValueError('Round operational snapshot changed during recovery')
        else:
            common.atomic_json(snapshot_path, operational)
        item = {'index': index, 'root': str(queue_root), 'label': 'W' + str(index),
                'done_before': report['done'], 'allocation': allocation,
                'operational_sha256': common.sha256(snapshot_path),
                'continuation_allowed': phase_index + 1 < round_limit}
        if workflow:
            item.update(phase=phase, phase_index=phase_index)
            allocation.update(phase=phase)
            state['workflow_state'] = 'BASE_RUNNING' if phase == 'base' else 'SL_RUNNING'
        if len(rounds) == index: rounds.append(item)
        else: rounds[index] = item
        if workflow:
            from aleatoric_nk_grid.prediction_admission import check_plan_storage
            try:
                state['storage_admission'] = check_plan_storage(plan, allocated_workers=allocation['workers'])
            except (ValueError, OSError) as exc:
                state.update(status='storage_blocked', blocked_reason=str(exc))
                journal.save(); return state
        state['status'] = 'running'; journal.save()
        job = journal.submit(item['label'], batch_args(spec, 'work', plan_path,
            dependency=current, allocation=allocation, policy=policy))
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
        operational = operational_inputs(root, item['root'], item)
        if operational.get('queue_id') != read(Path(item['root']) / 'queue-id.json')['queue_id']:
            raise ValueError('Round operational snapshot belongs to another queue')
        journal.submit('Gwait-' + os.environ['SLURM_JOB_ID'],
            batch_args(plan['launch'], 'control', plan_path, dependency=os.environ['SLURM_JOB_ID'],
                       policy=operational['policy']))
    from discoverer_resources import duration
    runtime.run(Path(item['root']), common.ROOT, common.ROOT, item['allocation']['workers'],
                validate_only=True, max_seconds=duration(item['allocation']['time_limit']),
                policy=operational['policy'], cost_profile=operational['cost_profile'],
                allocation={**item['allocation'], 'job_id': os.environ['SLURM_JOB_ID']},
                continuation_allowed=item.get('continuation_allowed', False))


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
