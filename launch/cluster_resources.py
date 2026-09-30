"""Size one multi-node Slurm allocation from the run's explicit node count.

The user chooses the total nodes, worker memory and time. Each round fits
workers, dispatcher and validator reserves onto those nodes using the
partition's node geometry, and may use fewer nodes when little work remains
or the run's CPU-hour budget is nearly spent. Slurm alone decides admission:
a request beyond the account's limits waits in the queue with its reason.
"""
import math
from copy import deepcopy

import slurm_resources as base


class CpuBudgetExhausted(ValueError):
    """The cumulative operational budget cannot admit another allocation."""


def phase_spec(spec, workflow, phase, *, round_index=0, policy=None):
    """Freeze separate lightweight SL resources, including a filesystem cap."""
    if phase not in ('base', 'sl'):
        raise ValueError('Unknown prediction workflow phase')
    if policy is not None:
        from aleatoric_nk_grid.scheduler_policy import validate_policy
        common = validate_policy(policy)
        if common['unified_compute']:
            result = deepcopy(spec)
            result['cluster'].update(memory_override=common['worker_memory'],
                                     time_limit=common['worker_time_limit'])
            result.setdefault('continuation', {})['worker_cap'] = common['worker_cap']
            result['prediction_phase'] = phase
            return result
    if phase == 'base':
        from aleatoric_nk_grid.prediction_workflow import validate_round_time_limits
        times = validate_round_time_limits(workflow, global_time_limit=spec['cluster']['time_limit'])
        if times is None: return spec
        if type(round_index) is not int or not 0 <= round_index < len(times):
            raise ValueError('Base phase round index is outside its frozen time-limit vector')
        result = deepcopy(spec)
        result['cluster']['time_limit'] = times[round_index]
        result['prediction_phase'] = 'base'
        result['prediction_phase_round_index'] = round_index
        return result
    if phase != 'sl': raise ValueError('Unknown prediction workflow phase')
    resources = workflow['sl_resources']; result = deepcopy(spec)
    result['cluster']['memory_override'] = resources['memory']
    result['cluster']['time_limit'] = resources['time_limit']
    cap = min(resources['worker_cap'], resources['io_concurrency'])
    old_cap = result.get('continuation', {}).get('worker_cap')
    if old_cap is not None: cap = min(cap, old_cap)
    result.setdefault('continuation', {})['worker_cap'] = cap
    result['prediction_phase'] = 'sl'
    result['prediction_io_concurrency'] = resources['io_concurrency']
    return result


def workers_for_work(work_seconds, wall_seconds, *, headroom=1.25):
    """Workers that the remaining work can actually keep busy for one round.

    Asking for more is what leaves an allocation idle rather than merely large:
    the finished tree-ordinal round held 17,279 workers for 9.43 hours to
    perform 1.58 hours of compute. Returns None when no measured profile has
    priced the queue, leaving the caller's task-count bound in charge.
    """
    if work_seconds is None: return None
    if not math.isfinite(work_seconds) or work_seconds <= 0:
        raise ValueError('Remaining work must be positive and finite')
    if not math.isfinite(wall_seconds) or wall_seconds <= 0:
        raise ValueError('Round wall time must be positive and finite')
    return max(1, math.ceil(headroom * work_seconds / wall_seconds))


def phase_allocation_options(policy, phase):
    """Resolve phase-local geometry without changing the scientific plan.

    Work sizing shrinks the fleet to the priced remaining work. Capacity sizing
    fills the node count up to the worker cap and does not shrink the fleet from
    a possibly stale per-cell cost estimate. The cost profile still determines
    task order and bounded claim batches.
    """
    from aleatoric_nk_grid.scheduler_policy import validate_policy
    policy = validate_policy(policy)
    options = {key: policy[key] for key in
               ('worker_cap', 'worker_memory', 'target_round_seconds')}
    options['worker_time_limit'] = policy['worker_time_limit'] if phase != 'sl' or policy['unified_compute'] else None
    options['sizing_mode'] = policy['sizing_mode']
    if phase == 'sl' and policy['sl_allocation'] is not None:
        options.update(policy['sl_allocation'])
        if options['sizing_mode'] == 'capacity':
            options['target_round_seconds'] = None
    # SL cells are short and bound by durable acknowledgement, not by workers:
    # a cap sized to that service rate keeps the throughput and frees the nodes.
    if phase == 'sl' and policy['sl_worker_cap'] is not None:
        options['worker_cap'] = min(options['worker_cap'] or policy['sl_worker_cap'], policy['sl_worker_cap'])
    return options


def resolve(spec, remaining, *, max_nodes, work_seconds=None, target_round_seconds=None,
            cpu_hours_remaining=None, control_jobs_reserved=2,
            worker_memory=None, worker_cap=None, worker_time_limit=None, validation_processes=0, dispatcher_shards=1,
            sizing_mode='work', run=base.query):
    '''Size one round's allocation (see _resolve_round). Each dispatcher shard needs its own worker nodes besides the
    controller's, so a round too small for its shards is sized without them.'''
    if type(dispatcher_shards) is not int or not 1 <= dispatcher_shards <= 8:
        raise ValueError('Invalid dispatcher shard count')
    if dispatcher_shards > 1 and not validation_processes:
        raise ValueError('Dispatcher shards need the reserved service step')
    arguments = dict(work_seconds=work_seconds, target_round_seconds=target_round_seconds,
                     cpu_hours_remaining=cpu_hours_remaining, control_jobs_reserved=control_jobs_reserved,
                     max_nodes=max_nodes, worker_memory=worker_memory, worker_cap=worker_cap,
                     worker_time_limit=worker_time_limit, validation_processes=validation_processes,
                     sizing_mode=sizing_mode, run=run)
    if dispatcher_shards > 1:
        try:
            result = _resolve_round(spec, remaining, dispatcher_shards=dispatcher_shards, **arguments)
            if result['nodes'] > dispatcher_shards:
                result['requested_dispatcher_shards'] = dispatcher_shards
                return result
        except ValueError:
            pass        # too small (or too constrained) for every shard to have worker nodes: one dispatcher then
    result = _resolve_round(spec, remaining, dispatcher_shards=1, **arguments)
    result['requested_dispatcher_shards'] = dispatcher_shards
    if dispatcher_shards > 1:
        result['shard_admission_note'] = 'Allocation cannot provide a worker-node group for each requested shard; admitted one dispatcher'
    return result


def _resolve_round(spec, remaining, *, max_nodes, work_seconds=None, target_round_seconds=None,
            cpu_hours_remaining=None, control_jobs_reserved=2,
            worker_memory=None, worker_cap=None, worker_time_limit=None, validation_processes=0, dispatcher_shards=1,
            sizing_mode='work', run=base.query):
    """Size one round's allocation. Geometry may be operational.

    ``worker_memory`` and ``worker_cap`` come from the round's policy snapshot
    and win over the frozen launch request. They change how many workers a
    round asks for and how much each reserves, never what the round computes,
    so a run can be resized between rounds instead of discarding finished
    cells. The frozen request remains the default when the policy is silent.
    ``max_nodes`` is the node count this allocation may use.
    """
    if sizing_mode not in ('work', 'capacity'):
        raise ValueError('Unknown allocation sizing_mode')
    if sizing_mode == 'capacity' and (type(worker_cap) is not int or worker_cap < 1
                                      or target_round_seconds is not None):
        raise ValueError('Capacity sizing requires an explicit positive worker_cap and no time target')
    if type(max_nodes) is not int or max_nodes < 1:
        raise ValueError('Explicit total node cap must be a positive integer')
    if type(validation_processes) is not int or not 0 <= validation_processes <= 32:
        raise ValueError('Invalid validation process reservation')
    if type(dispatcher_shards) is not int or not 1 <= dispatcher_shards <= 8:
        raise ValueError('Invalid dispatcher shard count')
    if dispatcher_shards > 1 and not validation_processes:
        raise ValueError('Dispatcher shards need the reserved service step')
    service_slots = dispatcher_shards * (1 + validation_processes)
    task_width = service_slots if validation_processes else 1
    cluster = spec['cluster']
    time_limit = worker_time_limit or cluster['time_limit']
    memory = worker_memory or cluster.get('memory_override') or '2G'
    cap = worker_cap if worker_cap is not None else spec.get('continuation', {}).get('worker_cap')
    nodes_info = base.node_geometry(cluster['partition'], run=run)
    # One worker per physical core; Slurm counts every hardware thread of it.
    cpu_per_task = max(int(n['threads']) for n in nodes_info)
    if validation_processes:
        # Slurm 20 steps count logical CPUs even with a one-thread
        # allocation hint. Keep the allocation CPU group large enough for the
        # service step, then verify physical affinity before any fit.
        task_width = service_slots * cpu_per_task
    mem = base.memory_mb(memory)
    overhead = base.memory_mb(spec['plan_memory'])
    slots = min(min(int(n['cpu']) // cpu_per_task, int((float(n['mem']) - overhead) // mem))
                for n in nodes_info)
    if slots < 1: raise ValueError('No node can fit worker memory plus dispatcher reserve')
    max_tasks = remaining + service_slots
    if cap is not None: max_tasks = min(max_tasks, cap + service_slots)
    wall_seconds = base.duration(time_limit)
    if target_round_seconds is not None:
        if (not math.isfinite(target_round_seconds) or target_round_seconds <= 0):
            raise ValueError('Target round seconds must be positive and finite')
    sizing_seconds = min(wall_seconds, target_round_seconds or wall_seconds)
    needed = workers_for_work(work_seconds, sizing_seconds) if sizing_mode == 'work' else None
    if needed is not None: max_tasks = min(max_tasks, needed + service_slots)
    # Include every controller already journaled plus the worker's successor and
    # its recovery guard. Their full requested durations are charged until an
    # accounting-backed receipt can narrow them; no average runtime is assumed.
    if not isinstance(control_jobs_reserved, int) or control_jobs_reserved < 0:
        raise ValueError('Reserved control jobs must be a nonnegative integer')
    control_hours = control_jobs_reserved * cpu_per_task * base.duration(spec['plan_time']) / 3600
    budget_seconds = None
    if cpu_hours_remaining is not None:
        if not math.isfinite(cpu_hours_remaining) or cpu_hours_remaining < 0:
            raise ValueError('Remaining CPU hours must be nonnegative and finite')
        budget_seconds = max(0., cpu_hours_remaining - control_hours) * 3600
        max_tasks = min(max_tasks, math.floor(budget_seconds / (wall_seconds * cpu_per_task)))
        if max_tasks < service_slots + 1:
            raise CpuBudgetExhausted('CPU-hour budget cannot fit worker, dispatcher and control reserves')
    slots = min(slots, max_tasks)
    if task_width > 1:
        slots = (slots // task_width) * task_width
        if slots < task_width:
            raise ValueError('Node cannot fit the reserved service CPU group')
    node_limit = max_nodes
    if budget_seconds is not None:
        # Every allocated node is billed in full for the whole wall time.
        node_limit = min(node_limit, math.floor(budget_seconds / (wall_seconds * slots * cpu_per_task)))
        if node_limit * slots < service_slots + 1:
            raise CpuBudgetExhausted('CPU-hour budget cannot fit worker, dispatcher and control reserves')
    tasks = min(max_tasks, node_limit * slots)
    if tasks < service_slots + 1:
        raise ValueError('Node count cannot fit worker, dispatcher and validator reserves')
    groups = math.ceil(tasks / task_width)
    nodes = math.ceil(groups * task_width / slots)
    node_memory = math.ceil(slots * mem + overhead)
    return {'workers': tasks - service_slots, 'nodes': nodes, 'max_nodes': max_nodes, 'tasks_per_node': slots,
            'controller_task_slots': service_slots, 'validation_processes': validation_processes,
            'dispatcher_shards': dispatcher_shards,
            'allocation_task_width': task_width, 'allocation_tasks': groups,
            'allocation_tasks_per_node': slots // task_width,
            'cpu_per_task': cpu_per_task, 'worker_step_memory_mb': math.ceil(slots * mem),
            'controller_memory_mb': math.ceil(overhead),
            'memory_mb_per_node': node_memory, 'time_limit': time_limit, 'qos': cluster.get('qos'),
            'allocated_cpu_bound': nodes * slots * cpu_per_task,
            'control_cpu_bound': cpu_per_task, 'target_round_seconds': target_round_seconds,
            'work_seconds': work_seconds, 'sizing_mode': sizing_mode,
            'reserved_cpu_hours': nodes * slots * cpu_per_task * wall_seconds / 3600,
            'reserved_control_cpu_hours': control_hours}
