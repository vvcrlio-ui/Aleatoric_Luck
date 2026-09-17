"""Conservative live admission for a single multi-node Slurm allocation.

Uses the existing scoped account/QoS checks, then bounds the whole allocation,
including its dispatcher, memory, physical-core billing and CPU-minute budget.
No site quota or node count is embedded in the submission path.
"""
import math
import re

import discoverer_resources as base


def default_qos(account, *, run=base.query):
    import getpass
    rows = base.records(run(['sacctmgr', '-nP', 'show', 'assoc', 'where',
        'account=' + account, 'user=' + getpass.getuser(), 'format=DefaultQOS']), 'qos')
    names = {r['qos'] for r in rows if r['qos'] not in ('', 'N/A', '(null)')}
    if len(names) != 1:
        raise ValueError('Cannot resolve one default QoS; supply --qos explicitly')
    return names.pop()


def minute_headroom(text, qos):
    blocks = re.split(r'(?=\bQOS=)', text, flags=re.I)
    block = next((b for b in blocks if re.match(r'QOS=' + re.escape(qos) + r'(?:\(|\s|$)', b, re.I)), None)
    if block is None:
        raise ValueError('Cannot resolve live QoS usage from scontrol show assoc_mgr: ' + qos)
    def cpu(field):
        value = re.search(r'(?:^|\s)' + field + r'=([^\s]+)', block)
        match = re.search(r'(?:^|,)cpu=([^,()]+)\(([^()]+)\)', value[1] if value else '')
        if match is None:
            raise ValueError('Cannot resolve ' + field + ' CPU limit/usage for ' + qos)
        limit = None if match[1].upper() in ('N', 'NONE', 'UNLIMITED', 'INFINITE') else float(match[1])
        used = float(match[2])
        if not math.isfinite(used) or used < 0 or (limit is not None and (not math.isfinite(limit) or limit < 0)):
            raise ValueError('Invalid CPU-minute accounting value for ' + qos)
        return limit, used
    limit, used = cpu('GrpTRESMins')
    run_limit, reserved = cpu('GrpTRESRunMins')
    limits = []
    if limit is not None: limits.append(limit - used - reserved)
    if run_limit is not None: limits.append(run_limit - reserved)
    return min(limits) if limits else None


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


def resolve(spec, remaining, *, work_seconds=None, run=base.query):
    cluster = spec['cluster']
    qos = cluster.get('qos') or default_qos(cluster['account'], run=run)
    live = base.snapshot(cluster['account'], qos, cluster['partition'], run=run)
    memory = cluster.get('memory_override') or '16G'
    cap = spec.get('continuation', {}).get('worker_cap')
    if cap is None and spec['profile'] != 'discoverer': cap = cluster['workers']
    # Retaining the old per-worker scoped bound deliberately underestimates
    # capacity; it cannot spend the old array's full allowance per allocation.
    bound = base.capacity(live, remaining=2**63 - 1, memory=memory,
        requested_time=cluster['time_limit'], worker_cap=None,
        extra_submit=3, extra_running=2)
    threads = max(int(n['threads']) for n in live['nodes'])
    cpu_per_task = max(threads, bound['allocated_cpu_per_worker_bound'])
    mem = base.memory_mb(memory)
    overhead = base.memory_mb(spec['plan_memory'])
    slots = min(min(int(n['cpu']) // threads, int((float(n['mem']) - overhead) // mem))
                for n in live['nodes'])
    if slots < 1: raise ValueError('No node can fit worker memory plus dispatcher reserve')
    max_tasks = min(remaining + 1, bound['workers'])
    if cap is not None: max_tasks = min(max_tasks, cap + 1)
    needed = workers_for_work(work_seconds, base.duration(bound['time_limit']))
    if needed is not None: max_tasks = min(max_tasks, needed + 1)
    per_job = [base.tres(r['MaxTRES']) for r in live['qos_rows']]
    # Associations may carry a per-job limit inherited from a parent account.
    parents = {r['Account']: r['ParentName'] for r in live['associations'] if not r['User']}
    ancestors = {cluster['account']}; current = cluster['account']
    while parents.get(current) and parents[current] not in ancestors:
        current = parents[current]; ancestors.add(current)
    per_job += [base.tres(r['MaxTRES']) for r in live['associations']
        if r['Account'] in ancestors and r['User'] in ('', live['user'])
        and r['Partition'] in ('', cluster['partition'])]
    slots = min(slots, max_tasks)
    for limits in per_job:
        if 'cpu' in limits: slots = min(slots, int(limits['cpu'] // cpu_per_task))
        if 'mem' in limits: slots = min(slots, int((limits['mem'] - overhead) // mem))
    max_node_mem = base.finite(live['partition'].get('MaxMemPerNode', ''))
    if max_node_mem not in (None, 0): slots = min(slots, int((max_node_mem - overhead) // mem))
    if slots < 1: raise ValueError('Per-job limits cannot fit worker and dispatcher memory')
    max_nodes = base.finite(live['partition'].get('MaxNodes', ''))
    if max_nodes is not None: max_tasks = min(max_tasks, max_nodes * slots)
    raw_usage = run(['scontrol', 'show', 'assoc_mgr'])
    budgets = [minute_headroom(raw_usage, r['Name']) for r in live['qos_rows']]
    minutes = base.duration(bound['time_limit']) / 60
    node_memory = math.ceil(slots * mem + overhead)
    # Binary search a monotone conservative bound using full last-node billing.
    def fits(tasks):
        nodes = math.ceil(tasks / slots)
        request = {'node': nodes, 'cpu': nodes * slots * cpu_per_task, 'mem': nodes * node_memory}
        units = max(request['node'], request['cpu'] / bound['allocated_cpu_per_worker_bound'], request['mem'] / mem)
        if units > bound['workers']: return False
        if any(request[k] > v for limits in per_job for k, v in limits.items() if k in request): return False
        # Reserve two short control jobs as well as the allocation.
        needed_minutes = request['cpu'] * minutes + 2 * cpu_per_task * base.duration(spec['plan_time']) / 60
        if any(b is not None and needed_minutes > b for b in budgets): return False
        return True
    low, high = 0, max_tasks
    while low < high:
        mid = (low + high + 1) // 2
        if fits(mid): low = mid
        else: high = mid - 1
    if low < 2: raise ValueError('No live capacity for one worker plus dispatcher and control reserves')
    nodes = math.ceil(low / slots)
    return {'workers': low - 1, 'nodes': nodes, 'tasks_per_node': slots,
            'memory_mb_per_node': node_memory, 'time_limit': bound['time_limit'], 'qos': qos,
            'allocated_cpu_bound': nodes * slots * cpu_per_task,
            'qos_remaining_cpu_minutes': budgets, 'live_worker_bound': bound}
