"""Operational scheduling controls; never part of scientific task identity."""
import math

from .shared_queue import QueueError


DEFAULTS = {
    'target_batch_seconds': 30., 'max_batch_tasks': 4096,
    'claim_bytes': 512 * 1024, 'submit_bytes': 512 * 1024,
    'flush_seconds': 10., 'status_seconds': 30., 'stale_status_seconds': 90.,
    'idle_fraction': .1, 'idle_samples': 4, 'startup_grace_seconds': 120.,
    'drain_grace_seconds': 60., 'drain_enabled': False,
    'restart_overhead_seconds': None, 'restart_cpu_hours': None,
    'target_round_seconds': None, 'max_cpu_hours': None, 'exclude_nodes': [],
    'max_nodes': 60,
}


def validate_policy(value=None):
    value = {} if value is None else value
    if not isinstance(value, dict) or set(value) - DEFAULTS.keys():
        raise QueueError('Unknown scheduler policy fields')
    policy = {**DEFAULTS, **value}
    for key, default in DEFAULTS.items():
        item = policy[key]
        if key == 'exclude_nodes':
            if not isinstance(item, list) or any(not isinstance(n, str) or not n
                    or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in n)
                    for n in item):
                raise QueueError('exclude_nodes must be explicit node names')
            policy[key] = sorted(set(item))
        elif key == 'drain_enabled':
            if type(item) is not bool: raise QueueError('drain_enabled must be boolean')
        elif item is None and default is None:
            continue
        elif isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) or item <= 0:
            raise QueueError('Scheduler policy ' + key + ' must be positive and finite')
    for key in ('max_batch_tasks', 'claim_bytes', 'submit_bytes', 'idle_samples', 'max_nodes'):
        if type(policy[key]) is not int: raise QueueError(key + ' must be an integer')
    if policy['max_batch_tasks'] > 4096 or policy['claim_bytes'] > 512 * 1024 or policy['submit_bytes'] > 512 * 1024:
        raise QueueError('Scheduler policy exceeds protocol safety limits')
    if not 0 < policy['idle_fraction'] < 1: raise QueueError('idle_fraction must be between zero and one')
    if policy['status_seconds'] * 2 >= policy['stale_status_seconds']:
        raise QueueError('Worker status needs stale-detection headroom')
    if policy['drain_enabled'] and any(policy[k] is None for k in
            ('restart_overhead_seconds', 'restart_cpu_hours', 'max_cpu_hours')):
        raise QueueError('Automatic drain needs measured restart costs and a CPU-hour budget')
    return policy


class TailMonitor:
    """Require sustained measured idleness, a successor and costed migration.

    Unknown or stale worker status cannot authorize an economic drain. A running
    expensive cell alone is not proof of a sick node. No predicted speedup is
    manufactured when the tail cannot be priced.
    """
    def __init__(self, policy, *, started):
        self.policy = validate_policy(policy)
        self.started = started
        self.samples = 0

    def observe(self, stats, *, now, allocation=None, continuation_allowed=False):
        p = self.policy
        incomplete = stats['done'] < stats['total']
        eligible = (now - self.started >= p['startup_grace_seconds'] and incomplete
                    and stats.get('effective_busy_fraction', 1.) < p['idle_fraction'])
        self.samples = self.samples + 1 if eligible else 0
        report = {'idle_samples': self.samples, 'idle_alert': self.samples >= p['idle_samples'],
                  'drain_recommended': False}
        if not (report['idle_alert'] and p['drain_enabled'] and continuation_allowed
                and stats.get('pending') == 0 and not stats.get('unknown_workers')
                and not stats.get('unacknowledged_chunks') and allocation):
            return report
        if (stats['done'] == 0 and allocation.get('prior_no_progress_rounds', 0) + 1
                >= allocation.get('max_no_progress_rounds', 2)):
            report['no_progress_budget_blocked'] = True
            return report
        work = stats.get('remaining_work_seconds')
        tail = stats.get('predicted_tail_seconds')
        busy = stats.get('computing_workers', 0)
        if work is None or tail is None or busy < 1 or work <= 0 or tail <= 0:
            return report
        # Keep the same number of tail workers, but stop billing idle workers.
        slots = allocation['tasks_per_node']
        cpu_per_slot = allocation['allocated_cpu_bound'] / (allocation['nodes'] * slots)
        small_cpu = math.ceil((busy + 1) / slots) * slots * cpu_per_slot
        keep = allocation['allocated_cpu_bound'] * tail / 3600.
        restart = p['restart_cpu_hours'] + small_cpu * (
            p['restart_overhead_seconds'] + work / busy + tail) / 3600.
        report.update(keep_cpu_hours=keep, restart_cpu_hours=restart,
                      restart_workers=busy)
        available = allocation.get('cpu_hours_remaining')
        spent = allocation['allocated_cpu_bound'] * max(0., now - self.started) / 3600.
        if available is None or restart + spent > available:
            report['budget_blocked'] = True
            return report
        # Full price of active cells is included in work: restarting is not free.
        report['drain_recommended'] = (restart < .5 * keep
            and keep - restart > p['restart_cpu_hours'])
        return report
