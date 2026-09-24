"""Operational scheduling controls; never part of scientific task identity."""
import math
import re

from .shared_queue import QueueError


CONTROL_NODE_RESERVE = 2

DEFAULTS = {
    'target_batch_seconds': 30., 'max_batch_tasks': 4096,
    'claim_bytes': 512 * 1024, 'submit_bytes': 512 * 1024,
    'flush_seconds': 10., 'status_seconds': 30., 'stale_status_seconds': 90.,
    'idle_fraction': .1, 'idle_samples': 4, 'startup_grace_seconds': 120.,
    'drain_grace_seconds': 60., 'drain_enabled': False,
    'restart_overhead_seconds': None, 'restart_cpu_hours': None,
    'target_round_seconds': None, 'max_cpu_hours': None, 'exclude_nodes': [],
    'max_nodes': 60,
    # Geometry is operational: it decides how many workers a round asks for and
    # how much memory each reserves, never what the round computes. Keeping it
    # out of the frozen plan lets a run be resized between rounds instead of
    # discarding every cell it already finished.
    'protocol_metrics': False, 'rpc_keepalive': False,
    'online_costs': False, 'online_cost_refresh_seconds': 15.,
    'online_cost_min_observations': 100, 'online_cost_safety_factor': 1.25,
    'online_cost_max_batch': 16,
    'heartbeat_aggregate_seconds': 0., 'validation_processes': 0,
    # Node relay (opt-in): each node forwards its workers' requests over a few
    # persistent connections. It needs a connection budget above the per-worker
    # default and a longer server-side keep-alive wait; the defaults here keep
    # today's behaviour exactly.
    'node_relay': False, 'max_connections': 128, 'keepalive_idle_seconds': 1.,
    # Concurrent result submissions the server admits (default: the historical constant). It budgets
    # connections, not throughput; the effective value is also capped at half of max_connections.
    'max_submissions': 32,
    # Validators reuse resolved shard paths and open shard descriptors (see prediction_cache.ReadContext).
    'validation_fast_reads': False,
    # Dispatcher CPU: light HTTP header parsing, one-write replies, validators receive the journal bytes.
    'dispatcher_fast_path': False,
    # Let the dispatcher use both hardware threads of its reserved core (the sibling is otherwise idle).
    'dispatcher_smt': False,
    # Independent dispatcher processes (shards) on the controller node, each with its own core, validators and
    # journal, serving a disjoint slice of the round's tasks to its own group of worker nodes.
    'dispatcher_shards': 1,
    'validation_timeout_seconds': 120., 'protocol_metrics_sample_modulo': 1,
    'max_claimed_tasks': None,
    'worker_memory': None, 'worker_cap': None, 'worker_time_limit': None,
    'base_round_limit': None, 'sl_round_limit': None,
    # Explicit SL allocation geometry. The same central dispatcher/relay path is
    # used; only admission changes, never the frozen cache/scientific contract.
    'sl_allocation': None,
    # One operational resource policy for both numerical phases. Legacy frozen
    # round snapshots retain their previous phase-specific behaviour.
    'unified_compute': False, 'sizing_mode': 'work',
    'parallel_verification': False, 'verification_block_bytes': 64 * 1024**2,
    'verification_round_limit': 3,
    # final_only runs: content-check the stopped base rounds while SL computes,
    # inside the node cap left by the SL allocation, so the final audit only
    # checks SL. Optional smaller SL ranges balance that last audit.
    'overlap_base_audit': False, 'sl_verification_block_bytes': None,
    # Worker ceiling for SL allocations only, also under unified_compute.
    'sl_worker_cap': None,
}


def validate_policy(value=None):
    value = {} if value is None else value
    if not isinstance(value, dict) or set(value) - DEFAULTS.keys():
        raise QueueError('Unknown scheduler policy fields')
    policy = {**DEFAULTS, **value}
    for key, default in DEFAULTS.items():
        item = policy[key]
        if key == 'sizing_mode':
            if item not in ('work', 'capacity'):
                raise QueueError('sizing_mode must be work or capacity')
        elif key == 'sl_allocation':
            if item is None:
                continue
            allowed = {'sizing_mode', 'worker_cap', 'worker_memory',
                       'worker_time_limit', 'target_round_seconds'}
            if (not isinstance(item, dict) or set(item) - allowed
                    or not {'sizing_mode', 'worker_cap'} <= set(item)):
                raise QueueError('sl_allocation requires sizing_mode and worker_cap; unknown fields are refused')
            if item['sizing_mode'] not in ('work', 'capacity'):
                raise QueueError('sl_allocation sizing_mode must be work or capacity')
            if type(item['worker_cap']) is not int or item['worker_cap'] < 1:
                raise QueueError('sl_allocation worker_cap must be a positive integer')
            # Reuse the ordinary operational resource validators.
            validate_policy({k: v for k, v in item.items() if k != 'sizing_mode'})
            if item['sizing_mode'] == 'capacity' and item.get('target_round_seconds') is not None:
                raise QueueError('capacity sizing cannot also specify target_round_seconds')
            policy[key] = dict(item)
        elif key == 'exclude_nodes':
            if not isinstance(item, list) or any(not isinstance(n, str) or not n
                    or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in n)
                    for n in item):
                raise QueueError('exclude_nodes must be explicit node names')
            policy[key] = sorted(set(item))
        elif key in ('drain_enabled', 'protocol_metrics', 'rpc_keepalive', 'node_relay', 'validation_fast_reads', 'dispatcher_fast_path', 'dispatcher_smt', 'unified_compute', 'parallel_verification', 'online_costs', 'overlap_base_audit'):
            if type(item) is not bool: raise QueueError(key + ' must be boolean')
        elif key in ('heartbeat_aggregate_seconds', 'validation_processes'):
            if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) or item < 0:
                raise QueueError('Operational concurrency must be nonnegative')
            if key == 'validation_processes' and (type(item) is not int or item > 32):
                raise QueueError('Validation processes must be an integer in 0..32')
            if key == 'heartbeat_aggregate_seconds' and item > 5:
                raise QueueError('Heartbeat aggregation window exceeds five seconds')
        elif item is None and default is None:
            continue
        elif key == 'worker_memory':
            if not isinstance(item, str) or not re.fullmatch(r'\d+[KMGT]?', item.strip()):
                raise QueueError('worker_memory must be a Slurm memory size such as 4G')
        elif key == 'worker_time_limit':
            match = re.fullmatch(r'(?:(\d+)-)?(\d+):(\d{2}):(\d{2})', item) if isinstance(item, str) else None
            if not match:
                raise QueueError('worker_time_limit must be a positive Slurm [days-]HH:MM:SS')
            days, hours, minutes, seconds = (int(v or 0) for v in match.groups())
            if minutes >= 60 or seconds >= 60 or not (days * 86400 + hours * 3600 + minutes * 60 + seconds):
                raise QueueError('worker_time_limit must be a positive Slurm [days-]HH:MM:SS')
        elif isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) or item <= 0:
            raise QueueError('Scheduler policy ' + key + ' must be positive and finite')
    for key in ('max_batch_tasks', 'claim_bytes', 'submit_bytes', 'idle_samples', 'max_nodes', 'max_connections',
                'max_submissions', 'dispatcher_shards', 'verification_block_bytes', 'verification_round_limit'):
        if type(policy[key]) is not int: raise QueueError(key + ' must be an integer')
    if not 1 <= policy['max_connections'] <= 4096:
        raise QueueError('max_connections must be within 1..4096')
    if type(policy['online_cost_min_observations']) is not int or policy['online_cost_min_observations'] < 100:
        raise QueueError('Online costs require at least 100 complete cold observations')
    if type(policy['online_cost_max_batch']) is not int or not 1 <= policy['online_cost_max_batch'] <= 64:
        raise QueueError('Online batch cap must be within 1..64')
    if policy['online_cost_safety_factor'] < 1:
        raise QueueError('Online cost safety factor cannot discount observed runtimes')
    if policy['online_cost_refresh_seconds'] < 5:
        raise QueueError('Online cost summaries must not refresh more often than every five seconds')
    if not 1 <= policy['max_submissions'] <= 1024:
        raise QueueError('max_submissions must be within 1..1024')
    if not .1 <= policy['keepalive_idle_seconds'] <= 30.:
        raise QueueError('keepalive_idle_seconds must be within 0.1..30')
    # Results wait in the worker's durable spool until the flush; a longer wait is
    # allowed (the lease is renewed meanwhile) but must stay well inside the lease.
    if policy['flush_seconds'] > 120.:
        raise QueueError('flush_seconds must not exceed 120')
    if not 1 <= policy['dispatcher_shards'] <= 8:
        raise QueueError('dispatcher_shards must be within 1..8')
    if policy['dispatcher_shards'] > 1 and not policy['validation_processes']:
        raise QueueError('dispatcher_shards needs the reserved service step (validation_processes >= 1)')
    if policy['dispatcher_shards'] > 1 and policy['heartbeat_aggregate_seconds']:
        raise QueueError('dispatcher_shards has not been combined with heartbeat aggregation')
    if policy['dispatcher_smt'] and not policy['validation_processes']:
        raise QueueError('dispatcher_smt needs the reserved service step (validation_processes >= 1)')
    if policy['node_relay'] and not policy['rpc_keepalive']:
        raise QueueError('node_relay needs rpc_keepalive: its upstream connections must persist')
    if policy['node_relay'] and policy['heartbeat_aggregate_seconds']:
        # Both start a per-node child process and the aggregator talks to the dispatcher on its own
        # connection, around the relay. That combination has not been measured, so it is refused.
        raise QueueError('node_relay replaces heartbeat aggregation; enable only one of them')
    if type(policy['protocol_metrics_sample_modulo']) is not int or policy['protocol_metrics_sample_modulo'] > 1024:
        raise QueueError('Metrics sampling modulo must be an integer in 1..1024')
    if policy['max_claimed_tasks'] is not None and type(policy['max_claimed_tasks']) is not int:
        raise QueueError('Claim limit must be an integer')
    if policy['worker_cap'] is not None and type(policy['worker_cap']) is not int:
        raise QueueError('worker_cap must be an integer')
    if policy['sl_worker_cap'] is not None and type(policy['sl_worker_cap']) is not int:
        raise QueueError('sl_worker_cap must be an integer')
    if not 1024 <= policy['verification_block_bytes'] <= 1024**3:
        raise QueueError('verification_block_bytes must be within 1 KiB..1 GiB')
    if policy['sl_verification_block_bytes'] is not None and (type(policy['sl_verification_block_bytes']) is not int
            or not 1024 <= policy['sl_verification_block_bytes'] <= 1024**3):
        raise QueueError('sl_verification_block_bytes must be an integer within 1 KiB..1 GiB')
    if policy['unified_compute']:
        if policy['sl_allocation'] is not None:
            raise QueueError('Unified compute cannot also have an SL-specific allocation')
        if any(policy[k] is None for k in ('worker_cap', 'worker_memory', 'worker_time_limit')):
            raise QueueError('Unified compute requires shared worker_cap, worker_memory and worker_time_limit')
    elif policy['sizing_mode'] != 'work':
        raise QueueError('Shared capacity sizing requires unified_compute')
    if policy['sizing_mode'] == 'capacity' and policy['target_round_seconds'] is not None:
        raise QueueError('Shared capacity sizing cannot also specify target_round_seconds')
    for key in ('base_round_limit', 'sl_round_limit'):
        if policy[key] is not None and type(policy[key]) is not int:
            raise QueueError(key + ' must be an integer')
    # The scheduler hands the resolver max_nodes minus two reserved control
    # nodes, so a cap of two or less resolves to zero and fails deep inside
    # sizing, after the controller has already been submitted. Say so here.
    if policy['max_nodes'] < CONTROL_NODE_RESERVE + 1:
        raise QueueError('max_nodes must leave room for %d control nodes plus at least one '
                         'worker node; got %d' % (CONTROL_NODE_RESERVE, policy['max_nodes']))
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
