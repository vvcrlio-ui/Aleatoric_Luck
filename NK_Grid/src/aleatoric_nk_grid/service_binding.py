"""Fail-closed physical-core accounting for dispatcher/validator service steps."""
import os
from pathlib import Path
import re
import socket
import subprocess

from .shared_queue import QueueError


def physical_cores(cpus=None, topology=Path('/sys/devices/system/cpu')):
    cpus = sorted(os.sched_getaffinity(0) if cpus is None else cpus)
    groups = {}
    for cpu in cpus:
        root = topology / ('cpu%d' % cpu) / 'topology'
        key = ((root / 'physical_package_id').read_text().strip(),
               (root / 'core_id').read_text().strip())
        groups.setdefault(':'.join(key), []).append(cpu)
    return groups


def process_binding():
    cpus = sorted(os.sched_getaffinity(0))
    return {'hostname': socket.gethostname(), 'pid': os.getpid(), 'cpus': cpus,
            'cores': sorted(physical_cores(cpus)), 'rank': os.environ.get('SLURM_PROCID'),
            'local_rank': os.environ.get('SLURM_LOCALID')}


def service_layout(validators, shards=1):
    """Physical-core layout of the service step: ``shards`` dispatchers, each followed by its own validators.

    With one shard this is exactly the historical layout. With more, ``dispatcher_cpu``/``validator_cpus`` describe
    shard 0 and ``shards`` lists every dispatcher with its own validators; ``cores`` always holds all service cores.
    """
    if os.environ.get('NKGRID_SERVICE_STEP') != '1':
        raise QueueError('Validators require an explicitly reserved service step')
    groups = physical_cores()
    expected = shards * (validators + 1)
    if len(groups) != expected:
        raise QueueError('Service step physical-core count differs: expected %d, got %d, affinity=%s' %
                         (expected, len(groups), sorted(os.sched_getaffinity(0))))
    ordered = sorted(groups.items(), key=lambda item: min(item[1]))
    per_shard = [{'dispatcher_cpu': ordered[i * (validators + 1)][1][0],
                  'dispatcher_cpus': sorted(ordered[i * (validators + 1)][1]),
                  'validator_cpus': [cpus[0] for _, cpus in ordered[i * (validators + 1) + 1:(i + 1) * (validators + 1)]]}
                 for i in range(shards)]
    layout = {'hostname': socket.gethostname(), 'step_id': os.environ.get('SLURM_STEP_ID'),
              'step_cpus': sorted(c for _, cs in ordered for c in cs),
              'cores': [key for key, _ in ordered], **per_shard[0],
              'job_task_slots': int(os.environ['NKGRID_JOB_NTASKS']) * int(os.environ.get('NKGRID_JOB_TASK_CPU_WIDTH', '1'))}
    if shards > 1:
        layout['shards'] = per_shard
    return layout


def expand_counts(text):
    result = []
    for item in text.split(','):
        match = re.fullmatch(r'(\d+)(?:\(x(\d+)\))?', item.strip())
        if not match:
            raise QueueError('Invalid Slurm per-node CPU list')
        result.extend([int(match[1])] * int(match[2] or 1))
    return result


def worker_hosts(hosts, cpu_counts, cpu_per_slot, service_host, service_slots, workers):
    if len(hosts) != len(cpu_counts) or service_host not in hosts or len(set(hosts)) != len(hosts):
        raise QueueError('Allocation host/CPU topology mismatch')
    result = []
    for host, cpus in zip(hosts, cpu_counts):
        if cpus % cpu_per_slot:
            raise QueueError('Allocation does not contain whole reserved cores')
        slots = cpus // cpu_per_slot - (service_slots if host == service_host else 0)
        if slots < 0:
            raise QueueError('Service reservation exceeds allocated cores')
        take = min(slots, workers - len(result))
        result.extend([host] * take)
    if len(result) != workers:
        raise QueueError('Insufficient disjoint worker cores after service reservation')
    return result


def worker_step(allocation, binding, workers, hostfile, *, environ=None, query=None):
    environ = dict(os.environ if environ is None else environ)
    query = query or (lambda command: subprocess.check_output(command, text=True))
    hosts = query(['scontrol', 'show', 'hostnames', environ['NKGRID_JOB_NODELIST']]).split()
    # Placement uses allocated job CPUs captured before entering the service
    # step, not TASKS_PER_NODE (a rounded task-distribution hint). The latter
    # need not sum to the requested task count on a partially populated node.
    # worker_hosts still checks actual capacity, whole cores and service slots.
    cpu_counts = expand_counts(environ['NKGRID_JOB_CPUS_PER_NODE'])
    placement = worker_hosts(hosts, cpu_counts,
        allocation['cpu_per_task'], binding['hostname'], allocation['controller_task_slots'], workers)
    hostfile = Path(hostfile)
    hostfile.write_text(''.join(h + '\n' for h in placement))
    for name in ('SLURM_NTASKS_PER_NODE', 'SLURM_TASKS_PER_NODE', 'SLURM_MEM_PER_CPU',
                 'SLURM_MEM_PER_NODE', 'SLURM_CPUS_PER_TASK', 'SLURM_NNODES',
                 'SLURM_NTASKS', 'SLURM_NPROCS', 'SLURM_CPU_BIND', 'SLURM_CPU_BIND_LIST',
                 'SLURM_CPU_BIND_TYPE', 'SLURM_CPU_BIND_VERBOSE'):
        environ.pop(name, None)
    environ['SLURM_HOSTFILE'] = str(hostfile)
    command = ['srun', '--ntasks=' + str(workers), '--cpus-per-task=1', '--ntasks-per-core=1',
        '--exclusive', '--nodes=' + str(len(set(placement))), '--distribution=arbitrary',
        '--threads-per-core=1', '--cpu-bind=cores',
        '--mem=' + str(allocation['worker_step_memory_mb']) + 'M']
    return command, environ
