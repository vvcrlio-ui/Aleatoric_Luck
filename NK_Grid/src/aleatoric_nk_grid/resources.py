"""Pure Slurm resource requests shared by planners and launchers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceRequest:
    cpus_per_task: int
    partition: str
    memory: str
    time_limit: str
    account: str
    constraint: str


def sbatch_resource_args(request: ResourceRequest) -> tuple[str, ...]:
    if request.cpus_per_task != 1:
        raise ValueError("dynamic workers must request exactly one CPU")
    if not all((request.partition, request.memory, request.time_limit, request.account)):
        raise ValueError("partition, memory, time_limit, and account are required")
    args = (f"--partition={request.partition}", "--cpus-per-task=1", f"--mem={request.memory}", f"--time={request.time_limit}", f"--account={request.account}")
    return args if request.constraint == "none" else (*args, f"--constraint={request.constraint}")
