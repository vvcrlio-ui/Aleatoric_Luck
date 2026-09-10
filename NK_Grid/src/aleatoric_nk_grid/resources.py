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
    qos: str | None = None
    single_node: bool = False


def sbatch_resource_args(request: ResourceRequest) -> tuple[str, ...]:
    if request.cpus_per_task != 1:
        raise ValueError("dynamic workers must request exactly one CPU")
    if not all((request.partition, request.memory, request.time_limit, request.account)):
        raise ValueError("partition, memory, time_limit, and account are required")
    args = (f"--partition={request.partition}", "--cpus-per-task=1", f"--mem={request.memory}", f"--time={request.time_limit}", f"--account={request.account}")
    if request.qos is not None:
        if not request.qos or any(c in request.qos for c in "\n\r\x00"):
            raise ValueError("qos must be a non-empty single-line string")
        args += (f"--qos={request.qos}",)
    if request.single_node:
        args += ("--nodes=1", "--ntasks-per-node=1", "--ntasks-per-core=1")
    return args if request.constraint == "none" else (*args, f"--constraint={request.constraint}")
