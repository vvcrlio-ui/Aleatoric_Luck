"""Persist allocation geometry; current usage may delay it, never resize it."""
import getpass
import math
from pathlib import Path
import re

import discoverer_resources as base
from cluster_resources import default_qos, minute_headroom


class CapacityWait(RuntimeError):
    pass


def snapshot(spec, *, run=base.query):
    c = spec["cluster"]
    qos = c.get("qos") or default_qos(c["account"], run=run)
    live = base.snapshot(c["account"], qos, c["partition"], run=run)
    constraint = c["constraint"]
    if constraint != "none" and not re.fullmatch(r"[\w.-]+", constraint):
        raise ValueError("Suite resource freezing accepts one node feature, or --constraint none")
    nodes = base.records(run(["sinfo", "-N", "-h", "-p", c["partition"], "-o", "%N|%c|%m|%Z|%f"]),
                         "name,cpu,mem,threads,features")
    nodes = {n["name"]: n for n in nodes if constraint == "none" or constraint in n["features"].split(",")}
    if not nodes:
        raise ValueError("No nodes match the selected partition/constraint")
    live["nodes"] = list(nodes.values())
    usage = run(["scontrol", "show", "assoc_mgr"])
    live["qos_minute_headroom"] = [minute_headroom(usage, row["Name"]) for row in live["qos_rows"]]
    return live


def limits(live):
    """Yield whole-job and scoped aggregate TRES/count/wall limits."""
    user, account, qos = live["user"], live["account"], live["qos"]
    part = live["partition"]["PartitionName"]
    parents = {r["Account"]: r["ParentName"] for r in live["associations"] if not r["User"]}
    ancestors = {account}; current = account
    while parents.get(current) and parents[current] not in ancestors:
        current = parents[current]; ancestors.add(current)
    selected = [r for r in live["associations"] if r["Account"] in ancestors
                and r["User"] in ("", user) and r["Partition"] in ("", part)]
    if not any(r["Account"] == account and r["User"] == user for r in selected):
        raise ValueError("No user/account/partition association")
    def descendants(name):
        found = {name}
        while True:
            children = {k for k, v in parents.items() if v in found}
            if children <= found:
                return found
            found |= children
    for row in live["qos_rows"]:
        jobs = [j for j in live["jobs"] if j["qos"] == qos] if row["Name"] == qos else [j for j in live["jobs"] if j["partition"].rstrip("*") == part]
        for suffix, scope in (("PU", [j for j in jobs if j["user"] == user]),
                              ("PA", [j for j in jobs if j["account"] == account])):
            yield row.get("MaxTRES" + suffix, ""), scope, row.get("MaxJobs" + suffix, ""), row.get("MaxSubmit" + suffix, ""), False, None
        yield row.get("GrpTRES", ""), jobs, row.get("GrpJobs", ""), row.get("GrpSubmit", ""), False, None
        yield row.get("MaxTRES", ""), [], "", "", True, base.duration(row.get("MaxWall", ""))
    for row in selected:
        accounts = descendants(row["Account"])
        jobs = [j for j in live["jobs"] if j["account"] in accounts
                and (not row["User"] or j["user"] == user)
                and (not row["Partition"] or j["partition"].rstrip("*") == part)]
        own = [j for j in jobs if j["user"] == user]
        yield "", own, row.get("MaxJobs", ""), row.get("MaxSubmitJobs", ""), False, None
        yield row.get("GrpTRES", ""), jobs, row.get("GrpJobs", ""), row.get("GrpSubmitJobs", ""), False, None
        yield row.get("MaxTRES", ""), [], "", "", True, base.duration(row.get("MaxWall", ""))


def check_partition(live):
    p = live["partition"]
    for field, target in (("AllowAccounts", live["account"]), ("AllowQos", live["qos"])):
        allowed = p.get(field, p.get(field.replace("Qos", "QOS"), "ALL"))
        if allowed not in ("ALL", "(null)") and target not in allowed.split(","):
            raise ValueError(field + " excludes requested account/QoS")
    if p.get("State", "UP") != "UP":
        raise CapacityWait("Partition is not UP")


def geometry(spec, live):
    check_partition(live)
    c = spec["cluster"]; mem = math.ceil(base.memory_mb(c["memory_override"] or "16G"))
    overhead = math.ceil(base.memory_mb(spec["plan_memory"]))
    threads = max(int(n["threads"]) for n in live["nodes"])
    slots = min(min(int(n["cpu"]) // threads, (int(n["mem"]) - overhead) // mem) for n in live["nodes"])
    max_mem = base.finite(live["partition"].get("MaxMemPerNode", ""))
    if max_mem:
        slots = min(slots, (max_mem - overhead) // mem)
    if slots < 1:
        raise ValueError("Worker memory plus controller reserve does not fit selected nodes")
    max_nodes = min(len(live["nodes"]), base.finite(live["partition"].get("MaxNodes", "")) or len(live["nodes"]))
    minimum_nodes = base.finite(live["partition"].get("MinNodes", "")) or 1
    constraints = list(limits(live))
    # Two controller CPUs/memory reservations are outside the worker allocation.
    candidates = []
    for per_node in range(1, slots + 1):
        for nodes in range(minimum_nodes, min(max_nodes, math.ceil((c["workers"] + 1) / per_node)) + 1):
            tasks = min(c["workers"] + 1, nodes * per_node)
            if tasks < max(2, nodes):
                continue
            request = dict(cpu=nodes * per_node * threads, mem=nodes * (per_node * mem + overhead), node=nodes)
            fits = True
            for tres, jobs, running, submitted, per_job, wall in constraints:
                extra = {} if per_job else dict(cpu=2 * threads, mem=2 * overhead, node=2)
                if any(request.get(k, 0) + extra.get(k, 0) > v for k, v in base.tres(tres).items() if k in request):
                    fits = False; break
                if any(base.finite(v) is not None and base.finite(v) < 3 for v in (running, submitted)):
                    fits = False; break
            if fits:
                candidates.append((tasks, -request["cpu"], -nodes, per_node, request))
    if not candidates:
        raise ValueError("Hard limits cannot fit a worker, dispatcher and controller reserves")
    tasks, _, negnodes, slots, request = max(candidates, key=lambda x: x[:3])
    return dict(format="bmrc-suite-resources-v1", account=c["account"], user=live["user"],
                partition=c["partition"], constraint=c["constraint"], qos=live["qos"],
                nodes=-negnodes, workers=tasks - 1, tasks_per_node=slots, cpus_per_task=1,
                ntasks_per_core=1, threads_per_core_bound=threads, memory_mb_per_node=slots * mem + overhead,
                allocated_cpu_bound=request["cpu"], worker_memory_mb=mem,
                controller_memory_mb=overhead, worker_cap=c["workers"])


def validate_saved(spec, saved):
    c = spec["cluster"]
    if saved.get("format") != "bmrc-suite-resources-v1":
        raise ValueError("Unknown shared resource format")
    expected = dict(account=c["account"], user=getpass.getuser())
    for field in ("partition", "constraint", "qos"):
        if field in spec["explicit_resources"]:
            expected[field] = c[field]
    if "workers" in spec["explicit_resources"]:
        expected["worker_cap"] = c["workers"]
    if "memory" in spec["explicit_resources"]:
        expected["worker_memory_mb"] = math.ceil(base.memory_mb(c["memory_override"]))
    if "plan_memory" in spec["explicit_resources"]:
        expected["controller_memory_mb"] = math.ceil(base.memory_mb(spec["plan_memory"]))
    for key, value in expected.items():
        if saved.get(key) != value:
            raise ValueError("Shared resources conflict at " + key + "; use a new --resources file")
    for key in ("nodes", "workers", "tasks_per_node", "memory_mb_per_node", "allocated_cpu_bound", "worker_memory_mb", "controller_memory_mb", "threads_per_core_bound", "worker_cap"):
        if type(saved.get(key)) is not int or saved[key] < 1:
            raise ValueError("Invalid shared resource field: " + key)
    if saved["nodes"] > saved["workers"] + 1 or saved["workers"] + 1 > saved["nodes"] * saved["tasks_per_node"]:
        raise ValueError("Shared allocation task/node counts disagree")
    if (saved.get("cpus_per_task") != 1 or saved.get("ntasks_per_core") != 1
            or saved["workers"] > saved["worker_cap"]
            or saved["allocated_cpu_bound"] != saved["nodes"] * saved["tasks_per_node"] * saved["threads_per_core_bound"]
            or saved["memory_mb_per_node"] != saved["tasks_per_node"] * saved["worker_memory_mb"] + saved["controller_memory_mb"]):
        raise ValueError("Shared allocation CPU/memory geometry is inconsistent")


def resolve(spec, *, run=base.query):
    import experiment as common
    from aleatoric_nk_grid.shared_queue import file_lock
    path = Path(spec["resources"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(path.with_name(path.name + ".lock")):
        if path.exists():
            saved = common.json.loads(path.read_bytes()); validate_saved(spec, saved)
        else:
            saved = geometry(spec, snapshot(spec, run=run)); common.atomic_json(path, saved)
    return saved


def admission(spec, allocation, *, run=base.query):
    """Validate the same allocation. Slurm handles temporary resource contention."""
    from copy import deepcopy
    effective = deepcopy(spec)
    effective["cluster"].update({k: allocation[k] for k in ("account", "partition", "constraint", "qos")})
    live = snapshot(effective, run=run); check_partition(live)
    eligible = [n for n in live["nodes"] if int(n["threads"]) <= allocation["threads_per_core_bound"]
                and int(n["cpu"]) >= allocation["tasks_per_node"] * allocation["threads_per_core_bound"]
                and int(n["mem"]) >= allocation["memory_mb_per_node"]]
    if len(eligible) < allocation["nodes"]:
        raise ValueError("Matching hardware no longer fits the frozen allocation")
    requested_time = base.duration(spec["cluster"]["time_limit"])
    walls = [base.duration(live["partition"].get("MaxTime", ""))]
    request = dict(cpu=allocation["allocated_cpu_bound"], mem=allocation["nodes"] * allocation["memory_mb_per_node"], node=allocation["nodes"])
    for tres, jobs, running, submitted, per_job, wall in limits(live):
        walls.append(wall)
        if any(request[k] > v for k, v in base.tres(tres).items() if k in request):
            raise ValueError("Frozen allocation exceeds changed hard limits; create a new resource file")
        # MaxSubmit limits can reject sbatch itself. Running/TRES contention is
        # left to Slurm rather than shrinking a pressure-test allocation.
        if base.finite(submitted) is not None and len(jobs) + 2 > base.finite(submitted):
            raise CapacityWait("No submission slot for frozen allocation")
    if any(w is not None and requested_time > w for w in walls):
        raise ValueError("Requested --time exceeds a partition/account/QoS limit")
    control_cpus = max(int(n["threads"]) for n in live["nodes"])
    needed = request["cpu"] * requested_time / 60 + 2 * control_cpus * base.duration(spec["plan_time"]) / 60
    if any(b is not None and needed > b for b in live["qos_minute_headroom"]):
        raise CapacityWait("Insufficient CPU-minute headroom for the frozen allocation; resources were not resized")
    return {"checked_at_utc": live["queried_at_utc"], "allocation": allocation,
            "time_limit": spec["cluster"]["time_limit"]}
