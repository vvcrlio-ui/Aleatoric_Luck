"""Live Slurm capacity for Discoverer. No site quota numbers are defaults.

Finite limits are conservatively intersected (including association ancestors
and partition QoS). This may undershoot a QoS override, never bypass it.
"""
from __future__ import annotations

from datetime import datetime, timezone
import getpass
import math
import re
import subprocess

QOS_FIELDS = "Name,MaxJobsPU,MaxSubmitPU,MaxJobsPA,MaxSubmitPA,GrpJobs,GrpSubmit,MaxWall,GrpTRES,MaxTRESPU,MaxTRESPA,MaxTRES,Flags"
ASSOC_FIELDS = "Cluster,Account,User,Partition,ParentName,MaxJobs,MaxSubmitJobs,GrpJobs,GrpSubmitJobs,GrpTRES,MaxTRES,MaxWall"


def query(command):
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=90)
    return result.stdout


def records(text, fields):
    names = fields.split(",")
    result = []
    for line in text.splitlines():
        values = line.strip().split("|")
        if len(values) == len(names) + 1 and values[-1] == "":
            values.pop()
        if len(values) != len(names):
            raise ValueError(f"Slurm returned {len(values)} fields; expected {len(names)}")
        result.append(dict(zip(names, values)))
    return result


def finite(value):
    if str(value).upper() in ("", "UNLIMITED", "INFINITE", "N/A", "NONE", "(NULL)", "-1"):
        return None
    return int(value)


def memory_mb(value):
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([KMGTPE]?)(?:[cn])?", str(value), re.I)
    if not match:
        raise ValueError(f"Unrecognized Slurm memory: {value!r}")
    return float(match[1]) * 1024 ** ("KMGTPE".index(match[2].upper()) - 1 if match[2] else 0)


def job_memory_mb(job):
    value = job["mem"]
    multiplier = float(job["cpu"]) if value.endswith("c") else float(job["node"]) if value.endswith("n") else 1
    return memory_mb(value) * multiplier


def duration(value):
    if str(value).upper() in ("", "UNLIMITED", "INFINITE", "N/A", "NONE", "(NULL)", "-1"):
        return None
    day, clock = str(value).split("-", 1) if "-" in str(value) else (None, str(value))
    parts = list(map(int, clock.split(":")))
    if day is not None:
        parts += [0] * (3 - len(parts))
        return int(day) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]
    return (parts[0] * 60 if len(parts) == 1 else parts[0] * 60 + parts[1]
            if len(parts) == 2 else parts[0] * 3600 + parts[1] * 60 + parts[2])


def clock_string(seconds):
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def tres(value):
    result = {}
    for item in value.split(","):
        if not item:
            continue
        key, number = item.split("=", 1)
        if number.upper() in ("UNLIMITED", "INFINITE", "N/A", "-1"):
            continue
        result[key] = memory_mb(number) if key == "mem" else float(number)
    return result


def key_values(text):
    return dict(re.findall(r"(?:^|\s)(\w+)=([^\s]+)", text))


def snapshot(account, qos, partition="cn", *, run=query, user=None):
    user = user or getpass.getuser()
    part = key_values(run(["scontrol", "show", "partition", partition, "-o"]))
    config = key_values(run(["scontrol", "show", "config"]))
    # scontrol config uses spaces around '=' unlike partition output.
    if "ClusterName" not in config:
        raw = run(["scontrol", "show", "config"])
        config = dict(re.findall(r"^\s*(\w+)\s*=\s*([^\n]+?)\s*$", raw, re.M))
    qnames = [qos]
    if part.get("QoS", "N/A") not in ("N/A", "(null)", "", qos):
        qnames.append(part["QoS"])
    qrows = records(run(["sacctmgr", "-nP", "show", "qos", "where", "name=" + ",".join(qnames),
                         "format=" + QOS_FIELDS]), QOS_FIELDS)
    if set(qnames) != {row["Name"] for row in qrows}:
        raise ValueError("Effective job/partition QoS could not be resolved")
    associations = records(run(["sacctmgr", "-nP", "show", "assoc", "format=" + ASSOC_FIELDS]), ASSOC_FIELDS)
    associations = [row for row in associations if row["Cluster"] == config.get("ClusterName")]
    if not associations:
        raise ValueError("Cannot resolve associations for the live cluster")
    jobs = records(run(["squeue", "-h", "-r", "-o", "%i|%u|%a|%q|%P|%T|%C|%m|%D|%j"]),
                   "id,user,account,qos,partition,state,cpu,mem,node,name")
    nodes = records(run(["sinfo", "-h", "-p", partition, "-o", "%c|%m|%Z"]), "cpu,mem,threads")
    return {"queried_at_utc": datetime.now(timezone.utc).isoformat(), "user": user, "account": account,
            "qos": qos, "partition": part, "qos_rows": qrows, "associations": associations,
            "jobs": jobs, "config": config, "nodes": nodes}


def capacity(live, *, remaining, memory, requested_time, worker_cap=None,
             extra_submit=2, extra_running=1):
    """Bound additional workers, reserving only actual pending control roles.

    All existing jobs, including this controller and pending jobs, count against
    their scoped limits. Reserving pending work is deliberately conservative.
    extra_submit covers the successor and recovery guard; extra_running covers
    the brief guard/worker overlap. Admission remains Slurm's responsibility.
    """
    user, account, qos = live["user"], live["account"], live["qos"]
    part, config, jobs = live["partition"], live["config"], live["jobs"]
    partition = part["PartitionName"]
    if part.get("State", "UP") != "UP":
        raise ValueError("Partition is not UP")
    for field, target in (("AllowAccounts", account), ("AllowQos", qos)):
        allowed = part.get(field, part.get(field.replace("Qos", "QOS"), "ALL"))
        if allowed not in ("ALL", "(null)") and target not in allowed.split(","):
            raise ValueError(f"Partition {field} excludes {target}")
    ancestors = {account}
    parents = {r["Account"]: r["ParentName"] for r in live["associations"] if not r["User"]}
    current = account
    while parents.get(current) and parents[current] not in ancestors:
        current = parents[current]; ancestors.add(current)
    selected = [r for r in live["associations"] if r["Account"] in ancestors
                and r["User"] in ("", user) and r["Partition"] in ("", partition)]
    if not any(r["Account"] == account and r["User"] == user for r in selected):
        raise ValueError("No live user/account/partition association")
    def descendants(name):
        result = {name}
        while True:
            more = {child for child, parent in parents.items() if parent in result}
            if more <= result:
                return result
            result |= more
    threads = max(int(n["threads"]) for n in live["nodes"])
    allocated_cpu = threads if "CR_CORE" in config.get("SelectTypeParameters", "").upper() else 1
    request = {"cpu": allocated_cpu, "mem": memory_mb(memory), "node": 1}
    choices = [(int(remaining), "remaining executable task groups")]
    if worker_cap:
        choices.append((int(worker_cap), "explicit worker cap"))
    maximum_array = finite(config.get("MaxArraySize", ""))
    if maximum_array is None:
        raise ValueError("Cannot resolve MaxArraySize")
    choices.append((maximum_array, "Slurm MaxArraySize"))
    walls = [duration(requested_time)]
    def count_limit(value, scope, label, reserve):
        limit = finite(value)
        if limit is not None:
            choices.append((limit - len(scope) - reserve, label))
    def tres_limit(value, scope, label):
        limits = tres(value)
        for key in ("cpu", "mem", "node"):
            if key not in limits:
                continue
            # Pending requests also reserve headroom. Group node usage is an
            # upper bound because multiple jobs may share the same node.
            used = sum(job_memory_mb(j) if key == "mem" else float(j[key]) for j in scope)
            choices.append((math.floor((limits[key] - used) / request[key]) - extra_running,
                            label + "/" + key))
    def per_job(value, label):
        for key, limit in tres(value).items():
            if key in request and request[key] > limit:
                raise ValueError(f"Worker request exceeds {label}/{key}")
    for row in live["qos_rows"]:
        partition_qos = row["Name"] != qos
        qjobs = [j for j in jobs if j["partition"].rstrip("*") == partition] if partition_qos else [j for j in jobs if j["qos"] == qos]
        scopes = [("PU", [j for j in qjobs if j["user"] == user]),
                  ("PA", [j for j in qjobs if j["account"] == account])]
        for suffix, scope in scopes:
            count_limit(row["MaxJobs" + suffix], scope, "QoS " + row["Name"] + " MaxJobs" + suffix, extra_running)
            count_limit(row["MaxSubmit" + suffix], scope, "QoS " + row["Name"] + " MaxSubmit" + suffix, extra_submit)
            tres_limit(row["MaxTRES" + suffix], scope, "QoS " + row["Name"] + " MaxTRES" + suffix)
        count_limit(row["GrpJobs"], qjobs, "QoS " + row["Name"] + " GrpJobs", extra_running)
        count_limit(row["GrpSubmit"], qjobs, "QoS " + row["Name"] + " GrpSubmit", extra_submit)
        tres_limit(row["GrpTRES"], qjobs, "QoS " + row["Name"] + " GrpTRES")
        per_job(row["MaxTRES"], "QoS MaxTRES")
        walls.append(duration(row["MaxWall"]))
    for row in selected:
        accounts = descendants(row["Account"])
        scope = [j for j in jobs if j["account"] in accounts and (not row["Partition"] or j["partition"] == partition)
                 and (not row["User"] or j["user"] == user)]
        own = [j for j in scope if j["user"] == user]
        label = "association " + "/".join(row[k] or "*" for k in ("Account", "User", "Partition"))
        count_limit(row["MaxJobs"], own, label + " MaxJobs", extra_running)
        count_limit(row["MaxSubmitJobs"], own, label + " MaxSubmitJobs", extra_submit)
        count_limit(row["GrpJobs"], scope, label + " GrpJobs", extra_running)
        count_limit(row["GrpSubmitJobs"], scope, label + " GrpSubmitJobs", extra_submit)
        tres_limit(row["GrpTRES"], scope, label + " GrpTRES")
        per_job(row["MaxTRES"], label + " MaxTRES")
        walls.append(duration(row["MaxWall"]))
    walls.append(duration(part.get("MaxTime", "")))
    partition_jobs = [j for j in jobs if j["partition"].rstrip("*") == partition and j["state"] != "PENDING"]
    tres_limit("cpu=" + part["TotalCPUs"], partition_jobs, "partition total capacity")
    if all(request["mem"] > float(n["mem"]) for n in live["nodes"]):
        raise ValueError("No partition node can fit worker memory")
    for key in ("MaxMemPerCPU", "MaxMemPerNode"):
        maximum = finite(part.get(key, ""))
        needed = request["mem"] / request["cpu"] if key.endswith("CPU") else request["mem"]
        if maximum not in (None, 0) and needed > maximum:
            raise ValueError(f"Worker memory exceeds partition {key}")
    workers, binding = min(choices)
    if workers < 1:
        raise ValueError("No worker capacity: " + binding)
    return {"workers": workers, "time_limit": clock_string(min(w for w in walls if w is not None)),
            "allocated_cpu_per_worker_bound": allocated_cpu, "memory_mb": request["mem"],
            "binding_limit": binding, "limits": [{"available_workers": n, "source": label} for n, label in choices],
            "queried_at_utc": live["queried_at_utc"], "existing_user_jobs": sum(j["user"] == user for j in jobs),
            "reserved_extra_submit_jobs": extra_submit, "reserved_extra_running_jobs": extra_running,
            "note": "Conservative scoped admission bound; immediate resource allocation is not guaranteed."}
