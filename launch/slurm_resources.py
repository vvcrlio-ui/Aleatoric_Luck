"""Slurm command output, sizes and durations, and a partition's node geometry."""
from __future__ import annotations

import re
import subprocess


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


def memory_mb(value):
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([KMGTPE]?)", str(value), re.I)
    if not match:
        raise ValueError(f"Unrecognized Slurm memory: {value!r}")
    return float(match[1]) * 1024 ** ("KMGTPE".index(match[2].upper()) - 1 if match[2] else 0)


def duration(value):
    day, clock = str(value).split("-", 1) if "-" in str(value) else (None, str(value))
    parts = list(map(int, clock.split(":")))
    if day is not None:
        parts += [0] * (3 - len(parts))
        return int(day) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]
    return (parts[0] * 60 if len(parts) == 1 else parts[0] * 60 + parts[1]
            if len(parts) == 2 else parts[0] * 3600 + parts[1] * 60 + parts[2])


def node_geometry(partition, *, run=query):
    """CPUs, memory in MB and hardware threads per core of each node in the partition.

    One line per node: grouped output of a mixed partition reports "24+".
    """
    nodes = records(run(["sinfo", "-h", "-N", "-p", partition, "-o", "%c|%m|%Z"]), "cpu,mem,threads")
    if not nodes:
        raise ValueError(f"sinfo reports no nodes in partition {partition}")
    return nodes
