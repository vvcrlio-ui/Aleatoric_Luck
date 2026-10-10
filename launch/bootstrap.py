"""Bootstrap job and worker allocations of one run.

The login node uses only the standard library to check arguments, the checkout
and the run directory. Numerical imports, pip, data preparation and the frozen
plan come from the bootstrap job, which then submits one worker allocation.
A resumed run submits another allocation for the cells still missing.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import experiment as common
import slurm_resources as resources
from slurm_submission import Journal, Slurm, _lock, batch_environment, now, read

def bootstrap_command(spec, request):
    output = Path(spec["output"])
    cluster = spec["cluster"]
    args = ["sbatch", "--parsable", "--job-name=nkgrid-bootstrap",
            "--nodes=1", "--ntasks-per-node=1", "--ntasks-per-core=1", "--cpus-per-task=1",
            "--account=" + cluster["account"], "--partition=" + cluster["partition"],
            "--mem=" + spec["plan_memory"], "--time=" + spec["plan_time"],
            "--chdir=" + str(output), "--output=" + str(output / "logs/bootstrap-%j.out"),
            "--error=" + str(output / "logs/bootstrap-%j.err"), "--export=ALL"]
    if cluster.get("qos"):
        args.append("--qos=" + cluster["qos"])
    if cluster["constraint"] != "none":
        args.append("--constraint=" + cluster["constraint"])
    return args + [str(common.ROOT / "launch/bootstrap.sbatch"), str(request),
                   spec["bootstrap"]["python_module"], str(common.ROOT / "launch/experiment.py")]



def submit_bootstrap(request, *, slurm=None):
    """Initial accepted-but-response-lost recovery, without creating another run."""
    request = Path(request).resolve()
    spec = read(request)
    common.validate_source(spec)
    path = request.parent / "bootstrap-journal.json"
    with _lock(request.parent / ".bootstrap.lock"):
        digest = common.sha256(request)
        if path.exists():
            state = read(path)
            if state["request_sha256"] != digest:
                raise ValueError("Bootstrap request identity changed")
        else:
            state = {"run_id": uuid.uuid4().hex, "jobs": {}, "request_sha256": digest,
                     "request": str(request), "created_at_utc": now()}
        journal = Journal(path, state, slurm or Slurm(spec))
        arguments = [a for a in bootstrap_command(spec, request)[2:] if not a.startswith("--job-name=")]
        return journal.submit("B0", arguments)


def launch(args, spec):
    if args.dry_run:
        print(json.dumps({"launch": spec, "actions": ["submit compute-node bootstrap",
              "reuse or create the shared dependency environment", "prepare data if requested", "freeze the plan",
              "submit one worker allocation of --nodes nodes"],
              "note": "No data reads, installation or submission. Workers per node follow the partition's cores and --memory."}, indent=2))
        return
    if sys.platform == "win32":
        raise ValueError("Run this command in a Linux cluster login shell; --dry-run works locally")
    source = common.frozen_source()
    if source["dirty"]:
        raise ValueError("Slurm requires a clean committed checkout; commit changes before submission")
    output = Path(spec["output"])
    if output.exists():
        raise FileExistsError(f"run directory already exists: {output}")
    if output.is_relative_to(common.ROOT):
        result = subprocess.run(["git", "check-ignore", "-q", str(output / "launch.json")], cwd=common.ROOT)
        if result.returncode:
            raise ValueError("Output inside the checkout must be Git-ignored; the default <catalog>/outputs/ is")
    if spec["bootstrap"]["wheelhouse"]:
        common.download_wheels(spec["bootstrap"]["wheelhouse"])
    spec.update(source=source, manifest_sha256=common.sha256(spec["manifest"]))
    if spec["schema"]:
        spec["schema_sha256"] = common.sha256(spec["schema"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    request = output / "launch.json"
    common.atomic_json(request, spec)
    job = submit_bootstrap(request)
    receipt = request.with_name(request.stem + ".submission.json")
    common.atomic_json(receipt, {"bootstrap_job": job, "request": str(request)})
    print(f"Run directory: {output}\nBootstrap job: {job}\nReceipt: {receipt}")


def bootstrap(request):
    if not request or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("bootstrap requires a request and a Slurm compute allocation")
    spec = json.loads(Path(request).read_text(encoding="utf-8"))
    common.validate_source(spec)
    options = spec["bootstrap"]
    if sys.platform == "win32" or not (3, 11) <= sys.version_info[:2] < (3, 15):
        raise ValueError("Bootstrap requires Linux and Python 3.11–3.14")
    if os.environ.get("PYTHON_MODULE", "") != options["python_module"]:
        raise ValueError("bootstrap Python module differs from submitted request")
    # Compute-node /tmp can be too small for dependency wheels. Keep both
    # pip's cache and temporary downloads/builds on the project filesystem.
    temporary = Path(spec["output"]) / "tmp" / ("bootstrap-" + os.environ["SLURM_JOB_ID"])
    temporary.mkdir(parents=True, exist_ok=True)
    for key in ("TMPDIR", "TEMP", "TMP"):
        os.environ[key] = str(temporary)
    import tempfile
    tempfile.tempdir = None
    print(f"Bootstrap temporary directory: {temporary}", flush=True)
    os.environ["PIP_CACHE_DIR"] = str(Path(options["shared_env_root"]) / "pip-cache")
    python, venv = common.ensure_shared_environment(options["shared_env_root"], options["wheelhouse"])
    environment = batch_environment()
    environment.update(VENV=str(venv), PYTHON=str(python), ENGINE_DIR=str(common.ROOT / "NK_Grid"),
                       PYTHONPATH=str(common.ENGINE_SRC))
    # Relaunch inside the verified venv before importing adapter/engine modules.
    common.command([python, common.ROOT / "launch/experiment.py", "execute", "--request", request], env=environment)


def worker_arguments(spec, plan_path, python):
    """sbatch arguments of a worker allocation: one worker per physical core, as many as memory allows."""
    cluster = spec["cluster"]
    nodes = resources.node_geometry(cluster["partition"])
    threads = max(int(node["threads"]) for node in nodes)
    memory = resources.memory_mb(cluster["memory"])
    per_node = min(min(int(node["cpu"]) // threads, int(float(node["mem"]) // memory)) for node in nodes)
    if per_node < 1:
        raise ValueError("No node of the partition fits one worker with the requested --memory")
    output = Path(spec["output"])
    arguments = ["--nodes=" + str(cluster["nodes"]), "--ntasks-per-node=" + str(per_node),
                 "--cpus-per-task=" + str(threads), "--mem=" + str(int(per_node * memory)) + "M",
                 "--time=" + cluster["time_limit"], "--account=" + cluster["account"],
                 "--partition=" + cluster["partition"], "--chdir=" + str(output),
                 "--output=" + str(output / "logs/work-%j.out"), "--error=" + str(output / "logs/work-%j.err"),
                 "--export=ALL"]
    if cluster.get("qos"):
        arguments.append("--qos=" + cluster["qos"])
    if cluster["constraint"] != "none":
        arguments.append("--constraint=" + cluster["constraint"])
    return arguments + [str(common.ROOT / "launch/cells.sbatch"), str(plan_path), python,
                        str(common.ENGINE_SRC), spec["bootstrap"]["python_module"]]


def submit_allocation(spec, arguments):
    """Submit the next worker allocation unless an earlier one is still queued or running."""
    output = Path(spec["output"])
    with _lock(output / ".bootstrap.lock"):
        path = output / "bootstrap-journal.json"
        journal = Journal(path, read(path), Slurm(spec))
        labels = sorted((label for label in journal.state["jobs"] if label.startswith("W")), key=lambda label: int(label[1:]))
        if journal.active(labels):
            raise ValueError("A worker allocation of this run is still queued or running")
        label = "W" + str(len(labels) + 1)
        job = journal.submit(label, arguments)
    print(f"Worker allocation {label}: {job}", flush=True)
    return job


def submit_workers(plan_path, *, python):
    """Called by the bootstrap job once the plan is frozen."""
    plan_path = Path(plan_path)
    spec = read(plan_path.parent / "launch.json")
    arguments = worker_arguments(spec, plan_path, python)
    common.atomic_json(plan_path.parent / "allocation.json", {"arguments": arguments})
    return submit_allocation(spec, arguments)


def resume(args):
    """Login node: another allocation with the frozen arguments, for the cells still missing."""
    plan_path = Path(args.resume).resolve()
    output = plan_path.parent
    spec = read(output / "launch.json")
    common.validate_source(spec)
    if args.account != spec["cluster"]["account"]:
        raise ValueError("--account differs from the run's account")
    if (output / "verified.json").exists():
        raise ValueError("The run is complete; a completed run cannot be resumed")
    arguments = read(output / "allocation.json")["arguments"]
    if args.dry_run:
        print(json.dumps({"resume": str(plan_path), "sbatch": arguments}, indent=2))
        return None
    return submit_allocation(spec, arguments)
