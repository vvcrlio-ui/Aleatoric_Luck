"""Compute-node bootstrap for the shared single-model cluster scheduler.

The login node uses only the standard library to check arguments, the checkout
and the run directory. Numerical imports, pip, data preparation and task-design
generation run inside the bootstrap allocation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import experiment as common
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
              "reuse or create the shared dependency environment", "prepare data if requested", "freeze single-model design", "start shared single-model scheduler"],
              "live_resources": {"workers": "unresolved until each round", "qos_account_partition_limits": "unresolved",
                                 "existing_jobs": "unresolved", "effective_wall_time": "unresolved"},
              "note": "No data reads, installation or submission. The planning placeholder worker count is not a resource decision. Each round resolves live limits and CPU-minute headroom, then submits one worker allocation; --workers is an optional cap and --rounds a hard bound."}, indent=2))
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
    python, venv = common.ensure_shared_environment(options["shared_env_root"])
    environment = batch_environment()
    environment.update(VENV=str(venv), PYTHON=str(python), ENGINE_DIR=str(common.ROOT / "NK_Grid"),
                       PYTHONPATH=str(common.ENGINE_SRC))
    # Relaunch inside the verified venv before importing adapter/engine modules.
    common.command([python, common.ROOT / "launch/experiment.py", "execute", "--request", request], env=environment)
