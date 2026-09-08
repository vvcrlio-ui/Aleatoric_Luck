"""Discoverer CPU bootstrap, sharing the BMRC experiment and dynamic queue engine.

Only standard-library code runs on the login node. Numerical imports, pip,
FFC preparation and task-table generation run inside the bootstrap allocation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

import experiment as common

DEFAULT_MODULE = "python/3/3.12/3.12.4"


def wall_seconds(value):
    """Accept documented Slurm minute and day/hour clock forms, max 48 hours."""
    if not re.fullmatch(r"(?:\d+-)?\d+(?::\d{1,2}){0,2}", value):
        raise ValueError("Discoverer time must be a finite Slurm duration, at most 48 hours")
    day, clock = value.split("-", 1) if "-" in value else (None, value)
    fields = [int(part) for part in clock.split(":")]
    if any(part >= 60 for part in fields[1:]):
        raise ValueError("invalid Slurm clock fields")
    if day is not None:
        fields += [0] * (3 - len(fields))
        seconds = int(day) * 86400 + fields[0] * 3600 + fields[1] * 60 + fields[2]
    elif len(fields) == 1:
        seconds = fields[0] * 60
    elif len(fields) == 2:
        seconds = fields[0] * 60 + fields[1]
    else:
        seconds = fields[0] * 3600 + fields[1] * 60 + fields[2]
    if not 0 < seconds <= 172800:
        raise ValueError("Discoverer jobs must be positive and at most 48 hours")
    return seconds


def configure(args):
    if args.target != "slurm":
        raise ValueError("--profile discoverer requires slurm")
    if not args.account or not re.fullmatch(r"[A-Za-z0-9_.-]+", args.account):
        raise ValueError("Discoverer requires explicit --account YOUR_PROJECT_ACCOUNT")
    args.qos = args.qos or args.account
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.qos):
        raise ValueError("Discoverer QoS must be a simple account/QoS name")
    if args.resume:
        if args.prepare_ffc or args.ffc_data_dir or args.refresh_env:
            raise ValueError("resume reuses prepared inputs and environment; no preparation or refresh allowed")
        if args.plan_time or args.plan_memory:
            raise ValueError("resume reuses bootstrap resources; do not override --plan-time/--plan-memory")
        return
    if args.schema and args.prepare_ffc:
        raise ValueError("choose --schema or --prepare-ffc, not both")
    if args.ffc_data_dir and not args.prepare_ffc:
        raise ValueError("--ffc-data-dir requires --prepare-ffc")
    # Ignore a BMRC constraint accidentally inherited from the login shell.
    if args.constraint not in (None, "none"):
        raise ValueError("Discoverer CPU uses --constraint none; unset NKGRID_CONSTRAINT")
    args.constraint = "none"
    args.partition = args.partition or "cn"
    if args.partition != "cn":
        raise ValueError("This Discoverer CPU profile supports partition cn")
    args.time_limit = args.time_limit or ("48:00:00" if args.preset in ("timing_full", "production") else "01:00:00")
    args.plan_time = args.plan_time or "02:00:00"
    for value in (args.time_limit, args.plan_time):
        wall_seconds(value)
    args.workers = args.workers or (496 if args.preset in ("timing_full", "production") else 16)
    args.rounds = args.rounds or 2
    args.memory = args.memory or "16G"
    args.plan_memory = args.plan_memory or "48G"


def batch_environment():
    # Slurm CLI flags override only named options; inherited SBATCH_* can inject
    # dependencies, arrays, export modes or a GPU constraint into another site.
    env = {key: value for key, value in os.environ.items() if not key.startswith("SBATCH_")}
    env.update({key: "1" for key in common.THREADS})
    env['SLURM_EXPORT_ENV'] = 'ALL'
    return env


def bootstrap_command(spec, request):
    output = Path(spec["output"])
    return ["sbatch", "--parsable", "--job-name=nkgrid-discoverer-bootstrap",
            "--nodes=1", "--ntasks-per-node=1", "--ntasks-per-core=1", "--cpus-per-task=1",
            "--account=" + spec["cluster"]["account"], "--qos=" + spec["cluster"]["qos"],
            "--partition=cn", "--mem=" + spec["plan_memory"], "--time=" + spec["plan_time"],
            "--chdir=" + str(output), "--output=" + str(output / "logs/bootstrap-%j.out"),
            "--error=" + str(output / "logs/bootstrap-%j.err"), "--export=ALL",
            str(common.ROOT / "launch/discoverer_bootstrap.sbatch"), str(request),
            spec["bootstrap"]["python_module"], str(common.ROOT / "launch/experiment.py")]


def resumed_spec(args, plan_path):
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    spec = json.loads((plan_path.parent / "launch.json").read_text(encoding="utf-8"))
    if spec.get("profile") != "discoverer":
        raise ValueError("resume requires a Discoverer launch.json beside plan.json")
    for field in ("account", "qos"):
        if getattr(args, field) != plan["submission"].get(field):
            raise ValueError(f"resume cannot override frozen {field}")
    if args.constraint not in (None, "none"):
        raise ValueError("resume cannot override frozen constraint")
    if args.venv and str(common.path_from_repo(args.venv)) != spec["bootstrap"]["venv"]:
        raise ValueError("resume cannot override frozen venv")
    spec["resume_plan"] = str(plan_path)
    spec["resume_plan_sha256"] = common.sha256(plan_path)
    spec["bootstrap"]["refresh_env"] = False
    return spec


def launch(args, spec):
    if args.resume:
        spec = resumed_spec(args, common.path_from_repo(args.resume))
    else:
        output = Path(spec["output"])
        # A per-run venv avoids reusing a partially installed/cancelled setup or
        # mutating an environment used by a previous queued/running experiment.
        spec["bootstrap"] = {
            "venv": str(common.path_from_repo(args.venv)) if args.venv else str(output / "venv"),
            "refresh_env": args.refresh_env,
            "python_module": os.environ.get("PYTHON_MODULE", DEFAULT_MODULE),
            "prepare_ffc": args.prepare_ffc,
            "ffc_data_dir": str(common.path_from_repo(args.ffc_data_dir or "FFCWS/data/private")),
        }
    if args.dry_run:
        print(json.dumps({"launch": spec, "actions": ["submit compute-node bootstrap",
              "install/validate environment", "prepare FFC if requested", "build plan and submit existing dynamic queue chain"],
              "note": "No data reads, installation or submission. Worker count is array concurrency, one CPU per worker."}, indent=2))
        return
    if sys.platform == "win32":
        raise ValueError("Run this command in a Linux cluster login shell; --dry-run works locally")
    source = common.frozen_source()
    if source["dirty"]:
        raise ValueError("Discoverer requires a clean committed checkout; commit changes before submission")
    output = Path(spec["output"])
    if args.resume:
        common.validate_source(spec)
    else:
        if output.exists():
            raise FileExistsError(f"run directory already exists: {output}")
        if output.is_relative_to(common.ROOT):
            result = subprocess.run(["git", "check-ignore", "-q", str(output / "launch.json")], cwd=common.ROOT)
            if result.returncode:
                raise ValueError("Discoverer output inside the checkout must be Git-ignored (use runs/)")
        spec.update(source=source, manifest_sha256=common.sha256(spec["manifest"]))
        if spec["schema"]:
            spec["schema_sha256"] = common.sha256(spec["schema"])
        output.mkdir(parents=True)
        (output / "logs").mkdir()
    request = output / ("resume-" + uuid.uuid4().hex + ".json" if args.resume else "launch.json")
    common.atomic_json(request, spec)
    result = common.command(bootstrap_command(spec, request), capture=True, env=batch_environment())
    job = result.stdout.strip()
    if not re.fullmatch(r"[0-9]+(?:;[A-Za-z0-9_.-]+)?", job):
        raise ValueError(f"Unrecognized sbatch receipt: {job!r}; inspect scheduler before retrying")
    receipt = request.with_name(request.stem + ".submission.json")
    common.atomic_json(receipt, {"bootstrap_job": job, "request": str(request)})
    print(f"Run directory: {output}\nBootstrap job: {job}\nReceipt: {receipt}")


def prepare_ffc(spec):
    import yaml
    from aleatoric_nk_grid.run_panels import resolved_panels
    panels = resolved_panels(Path(spec["manifest"]), only={spec["panel"]}, preset=spec["preset"])
    if len(panels) != 1:
        raise ValueError("FFC preparation requires exactly one declared panel")
    _, config = panels[0]
    match = re.fullmatch(r"ffc_(median_mode|median_missing_indicator|tree_ordinal)_(.+)", spec["panel"])
    if not match or match[2] != config.outcome:
        raise ValueError("--prepare-ffc requires a standard FFC panel name and outcome")
    models = spec["models"] or list(config.models)
    if len(set(models)) != len(models) or not set(models).issubset(config.models):
        raise ValueError("--models must be a unique subset of the panel models")
    config_path = common.ROOT / "FFCWS/adapter/config/ffc.yaml"
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = Path(spec["bootstrap"]["ffc_data_dir"])
    for key, name in (("background", "background.dta"), ("train", "train.csv"), ("test", "test.csv")):
        source = data / name
        if not source.is_file():
            raise FileNotFoundError(f"Missing FFC input: {source}")
        document["paths"][key] = str(source)
    root = Path(spec["output"]) / "prepared"
    root.mkdir(exist_ok=False)
    for key, name in (("output_root", "work"), ("ard_root", "ard"), ("schema_root", "schema")):
        document["paths"][key] = str(root / name)
    document["outcomes"] = [config.outcome]
    document["strategies"] = [match[1]]
    prepared_config = root / "adapter.yaml"
    prepared_config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    common.command([sys.executable, common.ROOT / "FFCWS/adapter/adapter.py", "--config", prepared_config,
                    "--strategy", match[1], "--validation-model", *models, "--min-n", str(config.min_n), "--seed", str(config.seed)])
    spec["schema"] = str(root / "schema" / (spec["panel"] + ".json"))
    spec["schema_sha256"] = common.sha256(spec["schema"])


def bootstrap(request):
    if not request or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("bootstrap requires a request and a Slurm compute allocation")
    spec = json.loads(Path(request).read_text(encoding="utf-8"))
    common.validate_source(spec)
    options = spec["bootstrap"]
    if sys.platform == "win32" or not (3, 11) <= sys.version_info[:2] < (3, 15):
        raise ValueError("Discoverer bootstrap requires Linux and Python 3.11–3.14")
    if os.environ.get("PYTHON_MODULE") != options["python_module"]:
        raise ValueError("bootstrap Python module differs from submitted request")
    # Compute-node /tmp can be too small for dependency wheels. Keep both
    # pip's cache and temporary downloads/builds on the project filesystem.
    temporary = Path(spec["output"]) / "tmp" / ("bootstrap-" + os.environ["SLURM_JOB_ID"])
    temporary.mkdir(parents=True, exist_ok=True)
    for key in ("TMPDIR", "TEMP", "TMP"):
        os.environ[key] = str(temporary)
    import tempfile
    tempfile.tempdir = None
    os.environ["PIP_CACHE_DIR"] = str(Path(spec["output"]) / "pip-cache")
    print(f"Bootstrap temporary directory: {temporary}", flush=True)
    python, venv = common.ensure_environment(argparse.Namespace(venv=options["venv"], refresh_env=options["refresh_env"]))
    environment = batch_environment()
    environment.update(VENV=str(venv), PYTHON=str(python), ENGINE_DIR=str(common.ROOT / "NK_Grid"))
    if spec.get("resume_plan"):
        if common.sha256(spec["resume_plan"]) != spec["resume_plan_sha256"]:
            raise ValueError("resume plan changed while queued")
        common.command(["bash", common.ROOT / "NK_Grid/slurm/submit_flat_task_table.sh", "--submit", spec["resume_plan"]],
                       cwd=Path(spec["resume_plan"]).parent, env=environment)
        return
    # Relaunch inside the verified venv before importing adapter/engine modules.
    common.command([python, common.ROOT / "launch/discoverer.py", "prepare-execute", request], env=environment)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "prepare-execute" or not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Internal compute-node entry point")
    spec = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    common.validate_source(spec)
    if spec["bootstrap"]["prepare_ffc"]:
        prepare_ffc(spec)
    common.atomic_json(Path(spec["output"]) / "prepared-launch.json", spec)
    common.execute(spec)
