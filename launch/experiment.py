"""Single-command bootstrap and shared single-model cluster orchestration.

Numerical work runs in the shared dependency environment and imports the engine from this checkout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[1]
THREADS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
ENGINE_SRC = ROOT / "NK_Grid/src"
# Launcher processes import the engine of this checkout, never one installed elsewhere.
if str(ENGINE_SRC) not in sys.path:
    sys.path.insert(0, str(ENGINE_SRC))


def command(args, *, capture=False, cwd=ROOT, env=None):
    return subprocess.run([str(a) for a in args], check=True, cwd=cwd, env=env,
                          text=True, capture_output=capture)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def path_from_repo(value):
    path = Path(value).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("target", choices=("slurm", "execute", "bootstrap", "recover-bootstrap"))
    p.add_argument("--profile", help="Site defaults loaded by run.sh from launch/profiles/NAME.sh")
    p.add_argument("--manifest", default="FFCWS/panels.yaml")
    p.add_argument("--panel", default="ffc_median_mode_gpa")
    p.add_argument("--preset", choices=("dev", "medium", "timing_full", "production", "pilot"), default="dev")
    p.add_argument("--schema", help="Use existing prepared data via its schema; never rewrite tracked schema")
    p.add_argument("--models", nargs="+", help="Optional explicit subset of the panel's models")
    p.add_argument("--allow-large-run", action="store_true")
    p.add_argument("--checkpoints", choices=("keep", "delete"),
                   help="Keep (default) or delete checkpoint data only after verified success")
    p.add_argument("--dry-run", action="store_true", help="Read-only launch preview; no installation, data reads or submission")
    p.add_argument("--resume", help="Resume a run from its plan.json")
    p.add_argument("--account", help="Required for Slurm, including resume; explicitly enter your authorized project account")
    p.add_argument("--qos", help="Slurm QoS; overrides the profile default")
    p.add_argument("--prepare", action="store_true", help="Prepare the selected panel on the bootstrap node")
    p.add_argument("--data-dir", help="Directory containing the raw files declared in the panel manifest")
    p.add_argument("--constraint")
    p.add_argument("--partition")
    p.add_argument("--time", dest="time_limit")
    p.add_argument("--workers", type=positive, help="Optional worker cap; live capacity determines the default")
    p.add_argument("--rounds", type=positive, help="Maximum continuation rounds; all clusters submit only the current worker allocation")
    p.add_argument("--memory", help="Base-worker memory request; default 2G")
    p.add_argument('--dispatcher-shards', type=int, choices=range(1, 9), metavar='1..8',
                   help='New shared Slurm run: requested dispatcher shards for both base and SL; >1 defaults to two validators per shard')
    p.add_argument('--scheduler-policy', help='New shared Slurm run: operational policy JSON; explicit shard option overrides this file')
    p.add_argument("--plan-memory")
    p.add_argument("--plan-time", help="Bootstrap and controller time; default 2h")
    p.add_argument("--request", help=argparse.SUPPRESS)
    return p


def wall_seconds(value):
    """Parse a positive finite Slurm duration in minutes or day/hour clock form."""
    if not re.fullmatch(r"(?:\d+-)?\d+(?::\d{1,2}){0,2}", value):
        raise ValueError("Time must be a finite Slurm duration")
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
    if seconds <= 0:
        raise ValueError("Job time must be positive")
    return seconds


def launch_spec(args):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.panel):
        raise ValueError("panel name may contain only letters, digits, _, . and -")
    if args.resume and (args.schema or args.models or args.preset != "dev"
                        or args.manifest != "FFCWS/panels.yaml" or args.panel != "ffc_median_mode_gpa"
                        or any((args.workers, args.rounds, args.partition, args.time_limit, args.memory,
                                args.dispatcher_shards, args.scheduler_policy, args.prepare, args.data_dir,
                                args.plan_time, args.plan_memory, args.constraint))):
        raise ValueError("resume reuses frozen design/resources; do not combine it with design/resource overrides")
    if not args.account or not args.account.strip():
        raise ValueError("Slurm requires explicit --account YOUR_PROJECT_ACCOUNT (including resume); no default account is used")
    if args.prepare and args.schema:
        raise ValueError("--prepare and --schema are mutually exclusive")
    if bool(args.prepare) != bool(args.data_dir):
        raise ValueError("--prepare requires --data-dir, and --data-dir requires --prepare")
    if args.preset == "production" and not args.allow_large_run and not args.dry_run:
        raise ValueError("production requires explicit --allow-large-run")
    defaults = os.environ if args.profile else {}
    partition = args.partition or defaults.get("NKGRID_PARTITION")
    if not args.resume and not partition:
        raise ValueError("Slurm requires --partition or a profile partition")
    maximum = defaults.get("NKGRID_MAX_TIME")
    time_limit = args.time_limit or (maximum if args.preset in ("timing_full", "production") else "01:00:00")
    if not args.resume and not time_limit:
        raise ValueError("timing_full/production requires --time or a profile maximum time")
    plan_time = args.plan_time or "02:00:00"
    for value in (time_limit, plan_time):
        if value is not None:
            seconds = wall_seconds(value)
            if maximum and seconds > wall_seconds(maximum):
                raise ValueError("Requested time exceeds the profile maximum")
    default_qos = defaults.get("NKGRID_QOS")
    qos = args.qos or (args.account if default_qos == "account" else default_qos)
    cluster = dict(account=args.account, qos=qos,
                   constraint=args.constraint or defaults.get("NKGRID_CONSTRAINT") or "none",
                   partition=partition, time_limit=time_limit,
                   workers=args.workers or 1, rounds=args.rounds or 2,
                   memory_override=args.memory or "2G")
    for value in [*cluster.values(), args.plan_time, args.plan_memory]:
        if isinstance(value, str) and ("\n" in value or "\r" in value or "\x00" in value):
            raise ValueError("scheduler fields must be single-line strings")
    manifest = path_from_repo(args.manifest)
    if not manifest.is_file():
        raise ValueError(f"manifest does not exist: {manifest}")
    output = manifest.parent / "outputs" / (args.panel + "-" + uuid.uuid4().hex[:12])
    result = dict(format_version=1, profile=args.profile, panel=args.panel, preset=args.preset,
                manifest=str(manifest), schema=str(path_from_repo(args.schema)) if args.schema else None,
                models=args.models, output=str(output), allow_large_run=args.allow_large_run,
                cluster=cluster, plan_memory=args.plan_memory or "48G",
                checkpoint_retention=args.checkpoints or "keep",
                plan_time=plan_time)
    result["continuation"] = {"worker_cap": args.workers, "max_rounds": args.rounds or 2,
                              "max_no_progress_rounds": 2, "max_control_failures": 3,
                              "max_control_jobs": 4 * (args.rounds or 2) + 8}
    result["bootstrap"] = {"shared_env_root": str(shared_env_root()),
                           "python_module": defaults.get("PYTHON_MODULE", ""),
                           "prepare": args.prepare,
                           "data_dir": str(path_from_repo(args.data_dir)) if args.data_dir else None}
    if args.dispatcher_shards is not None or args.scheduler_policy is not None:
        if args.resume:
            raise ValueError('Dispatcher options apply to a new shared Slurm run; existing rounds retain their snapshot')
        policy = json.loads(path_from_repo(args.scheduler_policy).read_text(encoding='utf-8')) if args.scheduler_policy else {}
        if not isinstance(policy, dict): raise ValueError('Scheduler policy must be a JSON object')
        if args.dispatcher_shards is not None:
            policy['dispatcher_shards'] = args.dispatcher_shards
            if args.dispatcher_shards > 1 and 'validation_processes' not in policy:
                policy['validation_processes'] = 2
        # The policy module and its shared-queue types have only stdlib imports.
        from aleatoric_nk_grid.scheduler_policy import validate_policy
        result['scheduler_policy'] = validate_policy(policy)
    return result


@contextmanager
def environment_lock(environment):
    import fcntl
    environment.parent.mkdir(parents=True, exist_ok=True)
    # Persistent inode: removing it after unlock would race another launcher.
    with (environment.parent / (environment.name + ".nkgrid-env.lock")).open("a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)



def shared_env_root():
    """Where cluster runs keep shared dependency environments, beside the checkouts by default."""
    return Path(os.environ.get("NKGRID_ENV_ROOT") or ROOT.parent / "nkgrid-envs")


def shared_environment_path(root):
    """Content-keyed location: same locked dependencies and Python module, same environment."""
    key = {"requirements": (ROOT / "NK_Grid/requirements.txt").read_text(encoding="utf-8"),
           "python_module": os.environ.get("PYTHON_MODULE", ""),
           "cpu_type": os.environ.get("MODULE_CPU_TYPE", "")}
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return Path(root).expanduser().resolve() / ("deps-" + digest)


def ensure_shared_environment(root):
    """Third-party dependencies only, installed once and never changed afterwards.

    Several experiments and checkouts use one environment, so it holds no project
    code: every process imports aleatoric_nk_grid from its own checkout through
    PYTHONPATH. A missing stamp means an interrupted build, rebuilt under the lock.
    """
    environment = shared_environment_path(root)
    python = environment / "bin/python"
    stamp = environment / ".nkgrid-shared-environment.json"
    expected = {"requirements": sha256(ROOT / "NK_Grid/requirements.txt"),
                "python_module": os.environ.get("PYTHON_MODULE", ""),
                "cpu_type": os.environ.get("MODULE_CPU_TYPE", "")}
    with environment_lock(environment):
        if not stamp.exists():
            pip = {**os.environ, "PIP_CACHE_DIR": os.environ.get("PIP_CACHE_DIR") or str(Path(root) / "pip-cache")}
            command([sys.executable, "-m", "venv", "--clear", environment])
            command([python, "-m", "pip", "install", "-r", ROOT / "NK_Grid/requirements.txt"], env=pip)
            command([python, "-m", "pip", "check"])
            atomic_json(stamp, expected)
        elif json.loads(stamp.read_text()) != expected:
            raise ValueError(f"Shared environment {environment} does not match its key; do not edit it in place")
        probe_code = ("import pathlib,sys,aleatoric_nk_grid as n,lightgbm,xgboost; "
                      "assert pathlib.Path(n.__file__).resolve()==pathlib.Path(sys.argv[1]).resolve(), 'engine imported from another checkout'; print('NKGRID shared environment ready')")
        command([python, "-c", probe_code, ENGINE_SRC / "aleatoric_nk_grid/__init__.py"],
                env={**os.environ, "PYTHONPATH": str(ENGINE_SRC)})
    return python, environment


def frozen_source():
    return {"commit": command(["git", "rev-parse", "HEAD"], capture=True).stdout.strip(),
            "dirty": bool(command(["git", "status", "--porcelain"], capture=True).stdout.strip())}


def validate_source(spec):
    if sha256(Path(spec["manifest"])) != spec["manifest_sha256"]:
        raise ValueError("manifest changed after launch; create a new run")
    if spec.get("schema") and sha256(spec["schema"]) != spec["schema_sha256"]:
        raise ValueError("schema changed after launch; create a new run")
    state = frozen_source()
    if state["commit"] != spec["source"]["commit"] or state["dirty"]:
        raise ValueError("checkout changed after submission or is dirty; use an immutable clean checkout")


def resolve_experiment(spec):
    from dataclasses import replace
    from aleatoric_nk_grid.run_panels import resolved_panels
    from aleatoric_nk_grid.ingest import load_input
    from aleatoric_nk_grid.validate_input import validate_input
    from aleatoric_nk_grid.nk_grid import resolve_input_grids, LARGE_RUN_THRESHOLD
    panels = resolved_panels(Path(spec["manifest"]), only={spec["panel"]}, preset=spec["preset"],
                             schema=Path(spec["schema"]) if spec["schema"] else None)
    if len(panels) != 1:
        raise ValueError(f"expected one panel named {spec['panel']}, found {len(panels)}")
    _, config = panels[0]
    models = tuple(spec["models"] or config.models)
    if len(set(models)) != len(models) or not set(models).issubset(config.models):
        raise ValueError("--models must be a unique subset of the declared panel models")
    config = replace(config, out=Path(spec["output"]) / "final.csv", models=models,
                     schema=Path(spec["schema"]) if spec["schema"] else config.schema,
                     n_jobs=1, allow_large_run=spec["allow_large_run"],
                     checkpoint_retention=spec["checkpoint_retention"])
    try:
        loaded = load_input(config.schema, config.outcome)
    except FileNotFoundError as exc:
        raise ValueError("Prepared input missing. Use --schema to point to existing ARD/schema, "
                         "or launch with --prepare --data-dir DIR.") from exc
    loaded, groups = validate_input(loaded, config.outcome, models=config.models,
                                  min_n=config.min_n, test_size=config.test_size, seed=config.seed)
    n_grid, k_grid = resolve_input_grids(config, loaded, groups)
    config = replace(config, n_grid=tuple(map(int, n_grid)), k_grid=tuple(map(int, k_grid)))
    count = len(n_grid) * len(k_grid) * config.n_seeds * config.n_draws * len(config.models)
    if config.repeat_plan is not None:
        count = len(n_grid) * len(k_grid) * len(config.repeat_plan) * len(config.models)
    if count > LARGE_RUN_THRESHOLD and not config.allow_large_run:
        raise ValueError(f"{count:,} model cells require --allow-large-run")
    print(json.dumps({"panel": spec["panel"], "model_cells": count, "N": n_grid.tolist(), "K": k_grid.tolist()}, ensure_ascii=False), flush=True)
    return config


def execute(spec):
    validate_source(spec)
    config = resolve_experiment(spec)
    from aleatoric_nk_grid.cluster_queue import prepare
    from cluster_scheduler import start
    output = Path(spec["output"])
    atomic_json(output / 'cluster-environment.json', {'python': str(Path(sys.executable).absolute()),
                'python_module': os.environ.get('PYTHON_MODULE', '')})
    plan_path = prepare(config, spec, ROOT)
    start(plan_path)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(argv)
    if args.target != 'slurm' and (args.dispatcher_shards is not None or args.scheduler_policy is not None):
        raise ValueError('Dispatcher options apply to new shared Slurm launches')
    if args.target == "recover-bootstrap":
        if not args.request:
            raise ValueError("recover-bootstrap requires --request")
        from bootstrap import submit_bootstrap
        print("Bootstrap job: " + submit_bootstrap(args.request))
        return
    if args.target == "bootstrap":
        if args.checkpoints is not None:
            raise ValueError("bootstrap reuses the frozen launch request; set --checkpoints on slurm instead")
        from bootstrap import bootstrap
        bootstrap(args.request)
        return
    if args.target == "execute":
        if not args.request:
            raise ValueError("execute requires --request")
        if args.checkpoints is not None:
            raise ValueError("execute reuses the frozen launch request; set --checkpoints on slurm instead")
        if sys.platform == "win32":
            raise ValueError("Full engine execution requires Linux/WSL")
        if not os.environ.get("SLURM_JOB_ID"):
            raise ValueError("execute requires a Slurm compute allocation")
        spec = json.loads(Path(args.request).read_text(encoding="utf-8"))
        validate_source(spec)
        if spec["bootstrap"]["prepare"]:
            from preparation import prepare_data
            prepare_data(spec)
        atomic_json(Path(spec["output"]) / "prepared-launch.json", spec)
        execute(spec)
        return
    if not args.resume and not any(a == "--preset" or a.startswith("--preset=") for a in argv):
        raise ValueError("New runs require an explicit --preset")
    spec = launch_spec(args)
    if args.resume:
        from cluster_scheduler import resume
        resume(args)
        return
    from bootstrap import launch
    launch(args, spec)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError, subprocess.CalledProcessError) as exc:
        print(f"NKGRID: {exc}", file=sys.stderr)
        raise SystemExit(2)
