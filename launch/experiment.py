"""Single-command environment bootstrap and existing-engine orchestration.

The bootstrap and dry run use only the standard library. Numerical work uses
the installed shared engine, never Windows lock/process compatibility shims.
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
    p.add_argument("target", choices=("local", "slurm", "execute", "bootstrap"))
    p.add_argument("--profile", choices=("local", "bmrc", "discoverer"), default="local")
    p.add_argument("--manifest", default="FFCWS/panels.yaml")
    p.add_argument("--panel", default="ffc_median_mode_gpa")
    p.add_argument("--preset", choices=("dev", "medium", "timing_full", "production", "pilot", "dev-dynamic"), default="dev")
    p.add_argument("--output", help="New run directory; defaults to <manifest directory>/outputs/<panel>-<unique ID>")
    p.add_argument("--schema", help="Use existing prepared data via its schema; never rewrite tracked schema")
    p.add_argument("--models", nargs="+", help="Optional explicit subset of the panel's models")
    p.add_argument("--venv", help="Environment path (or VENV); created only when absent")
    p.add_argument("--refresh-env", action="store_true", help="Reinstall fixed dependencies into the selected environment")
    p.add_argument("--allow-large-run", action="store_true")
    p.add_argument("--max-jobs", type=positive, help="Local-only bound on model cells")
    p.add_argument("--checkpoints", choices=("keep", "delete"),
                   help="Keep or delete checkpoint data only after verified success; omission preserves the configured/default behavior")
    p.add_argument("--dry-run", action="store_true", help="Read-only launch preview; no installation, data reads or submission")
    p.add_argument("--resume", help="Slurm: reuse an existing plan JSON; do not regenerate the task table")
    p.add_argument("--account", help="Required for Slurm, including resume; explicitly enter your authorized project account")
    p.add_argument("--qos", help="Discoverer QoS; defaults to the explicit account")
    p.add_argument("--prepare-ffc", action="store_true", help="Discoverer: prepare selected FFC panel on a compute node")
    p.add_argument("--ffc-data-dir", help="Directory containing background.dta, train.csv and test.csv")
    p.add_argument("--constraint", default=os.environ.get("NKGRID_CONSTRAINT"))
    p.add_argument("--partition")
    p.add_argument("--time", dest="time_limit")
    p.add_argument("--workers", type=positive, help="Worker cap; Discoverer timing_full/production resolves live capacity by default")
    p.add_argument("--rounds", type=positive, help="Maximum continuation rounds; Discoverer submits only the current round")
    p.add_argument("--memory", help="Optional worker memory request, e.g. 16G")
    p.add_argument("--plan-memory")
    p.add_argument("--plan-time", help="Planning job time; production 8h, other presets 1h")
    p.add_argument("--request", help=argparse.SUPPRESS)
    return p


def launch_spec(args):
    if args.profile == "discoverer":
        from discoverer import configure
        configure(args)
    elif args.qos or args.prepare_ffc or args.ffc_data_dir:
        raise ValueError("--qos, --prepare-ffc and --ffc-data-dir require --profile discoverer")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.panel):
        raise ValueError("panel name may contain only letters, digits, _, . and -")
    if args.target == "slurm" and args.max_jobs:
        raise ValueError("--max-jobs is local-only; use --preset dev for a small Slurm run")
    if args.resume and args.target != "slurm":
        raise ValueError("--resume accepts a Slurm plan; local runs use the engine checkpoint entry point")
    if args.resume and (args.output or args.schema or args.models or args.preset != "dev"
                        or args.manifest != "FFCWS/panels.yaml" or args.panel != "ffc_median_mode_gpa"
                        or any((args.workers, args.rounds, args.partition, args.time_limit, args.memory))):
        raise ValueError("resume reuses frozen design/resources; do not combine it with design/resource overrides")
    if args.target == "slurm" and (not args.account or not args.account.strip()):
        raise ValueError("Slurm requires explicit --account YOUR_PROJECT_ACCOUNT (including resume); no default account is used")
    if args.target == "slurm" and not args.resume and not args.constraint:
        raise ValueError("Slurm requires --profile bmrc or explicit --constraint")
    if args.preset == "production" and not args.allow_large_run and not args.dry_run:
        raise ValueError("production requires explicit --allow-large-run")
    production = args.preset == "production"
    cluster = dict(account=args.account, constraint=args.constraint,
                   partition=args.partition or ("long" if production else "short"),
                   time_limit=args.time_limit or ("10-00:00:00" if production else "01:00:00"),
                   workers=args.workers or (600 if production else 32), rounds=args.rounds or (4 if production else 2),
                   memory_override=args.memory)
    if args.profile == "discoverer":
        cluster.update(qos=args.qos, single_node=True, workers=args.workers or 1)
    for value in [*cluster.values(), args.plan_time, args.plan_memory]:
        if isinstance(value, str) and ("\n" in value or "\r" in value or "\x00" in value):
            raise ValueError("scheduler fields must be single-line strings")
    manifest = path_from_repo(args.manifest)
    if not manifest.is_file():
        raise ValueError(f"manifest does not exist: {manifest}")
    output = path_from_repo(args.output) if args.output else manifest.parent / "outputs" / (args.panel + "-" + uuid.uuid4().hex[:12])
    result = dict(format_version=1, target=args.target, profile=args.profile, panel=args.panel, preset=args.preset,
                manifest=str(manifest), schema=str(path_from_repo(args.schema)) if args.schema else None,
                models=args.models, output=str(output), allow_large_run=args.allow_large_run,
                max_jobs=args.max_jobs, cluster=cluster, plan_memory=args.plan_memory or "16G",
                checkpoint_retention=args.checkpoints or "default",
                plan_time=args.plan_time or ("08:00:00" if production else "01:00:00"))
    if args.profile == "discoverer":
        result["continuation"] = {"worker_cap": args.workers, "max_rounds": args.rounds or 2,
                                  "max_no_progress_rounds": 2, "max_control_failures": 3,
                                  "max_control_jobs": 4 * (args.rounds or 2) + 8}
    return result


def validate_resume_checkpoints(plan_path, requested=None):
    """A resumed dynamic run keeps the policy frozen in its snapshot."""
    plan_path = Path(plan_path)
    if not plan_path.is_file():
        raise ValueError(f"plan does not exist: {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    snapshot_path = plan.get("snapshot")
    if not snapshot_path:
        if requested is None:
            return
        raise ValueError("cannot confirm frozen checkpoint policy: plan has no snapshot")
    snapshot_path = Path(snapshot_path)
    if not snapshot_path.is_absolute():
        snapshot_path = plan_path.parent / snapshot_path
    if not snapshot_path.is_file():
        raise ValueError(f"cannot confirm frozen checkpoint policy: snapshot does not exist: {snapshot_path}")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    output_dir = snapshot.get("output_dir")
    if output_dir:
        output_dir = Path(output_dir)
        if not output_dir.is_absolute():
            output_dir = snapshot_path.parent / output_dir
        if (output_dir / "checkpoint-archive.json").exists():
            raise ValueError("run already completed and its checkpoints were archived/deleted; final CSV is retained; do not resume training")
    frozen = snapshot.get("config", {}).get("checkpoint_retention", "default")
    if frozen not in ("default", "keep", "delete"):
        raise ValueError(f"invalid frozen checkpoint_retention: {frozen!r}")
    # Historical dynamic runs retain WAL when no retention policy was set.
    effective = "keep" if frozen == "default" else frozen
    if requested is not None and requested != effective:
        raise ValueError(f"resume cannot override frozen checkpoint policy ({effective}); omit --checkpoints or use --checkpoints {effective}")
    return effective


def ensure_environment(args):
    """Install only on first use or explicit refresh; bind to source and lock files."""
    environment = path_from_repo(args.venv or os.environ.get("VENV", ".venv-linux"))
    python = environment / "bin/python"
    stamp = environment / ".nkgrid-launch-environment.json"
    expected = {"repo": str(ROOT), "requirements": sha256(ROOT / "NK_Grid/requirements.txt"),
                "project": sha256(ROOT / "NK_Grid/pyproject.toml"),
                "python_module": os.environ.get("PYTHON_MODULE", ""),
                "cpu_type": os.environ.get("MODULE_CPU_TYPE", "")}
    with environment_lock(environment):
        created = False
        if not python.is_file():
            if environment.exists():
                raise ValueError(f"Existing directory is not a Linux venv: {environment}; choose --venv explicitly")
            command([sys.executable, "-m", "venv", environment])
            created = True
        recorded = json.loads(stamp.read_text()) if stamp.exists() else None
        probe = command([python, "-c", "import importlib.metadata as m,json; print(json.dumps({d.metadata['Name'].lower():d.version for d in m.distributions()}))"], capture=True)
        installed = json.loads(probe.stdout)
        required = dict(line.strip().lower().split("==") for line in
                        (ROOT / "NK_Grid/requirements.txt").read_text().splitlines() if "==" in line)
        matches = all(installed.get(name) == version for name, version in required.items())
        if not created and not args.refresh_env and ((recorded is not None and recorded != expected) or not matches):
            raise ValueError("Existing environment differs from this checkout/locked dependencies. Choose a new --venv, or use --refresh-env only when no running jobs use it.")
        if created or args.refresh_env:
            command([python, "-m", "pip", "install", "-r", ROOT / "NK_Grid/requirements.txt"])
            command([python, "-m", "pip", "install", "--no-deps", "-e", ROOT / "NK_Grid"])
        # Verify an existing unmarked environment without reinstalling into it.
        probe_code = ("import pathlib,sys,aleatoric_nk_grid as n,lightgbm,xgboost; "
                      "assert pathlib.Path(n.__file__).resolve()==pathlib.Path(sys.argv[1]).resolve(), 'editable installation points at another checkout'; print('NKGRID environment ready')")
        command([python, "-c", probe_code, ROOT / "NK_Grid/src/aleatoric_nk_grid/__init__.py"])
        command([python, "-m", "pip", "check"])
        atomic_json(stamp, expected)
    return python, environment


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


def frozen_source():
    return {"commit": command(["git", "rev-parse", "HEAD"], capture=True).stdout.strip(),
            "dirty": bool(command(["git", "status", "--porcelain"], capture=True).stdout.strip())}


def validate_source(spec):
    if sha256(Path(spec["manifest"])) != spec["manifest_sha256"]:
        raise ValueError("manifest changed after launch; create a new run")
    if spec.get("schema") and sha256(spec["schema"]) != spec["schema_sha256"]:
        raise ValueError("schema changed after launch; create a new run")
    state = frozen_source()
    if state["commit"] != spec["source"]["commit"] or (spec["target"] == "slurm" and state["dirty"]):
        raise ValueError("checkout changed after submission or is dirty; use an immutable clean checkout")


def resolve_experiment(spec):
    from dataclasses import replace
    from aleatoric_nk_grid.run_panels import resolved_panels
    from aleatoric_nk_grid.ingest import load_input
    from aleatoric_nk_grid.validate_input import validate_input
    from aleatoric_nk_grid.nk_grid import resolve_input_grids, LARGE_RUN_THRESHOLD
    panels = resolved_panels(Path(spec["manifest"]), only={spec["panel"]}, preset=spec["preset"])
    if len(panels) != 1:
        raise ValueError(f"expected one panel named {spec['panel']}, found {len(panels)}")
    _, config = panels[0]
    models = tuple(spec["models"] or config.models)
    if len(set(models)) != len(models) or not set(models).issubset(config.models):
        raise ValueError("--models must be a unique subset of the declared panel models")
    config = replace(config, out=Path(spec["output"]) / "final.csv", models=models,
                     schema=Path(spec["schema"]) if spec["schema"] else config.schema,
                     n_jobs=1, allow_large_run=spec["allow_large_run"],
                     checkpoint_retention=(config.checkpoint_retention
                                           if spec.get("checkpoint_retention", "default") == "default"
                                           else spec["checkpoint_retention"]))
    try:
        loaded = load_input(config.schema, config.outcome)
    except FileNotFoundError as exc:
        raise ValueError("Prepared input missing. Use --schema to point to existing ARD/schema, "
                         "or prepare the private dataset with its adapter once. No input is fabricated or overwritten.") from exc
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
    if spec["target"] == "local":
        from aleatoric_nk_grid.nk_grid import run_nk_grid
        run_nk_grid(config, max_jobs=spec["max_jobs"], allow_large_run=spec["allow_large_run"])
        return
    from aleatoric_nk_grid.chunk_planning import ClusterPolicy, build_dynamic_plan
    output = Path(spec["output"])
    plan = build_dynamic_plan(config, n_grid=config.n_grid, k_grid=config.k_grid,
                             cluster=ClusterPolicy(**spec["cluster"]), table_path=output / "tasks.parquet",
                             snapshot_path=output / "snapshot.json", output_dir=output / "out", panel=spec["panel"])
    plan_path = output / "plan.json"
    atomic_json(plan_path, plan)
    if spec.get("profile") == "discoverer":
        from discoverer_continuation import start
        start(spec, plan_path)
    else:
        command(["bash", ROOT / "NK_Grid/slurm/submit_flat_task_table.sh", "--submit", plan_path], cwd=output)


def slurm_command(spec, request):
    cluster = spec["cluster"]
    args = ["sbatch", "--parsable", "--job-name=nkgrid-plan-submit", "--cpus-per-task=1",
            "--chdir=" + spec["output"], "--output=" + str(Path(spec["output"]) / "logs/plan-%j.out"),
            "--error=" + str(Path(spec["output"]) / "logs/plan-%j.err"),
            "--account=" + cluster["account"], "--partition=" + cluster["partition"],
            "--mem=" + spec["plan_memory"], "--time=" + spec["plan_time"]]
    if cluster["constraint"] != "none":
        args += ["--constraint=" + cluster["constraint"]]
    return args + [str(ROOT / "launch/prepare_and_submit.sbatch"), str(request)]


def main(argv=None):
    args = parser().parse_args(argv)
    if args.target == "bootstrap":
        if args.checkpoints is not None:
            raise ValueError("bootstrap reuses the frozen launch request; set --checkpoints on slurm instead")
        from discoverer import bootstrap
        bootstrap(args.request)
        return
    if args.target == "execute":
        if not args.request:
            raise ValueError("execute requires --request")
        if args.checkpoints is not None:
            raise ValueError("execute reuses the frozen launch request; set --checkpoints on local or slurm instead")
        if sys.platform == "win32":
            raise ValueError("Full engine execution requires Linux/WSL")
        execute(json.loads(Path(args.request).read_text(encoding="utf-8")))
        return
    spec = launch_spec(args)
    if args.resume:
        resume_retention = validate_resume_checkpoints(path_from_repo(args.resume), args.checkpoints)
        if resume_retention is not None:
            spec["checkpoint_retention"] = resume_retention
    if args.profile == "discoverer":
        from discoverer import launch
        launch(args, spec)
        return
    if args.dry_run:
        print(json.dumps({"launch": spec, "resume": args.resume, "venv": args.venv or os.environ.get("VENV", ".venv-linux"),
                          "actions": ["validate/reuse or create environment", "reuse plan" if args.resume else
                                      ("run locally" if args.target == "local" else "submit planning job, then existing job chain")],
                          "note": "Read-only preview; input availability and resolved cell count checked at execution."}, indent=2))
        return
    if sys.platform == "win32":
        raise ValueError("Full local execution requires Linux/WSL; use run.ps1 with an installed WSL distribution")
    if not (3, 11) <= sys.version_info[:2] < (3, 15):
        raise ValueError("Python 3.11–3.14 required; select the cluster module or NKGRID_BOOTSTRAP_PYTHON")
    source = frozen_source()
    if args.target == "slurm" and source["dirty"]:
        raise ValueError("Slurm requires a clean committed checkout; commit changes before submission")
    if not args.resume:
        output = Path(spec["output"])
        if output.exists():
            raise FileExistsError(f"run directory already exists: {output}; choose a new --output or --resume its plan")
        if args.target == "slurm" and output.is_relative_to(ROOT):
            ignored = subprocess.run(["git", "check-ignore", "-q", str(output / "launch.json")], cwd=ROOT)
            if ignored.returncode != 0:
                raise ValueError("Slurm output inside the checkout must be Git-ignored (use runs/ or aleatoric-production/) so launching does not dirty the frozen checkout")
    if args.resume:
        plan = path_from_repo(args.resume)
        if not plan.is_file():
            raise ValueError(f"plan does not exist: {plan}")
        submission = json.loads(plan.read_text(encoding="utf-8")).get("submission", {})
        for field in ("account", "constraint"):
            supplied = getattr(args, field)
            if supplied is not None and supplied != submission.get(field):
                raise ValueError(f"resume cannot override frozen {field}")
    python, venv = ensure_environment(args)
    environment = {**os.environ, "VENV": str(venv), "PYTHON": str(python), "ENGINE_DIR": str(ROOT / "NK_Grid"),
                   **{key: "1" for key in THREADS}}
    if args.resume:
        command(["bash", ROOT / "NK_Grid/slurm/submit_flat_task_table.sh", "--submit", plan], cwd=plan.parent, env=environment)
        return
    output = Path(spec["output"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    spec.update(source=source, manifest_sha256=sha256(spec["manifest"]))
    if spec["schema"]:
        spec["schema_sha256"] = sha256(spec["schema"])
    request = output / "launch.json"
    atomic_json(request, spec)
    print(f"Run directory: {output}", flush=True)
    if args.target == "local":
        command([python, Path(__file__).resolve(), "execute", "--request", request], env=environment)
    else:
        result = command(slurm_command(spec, request), capture=True, env=environment)
        job_id = result.stdout.strip()
        if not re.fullmatch(r"[0-9]+(?:;[A-Za-z0-9_.-]+)?", job_id):
            raise ValueError(f"Unrecognized sbatch receipt: {job_id!r}; inspect scheduler before retrying")
        atomic_json(output / "submission.json", {"planning_job": job_id, "request": str(request)})
        print(f"Planning job: {job_id}; after planning, the existing Slurm chain is submitted automatically.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileExistsError, subprocess.CalledProcessError) as exc:
        print(f"NKGRID: {exc}", file=sys.stderr)
        raise SystemExit(2)
