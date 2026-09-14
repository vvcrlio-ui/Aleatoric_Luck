"""One-command BMRC suite bootstrap and journaled bounded continuation.

Only the standard library is needed on the login node. Numerical imports and
environment provisioning are confined to allocated compute nodes.
"""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import uuid

import experiment as common

sys.path.insert(0, str(common.ROOT / "NK_Grid/src"))
FORMAT = "multi-panel-slurm-v1"
OUTCOMES = ("grit", "materialHardship", "eviction", "layoff", "jobTraining")
STRATEGIES = ("median_mode", "median_missing_indicator", "tree_ordinal")
PANELS = tuple("ffc_" + s + "_" + o for s in STRATEGIES for o in OUTCOMES)


def read(path):
    return json.loads(Path(path).read_bytes())


def explicit(argv, option):
    return any(a == "--" + option or a.startswith("--" + option + "=") for a in argv)


def spec_for(args, argv):
    if args.target != "slurm" or args.profile != "bmrc":
        raise ValueError("ffc_non_gpa suite requires slurm --profile bmrc")
    if not explicit(argv, "preset"):
        raise ValueError("Suite requires an explicit --preset")
    if args.suite and explicit(argv, "panel"):
        raise ValueError("--suite and --panel are mutually exclusive")
    panels = list(PANELS) if args.suite else [args.panel]
    if not args.suite and (not explicit(argv, "panel") or args.panel not in PANELS):
        raise ValueError("Raw BMRC single-panel preparation requires one of the fifteen non-GPA --panel names")
    if args.schema or args.models or args.max_jobs or args.run or args.manifest != "FFCWS/panels.yaml":
        raise ValueError("ffc_non_gpa uses its 15 declared panels and nine models; use --panel for overrides")
    if not args.account or not args.account.strip():
        raise ValueError("An explicit --account is required")
    if not args.ffc_data_dir:
        raise ValueError("Suite requires --ffc-data-dir with background.dta, train.csv and test.csv")
    if args.preset == "production" and not (args.allow_large_run or args.dry_run):
        raise ValueError("production requires --allow-large-run")
    full = args.preset in ("timing_full", "production")
    production = args.preset == "production"
    root = common.path_from_repo(args.output) if args.output else common.ROOT / "FFCWS/outputs" / ("ffc_non_gpa-" + uuid.uuid4().hex[:12])
    resource_path = common.path_from_repo(args.resources) if args.resources else (common.ROOT / "FFCWS/outputs" / "bmrc-resources.json" if full else root / "resources.json")
    cluster = dict(account=args.account, qos=args.qos, partition=args.partition or ("long" if full else "short"),
                   constraint=args.constraint or "skl-compat", workers=args.workers or (600 if full else 32),
                   memory_override=args.memory or "16G", rounds=args.rounds or (4 if production else 2),
                   time_limit=args.time_limit or ("10-00:00:00" if production else "24:00:00" if full else "01:00:00"))
    from discoverer_resources import duration, memory_mb
    for v in cluster.values():
        if isinstance(v, str) and any(c in v for c in ("\n", "\r", "\x00")):
            raise ValueError("Slurm arguments must be single-line values")
    if not duration(cluster["time_limit"]) or duration(cluster["time_limit"]) <= 0:
        raise ValueError("--time must be a positive finite duration")
    if memory_mb(cluster["memory_override"]) <= 0:
        raise ValueError("--memory must be positive")
    if memory_mb(args.plan_memory or "16G") <= 0 or (duration(args.plan_time or "08:00:00") or 0) <= 0:
        raise ValueError("Planning memory and time must be positive")
    value = dict(format=FORMAT, suite=args.suite, target="slurm", profile="bmrc", preset=args.preset,
                 output=str(root), resources=str(resource_path), data=str(common.path_from_repo(args.ffc_data_dir)),
                 panels=panels, cluster=cluster, checkpoint_retention=args.checkpoints or "keep",
                 allow_large_run=args.allow_large_run, plan_memory=args.plan_memory or "16G",
                 plan_time=args.plan_time or "08:00:00", source=None,
                 explicit_resources=[k for k in ("partition", "constraint", "qos", "workers", "memory", "plan_memory") if explicit(argv, k)],
                 bootstrap=dict(python=str(Path(sys.executable).absolute()), python_module=os.environ.get("PYTHON_MODULE", ""),
                    venv=str(common.path_from_repo(args.venv)) if args.venv else str(root / "env"),
                    owned_environment=not bool(args.venv), refresh_env=args.refresh_env))
    if resource_path.exists():
        from suite_resources import validate_saved
        saved = read(resource_path); validate_saved(value, saved)
        cluster.update({k: saved[k] for k in ("account", "partition", "constraint", "qos")})
        value["plan_memory"] = str(saved["controller_memory_mb"]) + "M"
    return value


def validate(spec):
    from aleatoric_nk_grid.suite_queue import check_files
    current = common.frozen_source()
    if current["dirty"] or current["commit"] != spec["source"]["commit"]:
        raise ValueError("Use the original clean checkout; code changed since launch")
    check_files(spec["files"])


def checked_plan(root):
    from aleatoric_nk_grid.suite_queue import check_files
    check_files([read(Path(root) / "plan-file.json")])
    return read(Path(root) / "plan.json")


def publish_plan(root, plan):
    """Publish the metadata binding before the plan becomes visible."""
    from aleatoric_nk_grid.suite_queue import file_metadata
    from aleatoric_nk_grid.shared_queue import sync_directory
    root = Path(root); stage = root / "plan.pending.json"
    common.atomic_json(stage, plan)
    metadata = file_metadata(stage); metadata["path"] = str((root / "plan.json").resolve())
    common.atomic_json(root / "plan-file.json", metadata)
    os.replace(stage, root / "plan.json"); sync_directory(root)


def batch_args(spec, mode, target, *, allocation=None, dependency=None, delayed=False):
    c = spec["cluster"]; root = Path(spec["output"])
    frozen = allocation or c
    args = ["--account=" + frozen["account"], "--partition=" + frozen["partition"],
            "--cpus-per-task=1", "--ntasks-per-core=1", "--export=ALL", "--no-requeue",
            "--chdir=" + str(root), "--output=" + str(root / "logs" / (mode + "-%j.out")),
            "--error=" + str(root / "logs" / (mode + "-%j.err"))]
    if frozen.get("qos"):
        args.append("--qos=" + frozen["qos"])
    if frozen.get("constraint") != "none":
        args.append("--constraint=" + frozen["constraint"])
    if dependency:
        args.append("--dependency=afterany:" + str(dependency))
    if delayed:
        args.append("--begin=now+10minutes")
    if allocation:
        args += ["--nodes=" + str(allocation["nodes"]), "--ntasks=" + str(allocation["workers"] + 1),
                 "--ntasks-per-node=" + str(allocation["tasks_per_node"]),
                 "--mem=" + str(allocation["memory_mb_per_node"]) + "M", "--time=" + c["time_limit"]]
    else:
        args += ["--nodes=1", "--ntasks=1", "--mem=" + spec["plan_memory"], "--time=" + spec["plan_time"]]
    python = spec["bootstrap"]["python"] if mode == "bootstrap" else str(Path(spec["bootstrap"]["venv"]) / "bin/python")
    return args + [str(common.ROOT / "launch/cluster_queue.sbatch"), python,
                   str(common.ROOT / "launch/suite.py"), mode, str(target), spec["bootstrap"]["python_module"]]


def state_for(spec):
    from aleatoric_nk_grid.suite_queue import file_metadata, check_files
    path = Path(spec["output"]) / "suite-state.json"
    if path.exists():
        state = read(path)
        if state["request_id"] != spec["request_id"]:
            raise ValueError("Run request changed")
        check_files(state.get("request_file", []))
    else:
        rounds = spec["cluster"]["rounds"]
        state = dict(format=FORMAT, run_id=uuid.uuid4().hex, request_id=spec["request_id"],
                     jobs={}, rounds=[], status="new", round_limit=rounds, control_limit=4 * rounds + 12,
                     completed=0, no_progress=0, error=None)
        request = path.with_name("launch.json")
        state["request_file"] = [file_metadata(request)] if request.exists() else []
    return path, state


def start(spec, *, resume=False, slurm=None):
    from discoverer_continuation import Journal, Slurm, TERMINAL, _lock
    root = Path(spec["output"])
    validate(spec)
    with _lock(root / ".suite-state.lock"):
        path, state = state_for(spec)
        journal = Journal(path, state, slurm or Slurm(spec["cluster"]["account"], spec["cluster"].get("qos")))
        journal.save()
        for label in list(state["jobs"]):
            job = journal.submit(label, [])
            if any(s not in TERMINAL for s in journal.slurm.states(job)):
                print("Suite already active: " + job); return job
        if (root / "plan.json").exists():
            frozen = checked_plan(root)
            spec = frozen["launch"]
        if (root / "verified.json").exists() and (spec["checkpoint_retention"] == "keep" or (root / "checkpoint-archive.json").exists()):
            print("Suite complete: " + str(root / "summary.json")); return None
        if resume:
            state["round_limit"] = max(state["round_limit"], len(state["rounds"]) + spec["cluster"]["rounds"])
            state["control_limit"] += 4 * spec["cluster"]["rounds"] + 12
            state["no_progress"] = 0
            state["progress_epoch"] = len(state["rounds"])
        mode = "control" if (root / "plan.json").exists() else "bootstrap"
        label = ("C" if mode == "control" else "B") + str(len(state["jobs"]))
        state.update(status="submitted", error=None); journal.save()
        job = journal.submit(label, batch_args(spec, mode, root / "launch.json"))
        print("Run directory: " + str(root) + "\nSubmitted job: " + job + "\nResume: bash run.sh slurm --profile bmrc --account " + spec["cluster"]["account"] + " --resume " + str(root))
        return job


def prepare(spec):
    import yaml
    from aleatoric_nk_grid import nk_grid as nk
    from aleatoric_nk_grid.config import config_to_json
    from aleatoric_nk_grid.execution_contract import runtime_environment
    from aleatoric_nk_grid.run_panels import resolved_panels
    from aleatoric_nk_grid.suite_queue import SuiteDesign, file_metadata, check_files, R2_COLUMNS
    root = Path(spec["output"]); ready = root / "prepared.json"
    if ready.exists():
        prepared = read(ready); check_files(prepared["files"])
        return prepared
    area = root / "prepared" / uuid.uuid4().hex; area.mkdir(parents=True)
    document = yaml.safe_load((common.ROOT / "FFCWS/adapter/config/ffc.yaml").read_text())
    document["outcomes"] = [o for o in OUTCOMES if any(p.endswith("_" + o) for p in spec["panels"])]
    document["strategies"] = [s for s in STRATEGIES if any(p.startswith("ffc_" + s + "_") for p in spec["panels"])]
    for k, name in (("background", "background.dta"), ("train", "train.csv"), ("test", "test.csv")):
        document["paths"][k] = str(Path(spec["data"]) / name)
    for k, name in (("output_root", "work"), ("ard_root", "ard"), ("schema_root", "schema")):
        document["paths"][k] = str(area / name)
    original = yaml.safe_load((common.ROOT / "FFCWS/panels.yaml").read_text())
    selected = [dict(p) for p in original["panels"] if p["name"] in spec["panels"]]
    if {p["name"] for p in selected} != set(spec["panels"]) or len(selected) != len(spec["panels"]):
        raise ValueError("Requested panel set differs from the declared non-GPA manifest")
    models = list(dict.fromkeys(m for p in selected for m in p["models"]))
    config = area / "adapter.yaml"; config.write_text(yaml.safe_dump(document, sort_keys=False))
    common.command([sys.executable, common.ROOT / "FFCWS/adapter/adapter.py", "--config", config,
                    "--validation-model", *models, "--min-n", "10", "--seed", "12345"])
    for p in selected:
        p["schema"] = str(area / "schema" / (p["name"] + ".json"))
        p["out"] = str(root / "panels" / p["name"] / "final.csv")
    original["panels"] = selected; original["model_params"] = str(common.ROOT / "FFCWS/model_params.yaml")
    manifest = area / "panels.yaml"; manifest.write_text(yaml.safe_dump(original, sort_keys=False))
    panels = []
    for name, config in resolved_panels(manifest, preset=spec["preset"]):
        config = replace(config, n_jobs=1, checkpoint_retention=spec["checkpoint_retention"], allow_large_run=spec["allow_large_run"])
        with nk.NKGridExecutionSession.open_from_config(config) as session:
            cfg = replace(config, n_grid=tuple(map(int, session.n_grid)), k_grid=tuple(map(int, session.k_grid)),
                          repeat_plan=tuple(session.repeat_pairs))
            paths = [config.schema, config.model_params, session.schema.table, session.schema.test_table,
                     session.schema.feature_manifest]
            schema_doc = read(config.schema)
            definition = schema_doc.get("feature_universe", {}).get("definition_file")
            if definition:
                paths.append(config.schema.parent / definition)
            files = [file_metadata(p) for p in paths if p is not None]
            panels.append(dict(name=name, dataset=session.dataset, task=session.task, config=config_to_json(cfg),
                N=list(map(int, session.n_grid)), K=list(map(int, session.k_grid)), repeats=list(map(list, session.repeat_pairs)),
                models=list(config.models), model_params=nk.resolved_model_params(session.selected_model_params),
                algorithm_version=session.algorithm_version, files=files,
                environment_overrides=nk.model_run_settings(config.models),
                public_columns=list(nk.public_result_columns(session.task))))
    design = SuiteDesign(panels)
    prepared = dict(panels=panels, count=design.count, runtime_environment=runtime_environment(),
                    files=list({f["path"]: f for p in panels for f in p["files"]}.values()))
    check_files(spec["files"])
    common.atomic_json(ready, prepared)
    return prepared


def bootstrap(request):
    from discoverer_continuation import Journal, Slurm, _lock
    from aleatoric_nk_grid.suite_queue import check_files, file_metadata
    spec = read(request); root = Path(spec["output"]); validate(spec)
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Bootstrap requires a compute allocation")
    with _lock(root / ".bootstrap-work.lock"):
        if not (root / "plan.json").exists():
            b = spec["bootstrap"]; venv = Path(b["venv"])
            scratch = root / "tmp" / ("bootstrap-" + os.environ["SLURM_JOB_ID"])
            scratch.mkdir(parents=True, exist_ok=True)
            os.environ["TMPDIR"] = str(scratch)
            os.environ["PIP_CACHE_DIR"] = str(root / "pip-cache")
            refresh = b["refresh_env"] or (b["owned_environment"] and (venv / "bin/python").exists() and not (venv / ".nkgrid-launch-environment.json").exists())
            if b["owned_environment"] and venv.exists() and not (venv / "bin/python").exists():
                # Repair a killed venv creation in this run's private directory.
                common.command([sys.executable, "-m", "venv", venv]); refresh = True
            python, _ = common.ensure_environment(argparse.Namespace(venv=str(venv), refresh_env=refresh))
            common.command([python, Path(__file__).resolve(), "prepare", request])
            prepared = read(root / "prepared.json"); check_files(prepared["files"])
            from suite_resources import resolve
            from discoverer_resources import duration
            allocation = resolve(spec)
            spec["cluster"].update({k: allocation[k] for k in ("partition", "constraint", "qos")})
            common.atomic_json(root / "resources.json", allocation)
            plan = dict(format=FORMAT, plan_id=uuid.uuid4().hex, launch=spec, allocation=allocation,
                        wall_seconds=duration(spec["cluster"]["time_limit"]), **prepared)
            publish_plan(root, plan)
        else:
            spec = checked_plan(root)["launch"]
        with _lock(root / ".suite-state.lock"):
            path, state = state_for(spec)
            journal = Journal(path, state, Slurm(spec["cluster"]["account"], spec["cluster"].get("qos")))
            job = journal.submit("Cafter-" + os.environ["SLURM_JOB_ID"], batch_args(spec, "control", request, dependency=os.environ["SLURM_JOB_ID"]))
            state.update(status="prepared", total=read(root / "plan.json")["count"]); journal.save()
            print("Suite prepared; controller " + job)


def control(request, *, slurm=None, backend=None, admit=None):
    from discoverer_continuation import Journal, Slurm, TERMINAL, _lock
    from suite_resources import admission, CapacityWait
    from aleatoric_nk_grid import suite_queue
    backend = backend or suite_queue; admit = admit or admission
    original = read(request); root = Path(original["output"]); plan = checked_plan(root)
    spec = plan["launch"]; validate(spec); backend.check_files(plan["files"])
    current = os.environ.get("SLURM_JOB_ID")
    if not current:
        raise ValueError("Control requires a Slurm job")
    with _lock(root / ".suite-state.lock"):
        path, state = state_for(spec)
        journal = Journal(path, state, slurm or Slurm(spec["cluster"]["account"], spec["cluster"].get("qos")))
        if current not in [journal.submit(k, []) for k in list(state["jobs"]) if k.startswith(("C", "G"))]:
            raise ValueError("Controller is not recorded in submission journal")
        if state["status"] in ("complete", "round_budget_exhausted", "no_progress", "control_budget_exhausted"):
            return state
        active = []
        for label in list(state["jobs"]):
            if label.startswith("W"):
                job = journal.submit(label, [])
                if any(s not in TERMINAL for s in journal.slurm.states(job)):
                    active.append(job)
        if active:
            if len(active) != 1:
                raise ValueError("Overlapping worker allocations")
            journal.submit("Gwait-" + active[0], batch_args(spec, "control", request, dependency=active[0]))
            return state
        if sum(k.startswith(("C", "G")) for k in state["jobs"]) >= state["control_limit"]:
            state["status"] = "control_budget_exhausted"; journal.save(); return state
        # Recover a controller dying before or after submitting workers.
        journal.submit("Gafter-" + current, batch_args(spec, "control", request, dependency=current, delayed=True))
        rounds = [r["root"] for r in state["rounds"] if r["label"] in state["jobs"]]
        if (root / "verified.json").exists():
            backend.cleanup(plan); state["status"] = "complete"; journal.save(); return state
        index = len(rounds); directory = root / "rounds" / ("round-" + str(index))
        report = backend.prepare_round(plan, rounds, directory)
        state["completed"] = report["done"]; state["panels"] = report["panels"]
        if not report["remaining"]:
            backend.finalize(plan, rounds); state["status"] = "complete"; journal.save(); return state
        if index >= state["round_limit"]:
            state["status"] = "round_budget_exhausted"; journal.save(); return state
        stalled = 0
        for old in reversed(state["rounds"][state.get("progress_epoch", 0):index]):
            if old["done_before"] != report["done"]:
                break
            stalled += 1
        if stalled >= 2:
            state["status"] = "no_progress"; journal.save(); return state
        try:
            live = admit(spec, plan["allocation"])
        except CapacityWait as exc:
            state.update(status="waiting_resources", error=str(exc)); journal.save(); return state
        item = dict(root=str(directory), label="W" + str(index), done_before=report["done"], admission=live)
        if len(state["rounds"]) == index:
            state["rounds"].append(item)
        else:
            state["rounds"][index] = item
        state.update(status="running", error=None); journal.save()
        job = journal.submit(item["label"], batch_args(spec, "work", directory, allocation=plan["allocation"], dependency=current))
        journal.submit("Gwait-" + job, batch_args(spec, "control", request, dependency=job))
        return state


def status(root):
    root = common.path_from_repo(root)
    spec = read(root / "launch.json")
    if spec.get("format") != FORMAT:
        raise ValueError("status --run expects a suite run")
    path, state = state_for(spec)
    value = dict(run=str(root), **state)
    if (root / "resources.json").exists():
        value["resources"] = read(root / "resources.json")
    if state["rounds"]:
        progress = Path(state["rounds"][-1]["root"]) / "control/progress.json"
        if progress.exists():
            value["current_round"] = read(progress)
    if (root / "verified.json").exists():
        value["publication"] = read(root / "verified.json")
    if (root / "last-error.json").exists():
        value["last_error"] = read(root / "last-error.json")
    print(json.dumps(value, indent=2))


def entry(args, argv):
    if args.target == "status":
        if not args.run:
            raise ValueError("status requires --run RUN_DIR")
        status(args.run); return
    if args.resume:
        root = common.path_from_repo(args.resume)
        spec = read(root / "launch.json")
        if spec.get("format") != FORMAT:
            raise ValueError("Directory resume requires a suite run")
        if args.account != spec["cluster"]["account"] or args.profile != spec["profile"]:
            raise ValueError("Resume requires the original explicit account/profile")
        if any(explicit(argv, k) for k in ("suite", "preset", "output", "panel", "resources", "models", "schema", "ffc-data-dir", "workers", "rounds", "partition", "time", "memory", "plan-memory", "plan-time", "venv", "refresh-env")):
            raise ValueError("Resume reuses saved inputs, resources and environment; omit new-run options")
        if args.checkpoints and args.checkpoints != spec["checkpoint_retention"]:
            raise ValueError("Resume cannot change checkpoint policy")
        if (root / "plan.json").exists():
            spec = checked_plan(root)["launch"]
        for key in ("qos", "constraint"):
            if explicit(argv, key) and getattr(args, key) != spec["cluster"].get(key):
                raise ValueError("Resume cannot change " + key)
        if args.dry_run:
            print(json.dumps(dict(launch=spec, actions=["resume full cell set minus existing valid result cells"]), indent=2)); return
        start(spec, resume=True); return
    spec = spec_for(args, argv)
    if args.dry_run:
        print(json.dumps(dict(launch=spec, actions=["compute-node environment and data preparation", "one shared fixed allocation", "cell-difference continuation", "publish per-panel CSV and paper R2"],
                             note="Read-only; node geometry resolves on BMRC or is reused from --resources"), indent=2)); return
    if os.name == "nt":
        raise ValueError("Submit from a BMRC Linux login shell")
    from aleatoric_nk_grid.suite_queue import file_metadata
    source = common.frozen_source()
    if source["dirty"]:
        raise ValueError("Suite submission requires a clean committed checkout")
    root = Path(spec["output"])
    if root.exists():
        raise FileExistsError("Run directory exists; use --resume " + str(root))
    for target in (root / "launch.json", Path(spec["resources"])):
        if target.is_relative_to(common.ROOT) and common.subprocess.run(["git", "check-ignore", "-q", str(target)], cwd=common.ROOT).returncode:
            raise ValueError("Run/resources inside checkout must be Git-ignored")
    spec["source"] = source; spec["request_id"] = uuid.uuid4().hex
    spec["files"] = [file_metadata(Path(spec["data"]) / f) for f in ("background.dta", "train.csv", "test.csv")]
    root.mkdir(parents=True); (root / "logs").mkdir()
    common.atomic_json(root / "launch.json", spec)
    start(spec)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("bootstrap", "prepare", "control", "work")); p.add_argument("path", type=Path)
    args = p.parse_args()
    try:
        if args.mode == "prepare":
            spec = read(args.path); validate(spec); prepare(spec)
        elif args.mode == "work":
            from aleatoric_nk_grid.suite_runtime import run
            manifest = read(args.path / "manifest.json"); plan = checked_plan(Path(manifest["plan"]).parent)
            validate(plan["launch"]); run(manifest["plan"], args.path)
        else:
            {"bootstrap": bootstrap, "control": control}[args.mode](args.path)
    except Exception as exc:
        # A small advisory receipt; never turn errors into successful completion.
        error_root = args.path.parent if args.path.is_file() else args.path.parents[1]
        common.atomic_json(error_root / "last-error.json", dict(stage=args.mode, error=str(exc),
                           resume="bash run.sh slurm --profile bmrc --account YOUR_ACCOUNT --resume " + str(error_root)))
        raise


if __name__ == "__main__":
    main()
