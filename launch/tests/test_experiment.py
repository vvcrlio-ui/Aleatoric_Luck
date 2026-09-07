"""Launcher adversarial boundary tests; no numerical engine or Slurm needed."""
import contextlib
from dataclasses import dataclass
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

SOURCE = Path(__file__).resolve().parents[1] / "experiment.py"
spec = importlib.util.spec_from_file_location("nkgrid_launch", SOURCE)
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


@dataclass
class StubConfig:
    models: tuple = ("ridge",)
    out: Path = Path("final.csv")
    schema: Path = Path("schema.json")
    n_jobs: int = 1
    allow_large_run: bool = False
    checkpoint_retention: str = "default"
    outcome: str = "y"
    min_n: int = 2
    test_size: float = 0.2
    seed: int = 1
    n_grid: tuple = (8,)
    k_grid: tuple = (2,)
    n_seeds: int = 1
    n_draws: int = 1
    repeat_plan: object = None


class SizeGrid(list):
    def tolist(self):
        return list(self)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "FFCWS").mkdir()
        (self.root / "FFCWS/panels.yaml").write_text("panels: []\n")
        self.root_patch = patch.object(launch, "ROOT", self.root)
        self.root_patch.start()

    def tearDown(self):
        self.root_patch.stop()
        self.temp.cleanup()

    def args(self, *args):
        return launch.parser().parse_args(list(args))

    def test_preview_does_not_install_read_data_create_output_or_submit(self):
        with patch.object(launch, "command", side_effect=AssertionError("subprocess during preview")), \
             patch.object(launch, "ensure_environment", side_effect=AssertionError("install during preview")), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            launch.main(["slurm", "--account", "mills.prj", "--constraint", "skl-compat", "--preset", "production", "--dry-run"])
        self.assertFalse((self.root / "runs").exists())
        self.assertEqual(json.loads(output.getvalue())["launch"]["cluster"]["workers"], 600)

    def test_production_requires_explicit_large_run(self):
        with self.assertRaisesRegex(ValueError, "allow-large-run"):
            launch.launch_spec(self.args("local", "--preset", "production"))

    def test_local_default_is_small_unique_run(self):
        a = launch.launch_spec(self.args("local"))
        b = launch.launch_spec(self.args("local"))
        self.assertEqual(a["preset"], "dev")
        self.assertNotEqual(a["output"], b["output"])
        self.assertEqual(a["checkpoint_retention"], "default")

    def test_checkpoint_options_recorded_and_invalid_values_rejected(self):
        for target in ("local", "slurm"):
            for retention in ("keep", "delete"):
                with self.subTest(target=target, retention=retention):
                    extra = ["--account", "a", "--constraint", "none"] if target == "slurm" else []
                    run = launch.launch_spec(self.args(target, "--checkpoints", retention, *extra))
                    self.assertEqual(run["checkpoint_retention"], retention)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.args("local", "--checkpoints", "always")

    def test_checkpoint_policy_reaches_resolved_config_and_preserves_older_requests(self):
        for supplied, configured, expected in (("keep", "delete", "keep"),
                                                ("delete", "keep", "delete"),
                                                ("default", "keep", "keep"),
                                                (None, "delete", "delete")):
            with self.subTest(supplied=supplied, configured=configured):
                run = launch.launch_spec(self.args("local"))
                if supplied is None:
                    del run["checkpoint_retention"]
                else:
                    run["checkpoint_retention"] = supplied
                config = StubConfig(checkpoint_retention=configured)
                modules = {
                    "aleatoric_nk_grid.run_panels": SimpleNamespace(resolved_panels=lambda *a, **kw: [("panel", config)]),
                    "aleatoric_nk_grid.ingest": SimpleNamespace(load_input=lambda *a: "loaded"),
                    "aleatoric_nk_grid.validate_input": SimpleNamespace(validate_input=lambda *a, **kw: ("validated", "groups")),
                    "aleatoric_nk_grid.nk_grid": SimpleNamespace(resolve_input_grids=lambda *a: (SizeGrid([8]), SizeGrid([2])), LARGE_RUN_THRESHOLD=100),
                }
                with patch.dict(launch.sys.modules, modules), contextlib.redirect_stdout(io.StringIO()):
                    resolved = launch.resolve_experiment(run)
                self.assertEqual(resolved.checkpoint_retention, expected)

    def test_checkpoint_policy_reaches_local_engine_and_slurm_plan_builder(self):
        for target in ("local", "slurm"):
            for retention in ("keep", "delete"):
                with self.subTest(target=target, retention=retention):
                    config = StubConfig(checkpoint_retention=retention)
                    runner, planner = Mock(), Mock(return_value={"format_version": 2})
                    modules = {
                        "aleatoric_nk_grid.nk_grid": SimpleNamespace(run_nk_grid=runner),
                        "aleatoric_nk_grid.chunk_planning": SimpleNamespace(ClusterPolicy=lambda **kw: kw, build_dynamic_plan=planner),
                    }
                    extra = ["--account", "a", "--constraint", "none"] if target == "slurm" else []
                    run = launch.launch_spec(self.args(target, "--checkpoints", retention, *extra))
                    with patch.dict(launch.sys.modules, modules), \
                         patch.object(launch, "validate_source"), \
                         patch.object(launch, "resolve_experiment", return_value=config), \
                         patch.object(launch, "atomic_json"), patch.object(launch, "command"):
                        launch.execute(run)
                    called = runner if target == "local" else planner
                    self.assertEqual(called.call_count, 1)
                    self.assertIs(called.call_args.args[0], config)
                    self.assertEqual(called.call_args.args[0].checkpoint_retention, retention)

    def test_execute_cannot_silently_override_frozen_checkpoint_policy(self):
        with patch.object(launch, "execute", side_effect=AssertionError("execution")):
            with self.assertRaisesRegex(ValueError, "frozen launch request"):
                launch.main(["execute", "--request", "launch.json", "--checkpoints", "keep"])

    def test_scheduler_identity_required(self):
        with patch.dict(launch.os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "Slurm requires"):
                launch.launch_spec(self.args("slurm"))

    def test_inherited_account_cannot_replace_explicit_input(self):
        with patch.dict(launch.os.environ, {"NKGRID_ACCOUNT": "inherited.prj", "SBATCH_ACCOUNT": "inherited.prj"}):
            for extra in ([], ["--dry-run"], ["--resume", "plan.json"]):
                with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, "explicit --account"):
                    launch.launch_spec(self.args("slurm", "--profile", "bmrc", *extra))

    def test_explicit_account_reaches_submission_despite_environment(self):
        with patch.dict(launch.os.environ, {"NKGRID_ACCOUNT": "inherited.prj", "SBATCH_ACCOUNT": "inherited.prj"}):
            run = launch.launch_spec(self.args("slurm", "--account", "chosen.prj", "--constraint", "none"))
        self.assertEqual(run["cluster"]["account"], "chosen.prj")
        self.assertIn("--account=chosen.prj", launch.slurm_command(run, self.root / "request.json"))

    def test_missing_account_refused_before_install_or_submission(self):
        with patch.object(launch, "ensure_environment", side_effect=AssertionError("install")), \
             patch.object(launch, "command", side_effect=AssertionError("submit")):
            with self.assertRaisesRegex(ValueError, "explicit --account"):
                launch.main(["slurm", "--profile", "bmrc"])

    def test_blank_account_is_not_explicit_input(self):
        for value in ("", "   "):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "explicit --account"):
                launch.launch_spec(self.args("slurm", "--account", value, "--constraint", "none"))

    def test_local_job_bound_not_silently_ignored_on_slurm(self):
        with self.assertRaisesRegex(ValueError, "local-only"):
            launch.launch_spec(self.args("slurm", "--max-jobs", "1"))

    def test_resume_rejects_design_and_resource_overrides(self):
        for option in (["--workers", "600"], ["--schema", "new.json"], ["--models", "ridge"],
                       ["--preset", "production"], ["--memory", "64G"], ["--output", "different"]):
            with self.subTest(option=option), self.assertRaisesRegex(ValueError, "resume reuses"):
                launch.launch_spec(self.args("slurm", "--resume", "plan.json", *option))

    def test_panel_cannot_escape_directory(self):
        for value in ["../other", "/absolute", "x\ny", "a;b"]:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "panel name"):
                launch.launch_spec(self.args("local", "--panel", value))

    def test_sbatch_arguments_do_not_interpolate_shell_payloads(self):
        payload = "account; $(touch SHOULD_NOT_EXIST)"
        run = launch.launch_spec(self.args("slurm", "--account", payload, "--constraint", "none"))
        cmd = launch.slurm_command(run, self.root / "path with spaces/request.json")
        self.assertIn("--account=" + payload, cmd)
        self.assertNotIn("--constraint=none", cmd)
        self.assertEqual(cmd[-1], str(self.root / "path with spaces/request.json"))
        self.assertNotIn("--wrap", cmd)

    def test_scheduler_newlines_rejected(self):
        with self.assertRaisesRegex(ValueError, "single-line"):
            launch.launch_spec(self.args("slurm", "--account", "x\ny", "--constraint", "none"))

    def test_output_collision_rejected_before_install(self):
        destination = self.root / "old-run"
        destination.mkdir()
        with patch.object(launch.sys, "platform", "linux"), \
             patch.object(launch, "frozen_source", return_value={"commit": "abc", "dirty": False}), \
             patch.object(launch, "ensure_environment", side_effect=AssertionError("installed before checking collision")):
            with self.assertRaises(FileExistsError):
                launch.main(["local", "--output", str(destination)])

    def test_slurm_dirty_checkout_refused_before_install(self):
        with patch.object(launch.sys, "platform", "linux"), \
             patch.object(launch, "frozen_source", return_value={"commit": "abc", "dirty": True}), \
             patch.object(launch, "ensure_environment", side_effect=AssertionError("install")):
            with self.assertRaisesRegex(ValueError, "clean committed"):
                launch.main(["slurm", "--account", "a", "--constraint", "none"])

    def test_manifest_content_change_is_detected(self):
        path = self.root / "FFCWS/panels.yaml"
        run = {"manifest": str(path), "manifest_sha256": launch.sha256(path), "schema": None,
               "source": {"commit": "abc"}, "target": "slurm"}
        path.write_text("panels: {}\n")
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            launch.validate_source(run)

    def test_checkout_change_while_queued_is_detected(self):
        path = self.root / "FFCWS/panels.yaml"
        run = {"manifest": str(path), "manifest_sha256": launch.sha256(path), "schema": None,
               "source": {"commit": "abc"}, "target": "slurm"}
        with patch.object(launch, "frozen_source", return_value={"commit": "def", "dirty": False}):
            with self.assertRaisesRegex(ValueError, "checkout changed"):
                launch.validate_source(run)

    def test_initial_slurm_submission_has_no_engine_or_plan_work(self):
        calls = []
        def record(args, **kwargs):
            calls.append(([str(a) for a in args], kwargs))
            return subprocess.CompletedProcess(args, 0, "12345;cluster\n", "")
        with patch.object(launch.sys, "platform", "linux"), \
             patch.object(launch, "frozen_source", return_value={"commit": "abc", "dirty": False}), \
             patch.object(launch, "ensure_environment", return_value=(self.root / "venv/bin/python", self.root / "venv")), \
             patch.object(launch.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)), \
             patch.object(launch, "resolve_experiment", side_effect=AssertionError("heavy plan on login node")), \
             patch.object(launch, "command", side_effect=record), contextlib.redirect_stdout(io.StringIO()):
            launch.main(["slurm", "--account", "a", "--constraint", "none", "--output", "runs/new", "--checkpoints", "delete"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0][0], "sbatch")
        self.assertTrue((self.root / "runs/new/logs").is_dir())
        self.assertEqual(json.loads((self.root / "runs/new/submission.json").read_text())["planning_job"], "12345;cluster")
        self.assertFalse((self.root / "runs/new/tasks.parquet").exists())
        self.assertEqual(json.loads((self.root / "runs/new/launch.json").read_text())["checkpoint_retention"], "delete")

    def test_local_launch_json_persists_explicit_checkpoint_policy(self):
        with patch.object(launch.sys, "platform", "linux"), \
             patch.object(launch, "frozen_source", return_value={"commit": "abc", "dirty": True}), \
             patch.object(launch, "ensure_environment", return_value=(self.root / "venv/bin/python", self.root / "venv")), \
             patch.object(launch, "command") as run, contextlib.redirect_stdout(io.StringIO()):
            launch.main(["local", "--output", "runs/local", "--checkpoints", "keep"])
        request = self.root / "runs/local/launch.json"
        self.assertEqual(json.loads(request.read_text())["checkpoint_retention"], "keep")
        self.assertEqual(run.call_args.args[0][-1], request)

    def checkpoint_resume_fixture(self, retention="default"):
        output_dir = self.root / "out"
        output_dir.mkdir(exist_ok=True)
        config = {} if retention is None else {"checkpoint_retention": retention}
        snapshot = self.root / "snapshot.json"
        snapshot.write_text(json.dumps({"config": config, "output_dir": "out"}))
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"snapshot": "snapshot.json", "submission": {"account": "original"}}))
        return plan, output_dir

    def test_resume_matching_policy_reuses_plan_without_rewriting_it(self):
        for frozen, requested in (("keep", "keep"), ("delete", "delete"), ("default", "keep"), (None, "keep")):
            with self.subTest(frozen=frozen, requested=requested):
                plan, _ = self.checkpoint_resume_fixture(frozen)
                original = plan.read_bytes()
                with patch.object(launch.sys, "platform", "linux"), \
                     patch.object(launch, "frozen_source", return_value={"commit": "abc", "dirty": False}), \
                     patch.object(launch, "ensure_environment", return_value=(self.root / "venv/bin/python", self.root / "venv")), \
                     patch.object(launch, "command") as run:
                    launch.main(["slurm", "--resume", str(plan), "--account", "original", "--checkpoints", requested])
                self.assertEqual(run.call_count, 1)
                self.assertEqual(plan.read_bytes(), original)

    def test_resume_checkpoint_override_refused_before_install_or_submission(self):
        for frozen, requested in (("keep", "delete"), ("delete", "keep"), ("default", "delete"), (None, "delete")):
            with self.subTest(frozen=frozen, requested=requested):
                plan, _ = self.checkpoint_resume_fixture(frozen)
                with patch.object(launch, "ensure_environment", side_effect=AssertionError("install")), \
                     patch.object(launch, "command", side_effect=AssertionError("submission")):
                    with self.assertRaisesRegex(ValueError, "frozen checkpoint policy"):
                        launch.main(["slurm", "--resume", str(plan), "--account", "original", "--checkpoints", requested])

    def test_resume_without_flag_preserves_frozen_delete_policy(self):
        plan, _ = self.checkpoint_resume_fixture("delete")
        with contextlib.redirect_stdout(io.StringIO()) as output:
            launch.main(["slurm", "--resume", str(plan), "--account", "original", "--dry-run"])
        self.assertEqual(json.loads(output.getvalue())["launch"]["checkpoint_retention"], "delete")

    def test_archived_run_refuses_resume_before_install_even_without_flag(self):
        plan, output_dir = self.checkpoint_resume_fixture("delete")
        (output_dir / "checkpoint-archive.json").write_text("{}")
        for options in ([], ["--checkpoints", "delete"], ["--dry-run"]):
            with self.subTest(options=options), \
                 patch.object(launch, "ensure_environment", side_effect=AssertionError("install")), \
                 patch.object(launch, "command", side_effect=AssertionError("submission")):
                with self.assertRaisesRegex(ValueError, "final CSV is retained"):
                    launch.main(["slurm", "--resume", str(plan), "--account", "original", *options])

    def test_explicit_resume_policy_cannot_be_silently_ignored_without_snapshot(self):
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"submission": {"account": "original"}}))
        with self.assertRaisesRegex(ValueError, "plan has no snapshot"):
            launch.main(["slurm", "--resume", str(plan), "--account", "original", "--checkpoints", "keep", "--dry-run"])

    def test_resume_calls_existing_submitter_without_new_request(self):
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"submission": {"account": "original"}}))
        with patch.object(launch.sys, "platform", "linux"), \
             patch.object(launch, "frozen_source", return_value={"commit": "abc", "dirty": False}), \
             patch.object(launch, "ensure_environment", return_value=(self.root / "venv/bin/python", self.root / "venv")), \
             patch.object(launch, "command") as run:
            launch.main(["slurm", "--resume", str(plan), "--account", "original"])
        self.assertEqual(run.call_count, 1)
        self.assertIn("--submit", run.call_args.args[0])
        self.assertEqual(run.call_args.args[0][-1], plan)
        self.assertFalse((self.root / "runs").exists())

    def environment_fixture(self, version):
        engine = self.root / "NK_Grid"
        engine.mkdir()
        (engine / "requirements.txt").write_text("numpy==2.4.4\n")
        (engine / "pyproject.toml").write_text("[project]\n")
        environment = self.root / "venv"
        (environment / "bin").mkdir(parents=True)
        (environment / "bin/python").touch()
        calls = []
        def record(args, **kwargs):
            calls.append([str(a) for a in args])
            return subprocess.CompletedProcess(args, 0, json.dumps({"numpy": version}), "")
        return environment, calls, record

    def test_existing_valid_environment_not_reinstalled(self):
        environment, calls, record = self.environment_fixture("2.4.4")
        with patch.object(launch, "environment_lock", return_value=contextlib.nullcontext()), \
             patch.object(launch, "command", side_effect=record):
            launch.ensure_environment(self.args("local", "--venv", str(environment)))
        self.assertFalse(any("install" in call for call in calls))
        self.assertTrue((environment / ".nkgrid-launch-environment.json").is_file())

    def test_existing_wrong_dependencies_not_silently_replaced(self):
        environment, calls, record = self.environment_fixture("0.0.1")
        with patch.object(launch, "environment_lock", return_value=contextlib.nullcontext()), \
             patch.object(launch, "command", side_effect=record):
            with self.assertRaisesRegex(ValueError, "Existing environment differs"):
                launch.ensure_environment(self.args("local", "--venv", str(environment)))
        self.assertFalse(any("install" in call for call in calls))
        self.assertFalse((environment / ".nkgrid-launch-environment.json").exists())

    def test_resume_account_override_refused_before_install(self):
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"submission": {"account": "original"}}))
        with patch.object(launch.sys, "platform", "linux"), \
             patch.object(launch, "frozen_source", return_value={"commit": "abc", "dirty": False}), \
             patch.object(launch, "ensure_environment", side_effect=AssertionError("install")):
            with self.assertRaisesRegex(ValueError, "frozen account"):
                launch.main(["slurm", "--resume", str(plan), "--account", "other"])


if __name__ == "__main__":
    unittest.main()
