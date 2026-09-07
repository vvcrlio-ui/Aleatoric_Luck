"""Launcher adversarial boundary tests; no numerical engine or Slurm needed."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "experiment.py"
spec = importlib.util.spec_from_file_location("nkgrid_launch", SOURCE)
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


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
            launch.main(["slurm", "--account", "a", "--constraint", "none", "--output", "runs/new"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0][0], "sbatch")
        self.assertTrue((self.root / "runs/new/logs").is_dir())
        self.assertEqual(json.loads((self.root / "runs/new/submission.json").read_text())["planning_job"], "12345;cluster")
        self.assertFalse((self.root / "runs/new/tasks.parquet").exists())

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
