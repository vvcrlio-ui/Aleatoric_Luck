"""Real temporary-file journal checks with a fake scheduler; no Slurm required."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[2] / "NK_Grid/slurm/submission_journal.py"
spec = importlib.util.spec_from_file_location("submission_journal", SOURCE)
journal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(journal)


class SubmissionJournalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = self.root / "out"
        self.output.mkdir()
        self.snapshot = self.root / "snapshot.json"
        self.snapshot.write_text(json.dumps({"output_dir": str(self.output)}))
        self.plan = self.root / "plan.json"
        self.payload = {"snapshot": str(self.snapshot), "execution_plan_id": "execution-one", "rounds": 1,
                        "submission": {"account": "explicit-project", "constraint": "none"}}
        self.payload.update({name: {"sbatch_args": ["--cpus-per-task=1"]}
                             for name in ("preparation", "verification", "finalization")})
        self.plan.write_text(json.dumps(self.payload))
        self.scheduler_calls = []

    def tearDown(self):
        self.temp.cleanup()

    def receipt(self):
        return next(self.root.glob("plan.submission-receipt-*.json"))

    def fake_chain(self, responses):
        active_receipt = [None]
        def run(command, **kwargs):
            if command[0] == "sbatch":
                self.scheduler_calls.append(command)
                pending = journal._read(active_receipt[0])["pending_submission"]
                self.assertIsNotNone(pending, "intent must be durable before contacting scheduler")
                code, output = responses[min(len(self.scheduler_calls) - 1, len(responses) - 1)]
                return subprocess.CompletedProcess(command, code, output, "simulated error" if code else "")
            path = kwargs["env"]["NKGRID_SUBMISSION_RECEIPT"]
            active_receipt[0] = path
            try:
                for label in ("prep-1", "work-1", "close-1", "verify", "finalize"):
                    journal.submit(path, self.plan, label, "generation-one", ["--dependency=afterany:123"])
            except ValueError:
                return subprocess.CompletedProcess(command, 2)
            return subprocess.CompletedProcess(command, 0)
        return run

    def run_chain(self, responses):
        with patch.object(journal.subprocess, "run", side_effect=self.fake_chain(responses)), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return journal.guard(self.plan, ["fake-chain"])

    def test_success_records_every_accepted_job_and_allows_existing_resume(self):
        self.assertEqual(self.run_chain([(0, "123\n")]), 0)
        receipt = journal._read(self.receipt())
        self.assertEqual(receipt["status"], "complete")
        self.assertEqual(len(receipt["jobs"]), 5)
        self.assertIsNone(receipt["pending_submission"])
        self.assertTrue(all(job["generation"] == "generation-one" for job in receipt["jobs"]))
        self.assertTrue(all(job["dependency"] == "afterany:123" for job in receipt["jobs"]))
        # A completed submission is compatible with the existing explicit
        # recovery workflow; it does not assert that its jobs have completed.
        self.assertEqual(self.run_chain([(0, "456\n")]), 0)

    def test_second_failure_keeps_first_job_and_blocks_blind_resubmission(self):
        self.assertNotEqual(self.run_chain([(0, "123\n"), (73, "")]), 0)
        receipt = journal._read(self.receipt())
        self.assertEqual(receipt["status"], "unknown")
        self.assertEqual([(job["label"], job["slurm_job_id"]) for job in receipt["jobs"]], [("prep-1", "123")])
        self.assertEqual(receipt["pending_submission"]["label"], "work-1")
        with patch.object(journal.subprocess, "run", side_effect=AssertionError("duplicate submission")):
            with self.assertRaisesRegex(ValueError, "prep-1=123"):
                journal.guard(self.plan, ["fake-chain"])

    def test_empty_malformed_and_ambiguous_job_ids_are_unknown(self):
        for raw in ("", "0\n", "not-a-job\n", "123\n456\n", "123;\n"):
            with self.subTest(raw=raw):
                # Independent identity avoids intentionally retained unknowns.
                self.payload["execution_plan_id"] = "case-" + hashlib.sha256(raw.encode()).hexdigest()
                self.plan.write_text(json.dumps(self.payload))
                self.scheduler_calls.clear()
                self.assertNotEqual(self.run_chain([(0, raw)]), 0)
                latest = max(self.root.glob("plan.submission-receipt-*.json"), key=lambda path: path.stat().st_mtime_ns)
                receipt = journal._read(latest)
                self.assertEqual(receipt["status"], "unknown")
                self.assertEqual(receipt["jobs"], [])
                self.assertEqual(receipt["pending_submission"]["stdout"], raw)

    def test_cluster_suffix_is_audited_but_never_passed_as_a_protocol_token(self):
        with patch.object(journal.subprocess, "run", side_effect=self.fake_chain([(0, "123;bmrc\n")])), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(journal.guard(self.plan, ["fake-chain"]), 0)
        self.assertEqual(output.getvalue().splitlines(), ["123"] * 5)
        job = journal._read(self.receipt())["jobs"][0]
        self.assertEqual((job["slurm_job_id"], job["sbatch_response"], job["cluster"]), ("123", "123;bmrc", "bmrc"))

    def test_copied_plan_cannot_bypass_unknown_but_new_execution_can(self):
        self.run_chain([(73, "")])
        copied = self.root / "copied"
        copied.mkdir()
        copy_plan = copied / "alias.json"
        copy_plan.write_bytes(self.plan.read_bytes())
        with patch.object(journal.subprocess, "run", side_effect=AssertionError("duplicate submission")):
            with self.assertRaisesRegex(ValueError, "incomplete or unknown"):
                journal.guard(copy_plan, ["fake-chain"])
        self.payload["execution_plan_id"] = "execution-two"
        self.plan.write_text(json.dumps(self.payload))
        self.assertEqual(self.run_chain([(0, "456\n")]), 0)

    def _open_receipt(self):
        with patch.object(journal.subprocess, "run", return_value=subprocess.CompletedProcess([], 2)), \
             contextlib.redirect_stderr(io.StringIO()):
            journal.guard(self.plan, ["unused"])
        path = self.receipt()
        payload = journal._read(path)
        payload["status"] = "submitting"
        journal._write(path, payload)
        return path

    def test_intent_write_failure_never_calls_scheduler(self):
        path = self._open_receipt()
        with patch.object(journal, "_write", side_effect=OSError("disk full")), \
             patch.object(journal.subprocess, "run", side_effect=AssertionError("scheduler contacted before intent")):
            with self.assertRaisesRegex(OSError, "disk full"):
                journal.submit(path, self.plan, "prep-1", "generation", [])

    def test_crash_after_acceptance_preserves_pending_intent(self):
        path = self._open_receipt()
        original_write = journal._write
        writes = []
        def fail_acceptance(target, payload):
            writes.append(payload)
            if len(writes) == 2:
                raise OSError("acceptance write failed")
            original_write(target, payload)
        with patch.object(journal, "_write", side_effect=fail_acceptance), \
             patch.object(journal.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "123\n", "")):
            with self.assertRaisesRegex(OSError, "acceptance write failed"):
                journal.submit(path, self.plan, "prep-1", "generation", [])
        self.assertEqual(journal._read(path)["pending_submission"]["label"], "prep-1")
        self.assertEqual(journal._read(path)["jobs"], [])

    def test_inherited_receipt_cannot_submit_another_or_changed_plan(self):
        path = self._open_receipt()
        self.payload["execution_plan_id"] = "other"
        self.plan.write_text(json.dumps(self.payload))
        with patch.object(journal.subprocess, "run", side_effect=AssertionError("wrong plan submitted")):
            with self.assertRaisesRegex(ValueError, "does not match"):
                journal.submit(path, self.plan, "prep-1", "generation", [])

    def test_real_lock_excludes_an_overlapping_submitter(self):
        path = self.output / "test-submission.lock"
        with journal._lock(path):
            with self.assertRaises(OSError):
                with journal._lock(path):
                    self.fail("overlapping submission lease")

    def test_failure_before_any_sbatch_intent_can_be_retried(self):
        with patch.object(journal.subprocess, "run", return_value=subprocess.CompletedProcess([], 2)), \
             contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(journal.guard(self.plan, ["invalid-plan"]), 2)
        self.assertEqual(journal._read(self.receipt())["status"], "failed_before_submission")
        self.assertEqual(self.run_chain([(0, "123\n")]), 0)

    def test_damaged_unknown_receipt_cannot_claim_it_never_submitted(self):
        self.run_chain([(73, "")])
        path = self.receipt()
        receipt = journal._read(path)
        del receipt["pending_submission"]
        journal._write(path, receipt)
        with patch.object(journal.subprocess, "run", side_effect=AssertionError("unknown receipt bypass")):
            with self.assertRaisesRegex(ValueError, "incomplete or unknown"):
                journal.guard(self.plan, ["fake-chain"])

    def test_empty_submitting_receipt_is_unknown_until_child_exit_is_confirmed(self):
        self._open_receipt()
        with patch.object(journal.subprocess, "run", side_effect=AssertionError("possibly orphaned chain duplicated")):
            with self.assertRaisesRegex(ValueError, "incomplete or unknown"):
                journal.guard(self.plan, ["fake-chain"])

    def test_killed_child_with_empty_receipt_may_have_an_orphaned_submit_process(self):
        with patch.object(journal.subprocess, "run", return_value=subprocess.CompletedProcess([], -9)), \
             contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(journal.guard(self.plan, ["killed-chain"]), -9)
        self.assertEqual(journal._read(self.receipt())["status"], "unknown")
        with patch.object(journal.subprocess, "run", side_effect=AssertionError("orphaned process bypass")):
            with self.assertRaisesRegex(ValueError, "incomplete or unknown"):
                journal.guard(self.plan, ["fake-chain"])

    def test_direct_submitter_rejects_an_archived_run_before_new_receipt(self):
        (self.output / "checkpoint-archive.json").write_text("{}")
        with patch.object(journal.subprocess, "run", side_effect=AssertionError("archived run submitted")):
            with self.assertRaisesRegex(ValueError, "terminally archived"):
                journal.guard(self.plan, ["fake-chain"])
        self.assertEqual(list(self.root.glob("*.submission-receipt-*.json")), [])


if __name__ == "__main__":
    unittest.main()
