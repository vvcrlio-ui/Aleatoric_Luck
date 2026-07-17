import tempfile
import unittest
from pathlib import Path

from NK_GRID.src.slurm_jobs import (
    build_slurm_jobs,
    load_job_snapshot,
    write_job_snapshot,
)


class SlurmJobTests(unittest.TestCase):
    def _write_manifest(self, root: Path, *, duplicate_outputs: bool = False) -> Path:
        second_output = "outputs/reg.csv" if duplicate_outputs else "outputs/clf.csv"
        manifest = root / "panels.yaml"
        manifest.write_text(
            "\n".join(
                [
                    "preset: dev",
                    "panels:",
                    "  - name: regression_panel",
                    "    data: data.csv",
                    "    dataset: synthetic",
                    "    outcome: outcome",
                    "    task: regression",
                    "    models: [ols]",
                    "    out: outputs/reg.csv",
                    "  - name: classification_panel",
                    "    data: data.csv",
                    "    dataset: synthetic",
                    "    outcome: employed",
                    "    task: classification",
                    "    models: [ols]",
                    f"    out: {second_output}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return manifest

    def test_expands_regression_and_classification_together(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            jobs = build_slurm_jobs(self._write_manifest(Path(temp_dir)))
            self.assertEqual(
                [(job.panel, job.config.task) for job in jobs],
                [
                    ("regression_panel", "regression"),
                    ("classification_panel", "classification"),
                ],
            )

    def test_rejects_duplicate_output_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = self._write_manifest(
                Path(temp_dir),
                duplicate_outputs=True,
            )
            with self.assertRaisesRegex(ValueError, "unique output paths"):
                build_slurm_jobs(manifest)

    def test_snapshot_does_not_change_with_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = self._write_manifest(root)
            snapshot = root / "jobs.json"
            original = write_job_snapshot(
                manifest,
                snapshot,
                allow_large_run=False,
            )
            manifest.write_text("preset: dev\npanels: []\n", encoding="utf-8")
            frozen = load_job_snapshot(snapshot)
            self.assertEqual(
                [(job.panel, job.model) for job in frozen],
                [(job.panel, job.model) for job in original],
            )
            self.assertEqual(snapshot.stat().st_mode & 0o777, 0o444)


if __name__ == "__main__":
    unittest.main()
