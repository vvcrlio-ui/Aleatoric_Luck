# Native sealed-WAL migration validation

Discoverer job `4445434` completed successfully on 2026-09-10 UTC in 3m09s.
The numerical/scheduler candidate was `45d4ed9ac8279b48c4a3a2f37a62b3dbf0e0d9a2`;
the frozen legacy source was `ada4cfb5f6868e3f76412d2fea46cd1eb22dd5ee`.
The probe at `checks/linux_legacy_resume_probe.py` has SHA-256
`dbcd0429bb896577b49c1344535d22887f4c517c1c19f0d41484c81c7d0084e0`.

The independent fixture used actual GPA inputs, OLS and Ridge, seed 12345,
draw 0, N={122,123}, K=47. The frozen old code created the immutable task
table and contracts and ran its real `run_slice`/WAL writer. After real
model fits, the fixture injected one failed Ridge result and interrupted
one two-model group between its durable STARTED and RESULT records.
These injections affected only the isolated validation run.

The candidate then used the real `generation_control`/`flat_task_table`
seal and verification protocol, `result_migration.export_sealed`,
`pending_resume.prepare`, the single-model dispatcher, and
`pending_resume.merge`. No export adapter was mocked.

- Active and unknown generations could not be exported.
- Sealed legacy verification correctly remained incomplete: one valid key,
  one failed key, and two missing keys from the interrupted group.
- The pending queue contained exactly three keys; the existing OLS success
  was never claimed again. Three real candidate model fits matched the old
  full public result rows except for the declared algorithm version.
- An incomplete merge could not publish a CSV. Identical copied old rows
  were deduplicated; conflicting copied old rows could not publish a plan.
- Dispatcher restart retained all three completions.
- The final four-row CSV had exact unique coverage, the original column
  order, one old-version row and three new-version rows, and no added
  source columns. The original fixture WAL hashes remained unchanged.

Evidence is in the independent server checkout under
`runs/legacy-resume-validation-4445434/`, mirrored beside the local source
checkout. Its `report.json` binds the export, resume and result hashes.
Final test CSV SHA-256:
`72facd09a9dbe449ee48c91e6dc193f5a685c31d0fe9a376cbb5264b01b027cc`.

This verifies a bounded Linux protocol and numerical continuation case.
It does not validate remaining-production scale, 698 clients, cross-node
TLS, Slurm continuation wiring, or a production cutover. Those gates remain
open and `deployment_ready` remains false.
