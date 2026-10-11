# Controller lockf repair acceptance

BMRC returned ACQUIRED for concurrent flock calls on compa002/compa003.
The same nodes returned BLOCKED for lockf. The repaired POSIX launcher uses
lockf for both bootstrap and controller submission journals.

The acceptance test calls the real controller_lock and Journal. It checks
contention timeout, waiting after exception release, a subprocess while the
parent holds its lock, 32 concurrent synthetic registrations, idempotent replay,
and release after a lock-owning process is terminated. It never submits real
jobs through the tested Journal.

Local controller/submission regression: 40 passed. Discoverer job 4455494:
12 Linux primitive regressions passed, followed by two-node acceptance PASS on
cn0383/cn0384. All 32 synthetic registrations were retained. This is a controller
concurrency check, not a model-training acceptance.

## BMRC: extract and test without changing the experiment checkout

From the existing checkout whose origin is OxfordDemSci/aleatoric_luck:

~~~bash
git fetch origin 'SMR&FFC'
# Set this to an existing shared scratch directory on BMRC.
scratch_parent=/absolute/path/to/shared/scratch
candidate=$(mktemp -d "$scratch_parent/controller-lockf-candidate.XXXXXX")
git archive FETCH_HEAD \
  launch/slurm_submission.py launch/slurm_resources.py launch/experiment.py \
  runs/controller-lockf-repair-20261001 |
  tar -x -C "$candidate"

sbatch --nodelist=compa002,compa003 \
  "$candidate/runs/controller-lockf-repair-20261001/bmrc-acceptance.sbatch" \
  "$candidate" "$candidate"
~~~

The job requests two nodes, one task per node, 512 MB per node and three minutes.
It creates a new synthetic run directory below candidate. The original
experiment state and original controller lock are never opened.

After completion, send these two files from the directory where sbatch ran:

- controller-lockf-acceptance-JOBID.out
- controller-lockf-acceptance-JOBID.err, even if empty

If present, also send acceptance.json from the acceptance directory printed in
the .out file. A valid PASS reports two different nodes, the source SHA-256,
32 synthetic registrations, and every listed check. A missing receipt or a
nonzero job exit requires reviewing the logs.

## Applying the launcher version

Use a clean checkout of the repaired commit for a new experiment. Keep every
job of a run on the same launcher version. Finish or stop jobs using the previous
flock version before switching its checkout; flock and lockf are different
lock interfaces. Completed experiments retain their original source and results.
