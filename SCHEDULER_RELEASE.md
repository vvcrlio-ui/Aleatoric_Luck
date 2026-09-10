# Scheduler and SVD release staging

User authorization: push to personal `vvcrlio-ui/Aleatoric_Luck`, default branch
`EuroHPC`, and have Discoverer fetch that repository into a separate checkout.
Do not alter the currently running frozen checkout. Code publication is separate
from activation of the production continuation.

## Resulting behavior

All workers consume one global pool. Each `(seed,draw,N,K,model)` is independently
leased, acknowledged and retried. Model classes are labels, with no reserved
worker partition. Estimated long tasks are eligible first; workers claim again
immediately after each result. Similar-cost cache affinity is bounded. The pool
does not split a fit/CV computation or shorten any model's training budget.

Optional cost profiles bind measured model/N/expanded-K durations and evidence
into the queue identity. The supplied local bootstrap has 36 model/cell points,
two repeats each; it does not establish full-grid timing accuracy on Discoverer.
New execution rows retain private fit/wall timing for later profile updates.
Profiles are frozen per plan, not silently changed during execution.

Raw cell inputs and outer diagnostic preprocessing may be cached. CV preprocessing
remains fold-local. Cache is off by default pending node-level contention/RSS
validation. Successful original NumPy SVD results are unchanged; finite-matrix
nonconvergence retries with scipy LAPACK `gesvd`, including Ridge inside SL.

`pending_resume` streams and validates the sealed old bundle, keeps a compact
completed-key bitset plus a scratch duplicate-hash index, then plans only missing
or failed model keys. A complete old experiment requires no new queue. Final
CSV output uses exactly the old public header and has no extra source columns;
source hashes, counts and original/new algorithm versions remain in sidecars.
Conflicts, malformed metrics, duplicate new keys or missing coverage prevent
publication of the final table. The CSV and its manifest receipt together form
the published result; a CSV without its validated receipt is not final.

The service supports private CA validation and hostname-checked TLS. Its CLI
refuses non-loopback plaintext binding. Tokens and private keys belong in the
private run directory, never Git. Queue backlog supports simultaneous arrivals;
idle polling is staggered rather than synchronized at 5 polls/sec/worker.

## Validation and activation boundary

56 local portable checks passed, including actual TLS certificate trust and
hostname rejection, crash recovery, partial-result resume, standard CSV merge,
SVD normal/fallback paths, and active-job cutover rejection. Six expected MLP
warnings come from a deliberately short integration fixture, not production
hyperparameter changes. Earlier complete-budget numerical comparisons and their
limits are in `LOCAL_VALIDATION.md` and `LOAD_BALANCE_698.md`.

On Discoverer, submit the bounded validation script only after checking real-time
quota. Its three arguments are the new repo, frozen old repo, and a fresh run
directory **inside the new repo's ignored runs/**. It copies 82 MB of prepared
inputs, isolates pytest dependencies, runs portable tests, then compares the real
native worker CLI's nine full public rows against the grouped native engine at
N122/K47, and verifies dispatcher restart. It does not submit a production run.

```bash
sbatch --account=ehpc-dev-2026d08-299 --qos=ehpc-dev-2026d08-299 --partition=cn \
  --output="$validation/slurm.out" --error="$validation/slurm.err" \
  NK_Grid/slurm/validate_single_model_scheduler.sbatch "$new_repo" "$old_repo" "$validation"
```

Production readiness still requires real legacy seal/export/resume validation,
remaining-design scale and Lustre persistence measurements, cross-node TLS and
current-round Slurm controller integration. Do not mark `deployment_ready=true`
based on local checks or the small native probe alone.

The user permits cutover from 2026-09-11 Paris onward, only after both boosters
have 2,000,000 unique valid results and **no old experiment job is running or
finishing**. Recheck under the old control lock. Never interrupt active jobs.
The existing half-hour monitor checks at minute 05 and 35; 10:00/18:00 reporting
continues. Final completion requires the normal CSV's full 18,000,000-key coverage.
