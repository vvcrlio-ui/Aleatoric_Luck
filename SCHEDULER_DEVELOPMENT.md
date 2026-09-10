# Single-model scheduler: development and validation

This branch is an opt-in implementation. It does not modify or submit the running
Discoverer experiment. The user approved development on 2026-09-10 and then
explicitly added conditional SVD recovery. Production cutover remains gated on
verified completion of XGBoost and LightGBM, zero running/finishing old jobs,
and acceptance of the checks below. Personal-repository push and isolated server
checkout are authorized; publishing code does not activate the new production run.

## Implemented components

- `shared_queue.py`: immutable streamed single-model design, cost-priority claims
  with bounded cache affinity, one in-flight lease per worker, heartbeat expiry,
  bounded infrastructure retries, restart fencing, checksum-chained durable
  events, idempotent result acknowledgments and drain mode.
- `queue_service.py`: bearer-authenticated worker API, loopback by default. No
  administrative pause/import operation is exposed to worker credentials. An
  optional local `--drain-file` stops claims while current results drain; removing
  it resumes claims. This file is authoritative when the option is supplied.
- `single_model_worker.py`: explicit plan/worker CLI using the existing validated
  CellExecutionSpec and numerical session. It never submits Slurm jobs.
- `cell_cache.py`: per-worker RAM LRU and optional shared node-local read-only
  joblib/mmap cache. Namespace binds the validated spec or actual input content.
  Only raw slices and outer diagnostics are reused; CV retains original missing
  values and fold-specific preprocessing. Shared disk admission is bounded;
  deletion/rotation happens after readers exit, not while mappings are live.
- `result_migration.py`: sealed legacy WAL exporter and offline two-pass import,
  preserving source identity and accepting only compatible numerical settings.
  Successful models in an old group import individually; failures stay pending.
- `svd_fallback.py`: original NumPy SVD first, scipy `gesvd` only after LinAlgError
  on a finite input matrix. Normal successes are returned without modification.
  Fold-local Ridge records a fallback count and logs actual activations; both
  standalone Ridge and the Ridge base learner inside SL use this path.

## Resource and durability boundary

There are many logical model tasks and a bounded number of Slurm workers.
Workers do not open the dispatcher's database. SQLite is a local rebuildable
index; manifest and fsynced events are authoritative. Acknowledgment follows
durable output. Exactly-once **acceptance**, not exactly-once computation, is the
promise. Expired/restarted worker credentials cannot overwrite accepted results.

The first dispatcher rebuilds its index by streaming the design and replaying
events. It does not yet have an incremental snapshot/compaction service. An
remaining-task restart must be budgeted and measured before production use;
small local tests do not establish production throughput. The new pending-only
planner leaves the sealed old bundle as the immutable base, avoiding millions of
import journal events; see `SCHEDULER_RELEASE.md` for the latest additions.

Shared input cache is trusted private node scratch. Do not point it at arbitrary
downloaded joblib files or expose its writable directory to other users. Its
initial global build lock intentionally avoids duplicate preparation. Benchmark
its contention and provision quiescent batch rotation on the cluster before use.

## Local validation

Run `python -m pytest checks/queue_checks.py checks/svd_checks.py checks/migration_checks.py checks/cache_checks.py checks/worker_checks.py`
with `NK_Grid/src` on PYTHONPATH. There are 35 passing local checks; results and
measurement limits are recorded in `LOCAL_VALIDATION.md`. The worker defaults to
no cross-task cache; enable its RAM/node options after workload-specific testing.
`checks/gpa_queue_benchmark.py` is a bounded Windows-only local throughput study:
three independently pinned physical cores, full seven-model parameters, four
GPA cells, two reversed-order repeats. All arms use in-process numerical fits.
It includes queue RPC and fsynced results, but excludes Linux native subprocess
isolation, Lustre behavior, inter-node transport, Slurm startup and shared-node
contention with other jobs. Startup/input loading is measured separately.

The saved original-version N122/K400 predictions are checked independently of
cross-arm equality. Maximum-grid and multi-seed performance are not inferred
from the four-cell benchmark. Historical real SVD failure evidence remains in
the original repository's `runs/ffc-gpa-production-20260909/` diagnostic folders;
these records are evidence, not already imported production repairs.

## Required before production cutover

1. Freeze/review the development commit and source compatibility certificate.
2. Complete real legacy seal -> export -> import -> resume tests on Linux, with
   failed and interrupted groups, duplicate results and crash injection.
3. Exercise the actual worker CLI, native process isolation and result schema.
4. Measure large-plan preparation/restart/import and dispatcher throughput. Size
   local scratch, journal retention, memory, request fanout and cache rotation.
5. Validate Discoverer node connectivity and authenticated transport; loopback
   HTTP tests do not authorize exposing an unencrypted service across networks.
6. Verify both boosting models' full expected key sets and valid result coverage.
7. Wait for all old running/finishing jobs to end naturally. Do not interrupt
   active workers/controllers. Under the exact old control lock, query the full
   job scope again; only when it is quiescent may an unstarted pending successor
   be cancelled and the old workflow stopped. Unknown states block cutover.
   Seal and verify the old checkpoint history.
8. Import into a new queue identity, run a small recovery batch, validate old/new
   provenance and complete coverage, then increase remaining-work concurrency.

Never patch the original analysis IDs, original WALs or frozen production source.
Do not treat completion of this development benchmark as production acceptance.
