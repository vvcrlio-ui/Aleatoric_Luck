# Aleatoric NK Grid

Shared, article-agnostic N×K sweep engine. Article folders own preprocessing,
schemas, panels and model parameters; this package owns validation, splitting,
per-cell preprocessing, modeling, metrics and checkpoints.

The locked environment supports Python 3.11–3.14. PyArrow 25 is pinned because
it provides wheels for the repository's Python 3.14 runtime as well as 3.11–3.13.

Run a panel manifest:

```bash
python -m pip install -e "NK_Grid[models,parquet]"
python -m aleatoric_nk_grid.run_panels \
  --manifest SMR/panels.yaml --dry-run
```

## Slurm

Submit the frozen panel/model mapping from any working directory:

```bash
bash NK_Grid/slurm/submit_nk_grid.sh \
  --manifest SMR/panels.yaml \
  --max-concurrent-per-class 20
```

Slurm execution has deliberately safer defaults than an interactive panel run:

- the frozen panel×model mapping is partitioned into up to three disjoint
  arrays: `parallel`, `serial` and `bart`; an empty resource class is not
  submitted;
- `parallel` models use the common `--cpus-per-task`/`--mem`/`--time`
  profile; `serial` models use `--serial-cpus-per-task` (default: 1) while
  inheriting common memory/time; BART can override all three with
  `--bart-cpus-per-task`/`--bart-mem`/`--bart-time`, with unset overrides
  inheriting the common profile;
- each worker replaces the submit-host `n_jobs` value with its allocated
  `SLURM_CPUS_PER_TASK`;
- completed outputs are reused by default; `--rerun-completed` is an explicit
  opt-in to recomputation;
- logs and execution use the engine directory as the fixed Slurm working
  directory;
- five minutes before the wall limit, USR1 asks the engine to stop after its
  current batch checkpoint. A watchdog waits 240 seconds by default; if the
  batch has not stopped by then, the worker explicitly requeues the array
  element from its last complete checkpoint. Set
  `--requeue-watchdog-seconds` to an integer from 0 through 240 to shorten that
  wait; the upper bound reserves 60 seconds for Slurm to process the requeue.
  Use `--max-restarts` to bound requeues;
- `--max-concurrent-per-class` is applied separately to every submitted
  resource-class array; it is a per-class throttle, not a global cap across
  all three arrays. The old ambiguous `--max-concurrent` option is rejected;
- each non-empty resource class receives its own Slurm job ID and receipt.
  That receipt records the class's effective policy/resources and an
  executable rerun command for every included array index, the advance-signal
  and watchdog settings, and model-affecting environment switches (including
  whether each optional switch was unset);
- before the first submission, the launcher verifies that the three resource
  classes cover every snapshot job. The separate `sbatch` calls are not
  transactional: if a later class fails, the error lists every class/job ID
  already submitted and the frozen snapshot path. Check `squeue` before
  retrying, then use the original `--snapshot PATH` together with
  `--resource-class parallel|serial|bart` and the same policy/resource flags
  for only the missing class; re-reading a possibly changed manifest is not a
  safe recovery;
- requeue, signal, log paths/open mode, working directory, exports, resources
  and array specification are passed explicitly on each `sbatch` command so
  submission behavior does not depend on stale directives in the worker
  script.

The worker uses the formally installed distribution. Source checkout fallback
is disabled unless `ALEATORIC_NK_GRID_SOURCE_FALLBACK=1` is explicitly set.
Set `PYTHON_MODULE` only on clusters that require a module to activate the base
interpreter used by the existing virtual environment.

The distinction between requeue eligibility, advance signals and explicit
requeue follows the official Slurm
[`sbatch`](https://slurm.schedmd.com/sbatch.html) and
[`scontrol`](https://slurm.schedmd.com/scontrol.html) semantics.

Seed-block sharding still requires a separate identity/output merge protocol
and is not emulated with `max_jobs`. The three resource classes are deliberately
coarse: finer per-model profiling or additional resource classes are follow-up
work.

## Runtime and checkpoint behavior

- Row/feature draw orders are deterministic within a run and are reused by a
  bounded `(seed, draw)` cache for thread and serial execution groups. Cached
  arrays are read-only. BART deliberately keeps process-local calculation:
  sending the order arrays with every loky task would trade permutation work
  for repeated serialization.
- Resume candidate scans read only `experiment_id`, the five cell-key columns
  and optional `status`. Resume planning never loads the metric columns, and it
  releases the projected index, completed-key set and full design list after
  deriving the pending list. A completed run is fast-reused only when this
  projected index exactly matches the current design: no duplicate, failed,
  missing or out-of-design cells. If a manifest claims completion but that
  index is not exact, the run fails integrity checking instead of silently
  rewriting or repeatedly slow-resuming the corrupt artifact. A manifest whose
  JSON root is not an object is treated as unreadable rather than allowed to
  abort unrelated candidate discovery.
- The lightweight completed-run check intentionally validates identity,
  cell keys and status rather than re-reading every metric column. It detects
  protocol/index corruption, but not external post-run edits to metric values;
  use an artifact checksum/archive policy if that threat is in scope.
- The execution `batch_size` remains the checkpoint-boundary and signal
  response unit. Every batch first publishes a small atomic WAL shard; after
  50 loose shards, a single writer atomically publishes one compact shard and
  only then removes those exact loose sources. Loose and compact shards live in
  separate subdirectories, so the per-batch threshold check scans at most the
  small loose set rather than every accumulated compact shard. A process
  interruption before publication leaves the sources intact; interruption
  during cleanup leaves harmless duplicate copies that checkpoint-key
  deduplication removes. The shard file and every newly created directory entry
  in its parent chain are fsynced before an older authority is removed.
- Consequently, a one-model production grid still makes 100,000 small
  checkpoint writes at the default batch size of 20, but stabilizes at about
  2,000 physical part files (brief publication peak: about 2,050). Raising the
  batch size globally to 1,000 would reach a similar file count while making
  slow serial/BART tasks lose up to 1,000 cells on a forced requeue, so it is
  not the production default. `timing_full` also uses 20; compaction, rather
  than a large recovery boundary, controls file count.
- Final materialization uses a temporary SQLite database as an on-disk reducer.
  Shards are validated and inserted in 2,000-row chunks, successful rows retain
  priority over failed retries, and the sorted CSV is emitted in 10,000-row
  chunks. Manifest diagnostics are aggregated in SQLite and final QA streams
  the sorted CSV, so neither finalization nor verification constructs the full
  metric table in RAM. The output filesystem must have temporary headroom for
  the SQLite database and atomically written CSV.
- On a Slurm checkpoint-boundary stop, the worker writes a lightweight
  resumable manifest from the projected checkpoint index and defers the full
  CSV merge to the next invocation. This keeps the 240-second watchdog from
  waiting on a multi-million-row materialization.
- Each declared output has an advisory filesystem writer lease. Duplicate
  array elements fail fast instead of concurrently finalizing or pruning the
  same checkpoint. The cluster filesystem must provide working POSIX
  `flock` semantics.
- After final CSV verification, the active `.parts` directory is atomically
  renamed to a hidden retired directory before best-effort deletion. An
  interruption during cleanup therefore cannot make a partial shard set
  override the already-verified final CSV.
- Standard models continue to use threads and Joblib task batching remains
  explicit at one cell. LightGBM and Super Learner fits run one at a time in a
  reusable spawn-based subprocess. A native abort or segmentation fault kills
  only that child. A cell exceeding `native_process_timeout_seconds` (six hours
  by default) is forcibly terminated as well. The parent recreates the worker,
  retries the cell once by default, and persists a normal failed row if both
  attempts crash or time out. The timeout and attempt count are recorded in
  experiment identity and the manifest. BART
  closure/memory redesign remains follow-up work; the checked-in SMR and FFCWS
  panels do not currently select BART.
- Production planning still materializes the job list and a completed-key set,
  so allow hundreds of MiB of scheduler-side memory in addition to the data and
  fitted models. Those duplicate planning structures are released before
  fitting. Use `--dry-run` to inspect the cell, checkpoint-write, stable part,
  peak part and maximum-uncheckpointed-cell estimates before submission.

The authoritative adapter contract is
[`../Adapter/ADAPTER.md`](../Adapter/ADAPTER.md).
