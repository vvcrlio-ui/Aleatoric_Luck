# How tasks are priced and scheduled

Within a round, worker processes take tasks from a queue service and send results back. This page describes the protocol between them, how the queue service estimates what each task costs, how the CPU-hour budget is charged, and when a round is ended early. None of this changes what a task computes: task identity, model settings, thread limits and success checks are the same however tasks are scheduled.

## Queue protocol

Runs use protocol 2. The queue service hands out leases on batches of tasks, sized by their estimated cost. A worker saves and submits each finished chunk of results while it goes on computing. Its heartbeats name the lease token, not a particular task.

Heartbeats are sent about every 600 seconds, with some jitter, and a lease lasts 3,600 seconds, so one delayed heartbeat does not cost a worker its lease. The queue service accepts only a limited number of result submissions at once (`max_submissions` in the [scheduler policy](policies/README.md)); when it is busy it replies with HTTP 503, and the worker keeps the result and tries again.

A chunk normally holds at most 512 KiB, including the whole request. A single larger row may use up to the server's 1 MiB limit. A row that cannot be sent at all is kept intact as evidence and never truncated or resent endlessly: the round drains, and the controller records `protocol_blocked` instead of starting another allocation with the same input.

## Estimating task costs

The controller prices each round from the timings of the run's own finished rounds, and from any profile placed in `cost-profile.json` in the run directory. A profile can be built from another run's result journal:

```sh
PYTHONPATH=NK_Grid/src python -m aleatoric_nk_grid.cost_profile \
  /path/to/stopped/round/results.jsonl -o /path/to/run/cost-profile.json
```

Use the full journal (`--stride 1`, the default). The builder keeps exact means, estimates quantiles from bounded samples and records their sampling error, and keeps successful, failed and skipped fits apart. Check that the dataset, grid, model settings and environment match before reusing another panel's profile. If any source in the profile belongs to a different plan, the whole import is ignored and the run relies on its own rounds.

Only a group of tasks with enough successful timings gets a batch price, taken from its 99th percentile. Tasks from a group without one are leased one at a time. A run without a suitable profile therefore gives the same results, but can move slowly through many cheap tasks.

For the prediction workflow, timings are grouped by phase, pipeline or variant, fold count, cache mode and training identity. Only tasks that did all their training from scratch set prices. Fits that reused cached predictions, resumed fits and incomplete out-of-fold work are kept in separate groups for diagnosis, so they cannot make a task look cheaper than it is. Each round's `operational.json` records the profile it used, its coverage and the policy.

## Budgets and policy changes

A scheduler policy passed with `--scheduler-policy` is saved as `scheduler-policy.json` beside `plan.json`. Fields and defaults are listed in [policies/README.md](policies/README.md). A small test policy might look like this:

```json
{
  "target_batch_seconds": 30,
  "flush_seconds": 10,
  "target_round_seconds": 14400,
  "max_cpu_hours": 100,
  "exclude_nodes": [],
  "drain_enabled": false
}
```

Choose `max_cpu_hours` for the actual panel; a production panel needs far more than this. The budget charges every allocated node in full for its wall time, and controller jobs for the time they reserved. When the actual allocation time is unknown, the full reserved time is charged. The budget can be lowered between rounds, but not raised, and removing the policy does not reset it.

A policy change applies only to rounds that have not started. Do not edit a submitted round's saved settings, the plan, the queue manifest or the checkout. `--rounds` or the manifest's phase limits set how many rounds the run may use. Per-round time limits for base rounds are described in [PREDICTION_CACHE.md](PREDICTION_CACHE.md#manifest-settings).

## Ending a round early

Every round drains before its allocation's time limit. When a round drains, workers keep their finished chunks, stop taking new tasks, get a grace period, and their job step is then ended so the controller can take stock and start a smaller allocation for the remaining work. An unresponsive filesystem can delay this; Slurm's time limit is the final bound.

A round can also be drained when its last tasks leave most workers idle. This is off by default. To turn it on, set `restart_overhead_seconds`, `restart_cpu_hours` and `max_cpu_hours` from the site's own measurements of start-up and scanning costs, and set `drain_enabled: true`. The round is then drained only when utilization stays low, the remaining work is priced, every worker's status is known, enough CPU budget remains, another round is allowed, and the no-progress limit has not been reached. A final batch in which every worker is still computing does not trigger it, and missing or stale worker reports block it.

## Progress and fault records

`control/<generation>/progress.json` counts computing, submitting, idle, unknown and slow workers, and records the estimated remaining work and any drain decision. A worker whose reported task has already been committed counts as unknown, not idle, until its next report. Reports from long tasks are jittered, and workers that find the queue empty poll less and less often.

A fault in the protocol itself notifies the queue service and leaves a shared fault marker. `control/round-result.json` records the job, generation and queue, where the elapsed time came from, the outcome and the reason for draining. The completed rows of the result journal decide what a resumed run still has to do and what goes into the final table. A result sent twice with the same lease token waits for the first copy to reach disk and is never stored twice.
