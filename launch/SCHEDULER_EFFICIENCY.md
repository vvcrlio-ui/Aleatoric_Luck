# Cost-bounded queue protocol

New shared-controller allocations use protocol 2: the dispatcher prices leases,
workers save and submit completed chunks independently of subsequent computation,
and heartbeats address the lease token rather than its first cell. Legacy
Dispatcher entry points explicitly retain protocol 1.

The scientific task identity, model settings, thread limits and success checks
are unchanged. Every round still runs its frozen checkout and plan. Existing
running panels must retain their original checkout.

## Build an operational profile

From a stopped result journal, with the new checkout on `PYTHONPATH`:

```sh
PYTHONPATH=NK_Grid/src python -m aleatoric_nk_grid.cost_profile \
  /path/to/stopped/round/results.jsonl -o /path/to/new-panel/cost-profile.json
```

Use full input (`--stride 1`, the default) for batch pricing. The builder uses
bounded float32 reservoirs, retains exact means, records quantile sampling error,
and separates successful, failed and skipped timing observations. Complete outer
worker timing is preferred; historical model-only timing remains explicitly
labelled. Validate the dataset, grid, model parameters and environment before
reusing another panel's profile.

Only an exact group with sufficient successful observations supplies a p99 batch
price. Unknown groups are leased one cell at a time. Old mean-only profiles still
support mean work estimates but do not authorize large batches. Thus deploying
without a suitable profile is safe but can materially reduce cheap-cell throughput.
The round's `operational.json` records profile coverage, source hashes and policy.

Prediction workflows additionally match phase, pipeline/variant, fold count,
cache mode and frozen training identity. Only complete cold work supplies their
mean/p99 prices. Cache hits, resumed fits, missing OOF and unclassified workflow
observations retain separate diagnostic timing groups; they cannot make a cold
task appear cheaper. Workflow profiles built before this distinction are
unpriced until rebuilt from rows that prove complete cold work. Ordinary legacy
profiles keep their previous interpretation.

## Optional policy beside the new plan

Place `scheduler-policy.json` beside `plan.json`. For example, a **local-policy
example, not a production resource recommendation**, is:

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

Choose a CPU-hour budget for the actual panel before launch. A 100-hour example
does not fit an 18-million-cell production panel. The existing account/QoS and
whole-allocation resource checks also apply. A budget can be tightened during
continuation; it cannot silently be raised or reset by removing the policy.
Missing trusted allocation-time evidence is charged at the full reserved duration.
Controls are conservatively charged by their reserved duration.

Policy/profile edits affect only rounds without a saved operational snapshot.
Never edit a submitted round's snapshot, frozen plan, queue manifest or checkout.
The existing `--rounds` option determines the total permitted rounds; this change
does not expand any frozen round budget.

New prediction workflows may freeze `execution.base_round_time_limits`, with
one `HH:MM:SS` entry per base round. For example, a separately approved plan can
freeze global wall time `08:00:00` and `['01:00:00', '08:00:00']`: its first
allocation drains at one hour, then the ordinary missing-task and live-capacity
checks size a smaller continuation. Each entry must fit the frozen global time
limit; current QoS/partition limits still apply. Submitted round count determines
the entry after a controller restart, and CPU-hour usage remains cumulative.
Omitting the field preserves the existing behavior.

The shared dispatcher requests one multi-node Slurm allocation. Job-count
limits reserve one allocation plus its controller jobs; they do not count its
worker processes as separate jobs, and `MaxArraySize` does not apply. Admission
checks every scoped CPU, memory and node headroom independently against the
whole allocation, including conservative billing for a partially filled last
node. Per-job limits, live wall-time limits, QoS CPU-minute headroom and the
cumulative run budget still apply. Legacy job-array admission retains its
original worker-count behavior.

Automatic economic tail drain is disabled until measured migration costs are
available. To enable it, supply positive `restart_overhead_seconds`,
`restart_cpu_hours` and `max_cpu_hours`, then set `drain_enabled: true`. These
values must come from the site's scan, initialization and control measurements.
The decision requires sustained low utilization, priced remaining work, no unknown
worker status, sufficient remaining CPU budget and an available successor round.
It also checks the no-progress limit. A normal final batch with everyone computing
does not authorize drain. Missing/stale telemetry blocks automatic economic drain.

Runtime keeps completed chunks durable, stops new claims, allows a drain grace,
and terminates the old step before the controller scans and starts a smaller
allocation. Allocation-deadline drain is always active. An unresponsive filesystem
can delay software shutdown; Slurm's allocation time limit is the final bound.

## Faults and evidence

`control/<generation>/progress.json` includes computing, submitting, idle, unknown
and straggler counts, profile work estimates and costed drain decisions. A worker
whose reported cell has already committed is conservatively unknown until its
next status; it is not counted as idle to justify a drain. Long-cell status reports
are jittered; empty-queue claim polling backs off.

Chunks normally use at most 512 KiB including the entire encoded RPC envelope.
A single larger row may use up to the server's 1 MiB limit. An unrepresentable row
is retained intact as blocked evidence; it is never truncated or endlessly replayed.
Terminal protocol faults notify the dispatcher and write a shared fault marker.
The round drains, and the controller records `protocol_blocked` without submitting
another worker allocation with the same poison input.

`control/round-result.json` records job/generation/queue identity, elapsed-time
source, outcome and drain reason. Completed journal rows remain the authority for
resumption and final publication. Same-token replays wait for the original durable
commit and cannot append duplicates while fsync is in flight.

## Validation and remaining gates

Run the tracked regressions in [validation/README.md](../NK_Grid/validation/README.md).
The synthetic multi-process/multi-node probe is
`NK_Grid/validation/efficiency_probe.py`; it never performs scientific fits or
submits jobs itself. It validates TLS, journal durability, partial submission,
heterogeneous leases and exact missing-task continuation. Its timings are not
evidence of production savings.

Native scientific fixed-task A/B and one complete production pilot remain gates
before claiming reduced scientific CPU-hour cost. In-round compute-process
supervision, automatic node quarantine and SMT packing remain separate changes;
the present recovery mechanism releases the whole allocation and resumes its
missing work. No model-thread or SMT binding defaults were changed.
