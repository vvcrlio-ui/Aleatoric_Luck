# Scheduler policies

A scheduler policy sets how a run uses a Slurm cluster: how many nodes and workers each round may take, how much memory and time each worker gets, and how the queue service is laid out. It does not change what is computed, so two runs that differ only in their policy give the same results. Pass one with `--scheduler-policy PATH.json`; without it, every field keeps its default, which suits small runs. Options given on the command line, such as `--dispatcher-shards`, override the same field in the file. A run saves the file it started with as `scheduler-policy.initial.json`.

[discoverer-cache.json](discoverer-cache.json) holds the values used for large runs on Discoverer: four queue service processes, one set of resource limits for the base models and the Super Learner, and a final check spread over the workers. To prepare a policy for another cluster or run size, copy [template.json](template.json) and replace its values; fields you leave out keep their defaults.

## Format

The file is one JSON object. Only the field names below and the others listed in `DEFAULTS` in [scheduler_policy.py](../../NK_Grid/src/aleatoric_nk_grid/scheduler_policy.py) are accepted; an unknown name or a value outside its range stops the launch with an error.

| Field | Default | Meaning |
|---|---|---|
| `unified_compute` | `false` | Base models and the Super Learner share one set of resource limits. Requires `worker_cap`, `worker_memory` and `worker_time_limit`. |
| `sizing_mode` | `"work"` | `work` sizes each round from the estimated remaining work; `capacity` fills the run's node count up to `worker_cap`. `capacity` requires `unified_compute`. |
| `max_nodes` | `null` | Lower cap on the run's `--nodes`, which counts the two nodes kept for control jobs. At least 3. A later edit can lower it but not raise it. |
| `worker_cap` | `null` | Most numerical workers in one round. `null` uses `--workers` or the profile's default. |
| `worker_memory` | `null` | Memory per worker as a Slurm size, such as `"3G"`. `null` uses `--memory`, or 2G for base workers. |
| `worker_time_limit` | `null` | Wall time of each worker allocation, `[days-]HH:MM:SS`. `null` uses `--time` or the profile's default. |
| `dispatcher_shards` | `1` | Number of queue service processes, 1–8. More than one requires `validation_processes`. |
| `validation_processes` | `0` | Result-checking processes per queue service, 0–32. |
| `node_relay` | `false` | Each node sends its workers' requests over a few shared connections. Requires `rpc_keepalive`. |
| `rpc_keepalive` | `false` | Workers keep their connection to the queue service open between requests. |
| `max_connections` | `128` | Connections the queue service accepts, 1–4096. |
| `max_submissions` | `32` | Results the queue service accepts at once, 1–1024; a value above half of `max_connections` is reduced to that half. |
| `parallel_verification` | `false` | Spread the final check of all results over the allocation's worker processes. |
| `max_cpu_hours` | `null` | CPU-hour budget for the run's workers. A later edit can lower it but not raise it. |
| `exclude_nodes` | `[]` | Node names never to request. |

Choose `--nodes`, `worker_cap`, `worker_memory` and `worker_time_limit` from the cluster's own limits: nodes per job, memory per core and maximum wall time. The launcher does not query account or QoS limits; a request beyond them stays pending in Slurm with its reason shown by `squeue`.
