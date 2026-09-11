# Two-node TLS and crash/spool recovery validation

Discoverer job `4445443` completed with exit `0:0` in four seconds on
`cn1105` and `cn1106`. It used two Slurm tasks, one per node, with 2 GiB
requested per node. Slurm allocated four logical CPUs in total. The engine
was frozen at `3173d266a286762a1a5a0a501bb0e93b7bed489e`.

The probe `checks/linux_cross_node_probe.py` has SHA-256
`c5cada0eff8e5b21cbf2f93a53fd83d10cc1d60155dd370d96c7c2ff33b5e87b`.
It created a new queue containing three explicitly synthetic model keys.
The service used the real `queue_service` CLI on one node; the other node
used the real client and `execute_worker` loop. SQLite was disposable
node-local tmpfs; the manifest, journal and worker spool were on Lustre.

The following checks passed:

- Trusted TLS communication across the two nodes; rejection of an
  untrusted certificate, wrong hostname, wrong bearer token and wrong queue.
- A second dispatcher on the other node could not acquire the queue's
  existing OS ownership lock on Lustre.
- A model result was durably acknowledged, and an identical repeat was
  accepted without creating a duplicate.
- An injected submit outage, before delivery to the server, left one
  result in the worker spool after the worker's bounded deadline.
- The probe killed only its own synthetic service child process with
  SIGKILL, then restarted the real CLI. The acknowledged result survived;
  the unacknowledged lease returned to pending along with the untouched key.
- The restarted service rejected the old heartbeat and old submission.
  Repeating the previously accepted result remained idempotent.
- A new invocation of the worker loop recomputed exactly the two pending
  keys, including the stale spooled result. It did not recompute the already
  acknowledged key. The spool was empty after acknowledgment.
- A further dispatcher rebuild exported exactly three unique synthetic
  results. Its journal was 11,606 bytes.

The temporary credential directory was mode 0700; token and private key
were mode 0600 and were removed after successful completion. Credentials
were not included in fetched evidence or printed to logs.

Evidence is in the independent server checkout's
`runs/cross-node-validation-4445443/`, mirrored locally beside the source
checkout. The report binds the probe and engine module hashes. Slurm stderr
contains working-directory warnings also observed in earlier validation
jobs; these are retained in the evidence. Both node tasks and all probe
assertions completed, but this is not a claim that Slurm integration is
free of operational warnings.

This validates a bounded two-node transport, ownership and service-crash
recovery path. It does not validate production Slurm continuation/controller
wiring, worker process crash recovery, full journal volume, certificate
rotation, large-node concurrency or numerical training throughput. No
production state, old WAL, experiment result or running job was changed.
Deployment readiness remains false pending the remaining acceptance work.
