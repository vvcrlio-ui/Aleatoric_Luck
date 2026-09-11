# Worker process recovery: local validation

The numerical worker previously generated a new worker ID for every process.
After a crash, its replacement could not reclaim the same live lease immediately.
Also, a result accepted just before the worker died was no longer returned by
`claim`, leaving the corresponding local receipt unreconciled.

The candidate now persists a queue-bound logical worker identity in each spool
directory. An exclusive operating-system lock prevents concurrent ownership of
that directory. `single_model_worker run --spool` acquires this lock before opening
the numerical session and retains it until the session closes. Restart the logical
slot with the same durable spool path; give every concurrent slot its own path.
Do not copy active spool directories. For node replacement the path must reside on
shared storage, with its identity and lock file retained.

At startup, `execute_worker` tries to submit retained receipts before claiming new
work. The dispatcher remains authoritative: it acknowledges an identical accepted
result, accepts a still-valid lease, and rejects an expired or superseded token.
Rejected receipts remain as evidence. An unfinished task reclaimed under a new
token must be recomputed; the old result cannot overwrite a later accepted result.
Malformed or foreign-queue spool data fails closed. No result schema, model
parameter, CV logic, or training budget changed.

## Evidence

`checks/worker_process_probe.py` uses real subprocesses and the real dispatcher,
HTTP client and worker loop, with one synthetic model task per scenario. It kills
only the child processes that it created, at synchronized barriers, and starts
replacement processes. A second process attempts to own each live slot and is
rejected; the replacement retains the same identity and rejects a wrong queue.

Local evidence: `../worker-process-validation-20260911T031300Z/report.json`.
Platform: Windows, Python 3.12.12. All six scenarios passed in 7.093 seconds.

| Crash/restart scenario | Verified outcome |
| --- | --- |
| During synthetic execution, before receipt | Interrupted task recomputed; one final result |
| Receipt persisted, before submit delivery | Receipt submitted after restart; no recomputation |
| Result accepted, before reply delivered | Idempotent acknowledgment and receipt cleanup; no recomputation |
| Accepted result, then dispatcher restart | Journal recovery and idempotent acknowledgment; no recomputation |
| Receipt persisted, lease subsequently expired | New lease recomputed; one final result |
| Expired receipt, another worker already won | Winning payload preserved; obsolete receipt retained |

The related existing regression suite passed **16/16** checks:
`checks/queue_checks.py`, `checks/worker_checks.py`, `checks/tls_checks.py`.
Log: `../worker-process-regression-20260911T031500Z.log`.
The new process probe is standalone and requires only the Python standard library.

## Remaining deployment gates

These are local Windows processes and synthetic results over loopback HTTP.
This does not establish Linux native model behavior, Slurm restart integration,
cross-node worker-slot locking on Lustre, or production throughput. Those checks
must be run against the final published candidate, using the same stable slot
paths in the Slurm launch configuration. Existing Linux receipts apply to their
recorded earlier commits and do not certify these new edits.

The source edits, this document and the new probe are local, uncommitted and
unpublished. The pending approval for commit `1b7fca8` does not include them.
`deployment_ready` and `worker_process_crash_validated` remain false; the separate
`worker_process_local_validation_passed` flag records only this bounded evidence.
