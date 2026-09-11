# Dispatcher readiness prerequisite

The measured 3,508,216-key rebuild and journal replay takes 2,494.464 seconds.
A Slurm controller must wait until this completes before admitting workers. A
short startup timeout or the presence of an old readiness file cannot prove that
the dispatcher is available for the current attempt.

The optional `queue_service` arguments `--ready-file` and `--generation` publish
an atomic readiness record only after the `Dispatcher` constructor completes its
full rebuild/replay and the HTTP/TLS socket is bound. `--advertise-host` supplies
the hostname that workers can resolve and that the TLS certificate covers. The
record contains the queue ID, controller-assigned launch generation, dispatcher
epoch and URL. It contains no bearer token or private key.

`python -m aleatoric_nk_grid.queue_readiness READY_FILE --queue-id QUEUE_ID
--generation GENERATION --token-file TOKEN_FILE --ca-file CA_FILE` verifies the
record and performs an authenticated live identity RPC. It checks that the queue
and service epoch match, then rereads the file to detect replacement during the
handshake. Non-loopback endpoints require HTTPS. The CLI prints the verified
record for the controller to consume. It neither submits nor cancels Slurm jobs.

The default wait is 5,400 seconds, longer than the measured 41m34s replay. A missing
file or temporarily unavailable endpoint is retried within a bounded wait;
identity mismatches and malformed records fail closed. Exhausting the wait never
authorizes worker admission. RPC timeout is bounded by the remaining wait.
Normal service exit removes its matching record. A process crash may leave the
file behind; a live identity check is still required and rejects that stale file.

## Local validation

`checks/readiness_checks.py` plus the existing queue, worker and TLS checks:
**21 passed in 11.47 seconds** on Windows. Evidence log:
`../readiness-regression-20260911T034500Z.log`.

Five new test functions cover delayed publication, absent-file timeout, queue and
generation mismatch, stale epoch, plaintext remote endpoint rejection, incorrect
authentication, cleanup ownership, the real service and readiness CLIs, a killed
service leaving a stale file, and failure during journal replay. The corruption
test initially targeted the wrong file (`journal.jsonl` rather than
`events.jsonl`); this was a test-fixture error, corrected before the successful
run. The earlier log is retained as `readiness-regression-20260911T034200Z.log`.

## Required Slurm integration

The controller still needs to assign a unique generation and readiness path to
each launch attempt; store them in durable current-round state; select resources
from current limits; wait for verified readiness; and only then submit the worker
array for that round. Keep one persistent shared-storage spool path per logical
worker slot. Before admission and restart, recheck jobs and controller ownership
under the exact run's control lock. Readiness alone is not a controller lock or
proof that the old experiment is idle.

Dispatcher deployment must retain the validated 48 GiB memory allocation, provide
space for the measured 16.538 GiB tmpfs index, and declare the tested journal
payload/heartbeat bounds. The 90-minute readiness wait is a deployment candidate,
not a newly measured performance bound. Slurm startup/restart wiring, Linux native
execution and cross-node testing against these edits remain outstanding.

All changes are local, uncommitted and unpublished. The existing pending
publication question for commit `1b7fca8` does not include them. Deployment and
production cutover gates remain false.
