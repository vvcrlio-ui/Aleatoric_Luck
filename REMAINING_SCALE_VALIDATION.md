# Remaining-key scale and bounded journal validation

Discoverer job `4445440` completed with exit 0 in 10m56s. The frozen engine
candidate was `fee77749a686499408f02f96782e9f11da96b7a2`. The probe
`checks/linux_remaining_scale_probe.py` has SHA-256
`74351ecb8a76210f8ed482bfb323fcf853b5336f2824d5804e4bb2b05eb2d837`.

The test derived the exact remaining keys at the fully audited first-round
capture on 2026-09-10 23:35:53 UTC from immutable assignment row groups,
validated completed-prefix positions, and the 19 failed-model keys. It
checked assignment hashes, row-group digests, the last completed cell,
full coverage, and per-model remaining counts. Numerical failures were
retained as pending work; malformed/protocol error records were rejected.
No production WAL was sealed or changed, and no production result was imported.

| Measurement | Observed value |
| --- | ---: |
| Remaining model keys | 3,508,216 |
| Streaming plan time | 121.523 s |
| Immutable task manifest | 337,507,580 bytes |
| Local SQLite index build | 255.524 s |
| Initial SQLite index | 2,244,231,168 bytes |
| Concurrent clients and peak simultaneous leases | 698 |
| Synthetic results durably accepted | 1,396 |
| Claim/heartbeat/submit/idempotent-submit workload | 7.542 s |
| Synthetic acceptance rate | 185.089 results/s |
| Logical request latency, median / p95 / maximum | 0.891 / 1.292 / 1.366 s |
| Transport retries | 0 |
| Tested durable journal | 3,489,857 bytes |
| Full index rebuild plus journal replay | 269.751 s |

Replayed result exports matched byte for byte. All 1,396 synthetic keys
were unique, and a lease held across the dispatcher restart was rejected.
These tests used one compute node, loopback HTTP, 16 allocated CPUs and
16 GiB allocated memory. They did not run model training.

The compute node's `/tmp` had only about 0.6 GB available, so the disposable
SQLite index lived on node-local `/dev/shm` and counted against the job's
memory budget. Queue manifests and fsynced journals lived on Lustre.
The process peak RSS was 297.3 MiB; that figure does not include the roughly
2.1 GiB tmpfs index. The launcher removed only its own temporary directory
when the job exited; durable evidence remains on project storage.

Evidence: `runs/remaining-scale-validation-4445440/` in the independent
server checkout, with its report and two small synthetic result exports
mirrored beside the local source checkout. The report binds the probe,
engine modules, audit summary and assignment hashes.

Initial jobs 4445436, 4445437 and 4445439 stopped during setup/preflight:
missing package metadata in the source archive, inadequate node `/tmp`,
and an overly strict check that rejected the known numerical-failure
catalog. They did not alter production and are retained in cutover state.

This establishes remaining-key plan/index capacity and a bounded 698-client
Lustre-journal round trip. The journal sample is about 3.5 MB, not the
complete production-lifetime journal. Cross-node TLS, actual Slurm
continuation wiring, full journal-volume recovery, a calibrated cost
profile and a final deployment receipt remain required. No training
speedup or production readiness is inferred from synthetic RPC throughput.
