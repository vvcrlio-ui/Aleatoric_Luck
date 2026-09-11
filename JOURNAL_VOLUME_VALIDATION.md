# Full-queue synthetic journal recovery capacity

Discoverer job `4445445` completed with exit `0:0` in 53m35s. The unchanged
engine was frozen at `fffe73232295f96e6339846abe125869c591cd5c`. Probe
`checks/linux_journal_volume_probe.py` has SHA-256
`2ea39bb2860e65bca37a81c7ca597190f3cc73e5acea39e66e35b9b49a4b1bbc`.

The fixture reused the validated 3,508,216-key manifest from job `4445440`.
This is a conservative size bound for the currently shrinking remainder,
not a claim that the old capture is the current set of missing results.
Every key had a synthetic lease, two heartbeats and one result containing
1,536 bytes of padding plus fields and a task fingerprint. The complete
fixture contained 14,032,865 events, including its initial restart event.

| Measurement | Observed value |
| --- | ---: |
| Model keys fully restored and verified | 3,508,216 |
| Immutable task manifest | 337,507,580 bytes |
| Complete journal prefix | 11,876,328,707 bytes |
| Fixture plan copy | 52.792 s |
| Buffered synthetic journal generation | 473.846 s |
| Real Dispatcher index rebuild and full replay | 2,494.464 s (41m34s) |
| Full result payload verification | 175.912 s |
| Rebuilt SQLite index | 17,757,044,736 bytes (16.538 GiB) |
| Python process peak RSS | 122.605 MiB, excluding tmpfs SQLite |
| Requested job memory | 48 GiB |

The probe first passed a ten-key native preflight. For the large fixture,
it appended an eight-byte uncommitted final frame. The real Dispatcher
removed that incomplete tail, preserved the entire committed prefix hash,
and appended its restart event. It restored all keys to done, with none
pending, leased, failed or exhausted. Every stored result matched its
original task and expected synthetic payload, and the ordered result stream
was hashed. The journal's complete prefix remained unchanged.

The manifest and journal lived on Lustre. SQLite lived in node-local
`/dev/shm` within the job memory allocation and was removed on exit. The
process RSS figure is not the total memory requirement: the index alone
exceeded 16 GiB. A deployment must budget for this observed index and its
other memory needs, and allow service readiness to follow a potentially
long replay. The tested 48 GiB allocation is the validated capacity point;
no smaller production allocation is established here. A short fixed startup
timeout would be incompatible with the measured 41-minute recovery.

Evidence is retained in the independent server checkout at
`runs/journal-volume-validation-4445445/`. Its report and Slurm logs are
mirrored locally beside the source checkout. Report SHA-256:
`03ed7f1189dc9e00143127ad4c4dc149ed993fbd07dbec0446588ace0f1f2ca3`.
The original committed journal-prefix SHA-256 is
`1ca9884346f0ceff1f00a13ad062ed107afa6c8912e74a00fc3161cd78cee5c1`.
Slurm's previously observed working-directory warning remains in the logs.

This is a recovery-capacity test at the stated event and payload bound.
The fixture writer flushes in batches and does not measure per-result
durability or RPC throughput. It does not establish all-round worst-case
retry volume, Slurm continuation integration, worker process crash recovery
or numerical model performance. All fixture output is synthetic and must
never enter the production experiment CSV. Deployment readiness remains
false until the remaining integration and configuration evidence is complete.
