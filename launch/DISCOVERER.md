# Discoverer site notes

Use `--profile discoverer` and an explicit authorized `--account`. Commands, preparation and recovery are described in [OPERATIONS.md](OPERATIONS.md).

| Profile value | Setting |
|---|---|
| Python module | `python/3/3.12/3.12.4` |
| Partition | `cn` |
| Constraint | Unset |
| Maximum requested time | 48 hours |
| Default QoS | The supplied account |

The profile is [profiles/discoverer.sh](profiles/discoverer.sh). Worker count follows live account, QoS, partition, CPU-minute and memory limits. The optional [cache policy](policies/discoverer-cache.json) requests resources for large cache/SL experiments through `--scheduler-policy`.

The site module's Python needs an available PyYAML installation for login-node `--prepare` checks. Use `NKGRID_BOOTSTRAP_PYTHON=/path/to/existing/python` to select an interpreter that already provides it. The compute bootstrap loads the module and builds or reuses the locked shared environment independently.

Run directories and raw inputs must be on compute-visible storage. Under `/valhalla/projects/PROJECT`, storage admission uses `lfs` project quotas. Bootstrap temporary downloads stay in the run directory and pip cache stays beside the shared environments.

Check current jobs and QoS headroom before submitting:

```bash
squeue -u "$USER"
scontrol show assoc_mgr qos=YOUR_ACCOUNT flags=qos
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,AllocCPUS
```

QoS `GrpTRESMins` and the project's allocation balance are separate limits. Inspect current values for the account being used.

Site references: [Slurm jobs](https://docs.discoverer.bg/writing_slurm_batch.html), [login node limits](https://docs.discoverer.bg/cpu-login-node-resource-limits.html).
