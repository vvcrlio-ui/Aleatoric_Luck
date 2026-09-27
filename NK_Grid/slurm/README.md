# Slurm scripts for grouped task tables

These scripts run plans in the grouped task-table format, where tasks are grouped into fixed tables and processed through a dynamic queue. When `run.sh` is given such a plan with `--resume`, it passes it to [legacy_submit_flat_task_table.sh](legacy_submit_flat_task_table.sh).

New experiments use single-model plans instead; they are started with `run.sh` and scheduled by [launch/cluster_scheduler.py](../../launch/cluster_scheduler.py), as described in [how runs are carried out](../../launch/README.md).
