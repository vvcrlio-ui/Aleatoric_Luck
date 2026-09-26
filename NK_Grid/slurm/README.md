# Slurm scripts of the grouped-task protocol

New experiments do not use this folder. They are started with `run.sh` and scheduled by [launch/cluster_scheduler.py](../../launch/cluster_scheduler.py); [how runs are carried out](../../launch/README.md) describes that process.

The scripts here belong to the earlier protocol, in which tasks were grouped into fixed task tables and processed through a dynamic queue. They stay because a run started under that protocol has to be resumed or audited with the code it started with. When such a run is resumed, `run.sh` passes its plan to [legacy_submit_flat_task_table.sh](legacy_submit_flat_task_table.sh). `submit_nk_grid.sh` only prints a pointer to `run.sh`.

Resuming older runs is covered in [launch/OPERATIONS.md](../../launch/OPERATIONS.md#older-entry-points).
