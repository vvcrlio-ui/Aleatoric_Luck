# Running an experiment on a cluster

Cluster execution distributes predefined N/K, seed, draw, and model combinations among worker processes. All workers share the same sampling, preprocessing, and model-training logic.

## Define the experiment, then distribute tasks

The code freezes every `(seed, draw, N, K, model)` combination. All new cluster submissions use one shared single-model queue, with estimated expensive tasks first and workers free to claim any model.

Slurm allocates resources according to the submission script. Workers obtain tasks, call the engine, save results, and continue with subsequent tasks. Before preparing another batch, the program excludes combinations with valid saved results.

## Resuming interrupted work

Workers save task states and results as they run. The program uses these records to track progress.

After a job ends or is interrupted, the code checks completed combinations and schedules only the remainder on resume. Combinations whose results were not saved are retrained.

## Publishing final results

The merge step compares results with the original design, checking for missing combinations, conflicting duplicates, and matching method identities. It generates the final CSV once publication requirements are met. A `skipped` record represents a combination omitted because of data conditions, rather than a completed fit.

If checkpoint cleanup is selected, it takes place after result validation.

## Adapting to different clusters

Cluster configuration covers accounts, partitions, CPU and memory requests, wall-time limits, environment loading, and data paths. BMRC, Discoverer and other configured Slurm clusters all use `launch/cluster_scheduler.py` for submission and resumption. Each round checks live scoped limits and CPU-minute headroom, reserves a dispatcher task, and submits one worker allocation. Resource checks are conservative; Slurm determines when the allocation starts.

Use `run.sh slurm` for new experiments. `submit_flat_task_table.sh [--submit] PLAN.json` routes saved single-model plans to the shared scheduler and historical grouped plans to their original journaled protocol. The pre-table `submit_nk_grid.sh` submission path is retired. Historical snapshot, input and code checks remain in force; recover existing jobs from their original frozen checkout, or explicitly migrate sealed results. Calibration scripts remain diagnostic tools. The [BMRC multi-panel suite](../../launch/BMRC.md) uses a shared pool across panels and a fixed resource file.

See [cluster_queue.py](../src/aleatoric_nk_grid/cluster_queue.py) for task preparation and publication, [launch](../../launch/README.md) for the launch flow, and the [root README](../../README.md) for examples.
