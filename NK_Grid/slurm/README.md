# Running an experiment on a cluster

Cluster execution distributes predefined N/K, seed, draw, and model combinations among worker processes. All workers share the same sampling, preprocessing, and model-training logic.

## Define the experiment, then distribute tasks

The code first determines all training combinations in the experiment and organizes them into tasks. Depending on the scheduling path, tasks are assigned in groups or claimed individually from a shared queue.

Slurm allocates resources according to the submission script. Workers obtain tasks, call the engine, save results, and continue with subsequent tasks. Before preparing another batch, the program excludes combinations with valid saved results.

## Resuming interrupted work

Workers save task states and results as they run. The program uses these records to track progress.

After a job ends or is interrupted, the code checks completed combinations and schedules only the remainder on resume. Combinations whose results were not saved are retrained.

## Publishing final results

The merge step compares results with the original design, checking for missing combinations, conflicting duplicates, and matching method identities. It generates the final CSV once publication requirements are met. A `skipped` record represents a combination omitted because of data conditions, rather than a completed fit.

If checkpoint cleanup is selected, it takes place after result validation.

## Adapting to different clusters

Cluster configuration covers accounts, partitions, CPU and memory requests, wall-time limits, environment loading, and data paths. The integration also controls batched submission and resumption. Task computation and result merging use shared code. Other clusters require environment configuration and submission scripts that match their requirements.

Use the root entry point for routine runs. `calibrate.sbatch` probes memory usage with synthetic data; `plan_production.sbatch` and `make_production_request.py` support manual planning; `submit_nk_grid.sh`, `run_nk_grid.sbatch`, and `finalize_seed_shards.sbatch` maintain legacy static runs.

See [flat_task_table.py](../src/aleatoric_nk_grid/flat_task_table.py) for the task-table implementation, [launch](../../launch/README.md) for the launch flow, and the [root README](../../README.md) for execution examples.
