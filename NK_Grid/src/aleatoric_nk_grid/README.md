# How the core code runs an experiment

This package carries out one panel: it reads the prepared data, lays out every training combination, fits the models, and checks the results. The same numerical code runs locally and on a cluster.

## Reading the data and the design

[ingest.py](ingest.py) loads the schema and the analysis table. [validate_input.py](validate_input.py) checks values, category states, IDs and outcome missingness before any sampling, so a problem in the data stops the run before any model is fit. [run_panels.py](run_panels.py) reads the panel catalog, [config.py](config.py) holds the chosen outcome, models and grid, and [grid_contract.py](grid_contract.py) checks the design. [nk_grid.py](nk_grid.py) builds the N and K grids, makes each seed's train/test split, orders the rows and variables for each draw, and takes the first N rows and K variables.

## One training combination

[preprocessing.py](preprocessing.py) groups the columns that belong to one variable, so dummy columns and missingness indicators are selected together while each group keeps its own imputation rule. Its `FoldPreprocessor` learns imputation and scaling from the rows it is fitted on and applies them to validation or test rows. Cross-validation inside a model starts again from the raw values in each fold, and a variable that is unobserved in the training rows is set to a fixed prior value or NaN on both sides.

[model_registry.py](model_registry.py) builds each model. [fold_local.py](fold_local.py) runs the cross-validated tuning and final refit of Ridge, Lasso and the neural network, and [mlp_estimator.py](mlp_estimator.py) sets up the network's balanced mini-batches. [robust_linear.py](robust_linear.py) and [svd_fallback.py](svd_fallback.py) recover linear fits when the standard solver fails on finite input. [native_process.py](native_process.py) runs models that use native libraries in separate processes, so a crash or timeout can be retried. [evaluation.py](evaluation.py) computes the metrics and keeps the training-mean and test-mean baselines of the two R² measures apart.

## Saved predictions and the Super Learner

Each base-model task saves its test predictions and out-of-fold training predictions before it reports back ([prediction_training.py](prediction_training.py), [prediction_worker.py](prediction_worker.py), [prediction_cache.py](prediction_cache.py)). [prediction_workflow.py](prediction_workflow.py) moves a run through its phases: base models, an index of their saved predictions, the Super Learner and the final check. [offline_sl.py](offline_sl.py) fits the Super Learner from the saved predictions alone. [parallel_verification.py](parallel_verification.py) and [base_seal.py](base_seal.py) spread the checks over the allocation's worker processes before the final CSV is published.

## Running on a cluster

[cluster_queue.py](cluster_queue.py) prepares a run and publishes its results. [shared_queue.py](shared_queue.py) holds the queue of model tasks and the record of accepted results. [single_model_worker.py](single_model_worker.py) takes a task, runs it and returns the result, and [slurm_queue_round.py](slurm_queue_round.py) starts the queue service and the workers inside one Slurm allocation. [scheduler_cost.py](scheduler_cost.py) and [cost_profile.py](cost_profile.py) estimate task durations from timings measured in the run, so the longest tasks can start first. [execution_contract.py](execution_contract.py) records the inputs, methods and design of a run, so results are only merged with results produced under the same settings.

The earlier grouped-task protocol uses [flat_task_table.py](flat_task_table.py), [worker_event_wal.py](worker_event_wal.py) and [generation_control.py](generation_control.py); that code stays so runs started under it can be resumed.

See the [experiment methods](../../README.md) and the [root README](../../../README.md).
