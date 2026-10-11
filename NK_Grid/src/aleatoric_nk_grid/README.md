# How the core code runs an experiment

This package carries out one panel on a Slurm cluster: it reads the prepared data, lays out every training combination, fits the models, and checks the results.

## Reading the data and the design

[ingest.py](ingest.py) loads the schema and the analysis table. [validate_input.py](validate_input.py) checks values, category states, IDs and outcome missingness before any sampling, so a problem in the data stops the run before any model is fit. [run_panels.py](run_panels.py) reads the panel catalog, [config.py](config.py) holds the chosen outcome, models and grid, and [grid_contract.py](grid_contract.py) checks the design. [nk_grid.py](nk_grid.py) builds the N and K grids, makes each seed's train/test split, orders the rows and variables for each draw, and takes the first N rows and K variables.

## One training combination

[preprocessing.py](preprocessing.py) groups the columns that belong to one variable, so dummy columns and missingness indicators are selected together while each group keeps its own imputation rule. Its `FoldPreprocessor` learns imputation and scaling from the rows it is fitted on and applies them to validation or test rows. Cross-validation inside a model starts again from the raw values in each fold, and a variable that is unobserved in the training rows is set to a fixed prior value or NaN on both sides.

[model_registry.py](model_registry.py) builds each model. [fold_local.py](fold_local.py) runs the cross-validated tuning and final refit of Ridge, Lasso and the neural network, and [mlp_estimator.py](mlp_estimator.py) sets up the network's balanced mini-batches. [robust_linear.py](robust_linear.py) and [svd_fallback.py](svd_fallback.py) recover linear fits when the standard solver fails on finite input. [evaluation.py](evaluation.py) computes the metrics and keeps the training-mean and test-mean baselines of the two R² measures apart.

## One cell

[nk_grid.py](nk_grid.py)'s session fits every base model of one cell (seed, draw, N, K). [prediction_training.py](prediction_training.py) fits each model on all N training rows and once in each out-of-fold (OOF) fold, reusing the full fit's selected penalty or number of rounds in the folds. [offline_sl.py](offline_sl.py) combines SL7 from the seven models' OOF predictions without fitting any base model.

## Running on a cluster

[cell_pipeline.py](cell_pipeline.py) freezes a run's plan, runs the workers that compute whole cells and append their rows and predictions to their own files, starts each later attempt with only the unfinished cells, and publishes `final.csv` once every cell is present exactly once. [execution_contract.py](execution_contract.py) records the inputs, methods and design of a run, so a worker computes only under the frozen settings.

See the [experiment methods](../../README.md) and the [root README](../../../README.md).
