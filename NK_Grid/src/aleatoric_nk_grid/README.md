# How the core code runs an experiment

`NKGridExecutionSession` executes the core computation. Local and cluster runs share the same sampling, preprocessing, and model-training steps.

## From inputs to training combinations

[ingest.py](ingest.py) reads the analysis table and feature definitions from the schema. [validate_input.py](validate_input.py) checks numeric values, category states, IDs, and missingness rates, then returns eligible data for the selected outcome. These checks precede sampling so input inconsistencies are identified before model fitting.

[run_panels.py](run_panels.py) selects the outcome, models, and repetition counts; [config.py](config.py) stores these choices. [nk_grid.py](nk_grid.py) determines the available N/K range, splits by seed, permutes by draw, and selects the first N rows and K sources.

Sources and columns are distinct. [preprocessing.py](preprocessing.py) groups columns that share preprocessing, then binds groups belonging to the same sampling source. An original variable's one-hot columns and missingness indicators can use different imputation methods but enter or leave the training input together.

## Missing-value handling in model pipelines

`FoldPreprocessor` implements type-specific imputation as a refittable step. It learns from the supplied training rows and applies the result to validation or test rows. Groups unobserved during training are set to a fixed prior value or NaN on both sides.

The outer preprocessed matrix is used for diagnostics. Internal cross-validation starts with the original NaN values and refits imputation and standardization within each training fold.

[model_registry.py](model_registry.py) selects an implementation by model and task type. [fold_local.py](fold_local.py) handles fold-specific preprocessing, parameter selection, and full refitting for Ridge, Lasso, and MLP. Models without internal parameter selection fit their pipeline once on the current N rows.

## MLP and ensemble models

[mlp_estimator.py](mlp_estimator.py) creates balanced batches from the actual row count in each fit, retaining native Adam, L2 regularization, and stopping behavior. Batch sizes are based on the rows in each fit rather than the outer N, because cross-validation training folds contain fewer rows.

[mlp_batch_cv.py](mlp_batch_cv.py) also retains explicit batch-size search and ensemble diagnostics. Current production regression settings use the fixed balanced-batch rule rather than searching batch sizes by default.

Super Learner fits combination weights using out-of-fold predictions from its base models, then refits those models on all training rows. Optional diagnostics save the out-of-fold predictions and weights from that fit.

Some models backed by native libraries run in isolated processes through [native_process.py](native_process.py), with bounded retries after crashes or timeouts. This execution mechanism preserves the declared training combinations.

## From predictions to results

After prediction, `nk_grid.py` computes metrics. [evaluation.py](evaluation.py) defines regression errors and their denominators. Training-mean and test-mean baselines are stored separately to distinguish the two R² measures.

[experiment.py](experiment.py) manages local checkpoints. The cluster path uses [flat_task_table.py](flat_task_table.py) to drive the same computation and [worker_event_wal.py](worker_event_wal.py) to save committed results. Only fully written, validated records count toward progress; training without saved results may be repeated after interruption.

[execution_contract.py](execution_contract.py) records input, method, and design identities so results from different runs are not combined solely because their columns match. [generation_control.py](generation_control.py) coordinates the start and sealing of each task generation. Final merging checks coverage of expected combinations and conflicting duplicates before publication. [checkpoint_retention.py](checkpoint_retention.py) then applies the selected checkpoint policy.

See the [experiment description](../../README.md) for methods and the [root tutorial](../../../README.md) for execution steps.
