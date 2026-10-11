# Saved predictions and the Super Learner

Every cell of a run saves each base model's predictions for the test rows and its out-of-fold (OOF) predictions for the training rows. The Super Learner (SL7) of the cell is combined from these predictions in memory, without fitting any base model again, and the predictions stay in the run for later analysis.

## Which models the Super Learner combines

The Super Learner combines the seven base models other than OLS (variant `standalone8-sl7-v1`). OLS keeps its own saved predictions and its own score, but it is left out of the combination: with more columns than rows it can predict far outside the range of the outcome.

Each of the seven models is fitted on all N training rows and once in each of five OOF folds. The fold fits use the penalty or number of boosting rounds chosen by the fit on all N rows; see [DECISIONS.md](../DECISIONS.md).

For continuous outcomes, the combiner is a nonnegative least-squares fit with an intercept on centered predictions; the weights are not rescaled to sum to one. For binary outcomes, it is an unpenalized logistic regression on the positive-class probabilities, with nonnegative weights and an intercept. Test outcomes are used only to score the combined prediction, after the combiner has been fitted.

If one of the seven models is skipped in a cell, for example because a binary outcome has too few rows of one class for its folds, the cell's SL7 row is `skipped` and its `error` names the first missing model.

`final.csv` has `phase`, `pipeline_id` and `variant_id` columns that tell base-model rows (`phase=base`, `pipeline_id=standalone8-v1/MODEL`) from SL7 rows (`phase=sl`, `variant_id=standalone8-sl7-v1`).

## What is saved for each cell

| Name | Contents |
|---|---|
| `train_ids` | IDs of the cell's N training rows, in the order of every OOF prediction |
| `oof_fold` | The OOF fold of each training row, when the cell has OOF predictions |
| `MODEL/holdout` | The model's predictions for the test rows |
| `MODEL/oof` | The model's OOF predictions for the training rows, when it has them |
| `super_learner/holdout` | SL7's predictions for the test rows |
| `super_learner/coefficients`, `super_learner/intercept` | SL7's weights, in the order ridge, lasso, random forest, extra trees, XGBoost, LightGBM, neural network |

Test predictions follow the seed's test rows, whose IDs are in `test-ids.npz` under the seed's number. Predictions are float64; binary outcomes store positive-class probabilities. A model that was skipped in a cell has no arrays there.

## How predictions are stored

Each worker writes two files in its allocation's `attempt-JOBID/` directory and only appends to them: `predictions-rNNNNN.bin` holds the raw bytes of the arrays, and `results-rNNNNN.jsonl` holds one line per cell with its result rows and, for every array, its byte offset, data type and shape in the `.bin` file. A cell's arrays are synced to disk before its line is written, so every complete line refers to complete arrays. A line cut off at the end of a file by an interrupted worker is ignored, and the cell is computed again by a resumed run.

When the run is published, `predictions.jsonl` in the run directory lists every cell once, in the order of `final.csv`, with the path of its `.bin` file and the same array positions.

## Reading predictions

```python
import json
import numpy as np
from aleatoric_nk_grid.cell_pipeline import read_arrays

run = "RUN_DIR"
index = {tuple(entry["cell"]): entry for entry in map(json.loads, open(f"{run}/predictions.jsonl"))}
arrays = read_arrays(run, index[(SEED, DRAW, N, K)])
test_ids = np.load(f"{run}/test-ids.npz")[str(SEED)]
```

`arrays` maps the names in the table above to NumPy arrays.

## Combining a different subset

A different combination can be fitted from the saved predictions without any base training, for example a four-model Super Learner:

```python
from aleatoric_nk_grid.offline_sl import recombine_predictions

models = ["ridge", "extra_trees", "lightgbm", "shallow_neural_network"]
answer = recombine_predictions(
    oof_predictions=np.column_stack([arrays[f"{m}/oof"] for m in models]),
    holdout_predictions=np.column_stack([arrays[f"{m}/holdout"] for m in models]),
    y_train=TRAINING_OUTCOMES_IN_TRAIN_ID_ORDER, model_names=models, selected_models=models,
    variant_id="sensitivity-sl4", task="regression")
```

Look up the training outcomes by `train_ids` in the prepared analysis table. Give each combination its own variant ID, and never pool results from different variants.
