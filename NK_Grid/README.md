# How N × K experiments work

The engine measures how prediction quality changes as more samples and variables become available for learning. Each training combination is defined by model, seed, draw, N, and K. All combinations follow the same preprocessing and evaluation rules.

## Split first, then sample

FFC uses the official train/test split. For SMR, the engine removes rows missing the selected outcome, then performs a seed-specific random split with 30% held out for testing by default. Experiments within a seed share the same test set as N changes.

Each seed can have multiple draws. A draw generates one random ordering of training rows and one of sources. The first N rows and K sources form the training input. Smaller samples are therefore nested within larger samples, and smaller source sets within larger ones, reducing variation caused by comparing unrelated subsets.

K counts original sampling sources. One-hot columns and their bound missingness indicators are selected together. The expanded column count is stored as `K_expanded`. Different encodings can have different dimensions at the same K.

By default, the N/K axes use integer points spaced on a logarithmic scale. Duplicate integers are adjusted to distinct points; an insufficient range raises an error. N cannot exceed the available training rows, and K cannot exceed the retained source count.

## Learn preprocessing from the current training data

Adapters preserve NaN. After sampling training rows, the engine learns type-specific imputation: medians for continuous columns, the most frequent complete category state for one-hot groups, and the dataset-declared mode or nearest observed level to the median for integer-encoded category columns (`ordinal`). Models requiring standardization also learn location and scale from training rows only.

Each cross-validation training fold learns these parameters independently and applies them to its validation fold. After parameter selection, preprocessing and the model are refitted on all N training rows, followed by test-set evaluation. Target standardization for regression MLP follows the same sequence.

If a variable is entirely missing in a small training fold, both sides are set to a prior value or NaN, so test data cannot supply information the training rows lack. A combination is skipped when none of the selected variables has an observed value in the training sample, or when a binary outcome has only one class there.

## Model fitting

Eight models are fit separately: OLS, Ridge, Lasso, Random Forest, Extra Trees, XGBoost, LightGBM and a shallow neural network. A ninth, the Super Learner, combines seven of them.

For continuous outcomes, Ridge, Lasso and the neural network choose their penalty by cross-validation on the training rows, and XGBoost and LightGBM choose their number of boosting rounds the same way. Each model is then refit on all N rows. The candidate values and fold counts are in each study's `model_params.yaml`. XGBoost and LightGBM receive missing values as NaN and handle them themselves.

The regression neural network has one hidden layer of 32 units. Each fit splits its training rows into mini-batches of at most 200 rows and of nearly equal size, so the last batch is never much smaller than the others. Training runs for at most 2,000 epochs and stops earlier once the training loss stops improving.

The Super Learner (SL7) combines the seven models other than OLS. Each of the seven is fit five more times, on about four fifths of the training rows each time, to predict the remaining rows; this gives an out-of-fold prediction for every training row. SL7 finds the nonnegative weights, plus an intercept, that best predict the training outcome from these predictions, and applies them to the seven models' test predictions. For binary outcomes it combines predicted probabilities with a logistic regression. It uses no test outcomes and retrains nothing.

For binary outcomes, the OLS, Ridge and Lasso names refer to logistic regressions with different penalties, and the base models use fixed settings instead of cross-validated tuning.

## Interpreting prediction quality

MSE is the mean squared prediction error. The project also saves R² measures with two baselines:

```text
r2_test / skill_train_mean
  = 1 - model MSE / test MSE of predictions using the current training-sample mean

r2_test_mean
  = 1 - model MSE / variance of the test outcomes
```

The baseline for `r2_test` changes with the training sample. The conventional test-set R² corresponds to `r2_test_mean`. MSE can be viewed alongside these measures.

Classification metrics include AUC, Brier score, log loss, and accuracy. Undefined metrics under particular data conditions are stored as NaN, which is distinct from a failed model fit.

Each result also has a status: `ok` for a completed fit, `skipped` when the sampled data do not allow a fit, and `failed` when fitting failed. Diagnostics also flag OLS fits with more columns than rows, constant predictions and convergence problems. An experiment is complete only when every combination in the design has a result.

See the [core code description](src/aleatoric_nk_grid/README.md) for implementation flow and the [root README](../README.md) for execution instructions.
