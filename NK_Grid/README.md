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

If a variable is entirely missing in a small training fold, both sides are set to a prior value or NaN. Test data do not supply information absent from training. A combination is skipped if the primary value representations of all selected sources are unobserved; classification combinations with only one training class are also skipped.

## Model fitting

Supported models include OLS, Ridge, Lasso, Random Forest, Extra Trees, XGBoost, LightGBM, Shallow NN, and Super Learner.

Regression Ridge and Lasso select regularization strength within their training folds and then refit. Ridge uses up to five folds and 63 alpha values, averaging fold MSEs with equal weights. Lasso uses up to five folds and 50 alpha values. Regression boosting models select training iterations through cross-validation. Standalone XGBoost and LightGBM preserve NaN as specified by the data definition.

The standalone regression MLP uses 32 hidden units and up to three folds to compare five alpha values. Each fit divides its actual training rows into batches of approximately equal size, capped at 200 rows. For example, 201 rows become 101+100, and 401 become 134+134+133, avoiding single-row final batches. Training uses at most 2,000 epochs with early stopping enabled.

Regression Super Learner generates out-of-fold predictions from Ridge, Extra Trees, LightGBM, and MLP, then fits a nonnegative linear combination of those predictions. Each base model is refitted on all N rows for test prediction. Its MLP uses a fixed alpha of 0.01 rather than the standalone MLP's alpha search. Preprocessing inside the ensemble is not necessarily identical to that of standalone models.

For classification, the OLS, Ridge, and Lasso names refer to logistic regressions with different regularization settings. Classification MLP and Super Learner use their own parameters rather than the regression balanced-batch configuration.

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

Each result also has a status: `ok` for a completed fit, `skipped` for insufficient data conditions, and `failed` for a failure. Separate diagnostics cover underdetermined OLS, constant predictions, and convergence. Experiment completeness is assessed against coverage of the full design.

See the [core code description](src/aleatoric_nk_grid/README.md) for implementation flow and the [root README](../README.md) for execution instructions.
