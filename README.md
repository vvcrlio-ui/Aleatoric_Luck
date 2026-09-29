# Aleatoric Luck

This repository runs a repeated-sampling prediction experiment. For an outcome in a dataset, we ask how well it can be predicted from a training sample of N observations using K of the available predictor variables, and how that changes as N and K grow. Each dataset is prepared by its own adapter; the engine then applies the same design to every dataset.

## Data

An adapter turns a dataset into an analysis table, a description of its variables, and a schema that ties them together ([adapter principles](Adapter/README.md)). The test set is either the one defined by the data provider or, when there is none, a random 30% of the observations with an observed outcome, drawn anew for each seed. Variables are screened and categories coded without using the test set.

K counts variables, not columns. The dummy columns of a categorical variable, together with any missingness indicators attached to it, enter and leave the model as one unit and count once. The adapter files call this unit a source.

Missing values stay missing in the prepared data. Imputation, scaling and model tuning are learned from the sampled training rows only, and inside cross-validation from each training fold only, so a small training sample does not borrow information from the rest of the data or from the test set ([why](Adapter/README.md#why-imputation-happens-during-training)).

## Experimental design

### The N × K grid

Each axis has 20 sizes spaced evenly on a log scale. N runs from 10 up to the whole training sample, and K from 1 up to all predictor variables.

### Nested samples

For each seed and draw, the training rows are put in one random order and the predictor variables in another. The training data for a grid point are the first N rows and the first K variables in those orders. Within a draw, a smaller sample is therefore always part of a larger one, and moving along the grid adds data instead of switching to an unrelated subset.

### Repetition

The production design repeats the grid for 100 seeds with 50 draws each. When the engine makes the train/test split, each seed also draws a new split; a test set defined by the data provider stays the same in every seed. With 400 grid points this gives 2,000,000 training samples per model, and 18,000,000 result rows for a panel with nine models. The spread across seeds and draws shows how much prediction quality at a given N and K depends on which observations and variables happen to be drawn.

## Models

Eight models are fit separately on every training sample: OLS, Ridge, Lasso, random forest, extra trees, XGBoost, LightGBM and a neural network with one hidden layer. For binary outcomes, OLS, Ridge and Lasso are logistic regressions with different penalties. For continuous outcomes, Ridge, Lasso and the neural network choose their penalty, and the two boosting models their number of rounds, by cross-validation within the training sample; for binary outcomes these models use fixed settings.

The ninth model, the Super Learner (SL7), is a weighted combination of the seven models other than OLS. Its weights are estimated from predictions each model makes for training rows it was not fit on, so it never sees test outcomes. The [experiment methods](NK_Grid/README.md#model-fitting) describe the tuning and the Super Learner in full.

## Reading the results

Every run writes to a new directory in its dataset folder, `<dataset>/outputs/<panel>-<ID>/`, which Git ignores. Each row of `final.csv` there is one model fitted at one seed, draw, N and K. `status` is `ok` for a completed fit, `skipped` when the sampled data do not allow a fit (for example, a binary outcome with only one class in the training sample), and `failed` when fitting failed. `K_expanded` is the number of model columns behind the K variables.

For continuous outcomes, `mse` is the mean squared error on the test set. `r2_test` compares it with predicting the training-sample mean for everyone; `r2_test_mean` compares it with the test-sample mean, which is the usual test R². For binary outcomes the table has `roc_auc`, `brier`, `log_loss` and `accuracy`. A metric that cannot be computed for a particular sample is stored as NaN, which is different from a failed fit.

## Running experiments

The [launch guide](launch/README.md) starts with a command that only previews a run, then a small real run, and then the steps up to a full production run. It also explains how to follow a run on the cluster and how to resume one that stopped.

## Where to read more

| Topic | Document |
|---|---|
| What adapters do, and why imputation waits for the training sample | [Adapter principles](Adapter/README.md) |
| Writing an adapter for a new dataset | [Adapter specification](Adapter/ADAPTER.md) |
| How a particular dataset is prepared | `adapter/README.md` in that dataset's folder |
| Sampling, preprocessing, models and metrics in detail | [Experiment methods](NK_Grid/README.md) |
| How the code carries out one experiment | [Core code](NK_Grid/src/aleatoric_nk_grid/README.md) |
| Launching, following and resuming a run | [Running experiments](launch/README.md) |
