# Aleatoric Luck

This repository runs a repeated-sampling prediction experiment. For each outcome we ask how well it can be predicted from a training sample of N people using K of the available predictor variables, and how that changes as N and K grow. The same design is applied to two studies: the Fragile Families Challenge (`FFCWS/`) and a study of social rigidity (`SMR/`).

## Research questions

### Fragile Families Challenge

How accurately can outcomes at age 15 be predicted from information observed from birth through age 9? The application also checks whether the conclusions depend on how categorical values and missing information are represented. The training and test samples are the ones defined by the Challenge; the [special-collection introduction](https://pmc.ncbi.nlm.nih.gov/articles/PMC10260255/) describes the study design.

### Social rigidity

How accurately can midlife socioeconomic outcomes be predicted from earlier-life and family-background information? The application follows the predictive approach of Zheng and Cheng, ["Social Rigidity Across and Within Generations: A Predictive Approach"](https://doi.org/10.1177/00491241251347984). It describes how persistent and predictable these outcomes are; the causal effects of individual predictors are outside its scope.

### Outcomes

| Study | Outcome | Column | Type |
|---|---|---|---|
| FFCWS | Grade point average | `gpa` | Continuous |
| FFCWS | Grit | `grit` | Continuous |
| FFCWS | Household material hardship | `materialHardship` | Continuous |
| FFCWS | Household eviction | `eviction` | Binary |
| FFCWS | Caregiver layoff | `layoff` | Binary |
| FFCWS | Caregiver job training | `jobTraining` | Binary |
| SMR | Log hourly wage | `Cm_lhourlywage` | Continuous |
| SMR | Log total personal income | `Cm_ltotalincome` | Continuous |

## Data

FFCWS predictors come from the background survey collected before the age-15 outcomes. Which variables are kept, and how their categories are coded, is decided from the Challenge's training sample only; the test sample is used only to score predictions. Each FFCWS outcome is run under three encodings of the same variables, which differ in how categories and missing values are represented ([the three encodings](FFCWS/adapter/src/ffcws_data_processor/strategies/README.md)).

SMR uses the numeric analysis matrix of the NLSY-based replication: 4,252 columns that represent 497 predictor variables. There is no predefined test sample, so for each seed the engine sets aside a random 30% of the people with an observed outcome as the test set.

In both studies K counts variables, not columns. The dummy columns of a categorical variable, together with any missingness indicators attached to it, enter and leave the model as one unit and count once.

Missing values stay missing in the prepared data. Imputation, scaling and model tuning are learned from the sampled training rows only, and inside cross-validation from each training fold only, so a small training sample does not borrow information from the rest of the data or from the test set ([why](Adapter/README.md#why-imputation-happens-during-training)).

## Experimental design

### The N × K grid

Each axis has 20 sizes spaced evenly on a log scale. N runs from 10 up to the whole training sample, and K from 1 up to all predictor variables.

### Nested samples

For each seed and draw, the training rows are put in one random order and the predictor variables in another. The training data for a grid point are the first N rows and the first K variables in those orders. Within a draw, a smaller sample is therefore always part of a larger one, and moving along the grid adds data instead of switching to an unrelated subset.

### Repetition

The production design repeats the grid for 100 seeds with 50 draws each. In SMR, each seed also makes a new train/test split; in FFCWS the Challenge's test sample is the same in every seed. With 400 grid points this gives 2,000,000 training samples per model and 18,000,000 result rows per panel across the nine models. The spread across seeds and draws shows how much prediction quality at a given N and K depends on which people and variables happen to be drawn.

## Models

Eight models are fit separately on every training sample: OLS, Ridge, Lasso, random forest, extra trees, XGBoost, LightGBM and a neural network with one hidden layer. For binary outcomes, OLS, Ridge and Lasso are logistic regressions with different penalties.

For continuous outcomes, Ridge, Lasso and the neural network choose their penalty, and the two boosting models their number of rounds, by cross-validation within the training sample, and are then refit on all N rows. For binary outcomes these models use fixed settings. Each study's settings are in its `model_params.yaml`.

The ninth model, the Super Learner (SL7), combines the seven models other than OLS. Besides its fit on all N rows, each of the seven is fit five more times, each time on about four fifths of the training sample, to predict the remaining rows. This gives an out-of-fold prediction for every training row. SL7 finds the nonnegative weights, plus an intercept, that best predict the training outcome from these out-of-fold predictions, and applies the same weights to the seven models' test predictions. For binary outcomes it combines predicted probabilities with a logistic regression. SL7 does not see test outcomes and does not retrain any model.

## Reading the results

Each row of `final.csv` is one model fitted at one seed, draw, N and K. `status` is `ok` for a completed fit, `skipped` when the sampled data do not allow a fit (for example, a binary outcome with only one class in the training sample), and `failed` when fitting failed. `K_expanded` is the number of model columns behind the K variables.

For continuous outcomes, `mse` is the mean squared error on the test set. `r2_test` compares it with predicting the training-sample mean for everyone; `r2_test_mean` compares it with the test-sample mean, which is the usual test R². For binary outcomes the table has `roc_auc`, `brier`, `log_loss` and `accuracy`. A metric that cannot be computed for a particular sample is stored as NaN, which is different from a failed fit.

## Quick start

Each experiment is one panel: one outcome under one encoding. `FFCWS/panels.yaml` lists the 18 FFCWS panels and `SMR/panels.yaml` the two SMR panels. A run goes through a dry-run that only prints the launch settings, a small `dev` run or a `timing_full` run that covers the full grid once, and then `production`.

| Preset | Seeds × draws | Grid |
|---|---|---|
| dev | 3 × 3 | 3 × 3, N and K at most 100 |
| timing_full | 1 × 1 | full 20 × 20 |
| production | 100 × 50 | full 20 × 20 |

A dry-run of the FFCWS GPA panel on Discoverer:

```bash
bash run.sh slurm --profile discoverer --account YOUR_PROJECT_ACCOUNT \
  --panel ffc_median_mode_gpa --preset timing_full \
  --scheduler-policy launch/policies/discoverer-cache.json \
  --dispatcher-shards 4 --dry-run
```

[How runs are carried out](launch/README.md) explains the stages and what happens on the cluster. [launch/OPERATIONS.md](launch/OPERATIONS.md) lists all commands and options, including other clusters, data preparation, resuming and output locations.

## Where to read more

| Topic | Document |
|---|---|
| What adapters do, and why imputation waits for the training sample | [Adapter principles](Adapter/README.md) |
| Writing an adapter for a new dataset | [Adapter specification](Adapter/ADAPTER.md) |
| FFCWS variable screening and missing values | [FFCWS adapter](FFCWS/adapter/README.md) |
| The three FFCWS encodings | [Encodings](FFCWS/adapter/src/ffcws_data_processor/strategies/README.md) |
| SMR variable definitions | [SMR adapter](SMR/adapter/README.md) |
| Sampling, preprocessing, models and metrics in detail | [Experiment methods](NK_Grid/README.md) |
| How the code carries out one experiment | [Core code](NK_Grid/src/aleatoric_nk_grid/README.md) |
| How runs are carried out | [Running experiments](launch/README.md) |
| Commands, clusters and recovery | [Operations](launch/OPERATIONS.md) |
