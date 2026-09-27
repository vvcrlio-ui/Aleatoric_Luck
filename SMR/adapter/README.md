# SMR data preparation

SMR uses the provider's existing analysis matrix, `asample2_withlag.csv`. The adapter selects predictors and outcomes using fixed definitions, describes category groups, and passes them to the shared engine. It does not reconstruct wages, income, lagged variables, or existing missingness indicators.

## Fixed variable definitions

The [feature definition file](config/asample2_withlag.json) lists predictor columns and their order, the two outcomes, and one-hot groups. The current definition contains 4,252 predictor columns and 29 one-hot groups, representing 497 sampling sources. Predictors outside one-hot groups are treated as individual continuous columns.

Loading checks column names, order, and category groups against this fixed list. Variable definitions remain fixed; the engine performs the train/test split.

## Row numbers

The matrix has no person identifier, so the adapter numbers the records in file order, starting at 1, before any row is filtered. The number, `smr_row_id`, is not a predictor and plays no part in sampling or in the train/test split. The engine records it for each training and test sample, so that predictions saved by separately trained models can be matched to the same person when the Super Learner combines them.

## Missing values

No missing-value codes are declared, so negative values are kept as values. Missing cells in the CSV stay NaN; a special code would be replaced only if it were declared for its column.

The adapter keeps rows with missing predictors and does not impute outcomes. The outcomes are `Cm_lhourlywage` and `Cm_ltotalincome`. For each outcome, the engine first checks its missingness rate: a rate above 50% raises an error; otherwise, rows missing that outcome are removed before the seed-specific train/test split. The two outcomes can therefore have different eligible samples.

A predictor column that is entirely missing is rejected by the input check. Partial missingness is handled within each training sample and cross-validation training fold: continuous columns use median imputation, and one-hot groups use the most frequent complete category state. Standalone LightGBM and XGBoost receive NaN and handle it themselves.

If a group is entirely missing in a small training fold, both training and validation/test values are set to the same prior value or NaN, so the test data cannot bring back a variable the training rows never saw. The source still counts toward K.

## Why categorical columns are grouped

One-hot groups retain all declared levels. An observed row must contain exactly one 1; a missing row may contain NaN across the entire group. All-zero and partially missing groups are not valid category states.

All dummy columns for an original variable are sampled together and count once toward K, including the reference-level column. Other predictors are treated as continuous variables.

## Lasso penalty search

For SMR, Lasso chooses its penalty from values set relative to alpha_max, the smallest penalty at which every coefficient is zero, computed separately in each training fold. It uses three folds and 25 values from alpha_max down to 0.001 × alpha_max, spaced evenly on a log scale. Each fold learns its own imputation and scaling. The value with the lowest mean validation MSE is chosen, and ties go to the stronger penalty. The final fit on all N training rows recomputes alpha_max from those rows. The convergence tolerance is 1e-4 and the iteration limit 20,000. The range is the same for every N and K, and no test outcome is used.
