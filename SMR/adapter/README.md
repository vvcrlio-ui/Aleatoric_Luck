# SMR data preparation

SMR uses the provider's existing `asample2_withlag.csv` analysis matrix. The adapter selects predictors and outcomes using fixed definitions, describes category groups, and passes them to the shared engine. It does not reconstruct wages, income, lagged variables, or existing missingness indicators.

## Fixed variable definitions

The [feature definition file](config/asample2_withlag.json) lists predictor columns and their order, the two outcomes, and one-hot groups. The current definition contains 4,252 predictor columns and 29 one-hot groups, representing 497 sampling sources. Predictors outside one-hot groups are treated as individual continuous columns.

Loading checks column names, order, and category groups against this fixed list. Variable definitions remain fixed; the engine performs the train/test split.

## Missing values

The current `missing_value_codes` mapping is empty, so negative values are not automatically treated as missing. Existing CSV missing values remain NaN. Special codes are replaced only when explicitly declared for a column.

The adapter retains rows with missing predictors and does not impute outcomes. The outcomes are `Cm_lhourlywage` and `Cm_ltotalincome`. For each outcome, the engine first checks its missingness rate: a rate above 50% raises an error; otherwise, rows missing that outcome are removed before the seed-specific train/test split. The two outcomes can therefore have different eligible samples.

Entirely missing predictor columns require an input check. Partial missingness is handled within each training sample and cross-validation training fold: continuous columns use median imputation, and one-hot groups use the most frequent complete category state. Standalone LightGBM/XGBoost preserve NaN.

If a group is entirely missing in a small training fold, both training and validation/test values are set to the same prior value or NaN. Test observations do not restore a variable unseen during training. The source still counts toward K.

## Why categorical columns are grouped

One-hot groups retain all declared levels. An observed row must contain exactly one 1; a missing row may contain NaN across the entire group. All-zero and partially missing groups are not valid category states.

All dummy columns for an original variable are sampled together and count once toward K, including the reference-level column. Other predictors are treated as continuous variables.
