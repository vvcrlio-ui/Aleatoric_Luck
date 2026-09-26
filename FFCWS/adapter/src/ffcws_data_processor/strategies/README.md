# The three FFC encodings

The three methods use the same training-pool screening result and change only the representation of retained variables. The same K represents the same number of original sources, but the expanded column count can differ. Shared row and column rules are described in the [adapter README](../../../README.md).

## median_mode: preserve numeric values and expand categories

Numeric variables retain their values and NaN. Categorical variables use full one-hot encoding based on the training vocabulary. For example, categories 1, 2, and 3 become `[1,0,0]`, `[0,1,0]`, and `[0,0,1]`.

Missing or unknown categories set the entire group to NaN. All known levels, including the reference category, are retained. During training, numeric columns use median imputation and category groups use the most frequent complete category state.

Implemented in [median_mode.py](median_mode.py).

## median_missing_indicator: retain reasons for missingness

Value encoding is the same as median_mode, with additional 0/1 indicators for declared missing codes or blank values observed in the training pool. For example, an original value of -9 produces NaN in the value column and 1 in the corresponding -9 indicator. Indicators must pass the training-pool prevalence screen; a new missing code in the test data does not add a column.

Value columns and indicators share the same original source. Three one-hot columns plus two missingness indicators produce five model columns, but K remains one, and all five are sampled together. Preprocessing treats them separately: the category group is imputed jointly, while indicators are treated as individual numeric columns.

Missingness indicators provide information about why a value is absent; they do not replace imputation of the value column. See [median_missing_indicator.py](median_missing_indicator.py).

## tree_ordinal: one column per categorical source

Categories in the training vocabulary are sorted by their original numeric codes and mapped to integers from 0 to L-1. Missing and unknown categories remain NaN. This reduces the expanded column count but introduces a numeric order that may not be intrinsic to an unordered category.

Current panels also run linear models and neural networks on this representation. Where imputation is required, it uses the observed legal code nearest the training median. Standalone LightGBM/XGBoost retain NaN. See [tree_ordinal.py](tree_ordinal.py).

## A simple comparison

Suppose source A has three categories and source B is continuous, and both are selected:

| Encoding | K | Expanded columns |
|---|---:|---:|
| median_mode | 2 | 3+1=4 |
| median_missing_indicator, with two retained missingness indicators | 2 | 3+1+2=6 |
| tree_ordinal | 2 | 1+1=2 |

None of the encoding functions learns imputation parameters. After encoding, the pipeline checks test-category coverage against training rows with an observed value for the selected outcome. The final test table may therefore contain additional NaN values.
