# FFCWS preprocessing and missing values

FFCWS inputs comprise a background-variable table and official train/test outcome tables. The code joins them by `challengeID`, preserving the official split and the row order of each outcome table. Only the official training pool determines variable retention and encoding; the same rules are then applied to test data.

The sequence is: **identify missing values → screen variables in the training pool → determine variable types and category vocabularies → generate three representations → join the selected outcome → validate and pass inputs to the training engine.**

## 1. Row and column handling

Families are not removed based on the number of missing predictors. Missing predictors are handled during training. Data are aligned by unique `challengeID`.

Each outcome uses families with an observed value for that outcome. For example, a family with observed GPA but missing grit can still be used in the GPA experiment. The engine checks the selected outcome separately in the training and test tables: missingness above 50% raises an error; otherwise, rows missing that outcome are removed. Outcomes are not imputed.

Column screening uses only the official training pool, with rules from the [configuration file](config/ffc.yaml):

| Check | Current rule |
|---|---|
| Missing-value recognition | Blanks, NA-like markers, and codes −9 through −1 are treated as missing; values that cannot be parsed as numbers also become NaN |
| Valid-value rate | After excluding recognized blanks and negative missing codes, discard the source if fewer than 50% of its original values are valid; exactly 50% passes |
| Variation | If conversion leaves no observed numeric values or only one distinct value, discard the entire source and its missingness indicators |
| Categorical-source retention | At least one observed level must satisfy `min(p,1-p) >= 0.01`, where p is its proportion of all training-pool rows |

Here, structural missingness includes blanks, NA markers, and codes -9 through -1. Numeric parsing failures become NaN, but parsing success rates do not determine column retention. The 50% threshold is based on structural missingness. The negative-code rule for predictors is not automatically applied to outcomes.

If a prepared column is entirely NaN, the engine reports an input problem.

## 2. Variable encoding

Numbers can represent quantities or category identifiers. For example, 1, 2, and 3 children are counts, whereas marital-status codes 1, 2, and 3 identify categories.

The code determines types in this order:

| Available information | Treatment |
|---|---|
| Explicit type in the configuration | `numeric` or `categorical` in `schema.variable_types` takes precedence over the 15-level heuristic |
| No declared type | Treat a variable as categorical if it has at most 15 observed levels and either Stata value labels or exclusively integer values; otherwise, treat it as numeric |

Type declarations should be checked against variable documentation. The default type map is currently empty, so the second rule still applies. The code does not automatically interpret variable descriptions. Category vocabularies are built only from the official training pool.

The same retained sources are represented in three ways:

| Representation | Numeric variables | Categorical variables | Additional missingness information |
|---|---|---|---|
| `median_mode` | Values and NaN | Full one-hot encoding; missing values set the entire group to NaN | None |
| `median_missing_indicator` | Values and NaN | Same as above | Retained 0/1 indicators for declared missing codes or blanks |
| `tree_ordinal` | Values and NaN | Categories sorted by original numeric code and mapped to 0, 1, 2, … | None |

Once a categorical source is retained, its one-hot representation keeps all known observed levels, including rare levels. Missingness indicators remain subject to the 1% binary-prevalence threshold.

In the missing-indicator representation, codes −9 through −1 become NaN in the value column. Separate 0/1 indicators are generated for specific codes observed in the training pool, preserving their different meanings as defined by the variable's labels. The value column is imputed during training; linear models and neural networks can also use these indicators.

For example, suppose a variable labels −1 as refusal and −2 as “don't know,” and both indicators pass screening:

| Original value | Value column before imputation | Refusal indicator | Don't-know indicator |
|---|---:|---:|---:|
| 10 | 10 | 0 | 0 |
| −1 | NaN | 1 | 0 |
| −2 | NaN | 0 | 1 |

These columns let models learn from reasons for missingness without treating −1 and −2 as quantities. Only `median_missing_indicator` adds these columns.

All three representations retain the same original sampling sources, although expanded column counts can differ. Value columns and missingness indicators are sampled together, and K counts the original variable once. See the [three encoding methods](src/ffcws_data_processor/strategies/README.md) for examples.

## 3. When and how imputation occurs

**The adapter identifies missing values and preserves NaN; imputation takes place during training.** The engine learns imputation rules after drawing the training sample. Each cross-validation training fold learns its own rules, which are then applied to its validation or test data.

| Type | Training-time treatment |
|---|---|
| Continuous variable | Impute the median of the current training rows |
| One-hot category group | Impute the most frequent complete category state in the training rows, rather than imputing columns independently |
| Integer category encoding (`ordinal`) | Find the training median, then select the nearest legal level observed in that training data; ties select the lower value |
| Standalone LightGBM/XGBoost | Preserve NaN as specified by the schema and let the model handle it |

For example, categories represented by `[1,0,0]`, `[0,1,0]`, and `[0,0,1]` have a missing state of `[NaN,NaN,NaN]`. Imputation selects a complete category state rather than filling the columns with `[0,0,0]`.

A column observed in the full training pool may be entirely missing in a small training sample. The code retains the column and its contribution to K, setting the corresponding preprocessing group to the same prior value on both training and validation/test sides. Models preserving NaN receive NaN on both sides. This prevents test observations from restoring information absent during training. A combination is skipped if the primary value representations of all selected sources are unobserved.

## 4. New categories in test data

For categories outside the training vocabulary, the code records counts and rates and converts them to missing values without extending the vocabulary. By default, an unknown rate strictly above 95% raises an error. The denominator is the count of parseable, nonmissing numeric values for that variable in the test data.

Coverage is checked again for each outcome. A category may occur in the official training pool only in rows missing the selected outcome. The code therefore checks coverage using training rows with an observed outcome. Uncovered categories in test rows with an observed outcome become NaN, subject to the same rate threshold. This check uses outcome availability, not outcome values, to determine category coverage.

## 5. Code locations

[pipeline.py](src/ffcws_data_processor/pipeline.py) coordinates the process. [common/schema.py](src/ffcws_data_processor/common/schema.py) identifies missingness, screens variables, and defines their representations. [common/io.py](src/ffcws_data_processor/common/io.py) joins official outcome tables. [contract.py](src/ffcws_data_processor/contract.py) handles category coverage for rows with observed outcomes and generates engine input definitions.

A new data version is published after all selected input checks pass. Reports record screening reasons and unknown categories, showing which variables were discarded and which test values became missing. See the [root quick start](../../README.md) for execution instructions.

Subsequent FFC panels use these rules when inputs are regenerated with the updated code. Existing ARD and schema files retain the preprocessing used when they were created. Select a `median_missing_indicator` panel to include missingness indicators.
