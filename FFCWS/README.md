# Fragile Families Challenge application

## Research question

How accurately can outcomes at age 15 be predicted from information observed
from birth through age 9? How does predictive performance change with the
number of observed training families (\(N\)) and the number of available
predictor variables (\(K\))?

The application also evaluates whether conclusions are sensitive to the
representation of categorical values and missing information.

| Outcome | Code in the analysis data | Type |
|---|---|---|
| Grade point average | `gpa` | Continuous |
| Grit | `grit` | Continuous |
| Household eviction | `eviction` | Binary |
| Household material hardship | `materialHardship` | Continuous |
| Caregiver layoff | `layoff` | Binary |
| Caregiver job training | `jobTraining` | Binary |

These outcomes and the predefined training and test samples follow the Fragile
Families Challenge. See the
[special-collection introduction](https://pmc.ncbi.nlm.nih.gov/articles/PMC10260255/)
for the study design.

## Data and research design

Eligible predictors come from the background survey data collected before the
age-15 outcomes. Predictor eligibility, categorical value sets, and
prevalence-based screening are determined using the predefined training sample
only. The test sample is reserved for evaluating predictive performance.

The analysis compares three representations of the same source information:

| Representation | Configuration ID | Treatment of categorical values and missing information |
|---|---|---|
| One-hot representation with within-sample imputation | `median_mode` | Categorical variables are represented by grouped indicator columns; missing values are imputed using the selected training sample |
| One-hot representation with missingness indicators | `median_missing_indicator` | Adds screened binary indicators that record whether a source value is missing |
| Ordinal representation for categorical variables | `tree_ordinal` | Uses stable integer codes for categorical values; missing and previously unseen values remain missing |

Configuration IDs serve file names and commands. Research text uses the
descriptive representation names in the first column.

The analysis repeatedly varies \(N\) and \(K\), fits each model on the selected
training data, and evaluates predictions on the predefined test sample. A
categorical variable's encoded columns are selected together and count as one
predictor variable. In the missingness-indicator representation, each declared
indicator is an additional predictor variable.

## Reproduction

```text
FFCWS/
├── adapter/
│   ├── config/ffc.yaml
│   ├── adapter.py
│   ├── src/
│   └── tests/
├── schema/
├── data/
│   ├── private/        source data; excluded from Git
│   ├── adapter_work/   generated validation records; excluded from Git
│   └── ard/            generated analysis-ready data; excluded from Git
├── panels.yaml
├── model_params.yaml
├── requirements.txt
└── README.md
```

From the repository root:

```bash
python -m pip install -r FFCWS/requirements.txt
python FFCWS/adapter/adapter.py
aleatoric-nk-grid-panels --manifest FFCWS/panels.yaml --dry-run
```

See the [FFCWS data-preparation guide](adapter/README.md) for representation
and validation details, and the
[N-by-K implementation guide](../NK_Grid/README.md) for execution options.
