# SMR data preparation

## Role in the research design

The SMR adapter converts the provider-supplied analysis matrix into the
versioned inputs required for the \(N\)-by-\(K\) analysis. It preserves the
substantive distinction between a predictor variable and the numeric columns
used to represent that variable in a statistical model.

The source matrix contains:

| Quantity | Count |
|---|---:|
| Model input columns | 4,252 |
| Predictor variables counted in \(K\) | 497 |
| Categorical variables represented by grouped indicator columns | 29 |

The 3,784 indicator columns belonging to those 29 categorical variables are
kept in atomic groups. When a categorical variable is selected, all of its
indicator columns are supplied to the model together.

## Data transformations

The source is already a numeric, analysis-ready matrix. The adapter therefore:

- retains the supplied numeric representation of continuous and scalar
  variables;
- records which indicator columns belong to the same categorical variable;
- records the two outcome columns and the eligible predictor variables; and
- validates the resulting data and analysis specifications.

The source matrix is free of non-finite values and the negative sentinel codes
commonly used to denote missingness. It enters the analysis with its supplied
values, rows, and predictor set unchanged. Any imputation parameters required
during analysis are estimated within the selected training sample.

The \(K\)-dimension treats the 497 declared predictor variables as eligible
units of prior information. This is a research-design assumption: it gives
each declared predictor variable an opportunity to enter the random
predictor-availability experiment, while preventing the arbitrary selection of
individual indicator columns from the same categorical variable.

## Inputs and generated files

```text
SMR/
├── adapter/
│   ├── config/asample2_withlag.json     reviewed variable specification
│   ├── adapter.py                       data-preparation entry point
│   ├── build_contract.py                specification-authoring utility
│   └── tests/
├── schema/                              generated, versioned analysis specifications
└── data/
    ├── private/asample2_withlag.csv     source data; excluded from Git
    └── ard/asample2_withlag/            generated data; excluded from Git
        ├── data.csv
        ├── feature_manifest.csv
        └── provenance.json
```

The schema and predictor-universe files are versioned because they define how
the generated data are interpreted. Git excludes the source and generated data
tables.

## Reproduction

From the repository root:

```bash
python SMR/adapter/adapter.py
```

To use validation settings that match a planned analysis:

```bash
python SMR/adapter/adapter.py \
  --validation-model ols ridge lasso elastic_net random_forest \
  --min-n 10 \
  --test-size 0.3 \
  --seed 12345
```

`build_contract.py` is used only when the reviewed source header or grouped
categorical-variable declarations change. Its output must be reviewed before
it replaces the existing specification:

```bash
python SMR/adapter/build_contract.py
```

The generated files follow the common
[adapter specification](../../Adapter/ADAPTER.md).
