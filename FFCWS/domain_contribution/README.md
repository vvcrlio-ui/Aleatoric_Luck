# How much each domain of information adds to FFC predictions

The FFCWS background file mixes many kinds of information about a family: its finances, housing, health, the parents' relationship, how the child is raised and how the child develops. This experiment asks how much each of these domains improves out-of-sample prediction of the six FFC outcomes. A domain's contribution here means a gain in predictive accuracy, not a causal effect.

The method follows Yan, "Revisiting the social determinants of health with explainable AI: a cross-country perspective", *American Journal of Epidemiology* 195 (2026): 681–688. Instead of removing one domain from a full model, it fits a model for every combination of domains and averages how much adding a domain helps across all of them.

## Domains

Each background variable is assigned a domain through the topics in the official [FFCWS metadata](https://metadata.ffcws.princeton.edu/). A variable carries up to two topics, and its first topic decides. Four pairs of topics that describe the same part of family life are merged, in line with the broader domains of the reference paper, whose socioeconomic domain holds income, wealth and occupation together and whose social-connections domain holds marriage and relations with family and friends. Childcare, with 14 variables, would otherwise stand alone.

The child development and education topics also hold questions about the parents themselves: their schooling and job training, their own cognitive tests and impulsivity, and the interviewer's impressions of the caregiver. The metadata records whom each question is about, and the 99 questions about a parent and not about the child form a domain of their own. Left in the child domain, they would credit the child's development with information about the parents, such as the parents' earlier job training when the outcome is the caregiver's job training.

| Domain | FFCWS topics | Variables |
|---|---|---:|
| Demographics | Demographics | 50 |
| Economy and employment | Finances, Employment | 600 |
| Housing and neighborhood | Housing and neighborhood | 344 |
| Family and social ties | Romantic relationships, Family and social ties | 494 |
| Parenting and childcare | Parenting, Childcare | 457 |
| Child development and education | Cognitive and behavioral development, Education and school, about the child | 417 |
| Parent education and traits | Cognitive and behavioral development, Education and school, about a parent | 99 |
| Health | Health and health behavior | 448 |
| Attitudes and expectations | Attitudes and expectations | 163 |
| Systems | Systems (mostly contact with the criminal justice system) | 58 |

The remaining 270 variables, interview paradata and survey weights plus 15 variables without a metadata entry, belong to no domain. [domains.csv](domains.csv) lists every variable with its topics and domain. The in-home variables of years 3 and 5 are named `hv3…` and `hv4…` in the FFC file but appear in the metadata without that prefix, so they are matched by wave and question number and checked against their labels.

## Design

For each outcome, families with an observed outcome are randomly split into 70% training and 30% test rows, stratified by the outcome when it is binary. One split is shared by every combination of domains, so two combinations are always compared on the same families. The split is redrawn for each of ten seeds. A random split is used rather than the official FFC holdout because the differences between domains are small, and averaging over many test sets reduces the noise that any single test set of about a thousand families carries.

With ten domains there are 1,023 non-empty combinations. For each one, LightGBM is trained on the training rows using only the variables of those domains, with the number of boosting rounds chosen by five-fold cross-validation, and scored on the test rows. Missing values are passed to LightGBM as they are. The empty combination predicts the training mean for every family, and one more model uses all 3,400 variables.

Continuous outcomes are scored by test R² relative to the training mean. Binary outcomes are scored by the area under the precision-recall curve, as in the reference paper, and by the Brier score rescaled against the training positive rate, the measure used in the Fragile Families Challenge.

## Reading the contributions

Write v(S) for the test score of the model that uses the domains in S. Adding domain d to S gains v(S + d) − v(S), and there are 512 combinations S that do not contain d. Four summaries are reported for each domain:

- the plain average of the 512 gains, which is the reference paper's measure;
- the Shapley value, which weights the gains so that every size of S counts equally; the Shapley values of the ten domains add up to the score of the model with all ten;
- the gain when d is the only domain;
- the gain when d is added last, after the other nine.

A domain whose information overlaps with others can score well alone and add little when added last. The averages sit between these two ends. Each summary is computed per seed; the results report its mean over seeds and the standard error across seeds. Because the test sets of different seeds overlap, that standard error understates the true uncertainty and serves only to show how stable a ranking is.
