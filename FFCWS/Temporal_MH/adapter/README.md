# FFCWS material hardship across childhood

This experiment asks how well a family's material hardship at one age can be predicted from everything the study had observed about the family by an earlier age. The data are the public release of the Future of Families and Child Wellbeing Study (ICPSR 31622, all-years file), which followed children born in 1998–2000 at birth and at ages 1, 3, 5, 9, 15 and 22.

## The landmark × outcome-age triangle

Two ages define each prediction. The landmark t is the age up to which information is used: every variable collected at birth, 1, 3, 5, 9 or 15, up to and including t. The outcome age T is the age at which hardship is measured, and t is always earlier than T. Hardship is an outcome at ages 1, 5, 9, 15 and 22, which gives 19 (t, T) cells, each a panel in [panels.yaml](../panels.yaml) named `hardship_t<t>_T<T>`.

Reading the triangle along a row holds the outcome fixed and adds earlier life history. Reading along a column holds the information fixed and moves the outcome further away. Cells with the same gap T − t compare an equal forecasting horizon at different stages of childhood; the uneven spacing of the waves leaves two such pairs, 1→5 with 5→9 and 3→9 with 9→15.

## Outcome

Material hardship is the share of 11 hardships the respondent reports for the past year, the measure used in the Fragile Families Challenge: receiving free food, going hungry, not paying the full rent, eviction, not paying a utility bill, utilities shut off, borrowing to pay bills, moving in with others, staying in a shelter, someone unable to see a doctor because of cost, and the phone being disconnected. Each item is coded yes or no, and a score requires an answer to all 11.

At ages 1, 5 and 9 the questions are in the mother's interview; at 15 and 22 they are in the primary caregiver's, who is the biological mother in close to nine of ten families. The wording differs slightly between waves: at age 1 the hunger and shut-off items do not say "because there was not enough money", and at age 5 the shut-off item asks only about electricity. The mother's age-3 interview asks only 8 of the 11 items, so age 3 serves as a landmark but not as an outcome age.

## Families

All outcome ages use the same 2,126 families: those with a complete hardship score at ages 1, 5, 9, 15 and 22. With one set of families, a difference between outcome ages cannot come from a change in who is being predicted, which attrition by age 22 would otherwise cause. Every panel has the same rows, so a seed draws the same 30% test families in every cell.

## Predictors

A variable belongs to the landmark of the wave the study metadata assigns it to, and a landmark uses its own variables and those of every earlier one, so the variable sets are nested. Variables are selected from the metadata alone: text and identifier fields are left out, as are the survey weights, sample flags and interview records that the metadata files under "Paradata and weights", because they describe the survey rather than the family.

The one use of the data in selecting variables is that a variable with no observed value among the 2,126 families is dropped, since an entirely empty column gives a model nothing to learn from. The resulting counts are 660 variables at birth, 2,225 by age 1, 5,874 by 3, 8,792 by 5, 11,859 by 9 and 13,228 by 15.

Codes −1 to −9 mark refusals, skips and other missing answers in every FFCWS file and become missing values. Continuous, binary and ordered variables enter as one numeric column each, and any other negative whole number in them is also treated as missing. Unordered categorical variables are split into one 0/1 column per category named in the Stata value labels, and a value outside those labels makes the whole variable missing for that family. Some ordered and binary questions accept a second kind of answer, labelled with codes of 100 and above: a parent asked on how many days a week they read to the child can answer 0 to 7, or "201 almost every day" and "203 at least once a month". These codes are not on the 0-to-7 scale, so such a variable is split into categories in the same way. A categorical variable with no labels has no defined categories and is left out. Imputation of missing values is learned within each training sample, as for every dataset ([why](../../Adapter/README.md#why-imputation-happens-during-training)).

The variable list, each variable's landmark and categories, and every exclusion are fixed in [config/hardship.json](config/hardship.json), so the same list is used however the families are split.

## Sizes within each cell

Each cell is fitted at 12 combinations of N and K. N takes six sizes from 50 to 1,488, the whole training sample of a split, evenly spaced on a log scale, which traces how error falls as families are added. K takes two values: every variable available at the landmark, and 660, the number available at birth. The first shows what the full life history to date predicts; the second gives every landmark the same number of variables, so a gain between landmarks reflects which information is available rather than how much.
