# Design decisions

Decisions that change how experiments are computed or run, with the evidence behind them. Each entry says whether it is in the code, proposed, or planned. The local evidence directories named below (`runs/...`) are kept in the working copy, not in Git.

## 1. SL7 out-of-fold fits reuse the full fit's selected parameters

Status: in the code since `3f4fa6c` (`algorithm_version: nk-grid-v13-fixed-oof-1`). Not yet run as a dev or timing_full experiment.

SL7 combines seven models through their out-of-fold (OOF) predictions on the training rows. Each model is fitted five more times, once per fold. Before this change, every one of those fits repeated the model's parameter search. Now each fold fit uses the value chosen when the model was fitted on all N rows:

- Ridge and the neural network: the penalty alpha (with the batch size when `mlp_batch_size: cv`);
- Lasso: the selected fraction of the largest useful penalty (`alpha_max`), applied to each fold's own `alpha_max`; on the absolute scale, the selected alpha;
- XGBoost and LightGBM: the number of boosting rounds;
- OLS, random forest and extra trees are not tuned and are unchanged.

The full fits, their test predictions and the eight standalone results are unchanged bit for bit. Only the OOF predictions that SL7 uses change.

Evidence:

- Trial, 2026-10-08 (`runs/sl-fixed-oof-params-20261008`): five panels (SMR hourlywage; FFC tree_ordinal GPA, materialHardship, layoff, eviction), 25 cells (N and K grid indices 3, 7, 11, 15, 19) × 30 seeds, both versions computed from the same full fit. Base-model fit time (full plus OOF, the seven SL7 models) fell by 70.3% (GPA), 70.3% (materialHardship), 74.2% (layoff), 71.7% (eviction) and 67.6% (SMR).
- SL7 changed by at most 0.001 in R² and 0.0005 in log loss for N ≥ 428. At small N it was worse for materialHardship (R² −0.066, se 0.027, at N = 22; −0.013, se 0.005, at N = 63) and slightly worse for GPA (−0.006, se 0.0045, at N = 58); SMR was slightly better (+0.019, se 0.007, at N = 27; +0.0075, se 0.0033, at N = 375). The median difference was about zero in every N band. Fixed-parameter OOF predictions fit the training rows slightly better than nested ones, and SL7 weights moved towards Ridge and Lasso in regression.
- Every trial task checked that refitting on all N rows with the fixed value reproduces the production full fit (26,250 tasks).
- Linux tests, job 4457888 (`runs/sl7-fixed-oof-gate-20261010`): no failure that 983bf1b does not also have.
- Reconciliation with the trial, jobs 4457889–4457893 (`runs/sl7-fixed-oof-reconcile-20261010`): 70 trial tasks rerun with the new engine; test predictions, fold assignments and OOF predictions are identical bit for bit, and the selected values match in 40 of 40 tasks where the trial recorded them.

Already finished experiments (materialHardship, GPA, SMR hourlywage) used nested OOF fits and are not recomputed. A paper's methods should say which version produced which results.

## 2. Proposal: 100 seeds × 5 draws instead of 100 × 50

Status: proposed, waiting for the team. The production preset in the code is still 100 × 50.

A seed sets the train/test split (for SMR) and, together with the draw, the random ordering of rows and variables; a draw only reorders. For FFC the test set is given by the data provider, so seed and draw play the same role.

Comparison at equal cost (500 seed-draw pairs), job 4457910 (`runs/seed-draw-design-20261010`), using the finished 100 × 50 runs of materialHardship, GPA and SMR hourlywage. For each model and (N, K) cell, a nested variance decomposition of r2_test gives the between-seed and within-seed variance. The quantity compared is the sampling variance of the cell-mean r2_test, that is, how much that mean would change if the same design were rerun with new seeds and draws. Both designs estimate the same quantity.

- FFC: no difference (median variance ratio 50×10 / 100×5 = 1.00; 5th–95th percentile 0.98–1.03).
- SMR: 100 × 5 has lower variance, and the gap grows with N and K. Median ratio, all models except OLS:

  | | K 1–5 | K 6–26 | K 36–186 | K 259–497 |
  |---|---|---|---|---|
  | N 10–37 | 1.00 | 1.00 | 1.01 | 1.01 |
  | N 52–269 | 1.00 | 1.03 | 1.06 | 1.20 |
  | N 375–1945 | 1.01 | 1.06 | 1.19 | 1.66 |
  | N 2703–5224 | 1.02 | 1.08 | 1.27 | 1.85 |

  When N approaches the training pool and K approaches all variables, draws of one seed train on nearly the same data; most of the variation comes from the split.
- 100 × 5 is the first five draws of the existing 100 × 50 design, so new results can be compared cell by cell with the finished runs.

Together with decision 1, 100 × 5 would bring a panel to roughly 3% of its previous model-fitting work. This is an estimate (0.1 × about 0.3), not a measurement; a timing_full run is needed before sizing production.

## 3. A simpler execution pipeline

Status: in the code on branch `per-cell-pipeline` (`29c2816`); checked against the previous pipeline in dev runs. The full pipeline used for the finished production runs is preserved at tag `multiround-pipeline-v1` (`3f4fa6c`), which already includes decision 1.

The current pipeline was built for 16 million base fits and 2 million SL7 fits per panel, run by about 14,000 workers on 300 nodes over several allocations. At the smaller scale most of that machinery is no longer needed.

Changes:

- One task per cell (seed, draw, N, K) instead of one per model: the worker fits the eight models one after another and fits SL7 from the seven OOF predictions in memory. There is no dependency between tasks, no separate SL phase and no index of saved predictions. Test and OOF predictions are still saved, appended to one file per worker rather than one file per cell.
- Workers claim cells from shared storage, largest estimated cost first, and each writes its own result file. This replaces the queue service, node relays, dispatcher shards, heartbeats and leases, and removes the requirement that compute nodes reach each other over TLS.
- One allocation per run; an interrupted run is resumed by skipping finished cells. This replaces the multi-round controller, its controller jobs and lock, round limits and online cost scheduling.
- One completeness check of every expected result before `final.csv` is written, instead of parallel verification and sealing.
- Run-level identity (commit, settings, input data) instead of a content identity per saved record.
- Removed: fold-level checkpoints, storage reservation and quota ledgers, and the isolated process for native libraries. The production cache path already bypassed that process (`nk_grid.py`, whenever folds were persisted), and the three cached production runs finished without it.
- Removed with them: the model code no run reached any more, namely the stacking Super Learner fitted as one model (with its own copies of the base learners and the optional SL4 control) and every model built without the fold-local preprocessor, and the `super_learner` parameters in `model_params.yaml`. SL7 has no parameters of its own.
- Every run now saves per-row predictions, so every dataset needs a row ID (`id_column`).

Why this is expected to be enough (estimates unless stated):

- Longest single cell, measured in the 2026-10-08 trial (seven SL7 models fitted one after another on one core, fixed OOF): 16 minutes for GPA, 18 minutes for materialHardship, 51 minutes for layoff, 79 minutes for eviction and 3.2 hours for SMR at N = 5224, K = 497. Every cell fits well inside one allocation.
- SMR hourlywage needed about 996 million worker-seconds of base fitting at 100 × 50 with nested OOF. With decisions 1 and 2 this becomes roughly 34 million worker-seconds, about 9,500 worker-hours, or about 200 node-hours on Discoverer. With 100 × 50 kept, it is about ten times that; the longest cell is the same, so one allocation per run still holds, but a run needs more nodes or a longer limit.
- Keeping a cell's OOF predictions in memory costs under 1 MB for the largest SMR cell. Peak memory is set by the largest single model fit, as now, provided each fitted model is released after predicting. To be confirmed with timing_full.
- The queue and controller code (about 6,500 lines across the engine and launcher) was the source of several recent failures, among them node-relay endpoints overwriting each other (2026-09-24), cross-node submission locking and `sinfo` parsing on BMRC.

The engine went from 18,145 to 6,349 lines of Python and the launcher from 1,985 to 877. Checks:

- Engine, launcher and adapter tests on Windows and Linux (Discoverer job 4457999), with no failure that the previous commit does not also have; new checks that one cell's rows, saved arrays and SL7 equal a fresh computation, that publication refuses missing, duplicate or failed cells, and an end-to-end run with an interruption and a resume (job 4458045).
- Dev runs of three panels with both pipelines on Discoverer (FFC tree_ordinal GPA and layoff, SMR hourly wage; 81 cells and 729 result rows each): `final.csv` is identical in every row and column, including the SL7 rows, and every base model's test and OOF predictions, fold assignments and training-row IDs are identical bit for bit. A dev run took 2 Slurm jobs instead of 15.
- Not yet measured: per-cell time and peak memory at full scale (timing_full).
