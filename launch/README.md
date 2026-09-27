# How runs are carried out

One panel's experiment is a very large number of small, independent model fits. The launch code turns a panel and a preset into those fits, runs them on a cluster, and publishes a single result table once every result has been checked. This page explains how that works. Commands and options are in [OPERATIONS.md](OPERATIONS.md).

## One task per fit

Each combination of seed, draw, N, K and model is one task. A production panel has 2,000,000 tasks for each of the eight base models and another 2,000,000 for the Super Learner. The tasks are independent, except that a Super Learner task needs the saved predictions of the seven models it combines.

## Stages before production

A run starts with a dry-run, which only prints the launch settings. A `dev` run then checks the whole path from data to results on a small grid, and a `timing_full` run fits every point of the full grid once, so memory use and run time across the whole N and K range are known before `production` repeats the grid for 100 seeds and 50 draws. Each stage writes to its own directory.

## Phases within a run

1. Base models: the eight models are fit on every training sample. Each task saves its test predictions and its out-of-fold training predictions as soon as it finishes.
2. Index: the saved base predictions are indexed so that the next phase can find them.
3. Super Learner: SL7 is fitted for every training sample from the saved predictions.
4. Verification: every expected result, including the saved predictions of all eight base models, is checked before `final.csv` and `verified.json` are written.

## Scheduling on the cluster

Workers take tasks from a shared queue and send back each result as soon as it is done. Tasks expected to take longest start first, and the estimates are updated from timings measured in the same run. Cluster time is requested in rounds: each round is one Slurm allocation with a time limit, and before each round the controller checks how much of the account's allowance is left.

## Interruptions and resuming

A task counts as done only once its result has been saved and accepted. If an allocation ends or a worker fails, the next round schedules only the tasks that are still missing, so at most the fits that were running at the time are repeated. A resumed run uses the inputs, code and settings it started with, and results are only combined with results produced under the same settings.

## More detail

- [OPERATIONS.md](OPERATIONS.md): commands, clusters, scheduling policy, resuming and output locations.
- [PREDICTION_CACHE.md](PREDICTION_CACHE.md): how predictions are saved and checked, and how SL7 uses them.
- [SCHEDULER_EFFICIENCY.md](SCHEDULER_EFFICIENCY.md): how task costs are estimated.
- [DISCOVERER.md](DISCOVERER.md) and [BMRC.md](BMRC.md): site notes, including the older BMRC multi-panel suite.
