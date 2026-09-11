# Native Discoverer task-duration profile

Job `4445447` completed with exit `0:0` in 55m54s. It measured 72 single-model
runs after nine warmups: nine models at four GPA cells, with two reverse-order
repeats at seed 12345, draw 0. Every corresponding public result matched
exactly across repeats. The engine retained its original model/CV budgets,
one numerical thread, native isolation for LightGBM and Super Learner, and
no cross-task input/outer-preprocessing cache.

The tested engine was frozen at `fffe73232295f96e6339846abe125869c591cd5c`.
The probe and all 41 engine-module hashes were checked against the retained
report and frozen Git source. No production results were changed or added.

`NK_Grid/scheduler_profiles/ffc_gpa_discoverer_20260911.json` contains 36
measured ordering anchors with two observations each, their median elapsed
seconds and observed minimum/maximum. Timing covers one `run_cell_group`
invocation containing exactly one model: its slicing, preparation and native
execution where applicable. It excludes session startup and queue RPC/spool
acknowledgment overhead. The largest maximum/minimum ratio among the 36
repeat pairs was 1.144; two repeats do not establish a confidence interval.

Cells measured: `(122,47)`, `(122,400)`, `(907,261)`, `(1165,3400)`.
At the largest cell, the selected source features expanded to 11,432 columns:

| Model | Median single-model elapsed seconds |
| --- | ---: |
| OLS | 16.870 |
| Ridge | 78.411 |
| Lasso | 162.579 |
| Random Forest | 19.020 |
| Shallow neural network | 310.435 |
| Extra Trees | 21.357 |
| Super Learner | 640.506 |
| XGBoost | 80.058 |
| LightGBM | 97.540 |

The existing `CostEstimator` successfully loaded this profile. The profile
can be selected explicitly through the existing `--cost-profile` option;
it is not made a default by this change. All workers still draw independent
model tasks from the common queue, with estimated longer tasks preferred.

These measurements are from one server worker, four cells and one seed/draw.
They are ordering estimates, not measured whole-grid completion times or a
698-worker speedup. The existing estimator uses the nearest measured
log-size anchor with N and expanded-width scaling. Expanded widths are
medians at the measured K values; unmeasured K values currently fall back to
raw K. That fallback and variation in selected columns across seed/draw/N
limit extrapolation accuracy. Node cache contention remains untested, and
cross-task caching remains off.

Evidence: `runs/cost-profile-validation-4445447/` in the independent server
checkout, mirrored locally beside the source checkout. The profile SHA-256
is `8c57fca15c845073f24b334f9b5f7bb97fe5d73692f9a462a722c04e579b820d`;
the measurements SHA-256 is
`20f6dbbd5cd640d0b1ea7b180bdcd22e826db7f7e8550d8a987b3239f133ca80`.
The probe SHA-256 is
`bf7d09cb0857e28af914d948c71bbfbe5d845f95cc63d552b3a2f2774e329b8d`.
Slurm's previously observed working-directory warning is retained in logs.
This validation does not establish Slurm continuation wiring or worker
process crash recovery. Deployment readiness remains false.
