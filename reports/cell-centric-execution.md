# cell-centric-execution 实现报告

## 1. 改动清单

- `NK_Grid/src/aleatoric_nk_grid/nk_grid.py`
  - 将 `_run_nk_grid_locked` 内原来的单模型 `run_one` 改为 `run_cell_group`：同一 `(seed, draw, N, K)` 只切片一次，按 `imputed` / `passthrough` mode 至多各预处理一次，再按原模型顺序串行 fit。
  - 完整 cell 作为并行执行原子单元，返回行进入 checkpoint 缓冲区后仍严格按原始 jobs 顺序和 `batch_size` 行数落盘。
  - 非 native estimator 每次接收 `prepared.X_train` / `prepared.X_test` 的 deep copy；lightgbm / super_learner 继续依靠 native 子进程序列化隔离，并用锁保护共享 `IsolatedProcessRunner`。
  - `_empty_diagnostics`、`_fit_predict_model_cell` 和结果行新增 `_preprocess_seconds`、`_preprocess_computed`、`_slice_seconds`、`_cell_wall_seconds`、`_peak_rss_bytes`；预处理与切片只在实际承担成本的行记账。
  - `_preserve_prior_timings` 保留 checkpoint shards 删除后无法重算的新 telemetry 汇总。
- `NK_Grid/src/aleatoric_nk_grid/experiment.py`
  - `_sqlite_diagnostics_summary` 与 `diagnostics_summary` 为 `diagnostics.by_model` 汇总 `preprocess_seconds_total`、`cell_wall_seconds_total`、`peak_rss_bytes_max`。
- `NK_Grid/tests/test_cell_centric_execution.py`
  - 新增 7 个测试，覆盖等价性、mode 调用计数、mutation 隔离与引用释放、并发不变性、telemetry、RSS、`max_jobs` / `batch_size` / resume。
- `reports/cell-centric-execution.md`
  - 本实现与验证报告。

## 2. 验收标准逐条核对

| # | 验收标准 | 结论 | 证据 |
|---|---|---|---|
| 1 | 每个 cell 的 `preprocess_cell` 调用次数不超过实际 mode 数 | 满足 | `test_preprocess_cell_runs_at_most_once_per_mode_in_cell_group`：`batch_size=1`、3 个模型、2 个 mode，调用序列实测为 `["ols", "lightgbm"]`，checkpoint 仍为 `[1, 1, 1]` 行三批 |
| 2 | 重组前后全部 metric 列逐位相同 | 满足 | `test_cell_group_metrics_exactly_match_model_at_a_time_execution`：3 模型 × 8 cells × 30 metric 列，共 720 个值以 `check_exact=True` 全量一致 |
| 3 | 一个 cell 组内 `_preprocess_seconds` 非零行数不超过 2 | 满足 | `test_preprocess_telemetry_counts_only_mode_misses_and_matches_manifest`：3 个 imputed 模型实测非零行数为 1，`_preprocess_computed=True` 行数为 1 |
| 4 | manifest 预处理汇总等于实际 miss 墙钟总和且不随模型数线性增长 | 满足 | 独立实测：8 模型实际预处理 0.055743000 s，manifest 为 0.055746542 s，相差 0.0064%；`test_preprocess_telemetry_counts_only_mode_misses_and_matches_manifest` 同时核对 manifest 与 checkpoint 行求和及 1/3 模型耗时口径 |
| 5 | estimator 执行后不修改共享 `prepared` | 满足 | `test_estimator_mutation_cannot_change_or_retain_shared_prepared_cell`：首个假 estimator 将 train/test 输入就地改为 `-999/-777`，第二个 estimator 逐位核对共享 prepared 快照未变 |
| 6 | cell 结束后不持有 `prepared` 引用 | 满足 | `test_estimator_mutation_cannot_change_or_retain_shared_prepared_cell`：cell 完成并 `gc.collect()` 后 `weakref()` 实测为 `None` |
| 7 | 现有 208 个测试全部通过 | 满足 | 激活仓库 `.venv` 后运行 `python -m pytest -q`：`215 passed, 14 warnings in 35.63s`（208 基线 + 7 新测试） |
| 8 | `--max-jobs` / `batch_size` / resume 语义不变 | 满足 | `test_max_jobs_batch_boundaries_and_resume_match_uninterrupted_output`：2 模型、`batch_size=3`、首次 `max_jobs=3` 得 3 行；恢复后的全部 30 个 metric 列与不中断运行 `check_exact=True` 一致；调用计数测试另证 `batch_size=1` 仍逐行 checkpoint |

## 3. 实测数字

测量环境：

- Python 3.14.3
- macOS 26.5.2 arm64
- synthetic 数据 128 行，固定内部切分后 96 个训练行，K=256
- 8 个非 passthrough 模型：ols、ridge、lasso、elastic_net、random_forest、shallow_neural_network、extra_trees、super_learner
- 使用真实 `preprocess_cell`；fit 替换为固定均值 estimator，以隔离预处理成本
- “改动前”口径用相同输入逐个单模型执行 8 次，复现原 `run_one` 对每个模型各预处理一次的路径；“改动后”在一个 cell group 中一次选择相同 8 个模型

| 指标 | 改动前 | 改动后 | 变化 |
|---|---:|---:|---:|
| 每 cell 的 imputed 预处理调用次数 | 8 | 1 | -87.5%，减少 7 次 |
| 8 模型累计预处理墙钟 | 0.422216875 s | 0.055743000 s | -86.8%，约 7.57× 加速 |
| manifest `preprocess_seconds_total` | 不适用 | 0.055746542 s | 与外部实测相差 0.000003542 s（0.0064%） |

最大 K 引用释放测量使用同一环境、K=256、2 个模型、6 个 draw，每个 cell 组落盘后 `gc.collect()` 并通过 macOS `libproc` 读取当前 RSS。六次读数依次为：

`211,763,200`、`211,795,968`、`211,795,968`、`211,779,584`、`211,795,968`、`211,795,968` bytes。

范围仅 32,768 bytes，序列在第 4 次下降，不是逐组增长。单元测试另以 K=64 重复同一 RSS 断言。私有数据中的 K=8053 / 16,106 展开列未访问。

## 4. 测试证据

- `test_cell_group_metrics_exactly_match_model_at_a_time_execution`
  - 对应“等价性”：小 panel 的 grouped 与 model-at-a-time 最终 CSV 全部 metric 精确比较。
- `test_preprocess_cell_runs_at_most_once_per_mode_in_cell_group`
  - 对应“调用计数”：monkeypatch 计数，2 个 mode 恰好 2 次；同时验证完整 cell 执行后仍按 `batch_size=1` 分三次 checkpoint。
- `test_estimator_mutation_cannot_change_or_retain_shared_prepared_cell`
  - 对应“并发 mutation”及引用释放：恶意 estimator 就地修改副本不污染 prepared；cell 后 weakref 释放。
- `test_cell_groups_are_deterministic_with_concurrent_outer_jobs`
  - 对应“并发不变性”：`n_jobs=2` 下两次运行的全部 metric 逐位一致。
- `test_preprocess_telemetry_counts_only_mode_misses_and_matches_manifest`
  - 对应“telemetry 口径”：只给 miss 行记时，manifest 等于 checkpoint 行之和，1 模型与 3 模型不呈线性增长；同时核对 wall total 与 peak max。
- `test_max_k_cell_groups_do_not_show_strictly_increasing_rss`
  - 对应“内存”：最大配置 K 下 6 个 cell 组后的当前 RSS 不严格递增。
- `test_max_jobs_batch_boundaries_and_resume_match_uninterrupted_output`
  - 覆盖验收标准 8：跨 cell 边界截断后恢复与不中断结果一致。

最终命令结果：

```text
215 passed, 14 warnings in 35.63s
```

无 skipped 或失败。14 个 warning 与基线相同：8 个 sklearn MLP `ConvergenceWarning`，6 个 LightGBM feature-name `UserWarning`。

## 5. 偏离方案之处与待澄清问题

- 输入不可变性采用方案首选：非 native estimator 前对 train/test 分别 `.copy(deep=True)`；SERIAL_OUTER_MODELS 不额外拷贝，由序列化跨进程隔离。
- 完整 cell 先一次执行并返回全部行，再由独立缓冲层严格按原 `batch_size` 落盘。因此 `batch_size` 和 checkpoint 行数语义不变；若停止信号在一个大 cell 的首个 checkpoint 块后到达，已计算但未落盘的同 cell 余下行会丢弃并在 resume 时重算。这不改变最终结果，但会在中断边界产生少量重复计算。
- `_peak_rss_bytes` 使用执行进程在该行结束时的累计 RSS 高水位；native 模型取 native 子进程返回值，其余取实际执行 fit 的进程。标准库高水位无法按单行重置，所以较晚的轻量行可能重复此前峰值。manifest 的 `peak_rss_bytes_max` 仍是运行峰值。请审查者确认是否需要未来改为额外采样线程实现严格的“单行区间峰值”；本方案范围内未引入新依赖或监控线程。
- 除上述口径说明外，无偏离方案之处，无其他待澄清问题。

## 6. 未覆盖与已知风险

- 按仓库边界未读取 `SMR/data/`、`FFCWS/data/`，所以没有在私有数据的 K=8053 / 16,106 展开列上实跑；本地内存验证最高为 K=256。
- BART-only 进程任务的 pickle 契约由既有 `test_bart_process_task_remains_pickleable_without_cached_order_ipc` 覆盖；BART 与 native 模型混在同一个 config 的真实第三方后端组合未做端到端重负载测试。
- `deep=True` 可隔离当前数值 DataFrame 的底层数组；若未来允许包含可变 Python object 的 object dtype predictor，pandas deep copy 不递归复制对象内容。
- 中断发生在“完整 cell 已算完、仅部分行已 checkpoint”时，resume 会重算未落盘行；这是保持 cell 原子复用和严格 checkpoint 批大小同时成立的成本。
- `_peak_rss_bytes` 是累计高水位而非可重置的单行局部峰值，见第 5 节。

## 7. 给审查者的重点

1. 重点检查 `run_cell_group` 中 mode miss/error 缓存、每模型 deep copy 和 `finally` 清理，确认失败路径也不会重复预处理或泄漏 prepared。
2. 重点检查完整 cell 的执行窗口与 checkpoint 缓冲分层，尤其 `batch_size=1`、`max_jobs` 截断和 stop/resume 边界。
3. 重点确认 `_peak_rss_bytes` 的累计高水位解释是否满足 timing run 的预算用途，以及 manifest 只累加 miss 行 `_preprocess_seconds` 的口径。
