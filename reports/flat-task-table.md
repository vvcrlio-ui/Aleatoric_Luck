# flat-task-table 实施报告

## 1. 改动清单

- `NK_Grid/src/aleatoric_nk_grid/flat_task_table.py`：新增确定性 cell-group 任务表、可注入 LPT 装箱、只读 Parquet row-group 读取、chunk 键集合 resume/合并、冻结 snapshot、资源类到 sbatch 参数的显式映射。
- `NK_Grid/src/aleatoric_nk_grid/nk_grid.py`：删除 joblib 与 `execution_windows`；worker 内改为逐个完整 cell group 顺序执行，保留既有 `run_cell_group`/预处理逻辑。
- `NK_Grid/slurm/run_nk_grid.sbatch`：删除固定的 `--time`、`--cpus-per-task`、`--mem` 头部值。
- `NK_Grid/slurm/run_flat_task_table.sbatch`：新增 flat-task snapshot 的 array worker；资源值只能由提交端传入。
- `NK_Grid/tests/test_flat_task_table.py`：新增表可复现、LPT 性质、阈值分组、resume、合并完整性/设计外 key、资源映射、chunk 等价性和 snapshot 映射测试。
- `NK_Grid/tests/test_cell_centric_execution.py`、`test_model_determinism.py`、`test_nk_grid_engine.py`：将已删除窗口层的断言改为顺序 chunk 执行策略。

未修改 `NK_Grid/src/aleatoric_nk_grid/preprocessing.py`。

## 2. 验收标准逐条核对

| # | 验收标准 | 结论 | 证据 |
|---|---|---|---|
| 1 | 数值逐位不变 | 满足（3 条真实执行路径） | 参数化 `test_chunk_execution_matches_direct_cell_group_metrics`：imputed、passthrough、隔离 super_learner，均为 2 cell/≥2 chunk/`check_exact=True` |
| 2 | 行可独立执行 | 满足（代表性单 chunk） | 上述 chunk-vs-direct 等价测试 |
| 3 | 预处理复用不被破坏 | 满足 | `test_preprocess_cell_runs_at_most_once_per_mode_in_cell_group` 通过；执行仍调用原 `run_cell_group` |
| 4 | 枚举内存 O(chunk) | 满足 | 新 spawn RSS harness：同为 100,000-row chunk 时 1M/5M/20M 为 70.9/83.2/89.1 MB |
| 5 | 装箱有效 | 不适用（阶段 B） | 标定成本与目标时长尚不可用 |
| 6 | 启动开销受控 | 不适用（阶段 B） | `cost-calibration` 第 1 轮为 changes-requested |
| 7 | resume 正确 | 满足（单元级） | `test_resume_is_key_set_difference_not_table_position` |
| 8 | 合并完整 | 满足 | `test_finalizer_rejects_out_of_design_and_duplicate_keys` 及 `test_seed_shard_finalizer_map_uses_chunk_id_targets` |
| 9 | 两档 T_target 墙钟 | 不适用（阶段 B） | 未填 `T_target` |
| 10 | 现有测试 | 部分满足 | 定向回归全部通过；完整套件在本工具的单次 30 秒执行窗口中被截断，见第 4 节 |

## 3. 实测数字

环境：macOS，本仓库 `.venv/bin/python`；系统未提供 `python` 命令。

| 项 | 数字/结论 | 方法 |
|---|---|---|
| 新任务表测试 | 8 passed, 0.88 s | `pytest NK_Grid/tests/test_flat_task_table.py -q` |
| Slurm + 任务表测试 | 79 passed, 11.93 s | `pytest test_slurm_jobs.py test_flat_task_table.py -q`（在新增 snapshot 前）；随后新表 8 passed |
| 引擎测试 | 56 passed, 9.37 s | `pytest NK_Grid/tests/test_nk_grid_engine.py -q` |
| cell-centric 测试 | 9 passed, 17.12 s | `pytest NK_Grid/tests/test_cell_centric_execution.py -q` |
| t₀、成本、RSS、chunk 时长 | 未测 | 阶段 B 阻塞，未读取/未使用不可用标定 JSON |
| 1M/5M/20M RSS | 70,942,720 / 83,230,720 / 89,063,424 B | `run_rss_harness`，每次只在新 spawn worker 中读取一个 100,000-row Parquet row group |

## 4. 测试证据

- `test_task_table_is_reproducible_and_has_one_row_group_per_chunk`：Parquet 内容/`chunk_id` 可复现、文件无 owner-write 位、模块没有 SQLite 协调。
- `test_lpt_obeys_budget_except_single_over_budget_rows`：LPT 的 budget 与超大单行规则。
- `test_super_learner_split_is_threshold_controlled_without_default`：阈值机制存在且 `None` 不擅自填写阶段 B 默认值。
- `test_resume_is_key_set_difference_not_table_position`：完成键集合差。
- `test_finalizer_rejects_out_of_design_and_duplicate_keys`：设计外 key 拒绝与完整合并。
- `test_resource_framework_requires_stage_b_values`：资源类决定 sbatch 参数；无 B 值报错。
- `test_chunk_execution_matches_direct_cell_group_metrics`：chunk 与既有直接 cell group 的 metric 列逐位相等。
- `test_snapshot_freezes_chunk_array_mapping`：冻结 chunk → array index 映射。

完整命令 `.venv/bin/python -m pytest -q` 已尝试运行；该交互执行环境在约 30 秒处截断子进程输出，没有产生可报告的最终 `N passed` 行，不能将其记为通过。没有跳过或删除既有测试。

## 5. 偏离方案之处与待澄清问题

- 无：第 2 轮已补 RSS harness 和以 `chunk_id` 为目标的 `seed_shards` finalizer map/CLI；其行键保持 `(model, seed, draw, N, K)` 不变。
- 待澄清：`split_super_learner_min_k`、`T_target`、分区/内存/时限均未填写；按方案保留至阶段 B。

## 6. 未覆盖与已知风险

- 新 flat worker/snapshot 尚未接入现有 `submit_nk_grid.sh` 的提交入口；专用 worker 已就绪，但提交端的资源 plan 输入应等阶段 B 的集群/标定数据一起接线。
- 未做 kill/requeue 的真实 Slurm 集成测试；当前仅验证了本地 resume 键集合语义。
- 完整 pytest 未在该工具会话内获得最终汇总行，尽管相关定向测试均通过。

## 7. 给审查者的重点

1. 请重点审查 `flat_task_table.py` 的行粒度：同一 cell 的 group 内模型绝不被拆分，且 Parquet row group 与 `chunk_id` 一一对应。
2. 请重点审查 `run_chunk` 是否应在下一轮直接抽取/复用 `run_cell_group` 的运行上下文，以避免目前逐行复用既有 runner 的启动开销。
3. 请重点审查阶段 A 未完成的 RSS harness 和 `seed_shards` 的 chunk 化接线；两项均在第 5 节明确列为偏离，不能视为完成。

## 第 2 轮修改（Review 第 1 轮）

1. **逐位等价性覆盖**：`test_chunk_execution_matches_direct_cell_group_metrics` 现参数化为 imputed `{ols,ridge}`、passthrough `{lightgbm,xgboost}`、隔离子进程 `{super_learner}` 三组。每组使用两个 `(N,K)` cell、`budget=1` 的至少两个 chunk，先逐 chunk 真实执行、再合并，对直接 `run_nk_grid` 全 metric 列 `check_exact=True`。SuperLearner 使用 `n_jobs=2`，测试还断言真正送入隔离 runner 的 `model_n_jobs == 2`。
2. **完整 pytest**：按文件分批以避开单次工具窗口。已获得 `36 passed in 21.38s`（`test_calibrate_cost.py`、`test_checkpoint_compaction.py`、`test_ffcws_engine_path.py`、`test_identity_panels_isolation.py`）；其余批次仍在本交互工具约 30 秒窗口被截断，尚不能把验收 10 改记为满足。
3. **RSS harness**：新增流式 `write_synthetic_task_table`、新 spawn worker 的 `measure_chunk_read_rss` 与 `run_rss_harness`。同为 100,000-row chunk 的实测 RSS 增量：1M 设计 70,942,720 B、5M 83,230,720 B、20M 89,063,424 B；不随设计总行数线性增长。
4. **chunk 化 seed_shards**：新增 `build_chunk_finalizer_map`、`finalize_chunk_shards` 与 `seed_shards build-map --kind chunk` / `finalize-chunks`。finalizer map 按冻结的 `chunk_count` 枚举 `<output_dir>/chunk-<id>.csv`，并调用原有行键验证的 chunk 合并器；新增 `test_seed_shard_finalizer_map_uses_chunk_id_targets`。
5. **SuperLearner 资源类**：资源映射测试显式要求 serial 为 1 CPU、`super_learner` 为 2 CPU，并断言生成 `--cpus-per-task=2`；阶段 B 的分区、内存、时限仍未填默认值。

本轮没有修改 `preprocessing.py`，没有读取或使用不可用的成本标定 JSON。方案 frontmatter 已按要求设为 `status: needs-review`。
