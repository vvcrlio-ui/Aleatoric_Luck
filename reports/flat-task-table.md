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
| 1 | 数值逐位不变 | 满足（dev-scale 代表性路径） | `test_chunk_execution_matches_direct_cell_group_metrics` 对全部 metric 列 `check_exact=True` |
| 2 | 行可独立执行 | 满足（代表性单 chunk） | 上述 chunk-vs-direct 等价测试 |
| 3 | 预处理复用不被破坏 | 满足 | `test_preprocess_cell_runs_at_most_once_per_mode_in_cell_group` 通过；执行仍调用原 `run_cell_group` |
| 4 | 枚举内存 O(chunk) | 未满足 | 未完成 1M/5M/20M RSS 实测；见第 5、6 节 |
| 5 | 装箱有效 | 不适用（阶段 B） | 标定成本与目标时长尚不可用 |
| 6 | 启动开销受控 | 不适用（阶段 B） | `cost-calibration` 第 1 轮为 changes-requested |
| 7 | resume 正确 | 满足（单元级） | `test_resume_is_key_set_difference_not_table_position` |
| 8 | 合并完整 | 满足（单元级） | `test_finalizer_rejects_out_of_design_and_duplicate_keys` |
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
| 1M/5M/20M RSS | 未测 | 尚未实现对应的测量 harness |

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

- 偏离：尚未完成验收 4 的 1M/5M/20M RSS dry-run 测量，因而 A6 未全部完成。这不是阶段 B 的标定工作，需在下一轮补齐。
- 偏离：chunk 合并目前由新 `finalize_chunk_shards` 实现键校验，尚未把既有 `seed_shards` 的 CLI/maps 通用化到 `chunk_id`。其行键保持 `(model, seed, draw, N, K)` 不变。
- 待澄清：`split_super_learner_min_k`、`T_target`、分区/内存/时限均未填写；按方案保留至阶段 B。

## 6. 未覆盖与已知风险

- 尚无生产规模 Parquet（1M/5M/20M）RSS 数字，无法证明大规模建表/读取的内存曲线。
- 新 flat worker/snapshot 尚未接入现有 `submit_nk_grid.sh` 的提交入口；专用 worker 已就绪，但提交端的资源 plan 输入应等阶段 B 的集群/标定数据一起接线。
- 未做 kill/requeue 的真实 Slurm 集成测试；当前仅验证了本地 resume 键集合语义。
- 完整 pytest 未在该工具会话内获得最终汇总行，尽管相关定向测试均通过。

## 7. 给审查者的重点

1. 请重点审查 `flat_task_table.py` 的行粒度：同一 cell 的 group 内模型绝不被拆分，且 Parquet row group 与 `chunk_id` 一一对应。
2. 请重点审查 `run_chunk` 是否应在下一轮直接抽取/复用 `run_cell_group` 的运行上下文，以避免目前逐行复用既有 runner 的启动开销。
3. 请重点审查阶段 A 未完成的 RSS harness 和 `seed_shards` 的 chunk 化接线；两项均在第 5 节明确列为偏离，不能视为完成。
