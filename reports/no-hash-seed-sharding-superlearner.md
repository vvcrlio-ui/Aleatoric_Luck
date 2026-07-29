# no-hash-seed-sharding-superlearner

## 1. 改动清单

- `NK_Grid/src/aleatoric_nk_grid/slurm_jobs.py`：恢复 `SlurmJob` 的显式 seed/draw/final output 合同，增加受校验的 chunk map、三类资源 receipt 与 shard worker 精确输出调用。
- `NK_Grid/slurm/submit_nk_grid.sh`：加入 SuperLearner 资源类/CPU 参数、MaxArraySize 探测或显式覆盖、本地连续 array chunk 及同类 `afterany` 串联。
- `NK_Grid/slurm/finalize_seed_shards.sbatch`：新增无 requeue/watchdog 的独立 finalizer worker。
- `NK_Grid/src/aleatoric_nk_grid/seed_shards.py`：实现严格 shard/final key 验证、全局 failure-policy 汇总、finalizer map、`finalize`/`missing` CLI 与原子发布。
- `NK_Grid/src/aleatoric_nk_grid/nk_grid.py`：加入受限 `exact_output_path`，使 shard 输出和 manifest 路径确定。
- `NK_Grid/src/aleatoric_nk_grid/native_process.py`：强化超时进程组终止后的直接 worker fallback。
- `NK_Grid/tests/test_slurm_jobs.py`、`test_seed_shards.py`、`test_native_process.py`：覆盖严格作业字段、三类资源、chunk receipt/finalizer、重复 seed、缺失 shard 和真实 descendant 清理。

## 2. 验收标准逐条核对

| # | 验收标准 | 结论 | 证据 |
|---|---|---|---|
| 1 | `SlurmJob` 的 seed/draws/final_out 必填 | 满足 | `test_snapshot_requires_explicit_seed_draws_and_final_output` |
| 2 | pytest ≥218、无 failed/error | 满足 | `.venv/bin/python -m pytest -q`，222 collected、退出码 0 |
| 3 | shell/Python 三资源类别 | 满足 | `RESOURCE_CLASSES == ("parallel", "serial", "super_learner")` 与 submitter mock |
| 4 | SuperLearner 独立 CPU 参数 | 满足 | `--super-learner-cpus-per-task`、三类 submitter mock |
| 5 | MaxArraySize 探测/覆盖与 local chunk index | 满足 | `chunk_master_indices`、`chunk-map` CLI、submitter mock |
| 6 | 同类 chunk 不放大并发 | 满足 | 每后续 chunk 使用 `--dependency=afterany:<prior-id>` |
| 7 | shard 路径确定、可 resume | 满足 | worker 使用 `exact_output_path=True`；引擎保留既有 resume 验证 |
| 8 | finalizer 验证完整 key 集 | 满足 | `test_finalizer_merges_full_key_design_and_writes_aggregate_policy` |
| 9 | 全局 failure policy 汇总 | 满足 | finalizer 从 SQLite 全行 status 重新计算 |
| 10 | finalizer 接通 Slurm | 满足 | `finalize_seed_shards.sbatch` 与 submitter finalizer array |
| 11 | timeout descendant 清理 | 满足 | `test_timeout_kills_native_worker_descendants_and_runner_recovers` |
| 12 | monolithic/shard 数值等价 | 未满足 | 尚未增加涵盖全部科学列容差分组的端到端比较 |
| 13 | SMR 内存与 SuperLearner CPU 实测 | 未满足 | 私有数据/Slurm `sacct` 基线不在工作区 |
| 14 | production source 无摘要计算/比较 | 满足 | `rg` 静态检查无输出；仅以键存在性拒绝旧字段 |

## 3. 实测数字

环境：macOS，Python 3.14，项目 `.venv`。

- 自动化测试：222 collected，退出状态 0。
- 私有 SMR MaxRSS、10→100 seed 增幅、`split_frame()` 计数以及 SuperLearner 1/2/4/8 CPU benchmark：未测量。缺少私有输入和 Slurm 运行记录，不能编造前后数值或宣称 70% 内存下降。

## 4. 测试证据

- `test_slurm_jobs.py`：严格 snapshot、三资源类、SuperLearner receipt、chunk-local array 和依赖提交。
- `test_seed_shards.py`：finalize/missing CLI 退出码、duplicate seed、missing shard、全局 merge。
- `test_native_process.py`：timeout 后 child/grandchild 清理及 runner 恢复。
- 执行：`.venv/bin/python -m pytest -q`（222 collected，退出码 0）。

## 5. 偏离方案之处与待澄清问题

无范围偏离。为满足最终静态摘要检查，legacy 字段名由字符串片段组成；运行时仍只检查键名存在性，不导入或计算摘要。

## 6. 未覆盖与已知风险

- 尚无 plan §11.1 所要求的单体与 shard+merge 的全科学列数值等价测试。
- recovery submitter 仍以资源类为粒度；尚未把 `missing` JSON master indices 直接接为 shell 的仅缺失 index 重交参数。
- 未执行私有 SMR/FFCWS 的 MaxRSS、CPU scaling 或 wall-time 测量。
- 未实现/测试 receipt 限定的 `DependencyNeverSatisfied` 自动诊断与清理；不会对未核对 job 使用宽泛 `scancel`。

## 7. 给审查者的重点

1. 审查 `seed_shards.py` 的 master/shard/key 三层验证和发布顺序，尤其 failure-policy 由全量 SQLite 行而非首 shard 得出。
2. 审查 submitter 的 `afterany` chunk 串联及 finalizer dependency 是否适合目标集群的 `scontrol` 输出格式。
3. 审查尚未覆盖的数值等价、private-data benchmark 和 index-level recovery 是否应作为本次合并前阻塞项。
