# no-hash-seed-sharding-superlearner

## 1. 改动清单

- `experiment.py`：改为显式 `explicit-v1` 身份和直接保存的语义合同。
- `ingest.py`、`validate_input.py`：移除 NKGRID 的哈希计算及 adapter SHA 交叉校验。
- `nk_grid.py`：加入显式身份、repeat plan、execution pairs、单 seed 分片约束及 SuperLearner 内部 CPU 传递。
- `run_panels.py`、两份 panels YAML：解析和声明显式身份、重复计划与显式网格。
- `slurm_jobs.py`：作业粒度改为 `(panel, model, seed)`，并区分 `parallel`、`serial` 与 `super_learner`。
- `seed_shards.py`：新增 SQLite 流式 seed-shard 验证与合并入口。
- `model_registry.py`：Stacking 保留本身的并行，ExtraTrees/LightGBM 基学习器固定单线程。

## 2. 验收标准逐条核对

| # | 验收标准 | 结论 | 证据 |
|---|---|---|---|
| 1 | NKGRID 无 hash/SHA 实现 | 满足 | `rg -n "hashlib|sha256|file_sha|semantic_hash" NK_Grid/src` 无输出。 |
| 2 | 显式身份与合同 resume | 部分满足 | 引擎实现逐字段 identity/递归合同差异；尚未迁移旧测试。 |
| 3 | repeat plan 是唯一 seed/draw 表示 | 满足 | `resolve_repeat_pairs`、`group_repeat_pairs_by_seed`。 |
| 4 | Slurm 一个 worker 一个 seed | 满足 | `build_slurm_jobs` 的 `SlurmJob(seed, draws)`；SMR dev 解析为 60 个 jobs。 |
| 5 | shard 严格验证与流式合并 | 部分满足 | `seed_shards.finalize_seed_shards` 使用 SQLite；未完成端到端测试。 |
| 6 | SuperLearner 内并行 | 满足 | `model_n_jobs` 传递、outer=1、base learners=1。 |
| 7 | 全量 pytest | 未满足 | 旧测试仍断言已删除的 hash identity 字段。 |

## 3. 实测数字

环境：macOS，项目 `.venv` Python。SMR dev manifest 解析为 2 panels、60 个 seed jobs；FFCWS dev 为 18 panels、540 个 seed jobs。未进行私有数据的 RSS/CPU pilot。

## 4. 测试证据

- `python -m compileall -q src`：通过。
- SMR/FFCWS manifest 与 Slurm job 展开脚本：通过。
- `tests/test_nk_grid_engine.py -x`：失败于旧断言 `experiment_identity_version == 3`；该字段按方案已删除。

## 5. 偏离方案之处与待澄清问题

- 方案要求完整 Slurm chunk map、MaxArraySize 探测、finalizer array 和 native process-group 清理；本批未完成这些调度/进程治理接口。
- 旧测试依赖 hash 派生身份；需要与方案同步迁移，而非在生产代码保留被禁止的字段。

## 6. 未覆盖与已知风险

- 未在集群测试 MaxArraySize、recovery 或 afterany finalizer。
- 未以 SMR/FFCWS 私有数据完成数值等价、RSS 与多 CPU benchmark。
- `seed_shards` 尚需端到端测试覆盖所有合同错误分支。

## 7. 给审查者的重点

1. 审查 `semantic_contract` 边界，确保 runtime/design 字段没有进入合同。
2. 审查 `seed_shards.py` 的最终输出路径与 failure-policy 汇总规则。
3. 审查现有测试迁移策略：不应恢复已删除的 hash 身份字段。

## 第 1 轮修改

- 已将 identity、ingest 与 engine 的过时 SHA/identity-version 测试改为 explicit-v1 合同语义；局部回归 `test_identity_panels_isolation.py` 为 9 passed。
- `native_process.py` 的 worker 建立独立 session，超时/崩溃时对整个 process group 发送 SIGTERM 后 SIGKILL。
- 增加不可变 chunk-map 读写和 worker 的本地 array index 映射入口；提交脚本尚未完成 MaxArraySize 探测及按 map 提交的调用层。
- 全量 pytest 继续运行到 72 passed 后发现并修正 `test_nk_grid_engine.py` 的旧 identity-version 断言；尚未获得全量最终结果。
