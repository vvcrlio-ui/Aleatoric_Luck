# no-hash-seed-sharding-superlearner

- 日期：2026-07-29
- 分支：`codex/no-hash-seed-sharding-superlearner`
- D1–D7 返工基线：`b1d4534`
- 本轮异常分流修正基线：`7a811ed`
- 本轮文档修正请求基线：`4a26b58`（当前等价重写 HEAD：`2bea830`）
- 正式实施规范：`plans/no-hash-seed-sharding-completion.md`
- 本轮范围：仅更新 Adapter 两份说明、`NK_Grid/README.md` 与本报告

结论：本轮未修改实现或测试；Adapter 文档不再生成引擎拒绝的摘要字段，
`NK_Grid/README.md` 已与 seed task → per-model finalizer → per-panel publish
执行链及 recovery 入口一致，全量 pytest 为 `293 passed`。此前 D1–D7 均已完成，
生产实现相对 `b1d4534` 为
`+122/-566`，净减少 444 行；异常分流轮次相对 `7a811ed` 为 `+12/-7`，净增加 5 行。
production preset 的 20,000,000-row panel publish
已完成，独立发布进程 MaxRSS 为 837,828,608 bytes；最终全量 pytest 连续三次均为
`293 passed`。本报告第 5 节列出两项按方案约束明确放弃的保证，等待审查决定是否接受。

## 1. 实现结果

- D1：删除 `FINALIZATION_STAGES`、finalization stage receipt、active-finalization
  scheduler query、两个 CLI 子命令，以及 shell 中的 recovery 前置守卫、stage receipt
  写入和 receipt 失败后的 `scancel`。recovery 恢复为直接提交本轮 seed chunks，
  随后无条件提交 finalizer array 和 publish array。
- D2：删除 existing-result no-op、artifact provenance、publish marker、publication
  residuals、`.previous` 备份/回滚和全部 `input_artifacts` 构造/透传。finalize/publish
  每次都重新合并、先 rename CSV、再 rename manifest；唯一 `mkstemp` 临时文件和只覆盖
  发布动作的 output lock 保留。
- D3：`_shard_state()` 重新捕获 `OSError` 并把单个不可读 shard 归入 `invalid_targets`；
  诊断继续扫描其余 shards。finalize/publish 自身的 I/O 错误仍返回 4。
- D4：`feature_universe` 只接受精确键集合 `{mode, definition_file}`，修正重复
  `definition_file` 的报错文本。SMR/FFCWS adapters 和 19 个已提交 schema 同步删除
  `definition_sha256`，使真实集群输入符合该精确合同。
- D5：加载 schema 时读取 feature-universe definition JSON，并把规范化内容直接放入
  `semantic_contract.feature_universe.definition`；定义内容变化会进入 finalizer 的现有
  逐字段 contract 比较，不新增摘要。
- D6：删除 `NKGridConfig.resume_group` 及 panel root/shared/config 三处透传；全仓库无残留读取。
- D7：删除 finalize/publish 中的 `expected_keys`、`actual_keys`、
  `all_expected_keys` 和 per-model Python key sets。SQLite `expected`/`rows` 表使用
  count 与双向 `NOT EXISTS` 检查缺失和越界；两表使用 `WITHOUT ROWID`，payload 只保存
  有序值数组。production publish 全程流式处理。
- 根 `README.md` 改为面向 Slurm 使用者的三节说明：研究问题与方法、安装/数据/两-seed
  smoke/production 提交与三级任务链、故障回传材料和路径；不列内部函数或设计决策。
- `experiment.py` 定义专用于 `output_run_lock()` 的 `OutputRunLockError`，锁竞争不再抛
  裸 `RuntimeError`；锁本身、锁范围和非阻塞行为均未改变。
- `seed_shards.main()` 使用互不重叠的 handler：`OutputRunLockError` 返回新专用非零码
  5，`OSError`/`sqlite3.Error` 返回 4，既有 incomplete/validation 分别返回 3/1；
  其他异常不捕获。三处 stderr 写入改用模块顶部 `import sys`。
- 上一轮删除 finalize/publish manifest 上两行只服务于 `b1d4534` 旧产物的
  `pop("input_artifacts", None)`。
- `Adapter/README.md` 与 `Adapter/ADAPTER.md` 删除 `definition_sha256` 生成示例、
  `hashlib` 依赖和失效摘要校验说明；改写为 definition 内容进入 semantic contract
  并在合并时逐字段比较，provenance 不再由引擎读取或校验。
- `NK_Grid/README.md` 把 Slurm task 单位改为 `(panel, model, seed)`，补齐三个资源类、
  seed shard → finalizer → publish 链、对应监控名称、missing 诊断和 snapshot recovery，
  并要求真实运行显式声明三个身份字段。

退出码合同保持为：

| 退出码 | 含义 |
|---:|---|
| 0 | finalize/publish 成功；`missing` 成功完成诊断 |
| 1 | identity/contract/design/key 违约 |
| 3 | shard 或 per-model final 缺失/不完整，可 recovery |
| 4 | 环境、权限、存储 `OSError` 或 `sqlite3.Error` |
| 5 | 另一进程持有发布租约，本次未发布 |

## 2. 正式验收矩阵

| 验收项 | 结论 | 证据 |
|---|---|---|
| D1 调度守卫净删除 | 满足 | production source 无 finalization status/stage receipt；submitter mock 精确断言调用中二者均不存在 |
| D1 recovery 仍提交收尾链 | 满足 | recovery shell 测试断言 seed recovery 后 finalizer 与 publish 均提交 |
| D2 每次重发 | 满足 | 同输入连续 finalize/publish 的 CSV 和 manifest inode 均变化 |
| D2 无 provenance/marker/backup/residual | 满足 | production source 检索为空；manifest 无 `input_artifacts`；失败测试无 `.previous` |
| D2 缺 manifest 不完整 | 满足 | per-model CSV 存在但 manifest 缺失时 publish 抛 `SeedShardIncompleteError` 且不发布 panel |
| H1 并发唯一临时 CSV | 满足 | finalize/publish 两并发线程各取得不同 temp path，清理后无残留 |
| H1 loser 观察旧 pair | 满足 | loser 精确捕获 `OutputRunLockError` 并读到 whole-old pair，winner 后为 whole-new pair |
| 并发 loser 退出语义 | 满足 | `test_concurrent_finalize_and_publish_loser_exits_five_without_traceback`：finalize/publish 各精确得到 `[0, 5]`，stderr 只有一行租约说明 |
| 并发最终产物完整 | 满足 | 同一测试精确断言最终模型/seed keys、manifest mode 和 materialized rows |
| 删除旧 manifest 死代码 | 满足 | 两个 `pop("input_artifacts", None)` 保持不存在；上一异常分流轮次删除对应的永真测试断言 |
| D3 单 shard I/O 容忍 | 满足 | 同一次 diagnosis 精确返回 `invalid_targets=[0]` 与 `missing_master_indices=[1]` |
| D4 精确 feature-universe 键 | 满足 | unknown `definition_sha256` 被精确错误拒绝；真实 SMR manifest 可解析为 60 jobs |
| D5 definition 内容入 contract | 满足 | 修改 definition 内容前后 contract 不同，且 contract 中内容与 JSON 对象精确相等 |
| D6 删除 `resume_group` | 满足 | production/test/source 全仓库检索无结果 |
| D7 SQL key 完整性 | 满足 | missing/out-of-design/duplicate 行为测试；AST 断言 finalize/publish 无四类 resident key-set 名称 |
| D7 production publish MaxRSS | 满足 | 10 models、20M rows 完整发布；MaxRSS 837,828,608 bytes |
| 多模型 panel 完整性 | 满足 | OLS/ridge 全部模型、seed rows 均存在于最终 CSV |
| failure policy | 满足 | per-model 汇总与 panel `passed` 合取测试 |
| CLI 异常边界 | 满足 | lock=5、`OSError`/`sqlite3.Error`=4、contract=1、incomplete=3；普通 `RuntimeError`/`RecursionError` 均向外传播 |
| 数值等价 | 满足 | OLS + SuperLearner，2 seeds × 2 draws × 2 N × 2 K，见 §4 |
| descendant cleanup | 满足 | timeout/crash/grandchild/SIGKILL fallback 与 runner recovery 保持通过 |
| SuperLearner 性能证据 | 满足 | 1/2 CPU 每档三次取中位数；2 CPU 未达 +15%，按停止规则不测 4/8 |
| 连续三次全量 pytest | 满足 | 三次均 `293 passed`，见 §3 |
| 实现行数上限 | 满足 | 相对 `7a811ed` 的生产实现为 `+12/-7`，净增加 5 行 |
| Adapter schema 示例可加载 | 满足 | 两份示例只生成 `{mode, definition_file}`，全文件仅保留一处“已移除 schema_sha256 校验”的现状说明 |
| 当前 Slurm 心智模型 | 满足 | README 明确一个 task 对应一个 seed、三个资源类和两级合并 |
| recovery 文档 | 满足 | README 给出 `missing` 诊断及 `--snapshot/--resource-class/--master-indices` 恢复命令 |
| 身份字段文档 | 满足 | Inputs 与 panel 说明列出 `experiment_id`、`data_version`、`model_spec_version` |
| 本轮代码/测试不变 | 满足 | `git diff --quiet -- NK_Grid/src NK_Grid/tests` 返回 0 |
| 本轮全量 pytest | 满足 | `293 passed, 70 warnings in 53.11s` |
| 用户工作树保护 | 满足 | 精确暂存清单不含用户 README/旧 reports 删除、requirements、`AGENTS.md`、`docs/` |

本轮每处文档修改的源码依据：

| 文档修正 | 权威源码行 |
|---|---|
| `feature_universe` 仅含两个键 | `NK_Grid/src/aleatoric_nk_grid/ingest.py:198-205` |
| definition 内容进入 contract；合并逐字段比较 | `NK_Grid/src/aleatoric_nk_grid/ingest.py:275-301`；`NK_Grid/src/aleatoric_nk_grid/seed_shards.py:264-300, 805-834` |
| provenance 摘要校验已移除 | `NK_Grid/src/aleatoric_nk_grid/ingest.py:15-36, 113-329` 的完整 schema 加载路径无 provenance 读取 |
| 一个 seed 一个 Slurm task，并写入 seed shard | `NK_Grid/src/aleatoric_nk_grid/slurm_jobs.py:75-82, 142-169, 257-266` |
| 三个资源类及 CPU 参数 | `NK_Grid/src/aleatoric_nk_grid/slurm_jobs.py:85-104`；`NK_Grid/slurm/submit_nk_grid.sh:14-16, 35-41, 207-217, 273-276` |
| seed arrays → finalizer → publish 依赖链及 job name | `NK_Grid/slurm/submit_nk_grid.sh:292-310, 320-380`；`NK_Grid/slurm/finalize_seed_shards.sbatch:10-31` |
| missing 诊断、指定 master indices 和 recovery 收尾链 | `NK_Grid/src/aleatoric_nk_grid/seed_shards.py:466-537, 978-993`；`NK_Grid/slurm/submit_nk_grid.sh:231-260, 320-380` |
| 三个显式身份字段的配置与合并校验 | `NK_Grid/src/aleatoric_nk_grid/run_panels.py:77-79, 108-110, 207-223`；`NK_Grid/src/aleatoric_nk_grid/seed_shards.py:264-300, 805-834` |

## 3. 自动化测试证据

本轮只改文档，没有新增或修改测试。文档冻结后运行一次全量测试，原始结果行：

```text
293 passed, 70 warnings in 53.11s
```

命令退出码为 0，即 `293 passed / 0 failed / 0 errors`。下列三行是上一轮实现冻结时的
连续三次历史证据：

实现和测试冻结后连续执行三次 `.venv/bin/python -m pytest -q`，三轮之间没有代码、
测试或工作树变更。以下是 pytest 的原始结果行：

```text
293 passed, 70 warnings in 48.16s
293 passed, 70 warnings in 50.29s
293 passed, 70 warnings in 48.59s
```

三次命令退出码均为 0，即每轮 `293 passed / 0 failed / 0 errors`。新增测试不是
字符串宽松包含检查：并发 CLI 测试精确比较排序后的 `[0, 5]` 返回码、唯一 stderr 行、
临时路径、完整模型/seed keys、manifest mode 和 materialized rows；异常边界测试精确
断言普通 `RuntimeError`/`RecursionError` 向外传播、专用锁异常为 5、SQLite 异常为 4；
重发测试比较 inode；
D3 比较完整 diagnosis 对象；D4/D5 比较精确 contract 对象；D7 运行真实 SQLite
缺失/完整检查并用 AST 排除指定 resident sets。

本轮冻结前的针对性结果：

```text
NK_Grid/tests/test_seed_shards.py: 47 passed, 56 warnings in 6.98s
```

## 4. 数值等价

自动化 fixture 使用 OLS 与 SuperLearner，每个模型包含：

```text
2 seeds × 2 draws/seed × 2 N × 2 K = 16 rows/model
```

monolithic 与单-seed shards → per-model finalize → panel publish 按
`(model, seed, draw, N, K)` 排序比较。key、status、error、整数、字符串、NaN
位置、completion 与 failure-policy counts 相同；同为 `n_jobs=1` 时全部科学列逐位相同。

容差组保持：

```text
bounded_dimensionless: rtol=0,     atol=1e-12
r2_like_unbounded:     rtol=1e-12, atol=1e-12
scale_dependent:       rtol=1e-10, atol=1e-12
```

所有 30 个科学输出列的 OLS 与 SuperLearner max abs/max rel 均为 0。最大误差并列时，
代表 key 分别为 `OLS=(ols,11,0,20,1)` 与
`SL=(super_learner,11,0,20,1)`：

| 科学列 | 组 | OLS max abs/rel | OLS key | SL max abs/rel | SL key |
|---|---|---:|---|---:|---|
| r2_test | r2_like_unbounded | 0 / 0 | OLS | 0 / 0 | SL |
| skill_score_pct | r2_like_unbounded | 0 / 0 | OLS | 0 / 0 | SL |
| rmse | scale_dependent | 0 / 0 | OLS | 0 / 0 | SL |
| mae | scale_dependent | 0 / 0 | OLS | 0 / 0 | SL |
| medae | scale_dependent | 0 / 0 | OLS | 0 / 0 | SL |
| max_error | scale_dependent | 0 / 0 | OLS | 0 / 0 | SL |
| nrmse | r2_like_unbounded | 0 / 0 | OLS | 0 / 0 | SL |
| spearman_rho | bounded_dimensionless | 0 / 0 | OLS | 0 / 0 | SL |
| pearson_r | bounded_dimensionless | 0 / 0 | OLS | 0 / 0 | SL |
| kendall_tau | bounded_dimensionless | 0 / 0 | OLS | 0 / 0 | SL |
| ccc | bounded_dimensionless | 0 / 0 | OLS | 0 / 0 | SL |
| explained_variance | r2_like_unbounded | 0 / 0 | OLS | 0 / 0 | SL |
| mean_bias | scale_dependent | 0 / 0 | OLS | 0 / 0 | SL |
| median_bias | scale_dependent | 0 / 0 | OLS | 0 / 0 | SL |
| pinball_q10 | scale_dependent | 0 / 0 | OLS | 0 / 0 | SL |
| pinball_q90 | scale_dependent | 0 / 0 | OLS | 0 / 0 | SL |
| d2_absolute_error | r2_like_unbounded | 0 / 0 | OLS | 0 / 0 | SL |
| pinball_q05 | scale_dependent | 0 / 0 | OLS | 0 / 0 | SL |
| pinball_q25 | scale_dependent | 0 / 0 | OLS | 0 / 0 | SL |
| pinball_q50 | scale_dependent | 0 / 0 | OLS | 0 / 0 | SL |
| pinball_q75 | scale_dependent | 0 / 0 | OLS | 0 / 0 | SL |
| pinball_q95 | scale_dependent | 0 / 0 | OLS | 0 / 0 | SL |
| ks_statistic | bounded_dimensionless | 0 / 0 | OLS | 0 / 0 | SL |
| wasserstein_distance | scale_dependent | 0 / 0 | OLS | 0 / 0 | SL |
| top_decile_hit_rate | bounded_dimensionless | 0 / 0 | OLS | 0 / 0 | SL |
| bottom_decile_hit_rate | bounded_dimensionless | 0 / 0 | OLS | 0 / 0 | SL |
| rsr | r2_like_unbounded | 0 / 0 | OLS | 0 / 0 | SL |
| cv_rmse | r2_like_unbounded | 0 / 0 | OLS | 0 / 0 | SL |
| mase | r2_like_unbounded | 0 / 0 | OLS | 0 / 0 | SL |
| pearson_r2 | bounded_dimensionless | 0 / 0 | OLS | 0 / 0 | SL |

## 5. 性能证据与需审查确认的问题

### 5.1 production panel publish

真实调用 `publish_panel()`，使用 SMR 的 10 个模型和 production cardinality：

```text
100 seeds × 50 draws × 20 N × 20 K × 10 models
= 2,000,000 rows/model
= 20,000,000 panel rows
```

完整原始结果：

```text
BENCHMARK_JSON={"benchmark": "production_panel_publish", "elapsed_seconds": 208.25895962503273, "max_rss_bytes": 837828608, "models": 10, "panel_rows": 20000000, "preset": "production", "published_csv_bytes": 521000028, "rows_per_model": 2000000, "total_cpu_seconds": 191.794924}
```

即墙钟 208.259 s、CPU 191.795 s、MaxRSS 837,828,608 bytes（799.02 MiB），最终
CSV 521,000,028 bytes。输入是完整 20M cardinality 的合成 per-model finals，不是
缩小规模 proxy。基准成功后删除自身唯一 `/private/tmp/nk-publish-production-*`
目录，未触碰用户数据。

外层 `/usr/bin/time -l` 在子进程成功、JSON 输出和清理完成后，因为受限环境不允许
`sysctl kern.clockrate` 自身返回 1；MaxRSS 来自独立 publish 子进程内部的
`resource.getrusage(RUSAGE_SELF)`，不使用该失败 wrapper 的统计。

### 5.2 既有 SMR/SuperLearner 证据

只读 SMR ARD 的既有实测保持有效：

| 模式 | MaxRSS | Elapsed |
|---|---:|---:|
| retained old 10-seed worker | 2.104 GiB | 3.204 s |
| new one-seed worker（10-seed repeat plan） | 1.322 GiB | 2.640 s |
| new one-seed worker（100-seed repeat plan） | 1.367 GiB | 3.545 s |

10→100 repeat plan 的单 worker MaxRSS 增长 3.385%；retained baseline 到新 worker
下降 37.165%。SuperLearner production params、8 cells、每档三次：

| CPUs | runs | median elapsed | median TotalCPU | median cells/hour | median MaxRSS |
|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 12.428 s | 12.321 s | 2,317.27 | 1.689 GB |
| 2 | 3 | 14.054 s | 18.406 s | 2,049.21 | 1.792 GB |
| 4/8 | 未运行 | 2 CPU 中位吞吐比 1 CPU 低 11.568%，按 §11.3 停止 | — | — | — |

### 5.3 问题与明确放弃的保证

1. D2 要求先 rename CSV、再 rename manifest，同时禁止 marker、备份和新检查。若目标
   原本已有旧 manifest，进程在第一个 rename 后崩溃，会形成“新 CSV + 旧 manifest”；
   现有 manifest 没有 CSV generation 标识，因此下游无法区分它与 matching pair。
   本轮按方案没有新增标识或探测，明确放弃“该特定崩溃窗口能被自动识别”的保证；显式
   重跑 finalize/publish 会重新生成 pair。请审查确认这是否是 D2 可接受的代价。
2. D5 按原文只把 feature-universe definition 内容纳入 semantic contract。provenance
   中旧 `schema_sha256` 的一致性检查没有恢复，因为恢复它会违反“不新增摘要机制”，而
   schema 的行为字段已进入 semantic contract。本轮明确放弃“仅凭 provenance hash
   检测 schema 文件字节级替换/损坏”的保证；请确认是否接受以 semantic contract
   作为唯一行为合同。

本轮异常分流没有新增偏离方案之处或待澄清问题；专用码选择 5，并已同步本地 §9.2 表格。

### 5.4 本轮文档核实发现

请求以 `4a26b58` 为基线，但当前分支为 `2bea830`；两者只有用户已核实的根
`README.md` 退出码说明不同，本轮保留该提交且未修改根 README。另一个代码事实是
`run_panels.py:77-79` 仍提供三个身份字段的开发默认值，因此解析器不会仅因 panel
manifest 省略它们而立即失败；本轮文档按真实研究运行要求写为“显式声明”，没有虚构
不存在的拒绝路径。

## 6. 静态检查与工作树保护

以下检查通过：

```text
.venv/bin/python -m py_compile <全部本轮 Python 实现>
git diff --check
production forbidden-name rg: no output
feature_universe definition_sha256 rg in schemas/adapters: no output
SMR real manifest resolution: 60 jobs
```

相对 `b1d4534` 的生产实现统计命令限定为 `NK_Grid/src`、`NK_Grid/slurm`、
`SMR/adapter`、`FFCWS/adapter`：

```text
production additions=122 deletions=566 net=-444
current-round additions=12 deletions=7 net=+5
```

本轮未修改 `SMR/requirements.txt` 或 `FFCWS/requirements.txt`。提交只精确暂存
`Adapter/README.md`、`Adapter/ADAPTER.md`、`NK_Grid/README.md` 与本报告；不包含根
`README.md`、任何实现或测试，也不包含用户删除的
`SMR/README.md`、`FFCWS/README.md`、`reports/cell-centric-execution.md`、
`reports/remove-bart.md`，也不包含未跟踪 `AGENTS.md`、`docs/`。

## 7. 未满足项与合并判断

D1–D7、README、production MaxRSS、数值等价、SuperLearner 三次中位数证据和连续三次
全量 pytest 均满足。本轮并发 loser 的退出码合同已变为专用非零码 5，普通
`RuntimeError`/`RecursionError` 不再被吞掉，SQLite 异常返回 4；实现相对 `7a811ed`
净增加 5 行，没有新增协调、重试、探测或诊断机制。

本轮文档修正已满足：请重点审查 Adapter 示例是否已彻底移除失效摘要字段，以及
`NK_Grid/README.md` 对 seed 任务单位、三级链和 recovery 起点的描述是否与上述源码行
一致。没有修改根 README、实现或测试。

没有把第 5.3 节两项放弃保证伪装为已满足；它们属于方案在禁止额外机制后留下的语义代价，
需要审查者确认。除这两项待确认问题外，本轮没有已知未满足的 §16 验收项。本报告不代替
审查批准；分支提交后停下，等待审查。
