# no-hash-seed-sharding-superlearner

- 日期：2026-07-29
- 分支：`codex/no-hash-seed-sharding-superlearner`
- 正式实施规范：`plans/no-hash-seed-sharding-completion.md`

结论：§2.2 的 H1/H2、四项非阻塞修正、严格测试和 SuperLearner 三次中位数
复测均已完成；最终全量 pytest 连续三轮 `290 passed`。但 production preset 的
20,000,000-row panel publish 在本机 16 GB 内存与有限 swap/磁盘条件下无法安全跑完，
没有取得完成态 MaxRSS；此前私有 SMR 10-seed 可复核基线的 MaxRSS 降幅 37.16%
也仍未达到候选 70% 门槛。因此本分支仍标记为**不可合并**。

## 1. 实现结果

- `SlurmJob.seed/draws/final_out` 保持必填；seed shards 写确定性精确路径，
  per-model final 写 `.seed-final/<panel>/<model>.csv`，不会再由同 panel 的模型互相覆盖。
- 发布为 seed chunks → per-model finalizer → per-panel publish 三级链。finalizer/publish
  都有独立 map、明确资源、`afterany` dependency、output lock、等价结果 no-op 和退出码合同。
- finalizer 严格验证 master seed/draw/output、explicit identity、semantic contract、N/K grid、
  completion、CSV schema、完整 cell key、status 和全量 failure policy；diagnostics 不再错误沿用
  单个 shard。
- CSV 与 manifest 都先写临时文件；锁内发布使用 marker 和旧 pair 回滚。失败保留全部 shards，
  interrupted marker 不会被判断为 complete。
- finalizer/publish 的待发布 CSV 改为 target 同目录 `mkstemp` 唯一文件；并发调用不再
  共享固定临时路径，并发 loser 只能观察到完整旧 pair，winner 发布后为完整新 pair。
- final/panel manifest 记录每个输入 artifact 的路径、`updated_at` 与行数；输入 key
  不变但内容按新 `updated_at` 替换时拒绝 no-op。`missing` JSON 同时报告当前 snapshot
  target 上残留的 `.previous` / `.publishing` 文件。
- CLI 将 contract violation、incomplete input 和 `OSError` 分别映射为退出码 1、3、4。
- submitter 使用 `parallel/serial/super_learner` 三类互斥完备分区，SuperLearner 有独立 CPU
  参数；MaxArraySize 可自动探测或显式覆盖；array task 使用 chunk-local 连续 indices，同类
  chunks 用 `afterany` 串联。
- recovery 只接受 machine-readable diagnosis 或显式 master indices，并再次过滤已完整
  shards；`DependencyNeverSatisfied` 诊断和可选清理由本 snapshot 的 receipts 限定 job IDs。
- finalizer/publish 各自写 snapshot-scoped stage receipt。recovery 在任何 `sbatch`
  之前用 receipt job IDs 查询 `squeue`；发现 finalizer/publish 非终态即整体拒绝，因此
  不会提交第二套并存链。两级资源与 worker 分离：finalizer 默认 `16G/1-00:00:00`，
  publish 默认 `32G/2-00:00:00`，且各有独立 CLI override。
- native worker 的 timeout/crash 都清理真实 grandchild，SIGTERM grace 后覆盖 SIGKILL fallback，
  runner 随后可恢复工作。
- NKGRID production source 未引入摘要计算或摘要值比较；legacy digest 字段仅按键存在性拒绝。

对应方案 §9.2 的 CLI 退出合同：

| 退出码 | 含义 | recovery 含义 |
|---:|---|---|
| 0 | 完成或等价 no-op | 不重试 |
| 1 | identity/contract/design/key 违约 | 不自动重试，人工修正输入 |
| 3 | shard/per-model final 缺失或未完成 | 可按 diagnosis 重交 |
| 4 | `OSError` 环境/存储故障 | 不伪装为 contract 违约；修复环境后再重试 |

## 2. 正式验收矩阵

| 验收项 | 结论 | 证据 |
|---|---|---|
| R1：多模型 panel 不覆盖 | 满足 | `test_per_model_final_outputs_are_unique_within_one_panel`、`test_multi_model_panel_publication_preserves_every_model` |
| R2：submitter mock 无裸 `return` 死代码 | 满足 | 三类 CPU、local array、receipt、snapshot、partial submission/recovery 断言均实际执行 |
| R3：`exact_output_path` 三用例 | 满足 | restriction、deterministic resume、identity mismatch 三项测试 |
| R4：descendant race 修复 | 满足 | helper 在触发 timeout 前确认 grandchild PID；最终连续三轮全量 pytest 见 §3 |
| 三资源类与独立 SuperLearner CPU | 满足 | Python partition、shell submitter、独立 receipt |
| MaxArraySize 与 chunk-local arrays | 满足 | 自动探测成功/失败/显式覆盖、0/1/M/M+1/sparse、local 越界 |
| chunk dependency/空类/partial submission | 满足 | 同类 `afterany`、空类省略、无 recovery jobs 报错、finalizer 失败阻止 publish |
| receipt 限定的 dependency diagnosis | 满足 | 只识别/清理当前 snapshot receipt job IDs |
| H1：唯一临时 CSV 与并发 pair 完整性 | 满足 | finalize/publish 并发各保留两个不同 temp path；loser 精确读取 whole-old pair，winner 后为 whole-new pair |
| H2：recovery 不产生并存 finalization 链 | 满足 | stage receipt 精确记录 stage/job/dependency/resource；active finalizer+publish 时 recovery 在 `sbatch` 前拒绝且提交计数为 0 |
| no-op 输入内容来源 | 满足 | final/panel manifest 记录 path/`updated_at`/rows；changed shard + unchanged key set 强制重发 |
| CLI 环境故障退出码 | 满足 | contract=1、incomplete=3、`OSError`=4，测试精确断言 stderr 与返回值 |
| 发布残留诊断 | 满足 | `missing.publication_residuals` 精确列出 snapshot target 的 CSV/manifest `.previous` 与 `.publishing` |
| finalizer/publish 独立资源 | 满足 | worker、finalizer、publish 的 sbatch argv 与两份 stage receipts 均精确断言不同 memory/time |
| deterministic shard/recovery | 满足 | exact path、resume、missing/incomplete-only recovery、显式 indices 再过滤 |
| finalizer/publish 严格验证 | 满足 | 42 个 seed-shard 测试覆盖 master、manifest、key、policy、atomic/no-op/concurrency/CLI |
| multi-model failure policy | 满足 | per-model 阈值判定和 panel `passed` 合取 |
| monolithic 与 shard+merge 数值等价 | 满足 | 2 seeds × 2 draws × 2 N × 2 K，OLS + SuperLearner，见 §4 |
| timeout/crash descendant cleanup | 满足 | 8 个 native-process 测试 |
| SMR 10→100 seeds 单 worker MaxRSS 增幅 ≤10% | 满足 | 1.322→1.367 GiB，+3.385%，`split_frame()` 均为 1 |
| 可复核旧基线到新 worker MaxRSS 下降 ≥70% | **未满足** | 2.104→1.322 GiB，下降 37.165% |
| SuperLearner 每档 ≥3 次取中位数 | 满足 | 1/2 CPU 各三次均 8/8 `ok`；2 CPU 中位吞吐下降 11.57%，按停止规则不继续 4/8 |
| production preset panel publish MaxRSS | **未满足** | 10 models、20M rows 真实运行使 16 GB 主机扩张 swap、系统卷仅余 357 MiB；为避免写满磁盘终止，未取得完成态 MaxRSS |
| production source 无摘要实现/调用 | 满足 | §6 静态命令无输出 |

## 3. 自动化测试证据

最终代码状态连续执行三轮：

```text
290 passed, 70 warnings in 48.68s
exit=0
290 passed, 70 warnings in 48.70s
exit=0
290 passed, 70 warnings in 48.09s
exit=0
```

三轮均为 `290 passed, 0 failed, 0 errors`，测试数量高于 218。每轮前等待 15 秒，
用于释放前一 pytest 进程和 production publish 压测后的 swap 压力；三轮之间没有代码
或工作树变更。

Targeted evidence：

```text
NK_Grid/tests/test_slurm_jobs.py: 74 passed
NK_Grid/tests/test_seed_shards.py: 42 passed
NK_Grid/tests/test_nk_grid_engine.py + test_native_process.py:
  included in 160 passed targeted run
§2.2 directly changed targeted total: 116 passed, 0 failed, 0 errors
```

## 4. 数值等价

自动化 fixture 使用 OLS 与 SuperLearner，每个模型包含：

```text
2 seeds × 2 draws/seed × 2 N × 2 K = 16 rows/model
```

旧式基线为每个 `(panel, model)` 一次多-seed monolithic 运行；新路径为单-seed shards、
per-model finalize、per-panel publish。按 `(model, seed, draw, N, K)` 排序后：

- key、status、error、整数、字符串字段逐位相同。
- NaN 位置逐列相同。
- completion 与 failure-policy counts 相同。
- 同为 `n_jobs=1`，两个模型全部科学列逐位相同。

容差组：

```text
bounded_dimensionless: rtol=0,     atol=1e-12
r2_like_unbounded:     rtol=1e-12, atol=1e-12
scale_dependent:       rtol=1e-10, atol=1e-12
```

下表的 representative key 在最大误差并列为 0 时取排序后的第一个 cell；
`OLS` key 为 `(ols,11,0,20,1)`，`SL` key 为 `(super_learner,11,0,20,1)`。

| 科学列 | 组 | OLS max abs | OLS max rel | OLS key | SL max abs | SL max rel | SL key |
|---|---|---:|---:|---|---:|---:|---|
| r2_test | r2_like_unbounded | 0 | 0 | OLS | 0 | 0 | SL |
| skill_score_pct | r2_like_unbounded | 0 | 0 | OLS | 0 | 0 | SL |
| rmse | scale_dependent | 0 | 0 | OLS | 0 | 0 | SL |
| mae | scale_dependent | 0 | 0 | OLS | 0 | 0 | SL |
| medae | scale_dependent | 0 | 0 | OLS | 0 | 0 | SL |
| max_error | scale_dependent | 0 | 0 | OLS | 0 | 0 | SL |
| nrmse | r2_like_unbounded | 0 | 0 | OLS | 0 | 0 | SL |
| spearman_rho | bounded_dimensionless | 0 | 0 | OLS | 0 | 0 | SL |
| pearson_r | bounded_dimensionless | 0 | 0 | OLS | 0 | 0 | SL |
| kendall_tau | bounded_dimensionless | 0 | 0 | OLS | 0 | 0 | SL |
| ccc | bounded_dimensionless | 0 | 0 | OLS | 0 | 0 | SL |
| explained_variance | r2_like_unbounded | 0 | 0 | OLS | 0 | 0 | SL |
| mean_bias | scale_dependent | 0 | 0 | OLS | 0 | 0 | SL |
| median_bias | scale_dependent | 0 | 0 | OLS | 0 | 0 | SL |
| pinball_q10 | scale_dependent | 0 | 0 | OLS | 0 | 0 | SL |
| pinball_q90 | scale_dependent | 0 | 0 | OLS | 0 | 0 | SL |
| d2_absolute_error | r2_like_unbounded | 0 | 0 | OLS | 0 | 0 | SL |
| pinball_q05 | scale_dependent | 0 | 0 | OLS | 0 | 0 | SL |
| pinball_q25 | scale_dependent | 0 | 0 | OLS | 0 | 0 | SL |
| pinball_q50 | scale_dependent | 0 | 0 | OLS | 0 | 0 | SL |
| pinball_q75 | scale_dependent | 0 | 0 | OLS | 0 | 0 | SL |
| pinball_q95 | scale_dependent | 0 | 0 | OLS | 0 | 0 | SL |
| ks_statistic | bounded_dimensionless | 0 | 0 | OLS | 0 | 0 | SL |
| wasserstein_distance | scale_dependent | 0 | 0 | OLS | 0 | 0 | SL |
| top_decile_hit_rate | bounded_dimensionless | 0 | 0 | OLS | 0 | 0 | SL |
| bottom_decile_hit_rate | bounded_dimensionless | 0 | 0 | OLS | 0 | 0 | SL |
| rsr | r2_like_unbounded | 0 | 0 | OLS | 0 | 0 | SL |
| cv_rmse | r2_like_unbounded | 0 | 0 | OLS | 0 | 0 | SL |
| mase | r2_like_unbounded | 0 | 0 | OLS | 0 | 0 | SL |
| pearson_r2 | bounded_dimensionless | 0 | 0 | OLS | 0 | 0 | SL |

私有 SMR 的 1 CPU 与有效 2 CPU SuperLearner 结果也逐位相同：8/8 rows 为 `ok`，
key/status/error 相同，以上 30 个科学列的 max abs/max rel 均为 0。

## 5. 私有数据性能验收

输入为只读的 `SMR/data/ard/asample2_withlag/data.csv`（7,463 rows、4,252 predictors），
输出全部写入 `/private/tmp`；未修改 SMR/FFCWS 私有数据。

### 5.1 SMR MaxRSS

相同 OLS panel/model，固定 `N=[10]`、`K=[1]`、每 seed 一个 draw/cell：

| 模式 | repeat-plan seeds | 实际 worker seeds | `split_frame()` | MaxRSS | Elapsed |
|---|---:|---:|---:|---:|---:|
| retained old multi-seed worker | 10 | 10 | 10 | 2.104 GiB | 3.204 s |
| new one-seed worker | 10 | 1 | 1 | 1.322 GiB | 2.640 s |
| new one-seed worker | 100 | 1 | 1 | 1.367 GiB | 3.545 s |

结论：

- 新 worker 的 `split_frame()` 恰好一次。
- repeat plan 10→100 时单 worker MaxRSS 增长 3.385%，满足 ≤10%，无 OOM。
- 本次可复核 10-seed retained baseline 到新 worker 下降 37.165%，未达到候选 70%。
  旧讨论中的 17–41 GB 没有对应 job IDs/`sacct` 证据，未用作计算基线。

### 5.2 SuperLearner CPU

私有 SMR，同一 seed/cells：2 draws × 2 N × 2 K = 8 cells，production model params。
每档独立运行三次；2 CPU 使用允许 loky 创建 worker process 的执行环境。每次先验证
8/8 rows 为 `ok`；1/2 CPU 的 key/status、NaN 位置相同，30 个科学列 max abs=0。

| CPUs | repeat | Elapsed | TotalCPU | CPU efficiency | MaxRSS | cells/hour | core-hours/1,000 cells |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 13.579 s | 12.741 s | 93.83% | 1.521 GB | 2,120.96 | 0.4715 |
| 1 | 2 | 12.428 s | 12.321 s | 99.13% | 1.689 GB | 2,317.27 | 0.4315 |
| 1 | 3 | 12.298 s | 12.226 s | 99.41% | 1.791 GB | 2,341.87 | 0.4270 |
| **1 median** | — | **12.428 s** | **12.321 s** | **99.13%** | **1.689 GB** | **2,317.27** | **0.4315** |
| 2 | 1 | 14.054 s | 18.406 s | 65.48% | 1.791 GB | 2,049.21 | 0.9760 |
| 2 | 2 | 13.834 s | 18.255 s | 65.98% | 1.794 GB | 2,081.80 | 0.9607 |
| 2 | 3 | 14.286 s | 18.531 s | 64.86% | 1.792 GB | 2,016.02 | 0.9921 |
| **2 median** | — | **14.054 s** | **18.406 s** | **65.48%** | **1.792 GB** | **2,049.21** | **0.9760** |
| 4 | — | 按停止规则不运行 | — | — | — | — | — |
| 8 | — | 按停止规则不运行 | — | — | — | — | — |

2 CPU 的中位 cells/hour 相对 1 CPU 为 -11.568%，小于 +15% 阈值；按 §11.3 停止规则，
不继续增加到 4/8 CPU。MaxRSS 未接近 48G 申请内存的 80%。生产默认保持
`--super-learner-cpus-per-task 1`。

### 5.3 production panel publish MaxRSS

使用真实 `publish_panel()`，SMR 的 10 个模型和 production cardinality：

```text
100 seeds × 50 draws × 20 N × 20 K × 10 models
= 2,000,000 rows/model
= 20,000,000 panel rows
```

输入为只含发布合同必需列的合成 per-model final CSV（合计约 525 MiB），以隔离 publish
阶段；不是缩小规模的 proxy。运行过程中 benchmark 目录增长到约 3.3 GiB，同时 16 GB
Mac 因 O(cells) key sets 产生内存压力并扩张 swap，系统卷可用空间从约 4.9 GiB 降到
357 MiB。为避免写满用户系统卷，在完成前终止，并删除唯一的
`/private/tmp/nk-publish-production-*` benchmark 目录；清理后可用空间恢复到 3.7 GiB。

结论：**未取得完成态 MaxRSS，§2.2 该项未满足。** 不能把终止前目录大小、swap 或
缩小规模的 RSS 当作 production panel publish MaxRSS。独立 publish 默认暂设
`32G/2-00:00:00`，但在高内存节点完成同规模实测前不能据此批准生产资源配置。

### 5.4 技术问题与需审查确认的歧义

1. production 20M-row publish 的完成态 MaxRSS 需要高内存且有充足 local scratch 的节点。
   请确认下一轮是否在集群节点复测，以及允许的最低内存/scratch 额度；本轮没有自行降低
   production cardinality，也没有把估算值写成实测。
2. H2 选择“发现 receipt-owned 非终态 finalizer/publish 就拒绝 recovery”。新提交链都有
   stage receipts；但 `9dc0046` 旧版本已经提交的链没有 stage receipt，代码无法从 snapshot
   反推出它的 job IDs。请确认是否要对“snapshot 有 worker receipts、但完全没有 stage
   receipts”的 legacy recovery 一律拒绝；这也会拒绝“worker 部分提交失败、从未交过
   finalizer”的合法 recovery，因此本轮没有自行改变该语义。
3. §2.2 要求独立默认值但没有指定额度。本轮取 finalizer `16G/1 day`、publish
   `32G/2 days` 并允许独立覆盖；publish 完成态 MaxRSS 未取得，以上值仍需集群实测确认。
4. §2.2 的明确验收要求是 `missing` JSON “列出” `.previous` / `.publishing` 残留，
   同段前文又使用“诊断与清理”措辞。为避免擅自删除可能用于人工恢复的 old pair，本轮只
   实现报告、不实现自动清理命令。请确认后续是否需要 receipt/marker-aware 的显式清理子命令。

## 6. 静态与工作树保护

以下命令无输出：

```bash
rg -n \
  '(^|[[:space:]])(import hashlib|from hashlib)|hashlib\.|file_sha256|semantic_sha256' \
  NK_Grid/src
```

另外通过：

```text
bash -n NK_Grid/slurm/submit_nk_grid.sh
bash -n NK_Grid/slurm/finalize_seed_shards.sbatch
.venv/bin/python -m compileall -q NK_Grid/src
git diff --check
```

实现提交只精确暂存 NKGRID 实现、测试和本报告；不包含用户对 SMR/FFCWS README、
requirements、旧 reports 的删除，也不包含未跟踪 `AGENTS.md`/`docs/`。

## 7. 未满足项与合并判断

未满足：

1. 本次可复核 retained 10-seed 基线的 MaxRSS 降幅为 37.165%，不是 70%。
2. production preset 的 20,000,000-row panel publish 在本机 16 GB 内存与有限
   swap/scratch 下无法安全完成，未取得完成态 MaxRSS；详见 §5.3。

SuperLearner 的证据要求已满足：1/2 CPU 每档三次取中位数，2 CPU 吞吐下降后按停止规则
不继续 4/8，不再把“未跑 4/8”列为未满足。

因此，尽管 §2.2 代码、严格测试、数值等价、连续三轮 pytest 和调度链均完成，本报告
仍不把分支标记为可合并。
