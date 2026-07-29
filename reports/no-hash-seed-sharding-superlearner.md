# no-hash-seed-sharding-superlearner

- 日期：2026-07-29
- 分支：`codex/no-hash-seed-sharding-superlearner`
- 正式实施规范：`plans/no-hash-seed-sharding-completion.md`

结论：工程实现和自动化验收满足；私有 SMR 10-seed 可复核基线的 MaxRSS
下降为 37.16%，未达到候选 70% 门槛；SuperLearner 在 2 CPU 档已触发停止规则，
未形成完整 1/2/4/8 扫描。因此本分支按规范仍标记为**不可合并**。

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
- submitter 使用 `parallel/serial/super_learner` 三类互斥完备分区，SuperLearner 有独立 CPU
  参数；MaxArraySize 可自动探测或显式覆盖；array task 使用 chunk-local 连续 indices，同类
  chunks 用 `afterany` 串联。
- recovery 只接受 machine-readable diagnosis 或显式 master indices，并再次过滤已完整
  shards；`DependencyNeverSatisfied` 诊断和可选清理由本 snapshot 的 receipts 限定 job IDs。
- native worker 的 timeout/crash 都清理真实 grandchild，SIGTERM grace 后覆盖 SIGKILL fallback，
  runner 随后可恢复工作。
- NKGRID production source 未引入摘要计算或摘要值比较；legacy digest 字段仅按键存在性拒绝。

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
| deterministic shard/recovery | 满足 | exact path、resume、missing/incomplete-only recovery、显式 indices 再过滤 |
| finalizer/publish 严格验证 | 满足 | 36 个 seed-shard 测试覆盖 master、manifest、key、policy、atomic/no-op/CLI |
| multi-model failure policy | 满足 | per-model 阈值判定和 panel `passed` 合取 |
| monolithic 与 shard+merge 数值等价 | 满足 | 2 seeds × 2 draws × 2 N × 2 K，OLS + SuperLearner，见 §4 |
| timeout/crash descendant cleanup | 满足 | 8 个 native-process 测试 |
| SMR 10→100 seeds 单 worker MaxRSS 增幅 ≤10% | 满足 | 1.322→1.367 GiB，+3.385%，`split_frame()` 均为 1 |
| 可复核旧基线到新 worker MaxRSS 下降 ≥70% | **未满足** | 2.104→1.322 GiB，下降 37.165% |
| SuperLearner 完整 1/2/4/8 CPU 扫描 | **未满足** | 2 CPU 比 1 CPU cells/hour 下降 7.478%，按停止规则未继续 4/8 |
| production source 无摘要实现/调用 | 满足 | §6 静态命令无输出 |

## 3. 自动化测试证据

最终代码状态连续执行三轮：

```text
282 passed, 70 warnings in 87.29s (0:01:27)
exit=0
282 passed, 70 warnings in 72.13s (0:01:12)
exit=0
282 passed, 70 warnings in 54.78s
exit=0
```

三轮均为 `282 passed, 0 failed, 0 errors`，测试数量高于 218。

Targeted evidence：

```text
NK_Grid/tests/test_slurm_jobs.py: 72 passed
NK_Grid/tests/test_seed_shards.py: 36 passed
NK_Grid/tests/test_nk_grid_engine.py + test_native_process.py:
  included in 160 passed targeted run
targeted total: 160 passed, 0 failed, 0 errors
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

| CPUs | Elapsed | TotalCPU | CPU efficiency | MaxRSS | cells/hour | core-hours/1,000 cells |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 13.356 s | 12.898 s | 96.57% | 1.496 GB | 2,156.30 | 0.4638 |
| 2 | 14.436 s | 18.410 s | 63.76% | 1.480 GB | 1,995.05 | 1.0025 |
| 4 | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 |
| 8 | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 |

2 CPU 的 cells/hour 相对 1 CPU 为 -7.478%，小于 +15% 阈值；按 §11.3 停止规则，
不继续增加到 4/8 CPU。MaxRSS 未接近 48G 申请内存的 80%。生产默认保持
`--super-learner-cpus-per-task 1`。

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
2. SuperLearner 在 2 CPU 已触发停止规则，因而没有 4/8 CPU 实测数字；完整
   1/2/4/8 扫描未完成。

因此，尽管代码、严格测试、数值等价、三轮 pytest 和调度链均完成，本报告不把分支
标记为可合并。
