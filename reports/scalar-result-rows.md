# scalar-result-rows 实现报告

## 1. 改动清单

- `NK_Grid/src/aleatoric_nk_grid/nk_grid.py`
  - 新增 `ROW_METADATA_FIELDS`，显式定义允许进入结果行的七个标量 metadata 字段。
  - 在 `_run_nk_grid_locked()` 构造一次 `row_metadata`，两个 `add_metadata()` 调用点只接收该对象；完整 `metadata` 仍原样传给 `_manifest_payload()`。
- `NK_Grid/src/aleatoric_nk_grid/model_registry.py`
  - `AdaptiveStackingRegressor.fit()` 的 LightGBM base pipeline 将 median imputer 输出设为 pandas。
  - `AdaptiveStackingClassifier.fit()` 做同样修改，使 LightGBM 的 fit/predict 容器和特征名契约一致。
- `NK_Grid/tests/test_scalar_result_rows.py`
  - 新增大合同两阶段运行测试和 definition 改变后的 resume 拒绝测试，覆盖 T1/T2。
- `NK_Grid/tests/test_super_learner_feature_names.py`
  - 新增回归/分类 SuperLearner warning 边界测试，以及 ndarray/pandas imputer 的自包含数值等价测试，覆盖 T3/T4。

实现分为两个独立 commit：

1. `afd333e Keep artifact contracts out of result rows`
2. `388aa9c Preserve LightGBM feature names in SuperLearner`

## 2. 验收标准逐条核对

| # | 验收标准 | 结论 | 证据 |
|---|---|---|---|
| 1 | part 与最终 CSV 不含两个嵌套列；writer row 的 metadata 键恰等于 allowlist，且无容器值 | 满足 | `test_large_contract_stays_in_manifest_not_result_rows` 检查 part/final header、实际 writer 入参的键集合及所有值类型；20 行均通过 |
| 2 | manifest 保留完整 `identity` / `semantic_contract`，合同继续传到 final/panel | 满足 | T1 对暂停前后 manifest 的两个完整对象做等值断言，并将 manifest 的 feature universe 与 schema 逐字段比较；既有 `test_monolithic_and_seed_shard_publication_are_numerically_equivalent` 通过 |
| 3 | definition 改变但 identity 不变时 resume 仍拒绝 | 满足 | `test_definition_change_with_same_identity_is_rejected_on_resume` 命中 `semantic_contract.feature_universe.definition.audit_features` 的精确字段路径，且确认 identity 未变 |
| 4 | 大于 128 KiB 合同的最小 panel 可端到端完成 | 满足 | 263,352-byte definition、20 行 panel 暂停后恢复完成；修复前恢复明确报 `field larger than field limit (131072)`，修复后完成并删除 `.parts` |
| 5 | 回归与分类目标 LightGBM warning 各为 0 | 满足 | `test_super_learner_lightgbm_uses_feature_names_consistently[regression/classification]`；修复前分别捕获 3/4 条，修复后均为 0 |
| 6 | R2 前后预测、概率与派生 metrics 逐元素等价 | 满足 | `test_lightgbm_pandas_imputer_is_numerically_equivalent[regression/classification]` 使用 `rtol=0, atol=0`；预测及概率最大绝对差为 0.0，MSE、accuracy、log-loss 相等 |
| 7 | 全量 pytest ≥293 passed 且 0 failed | 满足 | `.venv/bin/python -m pytest -q`：`299 passed, 64 warnings in 50.60s`，0 failed |
| 8 | 全量 warning summary 不出现两条完整 feature-name 消息 | 满足 | 全量 summary 仅有 8 条 MLP `ConvergenceWarning` 和 56 条 SciPy fallback `RuntimeWarning`；目标 `LGBMRegressor` / `LGBMClassifier` 消息均未出现 |
| 9 | 报告包含修改前后体积实测 | 满足 | 见第 3 节：同一 20-row fixture 的 part 从 5,650,618 B 降至 16,516 B，减少 99.7077% |

## 3. 实测数字

环境：Apple arm64，macOS 26.5.2，Python 3.14.3，scikit-learn 1.8.0，LightGBM 4.6.0，pandas 3.0.2。fixture 使用 40 行训练数据、2 个实际预测变量、1,100 个 `keep=false` 审计字段；`feature_universe.json` 为 263,352 bytes，运行 20 个 draw，结果为 20 行。

### 3.1 checkpoint / final 体积

| 项目 | 修改前 | 修改后 |
|---|---:|---:|
| checkpoint part bytes（20 行） | 5,650,618 B | 16,516 B |
| checkpoint bytes/row | 282,530.90 B | 825.80 B |
| checkpoint 降幅 | — | 99.7077%（缩小 342.13 倍） |
| final CSV bytes（20 行） | 无法物化：命中 131,072-byte field limit | 14,323 B |
| final CSV bytes/row | 无法物化 | 716.15 B |

按 2,000,000 行/model 线性外推：

- 修改前 checkpoint/CSV 下界：565,061,800,000 B，约 565.06 GB/model；最终 CSV 在此之前已因 field limit 失败。
- 修改后 checkpoint：1,651,600,000 B，约 1.65 GB/model。
- 修改后 final CSV：1,432,300,000 B，约 1.43 GB/model。

该外推只反映本合成 fixture 的行宽，不替代 production 各 schema 的正式 pilot 测量。

### 3.2 marker 与列实测

| 产物 | 修改前 semantic marker | 修改前 identity marker | 修改后 semantic marker | 修改后 identity marker |
|---|---:|---:|---:|---:|
| checkpoint part | 20 | 20 | 0 | 0 |
| final CSV | 不适用（物化失败） | 不适用（物化失败） | 0 | 0 |

semantic marker 为 definition 中唯一的 `scalar_result_rows_contract_marker`；identity marker 使用只存在于完整 identity 对象中的 `test-data-v1`。修改后 part/final header 同时确认不含 `semantic_contract`、`identity`。

### 3.3 R2 warning 与数值

| 路径 | 修改前目标 warning | 修改后目标 warning | 最大绝对数值差 |
|---|---:|---:|---:|
| SuperLearner 回归 | 3 | 0 | 0.0（predict） |
| SuperLearner 分类 | 4 | 0 | 0.0（predict / predict_proba） |

## 4. 测试证据

新增测试与方案映射：

- T1：`test_large_contract_stays_in_manifest_not_result_rows`
  - 两阶段运行；检查 part、writer row、final CSV、manifest 完整对象和完成后清理。
- T2：`test_definition_change_with_same_identity_is_rejected_on_resume`
  - 同 identity 下修改 audit definition，同时重建合法 canonical definition，确认 sidecar 逐字段比较拒绝 resume。
- T3：`test_super_learner_lightgbm_uses_feature_names_consistently`
  - `regression` / `classification` 两个参数实例；`n_jobs=1`，按 `UserWarning` 类和完整消息精确计数。
- T4：`test_lightgbm_pandas_imputer_is_numerically_equivalent`
  - `regression` / `classification` 两个参数实例；旧路径用 `pytest.warns()` 精确捕获完整消息，新旧结果以零容差比较。

定向回归：

```text
17 passed, 64 warnings in 8.05s
```

该命令包含 `test_prior_invariance.py`、既有 monolithic ↔ seed-shard 等价测试、T1–T4。目标 LightGBM warning 未出现在 summary。

全量回归：

```text
299 passed, 64 warnings in 50.60s
```

未跳过测试，无失败、无 error。64 条 warning 为 8 条既有 MLP convergence warning 和 56 条既有 SciPy `ks_2samp` fallback warning。

## 5. 偏离方案之处与待澄清问题

生产代码无偏离。

测试 fixture 采用 1,100 个 `keep=false` audit feature 放大 definition，而只保留 `X_a` / `X_b` 两个实际模型输入。这样合同真实超过 200 KiB，同时避免把 T1 变成高维模型性能测试；不改变方案验证的序列化路径。

本机没有 `python` 命令，因此测试命令使用语义等价的 `.venv/bin/python -m pytest -q`。无待澄清问题。

## 6. 未覆盖与已知风险

- 未执行集群干净切换；上线仍须严格完成方案第 7 节的取消旧作业、清空活动输出根和禁止共享 checkout 热切换步骤。
- 未测 production 最坏 FFCWS schema 的 finalizer MaxRSS、wall time 或 100 份 shard manifest 常驻内存；这是方案明确留给 production pilot 的独立证据项。
- `error` 列仍可无界增长，stdlib CSV 的 128 KiB 单字段限制仍保留；本方案未改变该既有风险。
- pandas 输出会把真实列名送入 LightGBM；含 `, " [ ] { } :` 的列名仍可能被 LightGBM 拒绝。本方案依赖已核验的 20 个生产 schema 列名干净这一前提，未添加运行时防御。
- 体积外推来自合成大合同 fixture；真实 production 行宽会随其他标量/metric 列内容变化。

## 7. 给审查者的重点

1. 请重点检查 `ROW_METADATA_FIELDS` 是否与约定的行级 schema 完全一致，以及两个 `add_metadata()` 调用点是否都只接收 `row_metadata`；这是 R1 的唯一承重边界。
2. 请检查 T1 的两阶段语义：生产清理未被 monkeypatch，第一阶段保留真实 part，第二阶段用同一 manifest 恢复并验证最终清理。
3. 请检查 R2 是否严格只修改了两个 LightGBM base pipeline；T4 的旧路径 warning 精确捕获和零容差比较用于证明没有隐藏数值变化。
