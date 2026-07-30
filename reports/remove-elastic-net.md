# remove-elastic-net 实现报告

## 1. 改动清单

- `NK_Grid/src/aleatoric_nk_grid/model_registry.py`
  - 从 `MODEL_NAMES`、`MODEL_PARAM_KEYS`、分类/回归构造分支中移除 `elastic_net`；删除 `AdaptiveElasticNetCV` 与 `ElasticNetCV` import；新增 `REMOVED_MODEL_NAMES` 和 `reject_removed_model`，让显式请求得到自解释错误。
- `NK_Grid/src/aleatoric_nk_grid/nk_grid.py`
  - 在 `_validate_config` 中调用 `reject_removed_model`，使 panel/run config 启动校验阶段响亮失败。
- `NK_Grid/src/aleatoric_nk_grid/validate_input.py`
  - 从 `REGRESSION_CV_MIN_N` 删除已移除模型的最小行数条目。
- `NK_Grid/model_params.yaml`、`FFCWS/model_params.yaml`、`SMR/model_params.yaml`
  - 删除 regression 与 classification 两段中的 `elastic_net` 参数块，不改变其他模型参数。
- `FFCWS/panels.yaml`、`SMR/panels.yaml`
  - 从共享模型列表锚点中删除 `elastic_net`，其余模型顺序不变。
- `FFCWS/adapter/README.md`、`SMR/adapter/README.md`
  - 仅从示例 `--validation-model` 清单中删除 `elastic_net`。
- `NK_Grid/tests/test_model_param_contract.py`
  - 删除旧 elastic-net 参数断言；新增 3 份 model params × 2 个 task 的 9 模型契约覆盖测试；新增已移除模型的拒绝路径测试。
- `NK_Grid/tests/test_nk_grid_engine.py`
  - 在 run-control 校验测试中加入已移除模型拒绝路径。
- `NK_Grid/tests/test_prior_invariance.py`
  - 从快速参数调整分支中删除 elastic-net 特例，参数化集合随 `SUPPORTED_MODEL_NAMES` 自然不再包含它。
- `reports/remove-elastic-net.md`
  - 本实现与验证报告。

## 2. 验收标准逐条核对

| # | 验收标准 | 结论 | 证据 |
|---|---|---|---|
| 1 | `grep -ri elastic NK_Grid/src NK_Grid/tests *.yaml */[a-z]*.yaml` 只剩拒绝信息 | 满足 | 在 zsh `null_glob` 下运行同口径命令，唯一输出为 `NK_Grid/src/aleatoric_nk_grid/model_registry.py` 的 `REMOVED_MODEL_NAMES` 一行 |
| 2 | panel 显式写 `elastic_net` 时启动报自解释错误，含方案文件名 | 满足 | `test_invalid_run_controls_fail_before_dry_run_arithmetic` 断言 `removed from the model space`；`REMOVED_MODEL_NAMES` 消息为 `see plans/remove-elastic-net.md` |
| 3 | `MODEL_NAMES` 长度为 9；`load_model_params` 对 9 个模型全部解析成功 | 满足 | `test_model_param_contract_covers_model_space_exactly`：3 份 YAML × regression/classification 均 `len(MODEL_NAMES) == 9` 且返回键集合恰为 `MODEL_NAMES` |
| 4 | super_learner 数值不变 | 满足 | dev preset 合成面板前后比较：`super_learner` 行包含在 729 rows/9 models/30 metric columns 的逐位相等比对中 |
| 5 | 其余 8 个模型数值不变 | 满足 | 同一 dev preset 比对覆盖 `ols,ridge,lasso,random_forest,xgboost,lightgbm,shallow_neural_network,extra_trees`，30 个 metric 列逐位相等 |
| 6 | 没有 `ElasticNetCV` import 或死代码；pytest 无 import 错误 | 满足 | `rg -n "ElasticNetCV|AdaptiveElasticNetCV" NK_Grid/src NK_Grid/tests` 无输出；全量 pytest `223 passed` |
| 7 | 测试全部通过，且不通过删除/跳过既有测试凑计数 | 满足 | `python -m pytest -q`：`223 passed, 14 warnings in 34.99s`；删除的只有 elastic-net 直接测试，其余测试保留并新增拒绝/契约覆盖 |

## 3. 实测数字

测量环境：

- Python 3.14.3（仓库 `.venv`，通过 `PATH=/Users/wanxiang/Documents/Aleatoric Project/Aleatoric_Luck/.venv/bin:$PATH` 调用 `python`）
- macOS arm64，本地合成数据 72 行、4 个 predictors、固定 seed `741`
- 基线：`main` worktree `891474f`；实现：`codex/remove-elastic-net`
- 配置：dev preset 口径，`n_seeds=3`、`n_draws=3`、`n_sizes_n=3`、`n_sizes_k=3`、`min_n=10`、`max_n=100`、`max_k=100`、`n_jobs=1`
- 比较模型：`ols,ridge,lasso,random_forest,xgboost,lightgbm,shallow_neural_network,extra_trees,super_learner`

数值不变性结果：

| 指标 | 结果 |
|---|---:|
| 比较输出行数 | 729 |
| 比较模型数 | 9 |
| 比较 metric 列数 | 30 |
| status 不一致行 | 0 |
| metric 不一致单元格 | 0 |

逐位比较的 metric 列：

`r2_test, skill_score_pct, rmse, mae, medae, max_error, nrmse, spearman_rho, pearson_r, kendall_tau, ccc, explained_variance, mean_bias, median_bias, pinball_q10, pinball_q90, d2_absolute_error, pinball_q05, pinball_q25, pinball_q50, pinball_q75, pinball_q95, ks_statistic, wasserstein_distance, top_decile_hit_rate, bottom_decile_hit_rate, rsr, cv_rmse, mase, pearson_r2`

比对命令输出：

```text
matched_rows=729 matched_models=9 metric_columns=r2_test,skill_score_pct,rmse,mae,medae,max_error,nrmse,spearman_rho,pearson_r,kendall_tau,ccc,explained_variance,mean_bias,median_bias,pinball_q10,pinball_q90,d2_absolute_error,pinball_q05,pinball_q25,pinball_q50,pinball_q75,pinball_q95,ks_statistic,wasserstein_distance,top_decile_hit_rate,bottom_decile_hit_rate,rsr,cv_rmse,mase,pearson_r2
```

## 4. 测试证据

新增/修改测试函数：

- `test_model_param_contract_covers_model_space_exactly`
  - 覆盖“参数契约”和“配置往返”：三份 `model_params.yaml` 对 regression/classification 均能解析 9 个模型，且无遗漏/多余。
- `test_removed_model_request_fails_with_self_explanatory_message`
  - 覆盖“拒绝路径”：直接请求已移除模型时抛出包含 `removed from the model space` 的 `ValueError`。
- `test_invalid_run_controls_fail_before_dry_run_arithmetic`
  - 新增一组参数，覆盖 panel/run config 显式选择已移除模型时的启动校验拒绝。
- `test_registered_model_predictions_are_invariant_to_full_missing_prior`
  - 删除 elastic-net 用例后，剩余 `SUPPORTED_MODEL_NAMES` 仍通过先验不变性测试。

命令结果：

```text
PATH='/Users/wanxiang/Documents/Aleatoric Project/Aleatoric_Luck/.venv/bin':$PATH PYTHONPATH='/Users/wanxiang/Documents/Aleatoric Project/Aleatoric_Luck_remove_elastic_net/NK_Grid/src' python -m pytest -q
223 passed, 14 warnings in 34.99s
```

无 skipped、无 failed、无 errors。14 个 warning 均来自既有 sklearn/LightGBM 路径：8 个 MLP `ConvergenceWarning`，6 个 LightGBM feature-name `UserWarning`。

## 5. 偏离方案之处与待澄清问题

- 方案称 `REMOVED_MODEL_NAMES` 与 `reject_removed_model` 已由 `remove-bart` 留下；但本任务按用户要求从 `main` 建分支，`main@891474f` 上没有该机制。为满足拒绝路径验收，本实现补了最小机制，并只接入参数加载、模型构造和 run config 校验。
- 方案要求更新 `docs/implementation/no-hash-seed-sharding-completion.md`；`main` worktree 中不存在 `docs/` 目录，因此未修改该文件。原目录里有未提交的 `docs/`，但本分支从干净 `main` worktree 实施，未把其他工作树的未提交文件卷入。
- 顶层没有匹配 `*.yaml` 的文件；验收 grep 在 zsh 下需启用 `null_glob` 才能按方案口径运行，不影响实际扫描范围。

## 6. 未覆盖与已知风险

- 数值不变性使用本地合成 dev preset，不读取 `SMR/data/` 或 `FFCWS/data/` 私有数据；真实 production 面板未重跑。
- 拒绝路径覆盖了 `load_model_params` 和 `NKGridConfig` run-control 校验；CLI `argparse choices=SUPPORTED_MODEL_NAMES` 对已移除模型会先给 argparse 的 invalid choice，而非 `REMOVED_MODEL_NAMES` 文案。方案验收点是 panel 启动路径，本次未扩大到 CLI 行为。
- `docs/implementation/no-hash-seed-sharding-completion.md` 在本基线不存在，审查时如需该未提交文档同步，需要在包含该文档的基线上补一轮。
- 本实现未改 identity、checkpoint、执行结构或剩余模型参数；风险主要集中在 reviewer 是否接受从 `main` 补最小 removed-model 机制这一偏差。

## 7. 给审查者的重点

1. 重点看 `model_registry.py` 中新增的 removed-model 机制是否足够小，且没有改变剩余模型构造行为。
2. 重点看 `test_model_param_contract_covers_model_space_exactly` 是否准确覆盖三份参数 YAML 的 9 模型契约。
3. 重点确认 `main` 上缺少 `docs/implementation/no-hash-seed-sharding-completion.md` 时，本报告第 5 节的处理是否符合审查预期。
