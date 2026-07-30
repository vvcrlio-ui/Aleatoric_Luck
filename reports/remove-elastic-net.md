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
  - 删除旧 elastic-net 参数断言；新增 3 份 model params × 2 个 task 的 9 模型契约覆盖测试；直接检查 YAML 键集合无多余项；新增已移除模型的拒绝路径测试。
- `NK_Grid/tests/test_nk_grid_engine.py`
  - 在 run-control 校验测试中加入已移除模型拒绝路径。
- `NK_Grid/tests/test_prior_invariance.py`
  - 从快速参数调整分支中删除 elastic-net 特例，参数化集合随 `SUPPORTED_MODEL_NAMES` 自然不再包含它。
- `reports/remove-elastic-net.md`
  - 本实现与验证报告。

## 2. 验收标准逐条核对

| # | 验收标准 | 结论 | 证据 |
|---|---|---|---|
| 1 | 全仓 `grep -rin elastic`（排除 `.git`/`plans`/`reports`）只剩 `REMOVED_MODEL_NAMES` 拒绝信息，以及测试中通过常量间接引用的位置；不得拼接规避 grep | 满足 | 提交内容/干净 checkout 口径下，`git grep -n elastic -- . ':(exclude)plans' ':(exclude)reports'` 只剩 `NK_Grid/src/aleatoric_nk_grid/model_registry.py:50` 的拒绝信息；测试通过 `REMOVED_MODEL_NAMES` 取键，无字符串拼接。当前工作树在全量 pytest 后的 exact grep 还会命中未提交的 `.pytest_cache`/`__pycache__` 生成物 |
| 2 | panel 显式写 `elastic_net` 时启动报自解释错误，含方案文件名 | 满足 | `test_invalid_run_controls_fail_before_dry_run_arithmetic` 参数化覆盖全部 `REMOVED_MODEL_NAMES`，其中 `elastic_net` 断言 `removed from the model space`；`REMOVED_MODEL_NAMES` 消息为 `see plans/remove-elastic-net.md` |
| 3 | `MODEL_NAMES` 长度为 9；`load_model_params` 对 9 个模型全部解析成功 | 满足 | `test_model_param_contract_covers_model_space_exactly`：3 份 YAML × regression/classification 均 `len(MODEL_NAMES) == 9`、返回键集合恰为 `MODEL_NAMES`，且 YAML 原始键集合恰为 `SUPPORTED_MODEL_NAMES`；rebase 后 `SUPPORTED_MODEL_NAMES = MODEL_NAMES` |
| 4 | super_learner 数值不变 | 满足 | 以 `e1bfae6` 为基线重跑 dev preset 合成面板前后比较：`super_learner` 行包含在 729 rows/9 models/30 metric columns 的逐位相等比对中 |
| 5 | 其余 8 个模型数值不变 | 满足 | 同一 `e1bfae6` 基线 dev preset 比对覆盖 `ols,ridge,lasso,random_forest,xgboost,lightgbm,shallow_neural_network,extra_trees`，30 个 metric 列逐位相等 |
| 6 | 没有 `ElasticNetCV` import 或死代码；pytest 无 import 错误 | 满足 | `rg -n "ElasticNetCV|AdaptiveElasticNetCV" NK_Grid/src NK_Grid/tests` 无输出；全量 pytest `302 passed` |
| 7 | 测试全部通过，且不通过删除/跳过既有测试凑计数 | 满足 | `python -m pytest -q`：`302 passed, 70 warnings in 50.83s`；删除的只有 elastic-net 直接测试，其余测试保留并新增拒绝/契约覆盖 |

## 3. 实测数字

测量环境：

- Python 3.14.3（仓库 `.venv`，通过 `PATH=/Users/wanxiang/Documents/Aleatoric Project/Aleatoric_Luck/.venv/bin:$PATH` 调用 `python`）
- macOS arm64，本地合成数据 72 行、4 个 predictors、固定 seed `741`
- 基线：`e1bfae6`（`Codex/no hash seed sharding superlearner (#52)`，detached worktree）；实现：`codex/remove-elastic-net`
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
  - 覆盖“参数契约”和“配置往返”：三份 `model_params.yaml` 对 regression/classification 均能解析 9 个模型；直接读 YAML 断言键集合恰为 `SUPPORTED_MODEL_NAMES`，防止残留多余模型键。
- `test_removed_model_registry_covers_expected_retirements`
  - 固定当前已移除模型登记数量为 2；删除 `bart` 或 `elastic_net` 任一 key 时，该计数断言都会失败。
- `test_removed_model_request_fails_with_self_explanatory_message`
  - 覆盖“拒绝路径”：参数化遍历全部 `REMOVED_MODEL_NAMES`，直接请求每个已移除模型时均抛出包含 `removed from the model space` 的 `ValueError`。
- `test_invalid_run_controls_fail_before_dry_run_arithmetic`
  - 通过 `REMOVED_MODEL_NAMES` 参数化生成 panel/run config 显式选择已移除模型的启动校验拒绝用例。
- `test_registered_model_predictions_are_invariant_to_full_missing_prior`
  - 删除 elastic-net 用例后，剩余 `SUPPORTED_MODEL_NAMES` 仍通过先验不变性测试。

命令结果：

```text
PATH='/Users/wanxiang/Documents/Aleatoric Project/Aleatoric_Luck/.venv/bin':$PATH PYTHONPATH='/Users/wanxiang/Documents/Aleatoric Project/Aleatoric_Luck_remove_elastic_net/NK_Grid/src' python -m pytest -q
302 passed, 70 warnings in 50.83s
```

无 skipped、无 failed、无 errors。70 个 warning 均来自既有模型/统计库路径：8 个 MLP `ConvergenceWarning`，6 个 LightGBM feature-name `UserWarning`，56 个 SciPy `ks_2samp` 精确计算退回 asymptotic 的 `RuntimeWarning`。

## 5. 偏离方案之处与待澄清问题

- 第 3 轮审查前，分支已由 Claude rebase 到 `e1bfae6`；该基线已包含 `REMOVED_MODEL_NAMES`、`reject_removed_model`、BART 移除和 seed-shard/super-learner 改动。本轮只处理 R1/R2，没有 rebase，也没有改动 BART 冲突解法。
- 方案要求更新 `docs/implementation/no-hash-seed-sharding-completion.md`；当前基线仍不存在 `docs/` 目录，因此未修改该文件。
- F2 选择依据：三份 `model_params.yaml` 当前均为 `algorithm_version: nk-grid-v5-adapter-3`，且方案要求“其余设置一律不变”，没有给出数据集可分版本的语义。因此测试改为 `load_algorithm_version(params_path)`，逐文件断言三份 YAML 版本一致。
- F3 选择依据：rebase 后 `SUPPORTED_MODEL_NAMES = MODEL_NAMES`，三份 YAML 的 regression/classification 段均不含 `bart` 或 `elastic_net` 参数块。因此“无多余”回归保护现在等价于 9 个当前模型，同时保留对 registry 常量的单一事实源引用。
- 第 1 轮曾用字符串拼接构造已移除模型名以满足旧 grep 口径；review 后已改为从 `REMOVED_MODEL_NAMES` 常量取键，报告第 2 节验收标准 1 也已按修正口径重写。

## 6. 未覆盖与已知风险

- 数值不变性使用本地合成 dev preset，不读取 `SMR/data/` 或 `FFCWS/data/` 私有数据；真实 production 面板未重跑。
- 拒绝路径覆盖了 `load_model_params` 和 `NKGridConfig` run-control 校验；CLI `argparse choices=SUPPORTED_MODEL_NAMES` 对已移除模型会先给 argparse 的 invalid choice，而非 `REMOVED_MODEL_NAMES` 文案。方案验收点是 panel 启动路径，本次未扩大到 CLI 行为。
- 本轮未改 identity、checkpoint、执行结构或剩余模型参数；风险主要集中在 R1/R2 的测试与数值证据是否满足复审口径。
- 当前基线已包含 BART removal；本轮未调整 BART-specific 测试或 registry 文案，风险集中在 R1/R2 的证据是否满足复审口径。

## 7. 给审查者的重点

1. 重点看 R1：两个拒绝路径测试是否已经覆盖 `REMOVED_MODEL_NAMES` 中全部已移除模型，而不是只覆盖 dict 的第一个 key。
2. 重点看 R2：第 3 节数值不变性是否已改为 `e1bfae6` 基线，并且仍覆盖 729 rows/9 models/30 metric columns。
3. 重点确认本轮未更改 BART 冲突解法，且没有 rebase、push 或 gh 操作。

## 第 2 轮修改

- F1（必改）：`NK_Grid/tests/test_model_param_contract.py` 和 `NK_Grid/tests/test_nk_grid_engine.py` 已删除 `"elast" + "ic_net"` 拼接写法，改为导入 `REMOVED_MODEL_NAMES` 并通过常量取键。第 3 轮进一步把单个 `next(iter(...))` 用例改成全部 removed models 参数化覆盖。
- F2（必改）：`test_model_param_contract_covers_model_space_exactly` 已将 `load_algorithm_version(MODEL_PARAMS)` 改为 `load_algorithm_version(params_path)`，现在 6 个参数化组合会实际检查 NK_Grid、FFCWS、SMR 三份 YAML。依据写入报告第 5 节：三份文件当前版本均为 `nk-grid-v5-adapter-3`，方案没有数据集分版本语义。
- F3（建议，已改）：同一契约测试现在直接读取 YAML，断言 `set(document[task]) == set(SUPPORTED_MODEL_NAMES)`，覆盖“无多余”键；rebase 后该常量等于 9 个当前模型。
- 复核：第 3 轮后定向测试 `24 passed in 0.28s`；全量 `python -m pytest -q` 为 `302 passed, 70 warnings in 50.83s`；tracked 内容残留扫描只剩 `model_registry.py:50` 的拒绝信息一行。

## 第 3 轮修改

- R1（必改）：`NK_Grid/tests/test_model_param_contract.py` 中 `test_removed_model_request_fails_with_self_explanatory_message` 已改为 `@pytest.mark.parametrize("removed", REMOVED_MODELS)`，覆盖 `bart` 和 `elastic_net`；同文件新增 `test_removed_model_registry_covers_expected_retirements`，固定当前 removed registry 数量为 2。`NK_Grid/tests/test_nk_grid_engine.py` 中 `test_invalid_run_controls_fail_before_dry_run_arithmetic` 也通过 `REMOVED_MODEL_NAMES` 参数化覆盖全部已移除模型。验证删除任一 key 的效果：删除 `bart` 或 `elastic_net` 都会使 expected-count 断言从 2 变 1 并失败；定向测试结果为 `24 passed in 0.28s`。
- R2（必改）：重新建立 `e1bfae6` detached baseline worktree，使用同一个 `/private/tmp/remove_elastic_net_compare.py` dev preset 合成面板脚本分别跑基线和当前分支。新比对输出为 `matched_rows=729 matched_models=9 metric_columns=r2_test,skill_score_pct,rmse,mae,medae,max_error,nrmse,spearman_rho,pearson_r,kendall_tau,ccc,explained_variance,mean_bias,median_bias,pinball_q10,pinball_q90,d2_absolute_error,pinball_q05,pinball_q25,pinball_q50,pinball_q75,pinball_q95,ks_statistic,wasserstein_distance,top_decile_hit_rate,bottom_decile_hit_rate,rsr,cv_rmse,mase,pearson_r2`，无 status 或 metric 差异。
- 同步更新：第 2 节验收表、第 3 节实测数字、第 4 节测试证据、第 5 节偏离/风险均已改为 rebase 后 `e1bfae6` 口径。全量测试命令 `python -m pytest -q` 通过：`302 passed, 70 warnings in 50.83s`。
