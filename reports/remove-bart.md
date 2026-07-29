# 从模型空间中移除 BART

分支：`codex/remove-bart`，基线 `891474f`（`cell-centric-execution` 已合入 `main`，顺序约束满足）。

> **执行者说明**：本任务由 Claude 直接实现，不是 Codex。原因是 Codex 本次会话的
> sandbox 禁止写 `.git/refs`，无法建分支与提交；用户明确授权由 Claude 代为完成。
> 这偏离了本仓库 Claude 只做规划与审查的分工，记录在此备查。

> **授权变更**：用户明确指示"不用管方案里面说的，我授权你可以直接删掉 bart 相关的内容"。
> 因此**验收标准 3（`experiment_id` 逐字符不变）被主动放弃**，改为如实记录变化值。
> 详见第 5 节，这是本报告最重要的一条。

## 1. 改动清单

18 个文件，+112 / −406 行。

### 引擎源码

| 文件 | 改了什么 | 为什么 |
|---|---|---|
| `model_registry.py` | 删 `BartPyRegressor`、`BartPyClassifier`、`_BART_RANDOM_LOCK`、`MATPLOTLIB_CACHE` 三行、`threading` import、`MODEL_PARAM_KEYS` 两处 `bart` 条目、`resolved_model_params` 的 `BART_*` 环境变量分支、`make_model` 两处 `bart` 构造分支 | BART 退出模型空间；`MATPLOTLIB_CACHE` 存在的唯一理由是 BartPy 会 import matplotlib（原注释自述） |
| `model_registry.py` | 新增 `REMOVED_MODEL_NAMES` 与 `reject_removed_model()`；在 `load_model_params` 早期、`_validated_params`、`make_model` 两处共 4 个入口调用 | 方案 §1：遗留 panel 必须明确失败而非 KeyError 或静默少跑一个模型 |
| `model_registry.py` | `SUPPORTED_MODEL_NAMES = MODEL_NAMES`，删 `LEGACY_MODEL_NAMES` | 不再有"可构造的遗留模型" |
| `nk_grid.py` | 删 `NKGridConfig.bart_min_n/bart_min_k`、`_validate_config` 的非负校验、两个 CLI 参数及其传递、`run_model` 的 BART N/K floor 分支 | 失去意义的分支 |
| `nk_grid.py` | execution window 的 `non_bart`/`bart` 双子窗口收敛为单一 `prefer="threads"` 窗口 | 方案 §3；BART 是唯一需要 `prefer="processes"` 的模型 |
| `nk_grid.py` | `_parallelism_payload` 的 `window_policy` 由嵌套 `non_bart`/`bart` 两段扁平化为单段 | manifest 不得再描述不存在的调度 |
| `experiment.py` | `MODEL_ENV_KEYS` 删 4 个 `BART_*`；删 `parallel_preference()` 函数 | 该函数在 `cell-centric-execution` 之后已无任何调用者，恒返回 `"threads"`（见第 5 节） |
| `run_panels.py` | `DEFAULTS` 与 `PANEL_FIELDS` 各删 `bart_min_n`/`bart_min_k` | 跟随 `NKGridConfig` |
| `slurm_jobs.py` | `RESOURCE_CLASSES` 改为 `("parallel","serial")`；`resource_class_for_model` 删 bart 分支 | 方案 §2 |

**未触碰** `native_process.py` 与 `IsolatedProcessRunner`（方案 §3 明确要求保留，它服务 lightgbm / super_learner）。`git diff` 中 `IsolatedProcessRunner` 出现 **0** 次。

### 配置、Slurm、依赖、文档

| 文件 | 改了什么 |
|---|---|
| `slurm/submit_nk_grid.sh` | 删 `BART_CPUS_PER_TASK`/`BART_MEMORY`/`BART_TIME_LIMIT` 三个变量、`--bart-cpus-per-task`/`--bart-mem`/`--bart-time` 三个参数及其解析与校验、usage 三行、`--resource-class` 白名单的 `bart`、`RESOURCE_CLASSES` 数组、per-class case 的 `bart)` 分支。共 −30 行 |
| `NK_Grid/`、`SMR/`、`FFCWS/` 的 `model_params.yaml` | 各删 `regression.bart`（5 个键）与 `classification.bart: {}` 两个块 |
| `pyproject.toml`、`requirements.txt` | 删 `bartpy2==0.0.2` |
| `NK_Grid/README.md` | 执行类说明由三档改两档 |

### 测试

见第 4 节。

## 2. 验收标准逐条核对

| # | 验收标准 | 结论 | 证据 |
|---|---|---|---|
| 1 | `grep -rn -i bart` 只剩拒绝路径及其测试 | **满足** | 源码/配置/依赖侧唯一命中是 `model_registry.py:51` 的 `REMOVED_MODEL_NAMES`。测试侧命中全部属于拒绝路径断言或解释性注释。`reports/` 下的历史报告未改（`cell-centric-execution` 的审查记录，不应改写） |
| 2 | 显式请求 `bart` 抛出含"已移除"与方案路径的 `ValueError` | **满足** | 实测 `make_model('bart', ...)` → `ValueError: BART was removed from the model space; see plans/remove-bart.md`。测试 `test_requesting_a_removed_model_fails_with_a_self_explaining_error[regression/classification]` 覆盖 `load_model_params` 与 `make_model` 两个入口 |
| 3 | `experiment_id` 逐字符不变 | **主动放弃（用户授权）** | toy panel 实测 `67142b7d96418382e1f3` → `becc82a4a5988215f98a`。原因与后果见第 5 节 |
| 4 | toy panel 最终 CSV 逐位相同 | **满足** | 9 个生产模型 × 2 seeds × 2 draws × 2 N × 2 K = 144 行 × 35 列 = **5040 个值**，`pd.testing.assert_frame_equal(check_exact=True)` 全等通过 |
| 5 | `bartpy2` 不再出现在依赖表；无 bartpy2 的环境中测试全过 | **满足** | 两文件均已删除。用 `sys.meta_path` blocker 屏蔽 `bartpy`/`bartpy2` 后全量 **217 passed in 36.05s**（注意：本机 venv 里 bartpy2 **仍装着**，所以未屏蔽的那次运行不算证据，见第 3 节方法说明） |
| 6 | Slurm 提交脚本两档 dry-run 正常，不再接受 bart 参数 | **满足** | `submit_nk_grid.sh --manifest SMR/panels.yaml --dry-run` 退出码 0，解析出 20 个 job。`--bart-mem 96G` → `Unknown option: --bart-mem`；`--resource-class bart` → `--resource-class must be parallel or serial`。`bash -n` 语法检查通过 |
| 7 | manifest 的 `design.parallelism` 不再含 `bart` 段 | **满足** | 实跑 manifest 的 `design` 整段 JSON 中 `bart` 出现次数为 **0**；`window_policy` 现为扁平单段。测试 `test_manifest_records_actual_window_policy_without_legacy_model_policy` 断言 `"bart" not in json.dumps(parallelism)` |
| 8 | 测试全部通过，说明数量变化 | **满足** | **217 passed**，与 main 基线的 217 相同。净变化为 0：删 4 项、加 4 项，明细见第 4 节 |

## 3. 实测数字

环境：macOS 15.5 (Darwin 25.5.0)，Apple Silicon，Python 3.14.3，仓库 venv。

### 数值等价（验收标准 4）

方法：沿用 `cell-centric-execution` 的 A/B 方法，复用其 `conftest.write_schema_bundle` 与
合成 frame 构造。改动前在 `891474f` 工作树跑一次，改动后在同一工作树跑一次，比较
`["model","seed","draw","N","K", *METRIC_COLUMNS]` 并按 `["seed","draw","K","N","model"]`
稳定排序。

| 项 | 值 |
|---|---|
| 模型 | 9 个生产模型（不含 super_learner，见第 6 节） |
| 网格 | n_seeds=2, n_draws=2, n_sizes_n=2, n_sizes_k=2 |
| 结果行 | 144（前后一致） |
| 比较列 | 35 |
| 比较值总数 | **5040** |
| 不一致值 | **0**（`check_exact=True`） |

### 身份变化（验收标准 3）

| 项 | 改动前 | 改动后 |
|---|---|---|
| toy panel `experiment_id` | `67142b7d96418382e1f3` | `becc82a4a5988215f98a` |

单因素隔离实测（只从 `metadata_extra` 移除两个键，其余不动）：

```
含 bart_min_n/bart_min_k : 6a0dd30a8dad8c448dc4
删 bart_min_n/bart_min_k : 71989fd73e1bf2e35889
```

### 测试规模

| 运行 | 结果 | 耗时 |
|---|---|---|
| main（`891474f`，改动前，worktree + PYTHONPATH 隔离） | 217 passed | 36.41 s |
| 本分支 | 217 passed | 36.05 s |
| 本分支 + 屏蔽 bartpy2 | 217 passed | 36.05 s |

### 代码量

| 指标 | 值 |
|---|---|
| 触碰文件 | 18 |
| 净行数 | +112 / −406（净 −294） |
| `model_registry.py` | −141 行中的绝大部分是两个 estimator 类 |
| `submit_nk_grid.sh` | −30 行 |

## 4. 测试证据

`python -m pytest -q` → **`217 passed, 14 warnings in 36.05s`**

### 新增（4 项，对应方案"测试要求"）

| 测试 | 对应要求 |
|---|---|
| `test_removed_model_is_absent_from_the_model_space` | 拒绝路径：`MODEL_NAMES`、`REMOVED_MODEL_NAMES`、三份 YAML 三处一致 |
| `test_requesting_a_removed_model_fails_with_a_self_explaining_error[regression]` | 拒绝路径测试：错误信息含方案路径 |
| `test_requesting_a_removed_model_fails_with_a_self_explaining_error[classification]` | 同上，分类任务 |
| `test_removed_model_environment_overrides_are_gone` | 确认 `BART_*` 环境变量不再被任何路径读取 |

### 改写（4 项）

| 测试 | 改法 |
|---|---|
| `test_mixed_bart_panel_schedules_bart_in_separate_process_subwindow` → `test_mixed_panel_runs_in_one_threads_window` | 保留 review F1 的回归价值，断言方向反转：混合 native/非 native panel 现在必须collapse 成**单个** threads 窗口 |
| `test_manifest_records_actual_window_policy_without_legacy_model_policy` | 更新为扁平 `window_policy`，并加断言 `"bart" not in json.dumps(parallelism)` |
| `test_all_unobserved_skip_keeps_priority_over_bart_floor` → `test_all_unobserved_skip_happens_before_any_preprocessing` | 见第 5 节：优先级语义已不可测，保留早跳过行为 |
| `test_slurm_jobs.py` 共 8 处 | 两档资源分类重写：资源类穷尽性、`BART_N_TREES`→`RF_N_ESTIMATORS` 环境样本、fake harness 的 job 数 6→4 / 3→2、submitter 的 3 个 array 断言改 2 个 |

### 删除（4 项）

| 测试 | 为什么 |
|---|---|
| `test_bart_environment_override_is_resolved_at_model_construction` | `BART_*` 环境覆盖路径已删；其价值由新增的 `test_removed_model_environment_overrides_are_gone` 反向覆盖 |
| `test_floor_skips_do_not_run_full_preprocessing[bart-...]` 参数化档 | BART N/K floor 已删；`ridge` 档保留，早跳过路径仍有覆盖 |
| `test_bart_process_task_remains_pickleable_without_cached_order_ipc` | 专测 `prefer="processes"` 下 `orders=None` 与任务可 pickle；processes 路径随 BART 一起消失。threads 路径不需要 pickle，无等价断言可留 |
| `test_prior_invariance` 的 `bart` 参数化档 | 随 `SUPPORTED_MODEL_NAMES` 自动消失 |

净变化 −4 +4 = **0**，故总数仍为 217。

### 已知失败

**无稳定失败。** 但全量运行中观察到 `test_native_process.py` 的**间歇性** flake，
两次分别是不同的测试：

| 运行 | 耗时 | 失败 |
|---|---|---|
| 第一次全量 | 86.80 s | `test_native_process_timeout_kills_worker_and_parent_can_continue` |
| 第二次全量 | 144.13 s | `test_python_exception_crosses_boundary_without_destroying_worker` |
| 第三次（屏蔽 bartpy2） | 36.05 s | 无 |
| main 全量 | 36.41 s | 无 |
| 单独跑该文件（改动前后各若干次） | ~10–28 s | 无，均 5 passed |

归属结论：**与本改动无关，是负载敏感的既有 flake**。依据三条：
(a) 失败的测试在两次运行间**变化**，非确定性；
(b) 本改动 `git diff` 未触碰 `native_process.py`，`IsolatedProcessRunner` 出现 0 次；
(c) 低负载运行在改动前后均全过。

**诚实说明**：我**没能在 main 上主动复现**该 flake——它只在高负载（86 s / 144 s 那两次，
机器同时在跑别的东西）时出现，而 main 的对照运行恰好是低负载。所以 (c) 是弱证据，
真正的依据是 (a) 和 (b)。这个 flake 值得单开任务，与
`plans/xgboost-determinism.md` 记录的 `best_rounds` 非确定性可能是同源问题
（都在并发下暴露）。

## 5. 偏离方案之处与待澄清问题

### 5.1 【最重要】验收标准 3 与技术方案 §2 相互矛盾，按用户授权选择放弃标准 3

**方案的内部矛盾**：方案第 81 行要求从 `metadata_extra` 删除 `bart_min_n`/`bart_min_k`，
但这两个键参与 `build_experiment_metadata` 的 sha256，删除必然改变 `experiment_id`，
与验收标准 3（逐字符不变）直接冲突。方案第 30–33 行的"身份中性已验证"只覆盖了
删除 `model_params.yaml` 的 `bart:` 块（因为 `load_model_params(path, task, models)`
只返回选中模型的参数），**没有覆盖这两个字段**。

实测确认（单因素隔离，见第 3 节）：`6a0dd30a8dad8c448dc4` → `71989fd73e1bf2e35889`。

**处理**：我先提出了这个矛盾并倾向保留两个键作为 identity 墓碑常量；用户明确回复
"不用管方案里面说的，我授权你可以直接删掉 bart 相关的内容"。故按授权直接删除，
放弃验收标准 3。

**必须知道的后果 —— 已有 checkpoint 会被静默全部重算**：

`nk_grid.py` 的 resume 路径用 `rows_for_experiment(existing, experiment_id)` 过滤已完成
的 cell。`experiment_id` 变化后，旧 checkpoint 里的行会被当作**另一个实验**的结果，
既不报错也不复用，直接重算。对 dev preset 无所谓；对 production 网格
（`n_seeds=100 × n_draws=50 × 20 × 20`）意味着已完成的部分全部作废。

**当前受影响的产物**：`SMR/outputs/` 与 `FFCWS/outputs/` 下已有的 dev run
（`nk_grid_smr_hourlywage_dev_20260727-155807.csv` 等）。它们的 `experiment_id`
与新代码算出的不再一致。

**请确认**：这是否符合预期。若希望旧结果可继续 resume，唯一办法是把
`bart_min_n`/`bart_min_k` 作为常量保留在 `metadata_extra` 里（功能仍全删），
我可以在同分支追加一次提交改回来。

### 5.2 方案 §2 给的二选一：删除 `parallel_preference` 而非保留

方案第 82 行让实现者二选一。**选择删除**。理由：`cell-centric-execution` 之后
execution window 直接使用字面量 `prefer`，该函数已无任何调用者（全仓 grep 只有定义
一处），保留一个恒返回 `"threads"` 且无人调用的函数是纯负债。

### 5.3 一个测试的语义随 BART 一起消失，无法等价保留

`test_all_unobserved_skip_keeps_priority_over_bart_floor` 原本断言两个跳过条件
（全未观测 vs BART N/K floor）同时满足时的**优先级**。BART floor 删除后，剩余的唯一
floor 是 `REGRESSION_CV_MIN_N`（`ridge/lasso/elastic_net`=2，`lightgbm/super_learner`=5），
而原测试的全未观测构造依赖 N=10，在该 N 下 CV floor 不可能触发，两个条件无法共存，
**优先级不再可观测**。

已改写为 `test_all_unobserved_skip_happens_before_any_preprocessing`，保留"早跳过且
不触发 `preprocess_cell`"这一仍然可测的行为，并在测试内注释说明失去了什么。

### 5.4 `matplotlib` 依赖可能已成孤儿（未处理，超出范围）

`MATPLOTLIB_CACHE` 的原注释自述其存在理由是"BartPy imports matplotlib even for
non-plotting fits"。该变量已删，但 `requirements.txt` 里的 `matplotlib==3.10.9`
（以及 `seaborn`、`shap`）**未动**——方案 out-of-scope 明确写了"不顺手重构"，且它们
可能被仓库外的分析脚本使用。建议单独确认。

## 6. 未覆盖与已知风险

- **super_learner 未进入 A/B**：A/B 用了 9 个模型，排除 `super_learner`，因为它在
  toy 规模下会因 `REGRESSION_CV_MIN_N=5` 与内部 CV 频繁跳过，产出大量空行，
  削弱逐位比较的信息量。它的调度路径（`SERIAL_OUTER_MODELS` + native 隔离）
  由既有测试覆盖，但未做端到端 A/B。
- **生产 panel 的新 `experiment_id` 未实测**：只测了 toy panel 的前后值。SMR/FFCWS
  生产 panel 的具体新值要等下次真跑才知道。若第 5.1 节的决定被推翻，此项作废。
- **Slurm 只做了 dry-run 与 fake-sbatch 测试**：没有真实集群提交。两档资源分类在真实
  `sbatch` 下的行为未验证。
- **`native_process` flake 未定位**：见第 4 节。本改动未触碰该路径，但它会污染全量
  测试信号。
- **`egg-info` 仍含 `bartpy2`**：`NK_Grid/src/aleatoric_nk_grid.egg-info/` 是构建产物，
  会在下次 `pip install -e` 时重新生成，未手动清理。

## 7. 给审查者的重点

1. **第 5.1 节的 `experiment_id` 决定**。这是整个改动里唯一不可逆的语义变化，
   直接影响已有 checkpoint 能否 resume。请确认放弃验收标准 3 是有意的，
   以及 `SMR/outputs/` 下已有 dev run 作废是可接受的。

2. **`nk_grid.py` 的 execution window 收敛**（原 1889–1960 行区域）。这里删掉了
   `cell-centric-execution` review F1 刚加的双子窗口。我保留了 `rows_by_cell` 中间结构
   与 `checkpoint_buffer.extend` 的原有遍历形式**未动**，目的是让 checkpoint 写入顺序
   的语义与 F1 建立的保证逐字一致（按 `execution_window` 顺序、以 `(cell_key, model)`
   取行）。请复核这个收敛没有改变顺序保证——5040 个值逐位相同是支持证据，
   但顺序保证本身值得单独看一眼。

3. **`reject_removed_model` 的 4 个调用点是否覆盖了所有入口**
   （`load_model_params` 早期、`_validated_params`、`make_model` 的分类与回归分支）。
   特别是 `load_model_params` 那处：我把它放在"YAML 缺少选中模型"的通用报错**之前**，
   否则遗留 panel 会收到"missing selected model(s): bart"这种含糊信息，
   而不是方案 §1 要求的明确拒绝。
