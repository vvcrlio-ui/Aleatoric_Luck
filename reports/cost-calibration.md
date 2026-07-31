# 成本标定：`calibrate_cost.py` 工具交付（本轮不含实际测量）

基线：`main` @ `83b4071`（`remove-elastic-net` 已合入，模型空间为 9 个：
ols / ridge / lasso / random_forest / extra_trees / shallow_neural_network /
super_learner / xgboost / lightgbm）。分支：`claude/cost-calibration`。

**本轮范围（人工在执行过程中口头确认，覆盖 plan 文档原有的"今晚缩减范围"指示）：**
只交付 `calibrate_cost.py` 工具本体、CLI、以及方案"测试要求"列出的全部测试场景，
**不在本机（MacBook Air，Apple Silicon / ARM）上运行任何生产规模的实际测量**，
**不产出** `NK_Grid/calibration/cost_model_<日期>.json`。原因见第 5 节。

## 1. 改动清单

- `NK_Grid/src/aleatoric_nk_grid/calibrate_cost.py`（新增）：
  - `guard_not_private_data` / `PrivateDataAccessError`：拒绝任何指向
    `SMR/data/`、`FFCWS/data/` 的路径。
  - `check_thread_env` / `enforce_thread_env`：读取
    `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`NUMEXPR_NUM_THREADS`，
    严格模式下不等于 `1` 就抛异常，非严格模式打印警告并把结果原样返回以便写入标定文件。
  - `time_budget`（`contextmanager`，基于 `signal.setitimer(SIGALRM)`）：
    单次测量超过 `max_seconds` 就抛 `MeasurementCensored`，用于删失路径。
  - `SyntheticDataParams` / `onehot_group_size_pool` / `generate_synthetic_bundle` /
    `assert_frame_determinism`：确定性合成数据生成器，产出 Parquet 训练表 +
    `feature_manifest.csv` + `feature_universe.json` + `schema.json`，可直接被
    `ingest.load_input` 消费。
  - `measure_t0`：在全新子进程里测 `t_load`/`t_split`/`t_orders`（各 5 次重复取
    中位数/最小/最大）；`t_import` 复用人工预测值，不重跑（见方案的"今晚缩减范围"）。
  - `build_session` / `measure_one_cell` / `run_stage_a` / `run_stage_b`：
    复用引擎公开 API（`load_input`、`validate_input`、`split_frame`、`draw_orders`、
    `preprocess_cell`）和两个既有私有辅助函数
    （`nk_grid._fit_predict_model_cell`、`nk_grid._process_peak_rss_bytes`）
    来测量单个 `(model, N, K)` 格点的 fit 时间、预处理时间与峰值 RSS,不复制引擎逻辑。
  - `fit_power_law` / `predict_power_law` / `fit_all_models` / `fit_preprocess_by_mode` /
    `fit_peak_rss`：幂律回归（OLS on log-log）与外推预测。
  - `build_validation_rows` / `build_calibration_payload` / `write_calibration_file` /
    `recompute_fit_cost_from_raw`：组装并写出方案 §5 定义的标定文件 schema；
    独立从 `raw_measurements` 复算回归系数用于往返校验。
  - `parse_args` / `main`：`python -m aleatoric_nk_grid.calibrate_cost` 入口，
    含 `--max-seconds`、`--stage-b-points`、`--fallback-feature-units`、
    `--allow-nonproduction-threads` 等 CLI 参数。
- `NK_Grid/tests/test_calibrate_cost.py`（新增）：14 个测试，覆盖方案"测试要求"
  的全部 6 条（见第 4 节）。
- 未修改任何既有文件；未产出 `NK_Grid/calibration/cost_model_*.json`
  （本轮未运行实际测量）。

## 2. 验收标准逐条核对

| # | 验收标准 | 结论 | 证据 |
|---|---|---|---|
| 1 | 不改变任何既有数值行为，新增模块纯附加，现有测试全部通过 | 满足 | `python -m pytest -q` → `316 passed`（基线 302 + 新增 14，0 failed/0 errors）；新模块未修改任何既有文件 |
| 2 | t0 分解为 4 分量，含中位数/最小/最大，总和与端到端时间之差 < 10% | 未满足（本轮未运行测量） | `measure_t0` 已实现并有单测覆盖调用路径，但未在生产维度上实际运行；见第 5、6 节 |
| 3 | 每个模型给出 `c_m,a_m,b_m,R²`，R²<0.8 单独说明 | 未满足（本轮未运行测量） | `fit_all_models`/`fit_power_law` 已实现并通过合成数据单测验证正确性，但未对 9 个真实模型跑 Stage A |
| 4 | 外推校验：Stage B 每个未删失点 `predicted/actual ∈ [0.5,2]` | 未满足（本轮未运行测量） | `run_stage_b`/`build_validation_rows` 已实现，无实际 Stage B 数据 |
| 5 | 峰值 RSS 给出三系数与 R² | 未满足（本轮未运行测量） | `fit_peak_rss` 已实现，无实际数据 |
| 6 | 删失点显式记录在标定文件与报告中，不静默丢弃 | 不适用（本轮无标定文件产出） | 机制已实现且被单测覆盖：`test_measure_one_cell_records_censoring_and_excludes_from_regression` |
| 7 | 标定文件可被 `json.load` 读取并通过 schema 校验；`raw_measurements` 足以复算回归 | 满足（机制层面；无生产文件） | `test_calibration_file_round_trip_recomputes_fit_cost`：写出/读回/用 `recompute_fit_cost_from_raw` 复算系数，与 `fit_cost` 一致（误差 < 1e-9） |
| 8 | 总测量墙钟时间被记录，超过 24 小时需说明原因 | 不适用（本轮未运行测量） | — |
| 9 | 产出中不含任何 metric 值或效果结论 | 满足 | 代码审查：`calibrate_cost.py` 全文不出现 `r2_test`/`rmse`/`mcfadden` 等 metric 字段；测试 `test_build_session_never_opens_private_data_paths` 等同样不涉及效果指标 |

## 3. 实测数字

**本轮未运行实际测量。** 按方案 §"报告要求"的格式要求，以下三张表保留完整结构，
每一行按方案要求填 **"未测——推迟到集群"**，不留空、不编造数字。

### 表 1：t0 分量表（4 行 + 合计）

| 分量 | median (s) | min (s) | max (s) | n_reps |
|---|---|---|---|---|
| t_import | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 |
| t_load | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 |
| t_split | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 |
| t_orders | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 |
| **合计 t0** | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 |

### 表 2：每模型 `c_m, a_m, b_m, R²`（9 行）+ 预处理两个 mode

| 模型 / mode | c (=exp(log_c)) | a (K 指数) | b (N 指数) | R² |
|---|---|---|---|---|
| ols | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 |
| ridge | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 |
| lasso | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 |
| random_forest | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 |
| extra_trees | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 |
| shallow_neural_network | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 |
| super_learner | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 |
| xgboost | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 |
| lightgbm | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 |
| preprocess: imputed | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 |
| preprocess: passthrough | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 |

### 表 3：阶段 B 预测 vs 实测（3 个点 × 9 个模型，含 ratio 列）

| N | K | 模型 | predicted (s) | actual (s) | ratio |
|---|---|---|---|---|---|
| 4242 | 8053 | ols | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 |
| 4242 | 8053 | ridge | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 |
| 4242 | 8053 | lasso | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 |
| 4242 | 8053 | random_forest | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 |
| 4242 | 8053 | extra_trees | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 |
| 4242 | 8053 | shallow_neural_network | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 |
| 4242 | 8053 | super_learner | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 |
| 4242 | 8053 | xgboost | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 |
| 4242 | 8053 | lightgbm | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 |
| 1000 | 8053 | ols | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 |
| 1000 | 8053 | ridge | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 |
| 1000 | 8053 | lasso | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 |
| 1000 | 8053 | random_forest | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 |
| 1000 | 8053 | extra_trees | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 |
| 1000 | 8053 | shallow_neural_network | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 |
| 1000 | 8053 | super_learner | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 |
| 1000 | 8053 | xgboost | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 |
| 1000 | 8053 | lightgbm | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 |
| 4242 | 3125 | ols | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 |
| 4242 | 3125 | ridge | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 |
| 4242 | 3125 | lasso | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 |
| 4242 | 3125 | random_forest | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 |
| 4242 | 3125 | extra_trees | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 |
| 4242 | 3125 | shallow_neural_network | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 |
| 4242 | 3125 | super_learner | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 |
| 4242 | 3125 | xgboost | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 |
| 4242 | 3125 | lightgbm | 未测——推迟到集群 | 未测——推迟到集群 | 未测——推迟到集群 |

测量方法与环境（供集群复测时对照）：Python 3.14.3（`.venv`），
scikit-learn 1.8.0 / pandas 3.0.2 / numpy 2.4.4 / xgboost 3.2.0 / lightgbm 4.6.0 /
pyarrow 25.0.0，机器为 MacBook Air（Apple Silicon, macOS 26.5.2, ARM64）。
`calibrate_cost.py` 在此机器上做过**探索性冒烟测试**（非本轮交付的正式测量,
数字不计入标定文件，仅用于验证代码路径可跑通）：合成 4242 行 × 8053 源
（展开后 16205 列）的 Parquet 生成耗时约 7.6s、`build_session`（含完整
`validate_input`）约 11.5s、单模型单格点在 N=4242,K=1000 处的 fit 时间范围
约 1.5s（shallow_neural_network）到 88s（super_learner）——这些数字仅证明工具
可在本机跑通、不会挂起或崩溃，**不作为标定结果使用**,理由见第 5 节。

## 4. 测试证据

新增/修改的测试函数（全部新增于 `NK_Grid/tests/test_calibrate_cost.py`），
对应方案"测试要求"的条目：

| 测试函数 | 对应"测试要求"条目 |
|---|---|
| `test_fit_power_law_recovers_known_exponents_noise_free` | 1. 回归可恢复已知指数（无噪声，误差 < 1e-6） |
| `test_fit_power_law_recovers_known_exponents_with_noise` | 1. 回归可恢复已知指数（加噪声后仍在容差内） |
| `test_fit_power_law_rejects_too_few_points`（附加） | 输入校验，非方案硬性要求，顺手补充 |
| `test_synthetic_bundle_generation_is_deterministic` | 2. 合成数据确定性（同 seed 两次生成，逐值/dtype 相等） |
| `test_synthetic_bundle_generation_differs_across_seeds`（附加） | 补充：不同 seed 产出不同数据，佐证 2 的可靠性 |
| `test_measure_one_cell_records_censoring_and_excludes_from_regression` | 3. 删失路径：monkeypatch 必然超时的假模型，断言记为 censored、不进入回归、出现在标定文件里 |
| `test_calibration_file_round_trip_recomputes_fit_cost` | 4. 标定文件往返：写出后读回，schema 完整，`raw_measurements` 复算系数与 `fit_cost` 一致 |
| `test_enforce_thread_env_refuses_to_start_when_not_production_set` | 5. 环境变量断言：严格模式下拒绝启动 |
| `test_enforce_thread_env_warns_and_records_when_not_strict` | 5. 环境变量断言：非严格模式下警告并记录 |
| `test_check_thread_env_ok_when_all_set_to_one`（附加） | 5. 佐证：生产环境设置下不拒绝、不警告 |
| `test_guard_rejects_smr_and_ffcws_data_paths` | 6. 不触碰私有数据：路径守卫拒绝 `SMR/data`、`FFCWS/data` |
| `test_guard_allows_schema_and_scratch_paths`（附加） | 6. 佐证：允许 `FFCWS/schema` 与合成数据临时目录 |
| `test_build_session_never_opens_private_data_paths` | 6. 端到端断言：`build_session` 实际打开的文件路径都不含 `SMR/data`/`FFCWS/data` |
| `test_onehot_group_size_pool_only_reads_schema_directory`（附加） | 6. 佐证：one-hot 尺寸分布来源只读 schema 目录 |

`python -m pytest -q` 结果（整仓库，含新增测试）：

```
316 passed, 70 warnings in 52.20s
```

基线（本次开工前，在 `main` @ `83b4071`）：`302 passed, 0 failed, 0 errors`
（`AGENTS.md` 文档记载的历史基线是 `216 passed, 1 failed`，但那是更早的
commit；自那以后合入的提交已经修好了 `test_slurm_jobs.py` 的失败，当前
`main` 是全绿的，已用 `git log`/`pytest` 核实）。无跳过、无已知失败。

## 5. 偏离方案之处与待澄清问题

**最大的偏离，需要审查者重点关注：本轮完全没有运行 Stage A / Stage B 实际测量，
没有产出 `NK_Grid/calibration/cost_model_<日期>.json`。**

方案原文和最初的任务指令都要求"今晚"在本机（MacBook Air）上用缩减范围
（`--max-seconds 300`、Stage B 只测 1/3 个点、维度探测 + 可能降维）跑一次实际测量,
产出量级可用的标定文件。在实现过程中，人工进一步明确指示：**不要执行实际测量**，
理由如下（人工原话精神，转述）：

- 本机是 Apple Silicon（ARM64）架构，生产 Slurm 计算节点是 x86 架构；线程调度、
  内存带宽、CPU 微架构差异足够大，即使是"量级"意义上的绝对耗时数字也可能有误导性
  （不只是常数因子的缩放，向量化路径、BLAS 后端实现都可能不同）。
- 方案本身 §2 已经写明"t0 必须在实际执行节点上再测一遍"，"若本方案只能在本地跑，
  必须在报告第 6 节写明't0 未在集群上验证'，且 `flat-task-table` 的 `T_target`
  在集群实测前不得定稿"。既然如此，本地测量数字对下游 `flat-task-table` 没有
  可直接使用的定稿价值,不如把这一轮的产出限定为"工具已就绪、可在集群上直接跑"。

因此本轮交付物是**工具本身**（`calibrate_cost.py` + CLI + 全部测试要求覆盖的测试），
不含标定文件。这与最初任务描述中"产出 `NK_Grid/calibration/cost_model_<UTC-date>.json`"
的要求不一致，是本轮最主要的范围偏离，请审查者确认是否可接受、或要求后续在集群上
补跑。

其余实现层面的偏离/选择（原方案未强制规定，均可复核）：

1. **训练表用 Parquet 而非 CSV**：`generate_synthetic_bundle` 把合成数据写成
   `train.parquet`（`ingest.read_table`/`table_columns` 原生支持 Parquet），
   而不是 CSV。原因：展开后 p≈16205 列 × 4242 行的宽表用 CSV 编码文本会产生
   > 1GB 的文件、读写显著变慢；Parquet 是引擎已支持的格式，不算改变引擎逻辑。

2. **复用两个私有辅助函数**：`nk_grid._fit_predict_model_cell`（拟合+预测一个 cell）
   和 `nk_grid._process_peak_rss_bytes`（跨平台 RSS 读取，macOS 上 `ru_maxrss`
   已是字节、Linux 上是 KB）。这两个函数已经是生产路径复用的既有实现（不是本轮新写），
   直接调用而不是复制一份逻辑，避免和引擎产生行为分叉。若审查者认为跨模块引用私有
   （下划线前缀）符号不妥，可以改为在 `nk_grid.py` 里去掉下划线公开，但那属于改动
   既有文件，本轮未做。

3. **删失机制用 `signal.SIGALRM`/`setitimer`**：只在 POSIX、且只在主线程里工作
   （macOS/Linux 满足，Windows 不支持）。方案没有规定用什么机制中止一个同步阻塞的
   `model.fit()`调用；这是一个合理选择,但如果集群运行环境是多线程调度或者
   `model.fit()`内部生成子进程,行为需要重新验证——特别是 lightgbm/xgboost
   在生产管线里是通过 `IsolatedProcessRunner` 隔离子进程运行的（见
   `nk_grid._run_native_model_cell_locked`），calibrate_cost.py **没有**复用该隔离
   子进程路径,而是直接在本进程内调用 `make_model`+`fit`,理由是简化实现、
   避免引入子进程 IPC 复杂度；代价是这两个模型的峰值 RSS 用 `RUSAGE_SELF`
   而不是方案 §4 指定的 `RUSAGE_CHILDREN` 测量，量级会偏低。已在代码里用
   `SUBPROCESS_MODELS` 常量标注、待审查。

4. **合成数据的类别比例/缺失率（待澄清问题 1 的回答）**：`continuous_fraction`
   固定为 0.622、one-hot 分组尺寸分布**直接采样自**
   `FFCWS/schema/ffc_median_missing_indicator.feature_universe.json` 里
   `sources` 数组中 `unit_type=="onehot_group"` 的真实分组尺寸列表（均值约 3.64，
   这是 schema 文件本身的结构统计量,不是私有数据里的取值）。**缺失率没有可查的
   依据**（真实缺失模式在 `FFCWS/data/`，本方案禁止读取）：本模块假设连续变量
   逐格 10% 缺失、one-hot 分组整组 8% 缺失，是**未经验证的占位假设**，请人工
   审查是否需要替换成更贴近真实缺失模式的数字（例如从 `FFCWS/adapter/` 的文档
   或论文里找一个已发表的缺失率范围）。

5. **待澄清问题 2（K∈{10,100,1000} 是否足以外推到 8053）无法回答**：
   这需要 Stage B 的实测 ratio 数据来判断，本轮没有该数据。工具已经支持任意
   `--stage-b-points`，一旦在集群上跑起来即可直接检验。

6. **待澄清问题 3（super_learner 是否需要单独成本形式）也无法回答**：
   `fit_all_models` 目前对 9 个模型使用同一种共享的幂律回归形式（不特殊处理
   `super_learner`），这是方案给出的默认做法（方案没有要求现在就做区分,只是
   提示"若阶段 A 的 R² 明显低,需要考虑"）；实际是否需要单独形式要等 Stage A
   数据出来后看 R² 才能判断，本轮无数据。

7. **`missing_rate_group`/`missing_rate_continuous`、`t_orders` 只测
   `(seed=0, draw=0)` 一次代表值**：`measure_t0` 中 `t_orders` 只测了一次
   `draw_orders` 调用（而不是像生产引擎那样每个 `(seed, draw)` 都测），
   与方案"每个 (seed, draw) 一次"的措辞不完全一致；由于 `draw_orders` 内部是
   纯 numpy 置换操作，跨 `(seed, draw)` 的耗时方差预期很小，用一次代表值加 5 次
   重复（跨 5 个全新进程）来估计分布，是为了控制测量总耗时的简化选择。

**没有其他偏离** —— 除上述 7 点外，`calibrate_cost.py` 的实现严格对齐方案 §2-§5
描述的分量、阶段、schema。

## 6. 未覆盖与已知风险

逐条对照方案的 9 个验收标准，说明本轮的验证状态（也是第 2 节的风险视角复述）：

| # | 验收标准 | 本轮验证状态 | 原因 |
|---|---|---|---|
| 1 | 不改变既有数值行为 | **已验证** | `pytest` 全绿，新模块未修改既有文件 |
| 2 | t0 四分量 + 10% 一致性 | **未验证** | 未运行实际测量；`measure_t0` 本身的子进程调用路径只在小合成数据上做过探索性冒烟（不计入标定），未做 5 次重复的正式统计，也没有对照"直接测得的端到端启动时间" |
| 3 | 每模型 c/a/b/R² | **未验证** | 未运行 Stage A；回归算法本身已用合成、已知真值的数据独立验证正确（见测试 1、2） |
| 4 | 外推校验 ratio∈[0.5,2] | **未验证** | 未运行 Stage B |
| 5 | 峰值 RSS 三系数 + R² | **未验证** | 未运行任何阶段；且即便跑了，lightgbm/xgboost 的 RSS 测量方式（`RUSAGE_SELF` 而非生产用的 `RUSAGE_CHILDREN`）也是已知的近似,见第 5 节第 3 条 |
| 6 | 删失点显式记录 | **机制已验证，无实测数据** | 单测证明删失路径正确；没有真实标定文件可核对"没有静默丢弃" |
| 7 | 标定文件可读、可复算回归 | **已验证（机制层面）** | round-trip 测试；没有生产标定文件可供人工再核对一次 |
| 8 | 总测量墙钟时间记录且 <24h 有说明 | **未验证** | 未运行任何测量，无墙钟时间数据 |
| 9 | 不含 metric 值 | **已验证** | 代码审查 + 测试均未涉及 r2_test/rmse 等字段 |

其他已知风险 / 未覆盖场景：

- 探索性冒烟测试（第 3 节末尾提到的数字）没有在生产线程环境（`OMP_NUM_THREADS=1`
  等）之外的设置下测过是否有明显差异；calibrate_cost.py 本身的环境守卫逻辑
  （`check_thread_env`/`enforce_thread_env`）只测过手工构造的假环境变量字典，
  没有在真实污染的 shell 环境里跑过一次完整 CLI 来确认它确实会在入口处退出。
- `--fallback-feature-units` 的降维路径（生成超时或超内存预算时切换到 1/4 维度）
  只在代码层面实现，未被单测直接覆盖（没有构造一个"生成必然超预算"的场景来断言
  fallback 触发、`dimension_fallback_triggered=True` 被正确写入 payload）；
  只做过一次真实的探索性验证（8053 源、7.6s、1.3GB，远低于阈值，所以 fallback
  分支实际没有被走到过）。这是一个已知的测试覆盖缺口。
- `SUBPROCESS_MODELS`（lightgbm、xgboost）峰值 RSS 用 `RUSAGE_SELF` 而不是
  `RUSAGE_CHILDREN`，如第 5 节所述，是已知偏差，未被修正也未被测试覆盖其偏差量级。
- Stage A/B 的实际运行时间、是否会触发 `--max-seconds` 删失、censored 点分布等，
  完全未知，留给集群实测。
- 只验证了 `internal_random` + `regression` 任务路径（合成 schema 固定
  `task="regression"`, `split_mode="internal_random"`）；`external_test`
  分割模式、`classification` 任务的 `load_input`/`validate_input`/`preprocess_cell`
  路径未被 `calibrate_cost.py` 或其测试触碰,如果 flat-task-table 需要覆盖分类任务
  的成本,需要另外扩展。

## 7. 给审查者的重点

1. **本轮"不产出标定文件"的范围收缩是否可接受**——这是与最初任务描述最大的出入，
   需要人工确认后续是"批准并安排集群跑一次"还是"仍要求本地先出一版能用的数字"。
2. **`SUBPROCESS_MODELS`/RSS 测量方式（第 5 节第 3 条）**：lightgbm、xgboost
   在生产里走隔离子进程,峰值 RSS 该用 `RUSAGE_CHILDREN`；本工具目前简化为进程内
   直接拟合、用 `RUSAGE_SELF`，测出来的 RSS 数字系统性偏低，需要确认这个简化在
   集群实测前是否要先修正,还是等实测阶段再评估影响。
3. **合成数据的缺失率假设（第 5 节第 4 条）**：`missing_rate_continuous=0.10`、
   `missing_rate_group=0.08` 是没有依据的占位数字，直接影响 K_unobserved 的分布,
   进而可能影响 preprocess_cost 的回归结果；如果有更可靠的缺失率来源（哪怕是
   一个已发表的粗略范围）,应该在集群实测前替换掉。

方案要求给出的补充结论句（第 7 节）：**由于本轮未测得 t0，暂时无法给出
`T_target` 的具体取值建议**（判据 `t0/T_target < 5%` 需要 t0 的实测中位数才能
代入计算，编造一个数字会违反"不得编造未测量的数字"的要求）；一旦集群上的
`measure_t0` 跑出 `t0_seconds.total.median`，`T_target` 应取
`t0_seconds.total.median / 0.05` 的量级（例如若集群 t0 中位数是 X 秒，
`T_target ≈ 20·X` 秒，使 t0 占比 < 5%），这个公式本身可以现在写进
`flat-task-table`，但数值代入必须等集群实测完成。
