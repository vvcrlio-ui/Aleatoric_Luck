# 计划:把 NK_Grid 拆成 general engine 与 upstream adapter(v5)

> 目标读者:Codex。本文件是实现规格。契约细节、插值状态机、experiment_id 规范化见配套 [`upstream-adapter-spec.md`](upstream-adapter-spec.md),以该文档为准。

## 修订记录

**v5 实现后加固(2026-07-24 code review)**
- outcome/ID 与 predictor 角色互斥；external ID 在 outcome 删除前验证。
- ordinal 契约收敛为有限数值 code；drop-first reference 平票顺序已钉死。
- XGBoost/LightGBM 回归 CV metric 固定 `rmse` 并在抽样前校验。
- schema 默认值规范化使身份算法升至 v3；完成 preset 可用 `rerun_completed=false` 显式复用。
- Slurm worker 以运行时 allocation 覆盖 `n_jobs`;Slurm 默认
  `rerun_completed=false`;固定 `--chdir`/日志/receipt 语义。
- 增加 checkpoint-boundary cooperative stop + USR1 显式 requeue、重启上限和
  array throttle。`--requeue` 本身不被误写成 TIMEOUT 自动续跑保证。
- Slurm 将同一冻结快照中的 panel×model 索引按 `parallel`/`serial`/`bart`
  分成至多三个独立 array；serial 默认 1 CPU，BART 使用独立资源配置，
  `--max-concurrent-per-class` 按类别分别生效（旧 `--max-concurrent` 拒绝），
  每个非空类别各写一份 receipt。
- USR1 后优先等待 batch checkpoint；默认 240 秒 watchdog 超时则从最后完整
  checkpoint 强制 requeue，可用 `--requeue-watchdog-seconds=0..240` 缩短；
  上限为调度器保留 60 秒处理 requeue。
  关键 `sbatch` 选项在命令行显式冻结；receipt 同时记录 signal/watchdog 和
  可影响模型的环境开关。
- production 热路径增加 `(seed, draw)` 有界抽样顺序缓存、resume 列投影、
  缺失感知预处理 fast path 与 loose/compact WAL 原子压实；这些实现优化不改变
  抽样、模型、指标、最终结果列或 `experiment_id`。
- 完成态复用要求 projected cell index 与当前设计精确相等；checkpoint/最终
  CSV 在替换旧 authority 前完成文件、逐级目录项落盘，并由 declared-output
  POSIX writer lease 排除并发写者。

**v5(回应第四轮审阅)**
- **K estimand = 设计口径**(审阅 1,已定):K = 抽中名义 source 数,不管是否带信息;"大 N 更多 source 可观测"是"更多数据→更强预测力"的合法部分,不剔除。`K_prior_filled`→**`K_unobserved`**(与模型无关的训练 cell 全缺失 source 数);引擎必须逐 cell 记录并暴露,不改主轴 K,聚合与敏感性分析属于下游任务。
- **全缺失 source 的 test 转换**(审阅 2):插值类模型把 train+test 该 source 都置同一先验常量;passthrough 两侧留 NaN。
- **per-source 先验 + ordinal/reference 字段**(审阅 3/4):`source_prior`/`ordinal_levels`/`is_reference` 入 manifest。
- **FFC manifest 迁移非"复制即可"**(审阅 6):现有 data_processor manifest 缺 `feature_manifest_version`/`feature_order`/`unit_type`/`drop_first`/`ordinal_levels`/`is_reference`/`source_prior`——必须**改 manifest 生成器 + 三套 strategy + 加迁移测试**,否则复制后过不了新契约。
- **legacy 列白名单硬编码**(审阅 11):验收 2 的比较列**钉死列名**,禁动态交集。
- **panels 字段集补全**(审阅 8)、**manifest_version 放 schema**(审阅 9)、**universe canonical 扩展**(审阅 10)、**keep=false 不强制存在**(审阅 7)——细节见 spec。

**v2–v4(历史)**:identity 升级+迁移;更正 model_params 已在 id;binary-only;`id_column`;schema 唯一位置;包名 `aleatoric_nk_grid`;两阶段列投影;严口径反泄漏;插值类型系统 + 先验占位;outcome 统一删除 + train/test 分阈值;failed 治理;复制而非移动;引擎行为变更边界。

## 目标
- **Part 1 engine** `NK_Grid/`(import 名 `aleatoric_nk_grid`):零领域知识,四篇共用,装协议。
- **Part 2 adapter** 每篇一文件夹:只产 ARD + schema。
- 判据:有拟合参数 / 需看某 cell 子样本 / 属实验协议 → engine;逐行确定性 / 跨 cell 一致 → adapter。

## 与 FFC_V1 的关系
FFC_V1=Phase A(合并两 fork+三移植项);本计划=Phase B(外提领域逻辑、确立契约、重构)。以 Phase A 合并引擎为起点,未实现则以 SMR 基线起。Phase B 独立保证 experiment_id 覆盖。

## 引擎行为变更边界
- **不变(有条件,审阅 5)**:SMR internal_random + continuous median 路径——**对每个选中 continuous source 至少有一个训练观测的 cell**,输出值与旧版逐行一致(验收 2,白名单)。**某 continuous source 在 cell 内全缺失的 edge case** 走 v5 先验占位 + test 强制转换,属**明确批准的新行为**,不要求与旧 `SimpleImputer` 一致。
- **经批准变更**:FFCWS 类型化插值(one-hot 原子、ordinal median_snap、全缺失先验占位、test 强制转换)是新模型输入算法,**必然改变 FFC 数值**,不追求与旧 FFC fork 一致。
- **新增引擎行为**:`validate_input`、experiment_id 规范化、outcome 统一删除、`K_unobserved` 记录、failed 治理。其余协议不动。

## 背景:隐式边界
| # | 假设 | 证据 | 违反 |
|---|---|---|---|
| 1 | 单表整体进内存 | 硬编码 `pd.read_csv`(约:905) | v5 Parquet 两阶段列投影 |
| 2 | 列名共享前缀 | `_predictor_columns`(:803–807) | 只能前缀 |
| 3 | 命中列全有限数值 | 无校验 | `run_one` `except` 转静默 failed 表 |
| 4 | predictor NaN 上游负责 | 仅 external `dropna`(:262);internal `split_frame:239` 不删 | outcome 口径不一致 |
已含并强化:`metadata_extra`(:963–980)已把 model_params 等进 id。v5 补 schema/imputer/predictor 有序/manifest hash + 规范化 + 版本号。

## 范围
### In-scope
1. `NK_Grid/src/aleatoric_nk_grid/` 建 ingest/contract 模块(read_table 两阶段投影 + schema 解析 + validate_input)。
2. `_predictor_columns` 泛化(columns 权威/缺省 prefix/并存报错)。
3. `validate_input()`(spec §11)。
4. 领域逻辑外提:FFC `data_processor/` 复制到 `FFCWS/adapter/` **并改造 manifest 生成器 + 三套 strategy 补 v5 新字段 + 迁移测试**(审阅 6);SMR 建近乎恒等 adapter。
5. `NK_Grid/` 提根做包,import 名 `aleatoric_nk_grid`、依赖版本锁。
6. 每篇自包含文件夹。
7. experiment_id 规范化 + `experiment_identity_version`(mode-aware)。
8. outcome 统一删除,train/test 阈值分开。
9. **`K_unobserved`** 逐 cell 记录并暴露(与模型无关;聚合/报告属下游);全缺失 source 先验占位 + **test 强制转换**(spec §7)。
10. failed 运行治理。
11. 产出 spec;SMR/FFCWS 双路径回归。

### Out-of-scope
- 不改 continuous 默认路径行为;不改 N/K 抽样、指标和统计诊断。原 v5
  不改科学协议;实现后审查批准了 Slurm 资源绑定、幂等重投、日志目录、
  checkpoint-boundary requeue，以及不改变输出的缓存、投影和 checkpoint
  存储/耐久性加固。
- 不把插值移出引擎、不把划分移进 adapter。
- 不删旧 fork;不实现 group-aware 划分/分层抽样/多分类/MICE-KNN。
- 不用 `max_jobs` 冒充 seed 分块;panel×model×seed shard、跨 shard identity/
  merge/final failure policy 仍另立设计。当前只实现三类粗粒度资源画像；
  基于实测 profiling 的逐模型 CPU/memory/time 优化属于后续工作。
- **`K_unobserved` 的职责仅限引擎逐 cell 记录并暴露;其按 N/K 的聚合、可视化、敏感性分析及论文解释属于下游分析任务,不在本次重构范围内。**

## 目标结构
```
Aleatoric_Luck/
├── NK_Grid/ src/aleatoric_nk_grid/{nk_grid,model_registry,experiment,run_panels,slurm_jobs,ingest,validate_input}.py
│           slurm/ tests/ requirements.txt(锁) pyproject.toml
├── SMR/    adapter/ schema/ panels.yaml model_params.yaml data/(ign) outputs/(ign) plans/
├── FFCWS/  adapter/{data_processor,configs,legacy} schema/ panels.yaml model_params.yaml data/(ign) outputs/(ign) plans/
├── Paper3_XXXX/ Paper4_XXXX/  docs/  README.md
```

## 技术方案
### 步骤 1:数据入口契约(engine)
- `ingest.read_table(path, columns=None)`:csv/parquet + **两阶段列投影**(先读 metadata/header 解析命中列含 id_column+未来 group,再只读所需)。
- `ingest.load_schema`:JSON Schema/Pydantic 严校验;`schema_version`/`feature_manifest_version` 未知 fail-fast;相对路径按 schema 目录解析。
- `ingest.resolve_predictors`:columns 权威/缺省 prefix/并存报错;有 manifest 按 `(source_order, feature_order)` 稳定排序。
- `validate_input`:spec §11 全部。
- `run_nk_grid()` 接线:读取→read_table;predictor→resolve_predictors;载入后→validate_input;**`split_frame` 补 `dropna(outcome)`**;experiment_id 规范化(mode-aware);结果行补 `K_unobserved`;**不改** draw order 的确定性结果、划分和指标语义；允许 run-local 缓存、执行分组和 WAL 存储结构等不改变最终协议的实现优化。
- 插值:全缺失 source 先验占位 + **test 强制置同一常量(插值类)/train+test 强制置 NaN(passthrough)**;`K_unobserved` 与模型无关。

### 步骤 2:领域逻辑外提(adapter)
引擎 `src/` 无领域残留(grep 自查)。FFC:复制 data_processor 进 FFCWS/ **且改造**——manifest 生成器与三套 strategy(median_mode / median_missing_indicator / tree_ordinal)在 CSV 中补 `feature_order`/`unit_type`/`drop_first`/`is_reference`/`reference_level`/`level_value`/`ordinal_levels`/`source_prior`(**`feature_manifest_version` 不进 CSV,由 schema 生成逻辑写入**,审阅 2),加迁移测试证明新 manifest 过 `validate_input`。旧 `prepare_ffc_*` 复制到 legacy/ 不接线。SMR 建 `prepare_smr.py` 主要写 schema。

### 步骤 3:仓库重构
`NK_Grid/` 提根,包名 `aleatoric_nk_grid`,依赖锁。每篇 panels/model_params 各一份;panels 只放旋钮(spec §13,含 `schema` 字段指向权威 schema),冲突 fail-fast。data/outputs gitignored。

### 步骤 4:K / failed 治理
- 插值 per-cell 类型化;全缺失 source 先验占位(train+test 同常量)、K 名义不变、记并暴露 `K_unobserved`;`K_unobserved==K`→skip 固定码。正式聚合/报告不由本次引擎重构实现。
- failed:run 末尾统一检查;分母 `failed/(ok+failed)`,skipped 不计,分母=0 判失败;`failed_count > failed_abs_threshold`(默认 50)OR `failed_ratio > failed_ratio_threshold`(默认 0.05)→非零退出、告警、checkpoint 保留、阈值入 manifest。两阈值可由 panels 覆盖(spec §13)。

### 步骤 5:Slurm 资源类别
- snapshot 仍冻结一份稳定的 panel×model 全局索引；提交器按模型将索引划入
  互斥的 `parallel`、`serial`、`bart` 三类，分别调用 `sbatch`，空类别跳过。
- `serial` 对应引擎中强制 outer-serial 的模型，默认每 task 1 CPU；
  `parallel` 使用通用 `--cpus-per-task`/`--mem`/`--time`；serial 通过
  `--serial-cpus-per-task` 覆盖 CPU 而沿用通用 memory/time；`bart` 可用
  `--bart-cpus-per-task`/`--bart-mem`/`--bart-time` 独立覆盖，未设置的
  BART 项回退到通用配置。
- `--max-concurrent-per-class` 是**每类 array**的节流值；三个类别同时存在时，
  全局同时运行数可能高于该值。旧 `--max-concurrent` 因语义含糊而 fail-fast。
  每个非空类别各有 Slurm job ID 和 receipt，receipt 只列该类别的全局
  snapshot 索引，并记录实际资源、节流和可执行 rerun 命令。
- worker 继续以实际 `SLURM_CPUS_PER_TASK` 覆盖配置中的 `n_jobs`；资源分类、
  并发节流和重跑策略均不进入 `experiment_id`。
- 提交前必须验证三类索引合并后无重无漏地覆盖 snapshot 全部 jobs。三次
  `sbatch` 不是原子事务：若后续类别提交失败，错误必须列出已经成功提交的
  `resource_class=job_id`；操作者先查 `squeue`，不得直接重跑整条提交命令，
  应以原 `--snapshot PATH --resource-class parallel|serial|bart` 只补投缺失
  类别，禁止从可能已变化的 manifest 重建恢复快照。
- `--requeue`、USR1 advance signal、日志路径/open mode、`--chdir`、环境导出、
  class 资源和 array spec 均通过 `sbatch` 命令行显式冻结，不依赖 worker
  文件中可能过期的默认 directive。receipt 记录 signal 提前量、0..240 秒的
  watchdog（默认 240）及所有可影响模型结果的可选环境开关和其 unset/value
  状态，逐 index rerun 命令须重建相同环境。
- 收到 USR1 后，引擎先 cooperative stop：等当前 batch 完成并落下原子
  checkpoint 后 requeue。若在 watchdog 时间内没有退出，worker 强制 requeue，
  放弃尚未完成的 batch，并从最后一份完整 checkpoint 恢复；重启次数上限仍生效。

## 验收标准
1. `NK_Grid/` 根包 import 名 `aleatoric_nk_grid`;旧两 fork 未删。
2. **SMR 回归(硬编码白名单,审阅 8/11)**:与现有 `SMR/NK_Grid` 相同 seed,**比较列为在测试源码中钉死的字面 `frozenset`**——测试须断言该白名单是**两侧列集合的子集**(任一侧缺白名单列即失败),然后**只截取白名单列比较**(允许两侧另有新 metadata 列,不参与比较);**禁止 `set(old.columns) & set(new.columns)` 动态交集**。字面清单 =
   - key/上下文列:`dataset, outcome, model, seed, draw, N, K, split_random_state, n_train_total, n_test_total, n_features_total`
   - 回归指标(把 `METRIC_COLUMNS` 的 30 个字面值拷入):`r2_test, skill_score_pct, rmse, mae, medae, max_error, nrmse, spearman_rho, pearson_r, kendall_tau, ccc, explained_variance, mean_bias, median_bias, pinball_q10, pinball_q90, d2_absolute_error, pinball_q05, pinball_q25, pinball_q50, pinball_q75, pinball_q95, ks_statistic, wasserstein_distance, top_decile_hit_rate, bottom_decile_hit_rate, rsr, cv_rmse, mase, pearson_r2`
   - 稳定诊断:`K_varying, underdetermined, constant_prediction, converged`
   - 状态:`status, error`
   - **排除**(新字段/计时/身份):`experiment_id, experiment_identity_version, K_unobserved, _fit_seconds, _best_rounds`、schema/imputation metadata。
   按 `(dataset,outcome,model,seed,draw,N,K)` 对齐、数值 `rtol=1e-9/atol=1e-12`、`status`/`error` 精确——值逐行一致;outcome 缺失用无缺失合成数据回归,删除语义单独测。
3. **FFCWS 回归**:external、改造后 manifest 过 validate、Parquet 投影、K 按 source 计、`K_unobserved` 记录、类型化插值生效(接受与旧 FFC 数值不同)。
4. **契约 fail-fast**:非数值/±inf/全 NaN predictor、非有限 regression outcome、缺 outcome/超 train 阈值、超 test 阈值、columns+prefix 并存、group 非 null、internal 非 fixed_a_priori 或 universe canonical hash 不符、external id 重叠/重复/缺失、未知 schema/manifest_version、行下限不足、manifest 缺新字段/source 内不一致 —— 各抽样前 raise,不产静默 failed 表。
5. **身份**:改 schema 语义/imputer/predictor→id 变;仅改 provenance→id 不变续跑;external 改 `test_size`→id 不变;identity_version 递增→不复用旧 checkpoint。
6. **binary-only**:非 `{0,1}` 报错。
7. **K/插值**:全缺失 source 先验占位、K 名义不变、**test 同置常量**、记 `K_unobserved`;`K_unobserved==K`→skip;`K_unobserved` 列存在且可被 `groupby` 聚合(**仅最小 sanity check,不做正式敏感性分析**);ordinal 无 0.5;onehot 原子无非法组合;并列按 `feature_order`。
8. **failed**:大量失败→非零退出;分母 `failed/(ok+failed)`、skipped 不计、分母 0 判失败;阈值入 manifest;checkpoint 保留。
9. **包隔离**:旧 fork 在 `PYTHONPATH`/cwd 时 `import aleatoric_nk_grid` 仍导根包;启动断言绝对路径。
10. 引擎无领域残留(grep);provenance 无原始 ID/绝对路径;受限数据不入库;全仓 pytest 通过。
11. **Slurm**:worker `n_jobs==SLURM_CPUS_PER_TASK`;旧/新 snapshot 在 worker
    默认不重算完成输出、显式 opt-in 才重算;任意 cwd 提交仍写 engine logs;
    同一 snapshot 的索引被无重无漏地划入 `parallel`/`serial`/`bart` 三类，
    每个非空类只提交一个 array，serial 默认 1 CPU、BART 资源可独立配置；
    `--max-concurrent-per-class` 分别写入每类 array spec，旧参数被拒绝；
    每类 receipt 只含本类索引并与实际 CPU/memory/time/throttle/rerun、
    signal/watchdog 和模型环境开关一致；关键 sbatch 选项全部由命令行显式给出；
    USR1 先等待 batch checkpoint，watchdog 默认/上限 240 秒（为 requeue
    保留 60 秒），超时从最后完整 checkpoint 强制 requeue，达重启上限不再
    循环；预提交覆盖检查失败时零
    submission，多 array 中途失败时列出已提交 job 并提示以 `squeue` 核实。

## 测试要求
双路径回归(SMR 硬编码白名单 / FFCWS 改造 manifest)· 契约每失败分支各一次 · predictor 三态 · 身份隔离(改语义变 / 仅 provenance 不变续跑 / external 改 test_size 不变 / identity_version 不复用)· 两阶段投影含 id · outcome 两路径删除 + 分阈值 + 行下限 · 插值状态机(全缺失先验占位 + test 同置常量、K_unobserved、K_unobserved==K skip、并列 tie-break、部分缺失非法、多 1 报错、drop_first 全 0 合法、ordinal 无 0.5、passthrough 两侧 NaN)· FFC manifest 迁移(新字段过 validate)· **prior-value invariance(改变全缺失 source 占位常量,当前注册表各模型预测不变;新增模型接入前必过此测试)**· K_unobserved 与模型无关(passthrough 也记)· **K_unobserved 可聚合的最小 sanity check(非敏感性分析,后者属下游任务、out-of-scope)**· binary 多类别报错 · failed(分母/skipped/分母 0/非零退出)· 包隔离 · Slurm runtime CPU/幂等重投、三类完整分区/逐类 throttle/逐类 receipt、USR1 cooperative-stop/watchdog fallback、显式 sbatch 冻结、环境可复现和 partial-submit shell 集成测试 · 合成数据不复制私有值。

## 风险与注意
1. Phase B 非纯结构重构:continuous 路径不变,FFC 类型化插值经批准变更,验收分别处理。
2. 划分归属只能引擎;插值归属只能引擎 per-cell。
3. **全缺失 source 的 test 必须同置常量**(插值类):否则模型在没学过的变量真值上外推,线性/NN 尤其不可解释。
4. experiment_id 扩展与 ingest 同批;mode-aware 排除 test_size;排除时间戳;不复用旧 checkpoint。
5. 每篇一律有 adapter,SMR 不例外。
6. **FFC manifest 必须改造非复制**:现有字段不足以过 v5 契约(审阅 6)。
7. 复制而非移动 data_processor 本体;旧两 fork 字节不动作回退。
8. failed 可执行阈值;包名 `aleatoric_nk_grid` + 启动路径断言;Parquet 两阶段投影。
9. **白名单硬编码**:验收 2 比较列钉死字面清单,禁动态交集(否则意外缺列被静默排除)。
10. binary-only;常量列不破坏 StandardScaler(零方差 scale=1);先冒烟后规模(preset: dev)。

## 交付
代码位于根 `NK_Grid/`(import 名 `aleatoric_nk_grid`)与各文章文件夹;配套 spec 同批;PR 说明 SMR(白名单逐行一致)/FFCWS(接受数值变更 + manifest 迁移)双路径回归、`grep` 自查、failed 阈值、包隔离、身份迁移、`K_unobserved` 列存在的最小 sanity check(非敏感性分析);旧 fork 保持不动。
