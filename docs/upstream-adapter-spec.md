# Upstream Adapter 设计规范(v5)

> 配套计划:[`split-nk-grid-engine-adapter.md`](split-nk-grid-engine-adapter.md)。
> 本文档定义 **adapter 与 engine 之间的契约**。每篇文章接入 NK_Grid,只需按本规范写一个 adapter。

## 修订记录

**v5 实现后加固(2026-07-24 code review)**
- predictor 与**全部** `outcome_columns`、`id_column` 必须角色互斥；命中即报错，不静默剔除。
- ordinal 落盘值与 `ordinal_levels` 统一为有限 JSON number；文本等级由 adapter 先确定性编码。
- schema 省略值与显式默认值先规范化再 hash；身份算法升为 `experiment_identity_version=3`，不续旧 checkpoint。
- drop-first one-hot 众数平票时，reference 作为被省略的首类别，先于所有 kept dummy。
- preset 默认保留历史“完成后再跑生成新输出”；`rerun_completed=false` 可显式复用已完成结果。
- Slurm 路径以 allocation 覆盖 `n_jobs`、默认复用完成结果，并用
  checkpoint-boundary USR1/requeue 保留长任务进度；这些调度字段不进入实验身份。
- Slurm 依模型执行特性提交 `parallel`/`serial`/`bart` 三类独立 array：
  serial 默认 1 CPU，BART 资源可独立覆盖；并发节流和 receipt 均按类别管理。
- 每类节流参数正式命名为 `--max-concurrent-per-class`，旧
  `--max-concurrent` 拒绝；USR1 的 batch-checkpoint 等待有默认 240 秒
  watchdog，超时从最后完整 checkpoint 强制 requeue。
- checkpoint 采用原子 WAL shard + 有界 loose/compact 压实；resume 只投影
  身份/key/status。缓存、压实、writer lease 与调度覆盖均不进入实验身份，
  且不得改变抽样、模型、指标或最终结果协议。

**v5(回应第四轮审阅)**
- **K 的 estimand 明确 = 设计口径**(审阅 1,已定):K = 抽中的**名义 source 数**,无论是否带信息(实验操作量:"交给模型 K 个随机 source + N 行")。研究目标是展示"N 增大时可预测性增强",其中"大 N 时更多 source 变得可观测"是该现象的**合法组成部分,非需剔除的混杂**。`K_prior_filled` 改名 **`K_unobserved`**,定义为**与模型无关**的"训练 cell 内全缺失 source 数";引擎必须逐 cell 记录并暴露,但不改变主轴 K、无需控制掉。聚合、敏感性分析与论文解释属于下游任务。见第 7 节。
- **全缺失 source 的 test 转换**(审阅 2):source 在训练 cell 全缺失时,**插值类模型把 train 和 test 的该 source 都强制置同一先验常量**(模型没学过它,不得在 test 真值上外推);**passthrough 模型两侧都留 NaN**(NaN-native 自然忽略)。
- **per-source 先验**(审阅 3):`continuous_prior` 从全局单值改为 **manifest 逐 source 声明**;量纲不同的连续变量各自定中性先验。
- **ordinal/one-hot 缺失字段补全**(审阅 4):manifest 加 `ordinal_levels`、`reference_level`/`is_reference`。
- **`K_unobserved` 与模型无关**(审阅 5):定义为训练 cell 全缺失 source 数(数据属性,非模型属性);passthrough 也照记。
- **`keep=false` 不强制存在**(审阅 7):存在性/orphan 校验只对 `keep=true` 行。
- **panels 合法字段集补全**(审阅 8):加 `name`/`schema`/`test_size`/`model_params`/`allow_large_run`/`dry_run`/`bart_min_n`/`bart_min_k`/failed 覆盖。
- **`feature_manifest_version` 放 schema**(审阅 9):不再用 CSV 表级 metadata,消除歧义。
- **universe canonical 对象扩展**(审阅 10):含 source/feature/类别值/ordinal 映射/reference/顺序,不止列名。
- **legacy 列白名单硬编码**(审阅 11):测试须钉死列名,禁动态交集。见计划验收 2。

**v2–v4(历史)**:严口径反泄漏;插值类型系统 + 先验常量占位(K 名义不变);身份规范化(mode-aware、排除时间戳、`experiment_identity_version`);schema 唯一位置;`id_column`;binary-only;包名 `aleatoric_nk_grid`;两阶段列投影;group fail-fast;outcome 统一删除 + train/test 分阈值;regression outcome 校验;行下限公式;failed 治理;manifest source 内一致性。

## 1. adapter 是什么 / 不是什么
**是**:每篇文章各自持有、可任意脏、带领域知识的预处理。raw → **ARD** + **schema**。
**不是**:不是引擎一部分;不做有拟合参数操作(插值/标准化统计量→引擎 per-cell);不做划分计算(→引擎);不做抽样/建模/指标/checkpoint(引擎协议)。

## 2. 唯一职责
```
版本控制:  <article>/schema/<dataset>.json    ← schema 唯一权威(panels 引用、被 hash)
raw ──▶ [adapter] ──▶ <article>/data/ard/<dataset>/   (gitignored)
                        ├── data.parquet / test.parquet(仅 external)
                        ├── feature_manifest.csv(满足 4.3 必需条件时)
                        └── provenance.json(来源 + schema/universe hash;不进 id)
```
schema 不放 ARD 内;provenance 记 schema hash,`validate_input` 交叉核对。

## 3. 契约 8 条
1. 表可被 `read_table` 读入。
2. predictor 规则唯一(columns 或 prefix,不并存)。
3. predictor 全有限数值;无 `±inf`/object/复数/全 NaN 列;类别缺失留 NaN。
4. outcome 存在;regression outcome 有限数值、无 ±inf、非 object;分类 binary `{0,1}`;删 NaN 后 ≥ 行下限(§11)。
5. external_test 结构一致(含 `id_column`;无 ID 重叠/重复/缺失;类别被 train 覆盖)。
6. 特征宇宙/词表来源合法(external train-only 冻结;internal fixed_a_priori + 可审计,§8)。
7. `group_column` 为 null。
8. 声明 `schema_version=1`;有 manifest 时 schema 声明 `feature_manifest_version=1`;不认即 fail-fast。

## 4. ARD 格式

### 4.2 `schema.json`(唯一权威)
| 字段 | 类型 | 必填 | 含义 |
|---|---|---|---|
| `schema_version` | int | ✅ | 合法值 `1` |
| `feature_manifest_version` | int/null | ✅ | 有 manifest 时合法值 `1`(审阅 9,放此不放 CSV) |
| `dataset` | str | ✅ | |
| `table`/`test_table` | str/null | ✅ | 相对 schema 目录解析 |
| `split_mode` | str | ✅ | internal_random / external_test |
| `task` | str | ✅ | regression / classification(仅 binary `{0,1}`) |
| `outcome_columns` | list | ✅ | |
| `id_column` | str/null | ✅ | external_test 必填 |
| `predictor_columns`/`predictor_prefix` | list/null | ✅* | 恰好一个 |
| `feature_manifest` | str/null | ✅ | 满足 4.3 必需条件时必备 |
| `exchangeable` | bool | ✅ | 抽样单位粒度 |
| `feature_universe` | object | ✅ | `{mode, definition_file, definition_sha256}`(§8) |
| `group_column` | null | ✅ | |
| `imputation` | object | ✅ | 状态机(§7);策略 + 全局回退默认;逐 source 先验在 manifest |
| `max_train_outcome_missing_ratio` | float | ⭕ | 默认 `0.5`,范围 `[0,1]`;train 侧 outcome 删除比例上限(审阅 1) |
| `max_test_outcome_missing_ratio` | float | ⭕ | 默认 `0.5`,范围 `[0,1]`;test 侧上限 |
| `continuous_priors` | object/null | ⭕ | 无 manifest 时的逐 source continuous 先验映射 `{source: value}`;缺省值及其模型适用条件见 §7 |

schema **不含** provenance/时间戳。

### 4.3 `feature_manifest.csv`
**必需条件(任一)**:source-level K;one-hot group;ordinal 类型化插值;任一列需非默认插值。即**有非 continuous 特征即必须提供**。

| 列 | 约束 |
|---|---|
| `source_column` | source 标识 |
| `feature_name` | 展开列名;`keep=true` 行须全表唯一且存在于数据 |
| `keep` | bool;false 仅审计,**不参与存在性/orphan/K 校验**(审阅 7) |
| `source_order` | int;source 间顺序;不同 source 不重复 |
| `feature_order` | int;组内顺序;组内唯一连续 |
| `unit_type` | continuous / ordinal / onehot_group |
| `drop_first` | bool;仅 onehot_group;同 source 一致 |
| `is_reference` | bool;仅 onehot_group;标记 non-drop-first 的 reference 特征(审阅 4);全组恰一个(drop_first=true 时可无,全 0 即 reference) |
| `reference_level` | 该 onehot_group 被省略/参考类的原始类别值(审阅 3);drop_first=true 时用于恢复被省略的 reference 类,否则记 `is_reference` 特征对应类别 |
| `level_value` | 每个 dummy 对应的原始类别值(审阅 3);使 canonical universe 能重建类别值集合;仅 onehot_group |
| `ordinal_levels` | 该 ordinal source 的合法等级有序集(审阅 4);仅 ordinal;**序列化为 canonical JSON 数组字符串**(审阅 7):元素必须是有限 JSON number(不接受 bool/字符串)、保序、无重复、非空;文本等级须由 adapter 先确定性映射为数值 code |
| `source_prior` | ⭕ 该 source 的先验占位值(审阅 3):continuous=常量;ordinal=某合法等级;onehot_group 用 reference。**可选**——仅在当前批准注册表并满足 §7 前提时不改变预测;新增模型须先过 invariance 测试 |

**source 内一致性**:同 source `unit_type`/`drop_first` 一致;continuous/ordinal source 恰一 `keep=true` 特征;onehot_group ≥1 且恰一 reference(或 drop_first);`keep=true` 全覆盖 predictor、无 orphan。

## 5. 边界表(同 v4)
无参数逐行变换→adapter;编码词表→adapter(受 §6 契约);特征宇宙筛选→adapter(受 §6);有参数填补(median/组原子/先验占位)→engine per-cell;划分机制→engine;官方 test 供给→adapter 提供、engine 消费;N×K 子采样→engine。

## 6. 准入 `exchangeable`
判在抽样单位(有 manifest 为 source unit)。true→跑;false→fail-fast。

## 7. 插值状态机 + K 语义

**K 的 estimand(设计口径,已定)**:K = 抽中名义 source 数,**不管是否带信息**。"大 N 更多 source 可观测"是"更多数据→更强预测力"现象的一部分,**不剔除**。引擎必须逐 cell 记录并暴露 `K_unobserved`(训练 cell 内全缺失 source 数,**与模型无关**),但**不改变主轴 K**;其聚合、可视化、敏感性分析及论文解释属于下游任务。

**schema.imputation**:
```json
{
  "continuous": "median",         // ∈ {median, mean}
  "ordinal": "most_frequent",     // ∈ {most_frequent, median_snap}
  "onehot_group": "atomic_mode",
  "model_overrides": { "lightgbm": "passthrough", "xgboost": "passthrough" }
}
```
逐 source 先验在 manifest `is_reference`/`reference_level`/`ordinal_levels`,可选 `source_prior`(审阅 3/4)。无 manifest 时全 continuous,先验取 schema `continuous_priors[source]`,未列出则默认 `0`。

> **关于"0 未必中性"(回应审阅 3,含更正)**:先验值仅用于全缺失回退,占位后整列恒为常量。**注意:常量列的取值并非对所有模型都无影响**——无截距、不居中的正则化模型会依赖它。反例(单常量特征 c 的无截距 Ridge):`β̂ = cΣy/(nc²+λ)`,预测 `βc = c²Σy/(nc²+λ)`,显然依赖 c;无截距 Logistic、某些正则化 NN、未来接入的原始特征模型亦可能如此,极大常量还可能浮点溢出。
> **在当前批准的模型注册表中安全**:只要 train/test 强制使用**同一有限先验常量**,并沿用现有管线——线性/NN 路径经 `StandardScaler` 把常量居中为 0;RF/ExtraTrees/XGBoost/LightGBM/BART 不用常量列分裂——占位值**不影响预测**。(注册表当前无 kNN;若未来接入,该维 train/test 相同则对距离贡献为 0,仍安全。)故默认 `0` 安全,`source_prior`/`continuous_priors` 仅作可读性/审计的可选覆盖。
> **门槛**:新增模型必须先通过 **prior-value invariance 测试**(改变全缺失 source 的占位常量,预测不变)方可接入注册表。

**正常 per-cell(子样本 ≥1 观测)**:continuous→子样本 median;ordinal→most_frequent 或 median_snap(吸附到 `ordinal_levels` 中最近**已观测**等级,并列取小,**绝不产 0.5**);onehot_group→atomic_mode(子样本众数类别,并列取 `feature_order` 最小;drop_first 的 omitted reference 视为首类别,与 dummy 平票时 reference 胜出)。

**全缺失回退(训练 cell 内该 source 零观测)**:
- 用 manifest 声明的**先验固定值**占位(非从数据估):onehot_group→reference(drop_first 全 0,否则 `is_reference` 特征置 1);continuous→`source_prior`;ordinal→`source_prior` 等级。成常量列,**计入 K → K 名义不变**。
- **test 一并强制转换(审阅 2)**:插值类模型对该 source 把 **train 与 test 都置同一先验常量**(模型没学过它,禁止在 test 真值上外推)。**passthrough 模型:把该 source 在 train 与 test 都强制置 NaN**(审阅 6:即使 test 原有观测值也覆盖为 NaN,非"仅保留原缺失",确保它对预测零贡献)。
- 记 `K_unobserved += 1`(与模型无关:是否 passthrough 都记)。
- **`K_unobserved == K`** → 模型零训练信号 → `status="skipped"`,固定码 `all_selected_sources_unobserved`。

**输入合法性(onehot_group,每行每 source)**:整组全 NaN(缺失)或合法指示(drop_first=false 恰一个 1;drop_first=true 至多一个 1、全 0=reference)。部分 NaN 混合、多个 1 → fail-fast。

> 常量列不得破坏 `StandardScaler`(零方差置 scale=1)。

## 8. internal_random 严口径 + 可审计声明
internal 下特征宇宙/词表必须 `feature_universe.mode=="fixed_a_priori"`,并:固定定义存**版本控制文件**、路径+SHA-256 入 schema;adapter 测试证明扰动行级数据不改固定定义;provenance 记 hash。
- **canonical 对象(审阅 10)**:不止列名——须含 **source、feature、类别值集合、ordinal 映射、reference 指定、source/feature 顺序**,做规范化序列化(UTF-8 排序、无多余空白)再 SHA-256;`validate_input` 从实际 resolved 定义重算比对。
- 需数据驱动 → 改 external_test 或预划固定 holdout。

## 9. 写新 adapter checklist
判 exchangeable → 判 split_mode → prepare(编码留 NaN、词表合规、宇宙合规、生成含新字段的 manifest)→ internal 存固定定义文件 + 扰动测试 → 写 schema(全字段、版本、group_column=null、external 填 id_column、不含 provenance)→ 写 provenance → 落 ard/ → panels 加旋钮面板(§13)→ dev 冒烟。

## 10. 两参考实现
SMR:internal_random / fixed_a_priori / prefix / 无 manifest / id_column=null / continuous median(**行为不变**)。
FFCWS:external_test / train_pool_screened / 显式 columns / **有 manifest(须补 v5 新字段)** / id_column 必填 / 类型化 + 先验占位(**经批准变更**)。

## 11. `validate_input()` 清单
**通用**:read_table 可读;`schema_version`/`feature_manifest_version` 被认识 + provenance schema hash 一致;schema↔panels 无字段冲突(§13);predictor 规则唯一;predictor 有限数值/无±inf/非全NaN;**regression outcome 有限数值/无±inf/非object**、分类 binary;outcome 删 NaN 后 ≥ 行下限、train 删除比例 > `max_train_outcome_missing_ratio`(默认 0.5)报错;manifest(必需时)满足 4.3(`keep=true` 才校验存在/orphan);`exchangeable`;`feature_universe` internal 必须 fixed_a_priori 且 canonical hash 一致;`group_column==null`;身份投影完整(附录)。
**行下限**:用 `train_test_split` 一致取整(或先 split 后验 `len(train)`),internal 与 external train 都要求 `n_train ≥ max(min_n, 选中模型 REGRESSION_CV_MIN_N)`;分类每类 ≥ CV 折数。
**external 附加**:test 含全 predictor+outcome+id_column、dtype/编码一致、删 NaN 后有行、test 删除比例 > `max_test_outcome_missing_ratio`(默认 0.5)报错;id 无重叠/重复/缺失;记 test 内容 hash;类别被 train 覆盖。
> per-cell 退化(常量列、单类别、小 N < CV、`K_unobserved==K`、passthrough NaN)不在 validate,由引擎 per-cell 跳过/诊断处理。

## 12. 两阶段列投影
先读 metadata/header 得列名 → schema 解析命中 predictor(prefix 展开)→ 只读 `outcome+predictors+id_column`(+未来 group)。禁整表读。

## 13. panels 合法字段集(审阅 8,补全)
- **panels 拥有**:`name`、`schema`(指向唯一权威 schema)、`preset`、`models`、`model_params`、`seed`、`n_seeds`、`n_draws`、`n_sizes_n`、`n_sizes_k`、`min_n`、`max_n`、`max_k`、`batch_size`、`n_jobs`、`test_size`(仅 internal 生效)、`allow_large_run`、`dry_run`、`rerun_completed`(交互/panel 默认 true,设 false 时复用同 identity 的完成输出;Slurm worker 为安全重投默认覆盖为 false,显式 `--rerun-completed` 才重算)、`bart_min_n`/`bart_min_k`(兼容 SMR)、`failed_abs_threshold`(int,可选)、`failed_ratio_threshold`(float,可选)、从 `schema.outcome_columns` 选一的 `outcome`、`out`。
- **schema 拥有(panels 出现即报错)**:`schema_version`、`feature_manifest_version`、`dataset`、`table`/`test_table`、`task`、`split_mode`、`outcome_columns`、`predictor_*`、`feature_manifest`、`exchangeable`、`feature_universe`、`group_column`、`imputation`、`id_column`、`max_train_outcome_missing_ratio`、`max_test_outcome_missing_ratio`、`continuous_priors`。

## 14. Slurm 执行策略
- 一份只读 snapshot 仍定义完整、稳定的 panel×model 全局索引。提交器按模型把
  索引无重无漏地划入 `parallel`、`serial`、`bart` 三个互斥资源类，并为每个
  非空类分别提交一个 array；空类不产生空 job。
- `serial` 是引擎注册为 outer-serial 的模型，默认每 task 1 CPU；
  `parallel` 使用通用 `--cpus-per-task`/`--mem`/`--time` 画像；serial 用
  `--serial-cpus-per-task` 覆盖 CPU、沿用通用 memory/time；`bart` 可用
  `--bart-cpus-per-task`/`--bart-mem`/`--bart-time` 独立覆盖，未设置的
  BART 项回退到通用画像。worker 最终仍以该 task 的
  `SLURM_CPUS_PER_TASK` 覆盖 snapshot 中的 `n_jobs`。
- `--max-concurrent-per-class` 分别应用于每类 array，不代表三个 array 合计
  的全局上限；旧 `--max-concurrent` 必须拒绝，不能静默解释成新参数。
  每个非空类各写一份 receipt，记录该类 Slurm job ID、全局 snapshot 索引、
  实际资源、节流/重跑策略及逐 index rerun 命令。
- 提交第一类前，三类索引的并集必须无重无漏覆盖 snapshot 全部 jobs，否则
  零提交并报错。多个 `sbatch` 调用本质上非原子：若中途失败，提交器明确列出
  已提交的 `resource_class=job_id`，操作者必须先用 `squeue` 核实状态，只补投
  缺失类别，不能直接重跑整组；补投入口为
  `--snapshot PATH --resource-class parallel|serial|bart`，必须复用原冻结快照，
  不得从可能已变化的 manifest 重建。
- 关键调度语义必须在 `sbatch` 命令行显式冻结，包括 requeue、USR1 advance
  signal、日志路径/open mode、工作目录、环境导出、class 资源与 array spec，
  不依赖 worker 脚本内可能过期的默认 directive。
- USR1 到达后先等待当前 batch 完成并写原子 checkpoint，再显式 requeue；
  `--requeue-watchdog-seconds` 接受 0..240，默认 240；该上限在 300 秒
  advance signal 窗口中固定保留 60 秒给调度器处理 requeue。watchdog 到期而
  worker 尚未 cooperative stop 时，强制 requeue 当前 element，未完成 batch
  丢弃，下次从最后完整 checkpoint 恢复。
- 原子 checkpoint/compact/final CSV 必须先 fsync 文件并逐级持久化新目录项，
  才能删除或隐藏旧 authority；同一 declared output 只允许一个持有 POSIX
  advisory lease 的 writer。完成结果的快速复用必须核对 projected cell index
  与当前 jobs 精确相等（行数、唯一 key、合法完成状态、无设计外 key），不能
  只信 manifest 计数；manifest 声称 complete 而索引不精确时必须 fail-fast，
  不得成功退出后形成永久慢恢复循环。
- receipt 除每类资源/节流/重跑策略外，还记录 advance-signal/watchdog 设置，
  以及所有可影响模型结果的可选环境开关的 unset/value 状态；receipt 生成的
  rerun 命令必须显式重建该环境，避免重投时继承登录节点的新值。
- 资源类是稳定、粗粒度的调度策略，不改变模型参数或 `experiment_id`。
  panel×model×seed block sharding、跨 shard 合并/最终失败治理仍不在本次范围；
  依据实测 profiling 继续细分逐模型资源画像也属于后续优化。

---

## 附:experiment_id 规范化
**(a) 覆盖**:`kind`·`algorithm_version`·训练数据 hash·external test hash·`outcome`·`split_seed`·`split_mode`·group 语义·**resolved model_params(env 覆盖后)**·**imputation spec**·**生效 predictor 有序列表**·**feature_manifest 内容 hash**·**schema 语义 hash**·**`experiment_identity_version`**。(更正:model_params 已在 id via `nk_grid.py:978`。)
**(b) mode-aware**:`test_size` 仅 internal 参与;external 排除。路径排除/规范化。
**(c) 去时间戳**:排除 `created_at`/绝对路径/provenance。
**(d) 身份升级不续旧**:回归比较排除 `experiment_id`;identity_version 递增即不复用旧 checkpoint。
