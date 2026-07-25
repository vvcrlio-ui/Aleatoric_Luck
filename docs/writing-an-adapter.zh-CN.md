# 为 NK Grid 引擎编写 adapter

*English version: [`writing-an-adapter.md`](writing-an-adapter.md)*

本指南讲解如何把一份新数据集接入共享的 `aleatoric_nk_grid` 引擎。写给已经有
数据集和研究问题、但不打算修改引擎的人。

规范性契约是 [`upstream-adapter-spec.md`](upstream-adapter-spec.md);本文档
是这份契约的实操路径。两者不一致时,以 spec 为准。

## adapter 是什么

引擎对你的数据一无所知。它只接收一个 **schema** 文件,并从中推导出其余一切:
表在哪里、哪些列是 outcome、哪些是 predictor、如何划分、如何插值。你的
adapter 唯一的工作就是产出这份 schema,外加一份分析就绪数据集(ARD)。

这个分工是严格的,原因在于引擎的结果必须能在各篇文章之间互相比较:

| 你的(adapter) | 引擎的 |
|---|---|
| 逐行确定性变换 | 任何带拟合参数的操作 |
| 类别编码、词表 | 插值(median、mode、先验) |
| 特征宇宙的选取 | 训练/测试划分 |
| 写 schema + provenance | N×K 子抽样、模型、指标、checkpoint |

判断归属的标准是:**如果它需要看子样本,或者带一个从数据估计出的参数,那就是
引擎的活。** 用中位数插值之所以是引擎的工作,正是因为这个中位数必须在每个
(N, K) cell 内部重新计算——提前算一次会跨 cell 泄漏信息。

## 写代码之前:三个决策

**1. 你的抽样单位是什么?** 引擎抽样的是 K 个 *source*,不是 K 列。如果某个
真实世界的变量展开成了多列(一个分类变量变成五个 dummy),这些列就是一个
source,必须一起被抽中。如果每一列都是独立的变量,你就不需要写 manifest。

**2. 内部划分还是外部划分?** `internal_random` 表示引擎按种子对你的表做划分。
`external_test` 表示你提供一份官方的留出测试表(就像 Fragile Families
Challenge 那样),引擎永远不会重新划分。外部模式要求提供 `id_column`。

**3. 你的所有 predictor 都是连续变量吗?** 如果是,你可以完全跳过 feature
manifest。如果有 ordinal 或 one-hot 变量,manifest 就是必需的——引擎正是靠它
才知道哪些列该一起移动、每种类型该怎么插值。

两个参考 adapter 划出了难度的两端:

- [`SMR/adapter/prepare_smr.py`](../SMR/adapter/prepare_smr.py) —— 170 行,
  内部划分、前缀选列、全连续、无 manifest。从这里开始看。
- [`FFCWS/adapter/`](../FFCWS/adapter/) —— 外部测试集、显式列清单、三种编码
  策略、带 one-hot 组和 ordinal 等级的 typed manifest。

## 目录结构

在仓库根目录下,每篇文章建一个文件夹:

```text
YourArticle/
├── adapter/            你的预处理代码(可以写得随意)
├── schema/             版本控制:<dataset>.json + feature universe
├── data/               gitignored
│   ├── private/        原始输入
│   └── ard/<dataset>/  data.csv、test.csv、feature_manifest.csv、provenance.json
├── outputs/             gitignored
├── panels.yaml         运行旋钮
└── model_params.yaml   模型超参数
```

schema 纳入版本控制;数据永远不纳入。schema 通过相对于 schema 所在目录的路径
来引用 ARD。

## 步骤 1 —— 把原始数据投影成 ARD

读取你的原始数据源,只保留 outcome 和 predictor 列,写成 CSV 或 Parquet 文件。
这里的每一步都必须是**逐行确定性**的:第 *i* 行写出的值只能依赖第 *i* 行自身
的原始值,绝不能依赖其他行。

缺失值留作 `NaN`。不要在这一步做插值——那是引擎逐 cell 的工作。predictor
最终必须是数值型;把类别编码成 dummy 或整数码,但缺失的类别要留 `NaN`,不要
臆造一个等级。

```python
header = pd.read_csv(source, nrows=0).columns.astype(str).tolist()
predictors = [c for c in header if c.startswith(("Aset", "Bset"))]
projected = pd.read_csv(source, usecols=[*outcomes, *predictors])
projected.to_csv(ard_dir / "data.csv", index=False)
```

对外部测试模式,用同样的方式写 `test.csv`,列和编码都要与 train 完全一致,
两侧都要带上 ID 列。

## 步骤 2 —— 写 feature manifest(仅当你有 typed 特征时需要)

每个展开后的列一行。引擎要求的列:

| 列 | 含义 |
|---|---|
| `source_column` | 这一列来自哪个真实世界的变量 |
| `feature_name` | 该列在 ARD 中的列名 |
| `keep` | 参与建模的列为 `true`;`false` 的行仅供审计 |
| `source_order` | 整数,每个 source 唯一——固定 source 间的顺序 |
| `feature_order` | 整数,在一个 source 内唯一且连续 |
| `unit_type` | `continuous`、`ordinal` 或 `onehot_group` |
| `drop_first` | 仅用于 `onehot_group`;同一 source 内必须一致 |
| `is_reference` | 当 `drop_first` 为 false 时,标记作为参考类的 dummy |
| `reference_level` | 作为参考类的原始类别值 |
| `level_value` | 每个 dummy 对应的原始类别值 |
| `ordinal_levels` | 合法等级的有序集合,以 canonical JSON 数组表示 |
| `source_prior` | 可选的占位值,用于该 source 全缺失时的回退 |

引擎强制执行的一致性规则:同一 source 内 `unit_type` 唯一;`continuous`/
`ordinal` source 恰好有一个 kept 特征;one-hot 组要么 `drop_first=true`,要么
恰好有一行 `is_reference`;`keep=true` 的行必须精确覆盖已解析的 predictor,
不能有孤儿行。

有两个细节经常绊倒人。`ordinal_levels` 必须是 *canonical* JSON——是
`[1,2,3]`,不是 `[1, 2, 3]`——因为这个字符串会被哈希进 feature universe。而且
ordinal 等级必须是有限数值;如果你的等级是标签,请在 adapter 里把它们映射成
整数码,标签本身留在 `level_value` 或你自己的文档里。

你可以为自己的审计需要添加额外的列。FFCWS 带了 `prevalence`、`strategy` 和
`mapping_id`;引擎会忽略它不认识的列。

## 步骤 3 —— 生成 feature universe

这是对你的 source、feature、类别值和顺序的一份规范化描述,会被哈希进 schema。
它的存在是为了让一次运行能够证明:它实际解析出的特征空间,就是你所声明的那
一个——在 `internal_random` 下尤其重要,因为没有别的机制能阻止一个依赖数据的
特征选择在多次运行之间悄悄改变。

用引擎自带的函数来生成它,而不是手写这个结构,这样它就不会与 validation 重新
计算出的结果产生偏差:

```python
from aleatoric_nk_grid.preprocessing import source_groups
from aleatoric_nk_grid.validate_input import canonical_feature_universe
from aleatoric_nk_grid.ingest import canonical_json

groups = source_groups(predictors, manifest, continuous_priors={})
universe = canonical_feature_universe(predictors, groups, manifest)
definition_path.write_text(canonical_json(universe))
```

然后把它的 SHA-256 记录进 schema。加载时引擎会根据你实际的 manifest 和
predictor 重新计算一次 universe 并比对;任何不一致都会在抽样开始之前失败。

## 步骤 4 —— 写 schema

schema 是唯一的语义权威。每个字段都是必填的(标记为可选的三个字段可以省略,
但没有任何未知字段会被容忍):

```json
{
  "schema_version": 1,
  "feature_manifest_version": null,
  "dataset": "my_dataset",
  "table": "../data/ard/my_dataset/data.csv",
  "test_table": null,
  "split_mode": "internal_random",
  "task": "regression",
  "outcome_columns": ["y1", "y2"],
  "id_column": null,
  "predictor_columns": null,
  "predictor_prefix": ["X_"],
  "feature_manifest": null,
  "exchangeable": true,
  "feature_universe": {
    "mode": "fixed_a_priori",
    "definition_file": "my_dataset.feature_universe.json",
    "definition_sha256": "…"
  },
  "group_column": null,
  "imputation": {
    "continuous": "median",
    "ordinal": "most_frequent",
    "onehot_group": "atomic_mode",
    "model_overrides": {"lightgbm": "passthrough", "xgboost": "passthrough"}
  },
  "max_train_outcome_missing_ratio": 0.5,
  "max_test_outcome_missing_ratio": 0.5,
  "continuous_priors": null
}
```

对几个真正有实际后果的字段的说明:

**`predictor_columns` 与 `predictor_prefix`** —— 二者恰好设置其一,不能并存。
前缀规则很方便,但会悄悄吸纳任何匹配该前缀的新列,所以如果你的列集合是稳定
的,更推荐用显式列表。两者都不能包含 outcome 或 ID 列;引擎会拒绝这种重叠,
而不是让 target 泄漏进特征集。

**`feature_universe.mode`** —— `fixed_a_priori` 表示特征集是在不看 outcome
的情况下决定的;`train_pool_screened` 表示它是仅从官方训练池推导出的。内部划
分的 schema *必须* 是 `fixed_a_priori`,因为在随机划分下,不存在一个可以安全
地称为"仅训练"的数据池。

**`imputation.model_overrides`** —— 只有 `lightgbm` 和 `xgboost` 可以设为
`passthrough`,因为只有它们原生支持 `NaN`。

**`task: classification`** 只接受二元 `{0,1}` 的 outcome。

schema 会被哈希进 experiment identity。改变语义(插值策略、predictor 集合)
会改变 identity 并开始一份新的 checkpoint;只改路径则不会。

## 步骤 5 —— 写 provenance

在 ARD 旁边,记录数据来自哪里、是哪份 schema 产出的:

```json
{
  "adapter": "smr",
  "dataset": "asample2_withlag",
  "source_sha256": "…",
  "schema_sha256": "…",
  "feature_universe_sha256": "…",
  "ard_sha256": "…"
}
```

引擎会拿 `schema_sha256` 去和它实际加载的 schema 交叉核对,所以一份过期的
ARD 配上一份被编辑过的 schema 会被抓出来。provenance 被刻意排除在
experiment identity 之外——重新生成它不会使一个正在运行的实验失效。不要把
原始 ID 或绝对路径放进 provenance。

## 步骤 6 —— 声明 panels

`panels.yaml` 只放运行旋钮。任何 schema 拥有的字段出现在这里都会报错,这正是
让语义只存在于一处的保证机制。

```yaml
model_params: model_params.yaml
preset: dev

panels:
  - name: my_outcome_run
    schema: schema/my_dataset.json
    outcome: y1
    models: [ols, ridge, random_forest, xgboost]
    out: outputs/nk_grid_my_outcome.csv
```

Panel 可以设置 `preset`、`seed`、`n_seeds`、`n_draws`、`n_sizes_n`、
`n_sizes_k`、`min_n`、`max_n`、`max_k`、`batch_size`、`n_jobs`、`test_size`
(仅内部划分生效)、failure 阈值,以及 `allow_large_run` / `dry_run`。preset
有 `dev`、`medium`、`pilot_full`、`production`。

## 步骤 7 —— 冒烟测试

```bash
aleatoric-nk-grid-panels --manifest YourArticle/panels.yaml --dry-run
```

这会解析每个 panel 并打印 cell 数量估计,不会触碰数据。然后真正跑一次 `dev`
preset,检查输出 CSV 及其 `.manifest.json` 附属文件。确认 `K_unobserved` 看起
来合理,并且 failed 行数为零——一整片带着同一条错误信息的 `failed` 行,通常
意味着 adapter 本该防住的一个契约问题。

## 当 validation 拒绝你的输入时

以下这些都会在任何模型运行之前触发,这是有意为之的设计——一张静默产出的
failed 行表,比直接崩溃要糟糕得多。

| 报错信息 | 原因 |
|---|---|
| `schema contains unknown fields` | 拼写错误,或者把一个 panel 拥有的字段放进了 schema |
| `must define exactly one of predictor_columns or predictor_prefix` | 两者都设置了,或者都没设置 |
| `Resolved predictors overlap protected outcome/ID columns` | 前缀规则误捕获了你的 outcome 或 ID |
| `predictor … must be finite numeric` | 有 object/string 列混进了 ARD |
| `predictor … contains ±inf` | 除法或 log 运算作用在了哨兵值上 |
| `predictor … is entirely missing` | 某列全部是 `NaN` |
| `regression outcome contains non-finite values` | target 里有 `inf` |
| `classification outcome must contain only binary {0,1}` | target 是多分类的 |
| `outcome missing ratio … exceeds …` | 超过一半的行没有 outcome;要么刻意调高阈值,要么修上游数据 |
| `Resolved feature universe does not match the canonical definition` | manifest 或 predictor 列表变了,但没有重新生成 universe 文件 |
| `definition_sha256 does not match definition_file` | universe 文件被编辑了,但 schema 里的哈希没更新 |
| `provenance schema_sha256 does not match` | 过期的 ARD 配上了被编辑过的 schema |
| `Kept manifest rows must exactly cover resolved predictors` | manifest 和数据已经彼此漂移 |
| `ordinal_levels … is not canonical JSON` | JSON 数组里有多余空格 |
| `contains an invalid one-hot state` | 某一行有两个 1,或者 dummy 部分缺失 |
| `External test source … contains category states absent from train` | 测试集里有一个模型从未见过的类别 |
| `id_column … contains duplicate IDs` | 标识符不唯一 |
| `Usable training rows … below required minimum` | 删除 outcome 缺失行后,剩余行数不够所选模型的 CV 要求 |

## 清单

- [ ] 已确定抽样单位;若有 source 展开成多列,已写好 manifest
- [ ] predictor 均为数值型,缺失留 `NaN`,未做任何插值
- [ ] predictor 规则不会命中任何 outcome 或 ID 列
- [ ] feature universe 用引擎自带函数生成,并已哈希进 schema
- [ ] schema 完整,predictor 规则恰好一条,`feature_universe.mode` 正确
- [ ] 外部模式:`test.csv` 结构与 train 一致,已设置 `id_column`,ID 无重叠
- [ ] 已写 provenance,内容是哈希而非原始标识符
- [ ] `panels.yaml` 只包含运行旋钮
- [ ] `--dry-run` 通过,随后 `dev` preset 产出干净结果、无 failed 行

## 接下来去哪看

- [`upstream-adapter-spec.md`](upstream-adapter-spec.md) —— 规范性契约,
  包含完整的插值状态机与 `K_unobserved` 语义。
- [`../NK_Grid/README.md`](../NK_Grid/README.md) —— 运行时、checkpoint 与
  Slurm 行为。
- `NK_Grid/tests/conftest.py` —— `write_schema_bundle()` 几行代码就能搭出一份
  完整合法的 bundle,是查看最小可运行示例最快的方式。
