## 背景

当前 FFC 数据路径由两部分组成：`src/prepare_ffc_analysis.py` 把 `background.dta` 转换为 `X_`、`C_`、`M_` 数值特征和 manifest，`src/prepare_ffc_nk_inputs.py` 再按六个 outcome 生成官方 train/test 文件。NK Grid 通过 `_predictor_columns()` 按前缀选择 predictor，通过 `_feature_groups()` 和 `--feature-manifest` 把展开列还原到 `source_column` 后抽取 K。

现有模型对缺失值的处理并不完全一致：线性、logistic 和 sklearn Random Forest pipeline 使用 `_median_imputer()`，在当前 `X_sub` 上拟合；XGBoost 和 LightGBM 的回归与分类路径都绕过该 imputer，直接使用各自的原生数值 `NaN` 处理。设计必须保留这种现状，不能通过离线填补让完整训练池的信息进入小 N 单元。

本 change 同时跨越 `data_processor/` 编码层和 `src/nk_grid.py` 输入层，但不得扩展为 NK Grid core 重构。当前工作树中的既有修改属于用户，实施时必须逐处保留。

## 目标 / 非目标

**目标：**

- 实现 `median_mode`、`median_missing_indicator`、`tree_ordinal` 三套确定性编码策略。
- 所有输出保留模型运行时所需的 `NaN`，不在离线阶段学习或应用填补统计量。
- 三套策略共享同一 source universe、规范化 `source_column` 和顺序，使 K 在策略间可比。
- 为六个 outcome 生成可直接交给 NK Grid 的 train/test 文件、manifest、schema 和 QA 产物。
- 让 NK Grid 以最小改动读取 CSV 与 Parquet。
- 明确主跨策略比较只使用 XGBoost 和 LightGBM。

**非目标：**

- 不实现 KNN、MICE、IterativeImputer、离线 scaling 或 outcome-supervised feature screening。
- 不提供 pandas 原生 categorical dtype，也不为 XGBoost 启用 `enable_categorical`。
- 不修改 `src/model_registry.py`、模型超参数、N/K 抽样、指标计算或 checkpoint 语义。
- 不修改 `src/run_panels.py`；现有 panel 解析已经可以传递不同后缀的输入路径。
- 不在本 change 中新增跨策略结果聚合器；公平包络线由结果分析按策略和公共模型集过滤。
- 不提交 FFC 私有数据或生成的中间文件。

## 决策

### 1. 分离共享 schema、策略编码与 outcome 物化

采用单向数据流：

```text
background.dta + Stata labels + official train IDs
                    │
                    ▼
        shared source schema / source manifest
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
   median_mode  median_missing  tree_ordinal
                    │
                    ▼
       strategy feature manifest + encoded features
                    │
                    ▼
        join official train/test outcomes (6 outcomes)
                    │
                    ▼
          strategy-specific NK input files
```

`data_processor/src/data_processor/common/` 负责 I/O、类型判定、缺失码解析、命名、manifest 和验证；三个 strategy 模块只实现从共享 schema 到策略表示的转换。`scripts/` 提供薄 CLI，可运行单一策略或全部策略。这样避免复制读取、ID 校验和 outcome join 逻辑，同时保留三个可独立测试的策略入口。

备选方案是复制现有两个 prepare 脚本并各改一份。该方案会产生三套逐渐漂移的类型推断、命名和 QA 逻辑，无法可靠保证 source-level K，因此不采用。

### 2. 共享 source schema 是固定实验定义，不是填补器

source schema 只使用官方 train ID 对应的 predictor 值和 Stata metadata，不读取 outcome 值，也不读取官方 test predictor 分布来决定类型或类别词表。它一次性确定：

- 原始列名、规范化安全列名和稳定顺序；
- source 类型：continuous、categorical 或 dropped；
- dropped 原因和现有阈值参数；
- categorical 的训练期 observed level 词表与 Stata label；
- 三套策略共同使用的 eligible source 集合。

所有策略必须为同一个 eligible source 生成至少一个 predictor；不得按策略单独删除 source。展开特征仍可记录 `keep` 和过滤原因，但若过滤会导致某个 source 在任一策略中没有 predictor，则该 source 必须在共享 schema 层对所有策略统一排除。

该固定 schema 会使用完整官方 train pool 的无监督 predictor 信息，因此它不等同于每个小 N 单元重新发现 feature universe。这样做是为了保持 K 轴和 train/test 列集合稳定，也与当前 baseline 的固定 manifest 设计一致。禁止项仍然是从完整 train pool 学习填补值、scale 参数或 outcome 相关筛选。

### 3. 统一缺失值识别

共享层把空字符串、pandas/Stata 空缺以及数值码 `-1` 至 `-9` 识别为 FFC missing。缺失原因在 source manifest 中保留。其他负数不自动视为缺失，避免误删潜在合法值。

各策略只能决定这些缺失在表示层是 `NaN` 还是显式 indicator/category；不得用观察值替换它们。

### 4. `median_mode` 表示

- continuous source 输出一个 `X_<safe_source>` float 列；缺失保持 `NaN`。
- categorical source 对训练 schema 中的 observed level 生成 `C_<safe_source>__<level>` one-hot 列。
- categorical 缺失或 test-only 未知 level 对应的整个 one-hot group 保持 `NaN`，不离线选择众数类别。
- 不生成 `M_` 列。
- 输出使用 CSV；所有 predictor 最终都是数值列。

在线性、logistic 和 Random Forest pipeline 中，现有 `_median_imputer()` 在当前 `X_sub` 内对每个 dummy 列拟合。对二元 dummy，median 等于该 dummy 的 mode；多分类 source 可能被填成全零 group，而不是严格恢复唯一的 modal category。全零 group 被定义为 reference/unknown 表示，这是不修改 `model_registry.py` 的明确折中。

对 XGBoost 和 LightGBM，`median_mode` 不会触发 median/mode 填补；两者直接使用这些 `NaN`。因此策略名描述的是面向现有非 boosting pipeline 的表示约定，而不是保证所有模型族采用相同填补算法。

备选方案是在模型 pipeline 中加入 group-aware categorical mode transformer。它可以恢复严格 modal category，但需要修改 `model_registry.py`，违反白名单，因此不采用。

### 5. `median_missing_indicator` 表示

- continuous source 输出 `X_<safe_source>`，缺失保持 `NaN`。
- 对 continuous source 中出现的 `-1` 至 `-9` 分别生成 `M_<safe_source>__neg_<n>`；空缺单独生成 `M_<safe_source>__blank`。
- categorical observed level 按 `median_mode` 方式 one-hot。
- categorical 的 `-1` 至 `-9` 分别生成独立 `C_...__neg_<n>` 水平；空缺单独生成 `C_...__missing`，不把九种缺失原因合并。
- indicator 和显式 missing-category 列为 0/1，不含 `NaN`；需要填补的 continuous 值仍为 `NaN`。
- 输出使用 CSV，并保留与当前 feature manifest 兼容的 `source_column`、`feature_name`、`kind`、`keep` 和 `reason` 字段。

该策略复用当前 baseline 的核心缺失表示，但把共享 schema、策略编码和 outcome join 拆开，确保它能与另外两套策略使用同一 source universe。

### 6. `tree_ordinal` 表示

- continuous source 输出 `X_<safe_source>` float64 列，缺失保持 `NaN`。
- categorical source 输出一个 `C_<safe_source>` float64 列。
- observed level 依据共享 schema 中的稳定顺序映射为 `0, 1, ..., L-1`；码值是整数语义，但使用 float64 承载 `NaN`，避免 pandas nullable integer 与 `xgb.DMatrix` 的兼容性差异。
- missing 和 test-only 未知 level 均映射为 `NaN`，并在 QA 中分别计数。
- ordinal mapping 写入独立 JSON，并在 feature manifest 中记录 mapping ID/hash；不使用 pandas categorical dtype。
- 输出使用 Parquet，由 `pyarrow` engine 保存。

XGBoost 回归直接构造 `xgb.DMatrix`，LightGBM 回归直接构造 `lgb.Dataset`；分类路径分别直接返回 `XGBClassifier` 和 `LGBMClassifier`。这四条路径都能消费普通数值列中的 `NaN`，且都不经过 `_median_imputer()`。因此无需修改 `model_registry.py`。

备选方案是原生 categorical dtype + `enable_categorical=True`。它需要修改 XGBoost fit/predict 两处 DMatrix 调用并扩大白名单，已被 proposal 排除。

### 7. Manifest 与 source-level K 契约

共享 `source_manifest.csv` 是三套策略的 canonical source inventory。每套策略各自输出 `feature_manifest.csv`，每个实际 predictor 恰好映射到一个 canonical `source_column`。必要字段为：

| 字段 | 含义 |
|---|---|
| `source_column` | 原始 FFC 变量名；跨策略稳定 |
| `feature_name` | 当前策略的实际 predictor 名 |
| `kind` | `X`、`C` 或 `M` |
| `strategy` | 三个策略 ID 之一 |
| `keep` | 是否进入 NK input |
| `reason` | 保留或排除原因 |
| `source_order` | canonical source 顺序 |

tree ordinal 的映射细节放在 JSON，不塞入 CSV 单元格。三个 manifest 中 `keep=True` 行所覆盖的 `source_column` 集合及顺序必须完全一致。

NK Grid 继续使用 `_feature_groups()`：K 统计 source 数，`K_expanded` 统计该 source 在当前策略展开后的列数。跨策略比较以 K 为横轴，同时保留 `K_expanded` 作为表示复杂度诊断。只有测试证明现有 `_feature_groups()` 无法消费上述 manifest 时，才允许对白名单内函数做最小修复。

### 8. 输出目录和命名

生成文件位于 gitignored 目录：

```text
data/intermediate_files/preprocessing/
├── source_manifest.csv
├── schema.json
├── median_mode/
│   ├── features.csv
│   ├── feature_manifest.csv
│   ├── qa_summary.json
│   └── nk_inputs/ffc_{train|test}_<outcome>.csv
├── median_missing_indicator/
│   ├── features.csv
│   ├── feature_manifest.csv
│   ├── qa_summary.json
│   └── nk_inputs/ffc_{train|test}_<outcome>.csv
└── tree_ordinal/
    ├── features.parquet
    ├── feature_manifest.csv
    ├── ordinal_mappings.json
    ├── qa_summary.json
    └── nk_inputs/ffc_{train|test}_<outcome>.parquet
```

所有 JSON 使用排序键和稳定缩进；所有表按官方输入 ID 顺序写出。metadata 记录原始文件 hash、配置 hash、schema hash、strategy ID、软件版本、行列数和生成时间。生成时间不参与内容 identity hash。

### 9. NK Grid 的最小消费改动

在 `src/nk_grid.py` 新增：

```python
def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(...)
```

`run_nk_grid()` 只在训练文件和外部测试文件两个读取点改用该 helper。`_predictor_columns()` 不改；三套策略通过现有 `X_`/`C_`/`M_` 前缀被识别。feature manifest 继续使用 CSV，因此 `_feature_groups()` 的 manifest 读取不需要 Parquet 分派。

`requirements.txt` 增加 `pyarrow`；当前 `.venv` 中尚未安装该依赖。`run_panels.py` 仅传递 Path，不读取数据，因此不修改。

### 10. Panel 与公平比较

`panels.yaml` 使用显式 panel，命名为 `ffc_<strategy>_<outcome>`：

- `median_mode` 和 `median_missing_indicator` 可运行现有支持的模型集合；
- `tree_ordinal` 只声明 XGBoost 和 LightGBM；
- 每个 panel 指向对应 strategy 的 input 和 feature manifest；
- `dataset` 包含 strategy ID，使 experiment metadata 和输出文件可区分。

主跨策略包络线只比较三套策略中 XGBoost/LightGBM 的同一 `(outcome, seed, draw, N, K, model)` 支持域。median 策略的其他模型不进入该包络线，仅作补充结果。由于 `run_panels.py` 不允许未知配置字段，不新增 `comparison_tier` 等 panel key。

### 11. 验证策略

测试分三层：

1. 纯单元测试：缺失码识别、安全命名、共享 schema、one-hot group NaN、九种缺失码分列、ordinal 映射和未知 level。
2. 跨策略契约测试：canonical source 集合与顺序一致，每个 predictor 恰好映射到一个 source，输出仍含预期 `NaN`，没有离线填补产物。
3. NK Grid 集成测试：CSV/Parquet 后缀分派、未知后缀报错、train/test schema 一致、`_predictor_columns()` 复用、`_feature_groups()` source-level K，以及 XGBoost/LightGBM 对小型 `tree_ordinal` Parquet 的 smoke run。

真实 FFC 数据只用于本地、可选的端到端 smoke test；自动测试使用合成数据，不复制任何私有值。

## 风险 / 权衡

- **[小 N 单元共享完整 train 的固定 schema]** → 这是保持 source-level K 和 train/test 列集合稳定的有意选择；schema 不使用 outcome，也不学习填补或 scale 参数，并通过 hash 完整记录。
- **[`median_mode` 的 dummy-wise median 不等于严格 modal category]** → 将全零 group 定义为 reference/unknown，测试并记录该行为；若未来要求严格 group mode，必须新开 change 并修改模型 pipeline。
- **[XGBoost/LightGBM 在 median 策略中不会执行 median 填补]** → 在结果解释和 metadata 中明确：公共树模型包络线比较的是编码/缺失表示，树模型继续使用原生 NaN routing。
- **[ordinal 码引入人为顺序]** → 接受该近似以保持 `model_registry.py` 零改动；映射固定、可审计，且只用于树模型。
- **[test-only category 变为 NaN]** → 不扩展训练期词表；QA 分 source 报告 unknown count，超过配置阈值时失败。
- **[Parquet 增加依赖和格式差异]** → 固定 `pyarrow` 依赖，使用 round-trip 测试验证 dtype、NaN、列顺序和 hash 输入。
- **[18 个显式 panel 增加配置体积]** → 不修改 `run_panels.py` 做矩阵展开；通过一致命名和 dry-run 测试降低维护风险。
- **[现有工作树有用户改动]** → 实施前后对每个白名单文件做 scoped diff，只添加必要行，不重写或格式化整文件。

## 迁移计划

1. 在 `data_processor/` 建立 package、配置和合成测试，不触碰 NK Grid core。
2. 实现共享 schema/source manifest，并验证三套策略的 canonical source 集合。
3. 依次实现三个 encoder、outcome 物化和 QA；输出到新的 strategy 目录，不覆盖现有 `data/intermediate_files/nk_inputs/` baseline。
4. 增加 `pyarrow` 和 `read_table()`，完成 CSV/Parquet 聚焦测试。
5. 在 `panels.yaml` 增加带 strategy ID 的显式 panel，并先运行 `--dry-run`。
6. 使用合成数据和受限 `--max-jobs` 完成 XGBoost/LightGBM smoke test；随后才在本地私有 FFC 数据上生成完整输入。
7. 对三套 manifest、六个 outcome 的行数、ID、列集合、NaN、hash 和公共 K 域执行最终 QA。

回滚不需要迁移原数据：删除新生成的 gitignored strategy 目录，并撤销白名单内新增的 loader、依赖和 panel 条目即可。现有 baseline 文件与运行路径始终保留。

## 开放问题

1. `median_mode` 是否接受上述精确定义：非 boosting pipeline 使用 dummy-wise median，而 XGBoost/LightGBM 使用原生 NaN routing？若要求所有模型都执行严格 categorical mode，需要扩大范围并修改 `model_registry.py`。（已确认：接受。）
2. 是否接受 source schema 在完整官方 train pool 上固定、但不使用 outcome/test、不做填补或 scaling？若要求每个 N 单元重新发现 schema，则 source-level K 将不再天然可比，并需要把 schema 构建移入 NK Grid。（已确认：接受。）
3. test-only category 的 QA 失败阈值采用零容忍还是仅报告？默认建议仅报告并映射为 `NaN`，因为小样本训练中出现未知 level 是预期情况。（已确认：默认仅报告，另设高位硬失败天花板。）
