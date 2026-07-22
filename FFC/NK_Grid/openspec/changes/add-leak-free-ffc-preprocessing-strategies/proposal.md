## 为什么

FFC NK-grid 实验需要把预处理策略明确纳入搜索空间，同时不能破坏现有学习曲线设计。不同编码策略应生成可复现、彼此独立的输入数据；所有需要从样本估计的填补统计量仍须在各模型的 `fit(X_sub)` 中拟合，确保小 N 单元不会借用完整训练池的信息。

## 变更内容

- 在 `data_processor/src/data_processor/strategies/` 下新增三套 FFC 编码策略：
  - `median_mode`：为现有子样本内拟合的中位数/众数式模型 pipeline 编码连续变量和分类变量，不生成缺失指标特征。
  - `median_missing_indicator`：保留现有 FFC 行为；数值变量的缺失生成指标，分类变量的 `-1` 至 `-9` 缺失码分别保留为独立编码水平，不合并为单一缺失类别。
  - `tree_ordinal`：把分类变量编码为确定性的整数 ordinal 码，以普通数值 dtype 保存；连续变量和分类变量的缺失值均保留为 `NaN`，并使用 Parquet 保存编码结果与数值类型。该策略不声称提供原生 categorical 支持。
- 所有策略输出均保持未填补状态。编码器可以统一缺失值表示并编码已观察类别，但不得拟合或应用中位数、众数、KNN、MICE、scaler 或其他从样本估计的填补与变换。现有模型 pipeline 继续只在当前 NK-grid `X_sub` 上拟合填补器。
- 为全部六个 FFC outcome 生成 train/test 输入，并输出确定性的 metadata、schema、QA 摘要和各策略 feature manifest。
- 所有策略继续使用稳定的 `X_` 与 `C_` predictor 前缀；`median_missing_indicator` 可以额外生成 `M_` 列。`tree_ordinal` 的 ordinal 映射记录在 manifest 中，不通过新前缀或 pandas categorical dtype 表达。
- 保证跨策略的 source-level K 可比。每个策略的 manifest 必须把所有输出 predictor 映射到同一套规范化 `source_column`；即使一个源变量展开为多列，NK Grid 仍通过现有 `_feature_groups()` 与 `--feature-manifest` 按源变量计数 K。
- 增加按文件后缀分派的表格读取，使 NK Grid 可以同时消费 CSV 和 Parquet，且不改变实验语义。
- 明确 `tree_ordinal` 的模型侧契约：XGBoost 和 LightGBM 的回归路径不经过 `_median_imputer()`，而是分别把普通数值列直接交给 `xgb.DMatrix` 和 `lgb.Dataset`；两者都能原生处理数值 `NaN`。因此该策略可以保持 NaN passthrough，且无需设置 `enable_categorical=True` 或修改 `src/model_registry.py`。
- 定义公平比较契约：
  - 跨策略主包络线只使用所有策略共同支持的模型交集：XGBoost 和 LightGBM；
  - median 策略运行其他模型所得结果仅作为策略内的补充结果；
  - `tree_ordinal` 不运行 sklearn Random Forest；该策略的公共比较范围有意限定为能直接消费 ordinal-coded + NaN passthrough 输入的 XGBoost 和 LightGBM。
- 本 change 不实现 KNN 或 iterative/MICE 策略。它们在高维场景下的计算性质，以及与子样本内拟合契约的不兼容，使其不属于当前范围。

## 范围约束

### Leak-free 预处理契约

- `data_processor` 生成的文件必须保留模型拟合时进行填补所需的 `NaN`。
- 在 NK Grid 选定 `(seed, draw, N, K)` 并构造 `X_sub` 之前，不得利用完整官方训练集学习任何填补统计量。
- 现有 `model_registry._median_imputer()` 必须继续位于模型 pipeline 内部，不得迁移到 `data_processor`，也不得在那里复制或预计算其结果。
- 本 change 不修改 `src/model_registry.py`。

### 上级文件白名单

把 OpenSpec root 放宽到 `NK_Grid`，并不意味着可以修改无关 core 代码。`data_processor/` 之外的实现改动严格限定在以下白名单：

- `src/nk_grid.py`
  - 新增一个小型 `read_table(path)` helper，按文件后缀选择读取方式；
  - 仅把 `run_nk_grid()` 中训练集和外部测试集的 `pd.read_csv(...)` 调用替换为 `read_table(...)`；
  - 复用现有 `_predictor_columns()`，不为 `tree_ordinal` 增加新前缀、dtype 分支或其他选择逻辑；该策略必须通过既有 `X_`/`C_` 前缀被原样识别；
  - 复用 `_feature_groups()` 和已有 `--feature-manifest` 接入；只有在存在可复现的兼容性缺陷、导致共享 source-level manifest 无法工作时，才允许修改该函数。
- `panels.yaml`
  - 仅添加或更新运行三套策略所必需的输入路径、策略标识、模型子集和 feature manifest 引用。
- `requirements.txt`
  - 仅添加所选 pandas I/O 实现需要的 Parquet engine 依赖。
- NK Grid tests
  - 仅添加或更新针对后缀分派、ordinal 映射确定性、数值 `NaN` passthrough、既有 predictor 选择、source-level K 分组和 leak-free 输入消费的聚焦测试。

未经先行修订本 proposal 并通过 review，不得修改 `src/nk_grid.py` 的其他函数、`src/model_registry.py` 的任何函数或其他无关 NK Grid 文件。实现过程中必须保留这些文件中已有的用户改动。

## 计划目录结构

```text
data_processor/
├── src/data_processor/
│   ├── common/
│   │   ├── io.py
│   │   ├── schema.py
│   │   ├── manifests.py
│   │   └── validation.py
│   └── strategies/
│       ├── median_mode.py
│       ├── median_missing_indicator.py
│       └── tree_ordinal.py
├── configs/
├── scripts/
└── tests/
```

生成的私有数据和中间数据继续存放在被忽略的 NK Grid data 目录中，绝不提交到版本库。

## 能力

### 新增能力

- `ffc-encoding-strategies`：为三套已批准的编码策略生成 leak-free、可复现的 FFC 输入，并保留缺失值供模型拟合时填补。
- `typed-nk-inputs`：读取 CSV 和 Parquet NK-grid 输入，保留普通数值 dtype 与 `NaN`，并按照稳定的 `X_`/`C_`/`M_` 命名契约选择 predictor。
- `comparable-preprocessing-search`：在不同展开表示之间保持 source-level K 一致，并执行跨预处理策略的共同模型比较包络线。

### 修改能力

无。当前 OpenSpec root 尚无现有主规格；本 change 将引入上述三项新能力。

## 影响

- 新实现和测试主要位于 `data_processor/`。
- NK Grid core 中仅允许触碰白名单列出的 `src/nk_grid.py` loader、predictor selection 和现有 feature manifest 接入点。
- 现有 `model_registry.py` pipeline 及其子样本内填补行为保持不变。
- `panels.yaml`、`requirements.txt` 和聚焦测试可以进行白名单中说明的最小集成改动。
- 策略输出会增加本地中间数据体积，尤其是 one-hot 编码版本；所有私有数据和生成产物仍保持 gitignored。
