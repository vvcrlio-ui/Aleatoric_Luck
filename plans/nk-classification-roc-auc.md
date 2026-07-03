# 方案:nk_grid 增加二分类(就业)ROC-AUC 分支(v2.0)

## 目标

白板第二列 `Pseudo R²(ROC-AUC)` 对应论文的附加分析:预测**"40–50 岁是否就业"(二分类)**。
现有 `nk_grid.py` 只做连续型结果。本轮加**一条分类分支**,复用同一套 N×K 对数网格 +
seed(分层划分)/draw 循环 + checkpoint,输出一个分类版长格式 CSV,含 ROC-AUC 等分类指标。

团队决定(v2.0):连续型 30 指标全保留(已在 main),本轮**新增**分类支持。

## 范围

### In-scope
1. `nk_grid.py` 增加 `--task {regression,classification}`(默认 `regression`,**回归行为完全不变**)。
2. classification 路径:
   - 结果变量 = 二分类就业列(参数 `--outcome`,默认列名见待确认 #1)。
   - **划分:分层** `train_test_split(..., stratify=y)`,仍是每个 seed 一套划分(seed 控划分不变)。
   - 子抽样:复用现有 `draw_orders`(排列取前缀)机制。
   - 模型:分类器,通过 `predict_proba` 取正类概率。
3. `model_registry.py` 增加分类器版本(本轮起 in-scope)。按名称映射:
   `ols→LogisticRegression(无罚)`、`ridge→LogisticRegression(L2)`、`lasso→(L1)`、
   `elastic_net→(elasticnet)`、`xgboost→XGBClassifier`、`lightgbm→LGBMClassifier`、
   `random_forest→RandomForestClassifier`、`bart→分类 BART(bartpy 支持则用,否则该模型返回 nan)`。
   回归器保持不变。
4. 分类指标集中在新函数 `compute_classification_metrics`,列(8 个):
   `roc_auc, pr_auc, brier, log_loss, balanced_accuracy, f1, accuracy, mcfadden_pseudo_r2`
   - `pr_auc` = average precision;概率型指标用正类概率;`f1/accuracy/balanced_accuracy` 用 0.5 阈值。
   - `mcfadden_pseudo_r2` = `1 − loglik_model / loglik_null`(null = 常数模型 = 训练正类基率)。
     这就是白板 `Pseudo R²(ROC-AUC)` 那一列。
5. CSV:新增 `task` 列;分类运行写**分类指标列**;默认输出到独立文件(如 `nk_grid_clf.csv`)。
   一个 task 一个 CSV,契合"每个 panel 一个 CSV"。
6. slurm:加一个分类版 sbatch(或给现有加 `--task` 参数示例)。

### Out-of-scope(不要做)
- 不改回归路径的现有 30 指标、列名、数值、行为。
- 不改网格 / seed / draw / checkpoint 骨架(复用)。
- 不动 `overall_prediction.py` / `sample_size.py` / `feature_sets.py` / SHAP。
- 只做二分类,不做多分类。

### 待确认(实现前若未回,按默认并在 PR 显式标注)
1. **就业二分类结果列的确切列名**(asample 数据里)。Codex 应在实际数据/代码里核对真实列名;
   拿不准就在 PR 提问,**不要硬编码一个猜测列名**当成确定。
2. **支持哪些模型作分类器**:默认全部 8 个按上表映射;不可行者优雅降级为 nan。
3. **分层划分**:默认 `stratify=y`(推荐);若 Charlie 另有意见以他为准。

## 健壮性
- 子样本或测试集**只有单一类别** → `roc_auc / pr_auc / log_loss` 无定义 → 返回 `nan`,**不得崩**
  (套用现有 try/except + `_empty_classification_metrics()` 填 NaN)。
- 模型无 `predict_proba` → 概率型指标 `nan`。

## 验收标准
1. `--task classification` 跑通,生成分类 CSV,含 `task` 列 + 上述 8 个分类指标列。
2. `--task regression`(默认)行为、30 个连续指标、列**完全不变**(回归回归测试全绿)。
3. 8 个分类指标定义正确;单一类别输入 → 概率型指标 `nan` 且不抛异常。
4. 划分为**分层**;不同 seed → 不同划分(可由 `split_random_state` 或划分索引验证)。
5. checkpoint key 不变(`model, seed, draw, N, K`),分类续跑不重复。
6. `model_registry` 新增分类器且回归器不受影响;不支持的模型/任务组合优雅降级。
7. 除 `nk_grid.py`、`model_registry.py`、测试(及可选 sbatch)外无改动;`pytest -q` 全绿。

## 测试要求
- `compute_classification_metrics` 已知值:小样例手算断言 `roc_auc`、`brier`、`accuracy`、
  `mcfadden_pseudo_r2`。
- 单一类别输入:`roc_auc/pr_auc/log_loss` 返回 `nan`,不抛异常。
- 端到端:小网格分类跑通,CSV 含 `task` 列 + 全部 8 个分类指标列,
  行数 = `n_seeds × n_draws × |N| × |K| × |models|`。
- 回归回归测试:默认 task 下现有全部测试仍绿、30 列不变。
- 分层验证:划分后训练/测试的正类比例接近总体比例。
