# 方案:N×K 联合扫描的统一结果表脚本(Figure 1 的数据源)

## 目标

为本论文(aleatoric luck / Zheng–Cheng 扩展)的 **Figure 1(以及后续同类图)** 产出数据源。
需要一个**统一的模板脚本**,在**同一套对数网格**上**同时**扫描:

- **N** = 训练样本量(sample size)
- **K** = 特征数(feature count)

每个 (N, K) 组合在多次随机中重复运行,输出一张**长格式 CSV**(每行一次运行),
列出一组预测质量指标。脚本作为模板复用,每个 figure/panel/dataset 产出一个独立 CSV。

这解决的问题:现在 N 和 K 是**分开两个脚本**跑的(`sample_size.py` 扫 N、
`feature_sets.py` 扫 K),且 N 用的是**线性**网格、seed 与 draw 合并成一层、只算回归 R²。
我们要把它们合并成一张能反映"可预测性如何随样本量与特征数共同变化"的表。

## 范围

### In-scope
1. 新增一个脚本(建议 `src/nk_grid.py`)+ 对应 `slurm/run_nk_grid.sbatch`,风格对齐现有脚本
   (argparse、`experiment.py` 的 checkpoint/metadata 辅助、joblib 并行)。
2. **N、K 共用同一套对数网格**:
   - K 复用 `feature_sets.py:82` 现成的 base-2 `np.logspace(0, log2(K_total), n_sizes, base=2)`。
   - N 从 `sample_size.py:208` 的 `np.linspace` **改为对数**,与 K 用同一刻度方案(同为 base-2 logspace,
     从最小点到 `len(X_train)`)。
   - 网格点数 `--n-sizes-n` / `--n-sizes-k` 作为参数(Charlie:算力允许下尽量多)。
3. **两层独立循环**(白板 SEED、DRAW# 两列):
   - `seed` 控**数据划分**:每个 seed 用 `random_state = seed` 重新做一次 70/30 划分。
   - `draw` 控**子抽样**:在该 seed 的训练集内,对每个 (N, K) 无放回抽样。
   - 即嵌套 `for seed → (重新划分) → for draw → for K → for N`。
4. **N 上限 = 完整训练集**(70% ≈ 5,200);K 上限 = 全部 `Aset*`/`Bset*` 特征。
5. **多指标**(连续型结果,Charlie:尽量多加,算这些成本可忽略)。默认列:
   `r2_test`(对训练均值 null,复用 `evaluation.py`)、`skill_score_pct`、`rmse`、`mae`、
   `medae`、`max_error`、`nrmse`、`spearman_rho`、`pearson_r`、`kendall_tau`、`ccc`、
   `explained_variance`、`mean_bias`、`median_bias`、`pinball_q10`、`pinball_q90`、
   `d2_absolute_error`。指标计算集中到一个函数/模块,便于扩展。
6. **长格式 CSV schema**(每行一次运行),至少含:
   `experiment_id, dataset, outcome, model, seed, draw, N, K, split_random_state,
   n_train_total, n_features_total, <各指标列>, status, error` + `build_experiment_metadata` 的元数据。
7. **dev↔production 一键切换**(顶端参数/CLI 默认):
   - dev 默认:`n_seeds=2, n_draws=2`、N/K 上限各 100、点数少、单模型(xgboost)。
   - production(BMRC,通过 CLI/sbatch 覆盖):`n_seeds≈100, n_draws≈50`、N/K 拉满、点数尽量多。
8. 沿用现有 checkpoint 机制,支持断点续跑(key 至少 `model, seed, draw, N, K`)。

### Out-of-scope(不要做)
- **不动** `overall_prediction.py`(论文主复现)、`sample_size.py` 的 power-law "误差下限 ε" 扩展、
  `domain_wise.py`、`SHAP_*` 等现有脚本。这是**新增**,不是重构删除旧脚本。
- 不改 `model_registry.py` 的模型定义/调参默认。
- 不引入新的数据清洗/特征工程;直接用现成的 `asample*.csv`。
- 分类(ROC-AUC)分支见"待确认",本轮**默认不实现**,除非待确认项拍定。

## 待确认(实现前若 Charlie 未回,按下述默认,并在 PR 里标注)
1. **seed→split 映射**:本方案按"每个 seed 重划一次 70/30"(Monte-Carlo 划分,给误差棒)。
   若 Charlie 另有指定(如固定划分、只控模型随机性),以他为准。
2. **二分类 ROC-AUC 列**:对应"40–50 岁是否就业"结果。本轮默认**只做连续型**;
   若确认要,则新增分类分支(分类器 + `roc_auc, pr_auc, brier, log_loss, balanced_accuracy, f1`),
   且该结果用**分层划分**(stratified)。
3. **同胞聚类**:NLSY79 含兄弟姐妹样本,随机划分可能有家庭层泄漏。默认用标准随机划分(对齐 Z&C);
   预留 `--group-split-col` 参数以便日后切到按家庭分组划分,但默认关闭。

## 验收标准(逐条可核查)
1. 存在 `src/nk_grid.py`,`python src/nk_grid.py --help` 正常显示上述参数。
2. 用小 dev 默认(`n_seeds=2 n_draws=2`,N/K 各 ≤100、点数少)在样例数据上跑通,
   生成一个非空 CSV,列包含 schema 中所有字段。
3. CSV 行数 = `n_seeds × n_draws × |N 网格| × |K 网格| × |models|`(去掉失败行前);
   每行的 `N`、`K` 取值落在对数网格上,`N ≤ len(X_train)`、`K ≤ 特征总数`。
4. N、K 两轴网格均为**对数刻度且同方案**;N 不再使用 `linspace`。
5. 不同 `seed` 对应**不同的 train/test 划分**(可通过 `split_random_state` 列或划分索引区分验证);
   同一 seed 下不同 `draw` 得到不同子样本。
6. 指标列齐全且数值合理(如 `r2_test ≤ 1`;`spearman_rho`、`pearson_r` ∈ [-1,1];
   `rmse ≥ 0`),null/失败情况写入 `status`/`error` 而不崩整个 run。
7. 中断后重跑能从 checkpoint 续上,不重复已完成的 (model, seed, draw, N, K)。
8. 现有脚本与测试不受影响(未改动)。

## 测试要求
- 单元测试:网格生成函数(N、K 均为对数、去重、上限裁剪、点数可配)。
- 单元测试:指标函数对已知输入返回预期值(至少 `r2_test`、`rmse`、`spearman_rho`、`pinball_q10`)。
- 单元测试:seed 改变 → 划分改变;draw 改变 → 子样本改变(可用小合成数据断言索引不同)。
- 集成测试:小网格 + 合成/样例数据端到端跑通,断言 CSV 行数与 schema。
- checkpoint 续跑测试:跑一半中断再跑,结果不重复且补全。
- 对齐仓库现有测试风格(`tests/`),`pytest` 全绿。
