# 方案:为 nk_grid 增补连续型结果指标(第二批)

## 目标

Charlie 要求"多加评估指标,反正算这些相比模型训练几乎零成本"。现有 `nk_grid.py` 已算 17 个连续型指标;
本轮在**同一次运行**里再补 **13 个连续型指标**,全部在模型预测向量上计算(不重训、不改抽样/网格/循环)。

## 范围

### In-scope
- 只改 `nlsy_replication/src/nk_grid.py` 里**集中式**的 `compute_regression_metrics` 与 `METRIC_COLUMNS`,
  并在 `tests/test_nlsy_replication.py` 补测试。
- 新增 13 列(名称与定义**严格如下**,`y` = 测试集真实值,`p` = 测试集预测值,`ytr` = 训练子样本真实值):

  | 列名 | 定义 |
  |---|---|
  | `pinball_q05` | `mean_pinball_loss(y, p, alpha=0.05)` |
  | `pinball_q25` | `alpha=0.25` |
  | `pinball_q50` | `alpha=0.50` |
  | `pinball_q75` | `alpha=0.75` |
  | `pinball_q95` | `alpha=0.95` |
  | `ks_statistic` | `scipy.stats.ks_2samp(y, p).statistic`(真实分布 vs 预测分布) |
  | `wasserstein_distance` | `scipy.stats.wasserstein_distance(y, p)` |
  | `top_decile_hit_rate` | 令 `t = y >= quantile(y,0.90)`、`q = p >= quantile(p,0.90)`;返回 `|t & q| / |t|`(top-10% 的召回率) |
  | `bottom_decile_hit_rate` | 令 `t = y <= quantile(y,0.10)`、`q = p <= quantile(p,0.10)`;返回 `|t & q| / |t|` |
  | `rsr` | `rmse / np.std(y)`(ddof=0);`std==0 → nan` |
  | `cv_rmse` | `rmse / np.mean(y)`;`mean==0 → nan` |
  | `mase` | `mae / mean(abs(y - mean(ytr)))`(朴素基线 = 训练均值预测,与 r2_test 的 null 一致);分母 `0 → nan` |
  | `pearson_r2` | `pearson_r ** 2`(pearson_r 为 nan 时 → nan) |

- 13 个名称按上表顺序**追加**到 `METRIC_COLUMNS`(这样 `_empty_metrics()` 会自动为失败行填 NaN)。
- 所有指标沿用现有健壮性约定:退化输入(常数 `y`、样本 <2、分母为 0)返回 `np.nan`,**不得让单次运行崩溃**。

### Out-of-scope(不要做)
- **不加** MAPE / sMAPE / WAPE(结果已是 log 值,百分比误差失真)。
- **不加**任何分类 / ROC-AUC 指标(那是另一个待确认项,不在本轮)。
- 不改抽样、网格、seed/draw 循环、CSV 之外的 schema、checkpoint 逻辑。
- 不动其它脚本(`sample_size.py`、`feature_sets.py`、`overall_prediction.py`、SHAP 等)。

## 验收标准
1. `compute_regression_metrics(y, p, ytr)` 返回键从 17 增至 **30**,含上表全部 13 个新键,定义与上表一致。
2. `METRIC_COLUMNS` 追加了 13 个新名(顺序如上),CSV 相应多出 13 列。
3. 失败行(`status=failed`)这 13 列均为 NaN(经由 `_empty_metrics`)。
4. 退化输入不崩:常数 `y`、`n<2`、`std/mean/分母=0` 时对应列为 `nan`,其余列照常。
5. 现有 17 列的名称、定义、数值不变;除 `nk_grid.py` 与测试文件外无改动。
6. `pytest -q` 全绿。

## 测试要求
- 扩展已有的 `test_nk_regression_metrics_known_values`,对**至少 5 个**新指标用可手算的小样例断言精确值,建议:
  - `pinball_q50`(= 0.5·MAE)、`pearson_r2`(= pearson_r²)、`rsr`(= rmse/std)、
    `top_decile_hit_rate`、`wasserstein_distance`。
- 新增退化用例:常数 `y`(令相关/rsr/cv_rmse 走 nan 分支)断言返回 `nan` 且不抛异常。
- 端到端测试断言新 CSV 含全部 30 个指标列。
