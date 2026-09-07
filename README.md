# NKGRID

重复抽样的N×K预测实验，包含SMR与FFCWS应用。

## 单命令启动

Linux本地（默认FFC GPA、dev规模）：

```bash
bash run.sh local
```

BMRC集群（默认dev，自动准备环境、提交计划生成作业，完成后提交计算与汇总链）：

```bash
# 将 YOUR_PROJECT_ACCOUNT 替换为你获准使用的项目账户（无默认值）
bash run.sh slurm --profile bmrc --account YOUR_PROJECT_ACCOUNT
```

先查看启动配置，不安装、不读取实验数据、不提交作业：

```bash
bash run.sh slurm --profile bmrc --account YOUR_PROJECT_ACCOUNT --preset production --dry-run
```

首次使用需要可读的私有数据和Adapter生成的schema/ARD。已有其他位置的输入用 `--schema` 指定，不要求复制或覆盖原文件。详见 [启动与维护说明](launch/README.md)，其中包含production、恢复、Windows/WSL、环境和验收边界。

研究设计：[NK_Grid](NK_Grid/README.md) · [SMR](SMR/README.md) · [FFCWS](FFCWS/README.md)。

运行命令可显式追加 `--checkpoints keep` 或 `--checkpoints delete`，选择成功并校验完成后
保留全部 checkpoint 或清理；最终逐单元 CSV 均保留，失败和未完成时均保留恢复数据。
它只控制文件保留，不改变模型拟合与评分。恢复、Slurm 归档及 bootstrap 所需数据见
[checkpoint 保留说明](launch/README.md#成功结束后的-checkpoint-保留)。

## 回归 MLP 的 batch 选择

NK_Grid、FFCWS、SMR 的新实验参数文件使用 `nk-grid-v7-mlp-batch-cv-1`。
在 `regression.shallow_neural_network` 和 `regression.super_learner` 下显式设置：

```yaml
mlp_batch_size: cv
mlp_batch_candidates: [32, 64, 128, 256]
```

独立 MLP 使用原有 `max_cv_folds: 3`，联合选择 alpha × batch；Super Learner
使用 `mlp_batch_cv_folds: 3`，每个 OOF 训练折及最终拟合分别选择 batch，
保留固定 alpha 和 `cv: 5` stacking。所有内折完整拟合预处理，以原始目标尺度的
折均验证 MSE 选择；平分取声明顺序第一个，不合并有效 batch 相同的候选。
候选列表必须非空、无重复且仅含正整数；折数必须是至少 2 的整数，布尔值无效。
若内折训练行不足 2，确定性使用首个 alpha/batch 并记录原因。
数值拟合失败或非有限预测使该候选失效并记录错误，全部失效则报错；最终拟合失败直接报错。

每个单元的 `mlp_diagnostics_json` 记录选择、有效 batch、alpha、实际迭代数、候选分数，
并区分 Super Learner 的 OOF 与全样本拟合。CV 诊断来自真实拟合，不重放搜索。
默认正常单元分别需要 61 和 78 次 MLP 拟合；候选/内折串行执行，2000 仍只是迭代上限。
搜索规则标识与参数进入现有运行身份；新算法不能恢复旧版本结果。

兼容固定策略可显式设 `mlp_batch_size: auto`、`full` 或正整数（例如 `32`）。
省略策略的自定义旧参数仍解析为 `auto`；随仓库提供的新参数文件显式采用 `cv`。
固定策略保持原数值路径，候选配置不参与搜索。分类行为不变。
实验写盘的 `batch_size` 是另一参数，与 MLP batch 无关。
