# Aleatoric Luck

> 内部审核版。本文说明当前代码已经实现的实验设计、比较逻辑和解释边界；具体统计汇总与作图规则由后续分析方案规定。

## 1. 研究问题

本项目考察样本外预测表现如何随两类研究资源变化：

- 训练样本数 \(N\)：研究者观察了多少个训练对象；
- 可用预测变量数 \(K\)：研究者能使用多少项既有信息。

对每个数据集、结果变量、数据表示和模型，实验在每个 \((N,K)\) 位置生成多次重复观测：

\[
S_{s,d}(N,K)=
\text{held-out predictive performance},
\]

其中 \(s\) 表示 seed，\(d\) 表示 draw。所有重复观测共同形成 \(N\times K\) 预测表现曲面；均值、中位数、区间和平台判据由后续分析方案规定，不由实验生成程序预先决定。

这张曲面支持三类比较：

| 比较 | 保持不变 | 改变 | 回答的问题 |
|---|---|---|---|
| 样本收益 | 同一 seed/draw 下的 \(K\) 与模型 | \(N\) | 增加训练对象还能改善多少预测表现？ |
| 测量收益 | 同一 seed/draw 下的 \(N\) 与模型 | \(K\) | 增加既有信息还能改善多少预测表现？ |
| 模型收益 | 同一 seed/draw 下的 \(N\) 与 \(K\) | 模型 | 更换函数形式或学习算法还能改善多少预测表现？ |

如果预测表现仍随 \(N\) 增加，有限样本仍是误差来源；如果仍随 \(K\) 增加，当前信息集仍不充分；如果模型之间仍有明显差距，模型选择仍会影响结论。只有在较大 \(N\)、较大 \(K\) 和多类模型之间都出现稳定平台时，数据才支持“在当前实验条件下，继续增加常规研究资源的边际收益已经较小”这一判断。

## 2. 一个实验格点如何产生

每条主结果由以下身份唯一确定：

\[
(\text{dataset},\text{outcome},\text{representation},
\text{model},seed,draw,N,K).
\]

每个格点按同一顺序执行：

1. 从预先声明的训练池和合格变量库开始。
2. 根据 `seed` 和 `draw` 生成训练样本与变量来源的随机排列。
3. 取样本排列的前 \(N\) 个训练对象，以及变量排列的前 \(K\) 个变量来源。
4. 只用这 \(N\) 个训练对象估计缺失值填补等数据依赖预处理。
5. 在同一训练格点内完成模型拟合及所需的交叉验证或调参。
6. 在未参与拟合的测试集上预测并计算指标。

同一个 `seed/draw` 下，所有 \(N\) 和 \(K\) 共享同一组随机排列。因此，小 \(N\) 训练样本是大 \(N\) 训练样本的子集，小 \(K\) 变量集也是大 \(K\) 变量集的子集。相邻格点可以据此作配对比较，避免把样本或变量的完全更换误当成增加 \(N\) 或 \(K\) 的效果。

```mermaid
flowchart LR
    A["声明训练池、测试集和变量库"] --> B["seed：确定外层重复"]
    B --> C["draw：排列训练样本和变量来源"]
    C --> D["取前 N 个样本、前 K 个变量来源"]
    D --> E["在格点训练样本内估计预处理和模型"]
    E --> F["在该重复对应的测试集上评估"]
    F --> G["汇总为 N×K 预测表现曲面"]
```

## 3. \(K\) 的统计单位

\(K\) 统计变量来源，不等于模型矩阵的列数。

一个分类变量经过 one-hot 编码后可能生成多列；这些列作为一个整体进入或离开模型，只计作一个 \(K\)。由某个原始变量生成的缺失指示列也属于同一个抽样单元：原始变量被抽中时，其保留的缺失指示列随之进入模型；原始变量未被抽中时，这些列也不进入模型。缺失指示列不单独占用 \(K\)。

实验从合格变量库中随机抽取 \(K\) 个变量来源，所以它估计的是：

> 从当前变量库随机获得 \(K\) 项测量时的平均样本外预测表现。

它不估计“经过监督式特征选择后，最优的 \(K\) 个变量能达到什么表现”。这种随机抽样还要求 schema 明确声明变量来源可作为交换单位。该声明是当前实验对变量抽样的操作性假设，不表示不同内容领域的变量在实质上完全同质。

## 4. seed、draw 与重复实验

`seed` 和 `draw` 共同定义重复实验，但承担的作用不同：

- 在 SMR 中，`seed` 决定70%训练、30%测试的内部随机划分，并参与训练样本排列、变量排列和模型随机数的生成。
- FFC 使用 Challenge 官方划定的训练集和测试集，`seed` 不参与数据集划分；它只参与官方训练池内的样本排列、变量排列和模型随机数的生成。
- `draw` 在同一个 seed 内重新排列训练样本与变量来源，并参与模型随机数的生成。

所以，draw 之间的变化可能同时来自训练样本选择、变量组合、排列顺序、交叉验证折叠和随机模型拟合。draw 方差不能单独解释为变量抽样不确定性或算法不确定性。

当前设计在 \(N\) 和 \(K\) 都达到最大值时仍保留全部 draw。此时样本集合和变量集合可能已经相同，但排列、交叉验证和模型随机性仍会产生残余变化。保留这些 draw，可以让右上角与相邻格点使用相同的重复口径，并提供残余算法变异的对照。

## 5. 两个经验应用

### 5.1 SMR：中年社会经济结果

SMR 沿用 Zheng 与 Cheng 的 NLSY 预测框架，用早年经历和家庭背景预测两个连续结果：

- 对数小时工资 `Cm_lhourlywage`；
- 对数个人总收入 `Cm_ltotalincome`。

分析数据包含 4,252 个模型输入列，对应 497 个变量来源。每个 seed 都重新执行一次内部随机划分：70% 为训练池，30% 为测试集。当前分析描述预测的持续性和边界，不估计任何单个预测变量的因果效应。

详见 [SMR/README.md](SMR/README.md)。

### 5.2 FFC：15 岁生命结果

FFC 使用 Fragile Families Challenge 的出生至9岁背景资料，预测15岁时的六个结果：

| 结果 | 代码 | 任务 |
|---|---|---|
| 学业成绩 | `gpa` | 连续 |
| 毅力 | `grit` | 连续 |
| 家庭被驱逐 | `eviction` | 二元 |
| 家庭物质困难 | `materialHardship` | 连续 |
| 照护者下岗 | `layoff` | 二元 |
| 照护者参加职业培训 | `jobTraining` | 二元 |

FFCWS 严格遵从 Challenge 的评测协议：官方训练集和测试集保持固定。所有 \(N\)、\(K\)、模型、seed 和 draw 都在同一官方测试集上评估；重复实验只改变训练侧抽样、变量组合和模型随机性。固定测试集保证所有格点和模型可以直接配对比较。

各个 \(N\) 格点从已经定义好的训练池和变量空间中抽样。因此，小 \(N\) 格点表示“在训练池内预先定义变量空间后，只用 \(N\) 个家庭拟合模型”，不表示从变量定义到模型拟合的全部过程都只观察了 \(N\) 个家庭。

FFCWS 比较三种数据表示：

| 数据表示 | 配置 ID | \(K\) 的变量来源数 | 完整变量库展开后的模型列数 | 类别值与缺失信息的处理 |
|---|---|---:|---:|---|
| one-hot + 格点内填补 | `median_mode` | 3,400 | 11,432 | 分类变量展开为成组指示列；缺失值用格点训练样本估计的统计量填补 |
| one-hot + 缺失指示变量 | `median_missing_indicator` | 3,400 | 16,085 | 在上一表示基础上加入经过训练池筛选的缺失指示列；这些列跟随所属原始变量进入或离开模型 |
| 有序数值编码 | `tree_ordinal` | 3,400 | 3,400 | 分类值使用固定整数编码；缺失值和未见值保持缺失，交由格点预处理或支持缺失值的模型处理 |

三种表示共享相同且顺序一致的3,400个原始变量来源。同一 `seed/draw/N/K` 会选择相同的训练家庭和原始变量子集；对于 `median_missing_indicator`，被选中变量对应的缺失指示列自动随行，不另占 \(K\)。因此三种表示可以在每个格点作配对比较，格点间差异表示同一批原始信息经过不同编码和缺失处理后的整套预测管线差异。

`median_missing_indicator` 的 manifest 内部有8,053个预处理组：3,400个原始变量的值表示，加上4,653个缺失指示组。8,053用于区分不同列需要怎样预处理，不是 \(K\) 的变量库大小。模型实际接收的展开列数还会因 one-hot 编码而变化，所以结果同时记录 `K` 和 `K_expanded`：前者始终按原始变量来源计数，后者记录该格点进入模型矩阵的实际列数。

详见 [FFCWS/README.md](FFCWS/README.md)。

## 6. 模型、预处理与信息边界

每个应用使用同一组九类模型：

1. OLS；
2. Ridge；
3. Lasso；
4. Random Forest；
5. Extra Trees；
6. XGBoost；
7. LightGBM；
8. 浅层神经网络；
9. Super Learner。

同一格点中的不同模型使用相同的训练对象、变量来源和测试对象。模型超参数记录在各应用的 `model_params.yaml` 中；需要数据驱动选择的参数只在格点训练数据内通过交叉验证确定。

adapter 负责固定的行级转换、特殊缺失码转换、类别编码和变量库声明，但不提前填补缺失值，避免数据泄露。数据依赖填补在每个 \((seed,draw,N,K)\) 格点内重新估计。OLS、Ridge、Lasso、Random Forest、Extra Trees、浅层神经网络和 Super Learner 使用格点训练样本估计的填补值；XGBoost 与 LightGBM 直接接收保留为 `NaN` 的缺失值，并在树的分裂过程中使用各自的原生缺失值处理。因而跨模型比较的是“模型及其预先规定的预处理方式”组成的完整预测管线，不是把所有模型放在同一份填补后矩阵上的纯算法比较。


## 7. 仓库结构与可复现输入

```text
Aleatoric_Luck/
├── Adapter/      数据适配器的共同契约
├── FFCWS/        Fragile Families Challenge 应用
├── NK_Grid/      N×K 重复抽样、拟合、评估和集群执行引擎
└── SMR/          Social Rigidity 应用
```

私有源数据放在以下未跟踪路径：

```text
SMR/data/private/asample2_withlag.csv
FFCWS/data/private/background.dta
FFCWS/data/private/train.csv
FFCWS/data/private/test.csv
```

adapter 将私有源数据转换为分析就绪数据、schema、变量 manifest 和变量空间定义。源数据、生成的数据和模型输出均不进入 Git。

## 8. 本地检查与运行

从仓库根目录使用 Python 3.11 或更高版本：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r SMR/requirements.txt
python -m pip install -r FFCWS/requirements.txt
```

生成并验证分析数据：

```bash
.venv/bin/python SMR/adapter/adapter.py
.venv/bin/python FFCWS/adapter/adapter.py
```

在不读取数据、不拟合模型的情况下检查解析后的设计规模：

```bash
.venv/bin/aleatoric-nk-grid-panels \
  --manifest SMR/panels.yaml \
  --dry-run

.venv/bin/aleatoric-nk-grid-panels \
  --manifest FFCWS/panels.yaml \
  --dry-run
```

当前 `panels.yaml` 默认使用 `dev` 规模。运行单个 panel：

```bash
.venv/bin/aleatoric-nk-grid-panels \
  --manifest SMR/panels.yaml \
  --only smr_hourlywage
```

测试命令：

```bash
.venv/bin/python -m pytest -q
```

adapter 的完整输入契约见 [Adapter/ADAPTER.md](Adapter/ADAPTER.md)，引擎配置和恢复方式见 [NK_Grid/README.md](NK_Grid/README.md)。

## 9. 分析规模

常用 preset 如下；研究报告应记录解析后的实际 seed、draw、\(N\) 和 \(K\)，不能只报告 preset 名称。

| Preset | Seeds | 每个 seed 的 draws | \(N\) 点数 | \(K\) 点数 | 每个结果—模型组合的格点数 |
|---|---:|---:|---:|---:|---:|
| `dev` | 3 | 3 | 3 | 3 | 81 |
| `medium` | 8 | 8 | 10 | 10 | 6,400 |
| `timing_full` | 1 | 1 | 20 | 20 | 400 |
| `production` | 100 | 50 | 20 | 20 | 2,000,000 |

正式运行前应复制并审核 panel 文件，将顶层 `preset` 改为 `production`，再检查解析后的总格点数和输出路径。超过 250,000 个格点必须显式传入 `--allow-large-run`。

## 10. Slurm 提交、恢复与结果完整性

先预览任务映射：

```bash
bash NK_Grid/slurm/submit_nk_grid.sh \
  --manifest SMR/panels.production.yaml \
  --dry-run
```

审核后提交：

```bash
bash NK_Grid/slurm/submit_nk_grid.sh \
  --manifest SMR/panels.production.yaml \
  --allow-large-run
```

提交器为每个 `(panel, model, seed)` 建立一个 seed 任务，并按计算需求分为 `parallel`、`serial` 和 `super_learner` 三类数组。seed 任务先写各自分片；随后，每个 `(panel, model)` 的 finalizer 合并所有 seed；最后，publisher 合并该 panel 的全部模型结果。任何分片不完整时，finalizer 或 publisher 都会拒绝发布部分结果。

每个分片携带 `experiment_id`、`data_version`、`model_spec_version`、解析后的设计和模型语义。合并阶段检查这些身份与语义是否一致；同一路径的重复发布由文件系统 lease 拒绝；完整结果使用原子替换发布。

诊断和恢复命令见 [NK_Grid/README.md](NK_Grid/README.md#monitoring-and-recovery)。发生故障时，应保留并提交以下原始材料：

- 对应的 `NK_Grid/logs/*.out` 和 `NK_Grid/logs/*.err`；
- `NK_Grid/logs/slurm-specs/` 下匹配的任务快照和提交回执 JSON；
- `scontrol show config` 中的 `MaxArraySize`；
- `squeue -j <job-id>` 的原始输出及退出码；
- 输出目录所在共享文件系统的挂载参数。
