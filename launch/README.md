# 单命令实验入口

新入口只编排既有共享引擎和Slurm协议，不改变抽样、评分、任务表或WAL恢复方式。入口代码与站点配置集中于 `launch/`；底层集群脚本保留在 `NK_Grid/slurm/`，避免破坏既有调用和历史运行。

## 命令

从仓库根目录运行。所有 Slurm 命令中的 `YOUR_PROJECT_ACCOUNT` 必须替换为你获准使用的项目账户。账户无默认值，不从 `NKGRID_ACCOUNT` 或 `SBATCH_ACCOUNT` 环境变量自动选取；新运行、dry-run 和恢复均须显式传入 `--account`，本地运行不需要。命令中的新输出目录由程序创建；不用预先mkdir、export、生成request或单独提交plan。

```bash
# Linux/WSL本地：首次自动创建.venv-linux并安装固定依赖，此后验证并复用
bash run.sh local

# 使用已有的隔离FFC输入，只计算少量单元
bash run.sh local --schema FFCWS/data/remediation_20260907/schema/ffc_median_mode_gpa.json --models ridge --max-jobs 5

# 站点dev实验：32 workers、2 rounds、short、1小时；不是production
bash run.sh slurm --profile bmrc --account YOUR_PROJECT_ACCOUNT

# 用户给定的production站点设置：600 workers、4 rounds、long、10天
bash run.sh slurm --profile bmrc --account YOUR_PROJECT_ACCOUNT --preset production --allow-large-run --output aleatoric-production/ffc_median_mode_gpa-new

# 参数仍可覆盖，例如较小的并发与显式内存
bash run.sh slurm --profile bmrc --account YOUR_PROJECT_ACCOUNT --workers 8 --rounds 2 --memory 16G

# 仅打印启动请求：无git更新、module切换、环境安装、数据读入或Slurm提交
bash run.sh slurm --profile bmrc --account YOUR_PROJECT_ACCOUNT --preset production --dry-run

# SMR也使用同一个入口
bash run.sh local --manifest SMR/panels.yaml --panel smr_hourlywage

# 可选的同步：当前必须是干净的SMR&FFC，只允许fast-forward；不会自动切换分支
bash run.sh slurm --profile bmrc --account YOUR_PROJECT_ACCOUNT --update

# 恢复同一Slurm计划，不重复生成tasks.parquet
bash run.sh slurm --profile bmrc --account YOUR_PROJECT_ACCOUNT --resume runs/EXISTING_RUN/plan.json
```

`--dry-run`是启动层的只读预览，不声称输入已经通过数值验证。实际N/K和总单元数在执行节点通过引擎统一解析并打印。production与超过引擎规模阈值的设计都要求显式 `--allow-large-run`。`--max-jobs`仅用于本地；Slurm不可用它伪装成有界实验。

交接时应写明使用本页的 `run.sh` 入口。它从 panel 的通用预设解析网格；直接调用
`chunk_planning` 的同名 `pilot` 使用另一组显式 N/K，二者不是同一设计。
以执行节点打印的实际 N/K 和冻结的 snapshot 为准，不要仅凭预设名称认定实验相同。

## 成功结束后的 checkpoint 保留

在原运行命令末尾显式加入 `--checkpoints keep` 或 `--checkpoints delete`：

```bash
bash run.sh local --checkpoints keep
bash run.sh local --checkpoints delete
bash run.sh slurm --profile bmrc --account YOUR_PROJECT_ACCOUNT --checkpoints delete
```

PowerShell/WSL 入口同样透传该选项。也可在 panel 条目中设置
`checkpoint_retention: keep` 或 `delete`，命令行显式值优先。
省略时，新本地运行沿用成功后清理分片的历史行为，新动态 Slurm 运行沿用保留 WAL
的历史行为；本地恢复会继承已有 manifest 的显式策略，Slurm 恢复复用 snapshot 冻结策略。
Slurm `--resume` 不能改变已经冻结的策略。

`keep` 保留最终完整 CSV 和每次写出的 checkpoint。本地会关闭删除源分片的自动压缩，
所以分片文件数可能显著增加；它不是保存每个 epoch 的模型权重。
`delete` 在计算期间照常写 checkpoint，只有完整成功、最终表和完成记录通过校验后才清理。
失败、中断、未完成或最终表校验失败均保留恢复数据。最终 CSV、模型选择诊断、
运行身份、计划、配置和日志不会因该选项被删除。

动态 Slurm 删除只针对封存清单中的精确 WAL 文件，并写入
`out/checkpoint-archive.json`，保留封存/验证记录及最终 CSV 的 SHA-256。
这标志着该运行已终态归档，不能再通过 prep/run/close/verify 或 launcher resume 继续训练；
重复原精确 finalize 命令可验证最终文件或继续中断的清理。
清理意图落盘后如删除中断，剩余 WAL 继续保留，不能当作未归档运行重新提交。
应从一开始使用 `keep` 才能保留所有原始分片；后来切换策略无法恢复此前已经压缩或删除的分片。

最终 CSV 是逐 `model/seed/draw/N/K` 单元的完整长表，不是只剩一组平均数。
有足够重复时，可用它分析抽样/随机化的重复实验不确定性，不需要中间 checkpoint。
同一 seed 共享划分，同一 seed/draw 的 N/K 单元使用嵌套样本与特征前缀，
因此不能把整张表的行当作独立观测随机打散；应按设计保留分组和模型配对关系。
配对重采样的索引原则参见 [SciPy bootstrap 文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)。

如果 bootstrap 的对象是测试集里的每个个体、并需重算 MSE/R²，则必须保留逐行
`row_id/y_true/y_pred`。现有 `prediction_export_cells` 可为选定本地单元导出最终
`.predictions.parquet`；普通 checkpoint 也不含这些行级预测，动态 Slurm 暂不支持该导出。
如果重采样训练集并重新调参训练，需要原始数据、配置和重新拟合；保留指标 checkpoint
不能替代训练。当前仓库尚无上述统计 bootstrap 分析程序。

本功能的 Windows 测试覆盖参数传递、本地策略判断和临时文件上的归档协议；
完整 POSIX 引擎、文件锁竞争及 BMRC/Slurm 依赖链仍需在目标环境验收。

## 环境与输入

`launch/profiles/bmrc.sh` 保存用户提供的 `Python/3.11.3-GCCcore-12.3.0` 和 `skl-compat`，不设置项目账户。不包含用户名和绝对仓库路径。模块加载完成后，默认venv为当前仓库的 `venv/aleatoric-${MODULE_CPU_TYPE}`，与模块的CPU类型对应；可通过 `--venv` 或 `VENV` 指向已经准备的环境。

首次创建环境时安装 `NK_Grid/requirements.txt` 和editable引擎。已有环境需核对实际安装版本、源码路径、module/CPU身份和依赖一致性；环境发生变化时拒绝静默重装。选择一个新的 `--venv` 最稳妥；确认没有运行中的任务使用旧环境后，也可显式 `--refresh-env`。不需要每次升级pip。环境安装使用文件锁，防止两个启动命令同时修改环境。

私有数据是前置输入，不从GitHub下载，不伪造。该入口接受现有Adapter结果；不自动重建或覆盖schema与ARD。若仓库的默认schema指向不存在的ARD，会提示使用 `--schema`。同一已准备数据只需准备一次，此后的实验均为单命令。更换表示或结果变量须选对应panel和schema。

新输出目录必须不存在，防止覆盖。Slurm在仓库内的输出必须位于Git忽略目录（如 `runs/`、`aleatoric-production/`、`aleatoric-pilot/`）或放到仓库外；否则创建输出会破坏干净工作区契约。

## Slurm执行链

登录节点：解析轻量启动参数 → 核对/准备venv → 记录启动请求 → 提交一个计划准备作业，立即返回job ID。

计划节点：核对排队期间的代码/配置 → 校验私有输入 → 解析网格 → 调用原 `build_dynamic_plan()` → 写 `plan.json` → 调用原 `submit_flat_task_table.sh --submit`，提交prep、worker、close、verify与finalize依赖链。

计划作业默认16G，production为8小时，其他规模为1小时；`--plan-memory` / `--plan-time`可配置，默认使用选定的计算分区和constraint。站点限制以实际Slurm配置为准。计划阶段也不会占用登录节点长时间生成大表。

运行目录包含：`launch.json`、`submission.json`、`logs/plan-JOBID.out`、`logs/plan-JOBID.err`，以及计划生成后的 `plan.json`、`snapshot.json`、`tasks.parquet`、`out/` 和最终CSV。后续作业日志由原脚本写入同一运行目录的 `logs/`。既有提交收据与恢复拒绝行为继续有效。

本实现要求计划计算节点可以执行 `sbatch` 提交后续链。若站点禁止计算节点提交，需要基于现场策略调整提交位置；不能在没有站点测试时声称已完成BMRC验收。

提交器在每次 `sbatch` 前持久化待确认请求，接受后立即记录 job ID、依赖、generation 和完整命令。
收据的 `complete` 只表示整条链已被调度器接受，不表示计算结束。
中途失败、空或异常 job ID、接受后收据写入失败会保留 `partial` / `unknown` 或待确认意图，
并输出已接受的任务；同一 execution 的自动重提交会被拒绝，包括 launcher `--resume`。
必须人工对照收据和调度器核对这些任务，不能把非零返回理解成没有任务被接受。
若尚未写出任何提交意图就失败，记录为 `failed_before_submission`，可以修复输入后重试。
已有完整收据仍兼容原恢复流程，但恢复前须确认原链不再活跃且满足封存前驱条件；
当前实现不自动查询 `squeue` / `sacct`，也不能全面阻止人工重复提交仍活跃的完整链。

恢复时显式输入的 `--account` 必须与原计划冻结的账户相同，不匹配会在环境安装或提交之前拒绝。同一个活动输出不允许任意修改资源和设计后恢复。`--resume`复用冻结的plan；不同worker数、模型、schema、时间或内存需要合法的新执行计划或新实验。升级代码后，旧plan可能因契约不兼容被拒绝，不能手工编辑哈希绕过。

## Windows

现有完整引擎依赖Linux/POSIX的锁、进程和资源计量。PowerShell入口调用WSL中的相同Linux路径，不提供假的 `fcntl` / `resource` 替代品。

```powershell
.\run.ps1 local --schema FFCWS/data/remediation_20260907/schema/ffc_median_mode_gpa.json --models ridge --max-jobs 5
```

必须先有一个可用的默认WSL Linux发行版和Python 3.11–3.14。绝对Windows路径会通过 `wslpath` 转换；相对路径相对于仓库。Windows虚拟环境不能复用为WSL虚拟环境；新入口默认使用独立的 `.venv-linux`。

本机尚未发现可用WSL发行版，因此Windows完整实验、POSIX安装锁及实际Slurm提交尚未验收；不将标准库单元测试或Windows数值测试当作这些环境的替代验证。

## MLP迭代预算

当前新实验已采用 `nk-grid-v7-mlp-batch-cv-1`，FFC/SMR 的面板版本为
`nkgrid-models-v5-mlp-batch-cv-1`：三份回归配置显式使用 `mlp_batch_size: cv`。
候选、内折选择与兼容固定策略见[根目录说明](../README.md#回归-mlp-的-batch-选择)。
以下记录此前的迭代预算与固定 batch 兼容行为；其中缺省 auto 指省略策略的旧参数，
不代表当前随仓库提供的配置。启动、账户显式输入和 production 授权规则保持原约定。

三份model_params中，回归Super Learner的MLP `max_iter` 从500提高至2000，与独立回归MLP一致。该值实际传递到五折模型及最终重拟合；诊断增加明确的 `max_iter`。FFC/SMR面板的 `model_spec_version` 更新为 `nkgrid-models-v4-sl-mlp2000`。

回归独立 MLP 和 Super Learner 均支持在各自的模型参数段设置 `mlp_batch_size`（正整数、`auto` 或 `full`），例如在 `regression.shallow_neural_network` 下增加 `mlp_batch_size: 32`。该值传入每个内折和最终重拟合；两种模型分别配置，不会相互覆盖。缺省仍为 `auto`。这与实验写盘的 `batch_size` 无关；固定为 200 与 `auto` 等效。实际拟合行数不足时，正整数 batch 会截断至该次拟合的样本数。固定 batch 不保证预测曲面单调，应在训练池预先校准并冻结参数，不使用正式测试集选择。

2000是epoch上限，并非必须执行2000轮；原收敛条件可能提前结束。提高上限不能单独证明欠拟合消失。batch默认仍为auto，学习率、网络结构、alpha、折数及分类模型预算保持原定义。

之前的GPA诊断使用旧500轮预算，应按原参数解释，不重写其历史报告，不将旧结果当作新预算验证。在旧诊断进程结束前，不移动其源码、虚拟环境或数据目录。

## 对抗性验证

```bash
python -m unittest discover -s launch/tests -v
python -m pytest -q NK_Grid/tests/test_super_learner_iteration_budget.py NK_Grid/tests/test_remediation_batch.py NK_Grid/tests/test_model_param_contract.py
```

启动器测试攻击：production未授权、输出冲突、路径穿越、Slurm换行与shell注入、恢复时改设计/资源、排队期间代码/manifest改变、错误环境被静默重装、dry-run产生写入、登录节点误建大plan。MLP测试观察实际折内和最终fit收到的预算；为验证参数传递而缩短拟合的测试不作为2000轮预测性能证明。

目标Linux环境还必须验证：干净提交上的本地dev单元、首次venv安装和再次复用、真实module/CPU导入、Slurm计划和依赖链、同plan恢复及输出完整性。未完成这些验证前，本入口处于已实现、目标环境待验收状态。
