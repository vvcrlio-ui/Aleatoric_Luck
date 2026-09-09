# Discoverer CPU 流水线

使用与 BMRC 相同的 `run.sh` 入口、任务表、动态 worker、WAL、close、verify 和 finalize 协议。新增 `--profile discoverer`，将安装依赖、可选 FFC 数据准备和计划生成全部放进 Slurm bootstrap 作业。登录节点只检查提交条件、写启动请求并提交一个 bootstrap 作业。

## 首次运行 FFC GPA timing_full

先进入集群上的干净、已提交的仓库。以下命令中的 `--ffc-data-dir` 接受包含 `background.dta`、`train.csv` 和 `test.csv` 的目录；这里对应当前上传位置。

```bash
cd /valhalla/projects/ehpc-dev-2026d08-299/aleatoric_luck
bash run.sh slurm --profile discoverer \
  --account ehpc-dev-2026d08-299 \
  --manifest FFCWS/panels.yaml \
  --panel ffc_median_mode_gpa \
  --preset timing_full \
  --checkpoints delete \
  --prepare-ffc --ffc-data-dir FFCWS/data \
  --output runs/discoverer-gpa-timing-001 \
  --dry-run
```

`--dry-run` 不安装、不读取实验数据、不生成目录、不提交作业。 如果登录节点默认 Python 过旧，预览会自动加载本站 Python module；本地已有 Python 3.11–3.14 时无需 module。确认预览后，去掉该参数才会实际提交。输出目录必须是新的；不要重复提交同一个实验来尝试恢复。

`timing_full` 沿用引擎定义：20×20 N/K 网格、1 seed、1 draw，模型取面板的完整模型列表；最终网格可能因去重而减少。Discoverer timing_full/production 默认自动解析 worker 数；`--rounds 2` 是最多两轮的保护上限，`--time 48:00:00` 是请求上限，仍受实时 MaxWall 限制。rounds 不改变 seed、draw 或模型数量，也不把未保存的单次训练变成连续 96 小时训练。

Discoverer 现在只提交当前轮的一个 worker array 及必要控制/恢复作业。控制作业封存并验证上一轮，完成则验证后发布；有剩余可执行任务才准备下一轮。每轮查询 QoS、用户和账户祖先关联、分区、Slurm 配置及已有作业，并在准备后提交 worker 前再次检查。worker 数取有效运行/提交数、CPU/内存/节点资源、剩余任务组的保守交集，计入实际辅助作业，不为未来轮次预留 array。700/1000 快照且没有其他限制或作业时通常约 698 workers；实际决策写入收据，不保证立即调度。显式 `--workers` 设置自动容量上限。

资源调整只改变 execution plan ID，保持 analysis ID、任务表、模型参数及设计不变。dry-run 将实时 worker、QoS/关联/分区和有效 wall time 显示为 unresolved；未传 `--workers` 的 timing_full/production 初始计划使用 `cluster.workers=1` 占位，实际 array 使用本轮冻结快照中的 worker 数。BMRC 保留旧提交器。

## 默认资源与环境

| 设置 | Discoverer 默认值 |
|---|---|
| Python module | `python/3/3.12/3.12.4` |
| partition | `cn` |
| account | 必须显式传 `--account`，不继承 BMRC 账户 |
| QoS | 默认等于 account；可显式 `--qos` |
| constraint | `none`，不向 Slurm 发送 constraint 参数 |
| worker | 单计算线程、16G；timing_full/production 自动容量，其他预设上限 16；默认最多 2 rounds |
| worker wall time | timing_full/production 为 48h；其他预设为 1h |
| bootstrap / controller | 1 CPU、48G、2h，可用 `--plan-memory` / `--plan-time` 调整 |
| venv | 默认 `<运行目录>/venv`，避免复用不完整的安装环境 |

每个作业显式携带 account、QoS、nodes、ntasks-per-node、ntasks-per-core 和内存。计算线程环境均为 1；按 CR_CORE 与硬件线程配置分配时可能计入两个逻辑 CPU，容量收据单独记录。每轮 worker 资源形成独立不可变合约。准备期间有效限制收紧时明确停止，不修改已经准备好的 assignment。当前 profile 接受的作业时长请求最多 48 小时，worker 还会按实时有效 MaxWall 缩短；700/1000 是已查快照，不是硬编码的 QoS 名额。

若管理员提供其他模块，可设置 `DISCOVERER_PYTHON_MODULE`；不要复用 BMRC 的 Python venv。安装临时文件位于 `<运行目录>/tmp/bootstrap-JOBID/`，pip 缓存位于 `<运行目录>/pip-cache/`，均使用项目存储，避免计算节点 `/tmp` 容量不足。依赖严格按仓库锁定版本安装，包不可下载或版本不可用会终止 bootstrap，不会自动放宽版本继续跑。

bootstrap 和 controller 在模块/Python 初始化前显式进入运行目录，以处理本站计算节点偶发的 Slurm 初始工作目录回退；worker 的 `--chdir`、Python、引擎和 snapshot 均绑定该次运行的绝对路径。

如有已经验证且没有作业正在修改的环境，可显式 `--venv /绝对路径` 复用。环境不匹配时会拒绝运行；`--refresh-env` 会重新安装，仅在确认没有其他作业使用该环境后使用。恢复模式不接受环境刷新。

## 数据准备与复用

`--prepare-ffc` 只准备选中 FFC 面板对应的表示和 outcome，并使用该面板声明的模型做输入验证。原始输入不移动、不覆盖；生成的 adapter 配置、ARD、schema 和验证报告位于 `<运行目录>/prepared/`。仓库中已跟踪的 schema 不改写。生成输入必须位于本次干净 checkout 内的 Git-ignored 运行目录，以满足不可变合约的仓库相对路径身份；原始输入目录可以在 checkout 外只读复用。

如果已有准备好的数据，省略 `--prepare-ffc`，改传：

```bash
bash run.sh slurm --profile discoverer \
  --account ehpc-dev-2026d08-299 \
  --panel ffc_median_mode_gpa --preset timing_full \
  --schema runs/discoverer-gpa-timing-001/prepared/schema/ffc_median_mode_gpa.json \
  --output runs/discoverer-gpa-timing-002
```

SMR 使用 `--manifest SMR/panels.yaml --panel ... --schema ...` 指向已经准备好的输入；FFC 专用准备开关不用于 SMR。

## CV 与动态均衡分批方法版本

`nk-grid-v8-ridge-5fold-1` 将生产路径的独立回归 Ridge 和 Super Learner
内部 Ridge 改为完整预处理流水线的 5 折交叉验证。每折只在训练行上拟合
填补和标准化，以一次 SVD 搜索原有 63 个 alpha；按各折 MSE 的等权均值
选择 alpha，再在全部训练行上重训。折不洗牌，少于 5 行时使用 N 折，
少于 2 行拒绝拟合。Super Learner 的外层 5 折及完整重训保持不变。

这是调参方法变更，不能与此前逐行留一验证的结果混为同一算法版本。
新实验使用新输出目录；正在运行的旧实验须保持其源码和参数不变。

当前 `nk-grid-v9-balanced-batch-1` 保留上述 Ridge 规则；panel 方法身份为
`nkgrid-models-v6-balanced-batch-1`。三份生产参数的 MLP/SL 均采用
`mlp_batch_size: balanced`：对每次实际训练的 m 行，分为 ceil(m/200) 批，
批大小最多相差 1，较大批在前，每轮恰好使用所有行一次。例如 201 行分为
101+100，401 行分为 134+134+133。200 是固定上限，不再交叉验证 batch。
这消除了 1 行尾批，但批数仍会在阈值处改变。

独立 MLP 保留三折搜索 5 个 alpha，随后完整重训（共 16 次 MLP fit）；
Super Learner 保留五折 OOF 和完整重训，MLP alpha 固定为 0.01（共 6 次）。
每次 fit 的特征和目标预处理均只在其训练行拟合。L2 保留原生按实际批大小
归一化；初始化、Adam、shuffle 和停止均保持原生：early_stopping=False、
tol=1e-4、n_iter_no_change=10，最大 2,000 轮。

实现复用锁定的 sklearn 1.8.0 训练循环，仅在独立函数命名空间替换分批迭代器，
不修改 sklearn 全局状态；其他版本明确报错，升级须重新验证数值一致性。
运行身份额外记录 mlp_batch_rule；auto、整数、full、cv 仍可显式用作旧对照。
fit_samples 和 effective_batch 仍仅为实验性 L2 对照，未采用为生产协议。

十 seed 配对确认中，K=500 的 200 附近总变差平均下降约 61.1%（9/10 seed），
400 附近平均下降约 11.3%，但仅 4/10 seed 改善。最大点 N=1165、K=3400 的
三折 CV MSE 平均由 0.387132 降至 0.386480（约 0.168%，5/10 seed 改善）。
该结果不保证全局平滑或全局最优，也不是独立外部测试准确率。
本地证据保存在 runs/batch-tenseeds-20260909；旧代码快照另存于
runs/balanced-adoption-20260909/baseline-758f0b4，旧结果保持原身份。
可选诊断保留实际 OOF、完整重训基础预测和元学习器系数，不额外重放训练。

## 日志、失败和恢复

- `launch.json`：冻结的启动配置、源码提交和输入身份。
- `launch.submission.json`：bootstrap 的 Slurm job ID。
- `bootstrap-journal.json`：bootstrap 原提交意图、唯一名称及接受/恢复状态。
- `logs/bootstrap-JOBID.out/.err`：环境安装、数据准备和计划生成日志。
- `prepared-launch.json`：生成 schema 后的执行请求。
- `plan.json`、`snapshot.json`、`tasks.parquet`：原动态队列计划。
- `continuation.json`：唯一运行 ID、预先冻结的预算、原 sbatch 意图、实时资源、每轮精确 generation、进度与失败。
- `rounds/N/snapshot.json`：本轮资源冻结快照；`logs/control-JOBID.*` 为自动控制日志。
- 其余作业收据、worker 日志和结果沿用原队列布局，位于运行目录及 `out/`。

```bash
squeue -u "$USER"
sacct -j BOOTSTRAP_JOB_ID --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

bootstrap 失败时不会进入后续计划/训练步骤。先核对日志、原提交收据和 Slurm 终态；只有确认原 bootstrap 已结束且没有后续控制链、尚无 plan 时，才在排除原因后使用新的输出目录重试。已安装完的环境可显式复用；取消安装留下的不完整 venv 不会被默认复用。响应是否被接受仍不确定时，先按下述原身份恢复流程核对，不能通过新目录重复发起实验。不要在原目录手工改哈希或覆盖生成输入。

已经生成 plan 的恢复方式：

```bash
bash run.sh slurm --profile discoverer \
  --account ehpc-dev-2026d08-299 \
  --resume runs/discoverer-gpa-timing-001/plan.json
```

如果初次使用自定义 QoS，恢复时必须传同一个 `--qos`。恢复重用冻结环境和 continuation 收据；已有控制/恢复作业时返回原 ID，不提交重复 bootstrap 或另一条链。不重新准备输入或任务表，旧版一次提交全部轮次的运行不能直接迁入新协议。terminal 状态不得盲目重试；先定位问题并核实作业身份。运行期间保持源码及输入不变。

每次 sbatch 前持久化原意图与唯一作业名；响应丢失时以 `squeue` 和 `sacct` 的名称/账户/用户/QoS 恢复 ID。array 使用 `JobID` 的根身份，不能用每元素不同的 `JobIDRaw`。未查到不代表未提交，绝不重发不确定意图。bootstrap 响应丢失时，在原干净 checkout 执行 `python launch/discoverer_continuation.py recover-bootstrap runs/运行目录/launch.json` 恢复原身份。

控制作业在准备/提交 worker 前先安排依赖自身 afterany 的延迟恢复 guard；正常 successor 依赖整个 worker array 的 afterany。正常时 guard 发现已建立后继即退出；部分提交失败时 guard 使用原身份补齐链。控制锁有界等待以处理同时触发；worker 还须通过原引擎精确 ready/generation 检查。续跑完全在集群端运行，不依赖本地电脑、SSH 或本 goal 在线。

默认最多两轮、连续两轮无进展停止、连续三次控制错误停止，最多 16 个控制/恢复作业提交意图（包含正常控制器及 guard，不包含 bootstrap 和 worker array）；控制推进次数也有同值上限。显式设置 `--rounds R` 时，控制上限为 `4R+8`。这些预算在启动前冻结。恢复提交序号单调递增，不随成功后清零的连续失败计数复用。OOM、BOOT_FAIL、FAILED、持久 TASK_ABORTED 或失败模型结果明确停止；TIMEOUT/NODE_FAIL 等仅在预算内重分配。未知接受状态做有界延迟查询，仍不确定则停止并保留意图供检查。

`continuation.json.status` 的含义：

| 状态 | 含义 |
|---|---|
| `ready` / `preparing` | 设计已冻结，正在安排或准备当前轮 |
| `workers_active` | 当前轮 worker 身份已记录，等待其终态及验证 |
| `recovery_required` | 已记录控制错误并安排有界恢复；不是完成 |
| `finalizing` | 全部结果已验证，正在发布最终输出 |
| `complete` | 验证、最终发布及所选检查点保留流程成功 |
| `round_budget_exhausted` / `no_progress` / `no_executable_tasks` | 按轮次/进度/可执行任务条件停止，实验可能未完整 |
| `control_budget_exhausted` / `quota_insufficient` / `unrecoverable` | 按控制预算、资源或失败证据停止，需要检查收据和日志 |

除 `complete` 外，停止状态都不能作为完整实验成功的凭据；恢复入口也不允许直接重启这些 terminal 状态。

生产规模仍要求显式 `--allow-large-run`。检查点沿用共享引擎的保留/恢复协议。可在首次启动时传 `--checkpoints delete`，只在验证成功后清理检查点并保留最终 CSV；`--checkpoints keep` 显式保留。恢复时不能更改已冻结的策略。

## 验证范围

本地 35 项调度测试覆盖模拟上限、剩余任务缩容、依赖、部分提交恢复、响应丢失、重复触发、guard 锁竞争、OOM 与无进展/预算停止。Linux 集成测试使用真实任务表、资源合约、OLS 训练、WAL、封存验证与发布，仅模拟 Slurm。2026-09-08 的有界计算节点验证通过 217 项模型/调度/引擎测试，随后当前源码快照通过另外 50 项测试，其中只读 WAL 观察测试使用真实结果增长、冻结身份校验，并确认不获取文件锁、不改写运行文件。第二份快照包含 controller 的目录重入修正；它不是对真实 Slurm 控制链的端到端验证。

这些结果没有验证一次真实 Slurm 自动多轮训练链；尚未提交新版正式 FFC GPA timing_full，也没有正式运行结果验收。批量/尾批科学决策已按用户要求暂停，本次整理只推进到 GitHub 推送前；不据此宣称已选择新的 MLP 协议或已完成正式实验。

站点参考：[作业资源](https://docs.discoverer.bg/writing_slurm_batch.html)、[登录节点限制](https://docs.discoverer.bg/cpu-login-node-resource-limits.html)。
Slurm 参考：[资源限制](https://slurm.schedmd.com/resource_limits.html)、[sacctmgr](https://slurm.schedmd.com/sacctmgr.html)、[array 身份](https://slurm.schedmd.com/job_array.html)。
