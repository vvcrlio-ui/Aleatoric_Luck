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

`timing_full` 沿用引擎定义：20×20 N/K 网格、1 seed、1 draw，模型取面板的完整模型列表；最终网格可能因去重而减少。Discoverer 对 timing_full/production 默认使用 `--workers 496 --rounds 2 --time 48:00:00`，最多 496 个单 CPU worker 并发，两轮依次执行。rounds 是续跑轮次，不改变 seed、draw 或模型数量。该预设覆盖完整 N/K 范围，不是几分钟必定完成的冒烟测试。

当前提交器一次提交全部轮次。按提交过程中作业均未结束计算，需要预留 `rounds × (workers + 2) + 3` 个作业名额（每轮 worker、prep、close，另加 verify、finalize、bootstrap）。496 × 2 合计 **999** 个；500 × 2 合计 1007 个，可能超过本项目当前每用户 1000 个提交作业上限。默认配置以没有其他排队或运行作业为前提；提交前用 `squeue -r -u "$USER"` 核对，账户限制有变化时重新调整。可显式传 `--workers`、`--rounds`、`--time` 覆盖默认值。

## 默认资源与环境

| 设置 | Discoverer 默认值 |
|---|---|
| Python module | `python/3/3.12/3.12.4` |
| partition | `cn` |
| account | 必须显式传 `--account`，不继承 BMRC 账户 |
| QoS | 默认等于 account；可显式 `--qos` |
| constraint | `none`，不向 Slurm 发送 constraint 参数 |
| worker | 1 CPU、16G 内存；timing_full/production 默认 496 workers，其他预设 16 workers；均为 2 rounds |
| worker wall time | timing_full/production 为 48h；其他预设为 1h |
| bootstrap | 1 CPU、48G、2h，可用 `--plan-memory` / `--plan-time` 调整 |
| venv | 默认 `<运行目录>/venv`，避免复用不完整的安装环境 |

每个作业显式携带 account、QoS、nodes、ntasks-per-node、ntasks-per-core 和内存。所有阶段的 QoS/资源写入冻结的执行计划，恢复时不会从当前 shell 猜测。48h 是本站账户当前已核实的上限；工作和 bootstrap 时限均在提交前校验。排队和并发仍受实际账户配额限制。

若管理员提供其他模块，可设置 `DISCOVERER_PYTHON_MODULE`；不要复用 BMRC 的 Python venv。安装临时文件位于 `<运行目录>/tmp/bootstrap-JOBID/`，pip 缓存位于 `<运行目录>/pip-cache/`，均使用项目存储，避免计算节点 `/tmp` 容量不足。依赖严格按仓库锁定版本安装，包不可下载或版本不可用会终止 bootstrap，不会自动放宽版本继续跑。

如有已经验证且没有作业正在修改的环境，可显式 `--venv /绝对路径` 复用。环境不匹配时会拒绝运行；`--refresh-env` 会重新安装，仅在确认没有其他作业使用该环境后使用。恢复模式不接受环境刷新。

## 数据准备与复用

`--prepare-ffc` 只准备选中 FFC 面板对应的表示和 outcome，并使用该面板声明的模型做输入验证。原始输入不移动、不覆盖；生成的 adapter 配置、ARD、schema 和验证报告位于 `<运行目录>/prepared/`。仓库中已跟踪的 schema 不改写。

如果已有准备好的数据，省略 `--prepare-ffc`，改传：

```bash
bash run.sh slurm --profile discoverer \
  --account ehpc-dev-2026d08-299 \
  --panel ffc_median_mode_gpa --preset timing_full \
  --schema runs/discoverer-gpa-timing-001/prepared/schema/ffc_median_mode_gpa.json \
  --output runs/discoverer-gpa-timing-002
```

SMR 使用 `--manifest SMR/panels.yaml --panel ... --schema ...` 指向已经准备好的输入；FFC 专用准备开关不用于 SMR。

## 日志、失败和恢复

- `launch.json`：冻结的启动配置、源码提交和输入身份。
- `launch.submission.json`：bootstrap 的 Slurm job ID。
- `logs/bootstrap-JOBID.out/.err`：环境安装、数据准备和计划生成日志。
- `prepared-launch.json`：生成 schema 后的执行请求。
- `plan.json`、`snapshot.json`、`tasks.parquet`：原动态队列计划。
- 其余作业收据、worker 日志和结果沿用原队列布局，位于运行目录及 `out/`。

```bash
squeue -u "$USER"
sacct -j BOOTSTRAP_JOB_ID --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

bootstrap 失败时不会进入后续计划/训练步骤。先检查日志；如果尚无 plan，使用新的输出目录重试，已安装完的环境可显式复用。取消安装留下的不完整 venv 不会被默认复用。不要在原目录手工改哈希或覆盖生成输入。

已经生成 plan 的恢复方式：

```bash
bash run.sh slurm --profile discoverer \
  --account ehpc-dev-2026d08-299 \
  --resume runs/discoverer-gpa-timing-001/plan.json
```

如果初次使用自定义 QoS，恢复时必须传同一个 `--qos`。恢复会提交计算节点 bootstrap 检查环境，再调用原提交器，不重新准备数据或生成任务表；原提交器负责并发、提交收据和 generation 检查。仍然活动的队列不要重复恢复。运行期间保持源码及已有输入不变。

生产规模仍要求显式 `--allow-large-run`。检查点沿用共享引擎的保留/恢复协议。可在首次启动时传 `--checkpoints delete`，只在验证成功后清理检查点并保留最终 CSV；`--checkpoints keep` 显式保留。恢复时不能更改已冻结的策略。

## 验证范围

本次变更只执行本地测试和只读预览，没有重新启动此前暂停的作业。真实计算节点上的安装、FFC 转换和完整动态队列仍需后续实际运行验收。

站点参考：[作业资源](https://docs.discoverer.bg/writing_slurm_batch.html)、[登录节点限制](https://docs.discoverer.bg/cpu-login-node-resource-limits.html)。
