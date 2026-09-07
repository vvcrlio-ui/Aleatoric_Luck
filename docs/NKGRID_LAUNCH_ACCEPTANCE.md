# MLP迭代预算与单命令入口：实现及验收记录

日期：2026-09-07。起始提交：`a3d6945`。本次改动在工作区，未提交、推送、合并或提交真实Slurm作业。

## 显式账户输入修订

按用户要求移除 BMRC 的 `mills.prj` 默认账户，同时移除 `NKGRID_ACCOUNT` 的自动读取。所有 Slurm 新运行、dry-run、恢复均须显式传入 `--account YOUR_PROJECT_ACCOUNT`；恢复账户必须与原计划一致。本地运行不需要账户。

本次启动器回归 22 项通过，新增检查覆盖环境变量不能代填账户、显式值传递到 sbatch、缺省账户在安装/提交之前拒绝，以及空白账户拒绝。实际 Bash dry-run 传入 `--account test.prj` 后请求正确；省略账户则明确失败。以下旧记录中的无账户 dry-run 命令属于修订前证据，当前示例见 launch/README.md。本次未执行真实 Slurm 提交，未 commit 或 push。

## 已实现

- 三份回归模型参数中，Super Learner内MLP的epoch上限500→2000；独立回归MLP仍为2000，分类预算不变。FFC/SMR模型版本更新为 `nkgrid-models-v4-sl-mlp2000`。
- 现有选择性诊断同时输出实际迭代数和配置上限。仍保留auto批量及原收敛条件；尚未证明真实GPA欠拟合消失。
- `run.sh`编排可选fast-forward更新、模块环境、venv验证/首次安装，以及本地或Slurm启动。`run.ps1`通过WSL转发，不模拟POSIX锁。
- `launch/`集中新入口实现、BMRC站点配置、计划提交作业和维护文档；原 `NK_Grid/slurm/`保留，以免破坏已有协议和调用。
- Slurm登录节点只提交准备作业；准备节点调用原引擎生成计划并自动调用原提交器。恢复复用已有plan；不重写WAL、租约或发布检查。

## 已执行验证

| 验证 | 结果 | 边界 |
|---|---|---|
| `python -m unittest discover -s launch/tests -v` | 18通过 | 标准库编排测试；操作系统/安装/sbatch调用使用记录器，非集群端到端测试 |
| `.venv-remediation/Scripts/python.exe NK_Grid/tests/run_portable_remediation.py -q NK_Grid/tests/test_super_learner_iteration_budget.py NK_Grid/tests/test_remediation_batch.py NK_Grid/tests/test_model_param_contract.py` | 49通过，79个预期的小预算收敛警告 | Windows真实数值及参数传播；portable包入口不用于验收POSIX行为 |
| Bash语法检查、`run.sh slurm --profile bmrc --preset production --dry-run` | 通过 | Windows上MSYS Bash仅用于语法/只读参数编排；未加载集群module |
| PowerShell AST解析 | 无语法错误 | 未验收WSL转发实际运行 |
| `git diff --check` | 无空白错误 | Git提示部分文本将随autocrlf转换；新增gitattributes固定sh/sbatch为LF |

初次数值回归中，旧测试把“500出现三次、2000出现一次”作为预算契约，出现3个失败。依用户要求更新为按任务/模型身份验证：两个回归MLP均2000、两个分类MLP均500；不再靠相同文本行计数，防止预算互换仍通过。实际折内与最终fit传播另有测试：观察到5次12行fit和1次15行fit均收到2000。该用例在记录配置后将实际训练缩到1epoch，仅证明参数传播，不是2000epoch性能对照。

启动器攻击覆盖：只读预览产生副作用、生产规模缺少明确授权、输出冲突、非法panel路径、Slurm参数shell插值/换行、恢复时改模型/资源/账号、排队期间改变manifest或代码、登录节点执行全量plan，以及已有错误环境被静默覆盖。合法环境复用、新请求提交收据、原plan恢复也有正向路径，避免仅测试“全部拒绝”。

## 尚未完成

- 真实Linux本地实验、首次venv安装/竞争锁、实际module/CPU兼容、Slurm准备作业与完整依赖链、恢复及完成性核验。
- 原GPA现象在2000预算下的成对对照；本次未重新启动或改写旧500预算诊断。
- 这台Windows尚未检测到可用WSL发行版；是否使用WSL待用户选择。
- 目录范围已确认：对应GitHub的开发仓库独立放在 `D:\Aleatoric_power_law\aleatoric_luck`。已保留Git历史、远程地址及未提交开发修改；223个复制文件逐一校验哈希。旧GPA进程仍使用原路径，因此上层旧checkout、私有数据、虚拟环境及活动日志保留为运行现场，未移动或删除。后续开发使用新子目录，旧实验结束后的新证据需另行核对迁入。

没有将上述未验收项折算为通过。使用方法与现场前提见 [launch/README.md](../launch/README.md)。
