# 方案:整理为 nk_grid_pipeline/、BART 最小 N/K 门槛、多 panel 自动编排脚本

## 背景

用户已手动把 `nlsy_replication/` 整个复制成 `nk_grid_pipeline/`(逐文件内容相同),
意图让 `nk_grid_pipeline/` 成为今后唯一工作目录,`nlsy_replication/` 里重复的内容可删除。
用户本地 `nk_grid.py` 还有一处未提交的改动:新增 `log_progress()`,给长任务打带时间戳的进度日志,
需保留并提交进代码库。

另外,读取本地 `nk_grid_pipeline/outputs/nk_grid_all_models_medium.csv` 排查发现:
BART 在 64 行里失败 30 行,错误均为 `RecursionError: maximum recursion depth exceeded`,
根因是 `bartpy2` 的 `UniformMutationProposer.propose()`(`samplers/treemutation/uniform/proposer.py`)
在找不到任何可分裂/可剪枝的树时,会无限自我递归重试而不是终止。观测到的失败格恰好等价于
`N < 10 或 K < 2`(样本量 N 的对数网格最小点恒为 1,K=1 时早晚会耗尽唯一可分裂变量),
成功格全部满足 `N >= 10 且 K >= 2`。

本方案一次性处理三件相关的事(顺序即依赖关系:先定好目录,脚本才知道放哪)。

## 目标与范围

### A. 目录整理:`nlsy_replication/` → `nk_grid_pipeline/`
- 用 `git mv` 把 `nlsy_replication/{src,slurm,README.md,requirements*.txt,colab_run.ipynb,__init__.py,logs}`
  迁移到 `nk_grid_pipeline/`,保留 git 历史(而不是删掉再新建)。
- 合并用户本地未提交的 `log_progress()` 改动到迁移后的 `nk_grid.py`。
- 删除用户手工建的、未纳入 git 的旧 `nk_grid_pipeline/` 副本内容(改由本次正式迁移替代),
  避免出现两份并存。
- 更新所有引用旧路径的地方:
  - `tests/test_nlsy_replication.py` 里的 `from nlsy_replication.src...` 改成
    `from nk_grid_pipeline.src...`(测试文件本身建议同步改名为 `tests/test_nk_grid_pipeline.py`,
    保留全部既有测试用例不变,只改 import 路径与文件名)。
  - 根目录 `README.md` 里对 `nlsy_replication/` 的路径引用。
  - `.gitignore` 里的 `!nlsy_replication/logs/.gitkeep` 改成 `!nk_grid_pipeline/logs/.gitkeep`
    (对应 `.gitkeep` 文件也随目录迁移)。
  - `nk_grid_pipeline/README.md`(即原 `nlsy_replication/README.md`)里所有提及自身路径的地方
    (如 `cd nlsy_replication`、slurm 示例里的 `PROJECT_DIR=.../nlsy_replication`)同步改名。
  - 各 `slurm/*.sbatch` 脚本注释/默认路径里的 `nlsy_replication` 字样。
- `data/`、`outputs/` 目录内容按现有 `.gitignore` 规则不纳入版本控制,迁移时按需保留本地文件即可,
  不需要在 PR 里提交这些内容。
- **说明(如实告知,不必改动)**:`nk_grid_pipeline/` 迁移后仍会包含 `overall_prediction.py`、
  `sample_size.py`、`feature_sets.py`、`domain_wise.py`、`SHAP_*.py` 等与 nk_grid 无直接关系的
  论文复现/早期扩展脚本——这是用户明确要求的合并方式(把整个目录重命名),不在本方案里拆分,
  如后续需要按用途再拆分目录属于新的一轮方案。

### B. BART 最小 N/K 门槛(修复已确认的必然失败)
- 在 `nk_grid.py` 的网格执行逻辑里,对 `model == "bart"` 的任务格,若
  `n_samples < BART_MIN_N`(默认 10)或 `k_features < BART_MIN_K`(默认 2),
  **跳过实际训练**,直接写一行 `status="skipped"`、`error="below BART minimum N/K floor"`,
  且所有指标列为 `NaN`(不占用一次真实的模型训练/MCMC 时间)。
- 门槛通过 CLI 暴露:`--bart-min-n`(默认 10)、`--bart-min-k`(默认 2),便于以后按数据调整。
- 其它模型不受影响;仍保留原有 try/except 兜底(万一某个未被门槛覆盖的组合仍然
  抛出 `RecursionError` 或其它异常,照旧记 `status="failed"`,不得让整个 run 崩溃)。
- CSV 的 `status` 列因此可能出现三种取值:`ok`、`failed`、`skipped`——`skipped` 专指
  "按门槛主动跳过",`failed` 专指"尝试了但抛异常"。

### C. 多 panel 自动编排脚本("设定好理想参数,自动跑,每个 figure/panel 一个 CSV")
- 新增 `nk_grid_pipeline/src/run_panels.py`,读取一份声明式清单
  `nk_grid_pipeline/panels.yaml`(或 `.json`,二选一,Codex 定),每个条目描述一个 figure/panel:
  ```yaml
  - name: smr_income
    data: data/asample2_withlag.csv
    dataset: asample2_withlag
    outcome: Cm_lhourlywage
    task: regression
    models: [ols, ridge, lasso, elastic_net, random_forest, xgboost, lightgbm, bart]
    preset: production
    out: outputs/nk_grid_smr_income.csv
  - name: smr_employment
    data: data/asample2_withlag.csv
    dataset: asample2_withlag
    outcome: <TBD 就业列名>
    task: classification
    models: [ols, ridge, lasso, elastic_net, random_forest, xgboost, lightgbm]
    preset: production
    out: outputs/nk_grid_clf_smr_employment.csv
  ```
- `preset` 映射到一套具体参数(集中定义一次,不在每个 panel 里重复写):
  - `dev`: `n_seeds=2, n_draws=2, n_sizes_n=4, n_sizes_k=4, max_n=100, max_k=100`
  - `medium`: 介于 dev 与 production 之间,用于本地/单机验证整条流水线跑得通、跑得动
    (具体数值 Codex 可参照本地 `nk_grid_all_models_medium.csv` 已用过的规模定,保持向后可比)。
  - `production`: `n_seeds=100, n_draws=50, n_sizes_n=20, n_sizes_k=20, max_n=0, max_k=0`
    (`0` = 不封顶,用满全部训练集/全部特征,对齐现有 slurm 脚本默认值)。
  - 单个 panel 可覆盖 preset 里的任意字段(比如某个 panel 想用更小的网格),但默认走 preset。
- `run_panels.py` 直接以 Python 函数调用复用现有 `run_nk_grid(config)`(不是拼命令行字符串再
  subprocess 调用),每个 panel 一个独立输出文件、独立 checkpoint,panel 之间互不影响。
- CLI 能力:
  - `--manifest nk_grid_pipeline/panels.yaml`(默认路径)
  - `--only <panel_name> [<panel_name> ...]`:只跑指定 panel(便于调试单个 figure)
  - `--dry-run`:只打印每个 panel 解析后的完整参数,不实际执行(方便核对"理想参数"对不对)
  - 复用 A 部分保留下来的 `log_progress()`,在 panel 级别也打印"开始/结束/跳过"的时间戳日志。
- 断点续跑:每个 panel 各自的输出 CSV 用现有 `nk_grid.py` 的 checkpoint 机制,
  中断后重跑 `run_panels.py` 不会重复已完成的行,也不会跨 panel 互相干扰。

## Out-of-scope(不要做)
- 不新增数据集拉取/下载逻辑;`panels.yaml` 里的 `data` 路径假定文件已存在于本地。
- 不实现除 regression/classification 之外的第三种 task。
- 不处理跨仓库(FFCWS/HRS/BC58 等姊妹仓)的编排,只覆盖本仓库内的多个 panel。
- 不修改 `overall_prediction.py` / `sample_size.py` / `feature_sets.py` / `domain_wise.py` /
  `SHAP_*.py` 的内部逻辑(只是随目录一起改名迁移,内容不变)。
- 不改动已有的 8 个分类指标 / 30 个连续指标的定义。

## 验收标准
1. 仓库中不再存在 `nlsy_replication/` 目录;`nk_grid_pipeline/` 是唯一工作目录,
   `git log --follow` 能看到 `nk_grid.py` 等文件的完整历史(证明是 `git mv` 而非删除重建)。
2. 全仓库不再有对 `nlsy_replication` 字符串的引用(README、测试、`.gitignore`、slurm 脚本),
   `grep -r nlsy_replication .`(排除 `.git/`)无结果。
3. `nk_grid_pipeline/src/nk_grid.py` 含 `log_progress()` 且在数据加载、建网格、分批次时均有调用
   (与用户本地版本行为一致或更完善)。
4. `--bart-min-n`/`--bart-min-k` 生效:小于门槛的 BART 任务格写 `status="skipped"`,
   不实际调用 bartpy2 拟合;门槛之上的格子行为不变。CSV 里能同时看到 `ok`/`failed`/`skipped` 三态。
5. `run_panels.py --dry-run --manifest <样例清单>` 能打印出每个 panel 解析后的完整参数
   (含 preset 展开后的具体数值),不实际跑模型。
6. 用一个含 2 个 panel(一个 regression、一个 classification)、`preset: dev` 的样例清单,
   `run_panels.py` 实际跑通,产出 2 个独立 CSV,行数与内容 schema 符合各自 task 的既有规范。
7. 中断一个 panel 的执行再重跑 `run_panels.py`,该 panel 不重复已完成的行,其它 panel 不受影响。
8. `pytest -q` 全绿(含迁移后更新的 import 路径)。

## 测试要求
- 迁移后:回归所有既有测试(30 列回归指标、8 列分类指标、checkpoint 续跑等)在新路径下原样通过。
- BART 门槛:构造 N<10 或 K<2 的合成小样例,断言对应行 `status=="skipped"` 且指标列全 NaN,
  且断言底层没有真的调用 bartpy2(比如 patch/mock 掉 fit,确认未被调用,或用极小样例但不设门槛时
  确认原先会失败/超时——至少要证明门槛确实跳过了本会失败的格子)。
- `run_panels.py`:
  - 用 2 个 panel 的样例清单 + `dev` preset,端到端跑通,断言产出 2 个 CSV、各自 schema 正确。
  - `--only` 只跑指定 panel,断言另一个 panel 未产生输出。
  - `--dry-run` 不产生任何输出文件,只打印/返回解析后的配置。
  - 断点续跑:跑一半中断再续跑,不重复行,且验证 panel 间互不影响。
