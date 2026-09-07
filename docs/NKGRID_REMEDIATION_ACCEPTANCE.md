# NKGRID 实施与验收报告

2026-09-07。六批代码已在 `SMR&FFC` 分别实现并本地提交，基线为 `d5df3df55e10bbce593f3fb8d5db7b10c550701d`。**这不是全部验收通过的声明。** 真实历史 GPA 下降、原集群 OOM、Linux 全引擎和 Slurm 恢复仍有未验收项目。

完整的 A1–F8 输入、基线/候选提交、命令、预期、实际、状态与证据位置见 [逐项验收表](remediation_evidence/acceptance-table.md) 和 [acceptance.json](remediation_evidence/acceptance.json)。部分证据一律标为“未验收”，不折算成通过。环境与日志校验见 [environment.json](remediation_evidence/environment.json)、[证据 SHA-256 索引](remediation_evidence/evidence-index.json)。

## 实现与已核实的边界

| 批次 | 首次实现提交 | 后续针对性修正 | 结果 |
|---|---|---|---|
| M1 | `6cc91f1` | `206c382`、`2544e53`、`934e38a` | 请求/网格/执行按实际训练容量及父变量抽样单位检查；只读历史审计 |
| M2 | `e27dc30` | `5fd3c6d`、`a782d4d` | 原折定义的逐折 boosting；统一曲线选轮；按父变量计宽、单独核算输入驻留 |
| M3 | `83648c2` | `e319141`、`f5f2a09` | 每次 fit 解析 auto/正整数/full；诊断 opt-in；默认仍 auto |
| M4 | `b3ff1e8` | `bcce47f`、`0f1d20c`、`c5bbf2e` | 类型感知折内预处理、X/y 缩放；完整管线 Ridge LOO、Lasso 路径和 stacking OOF |
| M5 | `1800267` | `d8612d7` | 显式 MSE/分母/两个相对分数；旧列不改义；新 serializer 往返及旧版本拒绝 |
| M6 | `95f997f` | `8c00b82`、`4502dcd` | 阶段计时、实际运行环境校验、统一 Slurm 数值线程；沿用原 snapshot 恢复 |

首次实施前完整阅读方案，检查了仓库及父目录适用的 `AGENTS.md`，未发现该文件。最初工作区只有用户的未跟踪方案文档；其要求保留并添加事实修正。未更换分支，未推送、合并或运行完整 production。使用用户提供的身份逐命令提交，没有改全局 Git 配置。

经核对：原 `sampling_units()` 已正确归并派生组；问题在调用点的分组计数及宽度估计。原 log2 网格生成和同 snapshot 恢复已存在，未重写。原 MLP batch 边界是真实机制，“它导致 GPA 下降”仍是假设。细节已同步 [方案事实修正](NKGRID_REMEDIATION_PLAN.md)。

M2 的原生旧方法对照在 M4 之前完成，日志 [M2-portable.txt](remediation_evidence/M2-portable.txt)；之后加强了独立 oracle 参数和真实 git 基线对照。曲线和预测门槛始终为 `rtol=1e-7, atol=1e-9`，选轮离散规则严格比较，没有放宽。LightGBM 原生 4.6 在 N 非 5 整除时余数行不进入 CV；本次保留其既有 CV 索引，最终 fit 仍使用全部 N。

M4 明确是方法变更。Ridge 每折重新学习填补/标准化，再用一次 SVD 计算原 alpha 网格；独立逐 alpha sklearn Ridge oracle 使用 `rtol=1e-10, atol=1e-12`。Lasso 保留降序 alpha 的 warm-start 路径。外层预处理结果仅用于原诊断，实际 CV/stacking 接收保留原始缺失的输入。类型、先验、无观测源、词表和父变量归属遵守原 Adapter 契约；特征宇宙仍以原训练池筛选为条件。

复查中还发现 M5 读取 serializer 层级错误，原“拒绝旧版本”测试没能发现合法新契约也被拒绝。新增真实 JSON 往返测试先产生 [失败证据](remediation_evidence/E6-roundtrip-before-fix.txt)，修正为读取 `public_result_schema.serializer_version` 后 [通过](remediation_evidence/E6-roundtrip-after-fix.txt)。这不替代 Linux 公共 CSV/WAL 发布验收。

## 测试及反向检验

Windows 使用 `.venv-remediation`，安装项目固定 requirements 的依赖。 [最终回归日志](remediation_evidence/final-portable-regression.txt) 为 174 通过，包括原预处理、模型参数和两个 Adapter 的现有回归；[共享输入/数值回归](remediation_evidence/shared-numerical-regression.txt) 另有 54 通过，覆盖输入验证、MLP alpha、先验不变性及旧 Ridge LOO。预期的小迭代 MLP 收敛警告保留，未屏蔽以制造通过。

完整规定命令曾实际执行，因 eager 包导入 `resource` 在 Windows 不可用而失败，见 [全套运行记录](remediation_evidence/full-suite-windows.txt)。portable runner 只建立指向真实源码的包命名空间，未替换 `resource`、`fcntl`、锁或恢复函数。5 个实际引擎/plan/恢复测试明确跳过；默认 deselect 的 slow 测试也不计入通过。

受控临时副本中，恢复分组计数、持有全部折模型、平均折最佳轮数、MLP 固定 200、clone 丢策略、完整 N 拟合生成 OOF、全 N 填补、全 N 缩放、交换分母、负分截断以及移除执行容量门、跳过内容校验、只看完成行数等 13 个错误均须被捕获。结果及实际 pytest 输出见 [mutations.json](remediation_evidence/mutations.json)。工具只把真正测试失败计作捕获，不把导入/收集错误计作成功。原始生产源码从未被 mutation runner 修改。

后三类反向检验在实际方法体/内容校验器/SQLite 键归约器层面完成；执行方法与 SQLite 函数在 Windows 由原 AST 无改动提取。它们不能代替完整 session、恢复入口和 WAL 发布链的目标 Linux 故障注入，相关项目仍标为未验收。

## 真实 FFC 输入与 GPA 诊断

用户提供 `data/ICPSR_31622/DS0015/31622-0015-Zipped_package.zip`。只在全新且隔离的 `FFCWS/data/remediation_20260907/` 提取 `background.dta`、`train.csv`、`test.csv`，用原 Adapter 配置生成 ARD/schema。原 zip、原 `FFCWS/schema`、历史结果、checkpoint 和 WAL 未覆盖。

重建三种宇宙与版本化 JSON **字节哈希及对象完全相等**：median_mode 3400 组/11432 列，median_missing_indicator 8053 组/16085 列，tree_ordinal 3400 组/3400 列；均为 3400 个抽样单位。见 [Adapter 日志](remediation_evidence/FFC-adapter.txt) 和 [原始输入与宇宙校验](remediation_evidence/FFC-provenance.json)。新增 `prepare_remediation_ffc.py` 提供相同流程的全新目录重建入口，其再次完整 Adapter 重建未重复执行。

完整设计预先固定为 median_mode GPA，N=180/199/200/201/220/250/251/252/300，K=10/100，seed=12345/23456，draw=0，auto/full，旧外层预处理/新折内预处理。使用原训练池去除缺失 GPA 后，固定 seed=731、20% 独立诊断验证集；样本/变量嵌套排列和模型 seed 使用原引擎函数。外部测试值不用于拟合、诊断分数或选择默认策略。

所有已完成单元、遗漏单元、失败率、两 seed 的完整 MSE 分布、成对 full-minus-auto 差、每折基模型误差、MLP batch/迭代/收敛、元模型系数/截距，统一记录于 [GPA-paired-summary.json](remediation_evidence/GPA-paired-summary.json)。`complete` 和 `missing_cells` 是完成度的唯一依据，未完成不得推断成功。原始 JSONL 存于隔离数据目录，摘要保存输入/源码/driver/配置哈希及当时提交；运行时存在工作区补充测试修改，不能仅凭提交名推定源码身份。

`wait_remediation_gpa.py` 可在既有两个 fold-local 进程结束后自动汇总并更新 ledger；它不启动新训练、不提交作业或 Git。完成状态见 `GPA-completion-status.json`。尚在增长的两个运行日志不列入未完成阶段的证据哈希索引；完成汇总时再纳入。源码字节审计见 [GPA-source-audit.json](remediation_evidence/GPA-source-audit.json)：科学模型/预处理文件仍匹配运行时哈希；其余后续变更包括 Lasso 路径、serializer 修正、诊断类型标注和计时，须按对应提交区分。

诊断额外重放确定性训练折，带来额外计算成本；两个 fold-local seed 并行，不把这些时间当作严格性能配对。完整管线 LOO 在较大 K 很慢，未为了提速改折数、alpha 网格或训练行数。`full` 与 `auto` 的 epoch 对应不同优化步数，不能解释为等步数优化器对照。元模型仍是非负线性回归，未加权重和为 1 的约束。

较早 smoke 的 N199/K10 两策略相同；N201/K10/seed12345 下 full 在旧新方法均比 auto 差。它们仅是探索性结果，且早期源码身份记录不足，不作为正式单独结论。保留全部设计的摘要用于正式比较。**当前不改变 auto 默认值，也不宣称已解释或修复用户所指历史 GPA 下降。** 尚缺历史评分列、表示、K/seed/draw、manifest 与依赖环境。

## 原生内存与时间

真实 GPA 缺失指示表示，固定 N=201/K=3400，float64，展开 16085 列，seed12345/draw0 派生模型 seed=2335205578，原 YAML 参数、1 个数值线程。四次测试分别启动新进程，从 git 加载指定实现；输入同一 legacy-preprocessed NPZ，哈希见 [native-input.txt](remediation_evidence/native-input.txt)。

| 模型 | 基线峰值 MiB / 秒 | M2 峰值 MiB / 秒 | 最佳轮数 | 最终预测 |
|---|---:|---:|---:|---|
| LightGBM | 518.67 / 6.172 | 282.44 / 34.722 | 都为 10 | 数组 SHA-256 相同 |
| XGBoost | 590.86 / 8.666 | 445.43 / 8.195 | 都为 7 | 数组 SHA-256 相同 |

原始四份 JSON： [LGBM 基线](remediation_evidence/native-lightgbm-baseline.json)、[LGBM M2](remediation_evidence/native-lightgbm-m2.json)、[XGB 基线](remediation_evidence/native-xgboost-baseline.json)、[XGB M2](remediation_evidence/native-xgboost-m2.json)。同时记录 CV 结束累计高水位、最终 fit 时间及最终 fit 后高水位；高水位不是独立阶段瞬时占用，不能相减求释放量。

这是 Windows OS **单进程原生 PeakWorkingSetSize**，包括输入 NPZ 加载，未包含外层完整 ARD/test 常驻副本、任意子进程树或 Slurm cgroup，也未重放原失败配置。LightGBM 因逐折跑满原 max_rounds 后重放聚合早停而明显变慢，不能只报告内存收益。B4 目标环境部分和 B7 仍未验收，OOM 未宣称解决。

## Linux / 集群补验及恢复操作

以下从仓库根目录执行，依赖使用 `NK_Grid/requirements.txt` 的固定版本；先确保当前提交及私有输入完整，日志使用新的隔离目录。不会要求重建或写回历史运行。

```bash
export PYTHONPATH="$PWD/NK_Grid/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
mkdir -p .pytest_cache/remediation-linux
python -m pytest -q NK_Grid/tests SMR/adapter/tests FFCWS/adapter/tests \
  > .pytest_cache/remediation-linux/full-suite.txt 2>&1
python -m pytest -q NK_Grid/tests/test_remediation_contracts.py NK_Grid/tests/test_remediation_posix.py \
  NK_Grid/tests/test_native_process.py NK_Grid/tests/test_flat_task_table.py \
  NK_Grid/tests/test_worker_event_wal.py NK_Grid/tests/test_generation_control.py \
  NK_Grid/tests/test_performance_paths.py NK_Grid/tests/test_flat_task_submission.py \
  > .pytest_cache/remediation-linux/entry-recovery.txt 2>&1
python NK_Grid/tests/run_remediation_mutations.py \
  > .pytest_cache/remediation-linux/mutations.json
```

既有协议用例覆盖超时杀子孙进程、schedule lease、精确 generation、激活故障边界、原子发布及缺键/冲突；必须实际运行并审阅输出。它们不能自动补足 F6 的“删一行再复制补总数”、F7 数值重排对照，以及上述三类 reducer 级 mutant 的完整入口验证；这些需进一步增加或组合实际攻击并记录，不能只运行命令就把 F1–F8 批量改成通过。

真实输入重建与重放（输出路径必须全新，脚本拒绝覆盖）：

```bash
python NK_Grid/tests/prepare_remediation_ffc.py \
  --archive data/ICPSR_31622/DS0015/31622-0015-Zipped_package.zip \
  --output FFCWS/data/remediation_linux_new
python NK_Grid/tests/run_gpa_diagnostic.py \
  --schema FFCWS/data/remediation_linux_new/schema/ffc_median_mode_gpa.json \
  --method legacy --output FFCWS/data/remediation_linux_new/gpa-legacy.jsonl
python NK_Grid/tests/run_gpa_diagnostic.py \
  --schema FFCWS/data/remediation_linux_new/schema/ffc_median_mode_gpa.json \
  --output FFCWS/data/remediation_linux_new/gpa-fold-local.jsonl
python NK_Grid/tests/prepare_remediation_memory_input.py \
  --schema FFCWS/data/remediation_linux_new/schema/ffc_median_missing_indicator_gpa.json \
  --output FFCWS/data/remediation_linux_new/boosting.npz
for model in lightgbm xgboost; do
  for commit in d5df3df55e10bbce593f3fb8d5db7b10c550701d e27dc30; do
    /usr/bin/time -v python NK_Grid/tests/benchmark_remediation_boosting.py \
      --commit "$commit" --model "$model" --seed 2335205578 \
      --input FFCWS/data/remediation_linux_new/boosting.npz \
      --output ".pytest_cache/remediation-linux/native-$model-$commit.json" \
      > ".pytest_cache/remediation-linux/native-$model-$commit.txt" 2>&1
  done
done
```

以上原生 benchmark 仍只是选定单元的模型进程。B7 必须拿到原失败 N/K/seed/draw、表示、CPU、ReqMem、原日志后，使用相同限制重放完整 worker；不可把本 N201 单元冒充原 OOM 单元。Slurm 完成后记录 `sacct -j "$JOB_ID" --format=JobID,State,ExitCode,Elapsed,ReqMem,MaxRSS,AllocCPUS`，并取集群支持的 cgroup 峰值/进程树监测。缺实际站点参数时不生成猜测的提交配置。

恢复**已有兼容计划**使用原入口，不调用 `make_production_request.py` 或重新生成 task table：

```bash
# PLAN 指向已有兼容 plan JSON；默认只打印已有 snapshot 的依赖链。
bash NK_Grid/slurm/submit_flat_task_table.sh "$PLAN"
# 仅在目标集群确认为隔离验收运行后提交同一计划：
# bash NK_Grid/slurm/submit_flat_task_table.sh --submit "$PLAN"
```

旧 serializer/spec 或旧实现提交绑定不兼容时，应在新输出目录建立独立实验；不手工修改 snapshot 版本、worker 数、哈希、激活状态或 WAL 以绕过拒绝。F8 需在同硬件/输入/FS记录冷启动和恢复，确认 task table 哈希与字节不变，收集 `nkgrid_phase` 日志。`.locked` 和外层计时是包含关系，SQLite/Arrow 子阶段也包含在总计时内，不能累加比较。当前没有现场阶段数据，因此未新增缓存或宣称全量索引瓶颈已解决。

## 历史结果影响范围

- M1：若历史 K 实际取分组数（例如 8053）或 N 超容量，标签与训练不符，应从后续分析隔离。不能直接把 K 改成 3400，因为错误 K 可能参与模型 seed 派生。未收到历史结果，尚未生成用户真实历史清单。
- 只读入口：`python NK_Grid/src/aleatoric_nk_grid/audit_history.py /path/to/historical.csv > /new/path/audit.json`。检查源文件哈希、manifest 提交、容量、重复键及证据缺项；不自动改写、删除或合并。缺容量/manifest 不是已证明有效。
- M2：已测旧方法数值相容，但 Git 实现绑定仍需遵守；不混入旧活动运行。
- M3：auto 未变。full 是显式候选；诊断 JSON 新列受 serializer 3 约束。诊断默认关闭。
- M4：内折填补/缩放及原生分箱边界改变，属于新算法身份 `nk-grid-v6-fold-local-1`，不与旧结果当作同方法汇总。完整 LOO 增加计算成本，尚未完成 production 容量评估。
- M5：`r2_test` 与 `skill_score_pct` 保持训练均值零模型定义；新 MSE/两个分母/两个相对分数明确区分。常数分母新分数 NaN；非有限预测拒绝。旧结果无原始预测时不保证可恢复全部新列；RMSE 平方只能标为派生 MSE。
- M6：新 CellExecutionSpec 2 绑定实际依赖/线程环境，公共 serializer 3 不接受旧格式混合恢复/发布。历史 CSV 可独立解释，不代表允许旧 snapshot 接续写新结果。

原始输入、旧 schema、历史 CSV、checkpoint/WAL 未覆盖。新增私有 ARD、NPZ、GPA JSONL 和虚拟环境留在本地，不纳入提交；提交只含代码、方案、验收记录及不含个人观测行的汇总证据。
