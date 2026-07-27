# NK Grid 生产运行风险修复审核

日期：2026-07-27（第二轮修订）

审核状态：待 Wanxiang 批准修订后的范围

Git 状态：所有修改均未提交

工作分支：`SMR&FFC`

修改基线：`5ad5a17393a401833960f9a4fd1aad3cf749ccd9`

## 0. 提交与运行的先后顺序

本文档在未提交状态下供审核，这与"生产必须运行 clean committed code"（R15）**不冲突**——两者处在流程的不同阶段。当前不提交是审核流程的要求，不是遗漏。正确顺序固定为：

```
① 审核本文档 + 代码
   ↓
② Wanxiang 分批批准实施范围
   ↓
③ 本轮先实施 R22 + Elastic Net 参数缩减并重新跑全量测试；
   R23–R26 保留为后续待审项
   ↓
④ 拆成两个独立 commit / tag：
     (a) SMR/FFCWS Adapter 重构
     (b) NK Grid Engine 风险修复
   ↓
⑤ 只有 ④ 完成、工作区 clean 之后，才允许在集群启动任何任务
```

**在 ④ 之前不得从共享工作区启动任何 Slurm 任务。** R15 已经证实过一次"array 排队启动期间共享源码变化"导致同一 array 的不同 task 执行跨代代码；当前工作区同时含有两个独立变更集，风险比那次更高。

§11 的审核文件清单按这两个变更集分组，便于分别 review 和分别提交。

## 1. 结论

Ava 使用的 `7dda462618dbd6dded0ad1b8da9d1445567e4581` 旧版代码确实存在两个会影响生产结果的代码风险：

1. 最终合并把所有 checkpoint CSV 同时载入 pandas，可能在全部模型单元已经算完之后触发 OOM。
2. LightGBM 在 C++ 层发生 segmentation fault 时会直接杀死 Python worker，普通 `try/except` 无法记录失败行。

最新版共享引擎在本次修改前已经具备 checkpoint 分片压缩、原子写入、Slurm 信号停止和重排恢复等保护，但仍没有完全消除上述两个风险。本次未提交修改补上了最终合并、恢复内存、原生崩溃隔离和原生子进程 timeout。用户已另行加入 `timing_full` preset，其 batch size 为 20。

本地测试证明了修复路径和故障隔离机制有效，但还不能等同于 48G、约 200 万结果行的集群生产验收。

对 Ava 提供的三轮 Slurm 日志和四份 snapshot 的后续审核还确认了运行层风险：
同一 array 启动期间共享源码发生变化；16 个任务在同一秒被统一取消；dev pilot
没有完成全部模型组合且未覆盖生产高 K；当前 production 规模对部分模型不可行；
警告日志已形成显著 I/O 噪声；Super Learner 已出现未展开错误详情的 failed rows。
这些问题不能由最终合并和原生进程隔离两项代码修复自动解决。

随后对已完成的 SMR/FFCWS Adapter 重构进行了专项审核。结论是：两个 Adapter 的
基本 contract、生成产物和共享 engine 接口均能运行，实际 20 个 Adapter panels 全部
通过输入验证；但 FFCWS 仍存在一个阻塞生产批准的方法学问题——它会按 outcome
missingness 主动改写 test predictors。除此之外，事务式发布、完整 provenance 校验、
默认 validation model 覆盖和 production manifest 入口仍有缺口。

**第二轮代码审核把范围从"文档修订"扩大到"还需新增代码改动"。** 新增五项
（R22–R26）：原生子进程 timeout runner、运行中 failure policy fail-fast、时间型
checkpoint、SQLite scratch 路径显式化、资源预检。R22 已在本轮实现并通过本地测试；
R23–R26 尚未实施，保留为后续待审项。

## 2. 风险、处理和预期效果

"阻塞范围"取值：**阻塞提交**（不改完不能形成生产 commit）／**阻塞 timing run**（不改完 timing run 结果不可信）／**阻塞 production**（timing run 可以先跑，正式重复数提交前必须解决）／**后续优化**。

| 编号 | 风险 | 判断 | 阻塞范围 | 处理 | 效果 |
|---|---|---|---|---|---|
| R1 | 最终 CSV 合并 OOM | 高，旧版已实际发生 | 已修复，待集群验收 | 用临时 SQLite 做磁盘去重和排序；2,000 行一批写入，10,000 行一批输出；诊断统计也在 SQLite 内完成 | 最终合并的 Python 内存不再随总行数线性增长，不再构造 95,000 个 DataFrame、完整结果表及其多份副本 |
| R2 | LightGBM segmentation fault 杀死整个任务 | 高，旧版已实际发生 | 已修复，待集群验收 | LightGBM 和包含原生模型的 Super Learner 在可复用的 `spawn` 子进程中串行运行；子进程崩溃后重建并默认重试一次 | 原生崩溃被限制在单个模型单元；两次均崩溃时转成普通 `failed` 行，父进程继续并写 checkpoint |
| R3 | 恢复时重新加载全部指标列 | 中到高，大规模恢复会长期占用额外内存 | 已修复，待集群验收 | 恢复只读取 experiment ID、五个 cell key 和 status；生成 pending 列表后主动释放完整设计列表、索引和 completed set | 恢复阶段不再加载几十个指标/诊断列；重复规划结构不会保留到模型拟合结束 |
| R4 | 大量小 checkpoint 文件和反复扫描 | 中，最新版共享引擎已部分解决 | 后续优化 | 保留现有"每 50 个 loose shard 原子压缩一次"的机制，并新增最终磁盘流式合并测试 | **旧版 `7dda462` 没有分片压缩，Ava 现有目录预计是 95,000 个 loose parts；新版压缩后稳定在约 1,900–2,000 个 physical parts。** preset 名义上限约 100,000 次写入；Ava 当前 resolved grid 实际 95,000 次 |
| R5 | batch 太大导致节点失败或强制重排时重复计算过多 | 中 | 见 R24 | production 和用户新增的 `timing_full` 均为 20；用分片压缩控制文件数，而不是扩大故障边界 | 已完成但尚未 checkpoint 的损失上限通常为 20 个单元；USR1 后的响应边界也更小。**固定 20 只是当前状态，不是最终方案，见 R24** |
| R6 | `NODE_FAIL` | **一般基础设施风险；本轮日志中尚未证实任何一次 NODE_FAIL** | 后续优化 | 依靠原子 checkpoint、Slurm requeue 和小 batch 恢复；`_13` 的终止原因必须先由 `sacct` 确认 | 不能防止节点故障，但只要输出目录和分片完整，重排可从最后一个完整 batch 继续 |
| R7 | ConvergenceWarning / LightGBM feature-name warning：既是可观测性问题，也是 I/O 风暴（原 R7 + R19 合并） | 中到高，约 104 万次 warning、531 MiB 日志 | 阻塞 production | warning 在 cell 内捕获并按 model/type 聚合进 checkpoint/QA；stderr 只打印首次和周期性摘要；结果和 manifest 中保留 `converged` / `nonconverged_rows` 统计 | 保留收敛证据，不把 warning 错当作任务崩溃，同时避免真正错误被重复 warning 淹没 |
| R8 | FFCWS 按 outcome-observed train subset 改写 test predictors | **Blocking，方法学语义尚未批准** | 阻塞 production（FFCWS）；不阻塞 SMR timing run | 当前代码将该 subset 未见的 test category 整组/ordinal 值改成 `NaN`；应先决定 engine 三层 unknown-category policy，再与重构作者对齐后决定是否删除 | 在决定前不能把 18 个 FFCWS panels 视为可正式发布；测试通过只证明 masking 被稳定执行 |
| R9 | `unknown_rate_threshold=0.95` 同时控制两种不同风险 | 高，护栏近乎无界 | 阻塞 production | 拆分 raw unknown-code rate 与 outcome coverage/masking rate；分别配置、测试和记录 | 避免只修一个调用点，也避免最多 95% 的已观测值被改写后才失败 |
| R10 | Adapter 直接覆盖正式产物，engine 只强制核对 provenance 的 schema hash | 高，可产生可读取但跨代的 artifact bundle | 阻塞 production | staging、完整验证、READY/artifact lock、原子发布；engine 校验 data/test/manifest/universe/config/source hashes | 中断或并发重跑不能静默混用新旧 ARD、manifest、universe 和 schema |
| R11 | FFCWS 每 outcome 保存完整 predictor 表和 manifest | 中，I/O、磁盘和一致性风险 | 后续优化（依赖 R8） | 仅在 R8 的三层策略获批并取消 outcome-specific masking 后，共享每 strategy 的 train/test 表和 manifest | 物理表从每 strategy 6 份降为 1 份；若保留 masking，本项不能实施 |
| R12 | 全局 category coverage hard fail 与 N/K cell 内 unseen category 静默放行不一致 | 高，未审计的方法学差异 | 阻塞 production | 把完整 train-pool integrity、outcome-observed subset、per-cell sample 分层；per-cell 只做预编码整数诊断 | 避免 Adapter 端"修好"全局检查，却继续在 190 万 cells 中静默使用另一套规则 |
| R13 | `SMR/panels.yaml` 和 `FFCWS/panels.yaml` 当前均固定为 `preset: dev`，提交命令没有 preset override | 中到高，容易误提交小规模任务 | 阻塞 timing run | 提供并审核独立 production manifest，或增加被 snapshot 明确记录的 preset override；production 必须 clean/committed | 防止把 dev 规模误认为 production，也避免临时编辑 tracked manifest 导致 dirty-worktree 冲突 |
| R14 | 两个 Adapter 默认只用 `ols` 做生成后验证 | 中，Adapter success 不等于所有声明模型可运行 | 阻塞 timing run | 默认从 panels 读取全部声明模型，或要求显式 validation profile 并把它写入 provenance | 尤其确保 classification + Super Learner 的 5-fold 类别下限在提交前验证 |
| R15 | array 排队启动期间共享源码发生变化 | **Blocking，已实际发生** | 阻塞提交 | snapshot 必须绑定 clean committed engine/adapter commit、worker script 和 artifact hashes；worker 启动时再次核对，不一致即整组 fail closed。当前工作区含两个独立变更集，必须先按 §0 拆分提交 | 防止同一 array 的不同 task 执行不同代码 |
| R16 | 16 个任务在同一秒被统一取消 | 中，终止来源未记录 | 后续优化 | 将 scheduler/user cancellation 与 model failure 分开分类；保存取消发起者、reason 和 `sacct` 状态 | 不把统一运维操作误报为 16 个独立模型崩溃 |
| R17 | production 规模对慢模型运行时间不可行 | **Blocking；日志强烈提示，待 `sacct` elapsed 校准确认**（原表述"已由日志速度证实"过强，见 §4.5） | 阻塞 production | 先运行 `1 seed × 1 draw × 20 N × 20 K` 的 full-range timing；保留 `batch_size=20`，再依据实测决定正式重复数 | 先得到完整 N/K 单轮成本，避免直接提交数月或数年的任务 |
| R18 | dev pilot 未完成全部模型且最大 K 仅 100 | 高，未覆盖实际崩溃区 | 阻塞 timing run | timing run 使用 `max_n=0`、`max_k=0`（即不截断）；全部 panel×model 必须到达最大 K，LightGBM 最大 K 单独设 Gate | 防止低 K pilot 通过后在 production 高 K 才暴露 native crash |
| R19 | *（已并入 R7）* | — | — | — | — |
| R20 | Super Learner 已出现 failed rows，但日志没有异常详情 | 高，结果风险待确认 | 阻塞 production | 从对应 checkpoint 行读取 `error`；按错误类别统计并对 native/数据/收敛失败采用不同门限 | 确认失败是否系统性集中在特定 N/K，而不是只看总数低于 50 |
| R21 | 当前只收到 logs/snapshots，没有 checkpoint/final/manifest | 高，恢复状态尚未验证 | 阻塞 production | 只读核对 part 连续性、header、行数、唯一 key、failed error 和 final manifest 后再决定恢复或重跑 | 当前 28.95% 只是日志确认的写入进度，不能当作已验证可恢复结果 |
| **R22** | **原生子进程没有任何 timeout，永久 hang 会烧光整个 wall time** | **高，比 segfault 更糟：segfault 有 R2 兜底，hang 无任何机制** | **已修复，待集群校准 timeout** | runner 已改为可 `kill()` 的 `multiprocessing.Process` + `Pipe`；加 per-cell timeout，超时后强杀、重试并在重试耗尽后记 `failed` 行；已新增 hang 测试 | 单 cell 卡死不再吃掉整个 wall time 且零产出 |
| **R23** | **failure policy 只在最终 materialization 之后检查** | **高，最坏情况是跑满数周后才因第 51 个失败报错** | **阻塞 production** | 增加运行中、按错误类别的 fail-fast（见 §8.2）；同时让绝对门限随总 cell 数缩放，避免 5% 门限在 190 万 cell 下永远不生效 | 系统性失败在小时级被发现，而不是数周之后 |
| **R24** | **固定 `batch_size=20` 对所有模型统一，是错误的旋钮** | **中到高，同时制造 R7 的 I/O 噪声和慢模型的长暴露窗口** | **阻塞 timing run** | 改为"达到 N cells **或** T 秒，先到者 checkpoint"（见 §8.3） | OLS 不再产生 95,000 次 fsync；Super Learner 不会数小时不落盘 |
| **R25** | **SQLite DB 建在 output 旁，但临时排序空间走 `TMPDIR`，两者是不同旋钮且都未显式配置** | **中到高，可能落到很小的节点本地 `/tmp` 或很慢的共享 FS** | **阻塞 production**（须先于 R26 实施） | 显式配置 DB 与 scratch 路径；把两处产生临时 B-tree 的查询（median `ORDER BY`、`GROUP BY status`）分别改为单列有界 materialization 和条件聚合（见 §8.4） | 最终化的磁盘行为可预测，不依赖集群默认 `TMPDIR`；external-sort 需求降到 0，output/scratch 两个 Gate 可独立定值 |
| **R26** | **没有提交前资源预检** | **中，磁盘/内存不足会在数周计算之后才暴露** | **阻塞 production**（须在 R25 之后实施） | 启动时**分别**检查 output FS 与 node-local scratch 的可用空间（按 §5.2 的双 filesystem Gate）、cgroup 内存上限，不满足直接 fail closed | 资源不足在第 0 分钟失败，而不是在最终化阶段 |

## 3. 已完成的具体修改

### 3.1 最终合并改为磁盘流式 reducer

文件：`NK_Grid/src/aleatoric_nk_grid/experiment.py`

- 不再执行"读取所有 part → `pd.concat` → `drop_duplicates` → `sort_values`"。
- 逐个流式验证 shard 的 header、关键字段、整数 key 和 status。
- 用 SQLite 主键
  `(experiment_id, model, seed, draw, N, K)` 去重，表声明为 `WITHOUT ROWID`
  （`experiment.py:756`）。
- 保留原有语义：已经成功的 `ok` / `skipped` 记录不会被较晚的失败重试覆盖；同优先级时较晚记录覆盖较早记录。
- 最终 CSV 按 experiment 和 cell key 排序，以同目录临时文件写入、`fsync` 后原子替换。
- manifest 需要的成功数、失败数、未收敛数、拟合时间和最佳迭代轮次在 SQLite 内聚合。
- 最终 QA 逐行验证行数、完成状态、唯一 key 和排序，不再用 pandas 重读整张结果表。
- 临时 SQLite 在正常完成或可捕获的 Python 异常后会清除；原始 checkpoint
  shards 在最终 CSV 验证通过前不会删除。

**排序键与主键一致，这是内存有界结论的支点。** `CHECKPOINT_SORT_COLUMNS =
["model", "seed", "draw", "N", "K"]`（`experiment.py:40`），最终输出的
`ORDER BY experiment_id, model, seed, draw, N, K`（`experiment.py:991`）与主键
逐列相同，加上 `WITHOUT ROWID`（表本身就是主键 B-tree），写 CSV 是纯索引扫描，
不产生任何临时排序文件。**任何一次改动排序键或主键都会把这里静默退化成外部
归并排序，改动时必须同步更新本节和相应测试。**

当前 pragma（`experiment.py:1045–1049`）：

| pragma | 值 | 说明 |
|---|---|---|
| `journal_mode` | `OFF` | 无回滚日志。DB 是一次性产物、shards 保留，可接受；但中断后残留的 DB **绝不能被复用**，当前每次用新 uuid 命名，不存在复用路径 |
| `synchronous` | `OFF` | 同上 |
| `temp_store` | `FILE` | 临时文件落盘而非内存，见 R25 |
| `cache_size` | `-262144` | 即 256 MiB page cache |
| `locking_mode` | `EXCLUSIVE` | 单进程独占 |

**列类型：除 `seed`/`draw`/`N`/`K` 为 `INTEGER` 外，其余全部列声明为 `TEXT`**
（`experiment.py:740–746`），ingest 直接写入从 CSV 读出的字符串
（`experiment.py:784–809`）。两个后果：

- 好的一面：**输出字段的词法表示不会因数值类型转换而改变**。注意这只是字段级
  保证，**不是** "final CSV 与 shards 逐字节一致"——finalization 还会重排、去重、
  重写 header 和 quoting，文件级字节相等本来就不成立。修改列类型前必须先读 §8.4
  的警告。
- 代价：库体积按文本计（实测约为 final CSV 的 1.33 倍，见 §5.2），且任何数值聚合
  都要经 Python UDF `nk_float()` 逐值转换，其中 median 还会因此退化成外部排序
  （R25，见 §8.4）。

内存边界：

- SQLite 插入缓冲：2,000 行。
- CSV 输出缓冲：10,000 行。
- SQLite page cache：上限约 256 MiB。
- 总结果行保存在磁盘数据库中，而不是 Python 对象和 DataFrame 中。

### 3.2 降低恢复阶段内存

文件：`NK_Grid/src/aleatoric_nk_grid/nk_grid.py`

- resume 只投影读取 7 个必要字段（experiment_id + 5 个 cell key + status），而不是完整指标表。
- 旧的单一 CSV 如需继续运行，采用流式复制迁入 `.parts`，不再先转换成 Python record 列表。
- pending 列表确定后释放 `jobs`、checkpoint index 和 completed set。
- manifest 可直接接收 SQLite 生成的汇总对象，不要求持有完整 DataFrame。

这解决的是"恢复和收尾阶段的额外内存"，不会减少单个模型拟合本身所需的内存。规划阶段仍存在一个 `jobs` + index + completed set + pending 同时存活的峰值时刻，其绝对值必须实测，见 §5.4。

### 3.3 隔离原生库崩溃

文件：

- `NK_Grid/src/aleatoric_nk_grid/native_process.py`
- `NK_Grid/src/aleatoric_nk_grid/nk_grid.py`

LightGBM 和 Super Learner 的每个 cell 通过一个单 worker、可复用的 `spawn`
子进程执行。实现使用 `multiprocessing.Process` + `Pipe`，不再使用无法可靠终止
已运行任务的 `ProcessPoolExecutor`：

1. 正常完成时只把 prediction 和少量诊断返回父进程。
2. 子进程 segmentation fault / native abort 时，父进程通过 pipe/sentinel 检测退出。
3. 单 cell 超过 `native_process_timeout_seconds` 时，父进程先 `terminate()`，宽限期后
   仍未退出则 `kill()`。
4. crash 或 timeout 后均销毁旧 worker，启动新解释器并重试该 cell。
5. 默认最多尝试 2 次，可用 `--native-process-max-attempts` 调整。
6. timeout 默认值暂定 21,600 秒（6 小时），可用
   `--native-process-timeout-seconds` 或 panel 配置调整；该值进入 experiment identity 和
   manifest，避免不同 timeout 的结果静默共用。
7. 两次都崩溃或超时时分别生成 `NativeProcessCrashed` /
   `NativeProcessTimedOut`，由现有 cell 级异常处理保存为 `failed` 行。

运行不会因此"静默成功"：failed 行仍进入现有 failure policy。默认失败数超过
`failed_abs_threshold=50` 或失败率超过 `failed_ratio_threshold=0.05` 时，运行会报错并保留最终 CSV 和 checkpoint shards 供诊断。

**R22 已关闭；本机制仍有一个已确认缺口，见 R23：**

- failure policy 只在最终 manifest 构造时求值（`nk_grid.py:1946`，
  `RunFailureThresholdExceeded` 的 docstring 即写明 "Raised after artifacts are
  persisted"）。运行中不会因失败过多提前停止。
- 另外，190 万 cell 下 5% 门限等于 95,000，绝对门限 50 永远先触发，`failed_ratio_threshold` 实际是死配置。

`max_attempts=2` 的默认值暂不改动。**不应仅凭 `_6`（K=472）和 `_16`（K=2739）两次
segfault 就断定所有原生崩溃必然可重现。** timing run 必须记录"重试成功率"（崩溃后
第二次尝试成功的比例），再据此决定默认重试次数。

### 3.4 batch size 现状

文件：`NK_Grid/src/aleatoric_nk_grid/run_panels.py`

- production 原来已经是 20，本次保持不变。
- 用户新增的 `timing_full` 为 `1 seed × 1 draw × 20 N × 20 K`，batch 为 20。
- dev、medium 和 CLI 默认也是 20。

不建议为了减少文件数把 production batch 提高到 500 或 1,000。分片压缩已经把约 100,000 次写入稳定为约 2,000 个物理文件；继续增大 batch 的主要结果是扩大节点失败、超时或强制重排时的重复计算窗口。

**但"所有模型统一 20"不是最终方案（R24）。** 两端都不合适：

- OLS 单 cell 是毫秒级，batch=20 意味着约 95,000 次写 + `fsync`，这正是 R7 描述的共享 FS I/O 噪声来源之一；
- Super Learner 单 cell 可能是分钟级，20 个 cell 的未落盘暴露窗口可达数小时。

正确的旋钮是"达到 N cells **或** T 秒，先到者触发 checkpoint"，见 §8.3。在该改动
落地前，`batch_size=20` 仅作为 timing run 的临时设置。

### 3.5 Elastic Net 参数空间缩减

文件：`NK_Grid/model_params.yaml`、`SMR/model_params.yaml`、
`FFCWS/model_params.yaml`

- 保留 `alpha_log10_min=-4`、`alpha_log10_max=1` 和
  `l1_ratio=[0.1, 0.5, 0.9]`，只把 regression Elastic Net 的
  `n_alphas` 从 50 降到 20。
- 每个 CV fold 的候选组合从 150 个降到 60 个，搜索拟合次数减少 60%；实际墙钟时间
  仍以 timing run 为准，不能直接假定也恰好减少 60%。
- 三份参数文件保持一致，`algorithm_version` 升为
  `nk-grid-v5-adapter-3`，防止旧参数结果被当作可恢复 checkpoint。

## 4. 验证结果

### 4.1 历史回归测试（当前不可复现，仅作记录）

以下是 Adapter 重构前、初轮运行时风险修复的历史测试记录。**该命令当前已不可
复现**：其中 `FFCWS/adapter/data_processor/src` 路径已经删除。当前权威回归结果以
§4.4 为准。

当时在 Python 3.14、本仓库 `.venv` 环境运行：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=NK_Grid/src:FFCWS/adapter/data_processor/src .venv/bin/python -m pytest -q -k 'not retained_legacy_forks_pass_in_isolated_processes'
```

结果：

- 198 passed
- 2 skipped（沙箱禁止创建本地子进程）
- 1 deselected
- 14 warnings，均为现有 sklearn convergence 或 LightGBM feature-name warning

随后在允许本地进程的环境单独运行 `NK_Grid/tests/test_native_process.py`，结果 2 passed。两组结果合并后，当时范围内共有 200 项测试通过。

### 4.2 新增的关键故障测试

- 模拟子进程直接执行 `os._exit(23)`：连续两次崩溃后父进程仍存活，并能创建新子进程继续接收任务。
- 模拟子进程 `sleep` 超过 timeout：确认父进程强制回收旧 worker、记录 timeout，
  随后能创建新 worker 继续运行。
- 验证普通 Python 异常跨进程返回时不销毁可复用 worker；`SystemExit` 等
  `BaseException` 不会被原样抛进父进程。
- 验证 timeout 配置改变会改变 experiment identity，不能静默复用旧 checkpoint。
- 使用仓库真实 LightGBM 参数完成一次跨进程 fit/predict。
- 用 20,000 行 checkpoint 做流式最终合并 smoke test，并把 `pd.concat` 替换为"调用即失败"，证明最终合并路径不依赖完整 DataFrame concat。
- 验证成功记录不会被较晚的 failed retry 覆盖。
- 验证最终 CSV 的排序、唯一 key、状态、诊断汇总和临时 SQLite 清理。
- 验证 production 与 `timing_full` 的 batch 都为 20，并锁定
  `timing_full = 1 seed × 1 draw × 20 N × 20 K`；配置级名义上限估算为：
  - 2,000,000 个模型单元；
  - 100,000 次 checkpoint 写入；
  - 约 2,000 个稳定物理 part；
  - 最多 20 个尚未 checkpoint 的单元。

该估算使用配置请求的 20 个 K levels。Ava 当前 SMR 数据经过整数网格生成和去重后
实际只有 19 个 K levels，因此实际为 1,900,000 cells、95,000 次写入。后续 dry-run
应同时报告 requested levels 和 resolved unique levels，避免把名义上限当作实际计划。

**尚未覆盖的测试场景**（对应 R23–R26，实施后补）：运行中 failure fail-fast、
时间型 checkpoint 触发、scratch 路径缺失时的 fail closed、
资源预检不足时的 fail closed。

### 4.3 静态检查

- `compileall` 通过。
- `git diff --check` 通过。
- 修改文件未发现超过 100 字符的代码行。

### 4.4 SMR/FFCWS Adapter 重构专项验证

针对当前工作区重新运行。**前提**：以下命令不再设置 `PYTHONPATH`，依赖 `.venv`
中已完成的 editable install（`pip install -e`）；在干净环境复现时必须先执行该安装，
否则 import 会失败。

```bash
PYTHONDONTWRITEBYTECODE=1 TMPDIR=/tmp .venv/bin/python -m pytest -q SMR/adapter/tests FFCWS/adapter/tests NK_Grid/tests/test_ffcws_engine_path.py NK_Grid/tests/test_ingest_validate.py NK_Grid/tests/test_preprocessing.py
```

结果：64 passed。

随后运行当前全仓库测试：

```bash
PYTHONDONTWRITEBYTECODE=1 TMPDIR=/tmp .venv/bin/python -m pytest -q
```

结果：

- 208 passed；
- 0 skipped；
- 14 warnings，均为已有 sklearn convergence 或 LightGBM feature-name warning。

实际产物审核：

- FFCWS 18 个 panel 和 SMR 2 个 panel 均使用声明的全部 10 个 models，通过
  `load_input()` + `validate_input()`。
- FFCWS 18 组 schema/provenance 当前 hash 一致；SMR schema、source、contract、
  ARD、manifest 和 universe 的 provenance hash 当前也一致。
- SMR tracked contract 可以由当前 source header 确定性重建：4,252 predictors、
  29 one-hot groups、497 sampling sources，与当前 ARD/schema 一致。
- FFCWS 同一 strategy 的 train ID + predictor projection 在六个 outcomes 间一致。
- FFCWS 同一 strategy 的 test predictor projection 出现 4 个不同 hashes。相对于同一
  baseline outcome，`median_mode`/`median_missing_indicator` 最多有 61 行、598 个
  predictor cells 不同；`tree_ordinal` 最多有 61 行、66 个 cells 不同。
- `outcome_category_coverage.csv` 显示每个 strategy 共执行 333 次
  outcome/source-row category masking，分布在 209 个非零 outcome/source
  combinations。最大单 source **outcome coverage / masking rate 约 0.713%**，低于
  当前 95% 阈值，但这不消除其 outcome-specific 方法学含义。
- **最大单 source raw unknown-code rate 约 1.115%**（与上一条是两个不同指标，共用
  同一个 95% 阈值，见 §5.6）。来源是三个 strategy 的
  `FFCWS/data/adapter_work/*/qa_summary.json` 中 `unknown_categories` 的最大值。
  从仓库根目录执行：

  ```bash
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
  import json
  from pathlib import Path

  for path in sorted(
      Path("FFCWS/data/adapter_work").glob("*/qa_summary.json")
  ):
      rows = json.loads(path.read_text())["unknown_categories"]
      maximum = max(rows, key=lambda row: row["unknown_rate"])
      print(path, maximum)
  PY
  ```

  三份输出的最大项相同：

  ```text
  source_column=hv3pvpercom_m
  unknown_count=9
  denominator=807
  unknown_rate=0.011152416356877323
  ```

  即 `9 / 807 = 1.1152416%`。对应输出文件为：
  `median_mode/qa_summary.json`、`median_missing_indicator/qa_summary.json`
  和 `tree_ordinal/qa_summary.json`。

因此专项审核结论是：

> 当前重构在工程上可加载、可验证、测试全绿；R8–R14 仍需在 production 批准前处理
> 或形成明确、可审计的接受决定。

### 4.5 Ava 旧任务日志实证审核

审核材料包括 Slurm array `24583685`、`24592181`、`24592482` 的 `.err/.out`
和四份 snapshot。收到的目录不包含实际 checkpoint parts、final CSV 或 manifest，
所以以下"已写入"是日志证据，不等于已经完成行级完整性验证。

当前 SMR production 的 resolved N/K grid 为 20 个 N levels、19 个去重后的 K levels。
因此每个 panel×model task 实际是 1,900,000 cells、95,000 batches；20 个 tasks
（2 panels × 10 models）合计 38,000,000 cells。

#### 4.5.1 Per-task 进度表

**本表当前不完整，缺 `sacct` 的 Elapsed / MaxRSS。这是 R17 判定从"已证实"降级为
"待校准确认"的直接原因。** 收到 §6 要求的材料后必须补全，再重新评估 R17。

| Array task | Panel / Model | 最后完整 batch | 已写 cells | 完成率 | 终止原因 | Elapsed | MaxRSS | 推算全量时长 |
|---|---|---|---|---|---|---|---|---|
| `24592482_6` | hourlywage / LightGBM | 13 | 260 | 0.014% | segfault（batch 14，含 K=472） | 待 `sacct` | 待 `sacct` | 待算 |
| `24592482_10` | totalincome / OLS | 95,000 | 1,900,000 | 100% | finalization OOM kill（**不是** OLS fitting） | 待 `sacct` | 待 `sacct` | 约一周（日志推算，待校准） |
| `24592482_13` | totalincome / Elastic Net | 74 | 1,480 | 0.078% | **未知**；日志无 `NODE_FAIL`、OOM、segfault 或 cancellation 终止行 | 待 `sacct` | 待 `sacct` | 待算 |
| `24592482_16` | totalincome / LightGBM | 17 | 340 | 0.018% | segfault（batch 18，含 K=2739） | 待 `sacct` | 待 `sacct` | 待算 |
| 其余 16 个 task | 待逐个填写 | 待填 | 合计 9,098,560 | 平均 29.9% | 全部于 `2026-07-27 09:28:33` 被取消 | 待 `sacct` | 待 `sacct` | 待算 |

聚合核对（供审计）：

```
1,900,000 (_10) + 260 (_6) + 340 (_16) + 1,480 (_13) = 1,902,080
11,000,640 − 1,902,080 = 9,098,560   （其余 16 个 task）
9,098,560 / 16 = 568,660 cells/task = 29.9%
11,000,640 / 38,000,000 = 28.95%
```

#### 4.5.2 由此得到的结论与其证据强度

- `_10` 写完 `95,000/95,000` batches 后才被 OOM kill，**确认 R1 的故障点在
  finalization，而不是 OLS fitting**。证据强度：充分。
- `_6`、`_16` 的 segfault 均发生在 LightGBM 且都在高 K batch，**确认 R2**。
  证据强度：充分。
- `_13` 只确认写完 batch 74、在 batch 75 期间停止。**不能在取得 `sacct` 前宣称
  NODE_FAIL/requeue**（对应 R6 的降级）。
- 其余 16 个 production tasks 在同一秒被取消，相同时间戳表明应先归类为一次统一
  scheduler/user cancellation，而不是 16 个模型故障（R16）。
- **关于 R17 的运行时长**：在 `_10`（OLS）跑完 100% 的同一时间窗口内，其余 16 个
  task 平均只到 29.9%，即约 3.3 倍 OLS 时长。**这支持"部分模型需要数周"，但
  支持不了"ridge/lasso/elastic_net/Super Learner 可达数月"**——后者要求它们只到
  2–5%，而本文档目前没有列出 per-task 的分模型速率。此外 array 并发上限可能让部分
  task 起步就晚，百分比在拿到 Elapsed 之前根本不可横比。
  因此 R17 的当前表述是：**日志强烈提示 production 规模不可行，待 `sacct` 的
  Elapsed 校准后确认。** timing run 的必要性不受影响。
- 另需说明：ridge/lasso/elastic_net 比树模型和神经网络更慢是反直觉的。若成立，
  最可能的原因是 CV path 加上不收敛打满 `max_iter`（与 229,487 次
  ConvergenceWarning 一致）。timing run 必须单独确认这一点，否则 Ava 会认为是笔误。

#### 4.5.3 运行纪律与 pilot 暴露的问题

- `24592181` 的早期 tasks **因 dirty worktree 被正确拒绝运行**（fail-closed 生效），
  同一 array 的后续 tasks 却读到 `nk_grid.py` 的 `SyntaxError: unmatched ')'`。
  这证明同一 array 排队启动期间共享源码发生变化；snapshot v1 没有阻止 worker 执行
  跨代代码（R15）。
- dev array `24583685` 只完成 16/20 tasks；hourlywage/Super Learner、
  totalincome/Random Forest、totalincome/Extra Trees、totalincome/Super Learner
  未完成。dev 最大 K 只有 100，没有覆盖两次 production LightGBM 崩溃所在的高 K（R18）。
- 不完整 production 日志中约有 229,487 次 `ConvergenceWarning` 和 814,989 次
  LightGBM feature-name warning，合计约 104 万次，日志目录约 531 MiB。warning 本身
  通常不是硬失败，但当前输出方式会产生显著 I/O 噪声并掩盖真正错误（R7）。
- hourlywage/Super Learner 至少出现 2 个 failed rows，totalincome/Super Learner
  至少出现 1 个 failed row（合计至少 3 个）；stderr 没有对应 exception 内容，必须
  读取 checkpoint 的 `error` 字段后才能分类（R20）。

#### 4.5.4 建议的 timing run

先增加一个明确标记为 timing、不得用于正式推断的 full-range run：

```text
n_seeds=1, n_draws=1, n_sizes_n=20, n_sizes_k=20,
max_n=0, max_k=0, batch_size=20
```

`max_n=0` / `max_k=0` 表示不截断，必须覆盖到最大 N/K。对当前 SMR grid（20 N × 19
resolved K），它约为 380 cells/panel/model，20 个 task 合计 7,600 cells。

该轮必须覆盖最大 N/K，并按 panel×model 记录 elapsed time、MaxRSS、失败类别、
原生崩溃重试成功率、规划阶段峰值 RSS 和 checkpoint/finalization 时间；完成后再审核
正式 `n_seeds × n_draws`，不能仅凭 nominal cell count 决定。

注意：1 seed × 1 draw 意味着每个 (N,K) 只测一次，没有方差估计。据此外推正式重复数
时应留出余量。

### 4.6 上一轮基线测试债务已解除

上一轮基线中，`NK_Grid/tests/test_legacy_forks.py` 曾尝试进入已经删除的
`FFC/NK_Grid`，导致全量测试需要排除该项。当前重构已经删除 legacy fork tests，
全仓库测试可以直接运行并通过，因此该债务不再是 blocker。

（本项为**已解决**事项，因此从原 §5 移到此处。）

## 5. 当前不能宣称已经完全解决的事项

### 5.1 尚未完成真实生产规模验收

20,000 行 smoke test 证明了算法路径是 out-of-core，但没有模拟 Ava 当前每 task
约 190 万行（preset 名义上限 200 万行）、48G 内存、集群共享文件系统和真实 Slurm
wall time。因此当前结论应是：

> 已修复已知代码机制，等待集群生产规模验收。

不能表述为：

> 已证明所有生产任务都不会 OOM 或崩溃。

### 5.2 磁盘空间成为新的显式约束

流式合并以磁盘换内存。最终化期间可能同时存在：

- 原 checkpoint shards。这里的 compaction 只减少文件数，不压缩 CSV 字节；本地
  totalincome dev 实测 parts 合计 14,293,098 bytes、final CSV 13,275,707 bytes，
  即 parts 约为 final 的 1.077 倍；
- 临时 SQLite。`WITHOUT ROWID` 因此没有额外的唯一索引副本，但**不能按"二进制存储
  小于等价文本"估算**：`_create_checkpoint_table`（`experiment.py:740–746`）只把
  `seed`/`draw`/`N`/`K` 声明为 `INTEGER`，**其余全部列都是 `TEXT`**，而 ingest
  直接把 CSV 读出的字符串写入（`experiment.py:784–809`），所以指标列在库内就是
  与 CSV 相同的文本。**16,000 行 dev 实测 17,666,048 B，即 final CSV 的 1.33 倍**；
- 旧 final CSV（若有，1 倍）；
- 新 final CSV 的原子写入临时文件（1 倍）。

**按 filesystem 拆分，而不是给一个合并倍数。** §8.4 会把 SQLite 移到 node-local
scratch，两个 filesystem 的约束此后互相独立：

| Filesystem | 并存产物 | 相对 final CSV | 全新运行 | resume / 已有输出目录重跑 |
|---|---|---:|---:|---:|
| **output FS** | shards 1.077 + 旧 final 1.0 + 新临时 final 1.0 | — | **2.08×** | **3.08×** |
| **scratch FS** | 临时 SQLite | 1.33 | **1.33×** | **1.33×** |

初始 Gate（含余量，向上取整）：

- **output FS：3.5 × final CSV/task。** 按 1.5–1.7 GiB/final CSV，即
  **每 task 约 6 GiB**；20 个 SMR tasks 若共用同一 output FS，聚合预检需
  **约 120 GiB**，且必须防止各 task 分别通过检查后共同耗尽空间。
- **scratch FS：2.0 × final CSV/task**（1.33 倍实测 + 余量，若采用 §8.4 的
  fallback 方案还需覆盖建索引的临时排序）。即**每 task 约 3.5 GiB**。
  **scratch 是 node-local，因此这一项按"同一节点并发 task 数"相乘，不是按 20 相乘。**

**这些是保守起点，不是最终保证。** Gate D 的大规模 finalization 必须实测峰值并
替换上述数字。在 §8.4 落地前，SQLite 仍建在 output 旁，此时 output FS 的实际需求
是上表两行之和（全新运行约 3.41 倍、resume 约 4.41 倍），原先的 5 倍单一 Gate
在这种情况下最坏余量只剩约 0.6 倍，是 §8.4 必须先于 §8.5 实施的直接原因。

现有两份完整 SMR dev final CSV 提供了可执行的初始行宽：

| 文件 | 数据行 | 文件大小 | 平均 bytes/row |
|---|---:|---:|---:|
| `nk_grid_smr_hourlywage_dev_20260717-120504.csv` | 16,000 | 13,315,351 | 832.21 |
| `nk_grid_smr_totalincome_dev_20260717-130352.csv` | 16,000 | 13,275,707 | 829.73 |

按该实测值线性外推，1,900,000 行 legacy regression final CSV 约为 1.47 GiB
（1.58 GB）。当前 v5 schema 新增了 identity 和 expanded-feature 等列，因此提交前
暂按 **1.5–1.7 GiB/final CSV** 估算。代入上面的双 filesystem Gate（按上界 1.7 GiB
向上取整）：

- **output FS：每 task 约 6 GiB**；20 个 SMR tasks 共用同一 output FS 时，提交级
  聚合预检至少 **120 GiB**，且必须防止多个 task 分别通过检查后共同耗尽空间。
- **scratch FS：每 task 约 3.5 GiB**，按同一节点上的并发 task 数相乘。

（此前写作"5 倍单一 Gate、8.5 GiB/task、聚合 170 GiB"，那是把两个 filesystem 合并
计算的结果，已按 §8.4 的路径拆开。在 §8.4 落地前 SQLite 仍建在 output 旁，此时
output FS 的实际需求是两行之和，约 8 GiB/task、聚合 160 GiB。）

timing run 应更新真实 v5 bytes/row；Gate D 再以实测峰值替换上述初始估算。

空间不足时，原始 shards 会保留，但最终 CSV 不能完成。进程如果被 `SIGKILL`，可能
遗留两类孤儿文件，都需要在确认没有同名任务运行后清理：

- 隐藏的 `.materialize-*.sqlite`；
- final CSV 的原子写入临时文件。

每次运行使用新的 uuid 命名 SQLite（`experiment.py:729`），不存在复用残留 DB 的
路径；但由于 `journal_mode=OFF`，残留 DB 本身可能处于损坏状态，**绝不能被手工
复用**。

### 5.3 原生子进程有序列化和峰值内存开销

父进程需要把当前 cell 的训练/测试数据发送给子进程，因此 LightGBM/Super Learner 会增加序列化时间，并可能短暂同时保留父、子两份当前 cell 数据。它隔离了 segmentation fault，但不能保证以下情况不会发生：

- 单个超大 cell 本身超过 Slurm cgroup 内存限制；
- 集群策略在子进程 OOM 时杀死整个 job step；
- 原生库不崩溃但永久 hang——**这一项已升级为 R22，将在 §8.1 修复并补测试，不再
  作为"接受的残余风险"**。

前两项需要真实集群的最大 K cell 验收。

### 5.4 生产规划仍不是完全流式

运行仍会创建完整 job/pending 列表，并在恢复时短暂创建 completed-key set。指标列已被
移除且重复结构会在拟合前释放，但存在一个 `jobs` + checkpoint index + completed set +
pending **同时存活**的峰值时刻。

**该峰值的绝对值必须实测，本文档不再给出估计数字。** timing run 必须记录规划阶段
峰值 RSS，并据此决定是否需要把规划改成分块流式。

### 5.5 FFCWS test predictors 仍随 outcome 改变（R8）

`FFCWS/adapter/src/ffcws_data_processor/contract.py` 中的
`enforce_outcome_train_category_coverage()`（当前第 27–125 行）会：

1. 对每个 outcome 重新选择 outcome-observed train rows；
2. 找出该 subset 中未见、但出现在 outcome-observed test rows 的 categorical state；
3. 将对应 one-hot group 或 ordinal predictor 改成 `NaN`。

调用仍位于 `pipeline.py:174–180` 的 outcome materialization 之后。该行为不是意外遗留：
函数 docstring 明确主张"只在 outcome-missing train rows 出现的 category 仍应视为
unknown"，测试也明确断言它必须被 mask。

因此测试通过不能被解释为"Q2 已解决"，只能证明当前 outcome-specific 语义稳定。它
造成同一个 test respondent 在不同 outcome panel 中拥有不同 predictor values。正式
生产前必须在以下两者中作出方法学决定：

- 采用三层策略：只用完整 official train pool 定义 vocabulary，不按 outcome
  missingness 改写 predictors；outcome/cell subset 的未见状态只记录诊断。
- 保留当前策略：把 outcome-specific preprocessing 明确写入正式 contract，并接受
  无法共享多 outcome 物理表及跨 outcome predictor 不一致。

建议采用第一项，但删除代码前必须与本轮 FFCWS 重构作者对齐并记录书面结论。

**本项阻塞 FFCWS production，不阻塞 SMR timing run。**

### 5.6 一个 95% 阈值控制两个不同语义的 raise（R9）

`FFCWS/adapter/contracts/ffc.yaml:51` 的 `unknown_rate_threshold: 0.95` 同时进入：

- `common/validation.py::unknown_qa_row()`：test raw unknown-code rate；
- `contract.py::enforce_outcome_train_category_coverage()`：
  outcome-observed coverage/masking rate。

前者是在防 raw mapping/join 或 vocabulary 问题，后者是在实施 outcome-specific
preprocessing。两者不能共享一个阈值。当前实际最大 rate 约 1.115%（raw unknown）和
0.713%（outcome coverage）——两个数字的实测来源见 §4.4，其中 raw unknown 的来源
命令待补——所以 95% 不会在正常偏差扩大时提供有效预警。

实施时必须同时点名修改这两个调用点：

- official train vocabulary 之外的 raw test state：默认 hard fail，或使用按 source
  审批的窄阈值；
- outcome/cell subset 内未见但 official vocabulary 合法的 state：记录诊断，不复用
  raw unknown threshold。

### 5.7 Adapter 发布和 provenance 还不是生产门禁（R10）

SMR（`prepare.py:277–345`）和 FFCWS（`pipeline.py:152–260`）都按
"table → manifest → universe/schema → provenance"的顺序直接写正式路径。进程中断
或两个生成进程重叠时，可能出现新旧文件组合。

当前生成的 hashes 全部匹配，但 engine 的 `_validate_provenance()`
（`validate_input.py:307–317`）只强制比较 `schema_sha256`。它没有强制验证
provenance 中已经存在的 data/test/ARD、manifest 和 universe hashes。FFCWS 的
per-dataset provenance 还没有记录 raw input hashes、contract/config hash、Adapter
commit 或 strategy-level metadata identity。

因此"当前 hash 正确"不能证明下一次中断生成会 fail closed。生产前需要：

1. 同 filesystem staging generation；
2. staging 内完整重读和验证；
3. 包含 source/config/code 与所有 artifact hashes 的 lock/READY；
4. 最后原子发布；
5. engine 在读取任何模型 cell 前验证全部 hashes。

**这五步都是尚未编写的代码**，见 §8.5 与 §9 的顺序约束。

### 5.8 FFCWS outcome-specific 物理副本（R11）

当前 FFCWS 仍生成：

- 3 strategies × 6 outcomes = 18 个 dataset directories；
- 每个目录一份完整 train/test predictor table；
- 每个目录一份相同 strategy 的 `feature_manifest.csv` 和 provenance。

当前 ARD 目录约 480 MiB；绝对数字会随 Parquet 版本和压缩变化，正确验收指标应是：
每个 strategy 的物理 train/test 表从 6 份降到 1 份、manifest 从 6 份降到 1 份，总体约为
旧布局的 1/6。

该改动严格依赖 R8：只要 outcome-specific masking 存在，六个 test predictor
projections 就不相同，不能安全合并。

### 5.9 全局与 per-cell category policy 断层（R12）

Engine 当前在 outcome 删除后执行全局 external category coverage hard check；FFCWS
通过 R8 的 masking 使其通过。但 N/K sampling 后，每个 cell 只使用 N 行训练数据，
小 N cell 缺少合法 categories 是常态，`preprocess_cell()` 对这种情况静默放行。

同一现象因此在全局是致命错误，在 cell 内却没有诊断。建议固定三层定义：

1. 完整 official train pool vocabulary integrity：hard contract；
2. outcome-observed subset 缺类别：允许，记录 panel/outcome QA；
3. N/K cell subset 缺类别：允许，记录定长聚合诊断。

Per-cell 诊断不得调用逐行 `_categorical_states()`。必须在 panel 级预编码整数 state
codes 和 test state set；cell 内只做 `np.unique(codes[train_idx])` 与预计算集合比较，
避免在每个约 1,900,000-cell task（preset 名义上限 2,000,000）中引入 Python
逐行循环。

### 5.10 当前没有明确的 production manifest 入口（R13）

`SMR/panels.yaml:2` 和 `FFCWS/panels.yaml:2` 当前根级均为 `preset: dev`。
`aleatoric-nk-grid-panels` 和 `submit_nk_grid.sh` 没有 `--preset production`
override；`--allow-large-run` 只授权大任务，不会把 dev 配置变成 production。

这是一个安全默认，但当前 README 没有给出可复现的 production manifest/命令。直接
提交当前 manifest 只会生成 dev grid；临时编辑 tracked manifest 又会与 production
clean-worktree policy 冲突。

建议保留 dev 默认，同时增加经过审核、tracked 的 production manifest，或增加会被
snapshot 完整记录的显式 preset override。提交 receipt 必须记录最终 resolved preset
和 cell 数。

**timing run 同样需要这个入口**，否则无法在不弄脏工作区的前提下跑 full-range grid。

### 5.11 Adapter 默认验证模型覆盖不足（R14）

`SMR/adapter/prepare.py` 和
`FFCWS/adapter/src/ffcws_data_processor/pipeline.py` 的
`validation_models` 默认值（分别为 `prepare.py:256`、`pipeline.py:61`）都只有
`("ols",)`；CLI 默认也只有 `ols`。但两个 `panels.yaml` 均声明 10 个 models。

当前实际数据经本次人工审计可通过全部 10 个 models，但 routine Adapter success 仍只
证明 OLS 的输入下限。对于 FFCWS classification panels，Super Learner 要求每类至少
满足 5-fold，而默认 OLS 验证只触发较弱的 2-fold 下限。

建议 Adapter 默认读取对应 panels 的完整 model set，或要求显式选择并记录
validation profile。使用的 models、`min_n`、seed 和 engine version/hash 应写入
validation report/provenance。

## 6. 检查 Ava 当前任务需要的文件和信息

本次已经收到完整度不一的 Slurm logs 和 snapshot，但所给目录没有 checkpoint
parts、final CSV 或 manifest。因此目前能确认运行故障和日志进度，不能确认分片
完整性、failed row 内容或最终结果可恢复性。

下一步先收集下面这些内容即可继续定位，不需要一开始传全部 95,000 个 part 文件
（旧版无分片压缩，预计确实是 95,000 个 loose parts）。

1. 四个重点 array task（`_6`、`_10`、`_13`、`_16`）的完整 Slurm
   stdout/stderr，不能只有截图或最后几十行。

2. **`sacct` 是主要来源**（用于补全 §4.5.1 的表，进而校准 R17）：

   ```bash
   sacct -j JOB_ID -P --units=M --format=JobID,JobName,State,ExitCode,Elapsed,Restarts,NodeList,ReqMem,MaxRSS
   ```

   注意：`MaxRSS` 只出现在 `.batch` / step 行上，不在主 job 行上；用 `-P` 输出
   管道分隔便于直接粘贴。请连同 `.batch` 行一起提供，不要过滤。

   `scontrol` **仅在 job 仍保留在 scheduler 内存中时可用**（受 `MinJobAge` 限制，
   常见默认 300 秒）。这些 job 已于 07-27 09:28 终止，大概率会返回
   "Invalid job id specified"。若仍在保留期内则一并提供：

   ```bash
   scontrol show job -dd JOB_ID
   ```

3. 实际提交时使用的 sbatch/launcher 脚本、array index 到 panel/model 的映射（`_13`
   已由日志头确认为 `smr_totalincome / elastic_net`；映射仍用于补全 §4.5.1 的其余
   16 个 tasks），以及提交 receipt/snapshot（如果旧版本已生成）。

4. `panels.yaml`、`model_params.yaml`、实际运行的 commit SHA、`pip freeze`，以及集群加载的 Python/CUDA/module 信息。

5. 每个问题任务的 output 目录清单和空间统计。GNU（集群 Linux）：

   ```bash
   find OUTPUT_DIR -maxdepth 2 -type f -printf '%p %s\n'
   du -sh OUTPUT_DIR OUTPUT_DIR/*.parts 2>/dev/null
   ```

   macOS / BSD find 没有 `-printf`，改用：

   ```bash
   find OUTPUT_DIR -maxdepth 2 -type f -exec stat -f '%N %z' {} +
   ```

6. **若存在**，提供 final CSV 和 manifest JSON。（据 §4.5，预计没有任何一个
   production task 生成过经验证的 final manifest，因此本项可能为空——为空本身
   也是需要确认的信息。）另外提供任意一个早期、一个中间、一个末期 part
   的 header 和前两行。数据敏感时可只提供 header、行数、文件大小和脱敏样例。

7. 对 LightGBM segfault，如果集群保留 core dump，再提供 core/backtrace 和
   LightGBM 动态库版本；如果没有 core dump，完整 stderr、ExitCode、节点名和
   LightGBM 版本也足以先确认故障类别。

8. 集群 `TMPDIR` / node-local scratch 的实际路径与可用空间，以及输出目录所在
   文件系统的可用空间（用于 R25/R26 的配置和预检 Gate）。

这些材料可以分别回答：是节点故障、cgroup OOM、Python 异常、原生库崩溃、final merge
OOM，还是 checkpoint/输出损坏。

## 7. 对 Ava 当前旧任务的影响

这份修改位于当前共享引擎 `NK_Grid/`，不会自动改变正在用旧提交 `7dda462...` 运行的任务，也不能假定新版 experiment identity、目录结构和旧的 95,000 个 part 可以直接互相 resume。

对现有任务建议：

1. 不要删除、移动或覆盖现有 `.parts` / checkpoint 文件。
2. `_13` 当前只能确认在 batch 75 中途停止；先用 `sacct` 核对 State、ExitCode、
   Restarts 和 NodeList。只有确认 NODE_FAIL 且 experiment identity/output path
   未变化后，才按原版本 requeue/resume。
3. `_10` 已经算完模型单元；最稳妥的是针对它的旧分片做只读备份，再用兼容旧 schema 的一次性流式 materializer 生成 final CSV，不应重跑 95,000 个 batch。
   **该 materializer 目前不存在**，属于 §8.6 的新增工作项，需要单独排期和测试；
   在它写好之前，`_10` 的分片只做备份，不做任何处理。
4. `_6`、`_16` 的 LightGBM 需要在审批后的隔离版本上重跑，或把原生子进程隔离补丁定向 backport 到旧运行分支。
5. 对同一秒取消的 16 个 tasks，先取得 cancellation reason/initiator，再决定 resume；
   不把它们登记为模型失败。
6. 从两个 Super Learner checkpoint 目录提取 failed rows 的 `error`，确认是否集中在
   特定 N/K。
7. 不要把新旧版本的 shard 混在同一输出目录，除非先完成明确的 schema/identity 迁移验证。

## 8. 新增改动状态（R22–R26 + 依赖项）

R22 已在本轮完成；R23–R26 尚未实施，§9 对应的验收 Gate 仍依赖它们完成。

### 8.1 原生子进程 timeout runner（R22）

状态：**已实现，未提交，已通过本地测试。**

- `ProcessPoolExecutor` 已替换为可回收的 `multiprocessing.Process` + `Pipe`。
- 新增 CLI/panel 参数 `native_process_timeout_seconds`。超时后执行
  `terminate()` → 有界宽限期 → `kill()`，丢弃 worker 并按现有尝试次数重试；
  重试耗尽后由 cell 异常路径写入 `failed` 行。
- 默认值暂定 6 小时，目的只是先提供硬上限；timing run 必须记录最大/高分位 cell
  时长，再决定生产默认值。
- hang、crash、普通异常、`SystemExit` 隔离和真实 LightGBM 跨进程拟合均有测试。

### 8.2 运行中 failure policy fail-fast（R23）

现状：`_failure_policy_violation` 仅在 `nk_grid.py:1946` 构造最终 manifest 时求值，
`RunFailureThresholdExceeded` 的 docstring 即写明 "Raised after artifacts are
persisted"。最坏情况是跑满数周后才因第 51 个失败报错。

改动：

- 在 checkpoint 写入处按批次求值失败计数，超限即停止并保留已有产物。
- 按错误类别分别设门限：`native_crash` / `native_timeout` / 数据错误 / 收敛失败
  应有不同容忍度，而不是共用一个总数。
- 让绝对门限随总 cell 数缩放，或明确记录"在当前规模下 `failed_ratio_threshold`
  是死配置"并移除，避免留下永不生效的配置项误导后续读者。
- 新增测试：注入超过门限的失败，断言运行在中途停止且已写 shards 完整可读。

### 8.3 时间型 checkpoint（R24）

改动：把 `batch_size` 语义从"固定 N cells"改为"达到 N cells **或** T 秒，先到者
触发 checkpoint"。新增 `checkpoint_max_seconds`（或等价命名）配置项，写入 manifest
和 snapshot。新增测试覆盖两个触发路径。

目标：OLS 不再产生约 95,000 次 `fsync`；Super Learner 不会数小时不落盘。

### 8.4 SQLite scratch 路径显式化（R25）

现状：DB 建在 output 旁（`experiment.py:729`），`temp_store=FILE`
（`experiment.py:1047`）。但**临时排序空间不跟随 DB 路径**，走
`SQLITE_TMPDIR` → `TMPDIR` → `/var/tmp` → `/tmp`，是两个独立旋钮，当前都未显式配置。

好消息：最终 CSV 的 `ORDER BY` 与主键逐列相同且表为 `WITHOUT ROWID`，写 CSV 是纯
索引扫描，**零临时排序**。

唯一吃临时空间的是诊断 median（`experiment.py:833–835`）：

```sql
SELECT nk_float(col) FROM checkpoint_rows WHERE ... ORDER BY nk_float(col) LIMIT 2 OFFSET ?
```

`nk_float` 是 Python UDF（`experiment.py:1042` 注册），因此无法走索引，SQLite 必须
对该 model 的全部非空值做外部排序，且每行触发一次 Python 回调。190 万行下这既费
临时空间也费时间。

需要 UDF 的原因是**所有指标列在库内都是 `TEXT`**（`experiment.py:740–746`，只有
`seed`/`draw`/`N`/`K` 是 `INTEGER`），必须在查询时逐值转 float。

需要 median 的列只有两个：`_fit_seconds` 和 `_best_rounds`
（`experiment.py:917–931`）。其余诊断都是 count/sum/min/max，不需要排序。

改动：

**(1) scratch 路径显式化。** 新增配置项分别指定 DB 目录与 scratch 目录（默认取
node-local scratch，不是共享 FS），启动时校验两者存在且可写，缺失即 fail closed。

**(2) median 改为「单列有界 materialization + exact median」（首选方案）。**

**注意命名：这不是"流式"。** 该方案会把单列完整读进内存，只是内存量有界且可
预测（`行数 × 8 bytes`）。1,900,000 个 float64 = **14.5 MiB**，**逐 model、逐列
处理**，任一时刻只保留一列：

```python
values = np.fromiter(...)          # 单列，14.5 MiB
median = np.median(values, overwrite_input=True)   # 避免额外复制
```

- 磁盘增量 0，SQLite external sort 需求 0，无建索引排序。
- 14.5 MiB 相对已预算的 256 MiB page cache 可忽略。这与 §3.1"不再用 pandas 重读
  整张结果表"的设计意图不冲突：那指的是几十列的完整 DataFrame，这里是单列。

**(2a) 不得直接对未经验证的 TEXT 使用 `CAST`。** SQLite `CAST(... AS REAL)` 对
非法输入不报错，实测（SQLite 3.53.0）有三类静默失败：

| 类别 | 输入 | `CAST AS REAL` | Python `float()` |
|---|---|---|---|
| 静默归零 | `'nan'` / `'inf'` / `'abc'` / `''` | `0.0` | `nan` / `inf` / ValueError / ValueError |
| **静默前缀截断** | `'5abc'` / `'1,000'` | **`5.0`** / **`1.0`** | ValueError |
| 溢出为非有限 | `'1e400'` | **`inf`** | `inf` |

第二类最危险：它产出一个看似合理的错值，在结果里无法察觉，而第一类至少会把
median 明显拉向 0。第三类说明"`CAST` 不会产出非有限值"的假设不成立，仅靠
`IS NOT NULL AND <> ''` 过滤挡不住。

因此必须：

1. **ingest 时**验证 `_fit_seconds`、`_best_rounds` 只能是空值或 finite numeric
   （`float()` 成功**且** `math.isfinite()` 为真——注意 `float('nan')` 不抛异常，
   单靠 `float()` 不够）；
2. malformed / non-finite 按既定策略处理，**建议 fail closed**，并把违规行的
   cell key 写进错误信息；
3. 只有验证通过后，SQL 才可以安全使用 `CAST(... AS REAL)`；
4. 测试覆盖：空字符串、`nan`、`NaN`、`inf`、`-inf`、`Infinity`、任意非法字符串、
   **前缀可解析字符串（`5abc`、`1,000`）**、科学计数法（`1e-05`、`1E+3`）、
   溢出（`1e400`）、前后空白（`'  2.5  '`）。

**(2b) median 不是唯一的临时 B-tree，`GROUP BY status` 也是。** 对真实 schema
（`WITHOUT ROWID` + 6 列复合主键）逐条跑 `EXPLAIN QUERY PLAN`，finalization 路径
共 9 条查询，**恰好 2 条**使用临时 B-tree：

| 位置 | 查询 | TEMP B-TREE |
|---|---|---|
| `experiment.py:834` | median，`ORDER BY nk_float(...)` | **YES**（`FOR ORDER BY`） |
| `experiment.py:957` | `SELECT status, COUNT(*) ... GROUP BY status` | **YES**（`FOR GROUP BY`） |
| `experiment.py:826` | count/sum/min/max | no |
| `experiment.py:864` | `boolean_count` | no |
| `experiment.py:879` | `nonconverged_count` | no |
| `experiment.py:894` | `SELECT DISTINCT model` | no（可由 PK 顺序满足） |
| `experiment.py:903` | 每 model `COUNT(*)` | no |
| `experiment.py:991` | final CSV `ORDER BY`（== PK） | no |
| `experiment.py:1061` | `SELECT DISTINCT experiment_id` | no（可由 PK 顺序满足） |

所以**只去掉 median 排序不够**。`GROUP BY status` 必须一并改成条件聚合：

```sql
SELECT
  COUNT(*),
  SUM(CASE WHEN status = 'ok'      THEN 1 ELSE 0 END),
  SUM(CASE WHEN status = 'skipped' THEN 1 ELSE 0 END),
  SUM(CASE WHEN status = 'failed'  THEN 1 ELSE 0 END)
FROM checkpoint_rows WHERE experiment_id = ?
```

两条替代查询均已验证为 `no`。**两处都改完之后，finalization 路径才真正不产生
external sort 文件。**

注：`EXPLAIN QUERY PLAN` 的结果与 SQLite 版本相关（上述实测为 3.53.0），因此
不能只在本地验一次就当作永久成立，必须落成 (5) 中的 query-plan 测试。

**(3) fallback：REAL shadow column + 复合索引（暂不实施）。** 仅在将来某个
checkpoint 需要同时对多 model 做大量分位统计时启用。16,000 行 dev 实测成本：

| 状态 | SQLite 大小 | 相对无 shadow |
|---|---:|---:|
| 原始 TEXT 表 | 17,666,048 B | 1.000× |
| 加两个 REAL shadow、无索引 | 17,752,064 B | 1.005× |
| 加 `_fit_seconds` 索引 | 18,616,320 B | 1.054× |
| 加两个索引 | 19,357,696 B | **1.096×** |

即 shadow 字段本身很便宜，**成本几乎全部来自索引**：约 106 bytes/row，外推 190 万行
约 **+192 MiB**，使 SQLite 相对 final CSV 从 1.33 倍升到 1.46 倍。原因是索引必须是
复合索引：

```sql
(experiment_id, model, _fit_seconds_real)
(experiment_id, model, _best_rounds_real)
```

只索引 shadow value 虽然能提供排序，但会扫到其他 experiment/model；而复合 key 的
前两列都是 TEXT，每条索引项都要把它们重存一遍——**索引成本由复合 key 宽度决定，
不是由 shadow value 决定**。

若启用该 fallback，还须注意：`CREATE INDEX` 自身需要临时排序，其峰值必须纳入
§5.2 的 scratch Gate。**不要改成"ingest 前建索引、随插入维护"**——shadow value 的
到达顺序相对其排序位置是随机的，190 万行会在超出 256 MiB page cache 后退化为随机
页写，很可能比一次性 bulk sort 更慢。

**(4) 不要把全部指标列改成 `REAL`。** 当前 TEXT 存储保证输出字段的词法表示不因
数值类型转换而改变；一旦转成 REAL 再输出，浮点会被重新格式化（例如 `1e-05` 与
`0.00001`、尾数末位差异），会静默改变输出内容并破坏与旧结果的比对。首选方案的
`CAST` 只发生在聚合查询内，不触及输出列，因此没有这个问题。

**(5) 新增测试**：

- scratch 目录不存在或不可写时 fail closed；
- ingest 数值验证的全部边界用例，见 (2a) 第 4 条；malformed 输入必须 fail closed
  且错误信息包含违规行的 cell key；
- median 结果与改动前逐值一致（含奇偶行数、全 NULL、单行等边界）；
- **query-plan 测试**：对 finalization 路径的每一条生产查询断言其
  `EXPLAIN QUERY PLAN` 不含 `USE TEMP B-TREE`。这条测试是防回归的关键——
  查询计划随 SQLite 版本变化，集群上的版本可能与开发机不同；
- **final CSV 与改动前 materializer 的输出逐字节一致**（回归比对；注意这不是与
  shards 比对——finalization 会重排、去重、重写 header 和 quoting）。

### 8.5 提交前资源预检（R26）

**必须先完成 §8.4**，否则 scratch 与 output 还在同一 filesystem 上，阈值要定两次。

改动：在启动时（而不是最终化时）检查并 fail closed。**两个 filesystem 分别检查，
不合并成一个倍数**（§5.2 给出初始值，按 1.5–1.7 GiB/final CSV）：

| 检查项 | Filesystem | 初始 Gate | 每 task | 聚合方式 |
|---|---|---|---|---|
| shards + 旧 final + 新临时 final | output FS（多为共享） | 3.5 × final CSV | 约 6 GiB | **× 同一 output FS 上的并发 task 数**（20 tasks 共用则约 120 GiB） |
| 临时 SQLite（+ fallback 的建索引排序） | node-local scratch | 2.0 × final CSV | 约 3.5 GiB | **× 同一节点上的并发 task 数**，不是 × 20 |

另外检查：

- cgroup 内存上限与 preset 规模匹配；
- 两个目录均存在且可写（与 §8.4(1) 共用同一校验）；
- 把检查结果、两个 filesystem 的实测可用空间和最终 resolved preset 写入
  snapshot/receipt。

必须同时做 **task 级**和 **submission 聚合级**检查，避免并发 task 分别通过检查后
共同耗尽空间——这一点只对 output FS 成立，node-local scratch 按节点并发数算。

timing run 必须更新 v5 实际行宽；Gate D 的大规模 finalization 验收再以实测峰值
定死上述阈值。

### 8.6 `_10` 专用一次性 materializer（§7.3 的依赖）

为兼容旧 schema 的 `_10` 分片单独编写一次性流式 materializer，只读旧 parts、
输出 final CSV，不写回原目录。需要独立测试与 code review，不与主线 engine 混用。
在它完成之前，`_10` 的分片只做只读备份。

### 8.7 依赖顺序

```
R8 方法学决议 ──▶ R11 物理表合并
R13 production/timing manifest ──▶ 任何 full-range timing run
R14 validation model 覆盖 ──▶ 任何 full-range timing run
R24 时间型 checkpoint ──▶ timing run（否则测到的 I/O 特征不代表最终方案）
R10 staging 发布 ──▶ production 提交
R22 / R23 ──▶ production 提交
R25 scratch 路径 + 两处去临时 B-tree ──▶ R26 资源预检 ──▶ production 提交
```

**R25 必须先于 R26。** 在 R25 落地前 SQLite 仍建在 output 旁，两个 filesystem 的
需求混在一起；先做 R26 等于要把阈值定两次。R25 去掉两处临时 B-tree 后还会把
scratch 上的 external-sort 需求降到 0（DB 文件本身仍常驻），直接改变 R26 要检查
的数值。

## 9. 集群验收 Gate（§8 实施完成后执行）

**前置条件：§8 的 R22–R26 已实施并通过新增测试，§0 的两个 commit/tag 已形成，
工作区 clean。**

### Gate A — Adapter 发布

1. 形成 R8/Q2 的书面方法学决议；若取消 masking，验证每个 strategy 六个 outcomes
   的 ID + predictor canonical projection hash 完全相同。
2. 使用 staging（§8.5 的 R10 实现）重新生成全部 Adapter artifacts，验证
   source/config/code 和所有输出 hashes，再原子发布。
3. 在提交节点对 SMR/FFCWS 全部声明 models 运行一次 Adapter/preflight 验证（R14）。

### Gate B — Full-range timing run

4. 使用明确的 timing manifest/snapshot（R13），运行
   `1 seed × 1 draw × 20 N × 20 K`、`max_n=0`、`max_k=0`、`batch_size=20`
   的 full-range grid。必须完整覆盖最大 N/K。
   记录并确认：resolved unique levels、models、cells、checkpoint writes、
   physical parts、max uncheckpointed cells；按 panel×model 记录 elapsed time、
   MaxRSS、warning/failed 分类、原生崩溃重试成功率、规划阶段峰值 RSS 和
   finalization 时间。
   **完成后必须回填 §4.5.1 的表并重新评估 R17，再批准正式 seeds/draws。**

### Gate C — 故障与恢复

5. 运行一个小 panel，主动终止并 requeue，确认只重算最后不超过一个 checkpoint
   窗口的 cell。
6. 运行最大 K 的 LightGBM cell，记录父/子进程 RSS、耗时和输出。
7. 用测试 wrapper 让 LightGBM 子进程主动退出，确认同一 Slurm job 记录 failed row 后继续下一个 cell。
8. **用测试 wrapper 让子进程 hang，确认超时后子进程被回收、记 `native_timeout`
   failed row 并继续**（R22）。
9. **注入超过门限的失败，确认运行在中途停止而不是跑到最终化**（R23）。

### Gate D — 48G 规模最终化

10. 复制一份大规模 checkpoint shards，单独执行 finalization，**分别记录 output FS
    与 node-local scratch 上的占用**。注意 scratch 上有**两类不同性质的占用，不能
    合并记录**：
    - `MaxRSS`（含 §8.4 单列有界 materialization 的数组，预期约 14.5 MiB/列）
    - **scratch 类别一：SQLite DB 文件峰值。** 这是常驻占用，不会为 0，实测后
      纳入 §5.2 的 scratch Gate
    - **scratch 类别二：SQLite external-sort / temp-file 峰值。** 采用 §8.4 的
      (2) + (2b) 两处改动后**应为 0**；若非 0，说明仍有查询在走外部排序，
      **必须先定位到具体 query plan 才能进入 Gate E**
    - shards 与新 CSV 临时文件峰值（output FS）
    - 总耗时（分列 ingest / 诊断 / CSV 写出）
    - 最终行数、唯一 key 和 status
    - 据此把 §5.2 的两个 Gate（output 3.5×、scratch 2.0×）分别换成实测值

### Gate E — 批准

11. 只有 Gate A–D 全部通过后，才决定正式重复数并替换生产版本。

## 10. R 编号到实施项与 Gate 的映射

| R | 状态 | 实施位置 | 验收 Gate |
|---|---|---|---|
| R1 | 已改 | §3.1 | Gate D |
| R2 | 已改 | §3.3 | Gate C-6/7 |
| R3 | 已改 | §3.2 | Gate B（峰值 RSS） |
| R4 | 已有 | 现有压缩机制 | Gate B |
| R5 | 被 R24 取代 | §8.3 | Gate B |
| R6 | 待证实 | 无代码改动 | 收到 `sacct` 后关闭 |
| R7 | **待实施** | warning 聚合（未排期） | Gate B |
| R8 | **待决议** | 方法学决定 → `contract.py` | Gate A-1 |
| R9 | **待实施** | `ffc.yaml` + 两个调用点 | Gate A-2 |
| R10 | **待实施** | staging/lock/原子发布 + `validate_input.py` | Gate A-2 |
| R11 | 待实施（依赖 R8） | FFCWS 物理布局 | Gate A-1 后 |
| R12 | **待实施** | 三层 policy + per-cell 整数诊断 | Gate A |
| R13 | **待实施** | production/timing manifest | Gate B-4 |
| R14 | **待实施** | 两个 Adapter 默认 validation models | Gate A-3 |
| R15 | 待执行 | §0 拆分提交 | 提交前 |
| R16 | 待材料 | 无代码改动 | 收到 `sacct` 后关闭 |
| R17 | **待校准** | 无代码改动 | Gate B-4 回填 §4.5.1 |
| R18 | 待执行 | timing 配置 | Gate B-4 |
| R20 | 待材料 | checkpoint `error` 提取 | 收到 checkpoint 后关闭 |
| R21 | 待材料 | 只读完整性核对 | 收到 checkpoint 后关闭 |
| **R22** | **已实施，待审核/集群校准** | §8.1 | Gate C-8 |
| **R23** | **待实施** | §8.2 | Gate C-9 |
| **R24** | **待实施** | §8.3 | Gate B-4 |
| **R25** | **待实施** | §8.4 | Gate D-10 |
| **R26** | **待实施** | §8.5 | 启动即生效 |

## 11. 审核文件清单（按变更集分组）

**按 §0 的顺序，这两组必须拆成两个独立 commit / tag。**

### 变更集 (a)：SMR/FFCWS Adapter 重构（作者：本轮 FFCWS/SMR 重构作者）

```
SMR/adapter/prepare.py
SMR/adapter/build_contract.py
SMR/adapter/tests/test_prepare.py
SMR/adapter/README.md
FFCWS/adapter/src/ffcws_data_processor/pipeline.py
FFCWS/adapter/src/ffcws_data_processor/contract.py
FFCWS/adapter/src/ffcws_data_processor/common/validation.py
FFCWS/adapter/tests/test_ffcws_prepare.py
FFCWS/adapter/tests/test_strategies.py
FFCWS/adapter/README.md
FFCWS/adapter/data_processor/**            （整目录删除）
FFCWS/schema/*.json                         （重新生成）
```

相关风险：R8、R9、R10（Adapter 侧）、R11、R14。

### 变更集 (b)：NK Grid Engine 风险修复（本次审核对象）

```
NK_Grid/src/aleatoric_nk_grid/experiment.py
NK_Grid/src/aleatoric_nk_grid/nk_grid.py
NK_Grid/src/aleatoric_nk_grid/native_process.py
NK_Grid/src/aleatoric_nk_grid/run_panels.py
NK_Grid/src/aleatoric_nk_grid/validate_input.py
NK_Grid/tests/test_checkpoint_compaction.py
NK_Grid/tests/test_native_process.py
NK_Grid/tests/test_performance_paths.py
NK_Grid/tests/test_identity_panels_isolation.py
NK_Grid/README.md
```

相关风险：R1、R2、R3、R4、R5、R10（engine 校验侧）、R12（per-cell 诊断）、R22–R26。

### R22 已新增、R23–R26 待新增（进入变更集 (b) 或独立 commit）

```
NK_Grid/src/aleatoric_nk_grid/native_process.py   （timeout runner 重写）
NK_Grid/tests/test_native_process.py               （已加入 timeout/hang 测试）
NK_Grid/tests/test_failure_policy_failfast.py     （新增）
NK_Grid/tests/test_time_based_checkpoint.py       （新增）
NK_Grid/tests/test_scratch_paths.py               （新增）
NK_Grid/tests/test_resource_preflight.py          （新增）
（一次性 `_10` materializer，独立脚本，不进主线 engine）
```

## 12. 可发给 Ava 的英文回复

> Thanks — your diagnosis is consistent with what we found. The old
> `7dda462...` run has two code-level failure modes: the all-at-once pandas
> final merge can OOM after all cells are complete, and a LightGBM native
> segfault can kill the worker before Python records a failed row.
>
> Fixes for both are written but **not yet committed**, and they are under
> review. Finalization now uses an on-disk SQLite reducer and streams the sorted
> CSV, while LightGBM/Super Learner run in an isolated child process with one
> retry. **Please do not start any job from the shared worktree.** Once the
> review is approved we will land these as two separate commits — the
> SMR/FFCWS adapter refactor and the NK Grid engine fixes — and tag them; only
> tagged, clean, committed code may be submitted to the cluster. This matters
> because we already have evidence that the shared source changed mid-array:
> in one earlier array the first tasks were correctly refused for a dirty
> worktree, while later tasks in the same array crashed with
> `SyntaxError: unmatched ')'`.
>
> The review also surfaced further code work that must land before any
> production submission: the isolated subprocess currently has no timeout, so a
> native hang would burn the whole wall-time allocation with no output; the
> failure policy is only evaluated after finalization, so a run could fail on
> its 51st failed cell only after weeks of compute; the SQLite scratch path is
> not explicitly configured; and there is no pre-submission resource check.
> A fixed `batch_size=20` for every model is also the wrong knob — it gives OLS
> ~95,000 fsyncs while leaving Super Learner hours without a checkpoint — so we
> are moving to "checkpoint after N cells or T seconds, whichever comes first."
>
> Two configuration blockers affect submission directly. Both `SMR/panels.yaml`
> and `FFCWS/panels.yaml` are currently `preset: dev` with no `--preset
> production` override in the submit path, so submitting as-is would silently
> produce a dev-sized grid. And both adapters validate generated artifacts with
> `ols` only, even though the panels declare ten models — so adapter success
> today does not prove the other nine can run, and for the FFCWS classification
> panels it does not exercise the Super Learner 5-fold class minimum. Separately,
> the FFCWS adapter currently rewrites test predictors based on outcome
> missingness, which is a methodology question we must settle before FFCWS
> results can be published. That one does not block the SMR timing run.
>
> The fuller logs also show several run-level issues. Sixteen production tasks
> were cancelled at the same timestamp, so we are treating that as one
> scheduler/user cancellation rather than sixteen model crashes. The dev pilot
> completed only 16/20 tasks and stopped at K=100, so it did not test the high-K
> region where both LightGBM tasks crashed.
>
> The supplied folder contains logs and snapshots but no checkpoint parts,
> final CSVs, or manifests. The logs confirm about 11,000,640/38,000,000 cells
> were checkpointed, but no production output is verified complete yet. The
> total-income OLS task wrote all 95,000 batches before finalization OOM, which
> confirms the failure was in finalization and not in OLS fitting. The terminal
> state of `_13` is still unknown until we see `sacct`; the current log alone
> does not prove NODE_FAIL or requeue. The Super Learner checkpoints also
> contain at least three failed rows whose `error` values still need review.
>
> What we most need from you is `sacct` output including the `.batch`/step
> lines, so we can attach real Elapsed and MaxRSS numbers to each array task.
> Right now we only have aggregate progress: in the same window where the OLS
> task finished 100%, the other sixteen averaged about 30%. That is enough to
> say production scale is very likely infeasible as designed, but not enough to
> put a number on the slowest models. Note `scontrol show job` will probably
> return "Invalid job id specified" for these jobs — they are past `MinJobAge` —
> so `sacct` is the primary source.
>
> Before choosing the final number of repetitions, we propose one full-range
> timing run with 1 seed, 1 draw, 20 N levels, 20 K levels, no N/K truncation,
> and batch size 20. That is roughly 380 cells per panel/model. It measures one
> complete N/K sweep and exercises the maximum K without launching the original
> multi-month production design.
>
> Local regression tests pass, including a simulated hard child-process crash
> and a real LightGBM subprocess fit. We still need a 48G cluster-scale
> finalization test before calling it production-validated. Please do not
> delete any existing part files: `_10` should be recoverable by streaming its
> completed shards (that one-off tool still has to be written, so for now just
> keep a read-only backup), `_13` should be held pending its Slurm state and
> checkpoint audit, and the crashed LightGBM tasks should be rerun or backported
> with the isolation fix.

## 13. 提交状态

本次没有执行 `git add`、`git commit` 或 `git push`。

本轮下一步：Wanxiang 审核 R22 与 Elastic Net 参数缩减代码。批准后再按 §0 拆分
commit/tag。R23–R26 仍为待审、待实施项；生产运行前仍需完成各自对应的 Gate。
