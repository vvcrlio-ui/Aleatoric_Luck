| 项目 | 状态 | 输入 / 预期 | 实际及限制 | 证据 |
|---|---|---|---|---|
| A1 | 通过 | 三种版本化宇宙、seed=37/K=25；均为3400个抽样单位且选择相同 | 通过；真实重建的三份宇宙也与版本化定义逐对象相等 | [M1-canonical-order.txt](remediation_evidence/M1-canonical-order.txt) |
| A2 | 通过 | 派生组倒序及提前source_order；规范父变量顺序与派生原子性不变 | 通过；独立列清单及父顺序断言 | [M1-canonical-order.txt](remediation_evidence/M1-canonical-order.txt) |
| A3 | 未验收 | 容量/容量+1；10行内部划分test_size=.3；合法边界通过；非法计划在写入前拒绝 | helper通过；真实planner测试在Windows跳过，Linux未验收 | [final-portable-regression.txt](remediation_evidence/final-portable-regression.txt) |
| A4 | 通过 | bool/0/负数/小数/NaN/重复/逆序/空；报错且不归一化非法输入 | 8类非法输入均拒绝；历史审计测试不改写源CSV | [final-portable-regression.txt](remediation_evidence/final-portable-regression.txt) |
| A5 | 未验收 | 10行其中2个缺失目标，外部test有2行；有效训练容量为8；N=9拒绝 | 已补真实session用例；Windows跳过；内部分类容量用例仍需Linux扩充 | [final-portable-regression.txt](remediation_evidence/final-portable-regression.txt) |
| A6 | 未验收 | 绕过CLI直接执行N=4/容量3和K=5/容量2；最终执行在抽样和fit前拒绝 | 真实入口Windows跳过；实际方法AST门检查及移除检查mutant通过，仍非完整session验收 | [final-portable-regression.txt](remediation_evidence/final-portable-regression.txt) |
| A7 | 未验收 | 合法旧设计及原抽样/模型seed；离散设计不变，旧方法预测在固定门槛内 | 直接git基线比较抽样/模型seed通过；B1有数值证据，但全引擎旧新合法设计端到端未验收 | [M1-final-boundaries.txt](remediation_evidence/M1-final-boundaries.txt) |
| B1 | 通过 | dense/wide/NaN/constant/min-N；原生lgb.cv/xgb.cv；曲线预测rtol=1e-7/atol=1e-9；轮数严格一致 | 通过；M2提交先于M4验证；补独立参数字面量及真实git基线NPZ对照 | [M2-portable.txt](remediation_evidence/M2-portable.txt) |
| B2 | 通过 | 首次/末次最优、并列和early-stop边界曲线；首个统一最优和原聚合早停规则 | 通过；平均折最优的受控错误被捕获 | [mutations.json](remediation_evidence/mutations.json) |
| B3 | 通过 | N=23、5折不同误差；保留原索引和平均RMSE，不替换为pooled-MSE | 通过；LightGBM原生余数行CV规则保留，最终训练全N | [final-portable-regression.txt](remediation_evidence/final-portable-regression.txt) |
| B4 | 未验收 | 真实GPA 201×16085 float64；weakref生命周期；最多1个折上下文且测目标进程树/阶段峰值 | 生命周期通过；Windows原生process峰值已测；目标Linux进程树/完整任务驻留未验收 | [native-lightgbm-m2.json](remediation_evidence/native-lightgbm-m2.json) |
| B5 | 未验收 | 第3折Python异常；原生kill/timeout；不基于部分曲线成功；既有失败路径完整 | XGBoost第3折注入通过；原生kill/timeout至WAL失败路径未验收 | [final-portable-regression.txt](remediation_evidence/final-portable-regression.txt) |
| B6 | 通过 | FFC缺失指示K=3400；K=25最宽及40个随机子集；全宽16085；最大K单位宽度覆盖任意子集 | 通过；独立集合枚举及真实NPZ均为16085列 | [M2-width.txt](remediation_evidence/M2-width.txt) |
| B7 | 未验收 | 原集群OOM单元及同CPU/内存限制；候选完成且记录旧新退出/峰值/时长 | 缺原OOM配置、日志和集群访问；未验收；不宣称OOM已解决 | [native-input.txt](remediation_evidence/native-input.txt) |
| C1 | 通过 | N199/200/201/250/251/252及实际折fit大小；auto边界正确，full每fit一批 | 通过；与独立sklearn MLP实际预测比较，未仅相信诊断标签 | [M3-wide-diagnostics.txt](remediation_evidence/M3-wide-diagnostics.txt) |
| C2 | 通过 | clone/pickle及折内/最终fit；配置和诊断不丢失 | 通过；clone丢策略mutant被捕获 | [mutations.json](remediation_evidence/mutations.json) |
| C3 | 通过 | auto、显式200、旧sklearn MLP；auto与200机制及数值兼容 | 通过；未改变引擎checkpoint batch_size；默认仍auto | [M3-wide-diagnostics.txt](remediation_evidence/M3-wide-diagnostics.txt) |
| C4 | 未验收 | 外部输入极端值及训练池内独立holdout；测试目标/文件不影响fit与策略选择 | predict前后完整模型pickle不变；driver不使用外部test值；尚无全引擎替换测试文件验收 | [M3-wide-diagnostics.txt](remediation_evidence/M3-wide-diagnostics.txt) |
| C5 | 通过 | 零目标/强信号N10 K30；边界batch用例；有限输出、真实fit诊断；不要求分数提高 | 通过；记录各折N、batch、未收敛；没有正分或单调性断言 | [M3-wide-diagnostics.txt](remediation_evidence/M3-wide-diagnostics.txt) |
| C6 | 未验收 | GPA训练池holdout731；9个N×K10/100×2seed×2策略×2方法；M4后完整重复成对诊断，保留全部seed和失败 | 完整度见GPA-paired-summary.json；默认auto不变；不据此宣称历史下降已修复 | [GPA-paired-summary.json](remediation_evidence/GPA-paired-summary.json) |
| D1 | 通过 | 内折validation X极值；实际LGBM树及MLP fit；训练统计和该折模型参数不变 | 通过；实际第一折模型与训练输入严格不变 | [M4-final-oracles.txt](remediation_evidence/M4-final-oracles.txt) |
| D2 | 通过 | 只影响该折validation侧y或与X共同极值；该折基模型及目标缩放不变 | 通过；MLP训练目标直接对照独立标准化公式；不错误要求最终调参不变 | [M4-final-oracles.txt](remediation_evidence/M4-final-oracles.txt) |
| D3 | 未验收 | 外部变换极值/全模型prediction变换；全部训练统计/alpha/权重不受外部X/y影响 | transform及SL模型状态不变已测；全引擎所有模型外部文件变更未验收 | [M3-wide-diagnostics.txt](remediation_evidence/M3-wide-diagnostics.txt) |
| D4 | 通过 | 全缺失源、未出现类别、ordinal、passthrough；旧类型/先验/NaN契约，列不漂移 | 通过；独立旧reference与折内transform相同 | [M4-final-oracles.txt](remediation_evidence/M4-final-oracles.txt) |
| D5 | 通过 | 全N与折内中位数不同的validation极值；真正折内填补可区别伪修复 | 通过；全N填补mutant被独立LOO oracle捕获 | [mutations.json](remediation_evidence/mutations.json) |
| D6 | 通过 | 显式逐alpha sklearn Ridge完整管线LOO；所有OOF预测/损失与选择一致 | 通过；rtol=1e-10/atol=1e-12，单SVD生产路径未减少折数 | [M4-final-oracles.txt](remediation_evidence/M4-final-oracles.txt) |
| D7 | 通过 | 无缺失且量纲/目标极端；MLP目标缩放；能检测X和y缩放泄漏 | 通过；独立无缺失LOO和实际MLP fit；全N缩放mutant被捕获 | [mutations.json](remediation_evidence/mutations.json) |
| E1 | 通过 | test=[0,2],train_mean=2,pred=[2,2]；MSE2/train-skill0/test-R2负1 | 通过；独立直接求和oracle | [M5-portable.txt](remediation_evidence/M5-portable.txt) |
| E2 | 通过 | 完美/训练均值/测试均值/很差预测；公式一致且不截断负分 | 通过；交换分母和负分截断mutant均被捕获 | [mutations.json](remediation_evidence/mutations.json) |
| E3 | 通过 | 常数目标、单样本、0分母、NaN/Inf预测；未定义记NaN；非有限拒绝 | 通过；未把零分母转换为高分 | [M5-portable.txt](remediation_evidence/M5-portable.txt) |
| E4 | 通过 | 目标/预测同时平移或非零缩放；相对分数不变，MSE按平方缩放 | 通过；独立求和oracle | [M5-portable.txt](remediation_evidence/M5-portable.txt) |
| E5 | 通过 | 仅改变训练目标均值；MSE/test-R2不变，train-skill改变 | 通过；旧列定义保留 | [M5-portable.txt](remediation_evidence/M5-portable.txt) |
| E6 | 未验收 | 新契约JSON往返；嵌套serializer1/2/999/None；合法新契约可读；旧新混合拒绝 | 发现并修复顶层读取错误；新往返通过；完整CSV/WAL混合发布仍需Linux | [E6-roundtrip-after-fix.txt](remediation_evidence/E6-roundtrip-after-fix.txt) |
| F1 | 未验收 | 同snapshot恢复，记录写任务表函数调用；字节不变、无重新plan枚举 | 代码事实：已有submitter复用snapshot；未做Linux入口调用跟踪 | [full-suite-windows.txt](remediation_evidence/full-suite-windows.txt) |
| F2 | 未验收 | 数据字节变化但size/mtime固定；参数/源顺序/seed变化；内容/语义漂移拒绝 | 内容哈希/不可变locator反例及跳过校验mutant通过；完整恢复入口所有变体未验收 | [E6-roundtrip-after-fix.txt](remediation_evidence/E6-roundtrip-after-fix.txt) |
| F3 | 通过 | requirements不变，声明的实际numpy版本不符；运行环境不匹配明确拒绝 | 契约层通过；线程声明静态回归和全部sbatch语法检查通过 | [E6-roundtrip-after-fix.txt](remediation_evidence/E6-roundtrip-after-fix.txt) |
| F4 | 未验收 | 并发恢复相同输出；schedule lease阻止重复发布 | POSIX锁不可在当前Windows验收；既有用例保留 | [full-suite-windows.txt](remediation_evidence/full-suite-windows.txt) |
| F5 | 未验收 | 任务表/激活/WAL/发布边界中断；原子恢复或明确拒绝；成功记录不丢 | 计时故障透传通过；真实协议中断矩阵及POSIX用例未运行 | [full-suite-windows.txt](remediation_evidence/full-suite-windows.txt) |
| F6 | 未验收 | 删成功行再复制另一行补总数；按完整任务键拒绝，不只看行数 | 真实SQLite归约器捕获重复补行，count-only mutant被捕获；完整WAL/发布入口未验收 | [M6-content-and-keys.txt](remediation_evidence/M6-content-and-keys.txt) |
| F7 | 未验收 | 任务重排及未授权worker数变更；设计数值不变；非法执行契约变更拒绝 | 执行契约/分配规则未删除；目标端到端未验收 | [full-suite-windows.txt](remediation_evidence/full-suite-windows.txt) |
| F8 | 未验收 | 相同硬件输入FS的冷启动/恢复；逐阶段耗时；恢复不重建plan | 计时已实现；无目标环境配对耗时，不宣称准备提速 | [full-suite-windows.txt](remediation_evidence/full-suite-windows.txt) |
