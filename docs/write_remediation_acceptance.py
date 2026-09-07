"""Regenerate acceptance ledger from recorded evidence, never run production."""
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'docs/remediation_evidence'
package = types.ModuleType('aleatoric_nk_grid')
package.__path__ = [str(ROOT / 'NK_Grid/src/aleatoric_nk_grid')]
sys.modules['aleatoric_nk_grid'] = package
from aleatoric_nk_grid.execution_contract import runtime_environment

BASE = 'd5df3df55e10bbce593f3fb8d5db7b10c550701d'
HEAD = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
batch = {'A':'934e38a','B':'a782d4d','C':'f5f2a09','D':'c5bbf2e','E':'d8612d7','F':'4502dcd'}
files = {'A':'contracts','B':'boosting','C':'batch','D':'folds','E':'metrics','F':'environment'}
portable = '.venv-remediation/Scripts/python.exe NK_Grid/tests/run_portable_remediation.py -q '
linux = 'PYTHONPATH=NK_Grid/src python -m pytest -q '
gpa = json.loads((EVIDENCE / 'GPA-paired-summary.json').read_text(encoding='utf-8'))
# status is deliberately binary here: partial evidence is NOT acceptance.
rows = [
('A1',True,'三种版本化宇宙、seed=37/K=25','均为3400个抽样单位且选择相同','通过；真实重建的三份宇宙也与版本化定义逐对象相等','M1-canonical-order.txt'),
('A2',True,'派生组倒序及提前source_order','规范父变量顺序与派生原子性不变','通过；独立列清单及父顺序断言','M1-canonical-order.txt'),
('A3',False,'容量/容量+1；10行内部划分test_size=.3','合法边界通过；非法计划在写入前拒绝','helper通过；真实planner测试在Windows跳过，Linux未验收','final-portable-regression.txt'),
('A4',True,'bool/0/负数/小数/NaN/重复/逆序/空','报错且不归一化非法输入','8类非法输入均拒绝；历史审计测试不改写源CSV','final-portable-regression.txt'),
('A5',False,'10行其中2个缺失目标，外部test有2行','有效训练容量为8；N=9拒绝','已补真实session用例；Windows跳过；内部分类容量用例仍需Linux扩充','final-portable-regression.txt'),
('A6',False,'绕过CLI直接执行N=4/容量3和K=5/容量2','最终执行在抽样和fit前拒绝','真实入口Windows跳过；实际方法AST门检查及移除检查mutant通过，仍非完整session验收','final-portable-regression.txt'),
('A7',False,'合法旧设计及原抽样/模型seed','离散设计不变，旧方法预测在固定门槛内','直接git基线比较抽样/模型seed通过；B1有数值证据，但全引擎旧新合法设计端到端未验收','M1-final-boundaries.txt'),
('B1',True,'dense/wide/NaN/constant/min-N；原生lgb.cv/xgb.cv','曲线预测rtol=1e-7/atol=1e-9；轮数严格一致','通过；M2提交先于M4验证；补独立参数字面量及真实git基线NPZ对照','M2-portable.txt'),
('B2',True,'首次/末次最优、并列和early-stop边界曲线','首个统一最优和原聚合早停规则','通过；平均折最优的受控错误被捕获','mutations.json'),
('B3',True,'N=23、5折不同误差','保留原索引和平均RMSE，不替换为pooled-MSE','通过；LightGBM原生余数行CV规则保留，最终训练全N','final-portable-regression.txt'),
('B4',False,'真实GPA 201×16085 float64；weakref生命周期','最多1个折上下文且测目标进程树/阶段峰值','生命周期通过；Windows原生process峰值已测；目标Linux进程树/完整任务驻留未验收','native-lightgbm-m2.json'),
('B5',False,'第3折Python异常；原生kill/timeout','不基于部分曲线成功；既有失败路径完整','XGBoost第3折注入通过；原生kill/timeout至WAL失败路径未验收','final-portable-regression.txt'),
('B6',True,'FFC缺失指示K=3400；K=25最宽及40个随机子集','全宽16085；最大K单位宽度覆盖任意子集','通过；独立集合枚举及真实NPZ均为16085列','M2-width.txt'),
('B7',False,'原集群OOM单元及同CPU/内存限制','候选完成且记录旧新退出/峰值/时长','缺原OOM配置、日志和集群访问；未验收；不宣称OOM已解决','native-input.txt'),
('C1',True,'N199/200/201/250/251/252及实际折fit大小','auto边界正确，full每fit一批','通过；与独立sklearn MLP实际预测比较，未仅相信诊断标签','M3-wide-diagnostics.txt'),
('C2',True,'clone/pickle及折内/最终fit','配置和诊断不丢失','通过；clone丢策略mutant被捕获','mutations.json'),
('C3',True,'auto、显式200、旧sklearn MLP','auto与200机制及数值兼容','通过；未改变引擎checkpoint batch_size；默认仍auto','M3-wide-diagnostics.txt'),
('C4',False,'外部输入极端值及训练池内独立holdout','测试目标/文件不影响fit与策略选择','predict前后完整模型pickle不变；driver不使用外部test值；尚无全引擎替换测试文件验收','M3-wide-diagnostics.txt'),
('C5',True,'零目标/强信号N10 K30；边界batch用例','有限输出、真实fit诊断；不要求分数提高','通过；记录各折N、batch、未收敛；没有正分或单调性断言','M3-wide-diagnostics.txt'),
('C6',gpa['complete'] and not gpa['failed_cells'],'GPA训练池holdout731；9个N×K10/100×2seed×2策略×2方法','M4后完整重复成对诊断，保留全部seed和失败','完整度见GPA-paired-summary.json；默认auto不变；不据此宣称历史下降已修复','GPA-paired-summary.json'),
('D1',True,'内折validation X极值；实际LGBM树及MLP fit','训练统计和该折模型参数不变','通过；实际第一折模型与训练输入严格不变','M4-final-oracles.txt'),
('D2',True,'只影响该折validation侧y或与X共同极值','该折基模型及目标缩放不变','通过；MLP训练目标直接对照独立标准化公式；不错误要求最终调参不变','M4-final-oracles.txt'),
('D3',False,'外部变换极值/全模型prediction变换','全部训练统计/alpha/权重不受外部X/y影响','transform及SL模型状态不变已测；全引擎所有模型外部文件变更未验收','M3-wide-diagnostics.txt'),
('D4',True,'全缺失源、未出现类别、ordinal、passthrough','旧类型/先验/NaN契约，列不漂移','通过；独立旧reference与折内transform相同','M4-final-oracles.txt'),
('D5',True,'全N与折内中位数不同的validation极值','真正折内填补可区别伪修复','通过；全N填补mutant被独立LOO oracle捕获','mutations.json'),
('D6',True,'显式逐alpha sklearn Ridge完整管线LOO','所有OOF预测/损失与选择一致','通过；rtol=1e-10/atol=1e-12，单SVD生产路径未减少折数','M4-final-oracles.txt'),
('D7',True,'无缺失且量纲/目标极端；MLP目标缩放','能检测X和y缩放泄漏','通过；独立无缺失LOO和实际MLP fit；全N缩放mutant被捕获','mutations.json'),
('E1',True,'test=[0,2],train_mean=2,pred=[2,2]','MSE2/train-skill0/test-R2负1','通过；独立直接求和oracle','M5-portable.txt'),
('E2',True,'完美/训练均值/测试均值/很差预测','公式一致且不截断负分','通过；交换分母和负分截断mutant均被捕获','mutations.json'),
('E3',True,'常数目标、单样本、0分母、NaN/Inf预测','未定义记NaN；非有限拒绝','通过；未把零分母转换为高分','M5-portable.txt'),
('E4',True,'目标/预测同时平移或非零缩放','相对分数不变，MSE按平方缩放','通过；独立求和oracle','M5-portable.txt'),
('E5',True,'仅改变训练目标均值','MSE/test-R2不变，train-skill改变','通过；旧列定义保留','M5-portable.txt'),
('E6',False,'新契约JSON往返；嵌套serializer1/2/999/None','合法新契约可读；旧新混合拒绝','发现并修复顶层读取错误；新往返通过；完整CSV/WAL混合发布仍需Linux','E6-roundtrip-after-fix.txt'),
('F1',False,'同snapshot恢复，记录写任务表函数调用','字节不变、无重新plan枚举','代码事实：已有submitter复用snapshot；未做Linux入口调用跟踪','full-suite-windows.txt'),
('F2',False,'数据字节变化但size/mtime固定；参数/源顺序/seed变化','内容/语义漂移拒绝','内容哈希/不可变locator反例及跳过校验mutant通过；完整恢复入口所有变体未验收','E6-roundtrip-after-fix.txt'),
('F3',True,'requirements不变，声明的实际numpy版本不符','运行环境不匹配明确拒绝','契约层通过；线程声明静态回归和全部sbatch语法检查通过','E6-roundtrip-after-fix.txt'),
('F4',False,'并发恢复相同输出','schedule lease阻止重复发布','POSIX锁不可在当前Windows验收；既有用例保留','full-suite-windows.txt'),
('F5',False,'任务表/激活/WAL/发布边界中断','原子恢复或明确拒绝；成功记录不丢','计时故障透传通过；真实协议中断矩阵及POSIX用例未运行','full-suite-windows.txt'),
('F6',False,'删成功行再复制另一行补总数','按完整任务键拒绝，不只看行数','真实SQLite归约器捕获重复补行，count-only mutant被捕获；完整WAL/发布入口未验收','M6-content-and-keys.txt'),
('F7',False,'任务重排及未授权worker数变更','设计数值不变；非法执行契约变更拒绝','执行契约/分配规则未删除；目标端到端未验收','full-suite-windows.txt'),
('F8',False,'相同硬件输入FS的冷启动/恢复','逐阶段耗时；恢复不重建plan','计时已实现；无目标环境配对耗时，不宣称准备提速','full-suite-windows.txt'),
]
records=[]
for identifier, passed, inputs, expected_result, actual, evidence in rows:
    group = identifier[0]
    target = f'NK_Grid/tests/test_remediation_{files[group]}.py'
    if identifier == 'B6': target = 'NK_Grid/tests/test_remediation_width.py'
    if identifier == 'E6': target = 'NK_Grid/tests/test_remediation_environment.py'
    commands = [portable+target]
    if identifier == 'C6':
        commands = ['.venv-remediation/Scripts/python.exe NK_Grid/tests/run_gpa_diagnostic.py --schema FFCWS/data/remediation_20260907/schema/ffc_median_mode_gpa.json --method legacy --output FFCWS/data/remediation_20260907/gpa-legacy-paired.jsonl'] + [
            f'.venv-remediation/Scripts/python.exe NK_Grid/tests/run_gpa_diagnostic.py --schema FFCWS/data/remediation_20260907/schema/ffc_median_mode_gpa.json --seeds {seed} --output FFCWS/data/remediation_20260907/gpa-fold-local-seed{seed}.jsonl' for seed in (12345,23456)]
    remaining = linux+target
    if identifier in ('F1','F4','F5','F6','F7','F8'):
        remaining = linux+'NK_Grid/tests/test_remediation_posix.py NK_Grid/tests/test_worker_event_wal.py NK_Grid/tests/test_generation_control.py'
    if identifier == 'A7':
        remaining = linux+'NK_Grid/tests/test_cell_centric_execution.py NK_Grid/tests/test_seed_shards.py'
    if identifier in ('C4','D3','E6'):
        remaining = linux+'NK_Grid/tests/test_nk_grid_engine.py NK_Grid/tests/test_worker_event_wal.py'
    records.append({'id':identifier, 'baseline_commit':BASE, 'candidate_commit':subprocess.check_output(['git','rev-parse',batch[group]],cwd=ROOT,text=True).strip(),
        'inputs':inputs, 'environment':'environment.json; source hashes in evidence-index.json and GPA identity records',
        'commands':commands, 'expected':expected_result, 'actual':actual,
        'status':'通过' if passed else '未验收', 'evidence':[evidence,'final-portable-regression.txt'],
        'remaining_command':None if passed else remaining,
        'remaining_scope':'见报告的 Linux/Slurm 补验命令；单个 pytest 文件不自动满足真实负载或协议矩阵'} )
environment={'repository':str(ROOT),'branch':subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
    'audit_head':HEAD,'baseline_commit':BASE,'runtime':runtime_environment(),'platform':platform.platform(),
    'python_executable':sys.executable,
    'worktree_status':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True).splitlines(),
    'limitations':['Windows portable package namespace bypass, no faked POSIX resource or locks',
                   '24 pre-existing default slow deselections are not acceptance',
                   'native benchmark is process high-water RSS, not Slurm cgroup or whole worker tree']}
(EVIDENCE/'environment.json').write_text(json.dumps(environment,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
(EVIDENCE/'acceptance.json').write_text(json.dumps(records,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
table=['| 项目 | 状态 | 输入 / 预期 | 实际及限制 | 证据 |','|---|---|---|---|---|']
for record in records:
    table.append(f"| {record['id']} | {record['status']} | {record['inputs']}；{record['expected']} | {record['actual']} | [{record['evidence'][0]}](remediation_evidence/{record['evidence'][0]}) |")
(ROOT/'docs/remediation_evidence/acceptance-table.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
live_logs = {'GPA-fold-local-seed12345.txt','GPA-fold-local-seed23456.txt'} if not gpa['complete'] else set()
index={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in EVIDENCE.iterdir()
       if p.is_file() and p.name!='evidence-index.json' and p.name not in live_logs}
(EVIDENCE/'evidence-index.json').write_text(json.dumps(index,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'items':len(records),'passed':sum(r['status']=='通过' for r in records),'not_accepted':sum(r['status']=='未验收' for r in records),'gpa_complete':gpa['complete']}))
