"""Summarize measured category/concurrency comparisons with their limits."""
from pathlib import Path
import json
import statistics
import os
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT.parent/'four-class-results'


def group_summary(runs,keys):
    groups={}
    for run in runs:groups.setdefault(tuple(run[k] for k in keys),[]).append(run)
    result=[]
    for key,values in groups.items():
        walls=[v['wall_seconds'] for v in values]
        workers=values[0].get('workers',4)
        result.append({**dict(zip(keys,key)),'workers':workers,'mean_seconds':statistics.mean(walls),
            'repeat_seconds':walls,'task_occupancy':sum(sum(v['worker_busy_seconds']) for v in values)/(workers*sum(walls)),
            'mean_cpu_seconds':statistics.mean(v['model_cpu_seconds'] for v in values)})
    return result


def main():
    from aleatoric_nk_grid.shared_queue import atomic_json,file_digest
    class_root=ROOT.parent/'four-class-benchmark-01';scale_root=ROOT.parent/'queue-concurrency-01'
    for root in (class_root,scale_root):
        complete=json.loads((root/'completion.json').read_bytes())
        assert complete['complete'] and complete['all_bitwise_equal'] and complete['model_runs']==288
    classification=json.loads((class_root/'results.json').read_bytes())
    concurrency=json.loads((scale_root/'results.json').read_bytes())
    # Verify full nine-model predictions also agree between the two studies.
    fingerprints={};comparisons=0
    for run in classification+concurrency:
        for packet in run['records']:
            row=packet['row'];key=tuple(row[k] for k in ('seed','draw','N','K','model'))
            if key in fingerprints:assert fingerprints[key]==row['_prediction_sha256']
            else:fingerprints[key]=row['_prediction_sha256']
            comparisons+=1
    assert comparisons==576 and len(fingerprints)==36
    for root in (class_root,scale_root):
        protocol=json.loads((root/'protocol.json').read_bytes())
        for name,sha in protocol['source_sha256'].items():
            assert file_digest(ROOT/'NK_Grid/src/aleatoric_nk_grid'/name)==sha,('source changed while testing',name)
    categories=group_summary(classification,['arm']);scaling=group_summary(concurrency,['workers','policy'])
    category_work={c:0. for c in ('sl','nn','boost','five')}
    baseline_runs=[r for r in classification if r['arm']=='global_single']
    for run in baseline_runs:
        for p in run['records']:category_work[p['category']]+=p['seconds']/len(baseline_runs)
    work_total=sum(category_work.values())
    fractions={key:value/work_total for key,value in category_work.items()}
    OUTPUT.mkdir(exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(OUTPUT/'matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'sl':'#456EDB','nn':'#A963C6','boost':'#EAA734','five':'#35A789'}
    labels={'global_single':'Unified queue','four_shared':'Four shared queues'}
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True,constrained_layout=True)
    for ax,run in zip(axes.flat,[next(r for r in concurrency if r['workers']==n and r['policy']==policy and r['repeat']==1)
        for n in (4,8) for policy in ('global_single','four_shared')]):
        for packet in run['records']:
            ax.barh(packet['worker'],packet['finish']-packet['start'],left=packet['start'],height=.65,
                color=colors[packet['category']],edgecolor='white',linewidth=.5)
        ax.axvline(run['wall_seconds'],color='#555555',linestyle=':',linewidth=1)
        ax.set_title(f"{labels[run['policy']]} / {run['workers']} cores / {run['wall_seconds']:.2f} s")
        ax.set_yticks(range(run['workers']));ax.set_ylabel('Worker');ax.invert_yaxis()
        ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);ax.set_xlabel('Elapsed seconds')
    handles=[plt.Rectangle((0,0),1,1,color=colors[key]) for key in colors]
    fig.legend(handles,['SL','Shallow NN','XGBoost + LightGBM','Other five'],loc='outside upper center',ncol=4)
    fig.suptitle('Same 36 GPA model tasks: measured task timelines (second repeat)',y=1.07)
    fig.savefig(OUTPUT/'task-timelines.png',dpi=160,bbox_inches='tight');plt.close(fig)
    atomic_json(OUTPUT/'summary.json',{'classification':categories,'concurrency':scaling,
        'observed_class_work_seconds':category_work,'observed_class_work_fraction':fractions,
        'model_runs':576,'distinct_model_cells':36,'all_predictions_bitwise_equal':True,
        'prior_seven_model_hash_comparisons':448,'production_modified':False,
        'source_evidence_sha256':{str(path.relative_to(ROOT.parent)):file_digest(path) for path in
            [class_root/'results.json',class_root/'protocol.json',class_root/'completion.json',
             scale_root/'results.json',scale_root/'protocol.json',scale_root/'completion.json']}})
    four={v['arm']:v for v in categories}
    by={(v['workers'],v['policy']):v for v in scaling}
    fixed_gain=1-four['four_shared']['mean_seconds']/four['four_dedicated']['mean_seconds']
    class_gap=four['four_shared']['mean_seconds']/four['global_single']['mean_seconds']-1
    increase={policy:1-by[(8,policy)]['mean_seconds']/by[(4,policy)]['mean_seconds'] for policy in labels}
    rows=[]
    name_map={'global_single':'统一单模型队列','four_dedicated':'四类各固定一个核（对照）',
              'four_shared':'四类动态共享核心','four_shared_cache':'四类共享核心＋输入缓存'}
    for value in categories:
        rows.append(f"| {name_map[value['arm']]} | {value['repeat_seconds'][0]:.2f} | {value['repeat_seconds'][1]:.2f} | {value['mean_seconds']:.2f} | {value['task_occupancy']:.1%} |")
    table='\n'.join(rows)
    rows=[]
    for value in scaling:
        rows.append(f"| {value['workers']} | {name_map[value['policy']]} | {value['repeat_seconds'][0]:.2f} | {value['repeat_seconds'][1]:.2f} | {value['mean_seconds']:.2f} | {value['task_occupancy']:.1%} |")
    scale_table='\n'.join(rows)
    text=f'''# 四类队列与并发数：本地实测

2026-09-10。按用户澄清，以整个实验尽快完成为目标，分类不意味着固定分核。
保持此前要求：每个 `(seed,draw,N,K,model)` 独立领取、执行、持久化和确认。
四类为 SL、Shallow NN、XGBoost＋LightGBM、其余五模型。

## 相同四核资源的分类对照

| 方案 | 第1轮/秒 | 第2轮/秒 | 平均/秒 | 任务时间占比 |
|---|---:|---:|---:|---:|
{table}

四类动态共享相对四类固定分核用时减少 {fixed_gain:.1%}；相对统一队列的平均时间差为
{class_gap:+.2%}（负数代表分类更快）。两次重复不足以把几个百分点的差异认定为稳定收益。
任务时间占比是累计模型任务墙钟时间除以预留核秒，不是 CPU 利用率。

本批统一队列两轮的分类工作量占比为：SL {fractions['sl']:.1%}、Shallow NN {fractions['nn']:.1%}、
Boosting {fractions['boost']:.1%}、其余五模型 {fractions['five']:.1%}。这是本批数据，不能固定成生产分核比例。
实际应根据未完成任务的预计累计耗时动态调整；某类完成后，其核转投其他类别。

## 四核与八核：重新配对测试

| 物理核数 | 方案 | 第1轮/秒 | 第2轮/秒 | 平均/秒 | 任务时间占比 |
|---|---|---:|---:|---:|---:|
{scale_table}

并发从4个物理核增加到8个，统一队列的总用时减少 {increase['global_single']:.1%}，
四类共享队列减少 {increase['four_shared']:.1%}。这比较的是同一批任务的墙钟完成时间，
不是云端账单，也不要求为节省核心牺牲用户关心的完成速度。

![任务时间线](task-timelines.png)

图中每条色块是一个独立模型任务。某个很长的模型任务一旦开始，不能仅靠增加空闲 worker
把它切开，因此只有36项的批次会有尾部等待；不能把此处的核数扩展比例直接用于全量实验。

## 具体策略与可复现性

- 统一队列与分类队列使用相同初始任务成本：模型权重乘 `N*sqrt(K)`。它是启发式，未声称已拟合
  全网格耗时。分类调度选择 `(本类待算＋在算的预计工作量)/(本类在算任务数＋1)` 最大的一类，
  再从该类领取最高成本任务。全部 worker 可跨类领取，未固定类别所属核心。
- 五模型和两个 boosting 仍是独立任务，未被捆成不可分割的大任务。分类路由是本地基准适配器；
  它在小清单上查询成本合计，不能原样用于1,800万任务，生产版需维护增量汇总。
- 一核同时运行一个模型、数值库单线程。后续并发测试覆盖本机8个物理核，未测试超线程上的
  16个并发进程；因此结论限于测试过的配置，不称全局最优。
- 全九模型、GPA真实输入、seed=12345、draw=0，四个格点 `(122,47)`、`(122,400)`、`(907,261)`、
  `(122,261)`。树数量、CV次数、MLP训练上限等均保留完整参数。
- 两个实验各288次正式拟合，合计 **576次**，36个不同模型格点，所有预测在两实验间也逐位一致。
  其中448次七模型结果还核对了此前保存的预测哈希。SVD条件回退版本在所有方案中相同。
- 两实验分别按正序/反序运行各两轮；输入加载、进程启动与热身单独记录。计时包含进程派发、
  Windows进程内数值计算、真实HTTP、结果暂存、日志fsync和确认；不包含Linux原生隔离、
  Lustre、跨节点网络与Slurm。分类测试全部常驻4个进程；并发测试常驻8个，4核方案仅4个执行任务。
- 缓存臂使用每worker 256MiB保留量与节点1GiB磁盘准入；只共享原始输入和外层诊断准备，
  所有CV折内预处理仍按原语义拟合。其他臂关闭跨任务缓存。该缓存不是整个进程内存的硬上限。
- 模型进程实际PID每200ms采样。工作集相加可能重复计共享页，私有提交不是RSS，采样会漏过短峰值。
  并发实验的active内存指标仅计实际执行的worker与父进程，未计其余空闲常驻池。

此次仅新增本地基准与分类路由检查；没有改动已验证的数值源码、当前集群作业或生产调度。
当前GPA若按既定条件等两种boosting完成后再切换，其队列应为空，所有核均可用于剩余三类。
生产验收边界仍见上级目录的 REPORT.md 和源码中的 SCHEDULER_DEVELOPMENT.md。

原始结果：`../four-class-benchmark-01/`、`../queue-concurrency-01/`。汇总 `summary.json`，
运行协议包含源码哈希，代码归档 `benchmark-source.zip`。测试只覆盖一组seed/draw和四个格点，
不将本地分类收益或并发收益外推到所有N/K或完整生产实验。
'''
    (OUTPUT/'REPORT.md').write_text(text,encoding='utf-8')
    with zipfile.ZipFile(OUTPUT/'benchmark-source.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ('gpa_queue_benchmark.py','four_class_queue_benchmark.py','four_class_router_checks.py',
                     'queue_concurrency_benchmark.py','local_cpu_inventory.py','summarize_four_classes.py'):
            archive.write(ROOT/'checks'/name,'checks/'+name)
        for p in (ROOT/'NK_Grid/src/aleatoric_nk_grid').glob('*.py'):
            archive.write(p,p.relative_to(ROOT).as_posix())
        archive.write(ROOT/'FFCWS/model_params.yaml','FFCWS/model_params.yaml')
    print(json.dumps({'classification':categories,'concurrency':scaling,'all_bitwise_equal':True},ensure_ascii=False))


if __name__=='__main__':main()
