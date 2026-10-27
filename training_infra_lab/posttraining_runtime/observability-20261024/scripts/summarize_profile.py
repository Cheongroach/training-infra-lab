"""Derive summaries from actual SQLite intervals and NCU raw CSV."""
from collections import defaultdict
import csv
import io
import json
from pathlib import Path
import sqlite3

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence'
RES=OUT/'artifacts/results'


def union(intervals):
    merged=[]
    for begin,end in sorted(intervals):
        if merged and begin<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],end)
        else:merged.append([begin,end])
    return merged


def category(name):
    low=name.lower()
    if 'nccl' in low:return 'NCCL'
    if 'attention' in low or 'flash' in low:return 'Attention'
    if any(x in low for x in ('gemm','xmma','nvjet_')):return 'GEMM'
    return '其他'


def nsight():
    databases=list((RES/'nsys').glob('*.sqlite'))
    if not databases:return {'available':False,'reason':'没有归档到 NSYS SQLite；请查看工具日志和 runtime-env 诊断。'}
    assert len(databases)==1
    path=databases[0]
    conn=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)
    tables={x[0] for x in conn.execute("select name from sqlite_master where type='table'")}
    (OUT/'nsys-schema.json').write_text(json.dumps({t:[x[1] for x in conn.execute('pragma table_info("'+t+'")')] for t in sorted(tables)},indent=2))
    strings=dict(conn.execute('select id,value from StringIds'))
    kcols={x[1] for x in conn.execute('pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)')}
    namecol='demangledName' if 'demangledName' in kcols else 'shortName'
    rows=list(conn.execute(f'select start,end,{namecol},deviceId,streamId from CUPTI_ACTIVITY_KIND_KERNEL'))
    if not rows:return {'available':False,'reason':'NSYS 导出存在，但没有 GPU kernel 记录。'}
    assert len({x[3] for x in rows})==1,'single learner capture must contain one device'
    ranges={k:[] for k in ('GEMM','Attention','NCCL','其他')}
    sums=defaultdict(lambda:[0,0])
    for start,end,nid,device,stream in rows:
        assert end>=start
        name=strings.get(nid,str(nid));low=name.lower()
        group=category(name)
        sums[name][0]+=end-start;sums[name][1]+=1;ranges[group].append((start,end))
    a=min(x[0] for x in rows);b=max(x[1] for x in rows)
    nvtx=[]
    if 'NVTX_EVENTS' in tables:
        cols={x[1] for x in conn.execute('pragma table_info(NVTX_EVENTS)')}
        text='text' if 'text' in cols else 'NULL'
        textid='textId' if 'textId' in cols else 'NULL'
        for st,en,txt,tid in conn.execute(f'select start,end,{text},{textid} from NVTX_EVENTS where end is not null'):
            label=txt or strings.get(tid,'')
            if 'grpo/actor_train/rollout=1/rank=0' in label:
                nvtx.append({'name':label,'start_ns':st,'end_ns':en,'duration_s':(en-st)/1e9})
                a=min(a,st);b=max(b,en)
    assert len(nvtx)==1,'exactly one selected learner NVTX range is required'
    assert all(a<=x[0]<=x[1]<=b for x in rows)
    phase_rows=[json.loads(x) for x in (RES/'nsys/phases-rank0.jsonl').read_text().splitlines()]
    phase=[x for x in phase_rows if x['rollout_id']==1 and x['profiled'] and x['rank']==0]
    assert len(phase)==1
    host_s=(phase[0]['monotonic_end_ns']-phase[0]['monotonic_start_ns'])/1e9
    assert host_s>=nvtx[0]['duration_s']
    scope={'host_scope_s':host_s,'nvtx_s':nvtx[0]['duration_s'],
           'outside_nvtx_s':host_s-nvtx[0]['duration_s'],
           'definition':'host scope includes CUDA Profiler API start/stop and bookkeeping; NVTX excludes these',
           'phases':phase_rows}
    copies=list(conn.execute('select start,end,bytes,copyKind,streamId,correlationId from CUPTI_ACTIVITY_KIND_MEMCPY'))
    memsets=list(conn.execute('select start,end,bytes,streamId from CUPTI_ACTIVITY_KIND_MEMSET'))
    copy_labels={x[0]:x[1] for x in conn.execute('select id,label from ENUM_CUDA_MEMCPY_OPER')}
    copy_groups={}
    for st,en,nbytes,kind,stream,corr in copies:
        group=copy_groups.setdefault(kind,{'direction':copy_labels[kind],'count':0,'bytes':0,'sum_ns':0})
        group['count']+=1;group['bytes']+=nbytes;group['sum_ns']+=en-st
    copy_summary=[{'direction':v['direction'],'count':v['count'],'bytes':v['bytes'],
                   'gib':v['bytes']/1024**3,'sum_s':v['sum_ns']/1e9} for v in copy_groups.values()]
    all_intervals=union([(x[0],x[1]) for x in rows+copies+memsets])
    gaps=[];cursor=a
    for st,en in all_intervals:
        st=max(st,a);en=min(en,b)
        if en<=a or st>=b:continue
        if st>cursor:gaps.append({'start_s':(cursor-a)/1e9,'end_s':(st-a)/1e9,'duration_s':(st-cursor)/1e9})
        cursor=max(cursor,en)
    if cursor<b:gaps.append({'start_s':(cursor-a)/1e9,'end_s':(b-a)/1e9,'duration_s':(b-cursor)/1e9})
    bins=[];bin_width=(b-a)/200
    merged={k:union(v) for k,v in ranges.items()}
    merged['Memcpy']=union([(x[0],x[1]) for x in copies])
    for i in range(200):
        lo=a+i*bin_width;hi=lo+bin_width
        activity={k:sum(max(0,min(en,hi)-max(st,lo)) for st,en in ints)/bin_width*100 for k,ints in merged.items()}
        bins.append({'t':((lo+hi)/2-a)/1e9,'activity':activity})
    api_sums=defaultdict(lambda:[0,0]);api_count=0;api_counts={}
    for table,kind in [('CUPTI_ACTIVITY_KIND_RUNTIME','runtime'),('CUPTI_ACTIVITY_KIND_DRIVER','driver')]:
        if table not in tables:continue
        count=0
        for st,en,nid in conn.execute('select start,end,nameId from '+table):
            name=strings.get(nid,str(nid))
            api_sums[(kind,name)][0]+=en-st;api_sums[(kind,name)][1]+=1;count+=1
        api_counts[kind]=count;api_count+=count
    kernel_top=[{'name':n,'total_ms':v[0]/1e6,'count':v[1]} for n,v in sorted(sums.items(),key=lambda x:-x[1][0])]
    api_top=[{'name':n[1],'kind':n[0],'total_ms':v[0]/1e6,'count':v[1]} for n,v in sorted(api_sums.items(),key=lambda x:-x[1][0])[:12]]
    intervals=union([(x[0],x[1]) for x in rows])
    capture={'available':True,'window_s':(b-a)/1e9,'kernel_count':len(rows),
             'window_definition':'bounding interval of selected NVTX and captured kernels',
             'api_count':api_count,'gpu_union_s':sum(en-st for st,en in intervals)/1e9,
             'api_counts':api_counts,'scope':scope,'copy_summary':copy_summary,
             'device_activity_union_s':sum(max(0,min(en,b)-max(st,a)) for st,en in all_intervals)/1e9,
             'device_activity_definition':'union of captured kernel, memcpy and memset intervals on learner device',
             'largest_activity_gaps':sorted(gaps,key=lambda x:-x['duration_s'])[:10],
             'kernel_sum_ms':sum(v[0] for v in sums.values())/1e6,
             'kernel_top':kernel_top,'api_top':api_top,'bins':bins,'nvtx':nvtx,
             'sqlite_path':str(path.relative_to(ROOT)),
             'report_path':str(path.with_suffix('.nsys-rep').relative_to(ROOT)),
             'device_id':rows[0][3], 'kernel_classification':'heuristic based on actual kernel names',
             'group_summary':[{'name':k,'count':len(v),'sum_ms':sum(en-st for st,en in v)/1e6,
                              'union_ms':sum(en-st for st,en in merged[k])/1e6} for k,v in ranges.items()]}
    # A short, actual interval excerpt with CPU API and GPU stream lanes.
    gemms=sorted([x for x in rows if category(strings.get(x[2],''))=='GEMM'])
    pivot=(gemms[len(gemms)//2] if gemms else sorted(rows)[len(rows)//2])[0]
    lo=max(a,pivot-50_000_000);hi=lo+150_000_000
    excerpt=[]
    for st,en,nid,dev,stream in rows:
        if st<hi and en>lo:
            name=strings.get(nid,str(nid))
            excerpt.append({'lane':f'GPU {dev} / stream {stream}','start_ms':(st-lo)/1e6,
                            'end_ms':(en-lo)/1e6,'name':name,'kind':'NCCL' if 'nccl' in name.lower() else 'GPU'})
    for st,en,nbytes,kind,stream,corr in copies:
        if st<hi and en>lo:
            excerpt.append({'lane':f'GPU memcpy / stream {stream}','start_ms':(st-lo)/1e6,
                            'end_ms':(en-lo)/1e6,'name':f'{copy_labels[kind]}: {nbytes} bytes','kind':'Memcpy'})
    if 'CUPTI_ACTIVITY_KIND_RUNTIME' in tables:
        rcols={x[1] for x in conn.execute('pragma table_info(CUPTI_ACTIVITY_KIND_RUNTIME)')}
        if 'globalTid' in rcols:
            for st,en,nid,tid in conn.execute('select start,end,nameId,globalTid from CUPTI_ACTIVITY_KIND_RUNTIME where start < ? and end > ?', (hi,lo)):
                excerpt.append({'lane':f'CPU thread {tid & 0xffffff}','start_ms':(st-lo)/1e6,
                                'end_ms':(en-lo)/1e6,'name':strings.get(nid,str(nid)),'kind':'CPU'})
    capture['excerpt']={'offset_s':(lo-a)/1e9,'width_ms':150,'events':excerpt,
                        'description':'actual 150 ms excerpt centered near a middle GEMM launch'}
    conn.close()
    return capture


def compute():
    path=RES/'ncu-raw.log'
    log=(RES/'ncu-capture.log').read_text(errors='replace') if (RES/'ncu-capture.log').exists() else ''
    if not path.exists():
        reason=('本次筛选 regex:.*gemm.* 未命中实际 nvjet_sm90 GEMM；NCU 返回 0，但原始日志明确记录 No kernels were profiled。'
                if 'No kernels were profiled' in log else '没有 NCU raw CSV 归档。')
        return {'available':False,'reason':reason,'raw_log_path':'evidence/artifacts/results/ncu-capture.log',
                'actual_kernel_name':'nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT',
                'next_kernel_filter':'regex:^nvjet_sm90_.*','no_retry':True,
                'scope':'isolated BF16 projection ran successfully, but no GPU counter measurement'}
    text=path.read_text(errors='replace');lines=text.splitlines()
    start=next((i for i,l in enumerate(lines) if '"Metric Name"' in l and '"ID"' in l),None)
    if start is None:return {'available':False,'reason':'NCU CSV 没有包含实际 kernel 指标行。'+log[-600:]}
    reader=csv.DictReader(io.StringIO('\n'.join(lines[start:])))
    samples={};units={}
    for row in reader:
        if not row.get('Metric Name'):continue
        sid=row['ID'];sample=samples.setdefault(sid,{'id':sid,'name':row['Kernel Name'],'metrics':{}})
        sample['metrics'][row['Metric Name']]=row['Metric Value'];units[row['Metric Name']]=row['Metric Unit']
    names=('gpu__time_duration.sum','sm__throughput.avg.pct_of_peak_sustained_elapsed',
           'dram__throughput.avg.pct_of_peak_sustained_elapsed','sm__warps_active.avg.pct_of_peak_sustained_active',
           'launch__registers_per_thread','launch__block_size','launch__grid_size',
           'launch__occupancy_limit_registers','launch__occupancy_limit_shared_mem',
           'launch__theoretical_occupancy')
    present=[n for n in names if n in units]
    if not samples or 'gpu__time_duration.sum' not in present:return {'available':False,'reason':'NCU 未采到算子执行时长；不能声明 GPU 计数器采集成功。'+log[-600:]}
    return {'available':True,'samples':list(samples.values()),'display_metrics':[{'name':n,'unit':units[n]} for n in present],
            'report_path':'evidence/artifacts/results/ncu/projection.ncu-rep',
            'scope':'isolated BF16 projection microbenchmark; not an actual training kernel'}


def main():
    summary={'nsys':nsight(),'ncu':compute(),'cost':json.loads((OUT/'cost.json').read_text()),'audit':True,
             'tools':json.loads((RES/'tools.json').read_text()),
             'verified':json.loads((OUT/'verified.json').read_text())}
    for tool in ('nsys','ncu'):
        summary['tools']['tools'][tool]['version_text']=(RES/f'{tool}-version.log').read_text().strip()
    (OUT/'profile-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps({k:{key:value for key,value in summary[k].items() if key not in ('kernel_top','api_top','bins','samples','display_metrics')}
                      for k in ('nsys','ncu')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
