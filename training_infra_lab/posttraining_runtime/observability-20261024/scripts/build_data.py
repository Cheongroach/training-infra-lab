"""Normalize actual metrics, retaining query expression and aggregation provenance."""
import csv
import datetime
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'evidence/grafana-history'
BASE=ROOT.parent/'grpo-ray-ready-ba214ae231ee'


def walk(v):
    if isinstance(v,dict):
        if 'dps' in v: yield v
        else:
            for x in v.values(): yield from walk(x)
    elif isinstance(v,list):
        for x in v:yield from walk(x)


def main():
    series=[]
    for file in sorted(RAW.glob('panel-*.json')):
        panel=int(file.stem.split('-')[1])
        for entry in json.loads(file.read_text())['total']:
            if not entry.get('payload'):continue
            query=json.loads(entry['payload'])
            for item in walk(entry.get('data')):
                assert item['tags'].get('pod_name',item['tags'].get('_pod_name'))=='trial-438559238-trialrun-438559238-worker-0'
                q=query['queries'][item.get('query',{}).get('index',0)]
                points=sorted([[int(t)*1000,v] for t,v in item['dps'].items()
                               if isinstance(v,(int,float)) and math.isfinite(v)])
                values=[v for _,v in points]
                name=item['metric'].rsplit('.',1)[-1]
                expr=q.get('multiFieldExpr','')
                if expr:name += ':'+expr
                series.append({'name':name,'metric':item['metric'],'field':expr,
                    'panel':panel,'aggregator':q.get('aggregator'),'tags':item['tags'],
                    'source_step_s':statistics.median([points[i+1][0]-points[i][0] for i in range(len(points)-1)])/1000,
                    'points':points,'summary':{'min':min(values),'max':max(values),'mean':statistics.mean(values)},
                    'source':f'evidence/grafana-history/{file.name}'})
    perf=[];train=[]
    events=BASE/'artifacts/results/events'
    for line in (events/'actor_cell00000_rank00000.jsonl').read_text().splitlines():
        x=json.loads(line)
        if x['type']=='metric':
            m=x['metrics']
            if 'perf/step_time' in m:perf.append({'rollout_id':x['rollout_id'],'timestamp':x['timestamp'],'metrics':m})
            if 'train/grad_norm' in m:train.append({'rollout_id':x['rollout_id'],'timestamp':x['timestamp'],'metrics':m})
    rollout=[]
    for line in (events/'rollout_executor.jsonl').read_text().splitlines():
        x=json.loads(line)
        if x['type']=='metric':rollout.append({'rollout_id':x.get('rollout_id'),'timestamp':x['timestamp'],'metrics':x['metrics']})
    verified=json.loads((BASE/'verified.json').read_text())
    data={'history':{'run':'70d36e11e929c5c3','trial':'438559238','series':series,'perf':perf,
            'train':train,'rollout':rollout,'verified':verified,'cost':json.loads((BASE/'cost.json').read_text())},
          'diagnostic':{'run':'49b6f8781696e68e','trial':'438689893'}}
    data['history']['grafana_url']=json.loads((RAW/'grafana-url.json').read_text())['grafana_url']
    current=ROOT/'evidence/run-latest.json'
    if current.exists():
        v=json.loads(current.read_text())['data']
        data['diagnostic']['status']=v['status']
    summary=ROOT/'evidence/profile-summary.json'
    if summary.exists():data['diagnostic'].update(json.loads(summary.read_text()))
    diagnostic_events=ROOT/'evidence/artifacts/results/events/actor_cell00000_rank00000.jsonl'
    if diagnostic_events.exists():
        diagnostic_perf=[]
        for line in diagnostic_events.read_text().splitlines():
            event=json.loads(line)
            if event['type']=='metric' and 'perf/step_time' in event['metrics']:
                diagnostic_perf.append({'rollout_id':event['rollout_id'],'timestamp':event['timestamp'],'metrics':event['metrics']})
        data['diagnostic']['perf']=diagnostic_perf
        (ROOT/'evidence/diagnostic-phase-events.json').write_text(json.dumps(diagnostic_perf,indent=2))
    upload=ROOT/'evidence/profiling-upload-short.json'
    if upload.exists() and upload.stat().st_size:
        data['diagnostic']['profiling_assets']=json.loads(upload.read_text())
    gpu=ROOT/'evidence/artifacts/results/gpu-1s.csv'
    if gpu.exists():
        cards={}
        with gpu.open() as f:
            for row in csv.DictReader(f,skipinitialspace=True):
                idx=row['index'];card=cards.setdefault(idx,{'name':row['name'],'uuid':row['uuid'],'metrics':{}})
                dt=datetime.datetime.strptime(row['timestamp'],'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=datetime.timezone.utc)
                for field in ('utilization.gpu [%]','utilization.memory [%]','memory.used [MiB]','power.draw [W]',
                              'temperature.gpu','clocks.current.sm [MHz]','clocks.current.memory [MHz]'):
                    if field in row:
                        try:value=float(row[field])
                        except ValueError:continue
                        card['metrics'].setdefault(field,[]).append([int(dt.timestamp()*1000),value])
        data['diagnostic']['cards']=cards
    (ROOT/'assets/data.js').write_text('window.OBS_DATA='+json.dumps(data,ensure_ascii=False,allow_nan=False)+';\n')
    (ROOT/'evidence/normalized-history.json').write_text(json.dumps(data['history'],ensure_ascii=False,indent=2))
    print('historical series',len(series),'points',sum(len(s['points']) for s in series))
    print([(s['name'],s['summary']) for s in series])


if __name__=='__main__':main()
