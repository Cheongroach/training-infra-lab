import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence'
CLI='/home/tiger/.merlin-cli/bin/merlin-cli'
SID=json.loads((OUT/'submission/create-response.json').read_text())['sid']


def call(args,body,name):
    with (OUT/(name+'.json')).open('w') as out,(OUT/(name+'.stderr')).open('w') as err:
        p=subprocess.run([CLI,'--control-plane','cn','job-v2',*args,'--json',json.dumps(body)],
                         stdout=out,stderr=err,timeout=65)
    assert p.returncode==0, (OUT/(name+'.stderr')).read_text()[-1000:]
    return json.loads((OUT/(name+'.json')).read_text())


run=call(['runs','get'],{'sid':SID},'run-latest')['data']
trial=str(run.get('runtime_context',{}).get('trial_id',''))
print(json.dumps({'sid':SID,'status':run.get('status'),'trial':trial},ensure_ascii=False),flush=True)
if trial:
    pods=call(['run-trial-pods','list'],{'filter':{'trial_sid_any':[trial]}},'pods-latest')['data']
    for pod in pods:
        print(json.dumps({k:pod.get(k) for k in ('sid','status','exit_status','start_at','stop_at')},ensure_ascii=False),flush=True)
        log=call(['run-trial-local-logs','get'],{'trial_pod_sid':str(pod['sid']),'stream':'stdout'},'log-source')
        url=log.get('data',{}).get('url')
        if url:
            data=urllib.request.urlopen(url,timeout=35).read()
            (OUT/'pod-stdout.log').write_bytes(data)
            lines=data.decode(errors='replace').splitlines()
            # Skip launcher echo, which embeds source strings and is not runtime evidence.
            starts=[i for i,l in enumerate(lines) if l.strip()=='GRPO_ENTRY_REACHED']
            if starts:
                actual=lines[starts[-1]:]
                print('\n'.join(actual[-18:])[-6500:],flush=True)
            else:
                print('entry marker not reached; stdout bytes',len(data),flush=True)
    if run.get('status') in ('done','failed','success','stopped','canceled'):
        call(['run-trials','get'],{'sid':trial},'trial-terminal')
        call(['run-timelines','list'],{'filter':{'job_run_sid':SID},'page_size':100},'timeline-terminal')
