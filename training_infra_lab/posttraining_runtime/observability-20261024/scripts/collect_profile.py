"""Collect only a terminal job; verify all five runtime patch changes before PT."""
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence'
BASE=ROOT.parent/'grpo-ray-ready-ba214ae231ee'
UPSTREAM=ROOT.parents[1]/'research/posttraining_20261022/upstream/miles'


def sha(b):return hashlib.sha256(b).hexdigest()


def main():
    run=json.loads((OUT/'run-latest.json').read_text())['data']
    assert run['status'] in ('done','success','failed','stopped','failed_to_launch')
    assert run['sid']=='49b6f8781696e68e'
    payload=json.loads((OUT/'submission/create.checked.json').read_text())
    expected=payload['job_config']['job_template_config']['entrypoint_full_script']
    actual=run['job_config']['job_template_config']['entrypoint_full_script']
    assert expected==actual
    terminal=json.loads((OUT/'trial-terminal.json').read_text())['data']
    assert terminal['launch_config']['EntrypointFullScript']==expected
    target=OUT/'artifacts.tar.gz'
    if not target.exists():
        subprocess.run(['hdfs','dfs','-get',
            'hdfs://haruna/home/byte_search_lifeservice/zhangyang.exp/training_infra_lab/miles-grpo/grpo-profile-932d2a8751/artifacts.tar.gz',
            str(target)],check=True,timeout=180)
    dest=OUT/'artifacts';dest.mkdir(exist_ok=False)
    names=set();size=0
    with tarfile.open(target,'r:gz') as archive:
        for m in archive:
            p=PurePosixPath(m.name)
            assert not p.is_absolute() and '..' not in p.parts and p.parts[0] in ('code','results')
            assert m.name not in names and len(names)<20000
            names.add(m.name)
            path=dest/m.name
            if m.isdir():path.mkdir(parents=True,exist_ok=True);continue
            assert m.isfile()
            size+=m.size;assert size<4*1024**3
            path.parent.mkdir(parents=True,exist_ok=True)
            with archive.extractfile(m) as src,path.open('wb') as dst:shutil.copyfileobj(src,dst)
            assert path.stat().st_size==m.size
    manifest=json.loads((OUT/'submission/source-manifest.json').read_text())
    assert manifest==json.loads((dest/'results/source-manifest.json').read_text())
    for name,digest in manifest.items():assert sha((dest/'code'/name).read_bytes())==digest,name
    patches=[]
    for module_name,report_name,attr in [('patch_broadcast_ipv6','runtime-patch.json','ORIGINAL_SHA256'),
                                     ('patch_publication_probe','publication-probe-patch.json','ORIGINAL_SHA'),
                                     ('patch_ray_launcher','ray-launcher-patch.json','ORIGINAL_SHA')]:
        spec=importlib.util.spec_from_file_location(module_name,dest/'code'/f'{module_name}.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        original=(UPSTREAM/module.RELATIVE_PATH).read_bytes()
        assert sha(original)==getattr(module,attr)
        changed=module.patched_source(original.decode()).encode()
        report=json.loads((dest/'results'/report_name).read_text())
        assert report['base_commit']==module.MILES_SHA and report['path']==module.RELATIVE_PATH
        assert report['before_sha256']==sha(original) and report['after_sha256']==sha(changed)
        patches.append(report)
    specdata=json.loads((dest/'code/profile-patch-spec.json').read_text())
    profile=json.loads((dest/'results/profile-patch.json').read_text())
    assert profile['base_commit']==specdata['base_commit']
    for s,r in zip(specdata['patches'],profile['patches'],strict=True):
        orig=(UPSTREAM/s['path']).read_text()
        assert sha(orig.encode())==s['before_sha256'] and orig.count(s['before'])==1
        changed=orig.replace(s['before'],s['after'],1)
        assert r=={'path':s['path'],'before_sha256':s['before_sha256'],'after_sha256':sha(changed.encode())}
    patches.append(profile)
    entry=(dest/'results/entry.log').read_bytes();stdout=(OUT/'pod-stdout.log').read_bytes()
    assert stdout.count(entry)==1,'entry artifact must be uniquely present in Pod stdout'
    artifact_hashes={str(p.relative_to(dest)):sha(p.read_bytes()) for p in dest.rglob('*') if p.is_file()}
    audit={'safe_archive':True,'files':len(names),'uncompressed_bytes':size,
           'source_hashes_verified':len(manifest),'patch_reports_verified':4,
           'runtime_file_changes_verified':5,'entry_sha256':sha(expected.encode()),
           'entry_log_bytes':len(entry),'entry_log_unique_in_actual_pod_stdout':True,
           'archive_sha256':sha(target.read_bytes()),'files_sha256':artifact_hashes,
           'run':run['sid'],'trial':run['runtime_context']['trial_id']}
    (OUT/'archive-audit.json').write_text(json.dumps(audit,indent=2))
    pods=json.loads((OUT/'pods-latest.json').read_text())['data']
    assert len(pods)==1
    pod=pods[0]
    def dt(v):return datetime.datetime.fromisoformat(v.replace('Z','+00:00'))
    duration=(dt(pod['stop_at'])-dt(pod['start_at'])).total_seconds()
    cost={'residence_s':duration,'gpu_hours':duration*4/3600,'gpus':4,'start_at':pod['start_at'],
          'stop_at':pod['stop_at'],'scope':'terminal Pod residence; includes setup and profiling'}
    (OUT/'cost.json').write_text(json.dumps(cost,indent=2))
    # Freeze independent verifiers from the prior audited baseline; do not edit them.
    verifier=ROOT/'scripts/audit-code';verifier.mkdir(exist_ok=True)
    for p in (BASE/'audit-code').iterdir():
        if p.is_file():shutil.copyfile(p,verifier/p.name)
    hashes={p.name:sha(p.read_bytes()) for p in verifier.iterdir() if p.is_file()}
    (OUT/'auditor-manifest.json').write_text(json.dumps(hashes,indent=2))
    program = ('import json,sys; from pathlib import Path; '
               'sys.path.insert(0,sys.argv[1]); from verify_publication_probe import audit; '
               'result=audit(Path(sys.argv[2]),json.loads(Path(sys.argv[3]).read_text()),'
               'Path(sys.argv[4]),Path(sys.argv[5])); '
               'Path(sys.argv[6]).write_text(json.dumps(result,indent=2)); '
               'print(json.dumps({k:result.get(k) for k in ("passed","errors")}))')
    subprocess.run(['python3','-c',program,str(verifier),str(dest),
                    str(OUT/'submission/source-manifest.json'),
                    str(ROOT.parent/'model-cache/Qwen3-4B'),str(UPSTREAM),
                    str(OUT/'verified.json')],check=True,timeout=150)
    assert all(sha((verifier/name).read_bytes())==digest for name,digest in hashes.items())
    print('ARCHIVE_PATCH_IDENTITY_AND_COST_VERIFIED',json.dumps(cost),flush=True)


if __name__=='__main__':main()
