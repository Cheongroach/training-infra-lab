"""Package an explicit allowlist; exclude environment dumps and training tensors."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
files={ROOT/p for p in ('observability-20261024.html','README.md','REPORT.md')}
for directory in ('assets','_shared'):
    files.update(p for p in (ROOT/directory).rglob('*') if p.is_file())
files.update((ROOT/'scripts').glob('*.py'))
files.update(p for p in (ROOT/'evidence/artifacts/code').glob('*') if p.is_file())
for filename in (
    'normalized-history.json','phase-events.json','diagnostic-phase-events.json','baseline-report.md',
    'archive-audit.json','auditor-manifest.json','cost.json','verified.json','profile-summary.json',
    'nsys-schema.json','checksum-source-excerpts.json','browser-validation.json',
    'profiling-upload-short.json','profiling-asset-verified.json',
):
    files.add(ROOT/'evidence'/filename)
for filename in ('create.checked.json','precheck-result.json','source-manifest.json','entry.sh'):
    files.add(ROOT/'evidence/submission'/filename)
files.update((ROOT/'evidence/grafana-history').glob('*.json'))
for filename in (
    'gpu-1s.csv','tools.json','nsys-version.log','ncu-version.log','profile-export.json',
    'cuda_gpu_kern_sum.log','cuda_api_sum.log','nvtx_sum.log','cuda_gpu_mem_time_sum.log',
    'ncu-capture.log','source-manifest.json','runtime-patch.json','publication-probe-patch.json',
    'ray-launcher-patch.json','profile-patch.json','publication-probe-contract.json','requested-args.json',
    'ncu/microbench.json','nsys/learner-rank0-6626.nsys-rep','nsys/learner-rank0-6626.sqlite',
    'nsys/phases-rank0.jsonl','nsys/phases-rank1.jsonl',
):
    files.add(ROOT/'evidence/artifacts/results'/filename)
assert all(p.is_file() and not p.is_symlink() for p in files)
assert all('.validation' not in p.parts and p.suffix not in ('.pt','.gz') for p in files)
for p in files:
    assert p.name not in ('pod-stdout.log','entry.log','log-source.json')
manifest={str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(files)}
manifest_path=ROOT/'evidence/delivery-manifest.json'
manifest_path.write_text(json.dumps(manifest,indent=2))
files.add(manifest_path)
archive=ROOT/'observability-20261024.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as output:
    for p in sorted(files):output.write(p,str(Path(ROOT.name)/p.relative_to(ROOT)))
with zipfile.ZipFile(archive) as output:
    assert output.testzip() is None
    assert len(output.namelist())==len(files)
    for name,record in manifest.items():
        assert hashlib.sha256(output.read(str(Path(ROOT.name)/name))).hexdigest()==record['sha256']
receipt={'file':archive.name,'files':len(files),'bytes':archive.stat().st_size,
         'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'crc_verified':True,
         'member_hashes_verified':len(manifest),'full_environment_logs_included':False,
         'training_tensors_included':False,'browser_validation':json.loads((ROOT/'evidence/browser-validation.json').read_text())}
(ROOT/'evidence/delivery-receipt.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt,indent=2))
