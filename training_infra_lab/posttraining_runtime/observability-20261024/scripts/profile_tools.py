"""Prepare tools once and export bounded captures with recorded return codes."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path('/tmp/grpo-smoke')
OUT = ROOT / 'results'


def run(cmd, name, timeout=90):
    started = time.time()
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, timeout=timeout)
        rc, text = result.returncode, result.stdout
    except subprocess.TimeoutExpired as e:
        rc, text = 124, (e.stdout or b'').decode(errors='replace') if isinstance(e.stdout, bytes) else (e.stdout or '')
    (OUT / (name + '.log')).write_text(text)
    return {'command': cmd, 'returncode': rc, 'duration_s': time.time() - started}


def locate(name):
    found = shutil.which(name)
    if found:
        return found
    patterns = {'nsys': ['cuda*/bin/nsys', 'nsight-systems*/target-linux-x64/nsys'],
                'ncu': ['cuda*/bin/ncu', 'nsight-compute*/ncu']}
    for base in ('/usr/local', '/opt/nvidia'):
        for pattern in patterns[name]:
            for path in Path(base).glob(pattern):
                if path.is_file() and os.access(path, os.X_OK):
                    return str(path)
    return None


def setup():
    records = []
    (OUT / 'nsys').mkdir(exist_ok=True)
    (OUT / 'ncu').mkdir(exist_ok=True)
    missing = [n for n in ('nsys', 'ncu') if not locate(n)]
    if missing:
        records.append(run(['sudo', 'apt-get', 'update'], 'apt-update', 150))
    for name in missing:
        prefix = 'nsight-systems-cli' if name == 'nsys' else 'nsight-compute'
        result = subprocess.run(['apt-cache', 'search', '--names-only', '^' + prefix],
                                capture_output=True, text=True, timeout=15)
        (OUT / f'apt-search-{name}.log').write_text(result.stdout + result.stderr)
        candidates = [s.split()[0] for s in result.stdout.splitlines() if s.startswith(prefix)]
        if candidates:
            package = prefix if prefix in candidates else sorted(candidates)[-1]
            records.append(run(['sudo', 'apt-get', 'install', '-y', '--no-install-recommends', package],
                               f'apt-install-{name}', 180))
    tools = {}
    for name in ('nsys', 'ncu'):
        path = locate(name)
        tools[name] = {'path': path}
        if path:
            if path != '/usr/local/bin/' + name and not shutil.which(name):
                records.append(run(['sudo', 'ln', '-s', path, '/usr/local/bin/' + name], f'link-{name}'))
            tools[name]['version'] = run([path, '--version'], name + '-version')
    enabled = bool(tools['nsys']['path'])
    (ROOT / 'code' / 'nsys-enabled.txt').write_text('1' if enabled else '0')
    records.append(run(['nvidia-smi', '-q'], 'nvidia-smi-q'))
    records.append(run(['cat', '/proc/driver/nvidia/params'], 'nvidia-driver-params'))
    (OUT / 'tools.json').write_text(json.dumps({'tools': tools, 'setup': records}, indent=2))
    print('TOOLS_READY', json.dumps(tools), flush=True)


def collect():
    tools = json.loads((OUT / 'tools.json').read_text())['tools']
    records = []
    nsys = tools['nsys']['path']
    if nsys:
        for path in sorted((OUT / 'nsys').glob('*.nsys-rep')):
            records.append(run([nsys, 'export', '--type', 'sqlite', '--output', str(path.with_suffix('.sqlite')),
                                str(path)], 'nsys-export', 90))
            for report in ('cuda_gpu_kern_sum', 'cuda_api_sum', 'nvtx_sum', 'cuda_gpu_mem_time_sum'):
                records.append(run([nsys, 'stats', '--report', report, '--format', 'csv', str(path)],
                                   report, 90))
    ncu = tools['ncu']['path']
    if ncu:
        records.append(run(['sudo', ncu, '--set', 'basic', '--kernel-name', 'regex:.*gemm.*',
                            '--launch-skip', '5', '--launch-count', '3', '--target-processes', 'all',
                            '--clock-control', 'none', '--cache-control', 'none',
                            '--export', str(OUT / 'ncu' / 'projection'),
                            '/opt/sglang/bin/python3', str(ROOT / 'code' / 'gemm_probe.py')], 'ncu-capture', 150))
        report = OUT / 'ncu' / 'projection.ncu-rep'
        if report.exists():
            records.append(run([ncu, '--import', str(report), '--page', 'raw', '--csv'], 'ncu-raw', 60))
    (OUT / 'profile-export.json').write_text(json.dumps(records, indent=2))
    print('PROFILE_EXPORT', json.dumps(records), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['setup', 'collect'])
    a = p.parse_args()
    setup() if a.stage == 'setup' else collect()
