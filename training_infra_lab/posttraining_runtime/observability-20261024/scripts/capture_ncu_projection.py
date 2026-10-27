"""Bounded follow-up capture; checks that real GPU counter records exist."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ncu', default='ncu')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    probe = out / 'projection_probe.py'
    probe.write_text(
        'import torch\n'
        'torch.manual_seed(20261024)\n'
        "x=torch.randn(256,2560,device='cuda',dtype=torch.bfloat16)\n"
        "w=torch.randn(9728,2560,device='cuda',dtype=torch.bfloat16)\n"
        'for _ in range(10): y=x @ w.T\n'
        'torch.cuda.synchronize()\n'
        'assert torch.isfinite(y).all().item()\n'
    )
    prefix = out / 'projection'
    command = [
        args.ncu, '--set', 'basic', '--kernel-name', 'regex:^nvjet_sm90_.*',
        '--launch-skip', '5', '--launch-count', '3', '--clock-control', 'none',
        '--cache-control', 'none', '--export', str(prefix), args.python, str(probe),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=150)
    log = result.stdout + result.stderr
    (out / 'capture.log').write_text(log)
    report = prefix.with_suffix('.ncu-rep')
    success = (
        result.returncode == 0 and report.is_file() and report.stat().st_size > 0
        and 'No kernels were profiled' not in log
    )
    record = {'command': command, 'returncode': result.returncode,
              'report_exists': report.is_file(), 'counters_verified': False,
              'scope': 'isolated BF16 projection, not an actual training kernel'}
    if success:
        exported = subprocess.run(
            [args.ncu, '--import', str(report), '--page', 'raw', '--csv'],
            capture_output=True, text=True, timeout=60,
        )
        (out / 'raw.csv').write_text(exported.stdout)
        (out / 'export.log').write_text(exported.stderr)
        record['export_returncode'] = exported.returncode
        record['counters_verified'] = (
            exported.returncode == 0 and '"Metric Name"' in exported.stdout
            and 'gpu__time_duration.sum' in exported.stdout
        )
    (out / 'result.json').write_text(json.dumps(record, indent=2))
    if not record['counters_verified']:
        raise SystemExit('No valid NCU counter capture; inspect capture.log and result.json')
    print('NCU_COUNTERS_VERIFIED', report)


if __name__ == '__main__':
    main()
