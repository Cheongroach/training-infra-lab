"""Bounded read-only observation; terminal artifacts are audited separately."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run')
    p.add_argument('--folder', type=Path, required=True)
    a = p.parse_args()
    start = time.monotonic()
    while time.monotonic() - start < 3600:
        step = time.monotonic()
        result = subprocess.run(['python', str(Path(__file__).resolve().parents[1] / 'inspect_job.py'),
                                 a.run, '--output', str(a.folder / 'observations')], timeout=180)
        if result.returncode == 0:
            status = json.loads((a.folder / 'observations/run.json').read_text())['data']['status']
            if status in ('done', 'success', 'failed', 'stopped', 'failed_to_launch'):
                print('TERMINAL_STATE ' + status, flush=True)
                break
        time.sleep(max(0, 45 - (time.monotonic() - step)))
    else:
        print('OBSERVER_LIMIT_REACHED', flush=True)


if __name__ == '__main__':
    main()
