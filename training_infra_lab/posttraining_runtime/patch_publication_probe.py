"""Preserve raw evaluation stop reasons without changing generation or scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

MILES_SHA = '2233d795dcf167be489d1859971c9233393f04c3'
RELATIVE_PATH = 'miles/rollout/generate_hub/single_turn.py'
ORIGINAL_SHA = '2bccc5555bb8459ca8c9092e8f95d88b6135d006cc1aeb1cadfc878594b77d91'
BEFORE = '    output = await post(url, payload, headers=compute_routing_headers(args, sample))\n'
AFTER = BEFORE + '''    if input.evaluation:
        sample.metadata["publication_probe_finish_reason"] = dict(output["meta_info"]["finish_reason"])
'''


def patched_source(content):
    if hashlib.sha256(content.encode()).hexdigest() != ORIGINAL_SHA or content.count(BEFORE) != 1:
        raise ValueError('Unexpected Miles rollout source; stop-reason capture refused')
    return content.replace(BEFORE, AFTER, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/root/miles'))
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    sha = subprocess.check_output(['git', '-C', str(args.root), 'rev-parse', 'HEAD'], text=True).strip()
    if sha != MILES_SHA:
        raise ValueError('Unexpected Miles commit')
    path = args.root / RELATIVE_PATH
    modified = patched_source(path.read_text())
    path.write_text(modified)
    args.report.write_text(json.dumps({
        'base_commit': sha, 'path': RELATIVE_PATH, 'before_sha256': ORIGINAL_SHA,
        'after_sha256': hashlib.sha256(modified.encode()).hexdigest(),
        'purpose': 'Retain raw evaluation EOS evidence; generation and reward unchanged',
    }, indent=2))


if __name__ == '__main__':
    main()
