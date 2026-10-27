"""Refuse any Miles file whose frozen base bytes do not match."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def patch_text(source, before, after, expected):
    if hashlib.sha256(source.encode()).hexdigest() != expected or source.count(before) != 1:
        raise ValueError('Profile patch base mismatch')
    result = source.replace(before, after, 1)
    compile(result, '<profile-patch>', 'exec')
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/root/miles'))
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    spec = json.loads(a.spec.read_text())
    commit = subprocess.check_output(['git', '-C', str(a.root), 'rev-parse', 'HEAD'], text=True).strip()
    assert commit == spec['base_commit']
    rows = []
    prepared = []
    for item in spec['patches']:
        path = a.root / item['path']
        result = patch_text(path.read_text(), item['before'], item['after'], item['before_sha256'])
        prepared.append((path, result))
        rows.append({k: item[k] for k in ('path', 'before_sha256')})
        rows[-1]['after_sha256'] = hashlib.sha256(result.encode()).hexdigest()
    for path, result in prepared:
        path.write_text(result)
    a.report.write_text(json.dumps({'base_commit': commit, 'patches': rows}, indent=2))


if __name__ == '__main__':
    main()
