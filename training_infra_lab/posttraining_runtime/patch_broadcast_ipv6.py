"""Apply the audited IPv6 rendezvous fix to this job's disposable Miles checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

RELATIVE_PATH = 'miles/backends/training_utils/weight_update/protocols/broadcast.py'
MILES_SHA = '2233d795dcf167be489d1859971c9233393f04c3'
ORIGINAL_SHA256 = 'c9349f1e3ff8405b1736458bab3029487c0dd18bd2424fb21730e65abf2f483e'


def patched_source(source):
    replacements = [
        ('from miles.utils.distributed_utils import init_process_group\n',
         'from miles.utils.distributed_utils import init_process_group\nfrom miles.utils.http_utils import wrap_ipv6\n'),
        ('    master_address = ray._private.services.get_node_ip_address()\n    with socket.socket() as sock:\n',
         '    master_address = ray._private.services.get_node_ip_address().strip("[]")\n    family = socket.AF_INET6 if ":" in master_address else socket.AF_INET\n    with socket.socket(family) as sock:\n'),
        ('        init_method=f"tcp://{master_address}:{master_port}",\n',
         '        init_method=f"tcp://{wrap_ipv6(master_address)}:{master_port}",\n'),
    ]
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError(f'Expected exactly one patch target: {old!r}')
        source = source.replace(old, new)
    return source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=Path('/root/miles'))
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    assert subprocess.check_output(['git', '-C', str(args.repo), 'rev-parse', 'HEAD'], text=True).strip() == MILES_SHA
    target = args.repo / RELATIVE_PATH
    before = target.read_bytes()
    assert hashlib.sha256(before).hexdigest() == ORIGINAL_SHA256, 'Refuse to patch an unexpected source version'
    after = patched_source(before.decode()).encode()
    target.write_bytes(after)
    report = {'base_commit': MILES_SHA, 'path': RELATIVE_PATH, 'before_sha256': ORIGINAL_SHA256, 'after_sha256': hashlib.sha256(after).hexdigest(), 'patch': 'IPv6 TCP rendezvous URL and socket family', 'gpu_verified': False}
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
