"""Bounded CPU-only Ray Jobs gate and local diagnostic collection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import socket
import time
from urllib import error, request
import uuid

RAY_VERSION = '2.58.1'
RAY_COMMIT = '4c7892ef427b19e7585ee28f548a60ab241929e8'
NO_AGENT = 'No available agent to submit job, please try again later..'
LOG_NAME = re.compile(r'^(dashboard[^/]*|raylet[^/]*|gcs_server[^/]*|runtime_env_agent[^/]*)\.(log|out|err)(\.\d+)?$')


def snapshot_ray_logs(destination, label, ray_root=Path('/tmp/ray')):
    """Keep bounded log tails; never replace a previous snapshot."""
    target = Path(destination) / label
    if target.exists():
        return
    target.mkdir(parents=True)
    report = {'files': {}, 'errors': [], 'max_files': 64, 'max_bytes_per_file': 256 * 1024}
    try:
        base = ray_root.resolve(strict=True)
        session = (base / 'session_latest').resolve(strict=True)
        if session.parent != base or not session.name.startswith('session_'):
            raise ValueError('Unexpected Ray session path')
        logs = session / 'logs'
        report['session'] = session.name
        for path in sorted(logs.iterdir()):
            if not LOG_NAME.fullmatch(path.name) or path.is_symlink() or not path.is_file():
                continue
            if len(report['files']) >= report['max_files']:
                break
            try:
                with path.open('rb') as handle:
                    original_bytes = os.fstat(handle.fileno()).st_size
                    handle.seek(max(0, original_bytes - report['max_bytes_per_file']))
                    content = handle.read(report['max_bytes_per_file'])
                (target / path.name).write_bytes(content)
                report['files'][path.name] = {
                    'bytes': len(content), 'original_bytes': original_bytes,
                    'sha256': hashlib.sha256(content).hexdigest(),
                }
            except OSError as exc:
                report['errors'].append(f'{path.name}: {type(exc).__name__}')
    except (OSError, ValueError) as exc:
        report['errors'].append(str(exc))
    (target / 'manifest.json').write_text(json.dumps(report, indent=2))


def collect_diagnostics(label):
    destination = os.environ.get('RAY_DIAGNOSTIC_DIR')
    if destination:
        try:
            snapshot_ray_logs(destination, label)
        except Exception as exc:
            print(f'RAY_DIAGNOSTIC_ERROR {type(exc).__name__}: {exc}', flush=True)


class HTTPFailure(RuntimeError):
    def __init__(self, code, body):
        self.code, self.body = code, body
        super().__init__(f'Ray HTTP {code}: {body}')


def check_readiness(address, output, budget=120, interval=1):
    if not 0 < budget <= 120:
        raise ValueError('Ray gate budget must be in (0, 120] seconds')
    if output.exists():
        raise ValueError('Refusing to replace an earlier readiness result')
    output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    deadline = start + budget
    nonce = 'RAY_READINESS_' + uuid.uuid4().hex
    submission_id = 'miles-readiness-' + uuid.uuid4().hex
    proof = {'passed': False, 'submission_id': submission_id, 'nonce': nonce,
             'budget_seconds': budget, 'num_cpus': 1, 'num_gpus': 0,
             'attempts': [], 'training_submitted_by_gate': False}
    opener = request.build_opener(request.ProxyHandler({}))
    known_job, terminal = False, False
    transport = (error.URLError, TimeoutError, ConnectionError, socket.timeout)

    def save():
        proof['elapsed_seconds'] = round(time.monotonic() - start, 3)
        output.write_text(json.dumps(proof, indent=2))

    def call(method, path, body=None, timeout=None):
        remaining = deadline - time.monotonic()
        if timeout is None and remaining <= 0:
            raise TimeoutError('Ray readiness deadline reached')
        timeout = timeout if timeout is not None else min(12, remaining)
        req = request.Request(address.rstrip('/') + path, method=method,
                              data=json.dumps(body).encode() if body is not None else None,
                              headers={'Content-Type': 'application/json'})
        try:
            with opener.open(req, timeout=timeout) as response:
                content = response.read(1024 * 1024 + 1)
                if len(content) > 1024 * 1024:
                    raise ValueError('Oversized Ray response')
                return json.loads(content)
        except error.HTTPError as exc:
            raise HTTPFailure(exc.code, exc.read(4096).decode(errors='replace')) from exc

    save()
    try:
        while time.monotonic() < deadline:
            try:
                version = call('GET', '/api/version')
                if (version.get('ray_version') != RAY_VERSION
                        or version.get('ray_commit') != RAY_COMMIT
                        or not version.get('session_name')):
                    raise ValueError('Unexpected Ray build/session')
                if 'version' in proof and proof['version'] != version:
                    raise ValueError('Ray session changed during readiness')
                proof['version'] = version
                path = '/api/jobs/' + submission_id
                try:
                    info = call('GET', path)
                except HTTPFailure as exc:
                    if exc.code != 404:
                        raise
                    # Reconcile this stable ID before every POST, including an ambiguous timeout.
                    payload = {
                        'submission_id': submission_id,
                        'entrypoint': 'python3 -c ' + shlex.quote(f'print({nonce!r}, flush=True)'),
                        'runtime_env': {}, 'entrypoint_num_cpus': 1, 'entrypoint_num_gpus': 0,
                    }
                    try:
                        submitted = call('POST', '/api/jobs/', payload)
                    except HTTPFailure as submit_error:
                        if submit_error.code != 500 or submit_error.body.strip() != NO_AGENT:
                            raise
                        proof['attempts'].append({'kind': 'no_agent', 'message': str(submit_error)})
                    else:
                        if submitted.get('submission_id') != submission_id:
                            raise ValueError('Readiness submission identity mismatch')
                        known_job = True
                        proof['attempts'].append({'kind': 'submitted'})
                else:
                    known_job = True
                    status = info.get('status')
                    proof['last_status'] = status
                    if status in ('FAILED', 'STOPPED', 'SUCCEEDED'):
                        terminal = True
                        if status != 'SUCCEEDED':
                            raise RuntimeError(f'Readiness job ended with {status}: {info.get("message")}')
                        logs = call('GET', path + '/logs')['logs']
                        if logs.splitlines().count(nonce) != 1:
                            raise ValueError('Readiness log nonce mismatch')
                        proof['logs'] = logs
                        proof['passed'] = True
                        save()
                        print(f'RAY_JOBS_READY {submission_id} {nonce}', flush=True)
                        return proof
                    if status not in ('PENDING', 'RUNNING'):
                        raise ValueError(f'Unexpected readiness status: {status}')
            except transport as exc:
                proof['attempts'].append({'kind': 'transport', 'message': str(exc)[:4096]})
            save()
            time.sleep(max(0, min(interval, deadline - time.monotonic())))
        raise TimeoutError(f'Ray Jobs did not pass the CPU gate in {budget} seconds')
    except Exception as exc:
        proof['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if known_job and not terminal:
            try:
                proof['stop_response'] = call('POST', '/api/jobs/' + submission_id + '/stop', timeout=3)
            except Exception as exc:
                proof['stop_error'] = f'{type(exc).__name__}: {exc}'
        save()
        collect_diagnostics('readiness')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--address', default=os.environ.get('RAY_ADDRESS', 'http://127.0.0.1:8265'))
    parser.add_argument('--snapshot-only', action='store_true')
    args = parser.parse_args()
    if args.snapshot_only:
        collect_diagnostics('archive')
    else:
        if args.output is None:
            parser.error('--output is required for the readiness gate')
        check_readiness(args.address, args.output)


if __name__ == '__main__':
    main()
