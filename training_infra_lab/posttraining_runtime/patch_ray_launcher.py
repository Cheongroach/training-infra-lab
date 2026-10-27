"""Hash-locked Miles cleanup fix; no retry of training submissions."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

MILES_SHA = '2233d795dcf167be489d1859971c9233393f04c3'
RELATIVE_PATH = 'miles/utils/external_utils/ray_job.py'
ORIGINAL_SHA = '2ddb56bf6777c4bb4aecbb67cfd71618639a4df7407ac8516148206cf168ba2b'
BEFORE = '''        client = JobSubmissionClient(address)
        try:
            client.submit_job(submission_id=submission_id, entrypoint=entrypoint, runtime_env=runtime_env)
            print(f"Ray job {submission_id} submitted to {address}", flush=True)
            await _follow_job(client, submission_id, stop_requested)
        finally:
            # Unlike the stop HTTP response, the CLI waits for a terminal job status.
            subprocess.run(
                [sys.executable, "-m", "ray.scripts.scripts", "job", "stop", "--address", address, submission_id],
                check=True,
                timeout=45,
            )
'''
AFTER = '''        client = JobSubmissionClient(address)
        submitted = False
        try:
            client.submit_job(submission_id=submission_id, entrypoint=entrypoint, runtime_env=runtime_env)
            submitted = True
            print(f"Ray job {submission_id} submitted to {address}", flush=True)
            await _follow_job(client, submission_id, stop_requested)
        finally:
            original_failure = sys.exc_info()[0] is not None
            if submitted:
                try:
                    # The CLI waits for a terminal status, unlike the stop HTTP response.
                    subprocess.run(
                        [sys.executable, "-m", "ray.scripts.scripts", "job", "stop", "--address", address, submission_id],
                        check=True,
                        timeout=45,
                    )
                except Exception as cleanup_error:
                    if not original_failure:
                        raise
                    print(f"RAY_JOB_CLEANUP_ERROR {cleanup_error}", flush=True)
'''
FINALLY_BEFORE = '''    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
'''
FINALLY_AFTER = '''    finally:
        from ray_readiness import collect_diagnostics
        collect_diagnostics("launcher")
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
'''


def patched_source(content):
    if (hashlib.sha256(content.encode()).hexdigest() != ORIGINAL_SHA
            or content.count(BEFORE) != 1 or content.count(FINALLY_BEFORE) != 1):
        raise ValueError('Unexpected Miles Ray launcher; cleanup patch refused')
    return content.replace(BEFORE, AFTER, 1).replace(FINALLY_BEFORE, FINALLY_AFTER, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/root/miles'))
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    commit = subprocess.check_output(['git', '-C', str(args.root), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != MILES_SHA:
        raise ValueError('Unexpected Miles commit')
    path = args.root / RELATIVE_PATH
    modified = patched_source(path.read_text())
    path.write_text(modified)
    args.report.write_text(json.dumps({
        'base_commit': commit, 'path': RELATIVE_PATH, 'before_sha256': ORIGINAL_SHA,
        'after_sha256': hashlib.sha256(modified.encode()).hexdigest(),
        'training_submission_retries': 0,
        'purpose': 'Stop only confirmed submissions, retain original exceptions, snapshot Ray logs',
    }, indent=2))


if __name__ == '__main__':
    main()
