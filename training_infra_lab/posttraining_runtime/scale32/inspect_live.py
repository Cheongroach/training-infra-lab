"""Record bounded phase evidence from an active Pod without dumping its environment."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess

from collect_preflight import CLI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--pod", type=int, required=True)
    parser.add_argument("--label", default="phase")
    args = parser.parse_args()
    current = json.loads((args.folder / "current.json").read_text())["data"]
    if current["status"] != "running":
        raise ValueError("Live inspection requires a running Job")
    pods = json.loads((args.folder / "pods.json").read_text())["data"]
    pod = pods[args.pod]
    if pod["status"] != 1:
        raise ValueError("Selected Pod is not active")
    manifest = json.loads((args.folder / "manifest.json").read_text())
    remote = manifest["remote_output"]
    tag = Path(remote).name
    script = f"""
import json, os, time
from collections import Counter
from pathlib import Path
r = Path({('/tmp/' + tag)!r})
s = Path({remote!r})
processes = []
for d in Path('/proc').iterdir():
    if not d.name.isdigit():
        continue
    try:
        name = (d/'comm').read_text().strip()
        if name.startswith(('python', 'ray', 'gcs', 'sudo')):
            processes.append({{'pid':int(d.name),'name':name,
                              'wait':(d/'wchan').read_text().strip()}})
    except (OSError, PermissionError):
        pass
logs = {{}}
for name in ('gcs_server.out','gcs_server.err','raylet.out','raylet.err'):
    p=Path('/tmp/ray/session_latest/logs')/name
    if p.is_file():
        logs[name]=p.read_text(errors='replace')[-5000:]
conversion = {{}}
for label, directory in (('local',r/'models/Qwen3-8B_torch_dist'),
                         ('shared',s/'converted-checkpoint'),
                         ('model_assets',r/'models/Qwen3-8B')):
    conversion[label]=[{{'path':str(p.relative_to(directory)),
                         'bytes':p.stat().st_size,'mtime':p.stat().st_mtime}}
                        for p in directory.rglob('*') if p.is_file()]
pointer=r/'models/Qwen3-8B_torch_dist/latest_checkpointed_iteration.txt'
conversion['pointer']=pointer.read_text() if pointer.is_file() else None
results = r/'results'
event_counts = Counter()
boundaries = []
for p in (s/'results/events').rglob('*.jsonl'):
    for line in p.read_text(errors='replace').splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        event_counts[event.get('type')] += 1
        if event.get('type') in ('train_group_step_end', 'weight_update_result'):
            boundaries.append({{k:event.get(k) for k in
                               ('timestamp','type','rollout_id','role',
                                'published_version','updated_cell_ids','failed_cell_ids')}})
print(json.dumps({{'observed_unix':time.time(),'arnold_id':os.environ.get('ARNOLD_ID'),
 'local_results':[{{'path':str(p.relative_to(r)),'bytes':p.stat().st_size,
                   'mtime':p.stat().st_mtime}} for p in results.rglob('*') if p.is_file()],
 'shared_results':[{{'path':str(p.relative_to(s/'results')),'bytes':p.stat().st_size}}
                   for p in (s/'results').glob('*') if p.is_file()],
 'event_counts':dict(event_counts),'boundaries':boundaries[-10:],
 'control':{{p.name:json.loads(p.read_text()) for p in (s/'control').glob('*.json')}},
 'entry_tail':(results/'entry.log').read_text(errors='replace')[-14000:],
 'processes':processes,'ray_log_tails':logs,'conversion':conversion}}))
"""
    body = {
        "job_run_id": current["sid"],
        "trial_id": current["runtime_context"]["trial_id"],
        "pod_name": pod["pod_name"],
        "exec": "python3 -c " + shlex.quote(script),
        "backend": "ssh",
        "exec_timeout_seconds": 30,
    }
    response = subprocess.run(
        [CLI, "--control-plane", "cn", "job-v2", "runs", "execute-script",
         "--json", json.dumps(body)],
        capture_output=True, timeout=65,
    )
    prefix = args.folder / f"live-{args.label}-pod-{args.pod}"
    prefix.with_suffix(".json").write_bytes(response.stdout)
    prefix.with_suffix(".stderr").write_bytes(response.stderr)
    print(response.stdout.decode(errors="replace"))
    response.check_returncode()


if __name__ == "__main__":
    main()
