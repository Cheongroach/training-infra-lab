"""Freeze an isolated profiling request from the successful readiness baseline."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STAGE = HERE / 'staging'
DEST = HERE.parent / 'evidence' / 'submission'
TAG = 'grpo-profile-932d2a8751'


def replace_once(text, before, after):
    assert text.count(before) == 1, repr(before)
    return text.replace(before, after, 1)


def main():
    STAGE.mkdir(parents=True, exist_ok=True)
    (STAGE / 'job-v4').mkdir(exist_ok=True)
    baseline = ROOT / 'grpo-ray-ready-ba214ae231ee'
    shutil.copyfile(baseline / 'create.json', STAGE / 'job-v4' / 'create.json')
    names = ('run_smoke.py', 'smoke-prompts.jsonl', 'model-manifest.json', 'rollout_gate.py',
             'qualification.py', 'bounded_rollout.py', 'patch_broadcast_ipv6.py', 'broadcast-ipv6.patch',
             'patch_publication_probe.py', 'ray_readiness.py', 'patch_ray_launcher.py')
    for name in names:
        shutil.copyfile(ROOT / name, STAGE / name)
    for name in ('profile_runtime.py', 'profile_patch.py', 'profile_tools.py', 'gemm_probe.py'):
        shutil.copyfile(HERE / name, STAGE / name)
    upstream = ROOT.parent / 'research/posttraining_20261022/upstream/miles'
    entries = [
        ('miles/utils/workers/ray_worker_manager.py',
         '            runtime_env={"env_vars": self.spec.env_var(self.launch_context)},',
         '            runtime_env=__import__("profile_runtime").actor_runtime_env(\n'
         '                self.spec.env_var(self.launch_context), actor_class, self.worker_in_cell_index),'),
        ('miles/backends/megatron_utils/actor.py',
         '            with timer("actor_train"):',
         '            with timer("actor_train"), __import__("profile_runtime").profile_train_range(rollout_id):'),
    ]
    patches = []
    for path, before, after in entries:
        content = (upstream / path).read_text()
        transformed = replace_once(content, before, after)
        ast.parse(transformed)
        patches.append({'path': path, 'before': before, 'after': after,
                        'before_sha256': hashlib.sha256(content.encode()).hexdigest()})
    (STAGE / 'profile-patch-spec.json').write_text(json.dumps({
        'base_commit': '2233d795dcf167be489d1859971c9233393f04c3', 'patches': patches}, indent=2))
    driver = (STAGE / 'run_smoke.py').read_text()
    driver = replace_once(driver, "    launch_options = {}", '''    options += ['--train-env-vars', json.dumps({
        'GRPO_NSYS_ENABLED': (data.parent / 'nsys-enabled.txt').read_text().strip(),
    })]
    (output / 'requested-args.json').write_text(json.dumps(options, indent=2))
    launch_options = {}''')
    (STAGE / 'run_smoke.py').write_text(driver)
    builder = (ROOT / 'build_smoke_job.py').read_text()
    builder = replace_once(builder, "    patch_command = (", "    names += ('profile_runtime.py', 'profile_patch.py', 'profile-patch-spec.json', 'profile_tools.py', 'gemm_probe.py')\n    patch_command = (")
    builder = replace_once(builder, "    train_command = f'python3", "    patch_command += '\\nsudo /opt/sglang/bin/python3 \"$SMOKE_ROOT/code/profile_patch.py\" --spec \"$SMOKE_ROOT/code/profile-patch-spec.json\" --report \"$SMOKE_ROOT/results/profile-patch.json\"'\n    patch_command += '\\npython3 \"$SMOKE_ROOT/code/profile_tools.py\" setup'\n    train_command = f'python3")
    builder = replace_once(builder, " {snapshot_command}\n timeout --signal", ''' {snapshot_command}
 if [[ -n "${{GPU_SAMPLE_PID:-}}" ]]; then kill "$GPU_SAMPLE_PID" 2>/dev/null; wait "$GPU_SAMPLE_PID" 2>/dev/null; fi
 timeout 30s /opt/sglang/bin/ray stop --force > "$SMOKE_ROOT/results/profile-ray-stop.log" 2>&1
 timeout --signal=TERM --kill-after=10s 420s python3 "$SMOKE_ROOT/code/profile_tools.py" collect
 timeout --signal''')
    builder = replace_once(builder, 'echo GRPO_ENTRY_REACHED', '''echo GRPO_ENTRY_REACHED
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem --format=csv,nounits --loop=1 -f "$SMOKE_ROOT/results/gpu-1s.csv" &
GPU_SAMPLE_PID=$!''')
    builder = replace_once(builder, 'timeout --signal=TERM --kill-after=110s 2700s', 'timeout --signal=TERM --kill-after=460s 3000s')
    builder = builder.replace("2700s entry cap", "3000s diagnostic entry cap; rank0 rollout1 NSYS and isolated BF16 NCU")
    ast.parse(builder)
    (STAGE / 'build_smoke_job.py').write_text(builder)
    subprocess.run([sys.executable, str(STAGE / 'build_smoke_job.py'), '--output', str(DEST),
                    '--tag', TAG, '--with-ipv6-patch', '--publication-probe', '--ray-readiness',
                    '--profile', 'dapo-bounded'], check=True)
    manifest = json.loads((DEST / 'source-manifest.json').read_text())
    assert len(manifest) == 16
    for name in manifest:
        if name.endswith('.py'):
            ast.parse((STAGE / name).read_text())
    subprocess.run(['bash', '-n', str(DEST / 'entry.sh')], check=True)
    entry = (DEST / 'entry.sh').read_text()
    inner = __import__('shlex').split(entry.split('\n', 1)[1])[-1]
    subprocess.run(['bash', '-n'], input=inner, text=True, check=True)
    block = inner.split("<<'UNPACK'\n", 1)[1].split('\nUNPACK', 1)[0]
    ast.parse(block)
    check = {'node_type': 'merlin_job_run', 'check_point': 'create',
             'job_run_context': json.loads((DEST / 'create.json').read_text())}
    (DEST / 'precheck-request.json').write_text(json.dumps(check, indent=2))
    (DEST / 'local-validation.json').write_text(json.dumps({
        'source_count': len(manifest), 'python_syntax': True, 'shell_syntax': True,
        'patch_bases': patches, 'submitted': False,
        'purpose': 'diagnostic, no performance improvement claim',
    }, indent=2))
    print(DEST)


if __name__ == '__main__':
    main()
