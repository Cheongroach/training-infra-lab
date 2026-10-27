"""Audit pinned source and proposed topology without importing or launching Miles.

This is preparation validation, not a CUDA, RDMA, algorithm or runtime test.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_FILES = (
    'train_async.py',
    'miles/utils/arguments.py',
    'miles/backends/training_utils/weight_update/updater.py',
    'miles/backends/training_utils/weight_update/session.py',
    'miles/backends/training_utils/weight_update/protocol.py',
    'miles/backends/training_utils/weight_update/protocols/broadcast.py',
    'miles/backends/training_utils/weight_update/protocols/p2p.py',
    'miles/backends/training_utils/weight_update/protocols/delta.py',
    'miles/backends/training_utils/weight_update/protocols/transports/mooncake.py',
    'miles/backends/training_utils/weight_update/protocols/utils/transfer_buffers.py',
    'miles/backends/fsdp_utils/hf_weight_iterator.py',
    'miles/rollout/fully_async_data_buffer.py',
    'miles/rollout/fully_async_rollout.py',
    'miles/rollout/submission_scheduler.py',
    'tests/e2e/fsdp/test_qwen3_0.6B_fsdp_disk_delta.py',
    'tests/e2e/torchtitan/test_qwen3_5_4B_disk_delta.py',
    'tests/e2e/megatron/test_qwen3_4B_disk_delta.py',
    'docs/advanced/disaggregated-rollout.md',
    'docs/advanced/p2p-weight-transfer.md',
    'docs/getting-started/installation.md',
)
# Backend-owned Megatron/SGLang flags are deliberately outside this static check.
GLOBAL_FLAGS = (
    '--actor-num-nodes', '--actor-num-gpus-per-node', '--rollout-num-gpus',
    '--rollout-num-gpus-per-engine', '--update-weight-transfer-mode',
    '--update-weight-buffer-size', '--update-weights-interval',
    '--rollout-batch-size', '--n-samples-per-prompt', '--global-batch-size',
    '--rollout-max-response-len', '--rollout-temperature', '--advantage-estimator',
    '--rm-type', '--prompt-data', '--hf-checkpoint', '--check-weight-update-equal',
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_source(snapshot, relative):
    raw = (snapshot / relative).read_bytes()
    entry = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    if relative.endswith('.py'):
        tree = ast.parse(raw, filename=relative)
        entry['symbols'] = [
            {'name': node.name, 'line': node.lineno, 'end_line': node.end_lineno}
            for node in ast.walk(tree)
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        ]
    return entry


def main():
    source = json.loads((ROOT / 'upstream/miles-source.json').read_text())
    plan = json.loads((ROOT / 'experiment-plan.json').read_text())
    require(plan['miles_commit'] == source['commit'], 'plan/source revision mismatch')
    require(plan['status'] == 'proposal_not_submitted', 'unexpected submission state')
    require(plan['backend'] == 'megatron' and not plan['colocate'],
            'packed broadcast requires non-colocated Megatron at this revision')
    require(not plan['lora'], 'this plan includes protocols without LoRA support')
    workload = plan['fixed_workload']
    require(workload['group_size'] >= 2, 'GRPO requires multiple samples per group')
    require(workload['global_batch_size'] == workload['group_size'] * workload['prompt_groups_per_rollout'],
            'one-rollout/one-global-batch proposal is inconsistent')
    topology = []
    for phase in plan['phases']:
        learner = phase['learner_nodes'] * phase['learner_gpus_per_node']
        parallel = phase['learner_tp'] * phase['learner_pp'] * phase['learner_cp']
        require(learner % parallel == 0, 'learner mesh is not divisible')
        dp = learner // parallel
        require(workload['global_batch_size'] % dp == 0, 'global batch is not divisible by DP')
        require(phase['learner_gpus_per_node'] % phase['learner_tp'] == 0, 'TP does not fit within node')
        require(phase['rollout_gpus'] % phase['rollout_tp'] == 0, 'rollout replica count is fractional')
        require(phase['total_gpus'] == learner + phase['rollout_gpus'], 'GPU total mismatch')
        topology.append(dict(phase=phase['id'], learner_gpus=learner, learner_dp=dp,
                             rollout_replicas=phase['rollout_gpus']//phase['rollout_tp'],
                             total_gpus=phase['total_gpus']))
    budget = plan['budget']
    main_phase = next(p for p in plan['phases'] if p['id'] == 'main')
    require(budget['example_main_gpu_hours'] == main_phase['total_gpus'] *
            budget['example_nights'] * budget['example_hours_per_night'], 'budget arithmetic mismatch')
    snapshot = ROOT / 'upstream/miles'
    entries = {relative: read_source(snapshot, relative) for relative in SOURCE_FILES}
    tree = ast.parse((snapshot / 'miles/utils/arguments.py').read_text())
    registered = {
        arg.value for node in ast.walk(tree)
        if isinstance(node, ast.Call) and (
            (isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument')
            or (isinstance(node.func, ast.Name) and node.func.id == 'reset_arg'))
        for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
    }
    require(not (set(GLOBAL_FLAGS)-registered), f'unregistered flags: {set(GLOBAL_FLAGS)-registered}')
    result = dict(
        validation='local_static_preparation_only', gpu_jobs_launched=0,
        upstream_tests_executed=False, source_commit=source['commit'],
        plan_sha256=hashlib.sha256((ROOT/'experiment-plan.json').read_bytes()).hexdigest(),
        topology=topology, example_main_gpu_hours=budget['example_main_gpu_hours'],
        global_cli_names_found=list(GLOBAL_FLAGS), files=entries,
        unverified=plan['launch_prerequisites'],
    )
    output = ROOT / 'source-audit.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('validation','source_commit','topology','example_main_gpu_hours')},indent=2))
    print(f'Wrote {output}')


if __name__ == '__main__':
    main()
