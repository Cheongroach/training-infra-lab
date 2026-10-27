"""Run three synchronous GRPO updates inside the validated Miles image.

Preparation and training are separate stages. This driver does not submit jobs
or declare semantic/performance success on a successful process exit.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

MILES_SHA = '2233d795dcf167be489d1859971c9233393f04c3'
MODEL_SHA = '1cfa9a7208912126459214e8b04321603b3df60c'
DATA_SHA = '67173a6074167cb280f1cf70542ccd1c1f91583defa8698f2428d268858e4fff'


def configure_publication_probe(options, data, output, max_response):
    if max_response != 4096:
        raise ValueError('The complete publication probe requires 4096 response tokens')
    eval_data = output / 'publication-probe.jsonl'
    rows = data.read_text().splitlines()[:4]
    if len(rows) != 4:
        raise ValueError('The publication probe requires four frozen prompts')
    eval_data.write_text('\n'.join(rows) + '\n')
    options += ['--eval-interval', '3', '--skip-eval-before-train',
                '--eval-prompt-data', 'publication_probe', str(eval_data),
                '--n-samples-per-eval-prompt', '1', '--eval-max-response-len', str(max_response),
                '--eval-temperature', '0']
    return {'samples': 4, 'max_response_tokens': max_response, 'temperature': 0,
            'weight_version': 3, 'natural_eos_required': True,
            'parseable_boxed_answer_required': True, 'capability_claim': False,
            'data_sha256': hashlib.sha256(eval_data.read_bytes()).hexdigest()}


def configure_recovery(options, phase, root, data, output):
    if phase == 'full':
        return [0, 1, 2]
    if options[options.index('--rm-type') + 1] != 'dapo':
        raise ValueError('Recovery requires the validated DAPO profile')
    options[options.index('--rollout-function-path') + 1] = 'recoverable_rollout.CheckpointedBoundedRolloutFn'
    options += ['--custom-megatron-before-train-step-hook-path', 'recovery_checks.before_step']
    checkpoint = root / 'training-checkpoint'
    if phase == 'checkpoint':
        options += ['--save', str(checkpoint), '--save-interval', '2',
                    '--debug-exit-after-rollout', '2',
                    '--custom-megatron-post-save-hook-path', 'recovery_checks.after_save']
        return [0, 1]
    tracker = checkpoint / 'latest_checkpointed_iteration.txt'
    if not tracker.is_file() or tracker.read_text().strip() != '1':
        raise ValueError('Recovery requires the checkpoint for completed rollout 1')
    from checkpoint_inventory import verify_inventory
    verify_inventory(checkpoint, json.loads((root / 'results/checkpoint-inventory.json').read_text()))
    for flag in ('--finetune', '--no-load-optim', '--no-load-rng'):
        options.remove(flag)
    options[options.index('--load') + 1] = str(checkpoint)
    eval_data = output / 'publication-probe.jsonl'
    # Reuse four frozen prompts solely to verify generation under final version 3.
    eval_data.write_text('\n'.join(data.read_text().splitlines()[:4]) + '\n')
    options += ['--exit-on-missing-checkpoint', '--ckpt-step', '1',
                '--use-checkpoint-opt-param-scheduler', '--eval-interval', '3',
                '--skip-eval-before-train', '--eval-prompt-data', 'publication_probe', str(eval_data),
                '--n-samples-per-eval-prompt', '1', '--eval-max-response-len', '512', '--eval-temperature', '0']
    return [2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=('prepare', 'train'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--staged-model', action='store_true')
    p.add_argument('--phase', choices=('full', 'checkpoint', 'resume'), default='full')
    p.add_argument('--profile', choices=('legacy', 'dapo-gated', 'dapo-bounded'), default='legacy')
    p.add_argument('--publication-probe', action='store_true')
    p.add_argument('--ray-readiness', action='store_true')
    args = p.parse_args()
    root, data = args.root.resolve(), args.data.resolve()
    if args.ray_readiness and not args.publication_probe:
        raise ValueError('The Ray readiness correction requires the complete publication probe')
    if args.publication_probe and (args.stage != 'train' or args.phase != 'full'
                                  or args.profile != 'dapo-bounded'):
        raise ValueError('The complete probe requires full train --profile dapo-bounded')
    if args.phase != 'full' and (args.stage != 'train' or args.profile != 'dapo-bounded'):
        raise ValueError('Recovery phases require train --profile dapo-bounded')
    if hashlib.sha256(data.read_bytes()).hexdigest() != DATA_SHA:
        raise ValueError('Prompt subset differs from the frozen preflight data')
    repo = Path('/root/miles')
    sha = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if sha != MILES_SHA:
        raise ValueError('Unexpected Miles revision')
    os.chdir(repo)
    from miles.utils.external_utils import command_utils
    model = root / 'models' / 'Qwen3-4B'
    output = root / 'results'
    if args.phase != 'full':
        output = output / args.phase
    output.mkdir(parents=True, exist_ok=True)
    config = command_utils.default_config()
    config.output_dir = str(output)
    backend = config.create_backend()
    if args.stage == 'prepare':
        if args.staged_model:
            proof = json.loads((output / 'assets-verified.json').read_text())
            assert proof['verified'] is True
        else:
            from huggingface_hub import snapshot_download
            snapshot_download('Qwen/Qwen3-4B', revision=MODEL_SHA, local_dir=model,
                              allow_patterns=['*.json', '*.safetensors', '*.model', '*.jinja', '*.txt'],
                              max_workers=4)
        files = {}
        for path in sorted(model.rglob('*')):
            if path.is_file() and '.cache' not in path.parts:
                with path.open('rb') as handle:
                    digest = hashlib.file_digest(handle, 'sha256').hexdigest()
                files[str(path.relative_to(model))] = {'bytes': path.stat().st_size, 'sha256': digest}
        (output / 'model-manifest.json').write_text(json.dumps({
            'repo': 'Qwen/Qwen3-4B', 'revision': MODEL_SHA, 'files': files}, indent=2))
        backend.convert_checkpoint(model_name='Qwen3-4B', megatron_model_type='qwen3-4B',
                                   num_gpus_per_node=2, dir_dst=str(model.parent), hf_checkpoint=str(model))
        return
    options = [
        '--hf-checkpoint', str(model), '--ref-load', str(model.parent / 'Qwen3-4B_torch_dist'),
        '--load', str(model.parent / 'Qwen3-4B_torch_dist'), '--finetune', '--no-load-optim', '--no-load-rng',
        '--prompt-data', str(data), '--input-key', 'prompt', '--label-key', 'label', '--apply-chat-template',
        '--rm-type', 'deepscaler', '--num-rollout', '3', '--rollout-batch-size', '4',
        '--n-samples-per-prompt', '4', '--global-batch-size', '16', '--rollout-max-response-len', '2048',
        '--rollout-temperature', '1', '--seed', '20261024', '--advantage-estimator', 'grpo',
        '--kl-loss-coef', '0.00', '--kl-loss-type', 'low_var_kl', '--entropy-coef', '0.00',
        '--eps-clip', '0.2', '--eps-clip-high', '0.28', '--optimizer', 'adam', '--lr', '1e-6',
        '--lr-decay-style', 'constant', '--weight-decay', '0.1', '--adam-beta1', '0.9', '--adam-beta2', '0.98',
        '--tensor-model-parallel-size', '2', '--sequence-parallel', '--pipeline-model-parallel-size', '1',
        '--context-parallel-size', '1', '--recompute-granularity', 'full', '--recompute-method', 'uniform',
        '--recompute-num-layers', '1', '--use-dynamic-batch-size', '--max-tokens-per-gpu', '4096', '--balance-data',
        '--attention-dropout', '0.0', '--hidden-dropout', '0.0', '--accumulate-allreduce-grads-in-fp32',
        '--attention-softmax-in-fp32', '--attention-backend', 'flash', '--actor-num-nodes', '1',
        '--num-gpus-per-node', '4', '--actor-num-gpus-per-node', '2', '--rollout-num-gpus', '2', '--rollout-num-gpus-per-engine', '1',
        '--sglang-mem-fraction-static', '0.6', '--update-weight-transfer-mode', 'broadcast',
        '--check-weight-update-equal', '--save-local-weight-checksum',
        '--save-debug-event-data', str(output / 'events'),
        '--save-debug-train-data', str(output / 'train-{rollout_id}-rank-{rank}.pt'),
        '--save-debug-rollout-data', str(output / 'rollout-{rollout_id}.pt'),
        '--save-debug-trajectory-data', str(output / 'trajectory-{rollout_id}.jsonl'),
    ]
    gated = args.profile in ('dapo-gated', 'dapo-bounded')
    if gated:
        options[options.index('--rm-type') + 1] = 'dapo'
        options[options.index('--rollout-max-response-len') + 1] = '4096'
        options += [
            '--reward-key', 'score',
            '--apply-chat-template-kwargs', '{"enable_thinking":false}',
            '--custom-rollout-log-function-path', 'rollout_gate.check_before_train',
        ]
        if args.profile == 'dapo-bounded':
            options += ['--rollout-function-path', 'bounded_rollout.BoundedRolloutFn']
        os.environ['PYTHONPATH'] = os.pathsep.join(
            [str(data.parent), os.environ.get('PYTHONPATH', '')]
        ).rstrip(os.pathsep)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
        rows = [json.loads(line) for line in data.read_text().splitlines()]
        suffix = '<think>\n\n</think>\n\n'
        rendered = [
            tokenizer.apply_chat_template(
                row['prompt'], tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            ) for row in rows
        ]
        if not all(prompt.endswith(suffix) for prompt in rendered):
            raise ValueError('The tokenizer did not render the non-thinking prefix')
        (output / 'generation-contract.json').write_text(json.dumps({
            'enable_thinking': False, 'prompts_checked': len(rows),
            'all_prompts_have_closed_thinking_prefix': True,
            'assistant_prefix_suffix': suffix,
        }, indent=2))
    expected_rollouts = configure_recovery(options, args.phase, root, data, output)
    if args.publication_probe:
        probe = configure_publication_probe(options, data, output, 4096)
        generation = json.loads((model / 'generation_config.json').read_text())
        probe['eos_token_ids'] = generation['eos_token_id']
        probe['tokenizer_eos_token_id'] = tokenizer.eos_token_id
        probe['model_revision'] = MODEL_SHA
        (output / 'publication-probe-contract.json').write_text(json.dumps(probe, indent=2))
    (output / 'smoke-contract.json').write_text(json.dumps({
        'profile': args.profile, 'requested_updates': 3,
        'phase': args.phase, 'expected_rollout_ids': expected_rollouts,
        'max_response_tokens': 4096 if gated else 2048,
        'reward_key': 'score' if gated else None,
        'rollout_gate_required': gated,
        'max_candidate_batches_per_update': 4 if args.profile == 'dapo-bounded' else 1,
    }, indent=2))
    (output / 'requested-args.json').write_text(json.dumps(options, indent=2))
    launch_options = {}
    if args.ray_readiness:
        os.environ['RAY_DIAGNOSTIC_DIR'] = str(output / 'ray-diagnostics')

        def before_submit():
            subprocess.run(
                [sys.executable, str(data.parent / 'ray_readiness.py'),
                 '--output', str(output / 'ray-readiness.json')],
                check=True, timeout=135,
            )

        launch_options['before_ray_job_submit'] = before_submit
    backend.execute_train(train_args=shlex.join(options), num_gpus_per_node=4,
                          megatron_model_type='qwen3-4B', train_script='train.py',
                          job_lifetime='launcher', config=config, **launch_options)
    (output / 'driver-completed.json').write_text(json.dumps({
        'driver_returned': True, 'requested_updates': 3, 'semantic_audit_pending': True,
        'phase': args.phase, 'expected_rollout_ids': expected_rollouts,
        'performance_claim': False}, indent=2))


if __name__ == '__main__':
    main()
