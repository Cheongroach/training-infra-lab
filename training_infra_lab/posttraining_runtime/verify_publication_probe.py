"""Require three genuine updates and four complete final-version answers."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

from verify_smoke import audit as audit_smoke
from run_smoke import DATA_SHA, MODEL_SHA


def boxed_answer(response):
    start = response.rfind('\\boxed{')
    if start < 0:
        return None
    start += len('\\boxed{')
    depth = 1
    for end in range(start, len(response)):
        depth += (response[end] == '{') - (response[end] == '}')
        if depth == 0:
            return response[start:end].strip() or None
    return None


def assess_probe(samples, prompts, contract):
    errors, records = [], []
    if len(samples) != 4 or {s['index'] for s in samples} != set(range(4)):
        errors.append('four unique probe identities required')
    for sample in samples:
        index = sample['index']
        problems = []
        n, tokens = sample['response_length'], sample['tokens']
        if type(index) is not int or index not in range(4) or sample['prompt'] != prompts[index]:
            problems.append('frozen prompt identity')
        if (sample['status'] != 'completed' or type(n) is not int
                or not 0 < n <= contract['max_response_tokens'] or len(tokens) <= n):
            problems.append('natural completion/token length')
        finish = sample.get('metadata', {}).get('publication_probe_finish_reason', {})
        if (finish.get('type') != 'stop' or type(finish.get('matched')) is not int
                or finish['matched'] not in contract['eos_token_ids']):
            problems.append('raw EOS stop reason')
        lp = sample.get('rollout_log_probs')
        if (lp is None or len(lp) != n or not all(
                isinstance(v, (int, float)) and math.isfinite(v) and v <= 1e-5 for v in lp)):
            problems.append('finite behavior probabilities')
        spans = [span for call in sample.get('weight_versions', []) for span in call['spans']]
        cursor = len(tokens) - n
        if not spans:
            problems.append('missing version spans')
        for span in spans:
            if (str(span['version']) != str(contract['weight_version'])
                    or span['abs_start'] != cursor or span['abs_end'] <= cursor
                    or span['abs_end'] > len(tokens)):
                problems.append('version/coverage')
            cursor = span['abs_end']
        if cursor != len(tokens):
            problems.append('incomplete version coverage')
        answer = boxed_answer(sample['response'])
        reward = sample.get('reward', {})
        if not answer or not isinstance(reward.get('pred'), str) or reward['pred'] in ('', '[INVALID]'):
            problems.append('parseable final boxed answer')
        if reward.get('score') not in (-1.0, 1.0):
            problems.append('DAPO score')
        records.append({
            'index': index, 'response_tokens': n, 'status': sample['status'],
            'finish_reason': finish, 'boxed_answer': answer,
            'pred': reward.get('pred'), 'score': reward.get('score'),
            'passed': not problems, 'errors': problems,
        })
        errors.extend(f'probe {index}: {problem}' for problem in problems)
    return {'passed': not errors, 'errors': errors, 'samples': records,
            'correct_answers': sum(r['score'] == 1.0 for r in records),
            'capability_claim': False}


def audit(root, manifest, tokenizer_path, upstream):
    result = {'passed': False, 'errors': [], 'performance_claim': False, 'capability_claim': False}
    errors = result['errors']
    for name, sha in manifest.items():
        path = root / 'code' / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            errors.append('source: ' + name)
    if errors:
        return result
    # The collector also checks archive identity and runtime patch hashes before this call.
    semantic = audit_smoke(root, manifest, require_final_publication=True)
    result['semantic_audit'] = semantic
    errors.extend(semantic['errors'])
    output = root / 'results'
    if 'ray_readiness.py' in manifest:
        from ray_readiness import RAY_VERSION, RAY_COMMIT
        readiness = json.loads((output / 'ray-readiness.json').read_text())
        if (readiness.get('passed') is not True or readiness.get('last_status') != 'SUCCEEDED'
                or readiness.get('num_cpus') != 1 or readiness.get('num_gpus') != 0
                or readiness.get('training_submitted_by_gate') is not False
                or readiness.get('version', {}).get('ray_version') != RAY_VERSION
                or readiness.get('version', {}).get('ray_commit') != RAY_COMMIT
                or not readiness.get('nonce', '').startswith('RAY_READINESS_')
                or readiness.get('logs', '').splitlines().count(readiness.get('nonce')) != 1):
            errors.append('Ray readiness CPU job evidence')
        result['ray_readiness'] = readiness
    contract = json.loads((output / 'publication-probe-contract.json').read_text())
    data = root / 'code/smoke-prompts.jsonl'
    probe_data = output / 'publication-probe.jsonl'
    if (hashlib.sha256(data.read_bytes()).hexdigest() != DATA_SHA
            or probe_data.read_text() != '\n'.join(data.read_text().splitlines()[:4]) + '\n'
            or hashlib.sha256(probe_data.read_bytes()).hexdigest() != contract['data_sha256']
            or contract['samples'] != 4 or contract['max_response_tokens'] != 4096
            or contract['temperature'] != 0 or contract['weight_version'] != 3
            or contract['eos_token_ids'] != [151645, 151643] or contract['tokenizer_eos_token_id'] != 151645
            or contract['model_revision'] != MODEL_SHA or not contract['natural_eos_required']
            or not contract['parseable_boxed_answer_required'] or contract['capability_claim']):
        errors.append('probe generation contract')
    requested = json.loads((output / 'requested-args.json').read_text())
    for flag, value in (('--eval-max-response-len', '4096'), ('--eval-temperature', '0'),
                        ('--eval-interval', '3'), ('--n-samples-per-eval-prompt', '1'),
                        ('--apply-chat-template-kwargs', '{"enable_thinking":false}')):
        if requested.count(flag) != 1 or requested[requested.index(flag) + 1] != value:
            errors.append('requested probe flag: ' + flag)
    generation = json.loads((output / 'generation-contract.json').read_text())
    if generation['enable_thinking'] or not generation['all_prompts_have_closed_thinking_prefix']:
        errors.append('thinking template contract')
    model_manifest = json.loads((root / 'code/model-manifest.json').read_text())
    for name in ('config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
                 'vocab.json', 'merges.txt'):
        if hashlib.sha256((tokenizer_path / name).read_bytes()).hexdigest() != model_manifest['files'][name]['sha256']:
            errors.append('local tokenizer provenance: ' + name)
    if errors:
        return result
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    rows = [json.loads(line) for line in probe_data.read_text().splitlines()]
    prompts = [tokenizer.apply_chat_template(row['prompt'], tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False) for row in rows]
    scoring_path = upstream / 'miles/rollout/rm_hub/math_dapo_utils.py'
    if hashlib.sha256(scoring_path.read_bytes()).hexdigest() != '3d6b61357e2ee8f1704ba0eb1726ea2dd65dbf9c76e963f20d5ff6598b8f7b23':
        errors.append('frozen DAPO scoring provenance')
        return result
    spec = importlib.util.spec_from_file_location('frozen_dapo', scoring_path)
    dapo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dapo)
    import torch
    samples = torch.load(output / 'rollout-eval_2.pt', map_location='cpu', weights_only=False)['samples']
    result['publication_probe'] = assess_probe(samples, prompts, contract)
    errors.extend(result['publication_probe']['errors'])
    for sample in samples:
        index = sample['index']
        if type(index) is not int or index not in range(4):
            continue
        if sample['tokens'][:-sample['response_length']] != tokenizer.encode(prompts[index], add_special_tokens=False):
            errors.append(f'probe {index}: prompt token identity')
        recomputed = dapo.compute_score(sample['response'], str(rows[index]['label']))
        if sample['reward'] != recomputed:
            errors.append(f'probe {index}: recomputed DAPO reward')
        answer = boxed_answer(sample['response'])
        if answer and dapo.normalize_final_answer(answer) != recomputed['pred']:
            errors.append(f'probe {index}: boxed/pred mismatch')
    result['scoring_source_sha256'] = hashlib.sha256(scoring_path.read_bytes()).hexdigest()
    result['generation_contract'] = contract
    result['passed'] = not errors
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('--source-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    here = Path(__file__).resolve().parent
    try:
        result = audit(args.root, json.loads(args.source_manifest.read_text()),
                       here / 'model-cache/Qwen3-4B',
                       here.parent / 'research/posttraining_20261022/upstream/miles')
    except Exception as exc:
        result = {'passed': False, 'performance_claim': False, 'capability_claim': False,
                  'errors': [f'{type(exc).__name__}: {exc}']}
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
