"""Synthetic rejection tests for the audit gate; these are NOT GPU evidence."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from verify_smoke import audit
from rollout_gate import assess_batch


class AuditGateTest(unittest.TestCase):
    def setUp(self):
        parent = Path(__file__).resolve().parent / 'verifier-tests'
        parent.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=parent)
        self.root = Path(self.tmp.name)
        self.results = self.root / 'results'
        (self.root / 'code').mkdir()
        (self.results / 'events').mkdir(parents=True)
        (self.root / 'code/test.py').write_text('synthetic fixture\n')
        self.manifest = {'test.py': hashlib.sha256((self.root / 'code/test.py').read_bytes()).hexdigest()}
        (self.results / 'driver-completed.json').write_text('{}')
        self.events = []
        for rid in [-1, 0, 1]:
            self.events.append({'type': 'weight_update_result', 'rollout_id': None if rid == -1 else rid, 'failed_cell_ids': [], 'updated_cell_ids': ['a', 'b'], 'published_version': rid + 1})
            self.events.append({'type': 'inference_engine_weight_checksum', 'rollout_id': rid, 'weight_version': rid + 1, 'engine_snapshots': [{'tensor_checksums': {'layer.weight': str(rid)}}] * 2})
        for rid in range(3):
            samples = [{'index': i, 'group_index': i // 4, 'response_length': 2, 'tokens': [1, 2, 3], 'status': 'completed', 'reward': i % 2, 'rollout_log_probs': [-1., -2.], 'weight_versions': [{'spans': [{'version': str(rid), 'abs_start': 1, 'abs_end': 3}]}]} for i in range(16)]
            torch.save({'samples': samples}, self.results / f'rollout-{rid}.pt')
            td = {'tokens': [[1, 2, 3]] * 16, 'sample_indices': list(range(16)), 'response_lengths': [2] * 16, 'loss_masks': [[1, 1]] * 16, 'advantages': [[1., -1.]] * 16, 'log_probs': [[-1., -2.]] * 16, 'rollout_log_probs': [[-1., -2.]] * 16}
            torch.save({'rollout_data': td}, self.results / f'train-{rid}-rank-0.pt')
            self.events.append({'type': 'metric', 'rollout_id': rid, 'metrics': {'train/grad_norm': 1.}})
            self.events.append({'type': 'train_group_step_end', 'rollout_id': rid, 'role': 'actor', 'cell_outcomes': {'0': ['normal', 'normal']}})
            for rank in range(2):
                self.events.append({'type': 'train_engine_local_weight_checksum', 'rollout_id': rid, 'source': {'rank_within_cell': rank}, 'state': {'param_hashes': {'layer.weight': str(rid)}, 'optimizer_hashes': [{'state_dict': {'state': {'0': {'exp_avg': str(rid), 'exp_avg_sq': str(rid)}}, 'param_groups': [{'step': rid + 1}]}}]}})

    def tearDown(self):
        self.tmp.cleanup()

    def run_audit(self):
        (self.results / 'events/test.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in self.events))
        return audit(self.root, self.manifest)

    def test_complete_fixture(self):
        self.assertTrue(self.run_audit()['passed'])

    def test_missing_rank(self):
        self.events = [e for e in self.events if not (e['type'] == 'train_engine_local_weight_checksum' and e['source']['rank_within_cell'] == 1)]
        self.assertFalse(self.run_audit()['passed'])

    def test_stale_engine(self):
        for e in self.events:
            if e['type'] == 'inference_engine_weight_checksum':
                for s in e['engine_snapshots']:
                    s['tensor_checksums']['layer.weight'] = 'unchanged'
        self.assertFalse(self.run_audit()['passed'])

    def test_nonfinite_behavior_probability(self):
        path = self.results / 'rollout-0.pt'
        data = torch.load(path, weights_only=False)
        data['samples'][0]['rollout_log_probs'][0] = float('nan')
        torch.save(data, path)
        self.assertFalse(self.run_audit()['passed'])

    def test_zero_learning_signal(self):
        path = self.results / 'train-0-rank-0.pt'
        data = torch.load(path, weights_only=False)
        data['rollout_data']['advantages'] = [[0., 0.]] * 16
        torch.save(data, path)
        self.assertFalse(self.run_audit()['passed'])

    def test_incomplete_group(self):
        path = self.results / 'rollout-0.pt'
        data = torch.load(path, weights_only=False)
        data['samples'].pop()
        torch.save(data, path)
        self.assertFalse(self.run_audit()['passed'])

    def test_incomplete_version_spans(self):
        path = self.results / 'rollout-0.pt'
        data = torch.load(path, weights_only=False)
        data['samples'][0]['weight_versions'][0]['spans'][0]['abs_end'] = 2
        torch.save(data, path)
        self.assertFalse(self.run_audit()['passed'])

    def test_missing_optimizer_rank_outcome(self):
        event = next(e for e in self.events if e['type'] == 'train_group_step_end')
        event['cell_outcomes']['0'] = ['normal']
        self.assertFalse(self.run_audit()['passed'])

    def test_failed_optimizer_rank(self):
        event = next(e for e in self.events if e['type'] == 'train_group_step_end')
        event['cell_outcomes']['0'] = ['normal', 'error']
        self.assertFalse(self.run_audit()['passed'])

    def test_sample_payload_mismatch(self):
        path = self.results / 'train-0-rank-0.pt'
        data = torch.load(path, weights_only=False)
        data['rollout_data']['tokens'][0] = [1, 4, 5]
        torch.save(data, path)
        self.assertFalse(self.run_audit()['passed'])

    def test_behavior_payload_mismatch(self):
        path = self.results / 'train-0-rank-0.pt'
        data = torch.load(path, weights_only=False)
        data['rollout_data']['rollout_log_probs'][0] = [-3., -4.]
        torch.save(data, path)
        self.assertFalse(self.run_audit()['passed'])

    def test_short_loss_masks(self):
        path = self.results / 'train-0-rank-0.pt'
        data = torch.load(path, weights_only=False)
        data['rollout_data']['loss_masks'].pop()
        torch.save(data, path)
        self.assertFalse(self.run_audit()['passed'])

    def test_changed_source_does_not_deserialize(self):
        (self.root / 'code/test.py').write_text('modified')
        with patch('verify_smoke.torch.load', side_effect=AssertionError('must not deserialize')) as load:
            self.assertFalse(self.run_audit()['passed'])
            load.assert_not_called()

    def test_changed_source(self):
        (self.root / 'code/test.py').write_text('modified')
        self.assertFalse(self.run_audit()['passed'])

    def add_gated_fixture(self):
        source = (Path(__file__).parent / 'rollout_gate.py').read_bytes()
        (self.root / 'code/rollout_gate.py').write_bytes(source)
        self.manifest['rollout_gate.py'] = hashlib.sha256(source).hexdigest()
        for rid in range(3):
            path = self.results / f'rollout-{rid}.pt'
            data = torch.load(path, weights_only=False)
            for sample in data['samples']:
                sample['reward'] = {'score': 1. if sample['index'] % 2 else -1., 'pred': '1'}
                sample['prompt'] = '<|im_start|>assistant\n<think>\n\n</think>\n\n'
            torch.save(data, path)
            report = assess_batch(data['samples'])
            report['rollout_id'] = rid
            (self.results / f'rollout-gate-{rid}.json').write_text(json.dumps(report))

    def test_gated_complete_fixture(self):
        self.add_gated_fixture()
        self.assertTrue(self.run_audit()['passed'])

    def test_gated_missing_report(self):
        self.add_gated_fixture()
        (self.results / 'rollout-gate-1.json').unlink()
        self.assertFalse(self.run_audit()['passed'])

    def test_gated_report_cannot_hide_bad_samples(self):
        self.add_gated_fixture()
        path = self.results / 'rollout-0.pt'
        data = torch.load(path, weights_only=False)
        for sample in data['samples']:
            sample['status'] = 'truncated'
            sample['reward'] = {'score': -1., 'pred': '[INVALID]'}
        torch.save(data, path)
        self.assertFalse(self.run_audit()['passed'])

    def test_gate_rejects_nonfinite_probability(self):
        self.add_gated_fixture()
        data = torch.load(self.results / 'rollout-0.pt', weights_only=False)
        data['samples'][0]['rollout_log_probs'][0] = float('nan')
        self.assertFalse(assess_batch(data['samples'])['passed'])

    def test_gate_rejects_duplicate_identity(self):
        self.add_gated_fixture()
        data = torch.load(self.results / 'rollout-0.pt', weights_only=False)
        data['samples'][0]['index'] = data['samples'][1]['index']
        self.assertFalse(assess_batch(data['samples'])['passed'])

    def test_optimizer_counter_does_not_prove_learning(self):
        for event in self.events:
            if event['type'] == 'train_engine_local_weight_checksum':
                event['state']['optimizer_hashes'][0]['state_dict']['state'] = {
                    '0': {'exp_avg': 'unchanged', 'exp_avg_sq': 'unchanged'}
                }
        self.assertFalse(self.run_audit()['passed'])

    def test_gated_wrong_thinking_prefix(self):
        self.add_gated_fixture()
        path = self.results / 'rollout-0.pt'
        data = torch.load(path, weights_only=False)
        data['samples'][0]['prompt'] = '<|im_start|>assistant\n'
        torch.save(data, path)
        self.assertFalse(self.run_audit()['passed'])

    def add_bounded_fixture(self):
        from qualification import assess_candidate
        self.add_gated_fixture()
        for name in ('bounded_rollout.py', 'qualification.py'):
            source = (Path(__file__).parent / name).read_bytes()
            (self.root / 'code' / name).write_bytes(source)
            self.manifest[name] = hashlib.sha256(source).hexdigest()
        directory = self.results / 'qualification'
        directory.mkdir()
        seen = set()
        for rid in range(3):
            path = self.results / f'rollout-{rid}.pt'
            data = torch.load(path, weights_only=False)
            for s in data['samples']:
                s['index'] += rid * 16
                s['group_index'] += rid * 4
                s['lineage'] = {'source_sample_index': s['index'], 'output_index': 0, 'output_count': 1}
            torch.save(data, path)
            tp = self.results / f'train-{rid}-rank-0.pt'
            td = torch.load(tp, weights_only=False)
            td['rollout_data']['sample_indices'] = [s['index'] for s in data['samples']]
            torch.save(td, tp)
            issued = [{k: s[k] for k in ('index', 'group_index')} for s in data['samples']]
            report = assess_candidate(data['samples'], issued, seen, rollout_id=rid, weight_version=rid)
            seen.update(s['index'] for s in data['samples'])
            name = f'candidate-{rid}-0.json'
            blob = json.dumps({'samples': data['samples'], 'issued': issued}).encode()
            (directory / name).write_bytes(blob)
            manifest = {'rollout_id': rid, 'weight_version': rid, 'max_candidates': 4,
                        'passed': True, 'attempts': [{
                            'attempt': 0, 'file': name, 'sha256': hashlib.sha256(blob).hexdigest(),
                            'sample_indices': [s['index'] for s in data['samples']],
                            'assessment': report, 'decision': 'admitted'}]}
            (directory / f'step-{rid}.json').write_text(json.dumps(manifest))

    def test_bounded_complete_fixture(self):
        self.add_bounded_fixture()
        self.assertTrue(self.run_audit()['passed'])

    def test_bounded_missing_candidate(self):
        self.add_bounded_fixture()
        (self.results / 'qualification/candidate-1-0.json').unlink()
        self.assertFalse(self.run_audit()['passed'])

    def test_bounded_admitted_payload_mismatch(self):
        self.add_bounded_fixture()
        path = self.results / 'qualification/candidate-0-0.json'
        candidate = json.loads(path.read_text())
        candidate['samples'][0]['tokens'][0] = 5
        blob = json.dumps(candidate).encode()
        path.write_bytes(blob)
        mp = self.results / 'qualification/step-0.json'
        manifest = json.loads(mp.read_text())
        manifest['attempts'][0]['sha256'] = hashlib.sha256(blob).hexdigest()
        mp.write_text(json.dumps(manifest))
        self.assertFalse(self.run_audit()['passed'])

    def add_rejected_candidate(self):
        from qualification import assess_candidate
        self.add_bounded_fixture()
        directory = self.results / 'qualification'
        original = directory / 'candidate-0-0.json'
        original.rename(directory / 'candidate-0-1.json')
        batch = json.loads((directory / 'candidate-0-1.json').read_text())
        for s in batch['samples']:
            s['index'] += 1000
            s['group_index'] += 250
            s['lineage']['source_sample_index'] = s['index']
            s['reward']['score'] = -1.
        batch['issued'] = [{k: s[k] for k in ('index', 'group_index')} for s in batch['samples']]
        blob = json.dumps(batch).encode()
        original.write_bytes(blob)
        mp = directory / 'step-0.json'
        manifest = json.loads(mp.read_text())
        manifest['attempts'][0].update(attempt=1, file='candidate-0-1.json')
        manifest['attempts'].insert(0, {
            'attempt': 0, 'file': original.name, 'sha256': hashlib.sha256(blob).hexdigest(),
            'sample_indices': [s['index'] for s in batch['samples']],
            'assessment': assess_candidate(batch['samples'], batch['issued'], set(),
                                            rollout_id=0, weight_version=0),
            'decision': 'rejected',
        })
        mp.write_text(json.dumps(manifest))

    def test_bounded_rejected_batch_accounted(self):
        self.add_rejected_candidate()
        result = self.run_audit()
        self.assertTrue(result['passed'], result['errors'])
        self.assertEqual(result['updates'][0]['rejected_batches'], 1)

    def test_bounded_rejection_reason_cannot_be_forged(self):
        self.add_rejected_candidate()
        path = self.results / 'qualification/step-0.json'
        manifest = json.loads(path.read_text())
        manifest['attempts'][0]['assessment']['errors'] = ['invented reason']
        path.write_text(json.dumps(manifest))
        self.assertFalse(self.run_audit()['passed'])


if __name__ == '__main__':
    unittest.main()
