"""Regress actual failed-v1 replay and reject forged restored-publication evidence."""
import ast
import copy
from pathlib import Path
import unittest

import patch_recovery_probe
from verify_recovery import audit, compose_events, read_events

BASE = Path(__file__).resolve().parents[1] / 'research/posttraining_20261022/upstream/miles'
V1 = Path('/tmp/miles-grpo-recovery-v1/artifacts')


class ProvenanceTest(unittest.TestCase):
    def setUp(self):
        self.before = read_events(V1 / 'results/checkpoint')
        self.after = read_events(V1 / 'results/resume')
        self.uid = '18debcd845b9466887528db5c7bc7f62'

    def checksum(self):
        return next(e for e in self.after if e['type'] == 'inference_engine_weight_checksum'
                    and e['debug_weight_update_id'] == self.uid)

    def test_actual_replay_is_deduplicated_and_new_initial_pair_is_removed(self):
        events, proof = compose_events(self.before, self.after)
        self.assertGreater(proof['replayed_events'], 0)
        self.assertEqual(proof['debug_weight_update_id'], self.uid)
        for event in self.before:
            self.assertEqual(events.count(event), 1)
        self.assertFalse(any(e.get('debug_weight_update_id') == self.uid for e in events))

    def test_replayed_history_alone_is_insufficient(self):
        with self.assertRaises(ValueError):
            compose_events(self.before, self.before)

    def test_changed_historical_event_is_rejected(self):
        old = next(e for e in self.after if e in self.before)
        old['rollout_id'] = 99
        with self.assertRaisesRegex(ValueError, 'changed contents'):
            compose_events(self.before, self.after)

    def test_checksum_needs_matching_update_id(self):
        self.checksum()['debug_weight_update_id'] = 'unrelated-transfer'
        with self.assertRaisesRegex(ValueError, 'coverage'):
            compose_events(self.before, self.after)

    def test_missing_checksum_does_not_reuse_saved_version_two(self):
        self.after.remove(self.checksum())
        with self.assertRaises(ValueError):
            compose_events(self.before, self.after)

    def test_equal_version_with_wrong_weights_is_rejected(self):
        self.checksum()['engine_snapshots'][0]['tensor_checksums']['forged.weight'] = 'wrong'
        with self.assertRaises(ValueError):
            compose_events(self.before, self.after)

    def test_different_incarnation_is_rejected(self):
        self.checksum()['engine_snapshots'][0]['workers_hash'] = 'other-worker'
        with self.assertRaises(ValueError):
            compose_events(self.before, self.after)

    def test_repeated_transfer_with_new_timestamp_is_rejected(self):
        extra = copy.deepcopy(self.checksum())
        extra['timestamp'] = '2026-10-24T03:48:30Z'
        self.after.append(extra)
        with self.assertRaisesRegex(ValueError, 'coverage'):
            compose_events(self.before, self.after)

    def test_actual_failed_v1_stays_failed(self):
        import json
        manifest = json.loads((V1 / 'results/source-manifest.json').read_text())
        result = audit(V1, manifest)
        self.assertFalse(result['passed'])
        self.assertTrue(result['resumed_initial_publication']['fresh_publication_verified'])
        self.assertIn('missing restore fingerprints rank 0', result['errors'])


class BaselinePatchTest(unittest.TestCase):
    def test_actual_driver_guard_preserves_fresh_check_only(self):
        original = (BASE / 'train.py').read_text()
        modified = patch_recovery_probe.patched_driver(original)
        tree = ast.parse(modified)
        guard = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                     and ast.unparse(n.test).startswith('args.start_rollout_id == 2'))
        from types import SimpleNamespace
        expression = compile(ast.Expression(guard.test), '<guard>', 'eval')
        recovery = 'recoverable_rollout.CheckpointedBoundedRolloutFn'
        for start, plugin, expected in ((0, recovery, False), (2, recovery, True),
                                        (1, recovery, False), (2, 'other', False)):
            self.assertEqual(eval(expression, {'args': SimpleNamespace(
                rollout_function_path=plugin, start_rollout_id=start)}), expected)
        self.assertEqual(modified.count('action="compare"'), original.count('action="compare"'))
        with self.assertRaises(ValueError):
            patch_recovery_probe.patched_driver(modified)


class RawGuardTest(unittest.TestCase):
    def setUp(self):
        from resume_publication_guard import canonical_hash
        self.inventory = {'verified': True, 'completed_rollout_id': 1, 'files': {'shard': 'sha'}}
        self.current = [{'cell_id': cell, 'tensor_checksums': {
            f'layer.{i}': '0' * 64 for i in range(290)}} for cell in ('a', 'b')]
        self.saved = {'completed_rollout_id': 1, 'policy_version': 2,
                      'checkpoint_inventory_sha256': canonical_hash(self.inventory),
                      'engines': copy.deepcopy(self.current)}

    def compare(self):
        from resume_publication_guard import compare_engine_commit
        compare_engine_commit(self.saved, self.current, self.inventory)

    def test_complete_commit_passes(self):
        self.compare()

    def test_changed_checkpoint_inventory_rejects_identical_engines(self):
        self.inventory['files']['shard'] = 'changed'
        with self.assertRaises(ValueError):
            self.compare()

    def test_missing_replica_rejects(self):
        self.current.pop()
        with self.assertRaises(ValueError):
            self.compare()

    def test_changed_full_raw_tensor_rejects(self):
        self.current[1]['tensor_checksums']['layer.0'] = '1' * 64
        with self.assertRaises(ValueError):
            self.compare()

    def test_missing_tensor_in_saved_and_current_rejects(self):
        for snapshot in self.current + self.saved['engines']:
            del snapshot['tensor_checksums']['layer.0']
        with self.assertRaises(ValueError):
            self.compare()


if __name__ == '__main__':
    unittest.main()
