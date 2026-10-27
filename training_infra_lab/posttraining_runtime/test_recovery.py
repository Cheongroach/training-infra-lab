import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from checkpoint_inventory import verify_inventory
from qualification_checkpoint import canonical, digest, load_journal, restore_journal, save_journal
from qualification import assess_candidate
from run_smoke import configure_recovery
from verify_recovery import audit, compare_restored
from verify_smoke import audit as audit_smoke
import test_verify_smoke as fixtures
import patch_recovery_probe

REPO = Path('/mlx_devbox/users/zhangyang.exp/vllm-omni/vllm-omni')
EVIDENCE = Path('/tmp/miles-grpo-v4-final/artifacts/results/qualification')


class JournalTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir='/tmp')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.journal = self.root / 'original/qualification'
        self.journal.mkdir(parents=True)
        for rid in (0, 1):
            manifest = json.loads((EVIDENCE / f'step-{rid}.json').read_text())
            shutil.copyfile(EVIDENCE / f'step-{rid}.json', self.journal / f'step-{rid}.json')
            for record in manifest['attempts']:
                shutil.copyfile(EVIDENCE / record['file'], self.journal / record['file'])
        self.cursor = {'epoch_id': 0, 'sample_offset': 20, 'sample_index': 80, 'sample_group_index': 20}
        self.contract = {'dataset_sha': 'frozen', 'lr': 1e-6}
        self.checkpoint = self.root / 'checkpoint/rollout_fn'
        save_journal(self.journal, self.checkpoint, completed=1, cursor=self.cursor,
                     seen=set(range(80)), contract=self.contract)
        self.results = self.root / 'resumed'
        self.results.mkdir()

    def test_real_journal_roundtrip(self):
        state, seen = restore_journal(self.checkpoint, self.results, cursor=self.cursor, contract=self.contract)
        self.assertEqual(seen, set(range(80)))
        self.assertEqual(state['next_rollout_id'], 2)
        self.assertEqual(state['next_weight_version'], 2)
        for source in self.journal.iterdir():
            self.assertEqual(source.read_bytes(), (self.results / 'qualification' / source.name).read_bytes())

    def test_previous_identity_is_rejected_after_load(self):
        _, _, seen = load_journal(self.checkpoint, cursor=self.cursor, contract=self.contract)
        batch = json.loads((self.journal / 'candidate-0-0.json').read_text())
        for sample in batch['samples']:
            for call in sample['weight_versions']:
                for span in call['spans']:
                    span['version'] = '2'
        report = assess_candidate(batch['samples'], batch['issued'], seen, rollout_id=2, weight_version=2)
        self.assertFalse(report['passed'])
        self.assertFalse(report['retryable'])
        self.assertIn('issued/returned/previous sample identity mismatch', report['fatal_errors'])

    def test_uncommitted_active_journal_is_preserved(self):
        active = self.results / 'qualification'
        shutil.copytree(EVIDENCE, active)
        old = (active / 'step-2.json').read_bytes()
        restore_journal(self.checkpoint, self.results, cursor=self.cursor, contract=self.contract)
        proof = json.loads((self.results / 'qualification-restored.json').read_text())
        self.assertFalse((active / 'step-2.json').exists())
        self.assertEqual(old, (self.results / proof['preserved_previous_journal'] / 'step-2.json').read_bytes())

    def test_install_failure_restores_original_active_journal(self):
        active = self.results / 'qualification'
        active.mkdir()
        (active / 'pending').write_text('preserve this')
        original = os.replace
        def replace(source, target):
            if Path(source).name.startswith('.qualification-restore-'):
                raise OSError('injected install failure')
            return original(source, target)
        with patch('qualification_checkpoint.os.replace', side_effect=replace):
            with self.assertRaises(OSError):
                restore_journal(self.checkpoint, self.results, cursor=self.cursor, contract=self.contract)
        self.assertEqual((active / 'pending').read_text(), 'preserve this')

    def test_contract_mismatch_leaves_active_untouched(self):
        active = self.results / 'qualification'
        active.mkdir()
        (active / 'pending').write_text('old')
        with self.assertRaises(ValueError):
            restore_journal(self.checkpoint, self.results, cursor=self.cursor, contract={'lr': 2e-6})
        self.assertEqual((active / 'pending').read_text(), 'old')

    def test_restored_cursor_mismatch(self):
        cursor = dict(self.cursor, sample_offset=21)
        with self.assertRaisesRegex(ValueError, 'cursor'):
            load_journal(self.checkpoint, cursor=cursor, contract=self.contract)

    def test_missing_candidate_is_rejected(self):
        (self.checkpoint / 'journal/candidate-1-0.json').unlink()
        with self.assertRaises(ValueError):
            load_journal(self.checkpoint, cursor=self.cursor, contract=self.contract)

    def test_state_hash_is_required(self):
        (self.checkpoint / 'state.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'hash'):
            load_journal(self.checkpoint, cursor=self.cursor, contract=self.contract)

    def test_forged_version_is_rejected_even_with_updated_hash(self):
        state = json.loads((self.checkpoint / 'state.json').read_text())
        state['next_weight_version'] = 3
        blob = canonical(state)
        (self.checkpoint / 'state.json').write_bytes(blob)
        (self.checkpoint / 'state.sha256').write_text(digest(blob))
        with self.assertRaisesRegex(ValueError, 'version'):
            load_journal(self.checkpoint, cursor=self.cursor, contract=self.contract)

    def test_boolean_identity_is_not_integer_identity(self):
        state = json.loads((self.checkpoint / 'state.json').read_text())
        state['seen_indices'][0] = False
        blob = canonical(state)
        (self.checkpoint / 'state.json').write_bytes(blob)
        (self.checkpoint / 'state.sha256').write_text(digest(blob))
        with self.assertRaisesRegex(ValueError, 'identities'):
            load_journal(self.checkpoint, cursor=self.cursor, contract=self.contract)

    def test_unaccounted_future_step_cannot_be_saved(self):
        (self.journal / 'step-2.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Unaccounted'):
            save_journal(self.journal, self.root / 'bad_checkpoint', completed=1,
                         cursor=self.cursor, seen=set(range(80)), contract=self.contract)


class LoadOptionsTest(unittest.TestCase):
    def test_resume_preserves_optimizer_rng_and_scheduler(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            root = Path(tmp)
            (root / 'training-checkpoint').mkdir()
            (root / 'training-checkpoint/latest_checkpointed_iteration.txt').write_text('1\n')
            output = root / 'results'; output.mkdir()
            data = root / 'data.jsonl'; data.write_text('{}\n' * 64)
            options = ['--rm-type','dapo','--rollout-function-path','bounded_rollout.BoundedRolloutFn',
                       '--load','base','--finetune','--no-load-optim','--no-load-rng']
            import hashlib
            tracker = root / 'training-checkpoint/latest_checkpointed_iteration.txt'
            inventory = {'verified': True, 'files': {tracker.name: {
                'bytes': tracker.stat().st_size, 'sha256': hashlib.sha256(tracker.read_bytes()).hexdigest()}}}
            (output / 'checkpoint-inventory.json').write_text(json.dumps(inventory))
            self.assertEqual(configure_recovery(options, 'resume', root, data, output), [2])
            for flag in ['--finetune','--no-load-optim','--no-load-rng']:
                self.assertNotIn(flag, options)
            self.assertIn('--use-checkpoint-opt-param-scheduler', options)
            self.assertEqual(options[options.index('--ckpt-step')+1], '1')
            self.assertEqual(len((output / 'publication-probe.jsonl').read_text().splitlines()), 4)

    def test_missing_trained_checkpoint_never_falls_back_to_base(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'rollout 1'):
                configure_recovery(['--rm-type','dapo','--rollout-function-path','bounded'],
                                   'resume', root, root / 'data', root)

    def test_probe_patch_is_exact_and_idempotency_is_rejected(self):
        original = (REPO / 'training_infra_lab/research/posttraining_20261022/upstream/miles' /
                    patch_recovery_probe.RELATIVE_PATH).read_text()
        modified = patch_recovery_probe.patched_source(original)
        compile(modified, 'actor.py', 'exec')
        self.assertNotEqual(original, modified)
        with self.assertRaises(ValueError):
            patch_recovery_probe.patched_source(modified)


class RestoreAuditTest(unittest.TestCase):
    def fixture(self):
        saved = {'rank': 0, 'world_size': 2, 'rollout_id': 1, 'policy_version': 2,
                 'state': {'param_hashes': {'layer.weight': 'weight', 'companion.version': 'old'},
                           'optimizer_hashes': [{'state': 'moment', 'step': 2}]},
                 'scheduler': {'num_steps': 32}, 'rng': {'cpu': 'rng-state'}}
        restored = copy.deepcopy(saved)
        restored.update(rollout_id=2, load_options={'finetune':False,'no_load_optim':False,
                        'no_load_rng':False,'use_checkpoint_opt_param_scheduler':True})
        return saved, restored

    def test_complete_restoration(self):
        self.assertEqual(compare_restored(*self.fixture()), [])

    def test_optimizer_counter_or_moment_mismatch(self):
        saved, restored = self.fixture()
        restored['state']['optimizer_hashes'][0]['step'] = 3
        self.assertIn('optimizer restoration mismatch', compare_restored(saved, restored))

    def test_scheduler_double_count_is_rejected(self):
        saved, restored = self.fixture()
        restored['scheduler']['num_steps'] = 48
        self.assertIn('scheduler restoration mismatch', compare_restored(saved, restored))

    def test_rng_mismatch_is_rejected(self):
        saved, restored = self.fixture()
        restored['rng']['cpu'] = 'different'
        self.assertIn('RNG restoration mismatch', compare_restored(saved, restored))

    def test_changed_source_cannot_deserialize_probe(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            with patch('verify_recovery.torch.load', side_effect=AssertionError('must not load')) as load:
                self.assertFalse(audit(Path(tmp), {'changed.py': 'untrusted'})['passed'])
                load.assert_not_called()

    def test_final_publication_is_required(self):
        fixture = fixtures.AuditGateTest('test_complete_fixture')
        fixture.setUp()
        try:
            fixture.run_audit()
            self.assertFalse(audit_smoke(fixture.root, fixture.manifest,
                                        require_final_publication=True)['passed'])
        finally:
            fixture.tearDown()

    def test_final_publication_and_changed_replicas_pass(self):
        fixture = fixtures.AuditGateTest('test_complete_fixture')
        fixture.setUp()
        try:
            fixture.events += [
                {'type':'weight_update_result','rollout_id':2,'failed_cell_ids':[],
                 'updated_cell_ids':['a','b'],'published_version':3},
                {'type':'inference_engine_weight_checksum','rollout_id':2,'weight_version':3,
                 'engine_snapshots':[{'tensor_checksums':{'layer.weight':'final'}}]*2},
            ]
            fixture.run_audit()
            result = audit_smoke(fixture.root, fixture.manifest, require_final_publication=True)
            self.assertTrue(result['passed'], result['errors'])
            self.assertTrue(result['final_update_published'])
        finally:
            fixture.tearDown()


if __name__ == '__main__':
    unittest.main()
