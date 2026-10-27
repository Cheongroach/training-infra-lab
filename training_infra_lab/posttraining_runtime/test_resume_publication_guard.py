"""Reject missing, stale or corrupted raw checkpoint engine evidence."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from checkpoint_inventory import verify_inventory
from resume_publication_guard import canonical_hash, compare_engine_commit, normalize_engine_results


class RawEngineGuardTest(unittest.TestCase):
    def fixture(self):
        checksums={f'layer{i}.weight':hashlib.sha256(str(i).encode()).hexdigest() for i in range(290)}
        results=[({'cell_id':cell},{'success':True,'ranks':[{'parallelism_info':[{'rank':0}], 'checksums':dict(checksums)}]}) for cell in ('a','b')]
        current=normalize_engine_results(results)
        inventory={'verified':True,'completed_rollout_id':1,'files':{'model':{'bytes':4,'sha256':'f'*64}}}
        saved={'completed_rollout_id':1,'policy_version':2,'checkpoint_inventory_sha256':canonical_hash(inventory),'engines':copy.deepcopy(current)}
        return results,inventory,saved,current

    def test_all_real_sha256_digests_match(self):
        _,inventory,saved,current=self.fixture()
        compare_engine_commit(saved,current,inventory)

    def test_missing_replica(self):
        results,_,_,_=self.fixture()
        with self.assertRaises(ValueError):normalize_engine_results(results[:1])

    def test_short_checksum_is_not_raw_sha256(self):
        results,_,_,_=self.fixture();results[0][1]['ranks'][0]['checksums']['layer0.weight']='short'
        with self.assertRaises(ValueError):normalize_engine_results(results)

    def test_missing_tensor(self):
        results,_,_,_=self.fixture();results[0][1]['ranks'][0]['checksums'].pop('layer0.weight')
        with self.assertRaises(ValueError):normalize_engine_results(results)

    def test_different_replicas(self):
        results,_,_,_=self.fixture();results[0][1]['ranks'][0]['checksums']['layer0.weight']='0'*64
        with self.assertRaises(ValueError):normalize_engine_results(results)

    def test_wrong_trained_commit_even_if_replicas_agree(self):
        _,inventory,saved,current=self.fixture()
        for e in current:e['tensor_checksums']['layer0.weight']='0'*64
        with self.assertRaises(ValueError):compare_engine_commit(saved,current,inventory)

    def test_same_weights_bound_to_wrong_checkpoint(self):
        _,inventory,saved,current=self.fixture();inventory['files']['model']['sha256']='0'*64
        with self.assertRaises(ValueError):compare_engine_commit(saved,current,inventory)

    def test_wrong_version(self):
        _,inventory,saved,current=self.fixture();saved['policy_version']=1
        with self.assertRaises(ValueError):compare_engine_commit(saved,current,inventory)

    def test_duplicated_identity(self):
        results,_,_,_=self.fixture();results[1][0]['cell_id']='a'
        with self.assertRaises(ValueError):normalize_engine_results(results)

    def test_checkpoint_bytes_change_before_restore(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            root=Path(tmp);p=root/'model';p.write_bytes(b'abcd')
            report={'verified':True,'files':{'model':{'bytes':4,'sha256':hashlib.sha256(b'abcd').hexdigest()}}}
            verify_inventory(root,report);p.write_bytes(b'abce')
            with self.assertRaises(ValueError):verify_inventory(root,report)

if __name__=='__main__':unittest.main()
