import copy
import shutil
import tempfile
import unittest
from dataclasses import replace

import torch

from .checkpoint import restore, save
from .contracts import digest
from .cost import Observation, features, fit_profile
from .planner import choose_profiled_plan
from .state import LocalCommitStore, StaleEpoch
from .test_runtime import example_batch


class StateCostTests(unittest.TestCase):
    def test_profile_split_context_and_domain_guards(self):
        context = {"model": "test_fixture", "kernel": "cpu", "defer_sync": False}
        observations = []
        for i, lengths in enumerate(((5,), (20,), (40,), (40, 40), (10,), (30,))):
            f = features(lengths, "padded")
            observations.append(Observation(
                str(i), "profile" if i < 4 else "validation", digest(context), lengths,
                float(f @ [2, 4, 0.1, 0.2]), int(f @ [1e7, 4e7, 1e6, 2e6]),
            ))
        model = fit_profile(observations, context)
        self.assertTrue(model.admitted)
        self.assertIsNone(model.predict((41,), digest(context)))
        with self.assertRaises(ValueError):
            model.predict((10,), "different-kernel")
        with self.assertRaises(ValueError):
            fit_profile(observations + [replace(observations[-1], split="profile")], context)
        with self.assertRaises(ValueError):
            fit_profile(observations[:-1] + [replace(observations[-1], split="evaluation")], context)
        bad = fit_profile(observations[:-1] + [replace(observations[-1], seconds=100)], context)
        self.assertFalse(bad.admitted)
        plan, report = choose_profiled_plan(
            example_batch(), 4, model, digest(context),
            memory_budget_bytes=10**9, safe_token_budget=40,
        )
        plan.validate(example_batch())
        self.assertIn("selected", report)
        # Full attention work must respect actual sequence boundaries.
        self.assertEqual(features((10, 20), "isolated_packed")[1] * 1024**2, 500)
        self.assertEqual(features((10, 20), "padded")[1] * 1024**2, 800)

    def test_dcp_replay_rng_adam_and_epoch_fencing(self):
        torch.set_num_threads(1)
        torch.manual_seed(12)
        model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(0.2), torch.nn.Linear(8, 1))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
        device = torch.device("cpu")
        contract = {"model": "cpu_dropout_fixture", "reduction": "sequence_mean"}

        def step():
            optimizer.zero_grad()
            model(torch.randn(3, 4)).square().mean().backward()
            optimizer.step()
            scheduler.step()

        for _ in range(3):
            step()
        with tempfile.TemporaryDirectory() as directory:
            store = LocalCommitStore(directory)
            epoch = store.acquire_epoch()
            metadata = {"next_update": 3, "policy_version": 3, "data_cursor": 24,
                        "plan_sha256": "fixture-plan", "batch_sha256": "fixture-batch",
                        "world_size": 1, "contract": contract}
            path = save(store, epoch, model, optimizer, metadata, device, scheduler)
            # Interrupted snapshot never changes the durable recovery point.
            incomplete = store.stage(epoch, 4)
            (incomplete / "partial").write_bytes(b"interrupted-write")
            self.assertEqual(store.read()["metadata"]["next_update"], 3)
            for _ in range(3):
                step()
            expected = copy.deepcopy(model.state_dict())
            expected_adam = copy.deepcopy(optimizer.state_dict())
            expected_lr = scheduler.get_last_lr()
            restore(store, model, optimizer, contract, device, scheduler)
            for _ in range(3):
                step()
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, expected[name], atol=0, rtol=0)
            for key, value in optimizer.state_dict()["state"].items():
                for field, tensor in value.items():
                    torch.testing.assert_close(tensor, expected_adam["state"][key][field], atol=0, rtol=0)
            self.assertEqual(scheduler.get_last_lr(), expected_lr)
            # Finished old-epoch data is forbidden from replacing the pointer.
            pending = store.stage(epoch, 6)
            shutil.copytree(path / "tensors", pending / "tensors")
            shutil.copyfile(path / "rank-0.pt", pending / "rank-0.pt")
            store.acquire_epoch()
            with self.assertRaises(StaleEpoch):
                store.commit(pending, epoch, dict(metadata, next_update=6))
            self.assertEqual(store.read()["metadata"]["next_update"], 3)
            with (path / "rank-0.pt").open("ab") as file:
                file.write(b"corruption")
            with self.assertRaises(ValueError):
                store.read()

    def test_metadata_without_referenced_shards_cannot_commit(self):
        import torch.distributed.checkpoint as dcp
        with tempfile.TemporaryDirectory() as directory:
            store = LocalCommitStore(directory)
            epoch = store.acquire_epoch()
            path = store.stage(epoch, 1)
            dcp.save({"value": torch.ones(16)}, checkpoint_id=path / "tensors")
            torch.save({"rng": "fixture"}, path / "rank-0.pt")
            shards = list((path / "tensors").glob("*.distcp"))
            self.assertTrue(shards)
            shards[0].unlink()
            metadata = {"next_update": 1, "policy_version": 1, "data_cursor": 1,
                        "plan_sha256": "fixture", "batch_sha256": "fixture",
                        "world_size": 1, "contract": {}}
            with self.assertRaisesRegex(ValueError, "Missing or truncated DCP shard"):
                store.commit(path, epoch, metadata)
            self.assertFalse((store.root / "latest.json").exists())


if __name__ == "__main__":
    unittest.main()
