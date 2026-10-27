"""CPU semantic gates; these do not establish device/runtime performance."""

import copy
import unittest
from dataclasses import replace

import torch
import torch.nn.functional as F

from .contracts import Trajectory, UpdateBatch
from .executor import execute_update
from .model import Actor, make_lm
from .planner import CapacityError, plan_update
from .rl import collate, surrogate_sum


def example_batch(lengths=(7, 19, 11, 9, 15, 5, 8, 13)):
    generator = torch.Generator().manual_seed(21)
    ts = []
    for i, length in enumerate(lengths):
        ids = tuple(torch.randint(2, 127, (length,), generator=generator).tolist())
        mask = tuple([0] * 2 + [1] * (length - 3))
        ts.append(Trajectory(
            str(i), str(i // 2), 0, ids, mask,
            tuple([-5.0] * (length - 1)), float(i % 2),
        ))
    return UpdateBatch(tuple(ts), 0, 0, 2)


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(12)
        cls.lm = make_lm(tiny=True)

    def test_full_vocabulary_shift_and_gradient(self):
        lm = copy.deepcopy(self.lm).eval()
        actor = Actor(lm, 3)
        data = collate(example_batch(), [0, 1], torch.device("cpu"))
        actual = actor(**{k: data[k] for k in ("input_ids", "attention_mask", "response_mask")})
        output = lm(input_ids=data["input_ids"], attention_mask=data["attention_mask"],
                    use_cache=False).logits[:, :-1].float()
        expected = -F.cross_entropy(
            output.reshape(-1, output.shape[-1]), data["input_ids"][:, 1:].reshape(-1),
            reduction="none",
        ).reshape_as(actual)
        mask = data["response_mask"].bool()
        torch.testing.assert_close(actual[mask], expected[mask], rtol=1e-5, atol=1e-6)
        a = torch.autograd.grad(actual.sum(), tuple(actor.parameters()))
        b = torch.autograd.grad(expected[mask].sum(), tuple(actor.parameters()))
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, rtol=1e-4, atol=3e-6)

    def test_fp32_head_matches_independent_projection_under_autocast(self):
        actor = Actor(copy.deepcopy(self.lm).to(dtype=torch.bfloat16), 3, fp32_head=True).eval()
        data = collate(example_batch(), [0, 1], torch.device("cpu"))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = actor(**{k: data[k] for k in ("input_ids", "attention_mask", "response_mask")})
            hidden = actor.backbone(input_ids=data["input_ids"], attention_mask=data["attention_mask"],
                                    use_cache=False).last_hidden_state
        logits = F.linear(hidden[:, :-1].float(), actor.head.weight.float())
        expected = -F.cross_entropy(logits.flatten(0, 1), data["input_ids"][:, 1:].reshape(-1),
                                    reduction="none").reshape_as(actual)
        mask = data["response_mask"].bool()
        torch.testing.assert_close(actual[mask], expected[mask], rtol=1e-5, atol=1e-6)
        a = torch.autograd.grad(actual.sum(), actor.head.weight, retain_graph=True)[0]
        b = torch.autograd.grad(expected[mask].sum(), actor.head.weight)[0]
        torch.testing.assert_close(a, b, rtol=0.02, atol=0.004)
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            generated, _ = actor(data["input_ids"], data["attention_mask"], generation=True)
        self.assertEqual(generated.dtype, torch.float32)

    def test_padding_and_gdn_batch_isolation(self):
        actor = Actor(copy.deepcopy(self.lm)).eval()
        batch = example_batch()
        data = collate(batch, [0, 1], torch.device("cpu"))
        with torch.no_grad():
            both = actor(**{k: data[k] for k in ("input_ids", "attention_mask", "response_mask")})
            for row, index in enumerate([0, 1]):
                single = collate(batch, [index], torch.device("cpu"))
                out = actor(**{k: single[k] for k in ("input_ids", "attention_mask", "response_mask")})
                torch.testing.assert_close(both[row, :out.shape[1]], out[0], atol=2e-6, rtol=1e-5)
            data["input_ids"][0] = 12
            changed = actor(**{k: data[k] for k in ("input_ids", "attention_mask", "response_mask")})
            torch.testing.assert_close(changed[1], both[1], atol=0, rtol=0)

    def test_cached_behavior_probability_matches_replay(self):
        actor = Actor(copy.deepcopy(self.lm)).eval()
        ids = torch.tensor([[2, 9, 21, 12, 87, 24]])
        with torch.no_grad():
            _, cache = actor(ids[:, :3], torch.ones_like(ids[:, :3]), generation=True)
            for end in range(4, ids.shape[1] + 1):
                cached, cache = actor(
                    ids[:, end - 1:end], torch.ones_like(ids[:, :end]),
                    generation=True, past_key_values=cache,
                )
                full, _ = actor(ids[:, :end], torch.ones_like(ids[:, :end]), generation=True)
                torch.testing.assert_close(
                    cached.log_softmax(-1), full.log_softmax(-1), rtol=1e-5, atol=2e-6,
                )

    def test_replanned_update_preserves_grpo_and_adam(self):
        for reduction in ("sequence_mean", "token_mean"):
            batch = replace(example_batch(), reduction=reduction)
            initial = Actor(copy.deepcopy(self.lm))
            results, states, moments = [], [], []
            for strategy, micro in (("fixed", 2), ("token", 1), ("verl", 3), ("slot_sorted", 1)):
                actor = copy.deepcopy(initial)
                optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-4, foreach=False)
                plan = plan_update(batch, 1, strategy=strategy, micro_batch=micro)
                results.append(execute_update(actor, optimizer, batch, plan, torch.device("cpu")))
                states.append([p.detach().clone() for p in actor.parameters()])
                moments.append([v["exp_avg"].clone() for v in optimizer.state.values()])
            for i in range(1, len(states)):
                self.assertAlmostEqual(results[0]["loss"], results[i]["loss"], places=6)
                for x, y in zip(states[0], states[i]):
                    torch.testing.assert_close(x, y, rtol=1e-4, atol=2e-6)
                for x, y in zip(moments[0], moments[i]):
                    torch.testing.assert_close(x, y, rtol=1e-4, atol=1e-7)

    def test_clipping_has_independent_reference(self):
        data = {
            "response_mask": torch.tensor([[1, 1, 0], [1, 0, 0]]),
            "advantages": torch.tensor([2.0, -3.0]),
            "old_log_probs": torch.zeros(2, 3),
        }
        log = torch.tensor([[0.5, -0.5, 1000.0], [0.5, 1000.0, 1000.0]], requires_grad=True)
        loss = surrogate_sum(log, data, "sequence_mean")
        expected = -(2 * 1.2 + 2 * torch.exp(torch.tensor(-0.5))) / 2 + 3 * torch.exp(torch.tensor(0.5))
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(torch.isfinite(log.grad).all())
        self.assertTrue((log.grad[data["response_mask"] == 0] == 0).all())

    def test_plan_and_provenance_rejections(self):
        batch = example_batch()
        for strategy in ("fixed", "token", "verl", "slot_sorted"):
            plan = plan_update(batch, 4, strategy=strategy)
            plan.validate(batch)
            with self.assertRaises(ValueError):
                replace(plan, slots=plan.slots + plan.slots).validate(batch)
        with self.assertRaises(CapacityError):
            plan_update(batch, 4, max_padded_tokens=8)
        with self.assertRaises(ValueError):
            replace(batch, trajectories=batch.trajectories[:-1])
        with self.assertRaises(ValueError):
            replace(batch, policy_version=1)
        with self.assertRaises(ValueError):
            replace(batch, trajectories=(batch.trajectories[0],) * 8)
        self.assertEqual(sum(batch.advantages()), 0.0)


if __name__ == "__main__":
    unittest.main()
