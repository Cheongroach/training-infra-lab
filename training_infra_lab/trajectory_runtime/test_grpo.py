"""Contracts specific to real sampling and scratch replay, independent of GPU timings."""
import tempfile
import unittest
from pathlib import Path

import torch

from .contracts import UpdateBatch
from .grpo import scratch_save, scratch_restore, scratch_compare
from .rollout import sample_group


class FakeTokenizer:
    eos_token_id = 2

    def apply_chat_template(self, *args, **kwargs):
        return [5, 6, 7]

    def decode(self, tokens, **kwargs):
        return "Final answer: 42" if tokens[-1] == 2 else "Final answer: 0"


class FakeActor(torch.nn.Module):
    def forward(self, ids, attention, **kwargs):
        step = attention.shape[1] - 3
        logits = torch.full((2, 10), -100.0)
        logits[0, 2] = 100.0
        logits[1, 8 if step < 2 else 2] = 100.0
        return logits, object()


class GRPOTests(unittest.TestCase):
    def test_real_fast_tokenizer_chat_template_contract(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast

        vocabulary = {word: i for i, word in enumerate((
            "<unk>", "<pad>", "<eos>", "x", "y", "fixture", "prompt", "answer", "z", "w",
        ))}
        backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>",
            chat_template="fixture prompt answer",
        )
        trajectories, _ = sample_group(
            FakeActor(), tokenizer, {"question": "fixture", "answer": "#### 42"},
            prompt_id="tokenizer-api", policy_version=0, group_size=2,
            device=torch.device("cpu"), max_new_tokens=1, autocast_dtype=None,
        )
        self.assertEqual(len(trajectories), 2)
        self.assertTrue(all(t.input_ids[:3] == (5, 6, 7) for t in trajectories))
        self.assertTrue(all(len(t.input_ids) == 4 for t in trajectories))

    def test_eos_masks_and_finished_row_waste(self):
        actor = FakeActor()
        trajectories, details = sample_group(
            actor, FakeTokenizer(), {"question": "fixture", "answer": "#### 42"},
            prompt_id="fixture", policy_version=0, group_size=2,
            device=torch.device("cpu"), max_new_tokens=5, autocast_dtype=None,
        )
        self.assertTrue(actor.training)
        self.assertEqual(details["decode_steps"], 3)
        self.assertEqual(details["wasted_decode_rows"], 2)
        self.assertEqual([sum(t.response_mask) for t in trajectories], [1, 3])
        self.assertEqual([t.reward for t in trajectories], [1, 1])
        self.assertEqual(UpdateBatch(tuple(trajectories), 0, 0, 2).advantages(), (0, 0))
        self.assertTrue(all(x == 0 for t in trajectories for x in t.old_log_probs))

    def test_length_limit_preserves_real_zero_reward(self):
        trajectories, details = sample_group(
            FakeActor(), FakeTokenizer(), {"question": "fixture", "answer": "#### 42"},
            prompt_id="fixture", policy_version=3, group_size=2,
            device=torch.device("cpu"), max_new_tokens=1, autocast_dtype=None,
        )
        self.assertEqual([t.reward for t in trajectories], [1, 0])
        self.assertEqual([t["finish_reason"] for t in details["trajectories"]], ["eos", "length"])
        self.assertEqual(trajectories[1].policy_version, 3)

    def test_truncation_cannot_earn_reward_from_an_accidental_number(self):
        class AccidentalTokenizer(FakeTokenizer):
            def decode(self, tokens, **kwargs):
                return "Final answer: 42"

        trajectories, details = sample_group(
            FakeActor(), AccidentalTokenizer(), {"question": "fixture", "answer": "#### 42"},
            prompt_id="truncation", policy_version=0, group_size=2,
            device=torch.device("cpu"), max_new_tokens=1, autocast_dtype=None,
        )
        self.assertEqual([t.reward for t in trajectories], [1, 0])
        self.assertEqual([r["reward_outcome"] for r in details["trajectories"]], ["correct", "truncated"])

    def test_scratch_restores_nonempty_adam_then_replays_exactly(self):
        torch.manual_seed(93)
        actor = torch.nn.Linear(4, 3)
        optimizer = torch.optim.AdamW(actor.parameters(), lr=0.001, foreach=False)
        inputs = torch.randn(2, 4)

        def step():
            optimizer.zero_grad(set_to_none=True)
            actor(inputs).square().mean().backward()
            optimizer.step()

        step()
        with tempfile.TemporaryDirectory() as directory:
            base, expected = Path(directory) / "base", Path(directory) / "expected"
            scratch_save(base, actor, optimizer)
            step()
            scratch_save(expected, actor, optimizer, gradients=True)
            for _ in range(3):
                scratch_restore(base, actor, optimizer)
                step()
                errors = scratch_compare(expected, actor, optimizer)
                self.assertTrue(all(v["max_abs"] == 0 for v in errors.values()))


if __name__ == "__main__":
    unittest.main()
