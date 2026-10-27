"""Check the portable loss against HF Qwen, including padding and recomputation."""

import copy
import json
import os

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from loss import SupervisedModel
from train import batch_at


def main():
    backend = os.environ.get("LAB_DEVICE", "cpu")
    if backend == "npu":
        import torch_npu  # noqa: F401
    device = torch.device(backend)
    torch.manual_seed(81)
    config = Qwen2Config(
        vocab_size=97, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        tie_word_embeddings=True, use_cache=False,
    )
    config._attn_implementation = "sdpa"
    reference = Qwen2ForCausalLM(config).to(device).train()
    records = []
    for length in (17, 68):
        ids = torch.randint(0, 97, (length,)).tolist()
        labels = ids.copy()
        labels[:8] = [-100] * 8
        records.append((ids, labels))
    fixed = batch_at(records, [0, 1], 256, 0, device, "fixed")
    expected = reference(**fixed).loss
    expected.backward()
    for mode, padding, activation in (
        ("full", "fixed", False),
        ("chunk", "fixed", False),
        ("full", "dynamic", False),
        ("selected", "dynamic", False),
        ("chunk", "dynamic", True),
    ):
        lm = copy.deepcopy(reference)
        lm.zero_grad(set_to_none=True)
        if activation:
            lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model = SupervisedModel(lm, mode, 7)
        batch = batch_at(records, [0, 1], 256, 0, device, padding)
        actual = model(**batch) / batch["labels"][:, 1:].ne(-100).sum()
        actual.backward()
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        for a, b in zip(lm.parameters(), reference.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=3e-4, atol=3e-6)
        print(json.dumps({
            "event": "qwen_hf_padding_recompute_gate_passed", "device": backend,
            "mode": mode, "padding": padding, "activation_checkpoint": activation,
            "loss": actual.item(), "shape": list(batch["input_ids"].shape),
        }), flush=True)


if __name__ == "__main__":
    main()
