"""Qwen3.5 text backbone and bounded-memory full-vocabulary log-probs."""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@contextmanager
def kernel_construction(kind):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m

    if kind not in ("torch", "fla"):
        raise ValueError(kind)
    names = ("FusedRMSNormGated", "causal_conv1d_fn", "causal_conv1d_update",
             "chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule")
    before = {name: getattr(m, name) for name in names}
    try:
        # The 5.2 fused norm constructor unconditionally allocates on CUDA.
        # Keep the native norm/conv on both devices; select the GDN separately.
        m.FusedRMSNormGated = None
        m.causal_conv1d_fn = m.causal_conv1d_update = None
        if kind == "torch":
            m.chunk_gated_delta_rule = m.fused_recurrent_gated_delta_rule = None
        else:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
            m.chunk_gated_delta_rule = chunk_gated_delta_rule
            m.fused_recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule
        yield
    finally:
        for name, value in before.items():
            setattr(m, name, value)


def make_lm(path=None, *, kernel="torch", tiny=False, dtype=torch.float32):
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    if tiny:
        config = Qwen3_5TextConfig(
            vocab_size=128, hidden_size=64, intermediate_size=128,
            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, linear_key_head_dim=16, linear_value_head_dim=16,
            linear_num_key_heads=2, linear_num_value_heads=4,
            layer_types=["linear_attention"] * 3 + ["full_attention"],
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                             "partial_rotary_factor": 0.5, "mrope_section": [2, 1, 1]},
        )
        config._attn_implementation = "sdpa"
    else:
        config = Qwen3_5TextConfig.from_pretrained(path, local_files_only=True)
    config.use_cache = False
    config.attention_dropout = 0.0
    with kernel_construction(kernel):
        if tiny:
            lm = Qwen3_5ForCausalLM(config).to(dtype=dtype)
        else:
            lm, info = Qwen3_5ForCausalLM.from_pretrained(
                path, config=config, local_files_only=True, trust_remote_code=False,
                dtype=dtype, attn_implementation="sdpa", output_loading_info=True,
            )
            errors = {key: sorted(value) if isinstance(value, set) else value
                      for key, value in info.items() if value}
            if errors:
                raise RuntimeError(f"Inexact text weight load: {errors}")
    return lm


class Actor(torch.nn.Module):
    def __init__(self, lm, chunk_tokens=128, fp32_head=False):
        super().__init__()
        self.backbone = lm.model
        self.head = lm.lm_head
        self.chunk_tokens = chunk_tokens
        self.fp32_head = fp32_head
        if chunk_tokens < 1:
            raise ValueError("chunk_tokens must be positive")

    def forward(self, input_ids, attention_mask, response_mask=None, *,
                generation=False, past_key_values=None):
        positions = (attention_mask.cumsum(-1) - 1).clamp_min(0)
        if past_key_values is not None:
            positions = positions[:, -input_ids.shape[1]:]
        output = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask,
            position_ids=positions, past_key_values=past_key_values,
            use_cache=generation, return_dict=True,
        )
        if generation:
            if self.fp32_head:
                with torch.autocast(input_ids.device.type, enabled=False):
                    logits = F.linear(output.last_hidden_state[:, -1].float(), self.head.weight.float())
                return logits, output.past_key_values
            return self.head(output.last_hidden_state[:, -1]), output.past_key_values
        hidden = output.last_hidden_state[:, :-1]
        selected = response_mask.bool()
        vectors = hidden[selected]
        targets = input_ids[:, 1:][selected]
        if vectors.shape[0] == 0:
            raise ValueError("Empty responses must use the executor's valid zero-weight dummy")
        values = []

        def log_probs(h, w, y):
            # Compute the complete vocabulary normalization, not a restricted head.
            if self.fp32_head:
                with torch.autocast(h.device.type, enabled=False):
                    return -F.cross_entropy(F.linear(h.float(), w.float()), y, reduction="none")
            return -F.cross_entropy(F.linear(h, w).float(), y, reduction="none")

        for begin in range(0, vectors.shape[0], self.chunk_tokens):
            h, y = vectors[begin:begin + self.chunk_tokens], targets[begin:begin + self.chunk_tokens]
            if torch.is_grad_enabled():
                value = checkpoint(log_probs, h, self.head.weight, y, use_reentrant=False)
            else:
                value = log_probs(h, self.head.weight, y)
            values.append(value)
        return torch.zeros_like(response_mask, dtype=torch.float32).masked_scatter(
            selected, torch.cat(values),
        )


def kernel_evidence(actor):
    paths = []
    for layer in actor.backbone.layers:
        if layer.layer_type == "linear_attention":
            block = layer.linear_attn
            paths.append({
                "gdn": block.chunk_gated_delta_rule.__module__ + "." + block.chunk_gated_delta_rule.__name__,
                "conv": "torch.nn.Conv1d" if block.causal_conv1d_fn is None else str(block.causal_conv1d_fn),
                "norm": type(block.norm).__name__,
            })
    return paths
