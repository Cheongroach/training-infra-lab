"""Regression: diagnostic fingerprints must not mutate optimizer state."""
import json
from unittest.mock import patch

import torch
from checkpointing import state_fingerprints

model = torch.nn.Linear(3, 2)
optimizer = torch.optim.AdamW(model.parameters())
model(torch.ones(1, 3)).sum().backward()
optimizer.step()
original = list(optimizer.state)
# Simulate FSDP exposing a temporary full view via the model while optimizer
# parameter groups retain the persistent shard.
model.weight = torch.nn.Parameter(model.weight.detach().clone())


def gather(result, item):
    result[0] = item


with patch("checkpointing.dist.get_world_size", return_value=1), patch(
    "checkpointing.dist.get_rank", return_value=0
), patch("checkpointing.dist.all_gather_object", side_effect=gather):
    before = state_fingerprints(model, optimizer)
    after = state_fingerprints(model, optimizer)
    assert before == after and list(optimizer.state) == original
    optimizer.state_dict()  # Raised KeyError with orphan entries.
    with torch.no_grad():
        optimizer.param_groups[0]["params"][0].add_(1)
    changed = state_fingerprints(model, optimizer)
    assert changed[0]["parameters"] != before[0]["parameters"]
print(json.dumps({"event": "fingerprint_read_only_gate_passed",
                  "transient_view": True, "detects_owned_parameter_change": True}))
