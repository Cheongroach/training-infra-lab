"""Isolated Qwen3-sized BF16 projection; not a captured training kernel."""
import json
from pathlib import Path
import torch

torch.manual_seed(20261024)
x = torch.randn(256, 2560, device='cuda', dtype=torch.bfloat16)
w = torch.randn(9728, 2560, device='cuda', dtype=torch.bfloat16)
for _ in range(10):
    y = x @ w.T
torch.cuda.synchronize()
Path('/tmp/grpo-smoke/results/ncu/microbench.json').write_text(json.dumps({
    'kind': 'isolated BF16 projection', 'training_kernel': False,
    'x': list(x.shape), 'w': list(w.shape), 'y': list(y.shape),
    'dtype': str(y.dtype), 'device': torch.cuda.get_device_name(),
    'torch': torch.__version__, 'cuda': torch.version.cuda,
    'finite': bool(torch.isfinite(y).all().item()),
}, indent=2))
