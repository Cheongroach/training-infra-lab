# Training Infra Lab

This is the executable-code branch.

The repository uses two core branches:

- `main`: first-party runtime, experiment builders, auditors, tests, Dockerfiles, and minimal runnable data.
- `doc`: tutorials, scientific reports, diagnostic step archives, figures, and curated structured evidence.

## Code layout

```text
training_infra_lab/
├── *.py
├── trajectory_runtime/
├── posttraining_runtime/
│   ├── scale32/
│   └── observability-20261024/scripts/
└── research/posttraining_20261022/
```

Large model weights, dependency caches, browser binaries, raw profiler traces, SQLite databases, and compressed run archives are excluded from Git history. Their immutable revisions or checksums are documented on the `doc` branch.

## Local tests

```bash
python3 -m unittest \
  training_infra_lab.trajectory_runtime.test_runtime \
  training_infra_lab.trajectory_runtime.test_state_cost \
  training_infra_lab.trajectory_runtime.test_grpo \
  training_infra_lab.trajectory_runtime.test_analysis \
  training_infra_lab.trajectory_runtime.test_training_audit -v
```

Some integration paths require pinned CUDA/CANN images, model assets, Merlin credentials, or cluster resources and are not expected to run in a plain CPU environment.
