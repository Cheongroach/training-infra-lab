# Main Branch Scope

The `main` branch contains 128 first-party executable or test files copied from the original workspace while preserving their relative paths.

Included:

- legacy FSDP2 SFT training and correctness tools;
- Qwen3.5 trajectory runtime, analyzers, and unit tests;
- post-training qualification, Ray readiness, recovery, and verification tools;
- 32-GPU experiment builders, collectors, auditors, and publication benchmark;
- observability collection and profiling scripts;
- Dockerfiles, source patches, and minimal fixed test input.

Excluded from Git:

- Qwen model weights and tokenizer caches;
- vendored upstream framework snapshots;
- Python/browser dependency caches;
- generated worker archives and duplicated source snapshots;
- raw `.pt`, NSYS, NCU, SQLite, trace, and compressed artifacts;
- temporary signed log URLs and credential-bearing environment dumps.

The `doc` branch records the migration boundary and curated evidence inventory.
