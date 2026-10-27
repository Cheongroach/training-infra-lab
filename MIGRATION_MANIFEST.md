# Documentation Branch Migration Boundary

## Included

- all first-party Markdown documents outside vendored upstream and dependency caches;
- the progressive tutorial and 14 independent diagnostic-step documents;
- report figures in PNG or SVG form;
- small CSV tables used by reports;
- curated JSON evidence with names such as `verified`, `summary`, `audit`, `diagnosis`, `diagnostics`, `cost`, `status`, `manifest`, `inventory`, and `validation`;
- nested paths relative to the original `training_infra_lab/` directory.

## Excluded

| Category | Reason |
|---|---|
| Qwen `.safetensors` and tokenizer caches | 23+ GiB; immutable upstream assets, not project source |
| `.validation/` browsers, fonts, wheels, and system packages | generated dependency cache |
| vendored framework snapshots under `research/.../upstream` | third-party source; upstream revision is documented |
| `.pt` tensors and raw checkpoints | unsafe/binary runtime artifacts |
| raw NSYS/NCU/SQLite/trace files | large profiler artifacts not suitable for ordinary Git |
| offline HTML metric data chunks | contain internal hostnames, process commands, and runtime topology |
| `.tar.gz` and `.zip` run archives | duplicated compressed evidence |
| signed log URL JSON | temporary credentials |
| raw create/dry-run/platform responses | may contain internal environment or transient service metadata |
| full stdout/stderr and environment dumps | noisy and may expose internal infrastructure |

## Integrity

The documentation records immutable model/framework revisions, job identities, run-level checksums, archive checksums, and selected source manifests. Exclusion from Git does not turn an unarchived binary into a claimed result: reports explicitly state when raw artifacts remain local-only.

## Branch separation

The `doc` branch intentionally does not contain executable Python runtime code. The same relative directory hierarchy is retained for documentation and curated evidence so links remain traceable to the corresponding code path on `main`.
