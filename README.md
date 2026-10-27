# Training Infra Lab Documentation

This is the documentation and evidence branch.

The repository uses two core branches:

- `main`: executable first-party code and tests.
- `doc`: tutorials, reports, diagnostic methods, figures, and curated evidence.

## Start here

- `training_infra_lab/一步步认识Training Infra项目-friendly.md`
- `training_infra_lab/diagnostics/README.md`
- `training_infra_lab/trajectory_runtime/measured-report-friendly.md`
- `training_infra_lab/posttraining_runtime/scale32/REPORT-friendly.md`
- `training_infra_lab/opd/OPD_DESIGN_AND_INTERVIEW-friendly.md`

## Diagnostic archive

`training_infra_lab/diagnostics/steps/` contains one file per independent diagnostic operation. Every file records:

1. input parameters;
2. processing steps;
3. output results;
4. final diagnostic conclusion;
5. underlying principle;
6. interpretation limits.

## Evidence policy

This branch preserves the original nested paths for source reports and curated structured evidence. Large or unsafe artifacts are represented by manifests and checksums rather than Git blobs.

See `MIGRATION_MANIFEST.md` for the exact inclusion and exclusion boundary.
