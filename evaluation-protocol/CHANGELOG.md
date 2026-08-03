# CHANGELOG.md

Reverse-chronological session log for `evaluation-protocol/`.

Use one entry per meaningful session or task bundle. Record exact dates, meaningful behavior-level changes, verification status, blockers, and the next recommended actions.

## 2026-08-03

### Summary

Added AI-first handoff documentation for future sessions and recorded the current validated state of `evaluation-protocol/`.

### What Changed

- added `AGENTS.md` as the local AI handoff entrypoint for this subtree
- added this `CHANGELOG.md` as a reverse-chronological session log
- documented current supported workflows, operating rules, next steps, and known-good commands
- captured the current status of Efficient Probing, visualization, and smoke-test usage
- recorded a new future direction for representation visualization with `PCA`, `t-SNE`, or `UMAP` over per-frame backbone features from a chosen video directory, using either the CLS token or global average pooled patch tokens

### Verification

- documentation consistency checked against the current `README.md`
- local status reflected the user-reported result that the current Efficient Probing path passed in the local environment
- no new code-path test was run as part of this documentation-only session

Verification status: partial

### Known Blockers / Risks

- future sessions must keep `README.md`, `AGENTS.md`, and `CHANGELOG.md` aligned as behavior changes
- the handoff docs are only as accurate as the most recent session update discipline

### Next Recommended Actions

- update these two files after every material `evaluation-protocol/` change
- if new probe or backbone behavior is added, record the supported combinations and verification status here
- when this representation-visualization feature is implemented, keep it as a separate workflow and document its config surface, supported representation types, and verification status

## 2026-07-09

### Summary

Expanded `evaluation-protocol/` into a fuller evaluation workflow with checkpointed training, visualization support, README documentation, and Efficient Probing integration.

### What Changed

- added checkpoint saving controls and checkpoint manifests for probe training
- added combined prediction-vs-ground-truth visualization for train, valid, and test splits
- added mode-safe batching and smoke tests for the evaluation flow
- documented installation, training, visualization, pooling modes, and feature extraction in `README.md`
- added `pool: patch` support and the Efficient Probing probe path

### Verification

- smoke and functional validation were performed during the 2026-07-09 implementation sessions
- the current Efficient Probing path was later reported as passing by the user in the local environment

Verification status: partial

### Known Blockers / Risks

- Efficient Probing compatibility must stay aligned across backbone output, training, checkpoint reload, and visualization
- future changes to token shapes or probe configs should be covered by smoke tests immediately

### Next Recommended Actions

- preserve the per-group training and visualization contract unless there is a deliberate redesign
- extend tests first when changing probe/backbone contracts
