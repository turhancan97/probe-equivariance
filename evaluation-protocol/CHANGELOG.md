# CHANGELOG.md

Reverse-chronological session log for `evaluation-protocol/`.

Use one entry per meaningful session or task bundle. Record exact dates, meaningful behavior-level changes, verification status, blockers, and the next recommended actions.

## 2026-08-04

### Summary

Added separate capped-array launchers plus combined metrics aggregation and backbone comparison plotting.

### Follow-up

- added repeated `--backbone NAME` selection to all three launchers
- omitted `--backbone` now preserves the all-backbone default
- training probe selection now reads each backbone config's actual `pool` value
- fixed Slurm array workers to source the shared launcher helper from the exported repository root instead of `/var/spool/slurmd/`
- added `aggregate_backbone_metrics.py` for one combined object-level metrics CSV
- updated `plot_backbone_metrics.py` to plot validation/test RMSE only, rank backbones best-to-worst, add size/family comparison views, and separate every pool mode
- included `pool` in aggregation deduplication keys so mean/cls/patch histories remain independent

### What Changed

- added independent launchers for `train_equivariance.py`, `visualize_equivariance.py`, and `visualize_representations.py`
- added a shared array engine with local execution, Slurm submission, dry-run output, per-task logs, and trailing Hydra override support
- enumerated all backbone configs automatically
- selected the probe from each backbone config's `pool` value and enabled checkpoint saving for training runs
- capped Slurm arrays with `--array=...%N` so each task requests one GPU without scheduling the full sweep concurrently
- documented reference-matching Slurm resource environment overrides

### Verification

- launcher scripts pass shell syntax validation
- dry-run output, default all-backbone expansion, single and repeated `--backbone` selection, invalid-selector rejection, local capped placeholder execution, array worker selection, and generated `--array=0-22%4` resource commands were checked locally
- aggregation and plotting smoke tests cover normalized CSV output and environment/mode comparison plots
- full Slurm submission was not run because this environment is not a Slurm controller

Verification status: partial

### Known Blockers / Risks

- actual `sbatch` behavior, cluster-specific resource names, and node exclusions must be validated on the target HPC system

### Next Recommended Actions

- run each launcher with `--dry-run` on the HPC login node, then submit a small capped training array before launching visualization arrays

## 2026-08-03

### Summary

Implemented the standalone per-frame representation visualization workflow.

### What Changed

- added `visualize_representations.py` and its Hydra configuration
- added pose-JSON-ordered frame loading for one motion directory
- added CLS and global patch-mean representations with native timm preprocessing
- added PCA, seeded t-SNE, and optional UMAP reduction
- added temporal plots, per-frame CSV output, and resolved config metadata
- added focused smoke tests and `scikit-learn` to the required dependencies
- documented the workflow and optional `umap-learn` installation

### Verification

- representation smoke tests cover frame ordering, both representation mappings, PCA outputs, seeded t-SNE, output files, invalid metadata, and missing UMAP
- `py_compile`, Hydra help/config composition, focused representation tests, existing evaluation smoke tests, and combined test discovery passed in `dinov3`

Verification status: passed

### Known Blockers / Risks

- UMAP execution requires installing the optional `umap-learn` package in the active environment

### Next Recommended Actions

- run the workflow against a real shared motion directory with the selected backbone

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
