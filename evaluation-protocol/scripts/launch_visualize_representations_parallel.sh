#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROBE_EVAL_PROJECT_ROOT:-${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}}"
ENTRYPOINT="visualize_representations.py"
LAUNCH_SCRIPT="${PROJECT_ROOT}/scripts/launch_visualize_representations_parallel.sh"
WORKFLOW_NAME="visualize_representations"
JOB_PREFIX="probe_repr_vis"
VIDEO_DIR="${VIDEO_DIR:-/shared/results/common/kargin/unreal_engine/dataset/probe-equivariance/FirstPersonMap/cube/camera_line}"
REPRESENTATION="${REPRESENTATION:-cls}"
REDUCTION_METHOD="${REDUCTION_METHOD:-pca}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results/representation_visualizations}"

handle_workflow_option() {
  PARSE_CONSUMED=0
  case "$1" in
    --video-dir)
      [[ $# -ge 2 ]] || { echo "--video-dir requires a value" >&2; exit 2; }
      VIDEO_DIR="$2"
      PARSE_CONSUMED=2
      return 0
      ;;
    --representation)
      [[ $# -ge 2 ]] || { echo "--representation requires a value" >&2; exit 2; }
      REPRESENTATION="$2"
      PARSE_CONSUMED=2
      return 0
      ;;
    --reduction)
      [[ $# -ge 2 ]] || { echo "--reduction requires a value" >&2; exit 2; }
      REDUCTION_METHOD="$2"
      PARSE_CONSUMED=2
      return 0
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || { echo "--output-dir requires a value" >&2; exit 2; }
      OUTPUT_DIR="$2"
      PARSE_CONSUMED=2
      return 0
      ;;
  esac
  return 1
}

build_run_specs() {
  case "$REPRESENTATION" in
    cls|patch_mean) ;;
    *) echo "Invalid --representation value: '$REPRESENTATION' (expected cls|patch_mean)" >&2; exit 2 ;;
  esac
  case "$REDUCTION_METHOD" in
    pca|tsne|umap) ;;
    *) echo "Invalid --reduction value: '$REDUCTION_METHOD' (expected pca|tsne|umap)" >&2; exit 2 ;;
  esac
  RUN_NAMES=()
  RUN_ARG_STRINGS=()
  for backbone in "${BACKBONE_CONFIGS[@]}"; do
    run_name="${backbone}_${REPRESENTATION}_${REDUCTION_METHOD}"
    add_run "$run_name" \
      "backbone=${backbone}" \
      "video_dir=${VIDEO_DIR}" \
      "representation=${REPRESENTATION}" \
      "reduction.method=${REDUCTION_METHOD}" \
      "output_dir=${OUTPUT_DIR}" \
      "${EXTRA_ARGS[@]}"
  done
}

print_workflow_help() {
  cat <<'USAGE'
Usage:
  bash scripts/launch_visualize_representations_parallel.sh [options] [Hydra overrides...]

Schedules one representation-visualization run for every config in
configs/backbone/.

Options:
  --backend local|slurm       Execution backend (default: slurm)
  --dry-run                   Print all commands without executing them
  --max-concurrent N          Slurm array cap / local process cap (default: 4)
  --backbone NAME             Select one backbone; repeat to select several
  --video-dir PATH            One motion directory to visualize
  --representation NAME       cls or patch_mean (default: cls)
  --reduction NAME            pca, tsne, or umap (default: pca)
  --output-dir PATH           Representation output root
  --help                     Show this help

Slurm environment overrides match the reference launcher:
  SLURM_PARTITION, SLURM_QOS, SLURM_ACCOUNT, SLURM_GPUS, SLURM_CPUS,
  SLURM_MEM, SLURM_TIME, SLURM_NTASKS, SLURM_EXCLUDE

Example:
  bash scripts/launch_visualize_representations_parallel.sh --dry-run \
    --video-dir /shared/results/common/kargin/unreal_engine/dataset/probe-equivariance/FirstPersonMap/cube/camera_line \
    --representation patch_mean --reduction pca
USAGE
}

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/launch_array_common.sh"
launch_array_workflow "$@"
