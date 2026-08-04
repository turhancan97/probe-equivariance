#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROBE_EVAL_PROJECT_ROOT:-${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}}"
ENTRYPOINT="train_equivariance.py"
LAUNCH_SCRIPT="${PROJECT_ROOT}/scripts/launch_train_equivariance_parallel.sh"
WORKFLOW_NAME="train_equivariance"
JOB_PREFIX="probe_eq_train"

effective_backbone_pool() {
  local backbone="$1"
  local override=""
  local arg
  for arg in "${EXTRA_ARGS[@]}"; do
    case "$arg" in
      backbone.pool=*) override="${arg#backbone.pool=}" ;;
    esac
  done
  if [[ -n "$override" ]]; then
    printf '%s\n' "$override"
  else
    backbone_pool "$backbone"
  fi
}

build_run_specs() {
  RUN_NAMES=()
  RUN_ARG_STRINGS=()
  for backbone in "${BACKBONE_CONFIGS[@]}"; do
    pool="$(effective_backbone_pool "$backbone")"
    case "$pool" in
      patch) probe="efficient_probing" ;;
      mean|cls) probe="regressor" ;;
      *) echo "Unsupported pool '$pool' in configs/backbone/${backbone}.yaml" >&2; exit 2 ;;
    esac
    run_name="parallel_${backbone}_${probe}"
    add_run "$run_name" \
      "${EXTRA_ARGS[@]}" \
      "backbone=${backbone}" \
      "probe=${probe}" \
      "experiment_name=${run_name}" \
      "experiment_model=${backbone}" \
      "training.save_checkpoints=true"
  done
}

handle_workflow_option() {
  PARSE_CONSUMED=0
  return 1
}

print_workflow_help() {
  cat <<'USAGE'
Usage:
  bash scripts/launch_train_equivariance_parallel.sh [options] [Hydra overrides...]

Schedules one run for every config in configs/backbone/. The launcher selects
probe=efficient_probing for configs with pool=patch and probe=regressor for
configs with pool=mean or pool=cls. A trailing backbone.pool=... Hydra
override also controls this probe selection.

Options:
  --backend local|slurm       Execution backend (default: slurm)
  --dry-run                   Print all commands without executing them
  --max-concurrent N          Slurm array cap / local process cap (default: 4)
  --backbone NAME             Select one backbone; repeat to select several
  --help                     Show this help

Slurm environment overrides match the reference launcher:
  SLURM_PARTITION, SLURM_QOS, SLURM_ACCOUNT, SLURM_GPUS, SLURM_CPUS,
  SLURM_MEM, SLURM_TIME, SLURM_NTASKS, SLURM_EXCLUDE

Other environment overrides:
  CONDA_ENV (default: dinov3), CONDA_BIN (default: conda),
  PYTHON_BIN (default: python), LOG_DIR (default: evaluation-protocol/logs)

Example:
  bash scripts/launch_train_equivariance_parallel.sh --dry-run \
    dataset.root=/shared/results/common/kargin/unreal_engine/dataset/probe-equivariance
USAGE
}

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/launch_array_common.sh"
launch_array_workflow "$@"
