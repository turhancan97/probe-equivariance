#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROBE_EVAL_PROJECT_ROOT:-${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}}"
ENTRYPOINT="visualize_equivariance.py"
LAUNCH_SCRIPT="${PROJECT_ROOT}/scripts/launch_visualize_equivariance_parallel.sh"
WORKFLOW_NAME="visualize_equivariance"
JOB_PREFIX="probe_eq_vis"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/results}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-parallel}"

handle_workflow_option() {
  PARSE_CONSUMED=0
  case "$1" in
    --result-root)
      [[ $# -ge 2 ]] || { echo "--result-root requires a value" >&2; exit 2; }
      RESULT_ROOT="$2"
      PARSE_CONSUMED=2
      return 0
      ;;
    --experiment-prefix)
      [[ $# -ge 2 ]] || { echo "--experiment-prefix requires a value" >&2; exit 2; }
      EXPERIMENT_PREFIX="$2"
      PARSE_CONSUMED=2
      return 0
      ;;
  esac
  return 1
}

build_run_specs() {
  RUN_NAMES=()
  RUN_ARG_STRINGS=()
  for backbone in "${BACKBONE_CONFIGS[@]}"; do
    run_name="${EXPERIMENT_PREFIX}_${backbone}"
    result_dir="${RESULT_ROOT}/equivariance_${run_name}"
    add_run "$run_name" \
      "result_dir=${result_dir}" \
      "${EXTRA_ARGS[@]}"
  done
}

print_workflow_help() {
  cat <<'USAGE'
Usage:
  bash scripts/launch_visualize_equivariance_parallel.sh [options] [Hydra overrides...]

Schedules one visualization run for every config in configs/backbone/. It
expects training outputs named results/equivariance_<prefix>_<backbone> unless
overridden.

Options:
  --backend local|slurm       Execution backend (default: slurm)
  --dry-run                   Print all commands without executing them
  --max-concurrent N          Slurm array cap / local process cap (default: 4)
  --backbone NAME             Select one backbone; repeat to select several
  --result-root PATH          Root containing equivariance_* results
  --experiment-prefix NAME    Training experiment prefix (default: parallel)
  --help                     Show this help

Slurm environment overrides match the reference launcher:
  SLURM_PARTITION, SLURM_QOS, SLURM_ACCOUNT, SLURM_GPUS, SLURM_CPUS,
  SLURM_MEM, SLURM_TIME, SLURM_NTASKS, SLURM_EXCLUDE

Example:
  bash scripts/launch_visualize_equivariance_parallel.sh --dry-run \
    --result-root /shared/results/probe-equivariance/results
USAGE
}

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/launch_array_common.sh"
launch_array_workflow "$@"
