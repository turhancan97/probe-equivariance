#!/usr/bin/env bash

# Shared array-launching engine. Workflow-specific launchers define:
#   build_run_specs
#   handle_workflow_option (optional)
#   print_workflow_help
# and set WORKFLOW_NAME, JOB_PREFIX, ENTRYPOINT, and LAUNCH_SCRIPT.

set -euo pipefail

BACKEND="${BACKEND:-slurm}"
DRY_RUN=0
WORKER=0
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
CONDA_ENV="${CONDA_ENV:-dinov3}"
CONDA_BIN="${CONDA_BIN:-conda}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"

SLURM_PARTITION="${SLURM_PARTITION:-}"
SLURM_QOS="${SLURM_QOS:-}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
SLURM_GPUS="${SLURM_GPUS:-1}"
SLURM_CPUS="${SLURM_CPUS:-10}"
SLURM_MEM="${SLURM_MEM:-64G}"
SLURM_TIME="${SLURM_TIME:-24:00:00}"
SLURM_NTASKS="${SLURM_NTASKS:-1}"
SLURM_EXCLUDE="${SLURM_EXCLUDE:-c22,c11,c12,c13,c17}"

EXTRA_ARGS=()
SELECTED_BACKBONES=()
RUN_NAMES=()
RUN_ARG_STRINGS=()
LOCAL_PIDS=()
LOCAL_NAMES=()
SUBMITTED=0
SUBMIT_FAIL=0
LOCAL_FAIL=0
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

discover_backbones() {
  local config_dir="${PROJECT_ROOT}/configs/backbone"
  if [[ ! -d "$config_dir" ]]; then
    echo "Backbone config directory not found: $config_dir" >&2
    exit 2
  fi
  mapfile -t BACKBONE_CONFIGS < <(
    find "$config_dir" -maxdepth 1 -type f -name '*.yaml' -printf '%f\n' \
      | sed 's/\.yaml$//' \
      | sort
  )
  if [[ "${#BACKBONE_CONFIGS[@]}" -eq 0 ]]; then
    echo "No backbone configs found in $config_dir" >&2
    exit 2
  fi

  if [[ "${#SELECTED_BACKBONES[@]}" -gt 0 ]]; then
    local requested
    local found
    local -a filtered=()
    for requested in "${SELECTED_BACKBONES[@]}"; do
      found=0
      for backbone in "${BACKBONE_CONFIGS[@]}"; do
        if [[ "$backbone" == "$requested" ]]; then
          filtered+=("$backbone")
          found=1
          break
        fi
      done
      if [[ "$found" -eq 0 ]]; then
        echo "Unknown backbone '$requested'. Available configs: ${BACKBONE_CONFIGS[*]}" >&2
        exit 2
      fi
    done
    BACKBONE_CONFIGS=("${filtered[@]}")
  fi
}

backbone_pool() {
  local backbone="$1"
  local config_path="${PROJECT_ROOT}/configs/backbone/${backbone}.yaml"
  local pool
  pool="$(sed -nE 's/^pool:[[:space:]]*([^[:space:]#]+).*/\1/p' "$config_path" | head -n 1)"
  if [[ -z "$pool" ]]; then
    echo "Could not determine pool from backbone config: $config_path" >&2
    exit 2
  fi
  printf '%s\n' "$pool"
}

add_run() {
  local run_name="$1"
  shift
  local rendered
  printf -v rendered '%q ' "$@"
  RUN_NAMES+=("$run_name")
  RUN_ARG_STRINGS+=("${rendered% }")
}

parse_args() {
  PARSE_CONSUMED=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --backend)
        [[ $# -ge 2 ]] || { echo "--backend requires a value" >&2; exit 2; }
        BACKEND="$2"
        shift 2
        ;;
      --dry-run)
        DRY_RUN=1
        shift
        ;;
      --max-concurrent)
        [[ $# -ge 2 ]] || { echo "--max-concurrent requires a value" >&2; exit 2; }
        MAX_CONCURRENT="$2"
        shift 2
        ;;
      --backbone)
        [[ $# -ge 2 ]] || { echo "--backbone requires a value" >&2; exit 2; }
        SELECTED_BACKBONES+=("$2")
        shift 2
        ;;
      --worker)
        WORKER=1
        shift
        ;;
      --help|-h)
        print_workflow_help
        exit 0
        ;;
      --)
        shift
        EXTRA_ARGS+=("$@")
        break
        ;;
      *)
        if declare -F handle_workflow_option >/dev/null 2>&1 && handle_workflow_option "$@"; then
          shift "$PARSE_CONSUMED"
        else
          EXTRA_ARGS+=("$1")
          shift
        fi
        ;;
    esac
  done
}

validate_common_args() {
  if [[ "$BACKEND" != "local" && "$BACKEND" != "slurm" ]]; then
    echo "Invalid --backend value: '$BACKEND' (expected local|slurm)" >&2
    exit 2
  fi
  if ! [[ "$MAX_CONCURRENT" =~ ^[0-9]+$ ]] || [[ "$MAX_CONCURRENT" -le 0 ]]; then
    echo "Invalid --max-concurrent value: '$MAX_CONCURRENT' (expected positive integer)" >&2
    exit 2
  fi
}

load_run_args() {
  local index="$1"
  RUN_ARGS=()
  # RUN_ARG_STRINGS is generated with printf %q, so this reconstructs exact
  # argument boundaries, including spaces and shell metacharacters.
  eval "RUN_ARGS=( ${RUN_ARG_STRINGS[$index]} )"
}

build_command() {
  local index="$1"
  load_run_args "$index"
  COMMAND=(
    "$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" "$PYTHON_BIN"
    "${PROJECT_ROOT}/${ENTRYPOINT}"
    "${RUN_ARGS[@]}"
  )
}

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

run_local_index() {
  local index="$1"
  local run_name="${RUN_NAMES[$index]}"
  local log_path="${LOG_DIR}/${run_name}_${TIMESTAMP}.log"
  build_command "$index"
  {
    echo "=== START $(date -Iseconds) ==="
    echo "workflow=${WORKFLOW_NAME}"
    echo "run=${run_name}"
    echo -n "command="
    print_command "${COMMAND[@]}"
    set +e
    "${COMMAND[@]}"
    local rc=$?
    set -e
    echo "=== END $(date -Iseconds) rc=${rc} ==="
    exit "$rc"
  } >"$log_path" 2>&1 &
  LOCAL_PIDS+=("$!")
  LOCAL_NAMES+=("$run_name")
  echo "[LOCAL] ${run_name} -> ${log_path}"
}

wait_for_local_slot() {
  while [[ "$(jobs -rp | wc -l)" -ge "$MAX_CONCURRENT" ]]; do
    sleep 2
  done
}

run_slurm_worker() {
  if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "--worker requires SLURM_ARRAY_TASK_ID" >&2
    exit 2
  fi
  local index="$SLURM_ARRAY_TASK_ID"
  if ! [[ "$index" =~ ^[0-9]+$ ]] || [[ "$index" -ge "${#RUN_NAMES[@]}" ]]; then
    echo "Invalid SLURM_ARRAY_TASK_ID=$index for ${#RUN_NAMES[@]} runs" >&2
    exit 2
  fi
  local run_name="${RUN_NAMES[$index]}"
  build_command "$index"
  echo "[SLURM ARRAY] workflow=${WORKFLOW_NAME} task=${index} run=${run_name}"
  echo -n "[SLURM ARRAY] command="
  print_command "${COMMAND[@]}"
  exec "${COMMAND[@]}"
}

submit_slurm_array() {
  command -v "$SBATCH_BIN" >/dev/null 2>&1 || { echo "Slurm submit command not found: $SBATCH_BIN" >&2; exit 2; }
  mkdir -p "$LOG_DIR"

  local last_index=$(( ${#RUN_NAMES[@]} - 1 ))
  local output_path="${LOG_DIR}/${JOB_PREFIX}_%A_%a_${TIMESTAMP}.log"
  local -a sbatch_cmd=(
    "$SBATCH_BIN"
    "--array=0-${last_index}%${MAX_CONCURRENT}"
    "--gpus=${SLURM_GPUS}"
    "--cpus-per-task=${SLURM_CPUS}"
    "--mem=${SLURM_MEM}"
    "--ntasks=${SLURM_NTASKS}"
    "--time=${SLURM_TIME}"
    "--job-name=${JOB_PREFIX}"
    "--output=${output_path}"
    "--chdir=${PROJECT_ROOT}"
  )
  [[ -n "$SLURM_EXCLUDE" ]] && sbatch_cmd+=("--exclude=${SLURM_EXCLUDE}")
  [[ -n "$SLURM_PARTITION" ]] && sbatch_cmd+=("--partition=${SLURM_PARTITION}")
  [[ -n "$SLURM_QOS" ]] && sbatch_cmd+=("--qos=${SLURM_QOS}")
  [[ -n "$SLURM_ACCOUNT" ]] && sbatch_cmd+=("--account=${SLURM_ACCOUNT}")
  sbatch_cmd+=("--export=ALL,PROBE_EVAL_PROJECT_ROOT=${PROJECT_ROOT}")
  sbatch_cmd+=("$LAUNCH_SCRIPT" --worker "${ORIGINAL_ARGS[@]}")

  echo "[SLURM] submitting ${#RUN_NAMES[@]} runs as one capped array"
  echo -n "[SLURM] command="
  print_command "${sbatch_cmd[@]}"
  if "${sbatch_cmd[@]}"; then
    SUBMITTED=1
  else
    SUBMIT_FAIL=1
    echo "[WARN] Slurm array submission failed" >&2
  fi
}

print_dry_run() {
  echo "[DRY-RUN] workflow=${WORKFLOW_NAME} runs=${#RUN_NAMES[@]} backend=${BACKEND} max_concurrent=${MAX_CONCURRENT}"
  for ((index = 0; index < ${#RUN_NAMES[@]}; index++)); do
    build_command "$index"
    echo "[DRY-RUN] task=${index} run=${RUN_NAMES[$index]}"
    echo -n "  command="
    print_command "${COMMAND[@]}"
  done
  echo "[DRY-RUN] Slurm resources: gpus=${SLURM_GPUS} cpus=${SLURM_CPUS} mem=${SLURM_MEM} time=${SLURM_TIME} ntasks=${SLURM_NTASKS} partition=${SLURM_PARTITION:-default} qos=${SLURM_QOS:-default} account=${SLURM_ACCOUNT:-default} exclude=${SLURM_EXCLUDE:-none}"
}

launch_array_workflow() {
  ORIGINAL_ARGS=("$@")
  parse_args "$@"
  validate_common_args
  discover_backbones
  build_run_specs

  if [[ "${#RUN_NAMES[@]}" -eq 0 ]]; then
    echo "No runs were generated" >&2
    exit 2
  fi

  # A worker must execute even if the parent invocation included --dry-run.
  if [[ "$WORKER" -eq 1 ]]; then
    DRY_RUN=0
    run_slurm_worker
    return 0
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    print_dry_run
    return 0
  fi

  if [[ "$BACKEND" == "slurm" ]]; then
    submit_slurm_array
  else
    mkdir -p "$LOG_DIR"
    for ((index = 0; index < ${#RUN_NAMES[@]}; index++)); do
      wait_for_local_slot
      run_local_index "$index"
    done
    for pid in "${LOCAL_PIDS[@]}"; do
      if ! wait "$pid"; then
        LOCAL_FAIL=$((LOCAL_FAIL + 1))
      fi
    done
  fi

  echo "Parallel array launch summary: workflow=${WORKFLOW_NAME}, backend=${BACKEND}, runs=${#RUN_NAMES[@]}, submitted=${SUBMITTED}, submit_fail=${SUBMIT_FAIL}, local_fail=${LOCAL_FAIL}, max_concurrent=${MAX_CONCURRENT}, dry_run=${DRY_RUN}"
  [[ "$SUBMIT_FAIL" -eq 0 && "$LOCAL_FAIL" -eq 0 ]]
}
