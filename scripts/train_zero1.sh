#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${1:?Usage: bash scripts/train_zero1.sh <nproc_per_node> [hydra_overrides...]}"
shift

EXTRA_ARGS=("$@")
NUM_MACHINES="${NNODES:-1}"
MACHINE_RANK="${NODE_RANK:-0}"
MAIN_PROCESS_IP="${MASTER_ADDR:-127.0.0.1}"
MAIN_PROCESS_PORT="${MASTER_PORT:-29500}"

is_integer() {
  [[ "${1}" =~ ^[0-9]+$ ]]
}

if ! is_integer "${NUM_MACHINES}" || ! is_integer "${MACHINE_RANK}"; then
  echo "Error: NUM_MACHINES (${NUM_MACHINES}) and MACHINE_RANK (${MACHINE_RANK}) must be integers." >&2
  exit 1
fi

extract_task_basename() {
  local cfg="$1"
  if [[ "${cfg}" == task/* ]]; then
    local name="${cfg#task/}"
    name="${name%.yaml}"
    echo "${name}"
    return 0
  fi
  return 1
}

TASK_BASENAME="train"
for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
  arg="${EXTRA_ARGS[$i]}"
  case "${arg}" in
    --config-name)
      if ((i + 1 < ${#EXTRA_ARGS[@]})); then
        next="${EXTRA_ARGS[$((i + 1))]}"
        if parsed="$(extract_task_basename "${next}")"; then
          TASK_BASENAME="${parsed}"
        fi
      fi
      ;;
    --config-name=*)
      cfg="${arg#--config-name=}"
      if parsed="$(extract_task_basename "${cfg}")"; then
        TASK_BASENAME="${parsed}"
      fi
      ;;
    task=*)
      cfg="${arg#task=}"
      cfg="${cfg%.yaml}"
      TASK_BASENAME="${cfg}"
      ;;
  esac
done

if [[ -z "${RUN_ID:-}" ]]; then
  if (( NUM_MACHINES <= 1 )); then
    RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
  else
    RUN_ID_SYNC_TIMEOUT="${RUN_ID_SYNC_TIMEOUT:-180}"
    RUN_ID_SYNC_PORT="${RUN_ID_SYNC_PORT:-$((MAIN_PROCESS_PORT + 11))}"

    export RUN_ID_SYNC_HOST="${MAIN_PROCESS_IP}"
    export RUN_ID_SYNC_PORT
    export RUN_ID_SYNC_TIMEOUT
    export RUN_ID_SYNC_MACHINE_RANK="${MACHINE_RANK}"
    export RUN_ID_SYNC_NUM_MACHINES="${NUM_MACHINES}"
    export RUN_ID_SYNC_TASK_BASENAME="${TASK_BASENAME}"

    RUN_ID="$(
      python - <<'PY'
import datetime
import os
from datetime import timedelta

import torch.distributed as dist

host = os.environ["RUN_ID_SYNC_HOST"]
port = int(os.environ["RUN_ID_SYNC_PORT"])
timeout_s = int(os.environ["RUN_ID_SYNC_TIMEOUT"])
machine_rank = int(os.environ["RUN_ID_SYNC_MACHINE_RANK"])
num_machines = int(os.environ["RUN_ID_SYNC_NUM_MACHINES"])
task_basename = os.environ.get("RUN_ID_SYNC_TASK_BASENAME", "train")

store = dist.TCPStore(
    host_name=host,
    port=port,
    world_size=num_machines,
    is_master=(machine_rank == 0),
    timeout=timedelta(seconds=timeout_s),
)
key = f"run_id::{task_basename}"
if machine_rank == 0:
    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    store.set(key, run_id)
run_id = store.get(key).decode("utf-8")
print(run_id)
PY
    )"

    echo "[run_id_sync] mode=tcpstore host=${RUN_ID_SYNC_HOST} port=${RUN_ID_SYNC_PORT} timeout_s=${RUN_ID_SYNC_TIMEOUT} run_id=${RUN_ID}"
  fi
fi

# Optional run-dir customization (used by scripts/train_fuyao_fastwam.sh):
#   RUNS_ROOT   base directory for run outputs (default ./runs)
#   RUN_NAME    suffix appended to the timestamped run id
RUNS_ROOT="${RUNS_ROOT:-./runs}"
RUN_NAME="${RUN_NAME:-}"
if [[ "${RUN_NAME}" == *"/"* ]]; then
  echo "Error: RUN_NAME must be a directory name, not a path: ${RUN_NAME}" >&2
  exit 1
fi
RUN_DIR_ID="${RUN_ID}"
if [[ -n "${RUN_NAME}" ]]; then
  RUN_DIR_ID="${RUN_ID}_${RUN_NAME}"
fi

echo "[launch] nproc_per_node=${NPROC_PER_NODE} num_machines=${NUM_MACHINES} machine_rank=${MACHINE_RANK} run_id=${RUN_ID} run_dir=${RUNS_ROOT}/${TASK_BASENAME}/${RUN_DIR_ID}"

# When ACCELERATE_PYTHON is set (fuyao: the conda env was copied and console
# scripts keep stale shebangs), invoke the accelerate entrypoint through the
# env's python explicitly.
ACCELERATE_CMD=(accelerate)
if [[ -n "${ACCELERATE_PYTHON:-}" ]]; then
  ACCELERATE_ENTRYPOINT="${ACCELERATE_ENTRYPOINT:?ACCELERATE_ENTRYPOINT is required when ACCELERATE_PYTHON is set}"
  if [[ ! -x "${ACCELERATE_PYTHON}" ]]; then
    echo "Error: ACCELERATE_PYTHON is not executable: ${ACCELERATE_PYTHON}" >&2
    exit 1
  fi
  if [[ ! -f "${ACCELERATE_ENTRYPOINT}" ]]; then
    echo "Error: accelerate entrypoint not found: ${ACCELERATE_ENTRYPOINT}" >&2
    exit 1
  fi
  ACCELERATE_CMD=("${ACCELERATE_PYTHON}" "${ACCELERATE_ENTRYPOINT}")
fi

# exec so bash never re-reads this file after the many-hour training command
# returns (the workspace copy may have been edited while the job was running).
exec "${ACCELERATE_CMD[@]}" launch \
  --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
  --num_processes "${NPROC_PER_NODE}" \
  scripts/train.py \
  "output_dir=${RUNS_ROOT}/${TASK_BASENAME}/${RUN_DIR_ID}" \
  "wandb.name=${TASK_BASENAME}" \
  "${EXTRA_ARGS[@]}"
