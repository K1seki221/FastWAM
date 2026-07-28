#!/usr/bin/env bash

# One-command fuyao submission for FastWAM train+eval.
#
# Usage (from a machine with the `fuyao` CLI):
#   bash scripts/submit_fuyao.sh [nproc] [hydra overrides...]
#
# Examples:
#   RUN_NAME=libero_uncond_repro_001 bash scripts/submit_fuyao.sh 8 task=libero_uncond_2cam224_1e-4
#   SKIP_EVAL=1 bash scripts/submit_fuyao.sh 8 task=libero_uncond_2cam224_1e-4     # train only
#   CKPT=/dataset_rc/.../step_021700.pt bash scripts/submit_fuyao.sh 8             # eval only
#   DRY_RUN=1 bash scripts/submit_fuyao.sh 8 task=...                              # print, don't submit
#
# The job runs scripts/deploy_fuyao_train_and_eval.sh (train -> latest ckpt -> LIBERO eval).
# Positional hydra overrides go to TRAINING; eval knobs go via env
# (NUM_TRIALS, MAX_TASKS_PER_GPU, EVAL_ARGS, EVAL_OUTPUT_DIR).

if [[ -z "${BASH_VERSION:-}" ]]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

# ----- site constants (edit once; conventions from the team's iron_vla run.sh) -----
export FUYAO_USER="${FUYAO_USER:-ruijie.zhang@xiaopeng.com}"
# The cluster checkout lives on persistent storage, not /workspace.
export REPO_ROOT="${REPO_ROOT:-/dataset_rc/${FUYAO_USER}/FastWAM}"
PROJECT="${PROJECT:-rc-embodied-foundation-model}"
SITE="${SITE:-fuyao_sh_n2}"
EXPERIMENT="${EXPERIMENT:-ruijie}"
DOCKER_IMAGE="${DOCKER_IMAGE:-infra-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/data-infra/fuyao:liuw50-260318-0232}"
# Optional flags — emitted only when non-empty (Xiaopeng's known-good FastWAM
# submission used none of them; the team's iron_vla submissions use
# e.g. GPU_TYPE=h200 QUEUE=rc-embodied-foundation-model-h200-p1 VOLUME=rc-perception).
GPU_TYPE="${GPU_TYPE:-}"
QUEUE="${QUEUE:-}"
VOLUME="${VOLUME:-}"
LABEL="${LABEL:-${RUN_NAME:-fastwam}}"
# Borrowed env for now; switch to your own once built
# (safe: the wrappers put the checkout's src/ first on PYTHONPATH).
export CONDA_ENV="${CONDA_ENV:-/dataset_rc/zhangxp7@xiaopeng.com/miniconda3/envs/wam}"

die() {
  echo "[submit] error: $*" >&2
  exit 1
}

is_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

[[ "$FUYAO_USER" != "CHANGE_ME@xiaopeng.com" ]] \
  || die "set FUYAO_USER (or edit the default at the top of this script)"

nproc="${NPROC_PER_NODE:-8}"
if (( $# > 0 )) && is_positive_integer "$1"; then
  nproc="$1"
  shift
fi
is_positive_integer "$nproc" || die "nproc must be a positive integer, got: $nproc"

# Env vars forwarded into the job command (only those currently set).
FORWARD_VARS=(
  FUYAO_USER CONDA_ENV REPO_ROOT CACHE_ROOT RUNS_ROOT
  WANDB_API_KEY WANDB_MODE WANDB_ENTITY WANDB_PROJECT
  RUN_NAME TASK_CONFIG
  NUM_TRIALS MAX_TASKS_PER_GPU EVAL_ARGS EVAL_OUTPUT_DIR
  CKPT SKIP_EVAL SKIP_APT_INSTALL USE_SYSTEM_NCCL
)

# The command after `--` is shell-evaluated on the job side (inline ENV=val
# prefixes rely on that), so %q-escape every value and argument.
job_cmd=()
for var in "${FORWARD_VARS[@]}"; do
  if [[ -n "${!var:-}" ]]; then
    job_cmd+=("${var}=$(printf '%q' "${!var}")")
  fi
done
# In-container dry run of the pipeline itself (distinct from submit-level DRY_RUN).
if [[ "${JOB_DRY_RUN:-0}" == "1" ]]; then
  job_cmd+=("DRY_RUN=1")
fi
job_cmd+=(bash "${REPO_ROOT}/scripts/deploy_fuyao_train_and_eval.sh" "$nproc")
for arg in "$@"; do
  job_cmd+=("$(printf '%q' "$arg")")
done

deploy_cmd=(
  fuyao deploy
  --gpus-per-node="$nproc" --nodes=1
  --project="$PROJECT" --site="$SITE"
  --docker-image "$DOCKER_IMAGE"
  --experiment "$EXPERIMENT"
  --label "$LABEL"
)
[[ -n "$GPU_TYPE" ]] && deploy_cmd+=(--gpu-type "$GPU_TYPE")
[[ -n "$QUEUE" ]] && deploy_cmd+=(--queue "$QUEUE")
[[ -n "$VOLUME" ]] && deploy_cmd+=(--volume "$VOLUME")
deploy_cmd+=(-- "${job_cmd[@]}")

echo "[submit] project=$PROJECT site=$SITE gpus=$nproc experiment=$EXPERIMENT label=$LABEL${GPU_TYPE:+ gpu_type=$GPU_TYPE}${QUEUE:+ queue=$QUEUE}${VOLUME:+ volume=$VOLUME}"
if [[ -z "${WANDB_API_KEY:-}" && "${WANDB_MODE:-online}" == "online" ]]; then
  echo "[submit] note: WANDB_API_KEY not set — the job will run with wandb DISABLED (pass WANDB_API_KEY=... or WANDB_MODE=offline to log)"
fi
printf '[submit] command:'
printf ' %s' "${deploy_cmd[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[submit] DRY_RUN=1; not submitted"
  exit 0
fi

command -v fuyao >/dev/null 2>&1 || die "fuyao CLI not found on this machine"
exec "${deploy_cmd[@]}"
