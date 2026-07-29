#!/usr/bin/env bash
# One-command fuyao submission for GR00T LIBERO runs (baseline or router).
#
# Examples:
#   WANDB_API_KEY=... RUN_NAME=baseline_libero_10 SUITE=10 bash scripts/submit_fuyao_groot.sh 8
#   USE_ROUTER=1 WANDB_API_KEY=... RUN_NAME=router_libero_10 SUITE=10 bash scripts/submit_fuyao_groot.sh 8
#   MAX_STEPS=100 RUN_NAME=timing SUITE=10 DRY_RUN=1 bash scripts/submit_fuyao_groot.sh 8
set -euo pipefail

export FUYAO_USER="${FUYAO_USER:-ruijie.zhang@xiaopeng.com}"
export REPO_ROOT="${REPO_ROOT:-/dataset_rc/${FUYAO_USER}/FastWAM}"
PROJECT="${PROJECT:-rc-embodied-foundation-model}"
SITE="${SITE:-fuyao_sh_n2}"
EXPERIMENT="${EXPERIMENT:-ruijie}"
DOCKER_IMAGE="${DOCKER_IMAGE:-infra-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/data-infra/fuyao:liuw50-260318-0232}"
GPU_TYPE="${GPU_TYPE:-h200}"
QUEUE="${QUEUE:-}"
VOLUME="${VOLUME:-rc-perception}"
LABEL="${LABEL:-${RUN_NAME:-groot}}"

die() { echo "[submit-groot] error: $*" >&2; exit 1; }

nproc="${NPROC_PER_NODE:-8}"
if (( $# > 0 )) && [[ "$1" =~ ^[1-9][0-9]*$ ]]; then nproc="$1"; shift; fi

FORWARD_VARS=(
  FUYAO_USER REPO_ROOT GROOT DATA_ROOT RUNS_ROOT
  SUITE RUN_NAME NUM_GPUS MAX_STEPS GLOBAL_BATCH_SIZE SAVE_STEPS
  USE_ROUTER ROUTER_LR ROUTER_LAYERS EXTRA_ARGS
  WANDB_API_KEY WANDB_MODE HF_TOKEN HF_HUB_OFFLINE HF_HOME
)
job_cmd=()
for var in "${FORWARD_VARS[@]}"; do
  [[ -n "${!var:-}" ]] && job_cmd+=("${var}=$(printf '%q' "${!var}")")
done
job_cmd+=("NUM_GPUS=$nproc" bash "${REPO_ROOT}/scripts/groot_fuyao_train.sh")

deploy_cmd=(
  fuyao deploy
  --gpus-per-node="$nproc" --nodes=1
  --project="$PROJECT" --site="$SITE"
  --docker-image "$DOCKER_IMAGE"
  --experiment "$EXPERIMENT" --label "$LABEL"
)
[[ -n "$GPU_TYPE" ]] && deploy_cmd+=(--gpu-type "$GPU_TYPE")
[[ -n "$QUEUE" ]] && deploy_cmd+=(--queue "$QUEUE")
[[ -n "$VOLUME" ]] && deploy_cmd+=(--volume "$VOLUME")
deploy_cmd+=(-- "${job_cmd[@]}")

echo "[submit-groot] suite=${SUITE:-10} run=${RUN_NAME:-<default>} router=${USE_ROUTER:-0} gpus=$nproc gpu_type=${GPU_TYPE:-<any>}"
printf '[submit-groot] command:'; printf ' %s' "${deploy_cmd[@]}"; printf '\n'

if [[ "${DRY_RUN:-0}" == "1" ]]; then echo "[submit-groot] DRY_RUN=1; not submitted"; exit 0; fi
command -v fuyao >/dev/null 2>&1 || die "fuyao CLI not found on this machine"
exec "${deploy_cmd[@]}"
