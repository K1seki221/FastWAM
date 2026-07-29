#!/usr/bin/env bash
# In-container runner for GR00T LIBERO fine-tuning on fuyao.
# Invoked by scripts/submit_fuyao_groot.sh; can also run directly on a kernel.
set -euo pipefail

export FUYAO_USER="${FUYAO_USER:-ruijie.zhang@xiaopeng.com}"
export REPO_ROOT="${REPO_ROOT:-/dataset_rc/${FUYAO_USER}/FastWAM}"
GROOT="${GROOT:-$REPO_ROOT/Isaac-GR00T}"
DATA_ROOT="${DATA_ROOT:-/dataset_rc/${FUYAO_USER}/libero_groot}"
RUNS_ROOT="${RUNS_ROOT:-/dataset_rc/${FUYAO_USER}/projects/groot_runs}"

# HF cache: cluster-shared; weights pre-downloaded -> offline by default.
export HF_HOME="${HF_HOME:-/dataset-cpfs3-rc/hf}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# Run knobs
SUITE="${SUITE:-10}"                       # 10 | goal | object | spatial
RUN_NAME="${RUN_NAME:-baseline_libero_${SUITE}}"
export NUM_GPUS="${NUM_GPUS:-8}"
export MAX_STEPS="${MAX_STEPS:-20000}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-640}"
export SAVE_STEPS="${SAVE_STEPS:-1000}"
USE_ROUTER="${USE_ROUTER:-0}"
ROUTER_LR="${ROUTER_LR:-1e-3}"
ROUTER_LAYERS="${ROUTER_LAYERS:-}"         # e.g. "0 6 12"; empty = all
EXTRA_ARGS="${EXTRA_ARGS:-}"

# wandb: on iff a key is provided or offline mode requested
if [[ -n "${WANDB_API_KEY:-}" || "${WANDB_MODE:-}" == "offline" ]]; then
  export USE_WANDB=1
else
  export USE_WANDB=0
fi

dataset="$DATA_ROOT/libero_${SUITE}_no_noops_1.0.0_lerobot"
[[ -d "$dataset" ]] || { echo "dataset not found: $dataset" >&2; exit 1; }
[[ -f "$dataset/meta/modality.json" ]] || { echo "missing $dataset/meta/modality.json" >&2; exit 1; }
[[ -x "$GROOT/.venv/bin/python" ]] || { echo "venv missing: run 'uv sync' in $GROOT first" >&2; exit 1; }

router_args=()
if [[ "$USE_ROUTER" == "1" ]]; then
  router_args=(-- --use-condition-router --router-lr "$ROUTER_LR")
  [[ -n "$ROUTER_LAYERS" ]] && router_args+=(--router-candidate-layers $ROUTER_LAYERS)
  [[ -n "$EXTRA_ARGS" ]] && router_args+=($EXTRA_ARGS)
elif [[ -n "$EXTRA_ARGS" ]]; then
  router_args=(-- $EXTRA_ARGS)
fi

cd "$GROOT"
source .venv/bin/activate
echo "[groot] suite=$SUITE run=$RUN_NAME gpus=$NUM_GPUS steps=$MAX_STEPS batch=$GLOBAL_BATCH_SIZE router=$USE_ROUTER wandb=$USE_WANDB"
exec bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path "$dataset/" \
  --embodiment-tag LIBERO_PANDA \
  --output-dir "$RUNS_ROOT/$RUN_NAME" \
  --state-dropout-prob 0.2 \
  "${router_args[@]}"
