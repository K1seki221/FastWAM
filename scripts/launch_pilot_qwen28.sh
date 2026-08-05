#!/usr/bin/env bash
# Qwen28 condition-router pilot arms (cc_memo/13 "Pilot phase").
#   A = fixed hard depth-aligned {7,14,21,28}   B = learnable, uniform K=4
#   C = fixed all-blocks->L28 (stock incumbent) D = learnable, uniform K=28
#   E = B with router lr 1e-4 (no 20x boost)    F = B with 500-step logit freeze
# 6-task PnP subset, 10K steps @ eff batch 256 (16 micro x 16 accum).
#
# Portable: override the env paths below per host. GPU/port per arm can be
# overridden with PILOT_GPU / PILOT_PORT.
#
# Usage: bash scripts/launch_pilot_qwen28.sh A|B|C|D|E|F
set -uo pipefail
arm="$1"

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
VENV_DIR="${VENV_DIR:-/data/ruijiezhang/env/groot}"
RUNS_ROOT="${RUNS_ROOT:-/data/ruijiezhang/groot_runs}"
GROOT_HF_HOME="${GROOT_HF_HOME:-/data/ruijiezhang/hf/hf_cache}"
GR1_DATA_GLOB="${GR1_DATA_GLOB:-/data/ruijiezhang/gr1_unified/gr1_unified.PnP*}"
# FFmpeg<8 shared libs for torchcodec; the libstdc++ preload works around
# conda-based venv pythons whose RPATH drags in an ancient libstdc++.
FFMPEG_LIB="${FFMPEG_LIB:-/data/ruijiezhang/env/ffmpeg7/lib}"

# Resolve the Qwen3-VL-2B snapshot from the HF cache unless given explicitly.
if [[ -z "${GR00T_BACKBONE_PATH:-}" ]]; then
  qrepo="$GROOT_HF_HOME/hub/models--Qwen--Qwen3-VL-2B-Instruct"
  if [[ -f "$qrepo/refs/main" ]]; then
    GR00T_BACKBONE_PATH="$qrepo/snapshots/$(cat "$qrepo/refs/main")"
  else
    GR00T_BACKBONE_PATH="$(ls -d "$qrepo/snapshots"/* 2>/dev/null | head -1)"
  fi
fi
[[ -f "$GR00T_BACKBONE_PATH/config.json" ]] \
  || { echo "Qwen3-VL-2B snapshot not found (set GR00T_BACKBONE_PATH)" >&2; exit 1; }

rlr=2e-3
case "$arm" in
  A) gpu=5; port=29521; layers="7 14 21 28"; init="--router-init-mode span --router-init-bias 16.0 --router-frozen" ;;
  B) gpu=7; port=29522; layers="7 14 21 28"; init="--router-init-mode span --router-init-bias 0.0" ;;
  C) gpu=4; port=29523; layers="7 14 21 28"; init="--router-init-mode last --router-init-bias 16.0 --router-frozen" ;;
  D) gpu=6; port=29524; layers="$(seq -s' ' 1 28)"; init="--router-init-mode span --router-init-bias 0.0" ;;
  E) gpu=3; port=29525; layers="7 14 21 28"; init="--router-init-mode span --router-init-bias 0.0"; rlr=1e-4 ;;
  F) gpu=2; port=29526; layers="7 14 21 28"; init="--router-init-mode span --router-init-bias 0.0 --router-freeze-steps 500" ;;
  *) echo "unknown arm $arm" >&2; exit 1 ;;
esac
case "$arm" in
  A) name=pilot_A_fixed_span ;;
  B) name=pilot_B_uniform_k4 ;;
  C) name=pilot_C_fixed_last ;;
  D) name=pilot_D_uniform_k28 ;;
  E) name=pilot_E_uniform_k4_baselr ;;
  F) name=pilot_F_uniform_k4_freeze500 ;;
esac
name="$name${PILOT_SUFFIX:-}"
gpu="${PILOT_GPU:-$gpu}"
port="${PILOT_PORT:-$port}"
# ROUTER_PCPROJ=1 => per-candidate identity-init proj adapters (v1.5-lite)
[[ "${ROUTER_PCPROJ:-0}" == "1" ]] && init="$init --router-candidate-proj"

ld_env=()
if [[ -d "$FFMPEG_LIB" ]]; then
  ld_env+=(LD_LIBRARY_PATH="$FFMPEG_LIB")
  [[ -f "$FFMPEG_LIB/libstdc++.so.6" ]] && ld_env+=(LD_PRELOAD="$FFMPEG_LIB/libstdc++.so.6")
fi

cd "$REPO_ROOT"
env \
  REPO_ROOT="$REPO_ROOT" \
  VENV_DIR="$VENV_DIR" \
  RUNS_ROOT="$RUNS_ROOT" \
  GROOT_HF_HOME="$GROOT_HF_HOME" \
  GR00T_BACKBONE_PATH="$GR00T_BACKBONE_PATH" \
  DATASET_GLOB="$GR1_DATA_GLOB" \
  EMBODIMENT_TAG=ROBOCASA_GR1_TABLETOP \
  NUM_GPUS=1 CUDA_VISIBLE_DEVICES=$gpu MASTER_PORT=$port \
  MAX_STEPS=10000 GLOBAL_BATCH_SIZE=16 SAVE_STEPS=2500 SAVE_TOTAL_LIMIT=2 \
  USE_ROUTER=1 ROUTER_LR=$rlr ROUTER_LAYERS="$layers" \
  WANDB_MODE="${WANDB_MODE:-online}" \
  "${ld_env[@]}" \
  RUN_NAME=$name \
  EXTRA_ARGS="--skip-weight-loading --tune-llm --tune-visual --backbone-lr 1e-5 --select-layer 28 --backbone-embedding-dim 2048 --dit-num-layers 28 --gradient_accumulation_steps 16 $init" \
  bash scripts/groot_fuyao_train.sh \
  > "$RUNS_ROOT/$name.log" 2>&1
echo "PILOT $arm EXITED: $?" >> "$RUNS_ROOT/pilot.status"
