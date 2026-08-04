#!/usr/bin/env bash
# In-container runner for GR00T LIBERO fine-tuning on fuyao.
# Invoked by scripts/submit_fuyao_groot.sh; can also run directly on a kernel.
set -euo pipefail

export FUYAO_USER="${FUYAO_USER:-ruijie.zhang@xiaopeng.com}"
export REPO_ROOT="${REPO_ROOT:-/dataset_rc/${FUYAO_USER}/FastWAM}"
GROOT="${GROOT:-$REPO_ROOT/Isaac-GR00T}"
# Venv lives OUTSIDE the repo (fuyao snapshots the submit dir; keep it small).
VENV_DIR="${VENV_DIR:-/dataset_rc/${FUYAO_USER}/projects/groot/venv}"
DATA_ROOT="${DATA_ROOT:-/dataset_rc/${FUYAO_USER}/libero_groot}"
RUNS_ROOT="${RUNS_ROOT:-/dataset_rc/${FUYAO_USER}/projects/groot_runs}"

# HF cache on the user's own /dataset_rc (always mounted in jobs);
# weights pre-downloaded -> offline by default.
# Hard-pin: job containers pre-set HF_HOME=/dataset_rc/hf (no user segment), which
# broke the pre-flight (bifrost-2026073102171101). Override via GROOT_HF_HOME only.
export HF_HOME="${GROOT_HF_HOME:-/dataset_rc/${FUYAO_USER}/hf}"
export HF_HUB_CACHE="$HF_HOME/hub"   # containers may pre-set HF_HUB_CACHE, which outranks HF_HOME
[[ -d "$HF_HOME/hub/models--nvidia--GR00T-N1.7-3B" ]] \
  || { echo "GR00T-N1.7-3B not in $HF_HOME/hub — copy the model caches there first" >&2; exit 1; }
# Point the Cosmos backbone at its local snapshot: repo-id loading needs a hub
# API call in transformers 4.57.3 (tokenizer mistral-regex probe) that raises
# under HF_HUB_OFFLINE=1 (killed bifrost-2026073102591700).
cosmos_repo="$HF_HOME/hub/models--nvidia--Cosmos-Reason2-2B"
if [[ -f "$cosmos_repo/refs/main" ]]; then
  cosmos_snap="$cosmos_repo/snapshots/$(cat "$cosmos_repo/refs/main")"
else
  cosmos_snap="$(ls -d "$cosmos_repo/snapshots"/* 2>/dev/null | head -1)"
fi
[[ -f "$cosmos_snap/tokenizer.json" ]] \
  || { echo "Cosmos-Reason2-2B snapshot incomplete/missing under $cosmos_repo" >&2; exit 1; }
export GR00T_BACKBONE_PATH="${GR00T_BACKBONE_PATH:-$cosmos_snap}"
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

# DATASET_PATH: explicit os.pathsep-joined dataset dirs (overrides SUITE);
# DATASET_GLOB: expand a glob of dataset dirs (e.g. ".../gr1_unified/gr1_unified.*").
# EMBODIMENT_TAG selects the embodiment (default LIBERO_PANDA).
EMBODIMENT_TAG="${EMBODIMENT_TAG:-LIBERO_PANDA}"
if [[ -n "${DATASET_GLOB:-}" ]]; then
  dataset=""
  for d in $DATASET_GLOB; do
    [[ -d "$d" ]] || continue
    [[ -f "$d/meta/modality.json" ]] || { echo "missing $d/meta/modality.json" >&2; exit 1; }
    dataset="${dataset:+$dataset:}$d"
  done
  [[ -n "$dataset" ]] || { echo "DATASET_GLOB matched nothing: $DATASET_GLOB" >&2; exit 1; }
  echo "[groot] DATASET_GLOB matched $(echo "$dataset" | tr ':' '\n' | wc -l) dirs"
elif [[ -n "${DATASET_PATH:-}" ]]; then
  dataset="$DATASET_PATH"
# SUITE=all -> StarVLA-style joint training on the 4 standard suites
# (finetune.sh --dataset-path accepts os.pathsep-joined dirs).
elif [[ "$SUITE" == "all" ]]; then
  dataset=""
  for s in spatial object goal 10; do
    d="$DATA_ROOT/libero_${s}_no_noops_1.0.0_lerobot"
    [[ -d "$d" ]] || { echo "dataset not found: $d" >&2; exit 1; }
    [[ -f "$d/meta/modality.json" ]] || { echo "missing $d/meta/modality.json" >&2; exit 1; }
    dataset="${dataset:+$dataset:}$d"
  done
else
  dataset="$DATA_ROOT/libero_${SUITE}_no_noops_1.0.0_lerobot"
  [[ -d "$dataset" ]] || { echo "dataset not found: $dataset" >&2; exit 1; }
  [[ -f "$dataset/meta/modality.json" ]] || { echo "missing $dataset/meta/modality.json" >&2; exit 1; }
fi
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  if [[ -x "$GROOT/.venv/bin/python" ]]; then
    VENV_DIR="$GROOT/.venv"   # legacy in-repo venv fallback
  else
    echo "venv missing: UV_PROJECT_ENVIRONMENT=$VENV_DIR uv sync (in $GROOT) first" >&2
    exit 1
  fi
fi

router_args=()
if [[ "$USE_ROUTER" == "1" ]]; then
  router_args=(-- --use-condition-router --router-lr "$ROUTER_LR")
  [[ -n "$ROUTER_LAYERS" ]] && router_args+=(--router-candidate-layers $ROUTER_LAYERS)
  [[ -n "$EXTRA_ARGS" ]] && router_args+=($EXTRA_ARGS)
elif [[ -n "$EXTRA_ARGS" ]]; then
  router_args=(-- $EXTRA_ARGS)
fi

cd "$GROOT"
source "$VENV_DIR/bin/activate"
echo "[groot] suite=$SUITE run=$RUN_NAME gpus=$NUM_GPUS steps=$MAX_STEPS batch=$GLOBAL_BATCH_SIZE router=$USE_ROUTER wandb=$USE_WANDB"
exec bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path "$dataset/" \
  --embodiment-tag "$EMBODIMENT_TAG" \
  --output-dir "$RUNS_ROOT/$RUN_NAME" \
  --state-dropout-prob 0.2 \
  "${router_args[@]}"
