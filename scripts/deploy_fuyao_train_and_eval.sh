#!/usr/bin/env bash

# Fuyao single-job pipeline: train FastWAM, then LIBERO-evaluate the final
# checkpoint — one PytorchJob, one container, same GPUs.
#
# Submit example:
#   fuyao deploy --gpus-per-node=8 --nodes=1 --project=<project> --site=<site> \
#     --docker-image <image> --experiment <name> -- \
#     FUYAO_USER=<you>@xiaopeng.com \
#     WANDB_MODE=offline \
#     RUN_NAME=libero_uncond_repro_001 \
#     bash /workspace/<you>@xiaopeng.com/FastWAM/scripts/deploy_fuyao_train_and_eval.sh 8 \
#       task=libero_uncond_2cam224_1e-4
#
# All positional hydra overrides go to TRAINING. Eval knobs go via env:
#   NUM_TRIALS=50            EVALUATION.num_trials
#   MAX_TASKS_PER_GPU=2      MULTIRUN.max_tasks_per_gpu
#   EVAL_ARGS="EVALUATION.replan_steps=10 ..."   extra eval hydra overrides (word-split)
#   EVAL_OUTPUT_DIR=...      default: <run_dir>/libero_eval
# Other knobs:
#   CKPT=/path/step.pt       skip training, evaluate this checkpoint only
#   SKIP_EVAL=1              train only
#   DRY_RUN=1                print resolved phases without running

if [[ -z "${BASH_VERSION:-}" ]]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

export FUYAO_USER="${FUYAO_USER:-CHANGE_ME@xiaopeng.com}"

DEFAULT_REPO_ROOT="/workspace/${FUYAO_USER}/FastWAM"
export REPO_ROOT="${REPO_ROOT:-${WORKSPACE_ROOT:-$DEFAULT_REPO_ROOT}}"
export RUNS_ROOT="${RUNS_ROOT:-/dataset_rc/${FUYAO_USER}/projects/fastwam/runs}"
export TASK_CONFIG="${TASK_CONFIG:-libero_uncond_2cam224_1e-4}"
export RUN_NAME="${RUN_NAME:-}"

die() {
  echo "[fuyao-pipeline] error: $*" >&2
  exit 1
}

is_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

hydra_override_value() {
  local key="$1"
  local fallback="$2"
  shift 2
  local arg
  for arg in "$@"; do
    case "$arg" in
      "$key="*|"+$key="*|"++$key="*)
        printf '%s\n' "${arg#*=}"
        return 0
        ;;
    esac
  done
  printf '%s\n' "$fallback"
}

[[ "$FUYAO_USER" != "CHANGE_ME@xiaopeng.com" ]] \
  || die "set FUYAO_USER (or edit the default at the top of this script)"
[[ -d "$REPO_ROOT" ]] || die "repository not found: $REPO_ROOT"
[[ -f "$REPO_ROOT/scripts/train_fuyao_fastwam.sh" ]] \
  || die "train wrapper not found under: $REPO_ROOT"
[[ -f "$REPO_ROOT/scripts/eval_fuyao_libero.sh" ]] \
  || die "eval wrapper not found under: $REPO_ROOT"

nproc="${NPROC_PER_NODE:-8}"
if (( $# > 0 )) && is_positive_integer "$1"; then
  nproc="$1"
  shift
fi
train_args=("$@")

# Resolve the task once for both phases (only the `task=<name>` form is supported).
task_value="$(hydra_override_value "task" "$TASK_CONFIG" "${train_args[@]}")"
task_basename="${task_value##*/}"
task_basename="${task_basename%.yaml}"
[[ -f "$REPO_ROOT/configs/task/${task_basename}.yaml" ]] \
  || die "task config not found: configs/task/${task_basename}.yaml"

# Pin RUN_ID so the run directory is known before training starts
# (scripts/train_zero1.sh honors a pre-set RUN_ID).
export RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
run_dir_id="$RUN_ID"
if [[ -n "$RUN_NAME" ]]; then
  [[ "$RUN_NAME" != *"/"* ]] || die "RUN_NAME must be a directory name, not a path: $RUN_NAME"
  run_dir_id="${RUN_ID}_${RUN_NAME}"
fi
RUN_DIR="$RUNS_ROOT/$task_basename/$run_dir_id"

latest_checkpoint() {
  local weights_dir="$1"
  ls -1 "$weights_dir"/step_*.pt 2>/dev/null | sort | tail -1
}

echo "[fuyao-pipeline] task: $task_basename"
echo "[fuyao-pipeline] run dir: $RUN_DIR"

# ---------------------------------------------------------------- train phase
if [[ -n "${CKPT:-}" ]]; then
  echo "[fuyao-pipeline] CKPT provided; skipping training: $CKPT"
  ckpt="$CKPT"
else
  echo "[fuyao-pipeline] === phase 1/2: training ==="

  # Git provenance (pattern from the team's iron_vla run.sh): record exactly
  # which code trained this run. Best-effort — skipped if git/.git is absent.
  if [[ "${DRY_RUN:-0}" != "1" ]] \
    && command -v git >/dev/null 2>&1 \
    && git -C "$REPO_ROOT" rev-parse HEAD >/dev/null 2>&1; then
    mkdir -p "$RUN_DIR"
    {
      echo "HEAD commit hash:"
      git -C "$REPO_ROOT" rev-parse HEAD
      echo
      echo "=== Uncommitted changes (git diff HEAD) ==="
      git -C "$REPO_ROOT" diff HEAD
    } > "$RUN_DIR/git_provenance.diff" 2>&1 || true
    echo "[fuyao-pipeline] git provenance -> $RUN_DIR/git_provenance.diff"
  fi
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '[fuyao-pipeline] would run: bash scripts/train_fuyao_fastwam.sh %q' "$nproc"
    printf ' %q' "${train_args[@]}"
    printf '\n'
  else
    bash "$REPO_ROOT/scripts/train_fuyao_fastwam.sh" "$nproc" "${train_args[@]}" \
      || die "training failed (exit $?); skipping evaluation"
  fi

  ckpt="$(latest_checkpoint "$RUN_DIR/checkpoints/weights" || true)"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    ckpt="${ckpt:-$RUN_DIR/checkpoints/weights/step_XXXXXX.pt}"
  fi
  [[ -n "$ckpt" ]] \
    || die "no checkpoint found under $RUN_DIR/checkpoints/weights — did training save?"
fi
echo "[fuyao-pipeline] checkpoint for eval: $ckpt"

# ----------------------------------------------------------------- eval phase
if [[ "${SKIP_EVAL:-0}" == "1" ]]; then
  echo "[fuyao-pipeline] SKIP_EVAL=1; done after training"
  exit 0
fi

echo "[fuyao-pipeline] === phase 2/2: LIBERO evaluation ==="
export EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$RUN_DIR/libero_eval}"

eval_extra=()
if [[ -n "${EVAL_ARGS:-}" ]]; then
  # shellcheck disable=SC2206  # intentional word splitting of EVAL_ARGS
  eval_extra=(${EVAL_ARGS})
fi

eval_cmd=(bash "$REPO_ROOT/scripts/eval_fuyao_libero.sh"
  "task=$task_basename" "ckpt=$ckpt" "${eval_extra[@]}")

printf '[fuyao-pipeline] eval command:'
printf ' %q' "${eval_cmd[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[fuyao-pipeline] DRY_RUN=1; evaluation was not started"
  exit 0
fi

# exec: the eval wrapper is the last phase; hand over signals/exit code to it.
exec "${eval_cmd[@]}"
