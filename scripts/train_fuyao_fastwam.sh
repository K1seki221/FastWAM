#!/usr/bin/env bash

# Fuyao single-node training entrypoint for FastWAM (adapted from Xiaopeng
# Zhang's fork; infrastructure only, no algorithm changes).
#
# Submit example (PytorchJob):
#   fuyao deploy --gpus-per-node=8 --nodes=1 --project=<project> --site=<site> \
#     --docker-image <image> --experiment <name> -- \
#     FUYAO_USER=<you>@xiaopeng.com \
#     WANDB_API_KEY=<your-key> \
#     RUN_NAME=libero_uncond_repro_001 \
#     bash /workspace/<you>@xiaopeng.com/FastWAM/scripts/train_fuyao_fastwam.sh 8 \
#       task=libero_uncond_2cam224_1e-4
#
# Useful overrides:
#   CONDA_ENV=/dataset_rc/<someone>/miniconda3/envs/wam   # reuse an existing env
#   WANDB_MODE=offline                                    # no API key needed; `wandb sync` later
#   SKIP_APT_INSTALL=1 DRY_RUN=1                          # inspect the resolved command

if [[ -z "${BASH_VERSION:-}" ]]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

# ----- user/site constants (edit FUYAO_USER or override everything via env) -----
export FUYAO_USER="${FUYAO_USER:-ruijie.zhang@xiaopeng.com}"

DEFAULT_REPO_ROOT="/dataset_rc/${FUYAO_USER}/FastWAM"
export REPO_ROOT="${REPO_ROOT:-${WORKSPACE_ROOT:-$DEFAULT_REPO_ROOT}}"
export CONDA_ROOT="${CONDA_ROOT:-/dataset_rc/${FUYAO_USER}/miniconda3}"
export CONDA_ENV="${CONDA_ENV:-$CONDA_ROOT/envs/wam}"
export PYTHON_BIN="${PYTHON_BIN:-$CONDA_ENV/bin/python}"
export ACCELERATE_ENTRYPOINT="${ACCELERATE_ENTRYPOINT:-$CONDA_ENV/bin/accelerate}"

export CACHE_ROOT="${CACHE_ROOT:-/dataset_rc/${FUYAO_USER}/projects/fastwam/.cache}"
export RUNS_ROOT="${RUNS_ROOT:-/dataset_rc/${FUYAO_USER}/projects/fastwam/runs}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT/torch}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$CACHE_ROOT/modelscope}"

# Shared read-only Wan2.2 weights + ActionDiT backbone on fuyao.
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/dataset_rc/vlm/fm_models/VIT/WAM/FastWAM/checkpoints}"

export TASK_CONFIG="${TASK_CONFIG:-libero_uncond_2cam224_1e-4}"
# W&B credential is deliberately NOT stored in the repo. Pass it at
# submission time (WANDB_API_KEY=... bash scripts/submit_fuyao.sh ... — the
# submit wrapper forwards it into the job), or use WANDB_MODE=offline and
# `wandb sync` later. With neither, wandb logging is disabled.
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-fast-wam}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

die() {
  echo "[fuyao] error: $*" >&2
  exit 1
}

is_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

has_hydra_override() {
  local key="$1"
  shift
  local arg
  for arg in "$@"; do
    if [[ "$arg" == "$key="* || "$arg" == "+$key="* || "$arg" == "++$key="* ]]; then
      return 0
    fi
  done
  return 1
}

install_system_packages() {
  local -a packages=(libegl1 libopengl0 libglvnd0 libgl1)

  if [[ "${SKIP_APT_INSTALL:-0}" == "1" ]]; then
    echo "[system] SKIP_APT_INSTALL=1; skipping system package installation"
    return 0
  fi
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[system] DRY_RUN=1; system package installation skipped"
    return 0
  fi
  command -v apt-get >/dev/null 2>&1 \
    || die "apt-get is unavailable; cannot install ${packages[*]}"

  local -a missing=()
  local package
  for package in "${packages[@]}"; do
    if ! command -v dpkg-query >/dev/null 2>&1 \
      || ! dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -q 'ok installed'; then
      missing+=("$package")
    fi
  done
  if (( ${#missing[@]} == 0 )); then
    echo "[system] EGL/OpenGL packages are already installed"
    return 0
  fi

  local -a root_prefix=()
  if (( EUID != 0 )); then
    command -v sudo >/dev/null 2>&1 \
      || die "root privileges are required to install ${missing[*]}"
    root_prefix=(sudo)
  fi

  local apt_mirror="${APT_MIRROR:-http://mirrors.cloud.aliyuncs.com/ubuntu}"
  apt_mirror="${apt_mirror%/}"
  local source_file
  while IFS= read -r -d '' source_file; do
    "${root_prefix[@]}" sed -E -i \
      -e "s|https?://([[:alnum:].-]+\\.)?archive\\.ubuntu\\.com/ubuntu|$apt_mirror|g" \
      -e "s|https?://security\\.ubuntu\\.com/ubuntu|$apt_mirror|g" \
      "$source_file"
  done < <(
    find /etc/apt -maxdepth 2 -type f \
      \( -name '*.list' -o -name '*.sources' \) -print0
  )

  echo "[system] apt mirror: $apt_mirror"
  echo "[system] installing: ${missing[*]}"
  "${root_prefix[@]}" env DEBIAN_FRONTEND=noninteractive apt-get \
    -o Acquire::Retries=3 update
  "${root_prefix[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install \
    -y --no-install-recommends "${missing[@]}"
}

activate_conda_wam() {
  [[ -d "$CONDA_ENV" ]] || die "Conda environment not found: $CONDA_ENV"
  [[ -x "$PYTHON_BIN" ]] || die "Python executable not found: $PYTHON_BIN"
  [[ -f "$ACCELERATE_ENTRYPOINT" ]] \
    || die "accelerate entrypoint not found: $ACCELERATE_ENTRYPOINT"

  # Persistent envs are copied around on fuyao, so generated console scripts
  # may retain stale absolute shebangs. Select the environment directly and
  # let train_zero1.sh invoke accelerate with this environment's Python.
  unset PYTHONHOME || true
  unset CONDA_EXE _CONDA_EXE CONDA_PYTHON_EXE _CONDA_ROOT _CE_M _CE_CONDA || true
  export CONDA_PREFIX="$CONDA_ENV"
  export CONDA_DEFAULT_ENV="$(basename "$CONDA_ENV")"
  export CONDA_SHLVL=1
  export PATH="$CONDA_ENV/bin:$PATH"
  export LD_LIBRARY_PATH="$CONDA_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export ACCELERATE_PYTHON="$PYTHON_BIN"

  # REPO_ROOT/src FIRST so this checkout's fastwam wins over any editable
  # install baked into a borrowed conda env.
  export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
  hash -r
}

setup_nccl() {
  local use_system_nccl="${USE_SYSTEM_NCCL:-1}"
  local system_nccl_lib="${SYSTEM_NCCL_LIB:-/lib/x86_64-linux-gnu/libnccl.so.2}"

  case "$use_system_nccl" in
    1|true|TRUE|yes|YES)
      [[ -r "$system_nccl_lib" ]] \
        || die "system NCCL not found: $system_nccl_lib (set USE_SYSTEM_NCCL=0 to use bundled NCCL)"
      if [[ ":${LD_PRELOAD:-}:" != *":$system_nccl_lib:"* ]]; then
        export LD_PRELOAD="$system_nccl_lib${LD_PRELOAD:+:$LD_PRELOAD}"
      fi
      echo "[system] NCCL: $system_nccl_lib"
      ;;
    0|false|FALSE|no|NO)
      echo "[system] NCCL: using the Conda/PyTorch bundled library"
      ;;
    *)
      die "USE_SYSTEM_NCCL must be true/false or 1/0, got: $use_system_nccl"
      ;;
  esac
}

visible_gpu_count() {
  if [[ -v CUDA_VISIBLE_DEVICES ]]; then
    [[ -n "$CUDA_VISIBLE_DEVICES" && "$CUDA_VISIBLE_DEVICES" != "-1" ]] \
      || die "CUDA_VISIBLE_DEVICES exposes no GPUs"
    local -a ids
    IFS=',' read -r -a ids <<< "$CUDA_VISIBLE_DEVICES"
    printf '%d\n' "${#ids[@]}"
    return
  fi

  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable"
  nvidia-smi --query-gpu=index --format=csv,noheader | wc -l
}

setup_wandb_login() {
  if [[ "$WANDB_MODE" != "online" ]]; then
    echo "[wandb] mode=$WANDB_MODE; login skipped"
    return 0
  fi
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[wandb] DRY_RUN=1; login skipped"
    return 0
  fi
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "[wandb] WANDB_API_KEY is unset; relying on existing credentials"
  fi
  "$PYTHON_BIN" - <<'PY'
import os
import wandb

logged_in = wandb.login(
    key=os.environ.get("WANDB_API_KEY") or None,
    relogin=bool(os.environ.get("WANDB_API_KEY")),
    host=os.environ.get("WANDB_BASE_URL") or None,
)
if not logged_in:
    raise RuntimeError(
        "W&B login failed. Set WANDB_API_KEY, or launch with WANDB_MODE=offline "
        "(sync later with `wandb sync`), or pass wandb.enabled=false."
    )
print("[wandb] login successful")
PY
}

[[ "$FUYAO_USER" != "CHANGE_ME@xiaopeng.com" ]] \
  || die "set FUYAO_USER (or edit the default at the top of this script)"
[[ -d "$REPO_ROOT" ]] || die "repository not found: $REPO_ROOT"
[[ -f "$REPO_ROOT/scripts/train_zero1.sh" ]] \
  || die "training launcher not found: $REPO_ROOT/scripts/train_zero1.sh"
cd "$REPO_ROOT"

if [[ "${NNODES:-1}" != "1" ]]; then
  die "this script supports one fuyao node; got NNODES=${NNODES}"
fi

nproc="${NPROC_PER_NODE:-8}"
if (( $# > 0 )) && is_positive_integer "$1"; then
  nproc="$1"
  shift
fi
is_positive_integer "$nproc" || die "NPROC_PER_NODE must be a positive integer, got: $nproc"

hydra_args=("$@")
if ! has_hydra_override "task" "${hydra_args[@]}"; then
  hydra_args=("task=$TASK_CONFIG" "${hydra_args[@]}")
fi

# wandb: enabled iff an API key is provided or mode is offline (pristine
# train.yaml defaults wandb.enabled=false). Explicit hydra overrides win.
wandb_enabled=false
if [[ -n "${WANDB_API_KEY:-}" || "$WANDB_MODE" != "online" ]]; then
  wandb_enabled=true
fi
if ! has_hydra_override "wandb.enabled" "${hydra_args[@]}"; then
  hydra_args+=("wandb.enabled=$wandb_enabled")
fi
if ! has_hydra_override "wandb.mode" "${hydra_args[@]}"; then
  hydra_args+=("wandb.mode=$WANDB_MODE")
fi
if [[ -n "$WANDB_ENTITY" ]] && ! has_hydra_override "wandb.workspace" "${hydra_args[@]}"; then
  hydra_args+=("wandb.workspace=$WANDB_ENTITY")
fi
if ! has_hydra_override "wandb.project" "${hydra_args[@]}"; then
  hydra_args+=("wandb.project=$WANDB_PROJECT")
fi

install_system_packages
activate_conda_wam
setup_nccl

gpu_count="$(visible_gpu_count)"
(( nproc <= gpu_count )) \
  || die "requested $nproc processes but only $gpu_count GPUs are visible"

if [[ "$wandb_enabled" == "true" ]]; then
  setup_wandb_login
fi

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  mkdir -p \
    "$XDG_CACHE_HOME" \
    "$HUGGINGFACE_HUB_CACHE" \
    "$TORCH_HOME/hub/checkpoints" \
    "$MODELSCOPE_CACHE" \
    "$RUNS_ROOT"
fi

echo "[fuyao] repository: $REPO_ROOT"
echo "[fuyao] Conda environment: $CONDA_ENV"
echo "[fuyao] Python: $PYTHON_BIN"
echo "[fuyao] cache root: $CACHE_ROOT"
echo "[fuyao] runs root: $RUNS_ROOT"
echo "[fuyao] Wan weights: $DIFFSYNTH_MODEL_BASE_PATH"
echo "[fuyao] visible GPUs: $gpu_count; processes: $nproc"
echo "[fuyao] W&B: enabled=$wandb_enabled entity=${WANDB_ENTITY:-<key-default>} project=$WANDB_PROJECT mode=$WANDB_MODE"

printf '[job] command: bash scripts/train_zero1.sh %q' "$nproc"
printf ' %q' "${hydra_args[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[job] DRY_RUN=1; training was not started"
  exit 0
fi

exec bash "$REPO_ROOT/scripts/train_zero1.sh" "$nproc" "${hydra_args[@]}"
