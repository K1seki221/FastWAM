#!/usr/bin/env bash

# Fuyao single-node training entrypoint for FastWAM.
#
# Default training command:
#   bash scripts/train_zero1.sh 8 task=libero_future_cache_2cam224_1e-4
#
# Useful overrides:
#   NPROC_PER_NODE=4 bash scripts/train_fuyao.sh
#   WANDB_MODE=offline bash scripts/train_fuyao.sh 8 task=libero_joint_2cam224_1e-4
#   RUN_NAME=vip_v1 bash scripts/train_fuyao_fastwam.sh 8 task=libero_uncond_2cam224_1e-4
#   SKIP_APT_INSTALL=1 DRY_RUN=1 bash scripts/train_fuyao.sh

if [[ -z "${BASH_VERSION:-}" ]]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

DEFAULT_REPO_ROOT="/workspace/zhangxp7@xiaopeng.com/FastWAM"

export REPO_ROOT="${REPO_ROOT:-${WORKSPACE_ROOT:-$DEFAULT_REPO_ROOT}}"
export CONDA_ROOT="${CONDA_ROOT:-/dataset_rc/zhangxp7@xiaopeng.com/miniconda3}"
export CONDA_ENV="${CONDA_ENV:-$CONDA_ROOT/envs/wam}"
export PYTHON_BIN="${PYTHON_BIN:-$CONDA_ENV/bin/python}"
export ACCELERATE_ENTRYPOINT="${ACCELERATE_ENTRYPOINT:-$CONDA_ENV/bin/accelerate}"

export CACHE_ROOT="${CACHE_ROOT:-/dataset_rc/zhangxp7@xiaopeng.com/projects/fastwam/.cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT/torch}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$CACHE_ROOT/modelscope}"

export TASK_CONFIG="${TASK_CONFIG:-libero_future_cache_2cam224_1e-4}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-yumio-wam}"
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

  # The persistent environment was copied from /root/miniconda3 and generated
  # console scripts retain their original absolute shebang. Select the
  # environment directly and let train_zero1.sh invoke accelerate with this
  # environment's Python.
  unset PYTHONHOME || true
  unset CONDA_EXE _CONDA_EXE CONDA_PYTHON_EXE _CONDA_ROOT _CE_M _CE_CONDA || true
  export CONDA_PREFIX="$CONDA_ENV"
  export CONDA_DEFAULT_ENV="wam"
  export CONDA_SHLVL=1
  export PATH="$CONDA_ENV/bin:$PATH"
  export LD_LIBRARY_PATH="$CONDA_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export ACCELERATE_PYTHON="$PYTHON_BIN"

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
  local enabled="$1"
  local mode="$2"

  case "${enabled,,}" in
    0|false|no|off)
      echo "[wandb] disabled by Hydra configuration"
      return 0
      ;;
    1|true|yes|on)
      ;;
    *)
      die "wandb.enabled must be boolean, got: $enabled"
      ;;
  esac

  case "$mode" in
    offline|disabled)
      echo "[wandb] mode=$mode; login skipped"
      return 0
      ;;
    online)
      ;;
    *)
      die "unsupported W&B mode: $mode"
      ;;
  esac

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[wandb] DRY_RUN=1; login skipped"
    return 0
  fi

  local restore_xtrace=0
  if [[ "$-" == *x* ]]; then
    restore_xtrace=1
    set +x
  fi

  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    echo "[wandb] WANDB_API_KEY detected; authenticating without exposing it"
  elif [[ -n "${WANDB_IDENTITY_TOKEN_FILE:-}" ]]; then
    echo "[wandb] using WANDB_IDENTITY_TOKEN_FILE"
  else
    echo "[wandb] WANDB_API_KEY is unset; checking existing credentials"
  fi

  local status=0
  "$PYTHON_BIN" - "${WANDB_VERIFY_LOGIN:-1}" <<'PY' || status=$?
import os
import sys

import wandb

value = sys.argv[1].strip().lower()
if value in {"1", "true", "yes", "on"}:
    verify = True
elif value in {"0", "false", "no", "off"}:
    verify = False
else:
    raise ValueError(f"WANDB_VERIFY_LOGIN must be boolean, got: {sys.argv[1]}")

api_key = os.environ.get("WANDB_API_KEY") or None
logged_in = wandb.login(
    key=api_key,
    relogin=api_key is not None,
    host=os.environ.get("WANDB_BASE_URL") or None,
    verify=verify,
)
if not logged_in and not os.environ.get("WANDB_IDENTITY_TOKEN_FILE"):
    raise RuntimeError(
        "W&B login failed. Set WANDB_API_KEY/WANDB_IDENTITY_TOKEN_FILE "
        "or launch with WANDB_MODE=offline."
    )
print(f"[wandb] login successful (verification={'enabled' if verify else 'disabled'})")
PY

  if (( restore_xtrace == 1 )); then
    set -x
  fi
  return "$status"
}

[[ -d "$REPO_ROOT" ]] || die "repository not found: $REPO_ROOT"
[[ -f "$REPO_ROOT/scripts/train_zero1.sh" ]] \
  || die "training launcher not found: $REPO_ROOT/scripts/train_zero1.sh"
cd "$REPO_ROOT"

if [[ "${NNODES:-1}" != "1" ]]; then
  die "scripts/train_fuyao.sh currently supports one Fuyao node; got NNODES=${NNODES}"
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
if ! has_hydra_override "wandb.mode" "${hydra_args[@]}"; then
  hydra_args+=("wandb.mode=$WANDB_MODE")
fi
if ! has_hydra_override "wandb.workspace" "${hydra_args[@]}"; then
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

wandb_enabled="$(hydra_override_value "wandb.enabled" "true" "${hydra_args[@]}")"
wandb_mode="$(hydra_override_value "wandb.mode" "$WANDB_MODE" "${hydra_args[@]}")"
wandb_entity="$(hydra_override_value "wandb.workspace" "$WANDB_ENTITY" "${hydra_args[@]}")"
wandb_project="$(hydra_override_value "wandb.project" "$WANDB_PROJECT" "${hydra_args[@]}")"
setup_wandb_login "$wandb_enabled" "$wandb_mode"

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  mkdir -p \
    "$XDG_CACHE_HOME" \
    "$HUGGINGFACE_HUB_CACHE" \
    "$TORCH_HOME/hub/checkpoints" \
    "$MODELSCOPE_CACHE"
fi

echo "[fuyao] repository: $REPO_ROOT"
echo "[fuyao] Conda environment: $CONDA_ENV"
echo "[fuyao] Python: $PYTHON_BIN"
echo "[fuyao] cache root: $CACHE_ROOT"
echo "[fuyao] rendering: MUJOCO_GL=$MUJOCO_GL PYOPENGL_PLATFORM=$PYOPENGL_PLATFORM"
echo "[fuyao] visible GPUs: $gpu_count; processes: $nproc"
echo "[fuyao] W&B: entity=$wandb_entity project=$wandb_project mode=$wandb_mode"

printf '[job] command: bash scripts/train_zero1.sh %q' "$nproc"
printf ' %q' "${hydra_args[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[job] DRY_RUN=1; training was not started"
  exit 0
fi

exec bash "$REPO_ROOT/scripts/train_zero1.sh" "$nproc" "${hydra_args[@]}"
