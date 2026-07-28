#!/usr/bin/env bash

# Fuyao single-node LIBERO evaluation entrypoint for FastWAM (adapted from
# Xiaopeng Zhang's fork; infrastructure only).
#
# Submit example:
#   fuyao deploy --gpus-per-node=8 --nodes=1 --project=<project> --site=<site> \
#     --docker-image <image> --experiment <name> -- \
#     FUYAO_USER=<you>@xiaopeng.com \
#     bash /workspace/<you>@xiaopeng.com/FastWAM/scripts/eval_fuyao_libero.sh \
#       task=libero_uncond_2cam224_1e-4 \
#       ckpt=/dataset_rc/<you>@xiaopeng.com/projects/fastwam/runs/.../checkpoints/weights/step_XXXXXX.pt \
#       EVALUATION.num_trials=50
#
# Env-var equivalents: TASK_CONFIG, CKPT, NUM_TRIALS, MAX_TASKS_PER_GPU, EVAL_OUTPUT_DIR.
# Inspect without side effects: DRY_RUN=1 bash scripts/eval_fuyao_libero.sh task=... ckpt=...

if [[ -z "${BASH_VERSION:-}" ]]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

export FUYAO_USER="${FUYAO_USER:-ruijie.zhang@xiaopeng.com}"

DEFAULT_REPO_ROOT="/dataset_rc/${FUYAO_USER}/FastWAM"
export REPO_ROOT="${REPO_ROOT:-${WORKSPACE_ROOT:-$DEFAULT_REPO_ROOT}}"
export CONDA_ROOT="${CONDA_ROOT:-/dataset_rc/${FUYAO_USER}/miniconda3}"
export CONDA_ENV="${CONDA_ENV:-$CONDA_ROOT/envs/wam}"
export PYTHON_BIN="${PYTHON_BIN:-$CONDA_ENV/bin/python}"

export CACHE_ROOT="${CACHE_ROOT:-/dataset_rc/${FUYAO_USER}/projects/fastwam/.cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT/torch}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$CACHE_ROOT/modelscope}"

# Eval loads the T5 text encoder live (sim_libero.yaml: load_text_encoder=true),
# so the shared Wan weights must be reachable.
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/dataset_rc/vlm/fm_models/VIT/WAM/FastWAM/checkpoints}"

export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/tmp/fastwam-libero-config}"
export TASK_CONFIG="${TASK_CONFIG:-libero_uncond_2cam224_1e-4}"
export CKPT="${CKPT:-}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}"

die() {
  echo "[fuyao-eval] error: $*" >&2
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
  local -a packages=(libegl1 libopengl0 libglvnd0 libgl1 libglu1-mesa tmux)

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
    echo "[system] EGL/OpenGL and tmux packages are already installed"
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

  unset PYTHONHOME || true
  unset CONDA_EXE _CONDA_EXE CONDA_PYTHON_EXE _CONDA_ROOT _CE_M _CE_CONDA || true
  export CONDA_PREFIX="$CONDA_ENV"
  export CONDA_DEFAULT_ENV="$(basename "$CONDA_ENV")"
  export CONDA_SHLVL=1
  export PATH="$CONDA_ENV/bin:$PATH"
  export LD_LIBRARY_PATH="$CONDA_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

  # Vendored LIBERO first, then this checkout's fastwam (wins over any editable
  # install baked into a borrowed conda env).
  export PYTHONPATH="$REPO_ROOT/third_party/LIBERO:$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
  hash -r
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

prepare_libero_config() {
  local benchmark_root="$REPO_ROOT/third_party/LIBERO/libero/libero"
  local config_file="$LIBERO_CONFIG_PATH/config.yaml"

  [[ -d "$benchmark_root" ]] \
    || die "vendored LIBERO benchmark not found: $benchmark_root"
  if [[ -s "$config_file" ]]; then
    echo "[libero] using existing config: $config_file"
    return 0
  fi
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[libero] DRY_RUN=1; would create config: $config_file"
    return 0
  fi

  mkdir -p "$LIBERO_CONFIG_PATH"
  {
    printf 'assets: %s/assets\n' "$benchmark_root"
    printf 'bddl_files: %s/bddl_files\n' "$benchmark_root"
    printf 'benchmark_root: %s\n' "$benchmark_root"
    printf 'datasets: %s/../datasets\n' "$benchmark_root"
    printf 'init_states: %s/init_files\n' "$benchmark_root"
  } > "$config_file"
  echo "[libero] created config: $config_file"
}

[[ "$FUYAO_USER" != "CHANGE_ME@xiaopeng.com" ]] \
  || die "set FUYAO_USER (or edit the default at the top of this script)"
[[ -d "$REPO_ROOT" ]] || die "repository not found: $REPO_ROOT"
[[ -f "$REPO_ROOT/experiments/libero/run_libero_manager.py" ]] \
  || die "LIBERO manager not found under: $REPO_ROOT"
[[ -f "$REPO_ROOT/experiments/libero/run_libero_parallel_test.sh" ]] \
  || die "LIBERO parallel launcher not found under: $REPO_ROOT"
cd "$REPO_ROOT"

if [[ "${NNODES:-1}" != "1" ]]; then
  die "LIBERO manager supports a single fuyao node; got NNODES=${NNODES}"
fi

hydra_args=("$@")
if ! has_hydra_override "task" "${hydra_args[@]}"; then
  hydra_args=("task=$TASK_CONFIG" "${hydra_args[@]}")
fi
if ! has_hydra_override "ckpt" "${hydra_args[@]}"; then
  [[ -n "$CKPT" ]] \
    || die "checkpoint is required; pass ckpt=/path/to/weights.pt or set CKPT"
  hydra_args+=("ckpt=$CKPT")
fi

install_system_packages
activate_conda_wam
prepare_libero_config

gpu_count="$(visible_gpu_count)"
is_positive_integer "$gpu_count" || die "invalid visible GPU count: $gpu_count"
if ! has_hydra_override "MULTIRUN.num_gpus" "${hydra_args[@]}"; then
  hydra_args+=("MULTIRUN.num_gpus=$gpu_count")
fi
if [[ -n "${NUM_TRIALS:-}" ]] \
  && ! has_hydra_override "EVALUATION.num_trials" "${hydra_args[@]}"; then
  is_positive_integer "$NUM_TRIALS" \
    || die "NUM_TRIALS must be a positive integer, got: $NUM_TRIALS"
  hydra_args+=("EVALUATION.num_trials=$NUM_TRIALS")
fi
if [[ -n "${MAX_TASKS_PER_GPU:-}" ]] \
  && ! has_hydra_override "MULTIRUN.max_tasks_per_gpu" "${hydra_args[@]}"; then
  is_positive_integer "$MAX_TASKS_PER_GPU" \
    || die "MAX_TASKS_PER_GPU must be a positive integer, got: $MAX_TASKS_PER_GPU"
  hydra_args+=("MULTIRUN.max_tasks_per_gpu=$MAX_TASKS_PER_GPU")
fi
if [[ -n "${EVAL_OUTPUT_DIR:-}" ]] \
  && ! has_hydra_override "EVALUATION.output_dir" "${hydra_args[@]}"; then
  hydra_args+=("EVALUATION.output_dir=$EVAL_OUTPUT_DIR")
fi

task_value="$(hydra_override_value "task" "$TASK_CONFIG" "${hydra_args[@]}")"
ckpt_value="$(hydra_override_value "ckpt" "$CKPT" "${hydra_args[@]}")"
requested_gpu_count="$(
  hydra_override_value "MULTIRUN.num_gpus" "$gpu_count" "${hydra_args[@]}"
)"
is_positive_integer "$requested_gpu_count" \
  || die "MULTIRUN.num_gpus must be a positive integer, got: $requested_gpu_count"
[[ "$requested_gpu_count" == "$gpu_count" ]] \
  || die "MULTIRUN.num_gpus=$requested_gpu_count must match the $gpu_count visible fuyao GPUs; adjust the fuyao resource request or CUDA_VISIBLE_DEVICES"
task_config_file="$REPO_ROOT/configs/task/${task_value#task/}"
task_config_file="${task_config_file%.yaml}.yaml"
[[ -f "$task_config_file" ]] \
  || die "task config not found: $task_config_file"
[[ -f "$ckpt_value" ]] || die "checkpoint not found: $ckpt_value"

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  command -v tmux >/dev/null 2>&1 \
    || die "tmux is required by run_libero_parallel_test.sh"
  mkdir -p \
    "$XDG_CACHE_HOME" \
    "$HUGGINGFACE_HUB_CACHE" \
    "$TORCH_HOME/hub/checkpoints" \
    "$MODELSCOPE_CACHE"
  "$PYTHON_BIN" -c \
    'import hydra, libero, torch; assert torch.cuda.is_available(); print(f"[env] torch={torch.__version__}, CUDA GPUs={torch.cuda.device_count()}")'
fi

echo "[fuyao-eval] repository: $REPO_ROOT"
echo "[fuyao-eval] Conda environment: $CONDA_ENV"
echo "[fuyao-eval] Python: $PYTHON_BIN"
echo "[fuyao-eval] LIBERO config: $LIBERO_CONFIG_PATH/config.yaml"
echo "[fuyao-eval] cache root: $CACHE_ROOT"
echo "[fuyao-eval] Wan weights: $DIFFSYNTH_MODEL_BASE_PATH"
echo "[fuyao-eval] rendering: MUJOCO_GL=$MUJOCO_GL PYOPENGL_PLATFORM=$PYOPENGL_PLATFORM"
echo "[fuyao-eval] visible GPUs: $gpu_count"
echo "[fuyao-eval] task: $task_value"
echo "[fuyao-eval] checkpoint: $ckpt_value"

printf '[job] command: %q experiments/libero/run_libero_manager.py' "$PYTHON_BIN"
printf ' %q' "${hydra_args[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[job] DRY_RUN=1; LIBERO evaluation was not started"
  exit 0
fi

exec "$PYTHON_BIN" experiments/libero/run_libero_manager.py "${hydra_args[@]}"
