#!/usr/bin/env bash
# In-container RoboCasa-GR1-tabletop eval runner for GR00T checkpoints on fuyao.
# Server (training venv, GPU) + rollout client (robocasa venv, EGL render).
# Env knobs: CKPT (required), RUN_NAME, N_EPISODES, N_ENVS, MAX_STEPS,
#            N_ACTION_STEPS, PORT, EVAL_OUT, SKIP_APT_INSTALL, TASKS_SUBSET.
set -uo pipefail

export FUYAO_USER="${FUYAO_USER:-ruijie.zhang@xiaopeng.com}"
export REPO_ROOT="${REPO_ROOT:-/dataset_rc/${FUYAO_USER}/FastWAM}"
GROOT="${GROOT:-$REPO_ROOT/Isaac-GR00T}"
VENV_DIR="${VENV_DIR:-/dataset_rc/${FUYAO_USER}/projects/groot/venv}"
CLIENT_VENV="${CLIENT_VENV:-$GROOT/gr00t/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv}"

export HF_HOME="${GROOT_HF_HOME:-/dataset_rc/${FUYAO_USER}/hf}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

CKPT="${CKPT:?CKPT (checkpoint dir) required}"
RUN_NAME="${RUN_NAME:-eval_gr1_$(basename "$(dirname "$CKPT")")_$(basename "$CKPT")}"
N_EPISODES="${N_EPISODES:-10}"
N_ENVS="${N_ENVS:-5}"
MAX_STEPS="${MAX_STEPS:-720}"
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"
# Shared host network namespace: randomize per job to avoid collisions.
PORT="${PORT:-$((20000 + RANDOM % 20000))}"
EVAL_OUT="${EVAL_OUT:-/dataset_rc/${FUYAO_USER}/projects/groot_evals/${RUN_NAME}}"

[[ -f "$CKPT/config.json" ]] || { echo "bad CKPT: $CKPT" >&2; exit 1; }
[[ -x "$VENV_DIR/bin/python" ]] || { echo "training venv missing" >&2; exit 1; }
[[ -x "$CLIENT_VENV/bin/python" ]] || { echo "robocasa client venv missing" >&2; exit 1; }
mkdir -p "$EVAL_OUT/logs"

if [[ "${SKIP_APT_INSTALL:-0}" != "1" ]]; then
  if ! ldconfig -p | grep -q libEGL.so.1; then
    sed -i "s|http://.*archive.ubuntu.com|http://mirrors.cloud.aliyuncs.com|g; s|http://security.ubuntu.com|http://mirrors.cloud.aliyuncs.com|g" /etc/apt/sources.list 2>/dev/null || true
    apt-get update -qq || true
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libegl1 libopengl0 libglvnd0 libgl1 libglu1-mesa libosmesa6 || true
  fi
fi
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

# 24 GR1 tabletop tasks (examples/robocasa-gr1-tabletop-tasks/README.md)
tasks=(
  PnPBottleToCabinetClose
  PnPCanToDrawerClose
  PnPCupToDrawerClose
  PnPMilkToMicrowaveClose
  PnPPotatoToMicrowaveClose
  PnPWineToCabinetClose
  PosttrainPnPNovelFromCuttingboardToBasketSplitA
  PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA
  PosttrainPnPNovelFromCuttingboardToPanSplitA
  PosttrainPnPNovelFromCuttingboardToPotSplitA
  PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA
  PosttrainPnPNovelFromPlacematToBasketSplitA
  PosttrainPnPNovelFromPlacematToBowlSplitA
  PosttrainPnPNovelFromPlacematToPlateSplitA
  PosttrainPnPNovelFromPlacematToTieredshelfSplitA
  PosttrainPnPNovelFromPlateToBowlSplitA
  PosttrainPnPNovelFromPlateToCardboardboxSplitA
  PosttrainPnPNovelFromPlateToPanSplitA
  PosttrainPnPNovelFromPlateToPlateSplitA
  PosttrainPnPNovelFromTrayToCardboardboxSplitA
  PosttrainPnPNovelFromTrayToPlateSplitA
  PosttrainPnPNovelFromTrayToPotSplitA
  PosttrainPnPNovelFromTrayToTieredbasketSplitA
  PosttrainPnPNovelFromTrayToTieredshelfSplitA
)
if [[ -n "${TASKS_SUBSET:-}" ]]; then
  read -r -a tasks <<< "$TASKS_SUBSET"
fi

cd "$GROOT"
echo "[eval-gr1] starting server for $CKPT (port $PORT)"
# Cap server CPU threads + run nice: the zmq serve loop busy-spins across
# all cores otherwise (~800% CPU) and starves co-located training jobs.
nice -n 15 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
"$VENV_DIR/bin/python" gr00t/eval/run_gr00t_server.py \
  --model-path "$CKPT" --embodiment-tag ROBOCASA_GR1_TABLETOP --use-sim-policy-wrapper \
  --port "$PORT" > "$EVAL_OUT/logs/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null' EXIT

for i in $(seq 1 120); do
  if ! kill -0 $SERVER_PID 2>/dev/null; then echo "[eval-gr1] server died:"; tail -20 "$EVAL_OUT/logs/server.log"; exit 1; fi
  "$CLIENT_VENV/bin/python" - <<PY 2>/dev/null && break
import socket; s=socket.socket(); s.settimeout(2); s.connect(("127.0.0.1", $PORT)); s.close()
PY
  sleep 5
done
echo "[eval-gr1] server up (pid $SERVER_PID)"

csv="$EVAL_OUT/results.csv"
echo "task,success_rate" > "$csv"
for task in "${tasks[@]}"; do
  log="$EVAL_OUT/logs/${task:0:90}.log"
  echo "[eval-gr1] $task"
  # NO_VIDEO=1 disables recording (the writer can kill forked sim workers on
  # some hosts; success rates don't need it).
  if [[ "${NO_VIDEO:-0}" == "1" ]]; then vdir="none"; else vdir="$EVAL_OUT/videos/${task:0:60}"; fi
  # CRITICAL for parallel eval: CUDA_VISIBLE_DEVICES does NOT restrict NVIDIA
  # EGL enumeration — without MUJOCO_EGL_DEVICE_ID every client renders on
  # physical device 0 and concurrent clients SIGABRT in read_pixels. Pin the
  # client to its own device and strip CVD so it cannot mislead.
  # 2026-08-12 findings: (a) with both NVIDIA and Mesa glvnd ICDs installed,
  # unpinned enumeration returns 16 devices and loads Mesa's dri2 path on
  # NVIDIA nodes — the documented 570-series concurrent-EGL crasher. Pin the
  # NVIDIA ICD. (b) EGL device order is PCI-bus order, NOT CUDA order — the
  # requested CUDA ordinal must be translated via EGL_CUDA_DEVICE_NV.
  export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"
  want_cuda="${EGL_DEVICE_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
  # env -u CUDA_VISIBLE_DEVICES is REQUIRED here: with CVD set, the driver
  # renumbers EGL_CUDA_DEVICE_NV to process-visible ordinals and the lookup
  # fails -> silent fallback -> renders land on the wrong physical GPU.
  egl_dev=$(env -u CUDA_VISIBLE_DEVICES "$CLIENT_VENV/bin/python" "$REPO_ROOT/scripts/egl_cuda_map.py" "$want_cuda" 2>/dev/null) || { echo "[eval-gr1] WARN: EGL map failed for CUDA $want_cuda, using raw index" >&2; egl_dev="$want_cuda"; }
  echo "[eval-gr1] render: CUDA $want_cuda -> EGL idx $egl_dev"
  # Cross-process creation lock: concurrent EGL context creation (even on
  # distinct devices) collides with other clients on this host. Hold a global
  # lock through env construction + first renders, then release — steady-state
  # rendering may overlap.
  # Concurrent EGL renderers on this host collide sporadically (silent
  # SIGABRT in read_pixels) — both at context creation AND mid-run. The
  # creation lock narrows the window; MAX_ATTEMPTS retries absorb the rest.
  rc=1
  for attempt in $(seq 1 "${MAX_ATTEMPTS:-3}"); do
    exec 9>/tmp/groot_egl_create.lock
    flock -w 1800 9
    setsid env -u CUDA_VISIBLE_DEVICES MUJOCO_EGL_DEVICE_ID="$egl_dev" \
    "$CLIENT_VENV/bin/python" gr00t/eval/rollout_policy.py \
      --n-episodes "$N_EPISODES" --policy-client-host 127.0.0.1 --policy-client-port "$PORT" \
      --max-episode-steps "$MAX_STEPS" --env-name "gr1_unified/${task}_GR1ArmsAndWaistFourierHands_Env" \
      --n-action-steps "$N_ACTION_STEPS" --n-envs "$N_ENVS" \
      --video-dir "$vdir" > "$log" 2>&1 &
    client_pid=$!
    sleep "${EGL_CREATE_GRACE:-90}"
    flock -u 9
    # Watchdog: a crashed spawn worker can deadlock AsyncVectorEnv.close()
    # in the parent forever (seen 2026-08-13). Bound the task wall time and
    # kill the whole process group on expiry (setsid gives it its own pgid).
    deadline=$(( $(date +%s) + ${TASK_TIMEOUT:-3600} ))
    while kill -0 "$client_pid" 2>/dev/null; do
      if [ "$(date +%s)" -gt "$deadline" ]; then
        echo "[eval-gr1]   TIMEOUT after ${TASK_TIMEOUT:-3600}s — killing client group $client_pid"
        kill -9 -- "-$client_pid" 2>/dev/null
        break
      fi
      sleep 20
    done
    wait $client_pid
    rc=$?
    [ $rc -eq 0 ] && break
    echo "[eval-gr1]   attempt $attempt failed (rc=$rc), retrying $task"
  done
  sr=$(grep -a "success rate:" "$log" | tail -1 | grep -oE "[0-9.]+$" || echo "NA")
  [[ $rc -ne 0 ]] && sr="ERR"
  echo "$task,$sr" >> "$csv"
  echo "[eval-gr1]   -> $sr"
done

n_ok=$(tail -n +2 "$csv" | grep -vc ",ERR" || true)
mean=$(tail -n +2 "$csv" | grep -v ",ERR" | cut -d, -f2 | awk '{s+=$1; n+=1} END {if (n>0) printf "%.4f", s/n; else print "NA"}')
echo "tasks_ok=$n_ok mean_success=$mean" | tee "$EVAL_OUT/summary.txt"
echo "[eval-gr1] done -> $EVAL_OUT"
