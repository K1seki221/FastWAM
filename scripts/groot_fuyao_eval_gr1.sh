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
  "$CLIENT_VENV/bin/python" gr00t/eval/rollout_policy.py \
    --n-episodes "$N_EPISODES" --policy-client-host 127.0.0.1 --policy-client-port "$PORT" \
    --max-episode-steps "$MAX_STEPS" --env-name "gr1_unified/${task}_GR1ArmsAndWaistFourierHands_Env" \
    --n-action-steps "$N_ACTION_STEPS" --n-envs "$N_ENVS" \
    --video-dir "$EVAL_OUT/videos/${task:0:60}" > "$log" 2>&1
  rc=$?
  sr=$(grep -a "success rate:" "$log" | tail -1 | grep -oE "[0-9.]+$" || echo "NA")
  [[ $rc -ne 0 ]] && sr="ERR"
  echo "$task,$sr" >> "$csv"
  echo "[eval-gr1]   -> $sr"
done

n_ok=$(tail -n +2 "$csv" | grep -vc ",ERR" || true)
mean=$(tail -n +2 "$csv" | grep -v ",ERR" | cut -d, -f2 | awk '{s+=$1; n+=1} END {if (n>0) printf "%.4f", s/n; else print "NA"}')
echo "tasks_ok=$n_ok mean_success=$mean" | tee "$EVAL_OUT/summary.txt"
echo "[eval-gr1] done -> $EVAL_OUT"
