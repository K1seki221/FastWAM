#!/usr/bin/env bash
# In-container LIBERO eval runner for GR00T checkpoints on fuyao.
# Server (training venv, GPU) + rollout client (libero venv, EGL render).
# Env knobs: CKPT (required), RUN_NAME, SUITES, N_EPISODES, N_ENVS, MAX_STEPS,
#            N_ACTION_STEPS, PORT, EVAL_OUT, SKIP_APT_INSTALL.
set -uo pipefail

export FUYAO_USER="${FUYAO_USER:-ruijie.zhang@xiaopeng.com}"
export REPO_ROOT="${REPO_ROOT:-/dataset_rc/${FUYAO_USER}/FastWAM}"
GROOT="${GROOT:-$REPO_ROOT/Isaac-GR00T}"
VENV_DIR="${VENV_DIR:-/dataset_rc/${FUYAO_USER}/projects/groot/venv}"
LIBERO_VENV="${LIBERO_VENV:-$GROOT/gr00t/eval/sim/LIBERO/libero_uv/.venv}"

export HF_HOME="${GROOT_HF_HOME:-/dataset_rc/${FUYAO_USER}/hf}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

CKPT="${CKPT:?CKPT (checkpoint dir) required}"
RUN_NAME="${RUN_NAME:-eval_$(basename "$(dirname "$CKPT")")_$(basename "$CKPT")}"
SUITES="${SUITES:-10 goal object spatial}"
N_EPISODES="${N_EPISODES:-10}"
N_ENVS="${N_ENVS:-5}"
MAX_STEPS="${MAX_STEPS:-720}"
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"
# Job containers share the host network namespace (iron_vla LCM lesson): a
# fixed port collides when two eval jobs land on one node. Randomize per job.
PORT="${PORT:-$((20000 + RANDOM % 20000))}"
EVAL_OUT="${EVAL_OUT:-/dataset_rc/${FUYAO_USER}/projects/groot_evals/${RUN_NAME}}"

[[ -f "$CKPT/config.json" ]] || { echo "bad CKPT: $CKPT" >&2; exit 1; }
[[ -x "$VENV_DIR/bin/python" ]] || { echo "training venv missing" >&2; exit 1; }
[[ -x "$LIBERO_VENV/bin/python" ]] || { echo "libero venv missing" >&2; exit 1; }
mkdir -p "$EVAL_OUT/logs"

# EGL/GL stack (job containers lack it; same set the FastWAM eval wrappers install)
if [[ "${SKIP_APT_INSTALL:-0}" != "1" ]]; then
  if ! ldconfig -p | grep -q libEGL.so.1; then
    sed -i "s|http://.*archive.ubuntu.com|http://mirrors.cloud.aliyuncs.com|g; s|http://security.ubuntu.com|http://mirrors.cloud.aliyuncs.com|g" /etc/apt/sources.list 2>/dev/null || true
    apt-get update -qq || true
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libegl1 libopengl0 libglvnd0 libgl1 libglu1-mesa libosmesa6 || true
  fi
fi
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

# LIBERO's first import calls input() when no config exists (fatal headless)
# and its makedirs races across vector-env workers. Point it at the
# pre-generated config on shared storage.
export LIBERO_CONFIG_PATH="/dataset_rc/${FUYAO_USER}/projects/groot/libero_config"
[[ -f "$LIBERO_CONFIG_PATH/config.yaml" ]] || { echo "missing $LIBERO_CONFIG_PATH/config.yaml" >&2; exit 1; }

# ---- task lists (examples/LIBERO/README.md, GR00T N1.7) ----
tasks_10=(
  LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket
  LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket
  KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it
  KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it
  LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate
  STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy
  LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate
  LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket
  KITCHEN_SCENE8_put_both_moka_pots_on_the_stove
  KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it
)
tasks_goal=(
  open_the_middle_drawer_of_the_cabinet
  put_the_bowl_on_the_stove
  put_the_wine_bottle_on_top_of_the_cabinet
  open_the_top_drawer_and_put_the_bowl_inside
  put_the_bowl_on_top_of_the_cabinet
  push_the_plate_to_the_front_of_the_stove
  put_the_cream_cheese_in_the_bowl
  turn_on_the_stove
  put_the_bowl_on_the_plate
  put_the_wine_bottle_on_the_rack
)
tasks_object=(
  pick_up_the_alphabet_soup_and_place_it_in_the_basket
  pick_up_the_cream_cheese_and_place_it_in_the_basket
  pick_up_the_salad_dressing_and_place_it_in_the_basket
  pick_up_the_bbq_sauce_and_place_it_in_the_basket
  pick_up_the_ketchup_and_place_it_in_the_basket
  pick_up_the_tomato_sauce_and_place_it_in_the_basket
  pick_up_the_butter_and_place_it_in_the_basket
  pick_up_the_milk_and_place_it_in_the_basket
  pick_up_the_chocolate_pudding_and_place_it_in_the_basket
  pick_up_the_orange_juice_and_place_it_in_the_basket
)
tasks_spatial=(
  pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate
  pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate
  pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate
  pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate
  pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate
  pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate
  pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate
  pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate
  pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate
  pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate
)

# ---- server ----
cd "$GROOT"
echo "[eval] starting server for $CKPT"
"$VENV_DIR/bin/python" gr00t/eval/run_gr00t_server.py \
  --model-path "$CKPT" --embodiment-tag LIBERO_PANDA --use-sim-policy-wrapper \
  --port "$PORT" > "$EVAL_OUT/logs/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null' EXIT

for i in $(seq 1 120); do
  if ! kill -0 $SERVER_PID 2>/dev/null; then echo "[eval] server died:"; tail -20 "$EVAL_OUT/logs/server.log"; exit 1; fi
  "$LIBERO_VENV/bin/python" - <<PY 2>/dev/null && break
import socket; s=socket.socket(); s.settimeout(2); s.connect(("127.0.0.1", $PORT)); s.close()
PY
  sleep 5
done
echo "[eval] server up (pid $SERVER_PID)"

# ---- rollout loop ----
csv="$EVAL_OUT/results.csv"
echo "suite,task,success_rate" > "$csv"
for suite in $SUITES; do
  arr="tasks_${suite}[@]"
  for task in "${!arr}"; do
    log="$EVAL_OUT/logs/${suite}__${task:0:80}.log"
    echo "[eval] $suite / $task"
    "$LIBERO_VENV/bin/python" gr00t/eval/rollout_policy.py \
      --n-episodes "$N_EPISODES" --policy-client-host 127.0.0.1 --policy-client-port "$PORT" \
      --max-episode-steps "$MAX_STEPS" --env-name "libero_sim/$task" \
      --n-action-steps "$N_ACTION_STEPS" --n-envs "$N_ENVS" \
      --video-dir "$EVAL_OUT/videos/${suite}__${task:0:60}" > "$log" 2>&1
    rc=$?
    sr=$(grep -a "success rate:" "$log" | tail -1 | grep -oE "[0-9.]+$" || echo "NA")
    [[ $rc -ne 0 ]] && sr="ERR"
    echo "$suite,$task,$sr" >> "$csv"
    echo "[eval]   -> $sr"
  done
done

# ---- summary ----
python3 - "$csv" > "$EVAL_OUT/summary.txt" <<PY
import csv, sys
from collections import defaultdict
rows = list(csv.DictReader(open(sys.argv[1])))
by = defaultdict(list)
bad = 0
for r in rows:
    try:
        by[r["suite"]].append(float(r["success_rate"]))
    except ValueError:
        bad += 1
means = {s: sum(v)/len(v) for s, v in by.items() if v}
for s, m in sorted(means.items()):
    print(f"{s}: {m:.4f} (n={len(by[s])})")
if means:
    print(f"OVERALL (suite mean): {sum(means.values())/len(means):.4f}")
if bad:
    print(f"WARNING: {bad} tasks with missing/error results")
PY
cat "$EVAL_OUT/summary.txt"
echo "[eval] done -> $EVAL_OUT"
