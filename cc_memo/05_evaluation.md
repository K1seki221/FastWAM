# Evaluation / experiments

Shared model contract: `model.infer_action(prompt, input_image, action_horizon, proprio, …, seed, …[, num_video_frames])` → `{"action": [T,D]}` normalized; `num_video_frames` passed only if present in `inspect.signature` (variant-dependent). Default `action_horizon = num_frames−1 = 32`, `num_video_frames = 9`, `num_inference_steps = 10`, prompt = `DEFAULT_PROMPT.format(task=…)`.

## LIBERO (`experiments/libero/`)

Flow: `run_libero_manager.py` (hydra, `sim_libero.yaml`; **requires `ckpt=` on CLI**; omitting `task=` silently uses the default `libero_uncond_2cam224_1e-4`) → writes task list `{suite},{task_id}` → env-var-parameterized `run_libero_parallel_test.sh` → tmux session **`libero_test_v3`** with one pane per task, scheduled across GPUs (least-loaded, `MULTIRUN.num_gpus` × `max_tasks_per_gpu`) → each pane runs `eval_libero_single.py task=… ckpt=… EVALUATION.task_suite_name=… EVALUATION.task_id=… CUDA_VISIBLE_DEVICES=<g>` → per-task `gpu{g}_task{t}_results.json` → on full success `summarize_results.py` → `summary.csv` / `summary.json` / `task_success_rates.csv` under `evaluate_results/libero/<task_choice>/<ts>/`.

Worker episode loop (`eval_libero_single.py`):
- Env: LIBERO `OffScreenRenderEnv` at 256×256, `env_num` must be 1. Max steps: 400 (spatial/object/goal) / 700 (libero_10/90). First `num_steps_wait=30` steps send dummy action `[0,0,0,0,0,0,-1]`.
- Obs → input: agentview + wrist images **rotated 180°** (`[::-1,::-1]`, matches training), center-crop-resize to per-camera shape, horizontal concat → [1,3,224,448], scaled to [-1,1]. Proprio = eef_pos(3) + axis-angle(3) + gripper_qpos(2) = 8-D, normalized via the processor.
- Replan: when queue empty, `infer_action` → denormalize → **gripper remap: `a[-1]·2−1`, then negate (`invert_gripper_action`), then optional `np.sign`** (`binarize_gripper: true`) → take first `replan_steps=10` (or ensembler mean).
- `ActionEnsembler`: plain unweighted mean of all chunks overlapping a timestep; `_cleanup` is dead code; disabled by default.
- `visualize_future_video=true` switches to `infer_joint` and logs future-video PSNR (requires `action_conditioned=false`).
- `dataset_stats.json` auto-discovery: explicit `EVALUATION.dataset_stats_path`, else searched in the first 4 parent dirs of `ckpt`.

Scheduler caveats: kills any existing `libero_test_v3` tmux session at start (concurrent runs clobber); completion detected by existence of `gpu*_task*_results.json` (re-running with same OUTPUT_DIR = crude resume); ANY task failure stops the scheduler; manager forwards all extra hydra overrides to workers except `task, ckpt, gpu_id, EVALUATION.task_suite_name, EVALUATION.task_id, MULTIRUN.*, hydra.*`. `summarize_results.py`: "Overall" = **unweighted mean of per-suite rates**; reads `CKPT`/`CONFIG` env vars for labels; `summary.csv` starts with a non-CSV title line.

## RoboTwin (`experiments/robotwin/`)

Flow: `run_robotwin_manager.py` (hydra, `sim_robotwin.yaml`) → task list from `third_party/RoboTwin/task_config/_eval_step_limit.yml` (ALL tasks when `EVALUATION.task_name` null) → per task two phases: **clean** (`demo_clean`) then **random** (`demo_randomized`), random always launched on the same GPU after clean succeeds → subprocess `eval_robotwin_single.py` → ensures symlink `third_party/RoboTwin/policy/fastwam_policy → experiments/robotwin/fastwam_policy` → shells out to `script/eval_policy.py` (cwd=`third_party/RoboTwin`) with `--overrides --key value` pairs (values `eval()`'d on the RoboTwin side; strings repr-quoted) → success rate parsed as the **last float-parseable line** of `<task>/_result_{clean|random}.txt` → `summary.{csv,json}` under `evaluate_results/robotwin/<ckpt_tag>/<run_ts>/`.

Policy plugin (`experiments/robotwin/fastwam_policy/deploy_policy.py`), RoboTwin interface = module functions:
- `get_model(usr_args)` → `WorldActionRobotWinPolicy`: re-composes the FastWAM hydra cfg itself (`initialize_config_dir` + `compose("sim_robotwin.yaml", overrides=["task=<sim_task>"])`), forces `load_text_encoder=True`, loads ckpt, builds `FastWAMProcessor` + stats (**`dataset_stats_path` required**). `replan_steps` clamped to [1, action_horizon].
- `eval(TASK_ENV, model, observation)` → `model.step(...)`: if queue empty → `get_instruction()`, stitch obs (head 320×256 top; left|right wrists 160×128 bottom → [1,3,384,320]), proprio = `joint_action.vector` (dual-arm qpos), `infer_action`, enqueue first `replan_steps=24`; each step `take_action(action, action_type="qpos")`. **No gripper remap** (unlike LIBERO).
- `reset_model(model)` → clears queue/counters.
- `should_request_observation()` (queue empty) powers `skip_get_obs_within_replan=true` — RoboTwin only renders obs when needed (fast eval, low-FPS saved videos by design).

RoboTwin harness facts: evaluates only **expert-verified seeds** (scripted expert must solve the seed first; seeds start at `100000·(1+seed)`) — "100 episodes" = 100 expert-solvable seeds. `instruction_type: unseen` (default, follows Motus) samples held-out language paraphrases; `seen` adds ~1–2 points (Lingbot-VA comparison caveat). Any worker failure → manager SIGTERMs everything and raises (fail-fast). Manager override blocklist: `ckpt, gpu_id, EVALUATION.task_name, EVALUATION.task_config, EVALUATION.output_dir, MULTIRUN.*, hydra.*` — note `task=` IS forwarded to workers (unlike LIBERO, where it's blocked and passed explicitly).

**Provisioning:** `third_party/RoboTwin/task_config/` and `assets/` are gitignored and ABSENT in a fresh checkout (`data/` exists but only holds the tracked `process_stuck.py`) — RoboTwin eval (including manager task discovery) cannot run until the official RoboTwin install/download steps are done. `policy/` upstream policies are stripped; only the fastwam symlink goes there. The docstring mentioning `policy/fastwam` in `eval_robotwin_single.py` is stale — the real name is `fastwam_policy`.

## Released-checkpoint eval commands (from README)

```bash
python experiments/libero/run_libero_manager.py task=libero_uncond_2cam224_1e-4 \
  ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  MULTIRUN.num_gpus=8
python experiments/robotwin/run_robotwin_manager.py task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
  MULTIRUN.num_gpus=8
```
