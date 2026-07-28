# FastWAM — Overview

Official codebase for **"Fast-WAM: Do World Action Models Need Test-time Future Imagination?"** (arXiv:2603.16666; Tianyuan Yuan, Zibin Dong, Yicheng Liu, Hang Zhao, 2026). Project page: https://yuantianyuan01.github.io/FastWAM/ · HF model `yuanty/fastwam` · HF datasets `yuanty/LIBERO-fastwam`, `yuanty/robotwin2.0-fastwam`. MIT license.

**One-sentence idea:** a world action model built on the Wan2.2-TI2V-5B video DiT, where a small action expert and the big video expert share attention in a Mixture-of-Transformers (MoT); in the flagship "uncond" variant the action expert attends only to first-frame (current-observation) video tokens, so at policy time actions are decoded with **no video imagination** — one video-expert prefill into a KV cache, then cheap action-only denoising.

- Git: `origin = https://github.com/K1seki221/FastWAM.git`, branch `main`, ~11 commits, HEAD `45d8e14` (PR #20 "fix_gpu_oom": ckpt `map_location` → cpu).
- Python package `fastwam` v0.1.0, src-layout (`src/fastwam`), `pip install -e .`. All deps pinned exact (`torch==2.7.1+cu128` — needs the cu128 wheel index). No console entry points; everything runs via `python scripts/*.py`, `bash scripts/*.sh`, `python experiments/...`.
- Benchmarks: **LIBERO** (mujoco 3.3.2, must match the released data — dir is literally `data/libero_mujoco3.3.2/`) and **RoboTwin 2.0** (vendored harness in `third_party/RoboTwin`, upstream commit `bf44be51`, policies stripped).

## Directory layout

```
configs/            # Hydra: train.yaml, sim_libero.yaml, sim_robotwin.yaml + data/ model/ task/ groups
scripts/            # train.py, train_zero{1,2}.sh, preprocess_action_dit_backbone.py,
                    # precompute_text_embeds.py, accelerate_configs/, ds_configs/
src/fastwam/
  runtime.py        # model factories (create_fastwam[_joint|_idm]), build_datasets, run_training
  trainer.py        # Wan22Trainer (Accelerate + DeepSpeed ZeRO)
  models/wan22/     # fastwam.py / fastwam_joint.py / fastwam_idm.py / action_dit.py / mot.py
                    # wan22.py, wan_video_dit.py, wan_video_text_encoder.py, wan_video_vae.py,
                    # schedulers/scheduler_continuous.py, helpers/{loader,io,state_dict_converters,gradient}.py
  datasets/lerobot/ # RobotVideoDataset -> BaseLerobotDataset -> vendored LeRobot v2.1
  utils/            # config_resolvers, samplers, logging, video_io/metrics, misc (work-dir registry)
experiments/
  libero/           # run_libero_manager.py -> run_libero_parallel_test.sh (tmux) -> eval_libero_single.py
  robotwin/         # run_robotwin_manager.py -> eval_robotwin_single.py -> fastwam_policy/ (symlinked into RoboTwin)
third_party/RoboTwin # vendored sim harness; task_config/ & assets/ are gitignored and ABSENT until provisioned
```

Gitignored (absent until created): `runs/`, `checkpoints/`, `data/`, `evaluate_results/`, plus globally `*.pt` and `*.txt`.

## End-to-end workflow (run everything from repo root — all paths are relative)

1. **Model prep** (once): `export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"`, then
   `python scripts/preprocess_action_dit_backbone.py --model-config configs/model/fastwam.yaml --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt --device cuda --dtype bfloat16`
   (downloads Wan2.2 weights, builds the interpolated ActionDiT backbone).
2. **Data**: extract datasets under `./data/` (see `03_data.md` for expected layouts).
3. **Text-embed cache**: `python scripts/precompute_text_embeds.py task=<task>` (or `torchrun --standalone --nproc_per_node=8 ...`). Mandatory — training never runs the T5 encoder.
4. **Train**: `bash scripts/train_zero1.sh 8 task=libero_uncond_2cam224_1e-4` → outputs to `runs/<task>/<run_id>/`. First run per task: `pretrained_norm_stats: null` in `configs/data/*.yaml`; afterwards point it at the generated `runs/.../dataset_stats.json`.
5. **Eval**: `python experiments/libero/run_libero_manager.py task=<task> ckpt=<path>` or `python experiments/robotwin/run_robotwin_manager.py task=<task> ckpt=<path>` → results under `evaluate_results/`. Released ckpts: `checkpoints/fastwam_release/{libero_uncond_2cam224,robotwin_uncond_3cam_384}.pt` + matching `*_dataset_stats.json`.

## Environment variables

| Var | Effect |
|---|---|
| `DIFFSYNTH_MODEL_BASE_PATH` | Base dir for Wan pretrained weights (default `./checkpoints/` — cwd-relative!) |
| `DIFFSYNTH_DOWNLOAD_SOURCE` | `modelscope` (default) or `huggingface` |
| `DIFFSYNTH_SKIP_DOWNLOAD` | `true` → offline, never download |
| `NNODES`/`NODE_RANK`/`MASTER_ADDR`/`MASTER_PORT`/`RUN_ID` | read by `train_zero*.sh`, but only for RUN_ID sync (TCPStore on `MASTER_PORT+11`) and validation — the accelerate yaml hardcodes `num_machines: 1`, so exporting these does NOT produce a real multi-node launch (each node starts an independent job sharing RUN_ID); true multi-node needs editing the accelerate config |
| `LEROBOT_HOME` | must NOT be set — `datasets/lerobot/constants.py` raises at import; use `HF_LEROBOT_HOME` |

## Naming decode

Task names `{bench}_{variant}_{cams+res}_{lr}`: `uncond` = base `FastWAM` (the paper's Fast-WAM, released ckpts), `joint` = `FastWAMJoint`, `idm` = `FastWAMIDM` (both are the "needs test-time imagination" ablations). `2cam224` = LIBERO 2×224² cams (video 224×448); `3cam_384` = RoboTwin 3-cam 384×320 mosaic.
