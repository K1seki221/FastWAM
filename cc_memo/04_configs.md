# Hydra config system

## Composition graph

```
train.yaml (@_global_)            defaults: [_self_, data: null, model: null, task: null]
  └─ task=<name>  (task/*.yaml, @_global_)
       defaults: [override /data: <libero_2cam|robotwin>, override /model: <fastwam|fastwam_idm|fastwam_joint>, _self_]
       + root-level trainer hyperparam overrides
sim_libero.yaml   (@_global_)     defaults: [train, override /task: libero_uncond_2cam224_1e-4, _self_] + ckpt/EVALUATION/MULTIRUN
sim_robotwin.yaml (@_global_)     defaults: [train, override /task: robotwin_uncond_3cam_384_1e-4, _self_] + same structure
```

- **`task=<name>` is mandatory for train/precompute** — bare `train.yaml` has `data: null, model: null` and instantiate fails. Model configs interpolate dims from data (`${data.train.processor.action_output_dim}` etc.), so model is invalid without a data group.
- Top-level configs and all task yamls are `# @package _global_` — task keys land at the config root (`batch_size`, not `task.batch_size`). The `data/` and `model/` group yamls have no @package directive and land under `data.*` / `model.*` normally.
- Hydra: `job.chdir: false`, `run.dir: .`, `output_subdir: null` — no `.hydra/` dirs, no cwd change; run dirs come solely from `output_dir`.
- Sim configs inherit ALL of train.yaml, flip `model.load_text_encoder=true`, `model.skip_dit_load_from_pretrain=true`, `model.action_dit_pretrained_path=null`, and add `ckpt` (required), `gpu_id`, `EVALUATION.*`, `MULTIRUN.*`. `${hydra:runtime.choices.task}` embeds the task name in `evaluate_results/...` output dirs.
- `eval_num_inference_steps: 10` in train.yaml is reused as `EVALUATION.num_inference_steps` in both sim configs via interpolation.

## The 6 task configs

Common to all: batch_size 16, num_workers 8, lr 1e-4 cosine, weight_decay 1e-2, grad_accum 1, `model.mot_checkpoint_mixed_attn: false`.

| task | data | model `_target_` | epochs | save_every | eval_every |
|---|---|---|---|---|---|
| `libero_uncond_2cam224_1e-4` | libero_2cam | `create_fastwam` | 10 | 2000 | 200 |
| `libero_idm_2cam224_1e-4` | libero_2cam | `create_fastwam_idm` | 10 | 2000 | 200 |
| `libero_joint_2cam224_1e-4` | libero_2cam | `create_fastwam_joint` | 10 | 2000 | 200 |
| `robotwin_uncond_3cam_384_1e-4` | robotwin | `create_fastwam` | 5 | 2500 | 500 |
| `robotwin_idm_3cam_384_1e-4` | robotwin | `create_fastwam_idm` | 5 | 2500 | 500 |
| `robotwin_joint_3cam_384_1e-4` | robotwin | `create_fastwam_joint` | 5 | 2500 | 500 |

## Key facts

- The 3 `configs/model/*.yaml` differ ONLY in `_target_` — don't hunt for architectural differences in config keys; uncond/idm/joint behavior lives in the classes/factories.
- `model.mot_checkpoint_mixed_attn` gates `use_gradient_checkpointing` in BOTH DiT sub-configs via interpolation — toggle it, not the sub-keys. Model yamls default it true; **every task yaml flips it to false**.
- `seperated_timestep` is misspelled on purpose — match the spelling when overriding.
- `action_scheduler` is required by the factories (ValueError if missing any of `train_shift`/`infer_shift`/`num_train_timesteps`); `video_scheduler` is optional and defaults each key to 5.0/5.0/1000. Shipped configs set both to 5.0/5.0/1000.
- Action horizon is NOT a model key — it derives from data: `num_frames: 33` + `action_video_freq_ratio: 4` → 32 actions / 9 frames.
- `tokenizer_max_len: 128` in model configs == `context_len: 128` in data configs.
- Custom resolvers (`src/fastwam/utils/config_resolvers.py`): `eval` = **Python builtin eval** (arbitrary code execution via `${eval:'…'}`); `oc.load` shadowed with a cwd-aware loader; `split`/`max`/`round_up`/`round_down`/`sum_shapes`/`max_action_dim`/`max_state_dim`. Registered only by `scripts/train.py` and `scripts/precompute_text_embeds.py`; `eval_libero_single.py` registers just a 3-resolver subset (`eval`/`max`/`split`); the other entrypoints register none. Currently unused by the shipped yamls.
- `EVALUATION` key highlights — sim_libero: `task_suite_name`, `task_id`, `num_trials: 50`, `replan_steps: 10`, `num_steps_wait: 30`, `binarize_gripper: true`, `use_action_ensembler: false`, `visualize_future_video: false`; sim_robotwin: `robotwin_root: third_party/RoboTwin`, `task_name: null` (manager: null = all tasks; the single-task entry `eval_robotwin_single.py` hard-requires `EVALUATION.task_name`), `task_config: demo_randomized`, `instruction_type: unseen`, `eval_num_episodes: 100`, `replan_steps: 24`, `skip_get_obs_within_replan: true`. Both: `text_cfg_scale: 1.0` (moot — no CFG), `dataset_stats_path: null` (auto-discovery near ckpt).
- `MULTIRUN`: `num_gpus: 8`, `max_tasks_per_gpu: 2`; libero also `task_suite_names: [libero_10, libero_goal, libero_spatial, libero_object]`, `create_only`, `task_file`.
- pyproject: no console scripts; exact `==` pins; `torch==2.7.1+cu128` needs `--extra-index-url https://download.pytorch.org/whl/cu128`.
