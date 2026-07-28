# Consolidated gotchas (cross-cutting foot-guns)

## Will bite you immediately

1. **Run everything from the repo root** (`/home/ruijiezhang/FastWAM`). All paths are relative (`./data`, `checkpoints/…`, `scripts/…`, `third_party/RoboTwin`), hydra chdir is off.
2. **`task=<name>` is mandatory** for train/precompute — `train.yaml` defaults data/model to null; instantiate fails without it.
3. **Never run `python scripts/train.py` directly** — the trainer dereferences the DeepSpeed plugin unguarded; use `bash scripts/train_zero{1,2}.sh N task=…`.
4. **Two mandatory preprocessing steps** before first training: ActionDiT backbone interpolation (`preprocess_action_dit_backbone.py` → `checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`) and T5 embed cache (`precompute_text_embeds.py task=…`). Training has no text encoder (`load_text_encoder: false`) — missing cache = FileNotFoundError per sample.
5. **First run per task: `pretrained_norm_stats: null`** in `configs/data/*.yaml`; then point it at the generated `runs/<task>/<run_id>/dataset_stats.json`. Val/test datasets hard-require a stats file.
6. `.gitignore` globally ignores **`*.pt` and `*.txt`** — new checkpoints/text files silently untracked.
7. RoboTwin eval needs provisioning: `third_party/RoboTwin/task_config/` and `assets/` are absent in a fresh checkout (and `data/` holds only the tracked `process_stuck.py`); the `fastwam_policy` symlink is auto-created by `eval_robotwin_single.py` (or manually via the README `ln -sfn` command). mujoco must be exactly **3.3.2** for LIBERO.

## Silent-wrongness traps

8. **Dataset errors are swallowed**: `__getitem__` failures substitute random samples (`BaseLerobotDataset`: 5 total attempts = requested idx + 4 random, then raises; `RobotVideoDataset` catches that and returns 1 random sample) — corrupt videos/caches skew training after a printed traceback only.
9. **Text-embed cache enc_id `wan22ti2v5b` is hardcoded** in the dataset lookup while the precompute script derives it from `model_id` — non-default model_id silently breaks the pairing. Cache key = sha256(full formatted `DEFAULT_PROMPT`) + `context_len` + enc_id.
10. **All Wan `load_state_dict` calls are `strict=False`** — shape/key mistakes leave modules randomly initialized without error. Model files are identified by MD5 of key-names+shapes, not filenames.
11. **Normalizer forward clamps to [-5,5]** (not invertible); states always use global stats.
12. **No CFG exists** despite `text_cfg_scale`/`negative_prompt`/`action_cfg_scale` params everywhere (only plain `Wan22Core.infer` implements it). Method names say "noise" but the model predicts **velocity (noise − x0)**.
13. Typo'd `dataset_dirs` → confusing HF Hub download attempt (path used as repo_id), not "no such file".
14. `resume=<weights .pt>` restores weights only (optimizer/step lost under ZeRO); use the `checkpoints/state/step_XXXXXX` directory for true resume.
15. If `data.val` is null, **val dataset IS the train dataset** (same object).

## Architecture invariants (don't fight these)

16. `seperated_timestep: true` (misspelled on purpose) + `fuse_vae_embedding_in_latents: true` are hard-required (NotImplementedError otherwise). Frame-0 latents: clean GT at timestep 0, excluded from video loss, re-clamped after every inference step, never attend to later frames/actions.
17. `FastWAM.infer_action` requires `video_attention_mask_mode='first_frame_causal'` — the video KV cache (post-RoPE keys) is only valid because first-frame tokens can't see later noisy frames.
18. `model.dit` is an alias of `model.mot`; checkpoints save under key `"mot"` (legacy `"dit"` loads into video expert only). Trainer's "dit-only" freeze therefore trains the whole MoT + proprio_encoder.
19. Resolution: multiples of **32** effectively (VAE ×16 spatial + DiT patch 2), code only checks %16; frames `T % 4 == 1`; action horizon must be a **multiple of** `num_frames−1` (`T_a % (T−1) == 0`); ActionDiT RoPE caps 1024 action tokens.
20. The 3 model yamls differ only in `_target_`; variant behavior is in code. `infer_action` signatures differ across variants (`num_video_frames` required by Joint/IDM, absent in base). `FastWAMIDM.video_cond_noise_prob = 0.5` is a hardcoded class attribute.
21. `model.mot_checkpoint_mixed_attn` (task configs: false) gates gradient checkpointing in both experts via interpolation — override this key, not the dit sub-keys.

## Eval-specific

22. LIBERO gripper chain: model output (0=close,1=open) → `·2−1` → negate → env convention (−1=open,+1=close) → optional sign binarize. RoboTwin qpos actions get **no** remap.
23. LIBERO parallel eval kills any existing tmux session `libero_test_v3` — no concurrent runs; completion/resume detected purely by `gpu*_task*_results.json` existence.
24. RoboTwin evaluates only expert-verified seeds; success rate = last float line of `_result_*.txt`; one worker failure aborts the whole run. Default instructions are **unseen** (`seen` ≈ +1–2 pts).
25. `dataset_stats.json` auto-discovery only checks the first 4 parent dirs of `ckpt` — otherwise pass `EVALUATION.dataset_stats_path`.
26. LIBERO obs images are rotated 180° to match training; env seed affects object placement even with fixed init states.

## Misc

27. `${eval:…}` resolver = Python builtin `eval` — configs can execute arbitrary code.
28. `utils/misc.py` imports boto3 at module top — boto3 must be installed for `fastwam.runtime` to import.
29. `LEROBOT_HOME` env var must not be set (import-time ValueError).
30. Warmup hardcoded 5% of total steps; `max_steps` overrides `num_epochs`; `save/eval/log_every` count optimizer steps; no EMA, no tensorboard.
31. `configs/data/robotwin.yaml` duplicates `train:` as `val:` — edit shared fields in both nodes.
32. Model weights are cast to bf16 outright (no fp32 master weights); keep `mixed_precision: null` in the accelerate yamls.
33. Dead code to not be confused by: `ActionHead` (action_dit.py), `ActionEnsembler._cleanup`, `accelerate_zero0.yaml`, `search_dataset_stats_cache_json`, diffusers state-dict converters (present, unregistered), legacy torch.load fallback in `eval_libero_single._load_model_checkpoint`.
