# Reproduction fidelity verification (2026-07-28)

Five-way verification that the fuyao LIBERO setup reproduces the ORIGINAL Fast-WAM (not Xiaopeng's variant): paper fetch, official-upstream diff, effective-config trace, wrapper audit, contamination check. **Verdict: faithful.**

## Upstream identity — CONFIRMED
- Official repo is `yuantianyuan01/FastWAM` (the local origin `K1seki221/FastWAM` is a byte-identical mirror at merge commit `45d8e14`, verified by cloning the official repo and diffing).
- This checkout differs from official HEAD in exactly 4 files, all intentional and path/launcher-only: `configs/data/libero_2cam.yaml` (dataset_dirs), `configs/model/fastwam.yaml` (action_dit_pretrained_path), `scripts/train_zero{1,2}.sh` (RUNS_ROOT/RUN_NAME/ACCELERATE_PYTHON/exec). `src/` is 100% identical to official.
- Zero contamination from `former/FastWAM`: no `vip`/`lambda_vip`/`future_cache`/`joint_bid`/`video_proprio` anywhere in `src/` or `configs/`; `save_every` 2000 (not Xiaopeng's 10000); `wandb.enabled` false in train.yaml.

## Effective LIBERO recipe (task=libero_uncond_2cam224_1e-4, 8 GPUs) vs paper (arXiv:2603.16666)

| Item | Code (composed) | Paper |
|---|---|---|
| Optimizer | AdamW, betas (0.9, 0.95), wd 1e-2 | AdamW, wd 0.01 (betas unstated) ✓ |
| LR | 1e-4 cosine (eta_min 1e-6), 5% linear warmup (hardcoded) | 1e-4 cosine (warmup unstated) ✓ |
| Batch | 16/GPU × 8 = **128 global** | unstated (README: 1 node × 8 GPUs) ✓ |
| Duration | 10 epochs ≈ ~21k steps at batch 128 | "20k steps" ≈ ✓ (released config is authoritative) |
| Precision / clip / seed | bf16 cast, clip 1.0, seed 42 | mixed precision, clip 1.0 ✓ |
| Flow matching | shift-warped uniform t (shift 5.0, 1000 steps), v-target, Gaussian-bump weight | says "logit-normal" — **paper/code wording divergence; the code trained the released ckpts, follow the code** |
| Action expert | hidden 1024, 30 layers, 24×128 heads (interp-init from Wan DiT) | d_a=1024, ~1B, 6B total ✓ |
| Data | 2 cams 512²→224², horizontal concat → 224×448; horizon 32; 9 frames (4× downsample); min/max norm; ctx 128 | h=32, 4× downsample → 9 frames ✓; "2cam224" per task name ✓ |
| Loss | lambda_video 1.0 + lambda_action 1.0 | joint video+action ✓ |
| Inference | infer_action: 1 video prefill + 10 action denoise steps, no CFG | no video steps; 10 steps, CFG 1.0 ✓ |
| Eval | 4 suites × 10 tasks × 50 trials = 2000; replan 10; binarize gripper; ensembler off | 2000 trials / 40 tasks ✓ (replan unstated) |

## Target numbers (paper Table 2, Fast-WAM w/o embodied pretraining)
Spatial **98.2** · Object **100.0** · Goal **97.0** · Long **95.2** · **avg 97.6** (RoboTwin avg 91.8). Success = repro within ~1–2 pts/suite. Paper latency ref: 190 ms on one RTX 5090D.

## Wrapper audit — injected values are training-inert
Only hydra overrides injected: `task=`, `wandb.*` (logging), `output_dir` (path), `wandb.name` (label). Env vars (threads, caches, PYTHONPATH, NCCL preload, ACCELERATE_PYTHON, RUN_ID/RUNS_ROOT) don't touch model init, data order, loss, or seeding. Residual notes:
1. `DIFFSYNTH_MODEL_BASE_PATH` is the one env var feeding model init — fidelity depends on the shared `/dataset_rc/vlm/fm_models/.../checkpoints` snapshot being identical to official Wan-AI releases. **One-time check on a kernel**: the loader already validates state-dict key structure by MD5; for weight-level certainty compare file sha256 against HF `Wan-AI/Wan2.2-TI2V-5B` LFS hashes.
2. `USE_SYSTEM_NCCL=1` (LD_PRELOAD) and `OMP_NUM_THREADS=16` can change floating-point summation order → bitwise-level differences only, not semantics. `USE_SYSTEM_NCCL=0` if comparing against a bundled-NCCL baseline.
3. Upstream-stock quirk: LIBERO eval tmux panes `source ~/.bashrc` then run bare `python` — a container bashrc that activates another conda env could swap library versions (our PYTHONPATH still forces this checkout's code). If eval workers crash oddly, check `~/.bashrc` in the image.
4. `WANDB_API_KEY` (passed at submission time, not stored in the repo): logging-only, no RNG consumed.
