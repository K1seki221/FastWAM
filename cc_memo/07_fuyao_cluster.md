# Running FastWAM on the fuyao cluster (from Xiaopeng Zhang's fork in `former/FastWAM`)

## Job submission (confirmed working example from Xiaopeng, 2026-07)

```
fuyao deploy --gpus-per-node=8 --nodes=1 \
  --project=rc-embodied-foundation-model --site=fuyao_sh_n2 \
  --docker-image infra-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/data-infra/fuyao:liuw50-260318-0232 \
  --experiment <name> -- \
  <ENV=val ...> bash /workspace/<user>/FastWAM/scripts/train_fuyao_fastwam.sh 8 task=<task> <hydra overrides>
```

- Env vars are passed inline before `bash` in the command after `--` (e.g. `CONDA_ENV=...`, `WANDB_API_KEY=...`, `RUN_NAME=...`).
- The docker image is a generic data-infra image; the conda env comes from `/dataset_rc` mounts, so the image is reusable as-is.
- Xiaopeng's conda env is reusable directly (`CONDA_ENV=/dataset_rc/zhangxp7@xiaopeng.com/miniconda3/envs/wam`) IF readable: the wrapper puts `$REPO_ROOT/src` first on `PYTHONPATH`, which shadows any editable fastwam install baked into the env — so the code that runs is always the checkout's, not the env owner's.
- SECURITY: never commit or reuse another person's `WANDB_API_KEY` (Xiaopeng's key was shared in chat 2026-07-28; runs with it land in their account — use your own key or `WANDB_MODE=offline`).

## Team submission conventions (from Ruijie's iron_vla `run.sh`, seen 2026-07-28)

- Fuller flag set than Xiaopeng's minimal command: `--gpu-type h200`, `--queue=<...>` (seen: `rc-perception-vla`, `rc-embodied-foundation-model-h200-p1`, `rc-perception-vla-4090` + `--over-subscription`), `--label=<run label>`, `--volume=rc-perception`, `--experiment=ruijie` (Ruijie's experiment namespace — use this, not `yinz`). Project `rc-embodied-foundation-model` and site `fuyao_sh_n2` match.
- Env passing there uses the `--envs KEY=VAL` flag; Xiaopeng's FastWAM flow uses inline `ENV=val` prefixes in the command after `--`. Both work; our `submit_fuyao.sh` keeps the inline style (proven with this exact payload).
- Their payloads are RELATIVE script paths submitted from the repo dir, and they write `current_diff.diff`/`last_commit.diff` (git HEAD + diffs) before `fuyao deploy` and delete them after — implying fuyao snapshots the submit-time cwd as the job workdir. The FastWAM flow instead references the persistent `/workspace/<user>/FastWAM` checkout by absolute path (no snapshot dependency).
- Fuyao containers set `core_pattern=/tmp/core.%p` with unlimited core dumps — long sim jobs should `ulimit -c 0` or crashing native processes (SAPIEN/Vulkan) fill ephemeral `/tmp` and get the pod evicted (exit 137). Worth adding to RoboTwin eval jobs if we ever run them.
- Their `run.sh` embeds OSS credentials — do not copy those into FastWAM scripts.
- Ported into `submit_fuyao.sh` accordingly: `EXPERIMENT` defaults to `ruijie`, `LABEL` defaults to `RUN_NAME`, and `GPU_TYPE`/`QUEUE`/`VOLUME` are optional flags emitted only when set. Git provenance (HEAD + uncommitted diff) is recorded by `deploy_fuyao_train_and_eval.sh` into `<run_dir>/git_provenance.diff` at training start.

## Ported into the pristine repo (2026-07-28, this checkout)

- Ruijie's fuyao account is **`ruijie.zhang@xiaopeng.com`** and their cluster checkout lives at **`/dataset_rc/ruijie.zhang@xiaopeng.com/FastWAM`** (on persistent storage, NOT /workspace like Xiaopeng's) — both baked in as script defaults on 2026-07-28. Run outputs/caches: `/dataset_rc/ruijie.zhang@xiaopeng.com/projects/fastwam/{runs,.cache}`.
- `scripts/train_fuyao_fastwam.sh` / `scripts/eval_fuyao_libero.sh`: rewritten, parameterized by `FUYAO_USER`; everything overridable via env (`REPO_ROOT`, `CONDA_ENV`, `CACHE_ROOT`, `RUNS_ROOT`). Exports `DIFFSYNTH_MODEL_BASE_PATH=/dataset_rc/vlm/fm_models/VIT/WAM/FastWAM/checkpoints`. wandb auto-enables only when `WANDB_API_KEY` is set or `WANDB_MODE=offline`; entity defaults to the key's default (never yumio-wam). Default task: `libero_uncond_2cam224_1e-4`. NOTE (2026-07-28): no credential is stored in the repo — `WANDB_API_KEY` must be passed at submission time (`WANDB_API_KEY=... bash scripts/submit_fuyao.sh ...`; the wrapper forwards it into the job). `submit_fuyao.sh` prints a warning when submitting online without a key (job runs with wandb disabled).
- `scripts/train_zero{1,2}.sh`: patched with `RUNS_ROOT` (default `./runs` — local behavior unchanged), `RUN_NAME` suffix, `ACCELERATE_PYTHON`/`ACCELERATE_ENTRYPOINT` support, and `exec` of the accelerate command.
- `configs/data/libero_2cam.yaml`: `dataset_dirs` → `/dataset_rc/vlm/vit/LIBERO-fastwam/...` (original relative paths in comment).
- `configs/model/fastwam.yaml`: `action_dit_pretrained_path` → absolute shared path.
- `third_party/LIBERO`: copied from Xiaopeng's tree (22 MB, 525 files). NOTE: it has `bddl_files`/`init_files` but **no `assets/`** — check Xiaopeng's `/workspace` copy or LIBERO's asset download on first eval.
- `scripts/deploy_fuyao_train_and_eval.sh`: single-job pipeline — pins `RUN_ID` up front (so the run dir is known deterministically; `train_zero1.sh` honors a pre-set `RUN_ID`), runs the train wrapper as a child process, picks the latest `checkpoints/weights/step_*.pt`, then execs the eval wrapper on it. Positional hydra args go to training; eval knobs via env (`NUM_TRIALS`, `MAX_TASKS_PER_GPU`, `EVAL_ARGS`, `EVAL_OUTPUT_DIR` default `<run_dir>/libero_eval`); `CKPT=...` skips training (eval-only), `SKIP_EVAL=1` trains only, `DRY_RUN=1` prints both phases.
- `scripts/submit_fuyao.sh`: one-command submission wrapper — bakes in the deploy flags (project `rc-embodied-foundation-model`, site `fuyao_sh_n2`, the shared docker image, `--gpus-per-node=<nproc> --nodes=1`), %q-escapes and forwards set env vars (`RUN_NAME`, `NUM_TRIALS`, `EVAL_ARGS`, `CKPT`, `SKIP_EVAL`, wandb vars, …) as inline `VAR=val` prefixes, and runs `fuyao deploy ... -- ... deploy_fuyao_train_and_eval.sh`. `DRY_RUN=1` prints without submitting; `JOB_DRY_RUN=1` makes the *job* dry-run in-container; `EXPERIMENT` defaults to `RUN_NAME`. Full chain: `submit_fuyao.sh` → `fuyao deploy` → `deploy_fuyao_train_and_eval.sh` → `train_fuyao_fastwam.sh` → `train_zero1.sh` → accelerate; then `eval_fuyao_libero.sh` → `run_libero_manager.py`.
- `src/` left 100% pristine (Xiaopeng's trainer/model changes NOT taken).
- Still required before first training job (on a Remote Kernel): precompute the T5 text-embed cache into the workspace checkout (`python scripts/precompute_text_embeds.py task=libero_uncond_2cam224_1e-4` with `DIFFSYNTH_MODEL_BASE_PATH` exported).

Source: `former/FastWAM` — Xiaopeng Zhang's fork. It mixes **algorithm changes (ignore for reproduction)** — VIP auxiliary loss (`vip.py`, `vip_dpt.py`, `lambda_vip`, `video_proprio` sample key), `fastwam_future_cache`, `fastwam_joint_bid` variants — with **fuyao infrastructure (this file)**. Verified by diffing against the pristine repo.

## Fuyao cluster model

- Jobs are submitted as single-node **PytorchJobs**: `fuyao deploy <resource flags> bash <script> <args>` (example in `eval_fuyao_libero.sh` header). Interactive "Remote Kernel" sessions also exist (Xiaopeng's README: run `train_zero1.sh` directly there).
- Both fuyao scripts **hard-refuse `NNODES != 1`** — single-node 8-GPU only.
- Job containers: Ubuntu with apt + root/sudo, ephemeral. Scripts switch the apt mirror to `http://mirrors.cloud.aliyuncs.com/ubuntu` and install missing packages at job start.
- Filesystems:
  - `/workspace/<user>@xiaopeng.com/` — code workspace, visible to jobs, **mutable while jobs run** (hence `exec` in the patched `train_zero1.sh`, so bash never re-reads a changed script file after the long accelerate command).
  - `/dataset_rc/<user>@xiaopeng.com/` — personal persistent storage: `miniconda3/envs/wam` (the conda env), `projects/fastwam/.cache` (XDG/HF/torch/modelscope caches), `projects/fastwam/runs/` (training outputs).
  - `/dataset_rc/vlm/` — **shared read-only data**: LIBERO-fastwam lerobot dataset at `/dataset_rc/vlm/vit/LIBERO-fastwam/{libero_spatial,libero_object,libero_goal,libero_10}_no_noops_lerobot`; Wan2.2 weights + ActionDiT backbone at `/dataset_rc/vlm/fm_models/VIT/WAM/FastWAM/checkpoints` (contains `ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`). **No dataset/weights download needed on fuyao.**

## The two entry scripts (both in `former/FastWAM/scripts/`)

### `train_fuyao_fastwam.sh` — training wrapper
`bash scripts/train_fuyao_fastwam.sh 8 task=libero_uncond_2cam224_1e-4` (submitted as PytorchJob or run on a kernel). What it does, in order:
1. Sets path constants (`DEFAULT_REPO_ROOT=/workspace/zhangxp7@xiaopeng.com/FastWAM`, `CONDA_ROOT=/dataset_rc/zhangxp7@xiaopeng.com/miniconda3`, env `wam`, `CACHE_ROOT=/dataset_rc/.../projects/fastwam/.cache`) — all overridable via env (`REPO_ROOT`, `CONDA_ENV`, `CACHE_ROOT`…). **Must be edited/overridden per user.**
2. Redirects `XDG_CACHE_HOME`/`HF_HOME`/`HUGGINGFACE_HUB_CACHE`/`TORCH_HOME`/`MODELSCOPE_CACHE` to persistent storage; sets `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`, `OMP_NUM_THREADS=16`, `NCCL_DEBUG=WARN`, `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`.
3. apt-installs EGL libs (`libegl1 libopengl0 libglvnd0 libgl1`) with the aliyun mirror (`SKIP_APT_INSTALL=1` to skip).
4. **Manual conda activation** — the env was *copied* from `/root/miniconda3`, so console-script shebangs are stale; the script exports `PATH`/`LD_LIBRARY_PATH`/`CONDA_PREFIX` directly and sets `ACCELERATE_PYTHON=$PYTHON_BIN` + `ACCELERATE_ENTRYPOINT=$CONDA_ENV/bin/accelerate` so the patched `train_zero1.sh` runs `python .../accelerate launch` instead of relying on the shebang. Also sets `PYTHONPATH=$REPO_ROOT/src:$REPO_ROOT`.
5. **NCCL**: `LD_PRELOAD`s system `/lib/x86_64-linux-gnu/libnccl.so.2` by default (`USE_SYSTEM_NCCL=0` to use the conda/pytorch bundled one).
6. wandb login in-script (`WANDB_API_KEY` env, or `WANDB_MODE=offline`); defaults entity `yumio-wam`, project `fast-wam`; injects `wandb.mode/workspace/project` hydra overrides unless given.
7. Defaults `task=$TASK_CONFIG` if no `task=` given (**Xiaopeng's default is `libero_future_cache_2cam224_1e-4` — his variant, override it**), then `exec`s `scripts/train_zero1.sh <nproc> <hydra args>`.
Useful knobs: `DRY_RUN=1` (print resolved command only), `RUN_NAME=<suffix>` (appended to the run dir id), `NPROC_PER_NODE`, `CONDA_ENV=path`, `WANDB_API_KEY`, `WANDB_MODE=offline`.

### `eval_fuyao_libero.sh` — LIBERO eval wrapper
`bash scripts/eval_fuyao_libero.sh task=<task> ckpt=<weights.pt> [MULTIRUN.num_gpus=N] [EVALUATION.num_trials=50]`. Same activation/cache/apt logic plus:
- apt also installs `libglu1-mesa` and **`tmux`** (needed by `run_libero_parallel_test.sh`).
- **No pip-installed LIBERO**: uses vendored `third_party/LIBERO` (present in Xiaopeng's tree, ~full upstream checkout) via `PYTHONPATH=$REPO_ROOT/third_party/LIBERO:...`.
- Generates a LIBERO config yaml at `$LIBERO_CONFIG_PATH` (default `/tmp/fastwam-libero-config/config.yaml`) pointing `assets`/`bddl_files`/`init_states` at the vendored tree.
- Enforces `MULTIRUN.num_gpus == visible GPU count` (dies otherwise — size the fuyao resource request to match).
- Env-var equivalents: `TASK_CONFIG`, `CKPT`, `NUM_TRIALS`, `MAX_TASKS_PER_GPU`, `EVAL_OUTPUT_DIR`.

## Infra changes inside the repo proper (worth porting to a pristine checkout)

- `scripts/train_zero1.sh` / `train_zero2.sh`: (a) `ACCELERATE_PYTHON`/`ACCELERATE_ENTRYPOINT` support; (b) `exec` the accelerate command; (c) `RUN_NAME` suffix for run dirs; (d) **output_dir hardcoded** to `/dataset_rc/zhangxp7@xiaopeng.com/projects/fastwam/runs/${TASK_BASENAME}/${RUN_DIR_ID}` — change per user.
- `configs/train.yaml`: `wandb.enabled: true`, `workspace: yumio-wam` (change/disable).
- `configs/data/libero_2cam.yaml`: `dataset_dirs` → `/dataset_rc/vlm/vit/LIBERO-fastwam/...`. `text_embedding_cache_dir` **unchanged** (`./data/text_embeds_cache/libero`, repo-root-relative → lives in the /workspace checkout; precomputed on a remote kernel before submitting). `pretrained_norm_stats` unchanged (absent → computed at startup each run).
- `configs/model/*.yaml`: `action_dit_pretrained_path` → `/dataset_rc/vlm/fm_models/VIT/WAM/FastWAM/checkpoints/ActionDiT_...pt` (+ VIP keys — ignore).
- `helpers/io.py`: default checkpoint base (when `DIFFSYNTH_MODEL_BASE_PATH` unset) → `/dataset_rc/vlm/fm_models/VIT/WAM/FastWAM/checkpoints`. Cleaner alternative: just export `DIFFSYNTH_MODEL_BASE_PATH` instead of patching.
- `configs/task/libero_uncond...yaml`: `save_every: 2000 → 10000` (disk saving; doesn't affect training math).
- `trainer.py` (infra bits only): guarded `deepspeed_plugin` deref (no longer crashes without DeepSpeed); `wandb.finish()` wrapped in try/except (remote tracking flush failures don't fail the job; `wandb sync` later).
- `run_libero_parallel_test.sh`: tmux session renamed `libero_${RUN_ID}` (unique, concurrent-safe, replaces fixed `libero_test_v3`) with EXIT/INT/TERM cleanup trap; explicit tmux presence check.
- `third_party/LIBERO`: vendored full LIBERO benchmark (pristine repo doesn't have it) — needed by the fuyao eval flow.

## Reproduction guidance (original Fast-WAM on fuyao)

Use the **pristine repo** as the code base and port ONLY: the two `*_fuyao_*.sh` scripts, the `train_zero1.sh` infra patch, `third_party/LIBERO`, and the path/wandb config edits. Do NOT take: `vip*.py`, `fastwam_future_cache*`, `fastwam_joint_bid*`, `lambda_vip`/`loss.vip` config blocks, `video_proprio` dataset changes, or Xiaopeng's modified `fastwam.py`/`mot.py`/`runtime.py`/`robot_video_dataset.py`. With `task=libero_uncond_2cam224_1e-4` on 8 GPUs the training hyperparams are already the paper's; the only faithful-repro deltas in his tree are `save_every` (10000 vs 2000) and wandb settings — both training-math-neutral. His `lambda_vip: 0.0` in the uncond task config disables the VIP loss, but his `fastwam.py`/`trainer.py` still differ from upstream — safest to not use his `src/` at all.
