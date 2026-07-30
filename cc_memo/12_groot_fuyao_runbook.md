# GR00T × condition-router on fuyao — operational runbook (as of 2026-07-30)

Companion to `11_groot_router_design.md` (design) — this file is the operational state + hard-won cluster lessons from getting the baseline running.

## Current state

- **Code**: router v1 implemented + committed; Isaac-GR00T (upstream `9c7e746` + router) vendored at `FastWAM/Isaac-GR00T`, ships via `git pull` to fuyao. 5/5 unit tests pass (incl. forward/backward gradient checks). NOT yet done: router-stats→wandb logging hook in `Gr00tTrainer.compute_loss`; checkpoint-load smoke of the identity-init path (happens implicitly on first router run).
- **Provisioned on fuyao** (`BASE=/dataset_rc/ruijie.zhang@xiaopeng.com`):
  - Repo: `$BASE/FastWAM` (pull to update)
  - Venv: `$BASE/projects/groot/venv` (**outside the repo** — see lesson 2), built with uv; python at `$BASE/projects/groot/uv_python`, uv cache `$BASE/projects/groot/uv_cache`
  - Data: `$BASE/libero_groot/libero_{10,goal,object,spatial}_no_noops_1.0.0_lerobot` (IPEC-COMMUNITY, ~1.8 GB total, `modality.json` in each `meta/`, libero_goal episode-82 patched). NOT the same as `/dataset_rc/vlm/vit/LIBERO-fastwam` (Fast-WAM preprocessing).
  - Weights: `$BASE/hf/hub/models--nvidia--{GR00T-N1.7-3B,Cosmos-Reason2-2B}` (copied `cp -a` from the interactive container's `/dataset-cpfs3-rc/hf/hub`)
  - Runs: `$BASE/projects/groot_runs/<RUN_NAME>`
- **Where we are**: baseline `libero_10` submission ready; awaiting first successful run + seconds-per-step, then the 8-run matrix (4 suites × {baseline, router}) per `11`'s plan.

## The submit chain

`scripts/submit_fuyao_groot.sh <nproc>` → `fuyao deploy` (project rc-embodied-foundation-model, site fuyao_sh_n2, gpu-type h200, volume rc-perception, experiment ruijie, docker liuw50-260318-0232) → `scripts/groot_fuyao_train.sh` (env pins + pre-flight checks, activates the external venv) → `Isaac-GR00T/examples/finetune.sh` (torchrun + DeepSpeed ZeRO-2).

Knobs (env): `SUITE=10|goal|object|spatial`, `RUN_NAME`, `MAX_STEPS` (20000 default; 100 = timing probe), `GLOBAL_BATCH_SIZE` (640), `USE_ROUTER=1 ROUTER_LR=1e-3 ROUTER_LAYERS="0 6 12"`, `WANDB_API_KEY` (wandb auto-on when set; project `finetune-gr00t-n1d7`) or `WANDB_MODE=offline`, `QUEUE=...` if quota needs it, `DRY_RUN=1` preview, `EXTRA_ARGS` passthrough.

Canonical commands:
```bash
# baseline                                    # router (later)
WANDB_API_KEY=<k> RUN_NAME=baseline_libero_10 SUITE=10 \   USE_ROUTER=1 WANDB_API_KEY=<k> RUN_NAME=router_libero_10 SUITE=10 \
bash scripts/submit_fuyao_groot.sh 8                       bash scripts/submit_fuyao_groot.sh 8
```
Direct on an interactive node: same env + `bash scripts/groot_fuyao_train.sh` in tmux.

## Lessons (each cost a failed attempt — do not relearn)

1. **Job containers ≠ interactive containers.** The interactive container pre-sets its own HF cache env (downloads land in `/dataset-cpfs3-rc/hf/hub` regardless of your `HF_HOME` export); deployed jobs don't see that cache (mount and/or env differ). Fix: weights copied to `$BASE/hf/hub`; the runner pins `HF_HOME` **and `HF_HUB_CACHE`** (the latter outranks the former) + `HF_HUB_OFFLINE=1`. Always `unset HF_HOME HF_HUB_CACHE` before submitting — the wrapper forwards stale exports.
2. **fuyao snapshots the submit directory** (>500 MiB check). The multi-GB venv must live outside the repo → `UV_PROJECT_ENVIRONMENT=$BASE/projects/groot/venv`; runner honors `VENV_DIR` (falls back to legacy in-repo `.venv`).
3. **uv persistence**: `UV_PYTHON_INSTALL_DIR` and `UV_CACHE_DIR` must point at `/dataset_rc`, else the venv symlinks a container-local interpreter and dies with the next job — the same stale-shebang disease as Xiaopeng's copied `wam` env (whose `pip` is permanently broken: shebang → nonexistent `/root/miniconda3`; reset `PATH` if it shadows yours).
4. **Networks**: PyPI is slow → `UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=4`, optionally `UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/`. **hf-mirror serves gated repos when `HF_TOKEN` is set** (no OSS transfer needed for Cosmos). All `hf download`/`uv sync` are resumable — loop `until <cmd>; do sleep 60; done`.
5. **gitignore traps when vendoring**: Isaac-GR00T's own `.gitignore` has `wheels/` — the torchcodec wheel had to be force-added (`4ea813a`); FastWAM's global `*.txt`/`*.pt` ignores are negated for `Isaac-GR00T/**`. The aarch64 torchcodec wheel is an LFS file — shallow clones get a pointer; real file fetched from `media.githubusercontent.com`.
6. `git pull` on fuyao is the only code-sync mechanism; the runner's pre-flight checks (dataset, modality.json, venv, weights) fail fast with explicit messages — read the first error line, not the torchrun spew.

## Success signals for the first router run

`condition_router params not in checkpoint - identity-initialized (28 keys)` and `create_optimizer: router group with 27 params at lr=0.001` in the log; then finite decreasing loss. Baseline target (official): libero_10 → **94.35** success (suite avg 97.0 across the four).
