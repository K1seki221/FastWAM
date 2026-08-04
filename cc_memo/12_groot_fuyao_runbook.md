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

## Phase A closed-loop results (2026-08-01, LIBERO 4 suites, 10 eps/task)

Stage-1 from-scratch runs (DiT+heads from random, Cosmos VLM trainable @1e-5,
4 suites joint, 30K steps, batch 256, ~3.5h on 8xH200), eval via
scripts/groot_fuyao_eval.sh (server+client, 1 GPU, ~2h):

| arm | spatial | object | goal | long | overall |
|---|---|---|---|---|---|
| s1_baseline (fixed tap) | 1.000 | 1.000 | 1.000 | 0.960 | **0.990** |
| s1_router (13-way, uniform init, lr 1e-3) | 1.000 | 1.000 | 0.960 | 0.980 | **0.985** |

Anchors: official finetuned N1.7 = 0.970; StarVLA-GR00T specialist = 0.965.
Readings: (1) the from-scratch recipe BEATS the official finetune — headline
reproducibility result; (2) baseline-vs-router = parity within noise — LIBERO
is saturated, cannot differentiate routing; (3) decisive router test moves to
RoboCasa-GR1-tabletop (official anchor 44.5%, corpus = gr1_unified, downloading).
Router final routing (30K): entropy 2.109, w L12=0.313 L11=0.191 L10=0.105 —
rediscovered the deep-layer preference from uniform, converged to a soft mixture.
Results CSVs: $BASE/projects/groot_evals/eval_s1_{baseline,router}_30k/results.csv
(job status shows JOB_FAILED due to a cosmetic summary-heredoc bash bug — the
40-task CSVs are complete and error-free; fix the heredoc after in-flight evals drain).

## Plan state (2026-07-31, decided with Ruijie)

- **No cluster job is submitted without Ruijie's explicit approval of that specific launch** (standing rule).
- Scope shifted: the meaningful router test is **stage-1-style training** (DiT from scratch, trainable VLM per N1.7's actual regime) — plain fine-tuning of the pretrained ckpt is a biased test (reader pre-committed to the last-layer tap). The fine-tune pair from 07-31 was demoted to infra-validation/adaptation-regime datapoint; baseline fine-tune cancelled by Ruijie at ~step 4000 (loss 0.029, ckpts kept), router-v4 fine-tune managed by Ruijie directly.
- **Blocker being settled first: the stage-1 training corpus** (GR00T's real pretraining data is unreleased). Candidates: nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim (the released slice of GR00T's own corpus) + LeRobot-format public sets (IPEC-COMMUNITY LIBERO etc.). Then relaunch a coherent experiment set with approval.
- From-scratch mechanics verified locally: `--skip-weight-loading --tune-llm --tune-visual` works; from-scratch build uses CODE-DEFAULT arch (select_layer 12 -> K=13, DiT 16 layers -> 8 cross blocks, 2.2B total all-trainable) NOT the released ckpt's 16/32; direct construction preserves ctor router init (uniform at bias 0), unlike from_pretrained.

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

1. **Job containers ≠ interactive containers.** The interactive container pre-sets its own HF cache env (downloads land in `/dataset-cpfs3-rc/hf/hub` regardless of your `HF_HOME` export); deployed jobs don't see that cache (mount and/or env differ). Fix: weights copied to `$BASE/hf/hub`; the runner pins `HF_HOME` **and `HF_HUB_CACHE`** (the latter outranks the former) + `HF_HUB_OFFLINE=1`. UPDATE 2026-07-30: the *job* container ALSO pre-sets `HF_HOME=/dataset_rc/hf` (no user segment) — the runner's old `${HF_HOME:-...}` default honored it and pre-flight-failed run `bifrost-2026073102171101`. The runner now hard-pins `HF_HOME` unconditionally; intentional override goes through `GROOT_HF_HOME` (forwarded by the submit wrapper instead of `HF_HOME`).
2. **fuyao snapshots the submit directory** (>500 MiB check). The multi-GB venv must live outside the repo → `UV_PROJECT_ENVIRONMENT=$BASE/projects/groot/venv`; runner honors `VENV_DIR` (falls back to legacy in-repo `.venv`).
3. **uv persistence**: `UV_PYTHON_INSTALL_DIR` and `UV_CACHE_DIR` must point at `/dataset_rc`, else the venv symlinks a container-local interpreter and dies with the next job — the same stale-shebang disease as Xiaopeng's copied `wam` env (whose `pip` is permanently broken: shebang → nonexistent `/root/miniconda3`; reset `PATH` if it shadows yours).
4. **Networks**: PyPI is slow → `UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=4`, optionally `UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/`. **hf-mirror serves gated repos when `HF_TOKEN` is set** (no OSS transfer needed for Cosmos). All `hf download`/`uv sync` are resumable — loop `until <cmd>; do sleep 60; done`.
5. **gitignore traps when vendoring**: Isaac-GR00T's own `.gitignore` has `wheels/` — the torchcodec wheel had to be force-added (`4ea813a`); FastWAM's global `*.txt`/`*.pt` ignores are negated for `Isaac-GR00T/**`. The aarch64 torchcodec wheel is an LFS file — shallow clones get a pointer; real file fetched from `media.githubusercontent.com`.
6. **Offline mode vs transformers 4.57.3 tokenizer loading** (killed runs `bifrost-2026073102591700` + `...03310200`, 2026-07-30): loading a tokenizer/processor BY REPO ID triggers `_patch_mistral_regex` -> `HfApi.model_info()` — a hub API call that cannot be cache-served, so `HF_HUB_OFFLINE=1` raises `OfflineModeIsEnabled`. Weights/config load fine from cache; only the tokenizer path breaks. Fix chain (all applied): (a) `launch_finetune.py` honors `GR00T_BACKBONE_PATH` (local Cosmos snapshot dir) — local paths skip the probe; (b) the runner resolves the snapshot via `refs/main` and exports it; (c) `setup.py _create_model` passes `model_name=config.model.model_name` into `AutoModel.from_pretrained` — otherwise the CHECKPOINT's config.json value (the repo id) wins and the override never reaches the collator; (d) `get_backbone_cls` substring loosened from `"nvidia/Cosmos-Reason2"` to `"Cosmos-Reason2"` (cache paths use `models--nvidia--Cosmos-Reason2-2B`). NOTE: checkpoints trained this way store the absolute snapshot path in config.json/processor_config.json — fine on-cluster, remap if exporting. Stock ckpts (e.g. `nvidia/GR00T-N1.7-LIBERO`) still carry repo ids — the eval/server path will need the same treatment when we wire it.
7. **`fuyao deploy` without `--queue` opens an interactive queue picker** — dies with `termios` errors in non-TTY shells (scripted/SSH submissions). Always pass `QUEUE=rc-embodied-foundation-model-h200-p1` (the h200 queue used by all runs so far) when submitting non-interactively.
8. `git pull` on fuyao is the only code-sync mechanism; the runner's pre-flight checks (dataset, modality.json, venv, weights) fail fast with explicit messages — read the first error line, not the torchrun spew.

## The router-enablement bug chain (2026-07-31 — cost 3 cancelled runs, READ THIS before touching setup.py)

Runs v1-v3 of "router_libero_10" were silently PLAIN BASELINES: `_create_model` loads the model
from the stock checkpoint whose config.json predates the router fields, and (unlike tune_*) the
router fields were not forwarded as from_pretrained kwargs -> `use_condition_router` fell back to
False. No error anywhere; loss curves indistinguishable from baseline (which they were). Diagnosed
with a one-shot `router-log probe` logging.warning in Gr00tTrainer.compute_loss (kept in the code).
Fixes, all in the vendored tree:
1. setup.py `_create_model`: forwards `use_condition_router / router_candidate_layers /
   router_init_bias / router_lr` as kwargs (router_lr matters too - create_optimizer reads it
   off the LOADED config).
2. `init_condition_router_from_vlln` now re-applies the FULL identity init (zero logits + bias
   on incumbent column + unit norms + vlln copy): HF from_pretrained re-initializes missing-key
   params AFTER the ctor, wiping the ctor bias (observed incumbent mass 0.088 instead of 0.773).
3. Real N1.7-3B checkpoint dims differ from code defaults: select_layer=16 (K=17 candidates incl.
   embeddings), 32 DiT layers -> 16 cross blocks. Identity mass at bias 4.0 = e^4/(e^4+16) = 0.7734.
4. RouterLLM/* wandb logging lives in Gr00tTrainer.compute_loss reading
   `action_head.condition_router.mixture_stats()` DIRECTLY off the module (input-independent);
   verified live: w_incumbent_mean 0.7734, off-incumbent 0.01416, entropy 1.156 at step 0.
First healthy router run: `bifrost-2026073105275141` (2026-07-31, v4).

## AV1 videos crash job-container dataloaders (2026-07-31 — cost 3 failed run-pairs)

The IPEC-COMMUNITY LIBERO suites ship AV1-encoded mp4s. torchcodec dlopens the
CONTAINER's libavcodec; the fuyao job image (liuw50-260318-0232) has an ffmpeg
whose AV1 decode fails on specific streams -> deterministic
`RuntimeError: Could not push packet to decoder` in a dataloader worker at a fixed
data position (same rank/worker/step every attempt), plus a red-herring
`cudaErrorContained` nvlink watchdog error during NCCL teardown. The SAME files
decode perfectly on the interactive box (newer ffmpeg) under ffmpeg CLI, torchcodec
sampled/full-episode decode, and the loader's exact kwargs — so local sweeps CANNOT
reproduce it; only job containers can. libero_10 AV1 happened to survive earlier
fine-tune runs; SUITE=all hit a bad stream in spatial/object/goal.
**Fix (applied): `examples/SimplerEnv/convert_av1_to_h264.py <root> -j 48` over
`/libero_groot` — LOSSLESS (-qp 0) in-place re-encode, 3,386 files, 3.3->7.5 GB.**
Rule: any new LeRobot dataset provisioned for fuyao training must be H264 (run the
converter after download; check with ffprobe). Note the script takes a POSITIONAL
root ('--root' in the SimplerEnv README is stale).

## gr1_unified provisioning lessons (2026-08-01)

NVIDIA's gr1_unified.* dirs (X-Embodiment-Sim) need THREE prep steps before the
GR00T trainer accepts them (smoke-verified end-to-end afterward):
1. `scripts/repair_lerobot_metadata.py <dir> --embodiment-tag ROBOCASA_GR1_TABLETOP`
   (file-index repair + relative-action stats).
2. **dtype fix**: their info.json declares observation.state/action as dtype
   "object" -> `generate_stats` silently skips them (it only stats "float*"
   features) -> loader dies with `KeyError: observation.state`. Fix: rewrite
   dtype to "float64" (shapes are fixed [44]), keep .json.bak backups.
3. `gr00t.data.stats.generate_stats(dir)` -> writes observation.state/action
   mean/std/min/max/q01/q99 into meta/stats.json. ASSERT the keys landed —
   step 2's silence is exactly the silent-no-op trap.
Runner support added: `DATASET_GLOB`/`DATASET_PATH`/`EMBODIMENT_TAG` env vars in
groot_fuyao_train.sh (forwarded by submit wrapper); videos already h264.

## GR1-tabletop eval assets (2026-08-01)

The GR1 sim needs 3 asset families; every upstream host except one is blocked:
- DigitalCousin zips (lightwheel/sketchfab): `nvidia/PhysicalAI-DigitalCousin-Assets`
  on HF — fetch via hf-mirror resolve URLs, unzip into
  `robocasa/models/assets/objects/{lightwheel,sketchfab}` (zip has the name-dir inside).
- Kitchen assets (objaverse/fixtures/textures/aigen/generative): upstream
  utexas.box.com is DEAD from the cluster (301 ok, CDN stalls at 0 bytes).
  **Mirror: `robocasa/robocasa-assets` on HF (9.9 GB, all five zips)** — resolve
  via hf-mirror. objaverse -> assets/objects/objaverse, fixtures -> assets/fixtures, etc.
- The setup script honors SKIP_DOWNLOAD_ASSETS=1 once assets are pre-staged; its
  patched download_groot_assets.py now points at hf-mirror anyway.

## Dataset download playbook (2026-07-31 saga)

- `hf`/`hfd` via hf-mirror FAIL on repos >1000 files: the mirror's pagination
  Link headers point at huggingface.co (blocked), and hfd chokes on truncated
  multi-MB metadata JSON (misreports as "requires authentication"). Small repos
  (<1000 files, e.g. the LIBERO suites) work fine.
- hf-mirror.com also has outages (down hours on 2026-07-31); huggingface.co
  unreachable from the cluster entirely.
- **Best rail: ModelScope** — CN-native, fast (~7 files/s), reliable. The
  `AI-ModelScope/*` namespace mirrors many HF repos 1:1: bridge_orig_lerobot,
  fractal20220817_data_lerobot, PhysicalAI-Robotics-GR00T-X-Embodiment-Sim all
  present; StarVLA/RoboTwin-{Clean,Randomized} exist under their own namespace.
  `modelscope download --dataset <id> [--include pat] --local_dir <dir>` in a
  tmux retry loop. Check availability:
  `curl https://modelscope.cn/api/v1/datasets/<ns>/<name>` -> 200/404.
- Fallback for HF-only repos: $BASE/projects/groot/mirror_list_dl.py — paginates
  the mirror tree API rewriting cursor URLs back to the mirror, then aria2c.
- After ANY dataset lands: H264-convert + verify (see AV1 lesson above).

## Success signals for the first router run

`condition_router params not in checkpoint - identity-initialized (28 keys)` and `create_optimizer: router group with 27 params at lr=0.001` in the log; then finite decreasing loss. Baseline target (official): libero_10 → **94.35** success (suite avg 97.0 across the four).

## Qwen3-VL 36-layer pivot (2026-08-03) — implementation complete, smokes green

Approved design: replace Cosmos-Reason2-2B with the OFFICIAL Qwen/Qwen3-VL-4B-Instruct
(shared cache /dataset_rc/robot/hf/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/
ebb281ec... — pure Qwen release, NOT iron_vla-tuned; iron_vla only inits from it).
36 LLM layers (hidden 2560) + 36-layer DiT -> 18 cross blocks -> router [18, 4].
Candidates [9 18 27 36] = hidden_states indices (0=embedding, i=layer i; 36=stock
final tap) = iron_vla {8,17,26,-1} translated. Whole model 6.02B, from-scratch DiT.

New knobs (all tyro CLI flags via EXTRA_ARGS):
- --select-layer 36 --backbone-embedding-dim 2560 --dit-num-layers 36
  (configs/finetune_config.py -> launch_finetune.py copies into model cfg;
  dit_num_layers writes diffusion_model_cfg["num_layers"]).
- --router-init-mode span: logits[b, b*K//B] = init_bias (iron_vla depth-aligned
  identity; 5/4/5/4 block split for B=18,K=4). Default "last" = GR00T incumbent.
- --router-frozen: fixed-mapping arm. Ctor freezes logits; set_trainable_parameters
  RE-APPLIES the freeze (its blanket requires_grad=True undid the ctor freeze —
  caught in construction smoke). Norms stay trainable. Optimizer groups filter on
  requires_grad, so frozen arm shows "router group 8 params" vs learned 9.
- Fixed arm uses bias 16.0 (softmax mass 0.9999996 = numerically hard); learned
  arm bias 3.0 (87% on aligned candidate, iron_vla convention).

Smokes on the dev L20X (30 steps, batch 4, gr1_unified, skip-weight-loading,
tune-llm+visual, backbone-lr 1e-5): BOTH ARMS GREEN. Learned: loss 1.095, router
group 9 @1e-3, backbone 713 @1e-5, config.json records all new fields. Fixed:
loss 1.094, RouterLLM w_mean L09/18/27/36 = .278/.222/.278/.222 (exact 5/4/5/4),
entropy 5.8e-6. Runs: projects/groot_runs/smoke_qwen36_{router,fixed}.

### Qwen36 launch recipe (pending approval)
Memory probes on L20X (144GB, 1 GPU, no ZeRO sharding): per-GPU batch 64 OOMs
(needs >141.5GB); batch 32 runs but peaks 142.2GB. Real jobs get ZeRO-2 sharding
across 8 GPUs (optimizer states /8), so per-GPU 32 is comfortable there — use
GLOBAL_BATCH_SIZE=256 + --gradient_accumulation_steps 2 = effective 512 (same
sample budget as s1_gr1_* pair; max_steps counts optimizer steps). NEVER launch
this arch at per-GPU 64. submit_fuyao_groot.sh now forwards GR00T_BACKBONE_PATH
(was missing -> job would silently fall back to Cosmos). Prior 3B GR1 pair:
60K steps in ~7.2h on 8xH200; 6B/36L estimate ~15-18h per run.
