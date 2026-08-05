# 13 — GR00T condition-router: local training launch & profile

This fork's research contribution lives in the vendored `Isaac-GR00T/`: a learned
**condition router** that lets each DiT cross-attention block choose which VLM
layer(s) to read, instead of GR00T's fixed "one tap layer feeds every block".
This doc covers how to launch training (locally and at scale) and the measured
training profile. Design rationale: `11_groot_router_design.md`; cluster ops:
`12_groot_fuyao_runbook.md`.

## What the router is

- Backbone exposes K candidate hidden states (`extra_hidden_layer_indices`;
  index 0 = embedding output, i = output of LLM layer i).
- Per DiT cross block `b` (blocks with cross-attention = every other block),
  the router holds logits `[num_cross_blocks, K]` and per-candidate LayerNorms.
  The conditioning fed to block `b` is `Σ_k softmax(logits[b])_k · norm_k(h_k)`.
- Identity init: logits are zero except an `init_bias` on one column, so the
  run starts ≈ equal to the non-router baseline and must *learn* to deviate.

Two supported architectures:

| | backbone | LLM layers kept | hidden | DiT layers | cross blocks | candidates |
|---|---|---|---|---|---|---|
| stock N1.7 | Cosmos-Reason2-2B | `--select-layer 12` (default) | 2048 | 16 | 8 | any subset of 0..12 |
| Qwen36 | Qwen/Qwen3-VL-4B-Instruct | `--select-layer 36` (all) | 2560 | `--dit-num-layers 36` | 18 | `9 18 27 36` |

Total params: stock ≈ 3.2B; Qwen36 = **6.02B** (4B VLM + 1.26B DiT + heads).

## Environment

- Python venv: `UV_PROJECT_ENVIRONMENT=<path> uv sync` inside `Isaac-GR00T/`
  (on fuyao we keep it at `/dataset_rc/$USER/projects/groot/venv` — *outside*
  the repo, both because fuyao snapshots the submit dir and to survive repo moves).
- HF caches: the runner hard-pins `HF_HOME`/`HF_HUB_CACHE` via `GROOT_HF_HOME`
  and runs `HF_HUB_OFFLINE=1`. Point `GR00T_BACKBONE_PATH` at a *local snapshot
  dir* of the backbone (repo-id loading makes a hub API call that crashes
  offline with transformers 4.57.x).
- Datasets: LeRobot v2 + GR00T `meta/modality.json`. All videos must be H264
  (job-image ffmpeg cannot decode some AV1; convert losslessly with `-qp 0`).

## UCSB box provisioning (2026-08-04, verified)

Everything large lives under `/data/ruijiezhang/` (28T volume). GPUs: use
**4–7 only** (94GB H100 NVL; 0–3 belong to others). Direct huggingface.co
access works — no mirror gymnastics needed.

| what | where |
|---|---|
| venv (uv sync of `Isaac-GR00T/`, torch 2.9.0+cu128) | `/data/ruijiezhang/env/groot` → pass as `VENV_DIR` |
| HF cache (GR00T-N1.7-3B, Cosmos-Reason2-2B, Qwen3-VL-4B-Instruct @ebb281ec) | `/data/ruijiezhang/hf/hf_cache` → pass as `GROOT_HF_HOME` |
| gr1_unified corpus (24 dirs, 42GB, PREPPED: dtype+stats+rel_stats done, all H264) | `/data/ruijiezhang/gr1_unified` → `DATASET_GLOB="/data/ruijiezhang/gr1_unified/gr1_unified.*"` |
| LIBERO 4 suites (modality.json + ep82 patch applied) | `/data/ruijiezhang/libero` → pass as `DATA_ROOT` |
| runs | `/data/ruijiezhang/groot_runs` → pass as `RUNS_ROOT` |
| uv wheel cache | `/data/ruijiezhang/uv_cache` (`UV_CACHE_DIR`) |

### Local 4-arm Qwen28 experiment (launched 2026-08-04)

Arch "Qwen28": Qwen3-VL-**2B** (28 layers, hidden 2048 — same width as Cosmos),
28L DiT from scratch -> 14 cross blocks, tune-all (~3.2B), fits ONE 94GB H100
natively (73GB peak @ micro 16) — no offload needed. Effective batch 512
(16 micro x 32 accum), 60K steps, router lr **2e-3** all arms, backbone 1e-5.
Measured 8.46 s/step -> ~5.9 days/arm. wandb: kiseki_rigel/finetune-gr00t-n1d7.

| arm | GPU | flags |
|---|---|---|
| A `qwen28_A_fixed_span` | 5 | span, bias 16, frozen — hard depth-aligned quarters {7,14,21,28} |
| B `qwen28_B_uniform_k4` | 7 | span, bias **0** = exact uniform over {7,14,21,28}, learnable |
| C `qwen28_C_fixed_last` | 4 | last, bias 16, frozen — all blocks read layer 28 (stock incumbent) |
| D `qwen28_D_uniform_k28` | 6 | bias 0 uniform over ALL layers 1..28 (K=28), learnable |

All four run in PARALLEL (GPUs 4-7 freed up mid-session). Launchers:
`/data/ruijiezhang/groot_runs/launch_gpu{5,7}.sh` (A/B; their queued 2nd legs
were detached by killing the wrapper bash — trainers reparented to init) and
`launch_gpu4_C.sh` / `launch_gpu6_D.sh`. Exit codes -> `chain_gpu{4,5,6,7}.status`.

**2026-08-05: full runs PAUSED at complete checkpoint-6000** (all four) to run
a fast pilot phase first. Resume later with `--resume-from-checkpoint` via
finetune.sh (add to EXTRA_ARGS is NOT enough — it's a first-class flag:
`bash examples/finetune.sh ... --resume-from-checkpoint`; runner passthrough
needs EXTRA_ARGS="--resume_from_checkpoint" tyro form) + reattach wandb with
`WANDB_RUN_ID=<id> WANDB_RESUME=must` (ids: A=jf6re4z4 B=24bwkudf C=af5j0nnq
D=kwqwmjrn). Loss curves to 6K: all four ~indistinguishable (expected — router
effects are small vs early-training dynamics).

### Pilot phase (launched 2026-08-05, ~12h)

Five arms as `pilot_*` runs via `/data/ruijiezhang/groot_runs/launch_pilot.sh A|B|C|D|E`:
6-task PnP subset (`DATASET_GLOB=.../gr1_unified.PnP*`, 6k episodes), 10K steps
@ eff batch 256 (16x16) ≈ 2.3 epochs, save 2500/limit 2, ports 29521-25, GPU map
A=5 B=7 C=4 D=6 E=3 (user granted GPU 3 mid-session). Purpose: fast idea
verification (loss separation + routing structure) before the 6-day full runs.
Arm E = `pilot_E_uniform_k4_baselr`: same as B but ROUTER_LR=1e-4 (= DiT base
lr, i.e. no 20x router-LR boost) — isolates the router-LR choice.
**Arms E and F were killed at ~step 400-500 on user request (2026-08-05)** —
GPUs 2/3 released; the LR and freeze-delay ablations moved to another server
(pull K1seki221/FastWAM main and use scripts/launch_pilot_qwen28.sh E|F).
Core pilots = A-D only.

### Architecture rev 2: per-candidate proj adapters (2026-08-05)

v1-arch pilots A-D killed at ~step 4-5K (curves preserved on wandb; dirs
`pilot_*` without suffix are v1-arch leftovers). Relaunched as
`pilot_*_pcproj` with NEW flag `--router-candidate-proj`
(`PILOT_SUFFIX=_pcproj ROUTER_PCPROJ=1` in the launcher): each candidate gets
an identity-init `Linear(2048,2048)` between its LayerNorm and the mix —
candidate-specific VLM->conditioning alignment, block-shared, router weights
still applied upstream of the blocks' shared to_k/to_v. Rationale: moving the
mix across the (linear, shared) to_k/to_v alone is a mathematical no-op; the
per-candidate adapters are what make placement meaningful. Costs: K=4 -> 16.8M
extra params, K=28 -> 117M. Optimizer: adapters ("projs." names) are EXCLUDED
from the router group -> base lr 1e-4 + normal decay; router group stays
logits+norms at 2e-3/wd0 (same param counts as before: 9 learned / 8 frozen).
Unit checks in-session: identity behavior at hard init, proj perturbation
affects only aligned blocks, group name filter.

Arm G = `pilot_G_uniform_k4_entropy_pcproj` (GPU 1, port 29527): B_pcproj plus
NEW flag `--router-entropy-coef 0.02` — loss -= coef(t) * mean routing entropy,
coef annealed linearly to 0 over max_steps (explore-then-commit). Computed in
Gr00tTrainer.compute_loss straight from the logits (static router => no
forward plumbing); zero gradient at exact uniform, restoring force once
weights drift, force fades as coef anneals. Unit-checked both properties.
LOGGING (since 2nd G launch; first G killed at ~2.7K for this): "loss"
(wandb train/loss) = pure TASK loss, directly comparable across arms;
"loss_with_entropy_bonus" = optimized objective; "router_entropy_bonus" =
coef(t)*H. Implemented via window accumulators in compute_loss + override in
Gr00tTrainer.log().
NOTE: measured VLM layer-norm scales (text probe, Qwen3-VL-2B): L1 15, L7 852,
L14 949, L21 1527, L27 3371, L28 2452 — 159x spread over pool 1..28; the
per-candidate LayerNorms are load-bearing (esp. arm D). Watch norm/proj gains
when interpreting W (effective contribution = w * gain).

Arm H = `pilot_H_uniform_k4_mixnorm_pcproj` (GPU 0, port 29528): B_pcproj plus
NEW flag `--router-mix-renorm` — mixture rescaled by 1/sqrt(sum w^2), which
decouples conditioning MAGNITUDE from routing entropy (mix of unit-RMS
decorrelated candidates has RMS ~ sqrt(sum w^2): one-hot 1.0, uniform K=4
0.5, K=28 0.19 — so un-renormed uniform arms start with 2-5x weaker
conditioning than fixed arms; magnitude confound in early A-vs-B loss gaps).
Exactly 1 at one-hot => stock identity preserved. Unit-checked: one-hot
no-op, uniform K=4 exactly 2x, grads flow. Also NOTE: vlln is BYPASSED in the
routed path (per-candidate norms replace it; no post-mix norm by design —
a post-mix LayerNorm would break exact stock identity).
Arm F = `pilot_F_uniform_k4_freeze500` (GPU 2, port 29526): same as B plus NEW
flag `--router-freeze-steps 500` (5% of steps, scale with budget) — logit grads
dropped pre-optimizer-step for the first N steps ("let the DiT settle before
the router chooses layers"), then released. Implemented 2026-08-05:
`RouterFreezeDelayCallback` in gr00t/experiment/trainer.py (grad-drop keeps
logits in the optimizer group; requires_grad=False would exclude them at
create_optimizer). Wired through finetune_config -> launch_finetune -> model
config, mirroring router_lr. Verify in logs: "router-freeze-delay: logits
frozen for first N" at start, "released at global_step=N" when opened;
RouterLLM curves must stay exactly uniform until N.
NOTE init change vs fuyao pairs: learned arms start UNIFORM (not span-87%).

Env quirks this box (all baked into the launchers):
- torchcodec needs conda FFmpeg 7: `LD_LIBRARY_PATH=/data/ruijiezhang/env/ffmpeg7/lib`
  AND `LD_PRELOAD=/data/ruijiezhang/env/ffmpeg7/lib/libstdc++.so.6` — the venv's
  python is miniconda-based and its RPATH drags in an ancient libstdc++ otherwise.
- Single-GPU DeepSpeed+CPU-offload machinery exists if ever needed again:
  `GROOT_FORCE_DEEPSPEED=1 GROOT_DS_OFFLOAD=cpu` (env-gated patches in
  experiment.py / base_config.py / trainer.py / finetune.sh) — unused for 2B.

`scripts/prep_gr1_unified.py` (this repo) = one-shot gr1_unified prep:
manifest completeness gate → info.json dtype fix (object→float64) → metadata
repair → stats+rel-stats regen with asserts → H264 fourcc sweep. Idempotent;
re-run after re-downloading anything. Remote manifest cached at
`/data/ruijiezhang/gr1_unified/.remote_manifest.json`.

## Local launch (single GPU)

Everything goes through `scripts/groot_fuyao_train.sh` (works on any box with a
GPU, not just fuyao). 30-step smoke of the **learned-router arm** on the GR1
corpus with the Qwen3-VL 36-layer architecture:

```bash
cd FastWAM
QSNAP=/path/to/hf/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/<hash>

DATASET_GLOB="/path/to/gr1_unified/gr1_unified.*" \
EMBODIMENT_TAG=ROBOCASA_GR1_TABLETOP \
GR00T_BACKBONE_PATH=$QSNAP \
RUN_NAME=smoke_qwen36_router NUM_GPUS=1 MAX_STEPS=30 GLOBAL_BATCH_SIZE=4 \
USE_ROUTER=1 ROUTER_LR=1e-3 ROUTER_LAYERS="9 18 27 36" \
EXTRA_ARGS="--skip-weight-loading --tune-llm --tune-visual --backbone-lr 1e-5 \
  --select-layer 36 --backbone-embedding-dim 2560 --dit-num-layers 36 \
  --router-init-mode span --router-init-bias 3.0" \
bash scripts/groot_fuyao_train.sh
```

The **fixed-mapping control arm** is identical except
`--router-init-bias 16.0 --router-frozen` (softmax(16,0,0,0) ≈ 0.9999996 →
numerically a hard depth-aligned mapping; frozen logits, norms still train —
so the two arms differ *only* in whether the mixing logits are learnable).

Stock-architecture LIBERO fine-tuning instead: drop the Qwen/arch flags and use
`SUITE=10|goal|object|spatial|all` (datasets under `DATA_ROOT`), e.g.

```bash
SUITE=10 RUN_NAME=router_libero_10 USE_ROUTER=1 ROUTER_LR=1e-3 NUM_GPUS=1 \
MAX_STEPS=20000 GLOBAL_BATCH_SIZE=64 bash scripts/groot_fuyao_train.sh
```

Useful env knobs (all forwarded by the fuyao submit wrapper too):
`DATASET_GLOB` / `DATASET_PATH` (`:`-joined dirs) / `SUITE`, `EMBODIMENT_TAG`,
`RUN_NAME`, `NUM_GPUS`, `MAX_STEPS`, `GLOBAL_BATCH_SIZE`, `SAVE_STEPS`,
`SAVE_TOTAL_LIMIT`, `USE_ROUTER`, `ROUTER_LR`, `ROUTER_LAYERS`, `EXTRA_ARGS`
(raw tyro flags passthrough), `WANDB_API_KEY` (wandb auto-on when set),
`GROOT_HF_HOME`, `GR00T_BACKBONE_PATH`.

## Router CLI reference (tyro flags, via `EXTRA_ARGS` / `launch_finetune.py`)

| flag | meaning |
|---|---|
| `--use-condition-router` | enable the router (requires AlternateVLDiT + vlln) |
| `--router-candidate-layers 9 18 27 36` | hidden_states indices exposed as candidates (0=embedding, i=layer i; omit → all of 0..select_layer) |
| `--router-init-bias B` | logit bias on the favored candidate at init; mixture mass = softmax([B,0,..]) → 3.0 ≈ 87% (soft, trainable), 16.0 ≈ 1.0 (hard) |
| `--router-init-mode last\|span` | which candidate gets the bias: `last` = deepest (GR00T incumbent), `span` = depth-aligned block-span i → candidate i (iron_vla style) |
| `--router-frozen` | freeze logits (fixed-mapping control); per-candidate norms stay trainable |
| `--router-lr 1e-3` | dedicated LR for router params (wd=0 group) |
| `--backbone-lr 1e-5` | dedicated LR for the VLM backbone |
| `--select-layer N` | keep LLM layers 1..N (truncates above N) |
| `--backbone-embedding-dim D` | backbone hidden size → DiT cross-attention dim (2048 Cosmos / 2560 Qwen3-VL-4B) |
| `--dit-num-layers N` | DiT depth; cross blocks = (N+1)//2 |
| `--skip-weight-loading` | stage-1/from-scratch: build DiT+heads fresh, only the VLM starts pretrained |
| `--tune-llm --tune-visual` | make the VLM trainable (stage-1 recipe) |
| `--gradient_accumulation_steps A` | effective batch = global_batch_size × A |

## Training profile

Recipe used for the stage-1 pairs (both LIBERO and GR1, baseline and router):

- **Optimizer**: AdamW, base LR `1e-4` (DiT/action head), weight decay `1e-5`,
  warmup ratio 0.05. Per-module groups: backbone `1e-5`, router `1e-3` (wd 0).
  Optimizer groups filter on `requires_grad` (frozen router logits drop out;
  log line `create_optimizer: router group N params` — 9 = learned, 8 = frozen).
- **Precision/parallelism**: bf16 mixed precision, DeepSpeed **ZeRO-2** (trainer
  default), torchrun data-parallel. Gradient checkpointing off.
- **Batch**: effective 512 (GR1 pairs) / 640 (LIBERO). For the 6B Qwen36 arch:
  `GLOBAL_BATCH_SIZE=256` + `--gradient_accumulation_steps 2`. `max_steps`
  counts optimizer steps, so 60K × eff-512 is the same sample budget either way.
- **Steps/cadence**: 60K steps, `SAVE_STEPS=2000`, `SAVE_TOTAL_LIMIT=40`
  (30 checkpoints retained), `state_dropout_prob 0.2`, color jitter on.

Measured memory (per GPU, 6B Qwen36 arch, bf16 + ZeRO-2 unsharded 1-GPU probe
on a 144GB L20X):

| per-GPU batch | result |
|---|---|
| 4 | fits easily (smoke) |
| 32 | fits, peak ≈ 142.2 GB — the 8-GPU ZeRO-2 run shards optimizer state, so much roomier there |
| 64 | **OOM** (needs >141.5 GB) — never launch this arch at per-GPU 64 |

Measured wall-clock (8×H200, effective batch 512, 60K steps):

| arch | time |
|---|---|
| stock 3B (Cosmos 2B + 16L DiT) | ≈ 7.2 h |
| Qwen36 6B (36L + 36L DiT) | ≈ 15–18 h (est. ~2.2× compute) |

Single L20X reference throughput (Qwen36, batch 4): ≈ 2.5 it/s.

## Telemetry (wandb project `finetune-gr00t-n1d7`)

`RouterLLM/*` metrics, logged every `logging_steps` alongside loss:

- `w_mean_L<idx>` — mean softmax mass on candidate layer `idx` across blocks.
- `w_incumbent_min` / `w_incumbent_mean` — mass on the incumbent (last) candidate.
- `entropy` — mean routing entropy across blocks (0 = hard one-hot; the frozen
  arm logs ≈ 5.8e-6, the learned span-init arm starts ≈ 0.53).

Sanity signatures at init (Qwen36, span): `w_mean` L09/L18/L27/L36 ≈
.278/.222/.278/.222 (the 5/4/5/4 block-span split of 18 cross blocks over 4
candidates).

## Evaluation

- LIBERO 40-task: `scripts/groot_fuyao_eval.sh` (`CKPT=<checkpoint dir>`).
- RoboCasa GR1-tabletop 24-task: `scripts/groot_fuyao_eval_gr1.sh`
  (server + robocasa client venv; see `12_groot_fuyao_runbook.md` for the
  one-time sim-asset provisioning).

## Gotchas (short list; full war stories in `12_groot_fuyao_runbook.md`)

- Router silently absent → check the one-shot `router-log probe … has_router=True`
  warning and the `create_optimizer: router group …` line at start.
- All dataset videos must be H264 (AV1 crashes the job-image decoder mid-run).
- `fuyao deploy` snapshots the *cwd* — submit via `scripts/submit_fuyao_groot.sh`
  (deploys from a tiny temp dir; never run raw deploy inside the repo).
- Shared host network namespace on fuyao → eval server ports are randomized.
