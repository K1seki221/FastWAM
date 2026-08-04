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
