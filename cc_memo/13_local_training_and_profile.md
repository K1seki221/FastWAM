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

**Arms A and C (frozen-router baselines) killed at ~1.5K/10K on user request
(2026-08-05; died pre-first-checkpoint, partial curves only)**: even frozen,
they carry trainable per-candidate norms + pcproj adapters (16.8M) the stock
model lacks — fine as learnability controls, wrong as "no router" anchors. Replaced by Arm Z = `pilot_Z_stock` (GPU 2, port
29529, launcher arm Z, USE_ROUTER=0): TRUE no-router stock path — single
select_layer-28 tap through vlln, no candidate norms, no adapters. Z-vs-C
would isolate adapter capacity; Z-vs-learned-arms = "router vs stock".
A/C partial curves (to ~7K) remain on wandb (gpwc160z / jwnldrta).
Arm A_hard = `pilot_A_fixed_span_hard` (GPU 5, port 29521): the MINIMAL
hard-wired depth-aligned baseline — frozen span logits (= hard wiring; no
router-free code path can express per-block taps), NO pcproj adapters,
per-tap norms at BASE lr 1e-4 (`PILOT_ROUTER_LR` launcher knob) so they
train exactly like Z's vlln. Capacity over Z: only 4 LayerNorms vs 1 (16K
params). Decomposition: Z vs A_hard = mapping; A_hard vs B = mapping +
capacity + learnability.

**2026-08-05 (later): baselines get per-tap projectors (iron_vla parity,
user decision)** — iron_vla's no-router configs ALWAYS have a per-tapped-layer
Linear projector (+ weightless shared RMSNorm); our A_hard/Z lack that
capacity. Fixes, run alongside (8 arms = all 8 GPUs):
- A_pcproj RERUN fresh on GPU 4 port 29531 (original died pre-checkpoint):
  frozen span + norms(2e-3) + identity-init projs(base lr) — capacity-matched
  fixed control for B AND the faithful iron_vla-fixed-mapping analog.
- Arm Y = `pilot_Y_last_proj` (GPU 3, port 29530, launcher arm Y +
  ROUTER_PCPROJ=1): last-layer baseline WITH projector, implemented as K=1
  candidate pool {28} — softmax of a single logit is EXACTLY 1.0 (no
  leakage, no dead params); one learnable norm + one identity-init proj.
  Y vs Z isolates the projector; Y vs A_pcproj isolates the mapping at
  matched capacity; A_pcproj vs B isolates learnability.
Z and A_hard kept running (free GPUs; complete the capacity ladder
Z -> A_hard -> Y/A_pcproj -> B).

### Sigmoid accumulation gates (2026-08-05, user-approved design)

NEW flag `--router-gate-mode sigmoid` (+ `--router-gate-init-hi/-lo`,
defaults 0.9/0.1): independent per-candidate gates g=sigmoid(logits) replace
the competitive softmax — "which SET of layers helps" (accumulation) vs
"which layer" (allocation). Design goal per user: beat the no-router
baseline; NOT magnitude-comparable with softmax arms by construction.
- Init = identity-accumulate: span-aligned gate 0.9, others 0.1 (both off
  the saturation rails; zero logits would mean 0.5 gates = 2x loud — trap).
- No renorm (accumulation semantics intact; identity init anchors start
  loudness ~0.92 of baseline).
- Guard: sigmoid + router_entropy_coef>0 raises (entropy bonus assumes a
  softmax distribution).
- Telemetry: RouterLLM/gate_sum (learned conditioning budget per block; the
  headline sigmoid metric), gate_min (dead-gate watch); entropy key becomes
  summed Bernoulli entropy; w_mean_L*/sqrt_sum_w2/rms_mix unchanged.
- USER DESIGN DISCUSSION (2026-08-05 late): baselines = Y (last+proj) and
  A_pcproj (span+proj, PRIMARY bar). First sigmoid experiment = the
  MAGNITUDE question, two arms, both identity-accumulate, gate lr = DiT lr
  1e-4 (not 2e-3 — note sigmoid grads carry g(1-g)~0.09 factor, so
  effective steps are small), no dead-gate insurance:
  * `pilot_S_sigmoid_k4_free` (GPU 2, port 29533): magnitude free.
  * `pilot_S_sigmoid_k4_renorm` (GPU 5, port 29534): + --router-mix-renorm.
  Init telemetry verified: gate_sum 1.2000 exact, free rms_mix 1.068,
  renorm rms_mix 1.167 (= 1.068/0.917 — the overcorrection nuance live).
- SECOND sigmoid pair (user-ordered, auto-launch when B/D finish ~7h):
  OPEN-EXPLORATION init = all gates 0.5 (hi=lo=0.5; max-gradient point;
  NOTE at all-0.5 sqrt(sum g^2)=1 so renorm is a NO-OP at init, engages as
  gates differentiate; both start ~1.5x baseline loudness):
  * `pilot_S_sigmoid_k4_open_free` (GPU 7 after B, port 29535)
  * `pilot_S_sigmoid_k4_open_renorm` (GPU 6 after D, port 29536)
  2x2 sigmoid matrix complete: {identity-accumulate, open} x {free, renorm}.
- THIRD sigmoid pair (2026-08-06, gate-LR sweep; 10K-run finding: at gate lr
  1e-4 the g(1-g)~0.09-0.25 factor left gates nearly frozen): gate LR 5e-3,
  open init 0.5, renorm ON:
  * `pilot_S_sigmoid_k4_open_renorm_glr5e3` (GPU 6, port 29537)
  * `pilot_S_sigmoid_k28_open_renorm_glr5e3` (GPU 7, port 29538, K=28 pool)
  Launcher gained PILOT_NAME / PILOT_LAYERS overrides.
- Arm T = `pilot_T_token_k4` (GPU 1, port 29539, launcher arm T + pcproj):
  PER-TOKEN content routing (user-approved design): score[b,k,s] =
  logits[b,k] + v_b.x_k[s]/sqrt(D), zero-init per-block probes, uniform
  logits prior, softmax per token, NO renorm, premixed once per backbone
  pass. Design notes: DiT-queried variant REJECTED because flow matching
  trains one random-t step and training-time DiT states are noise-dominated
  (Beta-skewed to high noise) — a DiT-state query is mostly untrainable;
  denoise-step (temb) query also deferred per user. T starts bit-identical
  to B => T-vs-B isolates content-dependence. Watch
  RouterLLM/alpha_token_std: ~0 means content routing unused.
- Arm T_k28 = `pilot_T_token_k28` (GPU 5, port 29540): T with the full
  candidate pool 1..28 (uniform prior 1/28 per token; 28 pcproj adapters).
  T_k28-vs-D isolates per-token routing at K=28; T_k28-vs-T_k4 pool size.
  Z and A_hard were killed by the user to free those GPUs (projector-less
  ladder rungs culled; partial curves on wandb).
- Dead-gate risk note: sigmoid grad ~ g(1-g) vanishes at BOTH rails — watch
  gate_min; a gate that closes early may never reopen.

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

- LOCAL GR1 sim eval (2026-08-06, working): setup via
  `gr00t/eval/sim/robocasa-gr1-tabletop-tasks/setup_RoboCasaGR1TabletopTasks.sh`
  (builds robocasa_uv/.venv + 8.8G assets in-repo; sanity test included).
  Fleet driver: `/data/ruijiezhang/groot_runs/eval_fleet.sh` (8 ckpts x 6 PnP
  tasks x 10 eps, GPUs 0-5, results -> /data/ruijiezhang/groot_evals/).
  LESSONS: video recording needs ffmpeg on PATH (use
  /data/ruijiezhang/env/ffmpeg7/bin) BUT multi-env AsyncVectorEnv workers
  SIGABRT in robosuite read_pixels (EGL multi-context, spawn workers, NO
  error text — expose with PYTHONFAULTHANDLER=1). Single-env rollouts are
  stable => always N_ENVS=1 locally; NO_VIDEO=1 + `--video-dir none` (new
  flag) skips recording. Smoke B ckpt: PnPCanToDrawer 1/2 in first try.
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


## First-wave pilot EVAL results (2026-08-06, complete)

6 trained PnP tasks x 10 eps, ckpt-10000, GR1 sim single-lane (EGL constraint).
Final means: H(mix-renorm) .567 | E*(ext) .467 | So(sigmoid open free) .450 |
Y(last+proj BASELINE) .433 | B .400 | G .400 | D .383 | Si .383 | F*(ext) .383 |
A(span BASELINE) .333 | Sr .333 | Sn .300.

Findings:
1. H is the only arm clearly above the incumbent baseline (+13.4 pts; 34/60
   vs 26/60). Same config as B except 1/sqrt(sum w^2) mix rescale.
2. Uncompensated learned routing (B/D) does NOT beat the last-layer
   incumbent. Span-aligned routing (A/Si/Sr) is the WORST direction.
3. Loudness pattern: loud arms (H ~1.55x early, So ~1.58x) hold ranks 1-2;
   attenuated arms cluster below baseline. "Early conditioning scale matters
   for from-scratch DiT" is a live alternative mechanism to compensation
   dynamics — arm X (H-exact, measured-energy renorm to exactly 1.0)
   discriminates.
4. NOISE CALIBRATION: So vs Sn are near-functional clones yet differ by 15
   pts (.450/.300) — the total noise floor (training chaos + 60-ep eval) is
   ~15 pts. H-vs-Y (+13.4) is AT that edge: needs X corroboration + 30-ep
   re-eval before any paper claim. Zero-shot on the 18 untrained Posttrain
   tasks: 0.0 across the board (B's accidental 24-task sweep).
5. G's entropy anneal ended ~uniform but scored .400 — annealed exploration
   gave no benefit; NOTE its anneal ended together with cosine LR decay
   (design flaw: never let those end simultaneously).
Second wave (T_k4, T_k28, sigmoid glr5e3 x2, X) trains overnight; X
auto-evals when the lane frees.


## FINAL 30-episode verdict (2026-08-07, pilot campaign COMPLETE)

180 eps/arm (6 tasks x 30 eps), ckpt-10000, single GPU-0 render lane.
Board: X(H-exact) .483 | Y(BASELINE) .467 | H .439 | Sg4 .406 | Sg28 .400 |
B .394 | So .378 | A(BASELINE 2, added 08-07) .350 | T_k28 .317 | T_k4 .306.
(10-ep only: G .400, D .383, Si .383, Sr .333, Sn .300; ext: E* .467,
F* .383.) A at 30 eps confirms its 10-ep .333: vs Y gap 11.7pp, z=2.25,
nominal p=.024 — but as a designed baseline pair (planned comparison) this
is the fairest read; under a 9-arm Bonferroni it is marginal (~.22).
Last-layer > span-fixed is now the best-supported baseline ordering.
CSV: /data/ruijiezhang/groot_evals/final_table.csv (+ per-arm pilot_*_e30/).

Statistics (paired per-task + pooled two-proportion, independently verified):
single-arm SE ~3.7pp; diff SE ~5.2pp => ~10.4pp needed for 2-sigma. NO pair
among {X,Y,H,B} is distinguishable: X-Y +1.7pp (z=0.32, pure noise), X-H
+4.4pp (z=0.85), X-B +8.9pp (z=1.70, p=.089, best gap in study), Y-B +7.2pp
(z=1.38). Resolving 5pp steps needs ~800+ eps/arm. X is the max of 9 arms
=> its .483 is winner's-curse inflated; expect regression on re-measure.

Findings (with adversarial-review hedges applied):
1. NOTHING BEATS THE INCUMBENT. No routing variant is distinguishable from
   Y; the only claim the data supports is "compensated routing reaches
   statistical parity" — equivalence is NOT established (X-Y CI ~[-9,+12]pp).
2. The ONLY Bonferroni-surviving result of the campaign: per-token routing
   (T arms) is genuinely harmful vs Y (z~2.9-3.1, corrected p~.02-.04).
   A(span fixed) is suggestive-worse (corrected p~.09). The sigmoid cluster
   and "K=28 never helps" are no-evidence-of-benefit observations only.
3. H-exact vs H (the approved question): YES at behavior level — identical
   learned routing (both: L28 .41-.42 > L21 .24 > L07 .17 > L14 .16;
   entropy 1.32/1.30 vs ln4=1.386; task loss .0237/.0238; same runtime).
   Success-rate equivalence unresolved at pilot power. TELEMETRY: X pins
   rms_mix at .93-1.00 per-block as designed (16k samples). H logs NO
   rms_mix/sqrt_sum_w2 (keys were added post-H) — its predicted ~1.42-1.55x
   overshoot is inferred from final weights, never measured.
4. Loudness hypothesis (first-wave finding 3) resolved at point-estimate
   level: exact-1.0 arm X tops the board while the loud arms (H, So) fell
   back at 30 eps — the 10-ep loud-arm ranking was noise. The stronger
   claim "routing penalty was a magnitude problem" was REFUTED in review:
   B<H<X is consistent with magnitude compensation but each step is
   non-significant and H-B reverses on 4/6 tasks.
5. Free routers RE-DERIVE the incumbent: both X and H converge to favoring
   L28 (w~.41) with soft mixing over the rest — the hand-picked last-layer
   tap is also the router's own preference.
6. Router preference caveat for the paper: high entropy (1.31 vs 1.386 max)
   means routers stayed near-soft; 10K steps + cosine decay may simply not
   be enough for commitment. Untested at 60K.

60K decision: if the router line continues, promote Y + X (X on
VERIFIED-MECHANISM grounds — its compensation behavior is the only one
telemetry-confirmed — NOT on success rate; X and H are indistinguishable).
Add H only if the L2-vs-exact question must stay testable at scale. Parked
full-scale runs at checkpoint-6000 predate the pcproj/renorm configs and
cannot resume into X/Y arms as-is.


## SCALE PHASE: 30K steps x 12 tasks (launched 2026-08-07)

Four arms, pilot recipe otherwise (eff batch 256, cosine over 30K, pcproj,
router lr 2e-3 / frozen for Y+A), SAVE_STEPS=5000 limit 2:
- scale30k12_X_k4_emanorm  GPU4  H-exact router K4 {7,14,21,28}
- scale30k12_Y_last_proj   GPU5  last-layer baseline
- scale30k12_A_fixed_span  GPU6  span-fixed baseline
- scale30k12_X_k28_emanorm GPU7  H-exact router K28 (all layers)

DATA: 12 tasks = 6 PnP*Close + 6 diverse Posttrain (CuttingboardToBasket,
CuttingboardToPot, PlacematToBowl, PlacematToTieredshelf, PlateToPan,
TrayToCardboardbox — covers all 4 source families, 6 receptacles). Symlink
dir /data/ruijiezhang/gr1_12task/ is the glob (launcher takes GR1_DATA_GLOB;
PILOT_MAX_STEPS/PILOT_SAVE_STEPS now override the launcher). 12k episodes
=> 30K steps ~ 1.6-2.6 epochs. The 12 UNUSED Posttrain tasks are now a
NEAR-distribution zero-shot set (novel combos of seen surfaces/receptacles).
A 6-task 30K wave was started first and killed 40 min in (user call:
12-task replaces it; wandb has the dead scale30k_* stubs).

EVAL: /data/ruijiezhang/groot_runs/scale30k12_eval60.sh armed on GPU-0 lane
— waits for checkpoint-30000, then serial 12 tasks x 60 eps per arm
(720 eps/arm, SE ~1.9pp; ~10h/arm). Status -> groot_evals/eval60.status.
ETA: training ~36h, evals drain ~40h after.
GOTCHA (2026-08-09): trainer writes checkpoint config.json FIRST; shards/
optimizer/processor land minutes later. An eval launched off config.json
existence loads a half-written ckpt and dies ("Unrecognized processing
class"). Drivers now wait for 2-min write quiescence + processor_config.


## SCALE VERDICT: 30K x 12 tasks, 60 eps (2026-08-10, campaign complete)

Board (720 eps/arm, ckpt-30000): X_k28 .311 (PnP .303/Post .319) |
X_k4 .296 (.308/.283) | A .233 (.197/.269) | Y .217 (.236/.197).

FIRST ROUTER WIN, with hedges (adversarially reviewed):
- Both routers > both baselines on pooled z (X_k28-Y +9.4pp z=4.07,
  X_k28-A z=3.32, X_k4-Y z=3.44, X_k4-A z=2.69; all survive 6-test
  Bonferroni). Task-paired tests: X_k28-Y t(11)=2.89; the weak leg is
  X_k4-A (t=2.00, p~.07, ns). Pre-registered-style contrast
  mean(routers)-mean(baselines) = +7.9pp, exact sign-flip permutation
  p=.018 (12 tasks); still .035 excluding the anomalous Can task.
- SINGLE SEED PER ARM: z covers eval sampling only (SE ~1.7pp/arm at 720
  eps); the 2-over-2 clustering weakens but does not defeat the seed
  objection (p~1/6 under arm exchangeability). Seed replicate (X_k28 + Y)
  is the missing piece for a paper claim.
- Cross-regime caveat: pilot->scale changed steps (10K->30K), tasks
  (6->12), eps (30->60) together; "task diversity is the cause" is
  HYPOTHESIS, not measurement. Degradation-vs-pilot is metric-dependent
  (absolute pp: A dropped least; relative: routers dropped least).
- Can->Drawer anomaly: BOTH baselines collapse (Y .033, A .000; Y also
  Cup->Drawer .033) while both routers score .18-.32. RECHECKS CONFIRM
  REAL (2026-08-10): fresh 60-ep re-runs reproduce it (Y30k .050,
  A30k .000) and Y's ckpt-25000 also fails (.000) — a genuine fixed-tap
  failure mode spanning checkpoints and eval runs, NOT an eval artifact
  or corrupted save. Paper gate (1) CLOSED.
- A-vs-Y: pilot's 11.7pp last-layer>span gap is GONE (+1.7pp, z=0.76).
  Do not claim "tap doesn't matter": routers say depth matters a lot
  (see below); claim only "neither fixed tap wins here".

TELEMETRY (both routers, ckpt-30000): commitment finally emerged —
K4 norm-entropy .758 (pilot .949), w_L28=.629 every block majority-
incumbent; K28 norm-entropy .853, late-band L19-28 holds 70.2% of mass,
monotone rise to L28, L25-27 shoulder (.372) outweighs L28 itself (.167);
mid-layers L13-18 are the trough. rms_mix no longer pinned at 1.0: K4
.846-.882, K28 .921-.955 (EMA renorm holds a stable band below unity;
drift scales with commitment). Task loss parity across all four arms
(.0129-.0132) — the success-rate gaps arise at ROLLOUT, not at fit.
Routers re-derive "late layers" but NOT "last layer only": the win is
consistent with spreading conditioning over adjacent late layers.

GOTCHA (2026-08-12): NVLink pairs on this box are (0,1), (2,3), (4,6),
(5,7) — NOT adjacent-numbered on NUMA node 1. DDP x2 on a non-NVLink
pair (e.g. 4,5 or 6,7) runs ~1.46x slower (3.6 vs 2.47 s/step on the
3.2B model). Always check `nvidia-smi topo -m` before pairing GPUs.

EGL PARALLEL-EVAL MYSTERY SOLVED (2026-08-12, research + live-verified):
1. Box has BOTH glvnd ICDs (10_nvidia.json + 50_mesa.json): unpinned EGL
   enumeration returns 16 devices (8 real + 8 Mesa /dev/dri dupes) and
   loads Mesa's dri2 path on NVIDIA nodes — documented 570-driver-series
   concurrent crasher. FIX: __EGL_VENDOR_LIBRARY_FILENAMES=
   /usr/share/glvnd/egl_vendor.d/10_nvidia.json (now in eval script).
2. EGL device order is PCI-bus, NOT CUDA: idx0=CUDA3, idx1=CUDA2,
   idx2=CUDA0, idx3=CUDA1, idx4=CUDA6, idx5=CUDA7, idx6=CUDA4, idx7=CUDA5.
   MUJOCO_EGL_DEVICE_ID indexes the EGL list => historic "GPU0 lane"
   actually rendered on CUDA3; the 08-12 "GPU2" eval rendered on CUDA0
   (occupied by another user's 80GB sglang) => garbage scores + ERRs.
   FIX: scripts/egl_cuda_map.py translates via EGL_CUDA_DEVICE_NV;
   eval script does this automatically now.
3. CORRECTION (same day, verified): the "EGL device 0 hazard" theory was
   WRONG. The real bug: the CUDA->EGL map lookup ran WITH
   CUDA_VISIBLE_DEVICES set, and under CVD the driver RENUMBERS
   EGL_CUDA_DEVICE_NV to process-visible ordinals => lookup fails =>
   silent fallback to the raw index. So "lane CUDA2" rendered on EGL
   idx2=CUDA0 (other user's sglang GPU: survived, slow) and "lane CUDA3"
   rendered on EGL idx3=CUDA1 (other user's 100%-util SpecForge GPU:
   deterministic SIGABRT). FIX: run egl_cuda_map.py under
   `env -u CUDA_VISIBLE_DEVICES` (eval script does this now and echoes
   the mapping: "[eval-gr1] render: CUDA X -> EGL idx Y" — CHECK IT).
   Enumeration orders: EGL = minor-number order, CUDA/nvidia-smi = PCI
   order; that is the whole scramble. n_envs=5 batch eval (AsyncVectorEnv
   spawn) works fine with the pin + correct mapping (15/15 eps, 4x
   throughput/lane); equivalence A/B vs n_envs=1 still pending.
Parallel driver: /data/ruijiezhang/groot_runs/scale60k24_eval_parallel.sh
(task-shards 24 tasks over free GPUs 2-7, per-lane shader cache, ERR
resweep, watchdog TASK_TIMEOUT kills hung client process-groups).

FINAL PARALLELISM VERDICT (2026-08-13, after many controlled attempts):
on this box/driver-state, ONLY one EGL render context total is stable.
Everything else eventually SIGABRTs in read_pixels within ~2-6 min:
n_envs=5 (5 worker contexts, even solo), 2x n_envs=1 lanes on separate
GPUs, 2 clients on one GPU. The 2-6 min horizon matches the first
episode-reset wave: robosuite defaults hard_reset=True (re-creates model
+ sim + RENDER CONTEXT every reset) => concurrent context
teardown/creation on the driver => known NVIDIA multi-context teardown
race. Short smokes "passed" only because they finished before the reset
storm (survivorship). n_envs=1 single lane never crashes.
=> Board evals run FORCE_LANES="2" LANE_N_ENVS=1 (~20h/arm, serial).
UNTESTED levers for the future (do NOT retry bare parallelism):
(a) hard_reset=False in the gym wrapper (removes context teardown
    entirely; also faster resets; likely unlocks n_envs>1);
(b) per-GPU containers (EGL namespace isolation, NVIDIA's own pattern);
(c) newer driver branch. Test during the seed-replicate wave.

## FULL-CORPUS PHASE: 60K steps x 24 tasks, DDP x2 (launched 2026-08-10)

Three arms (X_k4 dropped; routers tied, K28 kept as the stronger/general
form), 2 GPUs each via DDP: scale60k24_X_k28_emanorm (GPU 2,3) |
scale60k24_Y_last_proj (GPU 4,5) | scale60k24_A_fixed_span (GPU 6,7).
Recipe: eff batch unchanged 256 = 32 global micro (16/GPU) x 8 accum
(launcher: PILOT_NUM_GPUS/PILOT_GBS/PILOT_ACCUM added; per_device =
GLOBAL_BATCH_SIZE // NUM_GPUS in experiment.py). 60K steps keeps the
validated ~640K samples/task, ~2 epochs over 24k episodes. SAVE 10K,
limit 2. Glob: gr1_unified.* (all 24). NOTE: no held-out Posttrain set
remains — zero-shot claims need another corpus (user accepted).
EVAL: scale60k24_eval60.sh armed (24 tasks x 60 eps = 1440 eps/arm,
SE ~1.3pp; ~20h/arm serial GPU-0 lane; quiescence-guarded) ->
eval60k24.status. ETA: train ~40h (DDP~2.3s/step), evals ~60h after.


Paper-gate checklist (from review): (1) Can rechecks [running]; (2)
leave-one-task-out table [done: X_k28-Y z=3.62 without Can, X_k4-A
z=1.60]; (3) permutation contrast [done]; (4) seed replicate X_k28+Y
[NOT STARTED — needs 2 GPUs x 36h]; (5) report per-task CIs + Holm;
(6) So/Sn twins on 12-task board for a seed-noise bound [optional].


## n_envs=5 rescue campaign (2026-08-13, in progress)

CRITICAL PROTOCOL FACT (code-verified, recon agents): robosuite
hard_reset=False is INVALID for RoboCasa eval — object instances, scene
layout/style, placements, and camera randomization are ALL sampled in
Tabletop._load_model() (tabletop.py:414-538), which only runs on the
hard-reset branch (robosuite base.py:270). Soft reset re-applies the
PRE-COMPUTED placement dict (tabletop.py:1034 "use pre-computed object
placements") — the scene freezes; only robot joint jitter varies. NEVER
eval with hard_reset=False.

Surgical alternative: GL-CONTEXT SINGLETON — keep hard resets (model/sim
rebuild = full randomization) but monkeypatch robosuite EGLGLContext to a
per-process cached instance with free() neutered. The crash driver is the
per-reset EGL context destroy/create churn, not the model rebuild.

Test matrix (6 x 30-min random-action soaks, no server, GPUs 4/5 after Y
training ends; never touches board GPUs 2/3):
M1 bare async5 control | M2 bare async5+singleton | M3 container async5 |
M4 container async5+singleton | M5 bare sync5 | M6 async5 spread 2 GPUs.
Harness: /data/ruijiezhang/groot_runs/nenv5_lab/{soak.py,run_matrix.sh},
verdicts -> nenv5_lab/status.log.

MATRIX RESULTS (2026-08-13): M1 bare async5 FAIL@5.7min | M2 +GLsingleton
FAIL@7.8 | M3 container FAIL@4.8 | M4 container+singleton FAIL@8.1 |
M5 bare SYNC5 PASS(30min) | M6 spread-2GPU FAIL@6.1.
RESET-CHURN THEORY DEAD: async crashes happen with ZERO episode resets
(soak env has no step limit; no episodes ended). THREE-RULE MODEL:
R1 torch-server + EGL render on SAME gpu => crash (all post-fix bare-lane
   deaths — those lanes were the first-ever co-resident configs).
R2 spawn-sibling renderers (AsyncVectorEnv workers) => crash 2-8min,
   regardless of device spread/containers/context reuse. THE n_envs killer.
R3 SAFE: serialized rendering (SyncVectorEnv) AND cross-process renderers
   on separate GPUs with split server/render (board rendered clean through
   the entire matrix crash storm = direct coexistence proof).
=> rollout_policy.py gains GROOT_VECTOR_MODE=sync (SyncVectorEnv for
   n_envs>1). Validation on real 60-ep Bottle: STABLE, 26.5min vs ~50min
   (1.9x), BUT score 0.10 vs board's 0.30 same ckpt/task (z~2.7) — repeat
   running; if 0.1 replicates, sync batching corrupts scores => reject and
   use R3 split multi-lane (2 lanes n_envs=1) for parallelism instead.


## THE bf16 CATASTROPHE & FIX (2026-08-14) — READ BEFORE ANY MULTI-GPU RUN

ROOT CAUSE of the universal 60Kx24 rollout collapse (X_k28 .054, A .076,
Y .037 vs healthy 30Kx12 .31/.23/.22): use_ddp defaults False =>
experiment.py:263 routes ANY nproc>1 run to DeepSpeed ZeRO-2 =>
module.bfloat16() casts ALL params AND buffers to raw bf16 + bf16 grad
accumulation. Rollout collapses while loss only mildly degrades
(0.020-0.025 vs 0.013). Proof: ckpt dtype/bytes-per-param, accelerate
grad-accum warning, EMA buffer (ema_energy) stuck 2-4x off in bf16
(rms_mix 0.48-0.76 instead of ~1.0), and the empirical control: 100K-run
ckpt-30000 scored ~0.00 on tasks the 30K12 ckpt scores 0.43-0.58 (same
eval stack; stack itself validated by control re-eval).
=> ALL multi-GPU results 2026-08-10..14 are artifacts. All single-GPU
results (pilot 10Kx6, 30Kx12 wave incl. the router win) remain VALID.

FIX (commit 40cee34, validated 600-step DDPx2): --use_ddp / PILOT_USE_DDP=1
=> true torch DDP (fp32 masters + bf16 autocast, fp32 grad accum, fp32
buffers) + ddp_find_unused_parameters=True (router arms bypass vlln =>
strict DDP reducer errors otherwise). Validation: fp32 ckpt (4 B/param,
optimizer.pt), 0 deepspeed markers, rms_mix 0.98-1.01, 2.36s/it DDPx2.
RULE: never launch nproc>1 without PILOT_USE_DDP=1; verify the first log
event has rms_mix~1.0 and no DeepSpeed strings.

Relaunched scale100k24_X_k28_rlr1e4 as DDPx4 (GPUs 4-7): 1.54s/it => ~1.8
days. Broken original archived as *_bf16broken. 60Kx24 wave re-run pending
user decision. Secondary known issues (unfixed, low priority): rank-
identical (t,eps,dropout) draws under multi-GPU (rank-offset generator
would fix); EMA energy mean ignores padding.
