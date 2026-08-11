# Handoff: everything since 2a6cd76 (for the remote agent)

Written 2026-08-11 for a Claude agent on another machine. Anchor commit:
`2a6cd76` (per-candidate projection adapters) — the state at the E/F
handoff. 25 commits later, HEAD is `0a301c4`. The requested anchor
`64fd874` does not exist in origin/main (probably a local commit on your
machine); everything below covers 2a6cd76..0a301c4.

Read `13_local_training_and_profile.md` for the full chronological log.
This file is the condensed mission brief.

## 1. Project in one paragraph

Condition-router research on GR00T N1.7 "Qwen28" arch: each DiT
cross-attention block may read a learned softmax mixture over VLM layer
outputs instead of the stock "layer 28 feeds everything". Architecture:
Qwen3-VL-2B backbone (28L, hidden 2048, fully tuned, no pretrained N1.7
weights) + from-scratch 28L DiT with 16 cross-attn blocks, flow matching,
~3.2B params, fits one 94GB H100 at micro-batch 16. Question: does routing
beat the last-layer incumbent? Answer so far: NO at 10K steps/6 tasks;
YES (significant) at 30K steps/12 tasks; 60K/24-task run in flight.

## 2. Code added since 2a6cd76 (all in Isaac-GR00T/gr00t/)

Router features (`model/gr00t_n1d7/gr00t_n1d7.py`, ConditionRouter):
- `gate_mode="sigmoid"` accumulation gates + `gate_init_hi/lo` (logit init)
- `token_query`: per-token routing via zero-init probes v_b [N,dim];
  scores = logits + (v_b · normed_candidate)/sqrt(D), premixed
- `mix_renorm` + `mix_renorm_mode`: "l2" = 1/sqrt(sum w^2) predicted
  rescale; "ema" = measured-energy compensation ("H-exact": EMA of actual
  mixture RMS, momentum 0.99, train-only updates, sentinel-seeded first
  batch) — pins delivered mixture RMS at single-candidate scale
- telemetry stashes: RouterLLM/w_mean_L*, entropy, sqrt_sum_w2, rms_mix
  (per-block), gate_sum/min, alpha_token_std; logged every 10 steps
  (16 dicts per event = one per cross block)

Trainer (`experiment/trainer.py`):
- RouterFreezeDelayCallback (grad-drop before optimizer step)
- optimizer groups: router logits at ROUTER_LR ("condition_router" in name
  AND "projs." not in name); projectors/norms at base LR
- entropy-anneal bonus (loss -= coef(t)*H) with separate logging: task
  loss logged as "loss", bonus split out (comparable across arms)

Launcher (`scripts/launch_pilot_qwen28.sh`) — arms A-H, S, T, X, Y, Z and
env overrides: PILOT_NAME, PILOT_GPU (accepts "2,3"), PILOT_PORT,
PILOT_ROUTER_LR, PILOT_LAYERS, PILOT_EXTRA, PILOT_SUFFIX, ROUTER_PCPROJ,
PILOT_MAX_STEPS, PILOT_SAVE_STEPS, PILOT_NUM_GPUS, PILOT_GBS, PILOT_ACCUM,
GR1_DATA_GLOB. per_device batch = GLOBAL_BATCH_SIZE // NUM_GPUS
(experiment.py); DDP x2 keeps eff batch 256 via GBS=32, ACCUM=8.

Eval infra (`scripts/groot_fuyao_eval_gr1.sh`, `eval/rollout_policy.py`):
- `--video-dir none` / NO_VIDEO=1 (video recording kills sim workers)
- MUJOCO_EGL_DEVICE_ID pinning (CUDA_VISIBLE_DEVICES does NOT restrict
  EGL enumeration!), EGL creation flock, MAX_ATTEMPTS retries,
  `nice -n 15` + OMP/MKL_NUM_THREADS=4 on the server (CPU starvation)

## 3. The arms (naming convention: run dirs / wandb names)

| Arm | What | Router |
|-----|------|--------|
| Y | last-layer + per-tap proj (INCUMBENT baseline) | K=1 {28}, frozen |
| A | span-fixed + per-tap projs (baseline 2) | hard one-hot {7,14,21,28}, frozen (init bias 16) |
| B/D | learnable softmax K4/K28, uncompensated | LR 2e-3, uniform init |
| G | B + entropy anneal 0.02 | — |
| H | B + L2 mix-renorm | — |
| X | B + EMA measured-energy renorm ("H-exact") | — |
| X_k28 | X with all-28 pool | — |
| S* family | sigmoid gates (ident/open x free/renorm, gate-LR 5e-3 pair) | — |
| T | per-token routing K4/K28 | — |
| Z | true stock path, no router module (abandoned) | — |

ALL arms (incl. baselines) carry per-candidate LayerNorm + identity-init
Linear(2048,2048) projectors (ROUTER_PCPROJ=1) for parameter parity.

## 4. Hyperparameters (all phases)

Shared: eff batch 256 | DiT/proj LR 1e-4 | backbone LR 1e-5 | router
logit LR 2e-3 (20x; frozen in Y/A) | cosine to ~0, single cycle | bf16 |
state dropout 0.2 | `--skip-weight-loading --tune-llm --tune-visual
--backbone-lr 1e-5 --select-layer 28 --backbone-embedding-dim 2048
--dit-num-layers 28` | embodiment ROBOCASA_GR1_TABLETOP.

| Phase | Steps | Tasks | GPUs/arm | Micro/accum | Save | Wall |
|-------|-------|-------|----------|-------------|------|------|
| Pilot | 10K | 6 | 1 | 16 x 16 | 2500, keep 2 | ~12h |
| Scale | 30K | 12 | 1 | 16 x 16 | 5000, keep 2 | ~37h |
| Full (RUNNING) | 60K | 24 | 2 (DDP) | 16/GPU x 8 | 10000, keep 2 | ~40h |

Design rule discovered: hold ~640K training samples per task (~2 epochs).
That is the regime where routers commit and the win appears.

Data: gr1_unified corpus, 24 tasks x 1000 eps (6 PnP*Close + 18
Posttrain PnPNovel*SplitA). 12-task subset = 6 PnP + diverse-6 Posttrain
(CuttingboardToBasket/Pot, PlacematToBowl/Tieredshelf, PlateToPan,
TrayToCardboardbox) — built as a symlink dir consumed via GR1_DATA_GLOB.

## 5. Results

### Phase 1 — pilot 10K x 6 tasks, 30-ep re-evals (180 eps/arm)

X .483 | Y .467 | H .439 | Sg4 .406 | Sg28 .400 | B .394 | So .378 |
A .350 | T_k28 .317 | T_k4 .306. (10-ep only: G .400, D .383, Si .383,
Sr .333, Sn .300; external E* .467, F* .383.)

Verdict (adversarially reviewed): NOTHING beats the incumbent; all
top-7 arms one statistical cluster (diff SE ~5.2pp). Only
Bonferroni-surviving result: per-token routing HARMFUL (z~2.9-3.1).
X=H behaviorally (identical learned weights/loss; telemetry-confirmed
rms_mix~1.0 for X only). 60-ep noise floor ~15pp (So/Sn twin lesson) —
never trust 60-episode comparisons.

### Phase 2 — scale 30K x 12 tasks, 60 eps/task (720 eps/arm) — ROUTER WIN

| Arm | 12-task | 6-PnP | 6-Posttrain |
|-----|---------|-------|-------------|
| X_k28 | .311 | .303 | .319 |
| X_k4 | .296 | .308 | .283 |
| A | .233 | .197 | .269 |
| Y | .217 | .236 | .197 |

Stats: X_k28-Y +9.4pp z=4.07; X_k4-Y +7.9 z=3.44; X_k28-A +7.8 z=3.32;
X_k4-A +6.3 z=2.69 (all Bonferroni-safe over 6 tests); routers tied
(z=.63), baselines tied (z=.76). PRIMARY pre-registered-style contrast
mean(routers)-mean(baselines)=+7.9pp, exact sign-flip permutation
p=.018 (p=.035 excluding Can task). Weak leg: X_k4-vs-A paired t p~.07.
Baseline task COLLAPSE verified real: Y and A both ~0 on Can->Drawer
(rechecked fresh 60 eps + Y ckpt-25K: .050/.000/.000), Y also .03 on
Cup->Drawer; routers score .17-.32 there.

Telemetry at 30K: routers finally COMMIT (pilot did not). X_k4: w(L28)
=.629, norm-entropy .758 (pilot .949). X_k28: 70.2% mass on L19-28,
monotone rise, L25-27 shoulder (.372) outweighs L28 (.167), norm-entropy
.853. rms_mix bands: X_k4 .846-.882, X_k28 .921-.955. Train loss parity
across ALL arms (.0129-.0132) => gaps arise at ROLLOUT, not fit.

Mandatory hedges: ONE SEED PER ARM (z covers eval sampling only; 2-over-2
clustering is p~1/6 under exchangeability); pilot->scale changed
steps+tasks+eps together, so "task diversity is the cause" is hypothesis.

### Phase 3 — full corpus (IN FLIGHT on this machine)

scale60k24_{X_k28_emanorm,Y_last_proj,A_fixed_span}, 60K x 24 tasks,
DDP x2, launched 2026-08-10. Eval: 24 tasks x 60 eps (1440/arm, SE
~1.3pp) auto-runs at ckpt-60000. NOTE: training all 24 spends the
held-out set — zero-shot claims need another corpus.

## 6. Open paper gates

1. Can rechecks — DONE (collapse real).
2. Leave-one-task-out — DONE (headline survives w/o Can; X_k4-A z=1.60).
3. Permutation contrast — DONE (p=.018).
4. SEED REPLICATE (X_k28 + Y retrain, different seed) — NOT STARTED;
   the single missing gate for the paper claim.
5. Report per-task CIs + Holm-corrected p in the paper.
6. Optional: So/Sn pilot twins evaluated on the 12-task board for a
   seed-noise bound.

## 7. Operational gotchas (will bite you)

- Checkpoint write race: trainer writes config.json FIRST; shards/
  optimizer/processor land minutes later. Wait for 2-min quiescence +
  processor_config.json before eval (drivers do this now).
- CUDA_VISIBLE_DEVICES does not restrict EGL; pin MUJOCO_EGL_DEVICE_ID.
  On the primary box only ONE render lane is stable (host-global EGL
  race); your machine may differ — test before parallelizing.
- N_ENVS=1 always (AsyncVectorEnv SIGABRTs in robosuite read_pixels);
  NO_VIDEO=1; PYTHONFAULTHANDLER=1 exposes silent SIGABRTs.
- torchcodec needs ffmpeg<8 libs on LD_LIBRARY_PATH; conda-venv pythons
  may also need LD_PRELOAD of a modern libstdc++.so.6 (launcher handles
  both if FFMPEG_LIB is set).
- 60-ep evals have ~15pp noise; decision-grade comparisons need >=30 eps
  x many tasks, ideally the permutation contrast across tasks.
- Trainer status lines append "PILOT <letter> EXITED: <code>" to
  RUNS_ROOT/pilot.status; per-arm logs at RUNS_ROOT/<name>.log.

## 8. Machine-local paths (primary box; adapt on yours)

venv /data/ruijiezhang/env/groot | ffmpeg7 /data/ruijiezhang/env/ffmpeg7 |
HF cache /data/ruijiezhang/hf/hf_cache | data /data/ruijiezhang/gr1_unified
(+ gr1_12task symlink dir) | runs /data/ruijiezhang/groot_runs | evals
/data/ruijiezhang/groot_evals (results.csv per arm; final_table.csv;
eval60.status / eval60k24.status). Launch example (12-task X_k28):

    env PILOT_NAME=scale30k12_X_k28_emanorm PILOT_GPU=7 PILOT_PORT=29554 \
      PILOT_MAX_STEPS=30000 PILOT_SAVE_STEPS=5000 ROUTER_PCPROJ=1 \
      PILOT_LAYERS="$(seq -s' ' 1 28)" \
      GR1_DATA_GLOB='<data>/gr1_12task/gr1_unified.*' \
      bash scripts/launch_pilot_qwen28.sh X
