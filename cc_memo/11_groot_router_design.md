# Condition-router × GR00T N1.7 — integration design (repo mapped 2026-07-28, commit 9c7e746)

Local clone: `/home/ruijiezhang/Isaac-GR00T`. Full reader reports in session scratchpad (`groot_{model-core,backbone,train-config,data-eval}.md`). All anchors file:line at that commit.

## Incumbent wiring (what the router replaces)

- `Qwen3Backbone` (gr00t/model/modules/qwen3_backbone.py:135) loads Cosmos-Reason2-2B (Qwen3-VL arch) and **physically pops LLM layers above `select_layer=12`** (:194-195) — the shipped N1.7-3B checkpoint has only 12 LLM layers.
- `forward` (:357-372) runs with **`output_hidden_states=True`** and keeps only `hidden_states[-1]` (:363). The full tuple — 13 tensors (embeddings + layers 1..12), each `[B,S,2048]`, pre-final-norm — is **already materialized and discarded**. Candidate taps are FREE (and the backbone is frozen in the LIBERO recipe → no extra backward either).
- `Gr00tN1d7ActionHead.process_backbone_output` (gr00t/model/gr00t_n1d7/gr00t_n1d7.py:175-180): single `vlln = LayerNorm(2048)` (+ `nn.Identity` vl_self_attention) applied ONCE; same tensor feeds all cross-attn blocks. Inference (`_encode_features` :307) computes it once and reuses it across all 4 Euler steps.
- `AlternateVLDiT.forward` (gr00t/model/modules/dit.py:379-405): 16 blocks, inner dim 1536 (32h×48d), adaLN timestep; **even blocks (0,2,…,14) = cross-attention** to `encoder_hidden_states` (dim 2048), odd = self-attn; mask schedule alternates per cross block: idx 0,4,8,12 attend **text** tokens, idx 2,6,10,14 attend **image** tokens (`attend_text_every_n_blocks=2`). So incumbent = "layer 12 → vlln → all 8 cross blocks (with fixed text/image mask schedule)".

## v1 static router (phase 1 — near-zero cost, identity-initializable)

Per cross-attn block b (8 of them) and candidate layer k (K ≤ 13): `cond_b = Σ_k softmax(router_logits)[b,k] · norm_k(h_k)`; substitute `cond_b` as that block's `encoder_hidden_states`. Premixed once per backbone pass ⇒ **zero inference-time overhead** (still one DiT call per denoise step, same attention count).

File-level changes (6):
1. `qwen3_backbone.py:363` — return all hidden states (stack `[K,B,S,2048]` or dict) alongside the incumbent `backbone_features`.
2. `gr00t_n1d7.py` action head — new `ConditionRouter` submodule (name-prefix `condition_router`): per-candidate LayerNorms + `router_logits: Parameter[8, K]`; extend `process_backbone_output` to emit the 8 per-block tensors. **Norm decision** (open): per-layer norms (intermediate-layer statistics differ) with the incumbent layer's norm initialized from the trained `vlln` weights, vs shared vlln for all — start per-layer, ablate.
3. `dit.py` `AlternateVLDiT.forward` — accept a per-block list for `encoder_hidden_states` (keep tensor input = old behavior for compat). Keep the text/image mask schedule fixed in v1 (routing over the mask choice is a possible second axis — discuss).
4. `configs/model/gr00t_n1d7.py` — new fields: `use_condition_router: bool = False`, `router_candidate_layers: list[int]`, `router_init_bias: float = 3.0`, `router_lr: float`. Auto-serialize to checkpoint config.json; tolerant `**kwargs` init handles old checkpoints. CLI: add to `FinetuneConfig` + forward in `launch_finetune.py:78-128`, or pass tyro flags via finetune.sh `EXTRA_ARGS` passthrough (:117-121) with zero script edits.
5. `gr00t/model/gr00t_n1d7/setup.py:97-120` — **critical**: `_create_model` hard-fails on ANY missing keys except `mask_token` when loading `nvidia/GR00T-N1.7-3B`; whitelist `condition_router.*` there and do identity-init at that point (mirror the mask_token pattern): logits bias `+router_init_bias` on the incumbent layer-12 column ⇒ run starts ≈ baseline (IronVLA lesson: zero-init gates never opened; use ~87% incumbent mass, not 100%).
6. `gr00t/experiment/trainer.py:152` — override `create_optimizer` in `Gr00tTrainer`: params with prefix `condition_router` get a dedicated group at `router_lr` (start 10× base = 1e-3, the IronVLA `lr_gate` convention). No override exists today; the override covers both single-GPU and the ZeRO-2 path (zero2_config.json has no optimizer section). Also: register router under the action head + handle in `set_trainable_parameters` (:121-149; new params default trainable, governed by no existing flag — add `tune_router`).

Diagnostics: return per-block softmax W + entropy in the action-head forward dict (:280-286) and log via the existing rank-0 `logging_steps` gate in `compute_loss` (trainer.py:289-306); wandb is on by default (project `finetune-gr00t-n1d7`). TB names from IronVLA: `Router/w_mean_*`, `/entropy`, `/W` heatmap.

## v2 token router (phase 2 — the contribution's depth)

Replace even-block `attn1` (diffusers `Attention`, dit.py:154-163) with K expert cross-attentions + per-token `Linear(1536→K)` router on the post-AdaLN x̂ (no extra norm — preserves timestep scale/shift), `top_k` knob. Runs in all 4 denoise steps ⇒ real extra compute (~K× cross-attn on 41 query tokens — still cheap: queries are only [B,41,1536]). Port of IronVLA `forward_token_routed` / `install_token_router`.

## The official LIBERO recipe (baseline to reproduce first)

- Base ckpt `nvidia/GR00T-N1.7-3B`; backbone `nvidia/Cosmos-Reason2-2B` is a **gated HF repo — needs HF auth**.
- Data: `IPEC-COMMUNITY/libero_{10,goal,object,spatial}_no_noops_1.0.0_lerobot` + copy `examples/LIBERO/modality.json` into each `meta/` (+ libero_goal episode-82 mp4 patch). State 8-d, action 7-d, cams `image`+`wrist_image`, 16-step chunks padded to horizon 40 (action_mask marks first 16), eval executes 8.
- Fine-tune (per suite): `NUM_GPUS=8 MAX_STEPS=20000 GLOBAL_BATCH_SIZE=640 SAVE_STEPS=1000 uv run bash examples/finetune.sh --base-model-path nvidia/GR00T-N1.7-3B --dataset-path <suite dir> --embodiment-tag LIBERO_PANDA --output-dir ... --state-dropout-prob 0.2`. Effective: torchrun + DeepSpeed ZeRO-2, adamw_torch lr 1e-4 cosine, warmup 5%, wd 1e-5, per-device batch 80, bf16 mixed (weights fp32; `load_bf16` forced off), grad ckpt OFF, VLM frozen (`tune_llm/visual=False`), trainable = projector + DiT + vlln (~0.6B).
- Eval: server `gr00t/eval/run_gr00t_server.py --model-path <ckpt> --embodiment-tag LIBERO_PANDA --use-sim-policy-wrapper` + client `rollout_policy.py` from a separate LIBERO venv (`gr00t/eval/sim/LIBERO/setup_libero.sh`), 10 episodes × 5 envs per task, max 720 steps. Official numbers: S 97.65 / O 98.45 / G 97.5 / L 94.35 = **97.0 avg**.
- Stack: Python 3.12, uv, torch 2.9.0+cu128, transformers 4.57.3, diffusers 0.35.1, flash-attn 2.8.3, deepspeed 0.17.6. Apache-2.0 (repo README states N1.7 fully commercial-licensable — good for release).
- Wall time: not published; estimate 0.5–1.5 days per 20k-step suite run on 8×H100 — **do a 100-step timing run first**.

## Experiment sequence

1. Env + data setup; reproduce baseline on one suite (libero_10 = hardest/Long) → match ~94.35.
2. Identity-init router, short run + eval → confirm no-regression (the "free lunch" anchor).
3. Full v1 router runs, all 4 suites, same budget; report Δ + learned-W analysis (which depths per block; text-blocks vs image-blocks preferences).
4. Ablations: K pool (all 13 vs subset), per-layer vs shared norm, router_lr, init bias; then v2 token router.
5. Beyond plain LIBERO (saturation!): LIBERO-Long/90 emphasis, LIBERO-Plus/PRO perturbations; optionally RoboTwin via existing FastWAM infra.

## Open design questions (for discussion)

1. Candidate pool: all 13 free taps vs a spread subset; add ViT-level features (would need vision-side taps — not free like LLM layers) — probably later.
2. Per-layer norms vs shared vlln (statistics differ; identity-init interacts).
3. Route only over layers (keep text/image mask schedule) vs also over the mask/modality axis.
4. Per-suite finetuning (official recipe) means 4 routers learned independently — a cross-suite comparison of learned W is itself an interesting figure.
5. `select_layer` interaction: raising it (>12) would un-truncate the LLM — extra compute + weights not in the shipped ckpt; keep 12 for the main result.
6. DeepSpeed resume with an added optimizer group mismatches pre-router checkpoints — plan fresh 20k runs.
