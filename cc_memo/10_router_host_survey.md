# Condition-router host survey — top VLM+action-DiT frameworks on LIBERO (web survey 2026-07-28)

Goal: pick a well-known VLM + action-DiT framework to host Ruijie's **condition router** (from IronVLA: v1 static = learned per-DiT-block softmax over K candidate VLM layers, identity-initializable; v2 token = per-token `Linear(D→K)` router inside each DiT block mixing K expert attentions; router params in a dedicated high-LR `lr_gate` group). Four-agent web sweep; numbers cross-checked against primary sources.

## LIBERO landscape (mid-2026)

- **Saturated**: >95% 4-suite avg is table stakes; SOTA band 97–99.9. Absolute #1 is LaST-R1 99.9 (AR + RL post-training, not a flow expert). RL post-training (πRL 98.3 on π0.5, SnapFlow 98.75) is what pushes 97→99.
- Best *verified flow-expert* numbers: GeoAlign 99.0 (GR00T-N1.6-based, research code), GEAR-VLA 98.7, X-VLA 98.1, VLANeXt 97.4 (Qwen3-VL-2B + 12-layer flow head, cross-attn+adaLN).
- Credibility advice from the field (ICLR-2026 VLA meta-review): match the host's official reproduction first; show deltas on LIBERO-Long / LIBERO-90; pair with LIBERO-Plus/PRO (perturbation robustness) or another benchmark — 99-vs-98 on the plain suites persuades no one.
- Every host hardcodes a different answer to "which VLM layer feeds which DiT block" — **that's the router's unifying pitch**: GR00T = one middle layer for all blocks; π0 = layer-locked i→i joint attention; FLOWER = pruned half + ONE global AdaLN vector; SmolVLA = first L/2 only; CogACT = a single final token.

## Top-3 hosts (recommendation order)

### 1. GR00T N1.7 — NVIDIA Isaac-GR00T (recommended)
- LIBERO **97.0 official** (S 97.65 / O 98.45 / G 97.5 / L 94.35, 20k steps bs640); first-party LIBERO recipe + released `nvidia/GR00T-N1.7-LIBERO` ckpt; LeRobot-format data (IPEC-COMMUNITY). 7.7k stars; PyTorch; code Apache-2.0, **weights NVIDIA Open Model License** (check release implications).
- Architecture: VLM (N1.7: Cosmos-Reason2-2B, **Qwen3-VL architecture** — same family as IronVLA, extract_layers code transfers) exports features from a single `select_layer` (middle); 16-layer flow-matching DiT with alternating cross-attn(→VLM tokens)/self-attn + adaLN timestep; 4 Euler steps.
- Router fit: **surgical**. Add taps at K VLM layers (+ per-layer projectors), replace the hardcoded single tap: v1 = 16×K softmax premixing per DiT block (identity-init to the incumbent `select_layer` ⇒ no-regression start); v2 = K-expert cross-attention per block. The "one middle layer for everything" incumbent is exactly the assumption the router attacks.
- Risk: fast-moving repo (backbone swapped N1.6→N1.7) — pin a version.

### 2. π0 / π0.5 — Physical Intelligence openpi (most famous)
- LIBERO: π0.5 **96.85 official** (98.8/98.2/98.0/92.4); LeRobot PyTorch port reproduces **97.5** (97/99/98/96, 6k steps bs256 on 8×H100). 13k stars — the baseline in essentially every VLA paper. JAX-first; PyTorch path validated on LIBERO but lacks FSDP/LoRA/EMA.
- Architecture: PaliGemma-3B + ~300M expert weights; **layer-locked joint blockwise attention** — expert layer i attends VLM prefix KV of layer i inside one fused attention op. No cross-attention module.
- Router fit: deeper story ("relax the layer-locked alignment" — expert layer i attends a learned mixture over VLM layers' KV), but **invasive**: means editing the fused attention, and KV mixing across layers is the expensive variant (mixture of attentions over K per-layer KVs). Highest fame, highest engineering risk.

### 3. FLOWER — KIT, CoRL 2025 (cleanest story, cheapest compute)
- LIBERO **96.9** (97.5/99.1/96.1/**94.9 Long** — best sub-1B Long score; LIBERO-90 94.7). 947M total; LIBERO+CALVIN finetune code; academic-scale repo.
- Architecture: Florence-2-Large, decoder dropped + ~50% of LM layers pruned, intermediate-fusion features → rectified-flow transformer conditioned by **one Global AdaLN-Zero vector shared across all layers**.
- Router fit: the single global bottleneck is the perfect foil — "from one hardcoded condition to learned per-block routing over K fusion depths"; the paper already ablates fusion depth manually, the router learns it. Small model = many seeds/ablations on 1–4 GPUs.

## Honorable mentions
- **SmolVLA** (LeRobot-native, interleaved cross-attn = easiest surgery, 450M) — but 87.3 baseline is weak by 2026 standards; best as a cheap prototyping vehicle before the real host.
- **StarVLA** (3.3k stars, MIT): purpose-built "Lego" VLA research host (pluggable Qwen/InternVL/Florence backbones × FAST/OFT/π-flow/GR00T heads, LIBERO supported) — worth a day of due diligence; baselines less recognized.
- **X-VLA** (98.1, ICLR 2026, LeRobot-integrated): strong but unified single-stream (soft-prompt transformer) — no separate conditioning interface to route.
- **CogACT**: single cognition token (opposite extreme, conceptually clean) but repo stale since 2024, no LIBERO harness.
- Not-matching shapes appearing in tables: OpenVLA-OFT 97.1 (regression), LaST-R1 99.9 / Dream-VLA / MMaDA (discrete-diffusion-in-VLM), GEAR-VLA 98.7 (latent-action bridge).

## Evaluation-strategy notes for the router paper
1. Reproduce the host's official LIBERO number first (GR00T: 97.0 @20k steps) — the credibility anchor.
2. Primary claims: LIBERO-Long + LIBERO-90 deltas, robustness under LIBERO-Plus/PRO perturbations, and **interpretability of the learned routing W** (which depths matter per block/token — the color-grounding/attention-ROI analysis style from the IronVLA investigation transfers).
3. v1 static router is near-zero-cost + identity-init ⇒ "free lunch" framing; v2 token router is the contribution's depth.
4. Ruijie also has RoboTwin infra (FastWAM side) — a second benchmark for generality if needed.

Full per-agent reports archived in session scratchpad (`survey_{leaderboard,flagships,recent,usability}.md`).
