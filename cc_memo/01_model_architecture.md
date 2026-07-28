# Model architecture

## Mixture-of-Transformers (MoT) core

Two "experts" run layer-locked in parallel for 30 layers (`src/fastwam/models/wan22/mot.py`, class `MoT`):

- **Video expert** — `WanVideoDiT` (`wan_video_dit.py`): hidden 3072, ffn 14336, 24 heads × 128 head-dim, 30 layers, patch [1,2,2], in/out dim 48, loaded from Wan2.2-TI2V-5B pretrained.
- **Action expert** — `ActionDiT` (`action_dit.py`): hidden 1024, ffn 4096, same 24×128 heads / 30 layers, initialized from the Wan DiT by **linear interpolation** (see below).

Per layer: each expert computes Q/K/V from its own tokens (own RoPE — video 3D, action 1D), Q/K/V are concatenated along the sequence dim into one joint attention in the shared space `24×128 = 3072` under a 2D bool mask, output is split back, and each expert applies its own cross-attn (to its own text context) + FFN. Hidden dims may differ across experts because mixing happens only in attention space. `MoT.expert_order` comes from dict insertion order — `{"video", "action"}` (video first), which the `[video|action]` mask layout silently assumes.

Key MoT methods: `forward` (joint pass), `prefill_video_cache` (video-expert-only pass, returns per-layer post-RoPE K/V), `forward_action_with_video_cache` (action Q/K/V + cached video K/V, mask row-sliced `[Sv:, :Sv+Sa]`). Mixed attention is checkpointed during training when `mot_checkpoint_mixed_attn` (task configs set it **false**).

## The three variants

Class hierarchy `FastWAM` → `FastWAMJoint(FastWAM)` → `FastWAMIDM(FastWAMJoint)`. The three `configs/model/*.yaml` are **byte-identical except `_target_`** — the variant lives entirely in the factory (`fastwam.runtime.create_fastwam{,_joint,_idm}`) and class code.

| Variant ("task" name) | Class / file | Action attends to | `infer_action` cost |
|---|---|---|---|
| **uncond** = paper's Fast-WAM | `FastWAM`, `fastwam.py` | first-frame video tokens only | 1 video prefill (KV cache) + N action-only steps — **no imagination** |
| joint | `FastWAMJoint`, `fastwam_joint.py` | all (noisy) video tokens | full video+action co-denoising, returns action only |
| idm | `FastWAMIDM`, `fastwam_idm.py` | a separate clean/teacher-forced cond-video sequence | Stage 1: denoise video alone (plain `video_expert.forward`, no MoT); Stage 2: prefill cache on imagined video, denoise action |

Attention-mask semantics (rows=queries; video→video always internal `first_frame_causal`; **video never attends to action**):
- uncond: action → {action, first `tokens_per_frame` video tokens}.
- joint: action → {action, all video}.
- idm training: sequence `[noisy_video | cond_video | action]`; action → {action, cond_video}; the two video copies don't see each other; only the noisy half is supervised. Cond video is re-noised per-sample with hardcoded `video_cond_noise_prob = 0.5` (class attr, not config). IDM doubles the video-token count in one MoT pass (~2× attention cost).

`infer_action` signatures diverge: base takes no `num_video_frames`; Joint/IDM require it. Call sites (eval code) check `inspect.signature`.

## Training loss (shared by all variants)

`training_loss` in `fastwam.py`: independent flow-matching timesteps for video and action; both noised; `loss = lambda_video * MSE_v(video) + lambda_action * MSE_v(action)` where the target is **velocity `noise − x0`** (rectified flow — despite method names like `_predict_joint_noise`). Per-branch timestep weights from a Gaussian-bump `training_weight`. Masking via `image_is_pad` / `action_is_pad`. Frame-0 latents are clamped to clean GT and excluded from video loss.

Scheduler (`schedulers/scheduler_continuous.py`, `WanContinuousFlowMatchScheduler`): shift-warp `phi(u,s)=s·u/(1+(s−1)·u)`, shift 5.0 both branches, 1000 train timesteps; `x_t=(1−σ)x0+σε`; inference = Euler `x ← x + v·Δσ` over σ:1→0 (Δσ negative), default 20 steps in `infer_joint`, eval uses 10.

**No CFG is implemented anywhere** — `negative_prompt` / `text_cfg_scale` / `action_cfg_scale` params are accepted and ignored in the FastWAM classes (only `Wan22Core.infer` in `wan22.py`, the plain video model, actually implements dual additive CFG).

## Model input contract (`FastWAM.build_inputs`)

Required batch keys: `video` [B,3,T,H,W] (H,W % 16 == 0 — effectively % 32 due to DiT patch 2; T % 4 == 1), `context` [B,L,4096] + `context_mask` [B,L] (**precomputed T5 — training never encodes text**; `load_text_encoder: false`), `action` [B,T_a,a_dim] with `T_a % (T−1) == 0`. Optional: `proprio` [B,T,p_dim] (required if `proprio_dim` set — becomes ONE extra context token via `nn.Linear(proprio_dim, 4096)`, only `proprio[:,0,:]` used), `action_is_pad`, `image_is_pad`.

Hard requirements baked in: `seperated_timestep=true` (sic, misspelled in config — keep the spelling) + `fuse_vae_embedding_in_latents=true` (any other combo → NotImplementedError in `WanVideoDiT.pre_dit`). Per-token timesteps with frame 0 forced to t=0. `FastWAM.infer_action` asserts `video_attention_mask_mode == 'first_frame_causal'` — the KV cache is only valid because first-frame tokens can't attend to later noisy frames (`build_video_to_video_mask`).

`model.dit` is an **alias of `model.mot`** (fastwam.py ~:47) so trainer freeze/optimizer code addressing `.dit` trains the whole MoT. Checkpoints: `save_checkpoint` writes `{"mot": state_dict, "step", "torch_dtype"[, "proprio_encoder"]}`; `load_checkpoint` accepts `"mot"` (strict=False) or legacy `"dit"` (loads into video expert only).

## ActionDiT initialization (scripts/preprocess_action_dit_backbone.py)

Identity layer mapping (30↔30). For every backbone key (all except `action_encoder.*`, `head.*`, which stay random) the same-named Wan DiT tensor is taken; shape-mismatched tensors (3072→1024, 14336→4096) are resized per-dimension by **1D linear interpolation, align_corners=True**, with `sqrt(d_src/d_tgt)` alpha scaling whenever the last dim of a ≥2-D tensor changes (`--apply-alpha-scaling` default true; 1-D tensors like biases are interpolated but never alpha-scaled). Payload `{"policy", "backbone_state_dict", "meta"}`; `ActionDiT.from_pretrained` validates `meta` exactly against config (hidden/ffn/layers/heads/head_dim/text_dim/freq_dim/eps) and key-set/shapes exactly. Relative `action_dit_pretrained_path` resolves against the **repo root**, not cwd. ActionDiT 1D RoPE cache caps action length at 1024 tokens. The `ActionHead` class in `action_dit.py` is dead code — the real head is a bare `nn.Linear` with no time modulation. Rerun the script when ActionDiT dims, source model, or scaling policy change.

## Wan2.2 backbone loading (`helpers/loader.py`, `helpers/io.py`)

- Components: DiT (`WanVideoDiT`), VAE (`WanVideoVAE38`), optional umT5-xxl text encoder (`WanTextEncoder`) + HF tokenizer (from the **Wan2.1**-T2V-1.3B repo, `google/umt5-xxl/` subdir).
- Checkpoint identification is **hash-based**: MD5 over sorted `key:shape` strings (not weights, not filenames) matched against `WAN22_MODEL_REGISTRY` (T5 `9c8818c2…`, DiT `1f5ab770…`, VAE `e1de6c02…`). Wrong key structure → "Cannot detect model type". All `load_state_dict` calls are `strict=False` — misconfigurations silently leave modules random.
- Expected on-disk layout under `$DIFFSYNTH_MODEL_BASE_PATH` (default `./checkpoints/`), with default `redirect_common_files=true`:
  - `Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model*.safetensors` (DiT, sharded ok)
  - `DiffSynth-Studio/Wan-Series-Converted-Safetensors/{models_t5_umt5-xxl-enc-bf16.safetensors, Wan2.2_VAE.safetensors}`
  - `Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/` (tokenizer)
- Auto-download from **modelscope by default** (`DIFFSYNTH_DOWNLOAD_SOURCE=huggingface` to switch, `DIFFSYNTH_SKIP_DOWNLOAD=true` for offline).
- Loader extras not exposed by `Wan22Core.from_wan22_pretrained` but used by the fastwam factories: `skip_dit_load_from_pretrain` (random-init DiT; sim-eval configs set true since the trained ckpt is loaded anyway) and `load_text_encoder` (false for training, true for sim eval).

## VAE (`WanVideoVAE38`)

z_dim 48, spatial ×16, temporal ×4 (Wan2.1's 16/×8 classes also present but unused). Latent for [3,T,H,W]: `[48, (T−1)/4+1, H/16, W/16]`. Encode is deterministic (mu only, normalized with hardcoded 48-dim mean/std); VAE frozen at construction. `encode(tiled=True)` raises NotImplementedError; decode stages latents on CPU (slow, RAM-heavy) and supports tiling. Input is spatially patchified 2×2 (12-ch conv1); T must satisfy T % 4 == 1.

## Misc facts

- `flash_attention` in `wan_video_dit.py` is actually `F.scaled_dot_product_attention` (compatibility mode only).
- `WanVideoDiT` ctor hard-asserts `has_image_input=False`, `require_clip_embedding=False`, `require_vae_embedding=False`+`fuse_vae_embedding_in_latents=True`.
- `action_conditioned` video-DiT path exists (GT actions appended to cross-attn context with group-causal mask, `action_group_causal_mask_mode: group_diagonal`) but is **false in all shipped configs**.
- `infer_joint` (base) optionally self-checks `infer_action` equivalence with the same seed (`test_action_with_infer_action=True`, warns if allclose(1e-2) fails; requires `seed`).
- Timestep broadcast [1]→[B] allowed only in eval; training asserts per-sample timesteps.
- `encode_prompt` zeroes padded T5 positions then sets mask to all-ones (Wan2.2 convention; dataset does the same to cached embeds).
