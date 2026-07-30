# IronVLA on LIBERO — interface facts & port plan (models/ read 2026-07-28)

Source: `former/models` = the `models/` subtree of Ruijie's IronVLA (iron_vla/xprobot) repo. Qwen3-VL backbone + task heads. Verified against code; file:line anchors relative to `former/models/`.

## Build / load / forward contract (what an eval adapter calls)

- Build: `from models.build_model import build_model; policy = build_model(policy_config)` — registry names `IRON_VLA`/`HEAD_ONLY` both → `IronVLA` (build_model.py:5-8). Whole model bf16; backbone built via `Qwen3_VL_Wrapper.from_pretrained(model_id, config=hf_config, ...)` (backbone/registry.py:16-21).
- Load ckpt: `policy.deserialize(state_dict, strict=True)` (policy/iron_vla.py:504) — auto-detects legacy flat layout (`backbone.*`, `proprio_input_encoder.*`) vs nested `vlm.*`; `serialize()` writes the flat layout. On-disk wrapper (raw vs dict with 'model'/norm keys) is train.py's choice — check the actual file.
- Inference: `out = policy.forward(compute_loss=False, **batch)`; result keyed by `head_cfg.task` → dict `{'type':'trajectory','pred': [B, action_horizon, action_dim], 'gt', 'mask'}`. **pred is in normalized model space** — normalization lives entirely dataset-side (`IronVLA.norm_stats = None`, iron_vla.py:177); adapter must denormalize.

## Batch keys for one action-only eval step (FlowMatchingAction head)

| Key | Shape / notes |
|---|---|
| `input_ids`, `attention_mask` | [B,S]; standard Qwen3-VL AutoProcessor output; prompt must contain the image placeholders AND `<|PROPRIO|>` token(s) |
| `pixel_values`, `image_grid_thw` | flat-patch tensor + [num_images,3], straight from AutoProcessor (2 images per sample for 2 cams) |
| `state` | **must be 3-D** `[B, N_proprio_tokens, 8]` (bmm in CategorySpecificLinear, common/moe.py:14-17); N must equal the number of `<|PROPRIO|>` placeholders per row |
| `embodiment_id` | LongTensor [B], < num_categories (50 bank); LIBERO gets its own id |
| `state_mask` | optional; absent ⇒ every row must carry exactly N placeholders |
| `action`, `action_mask` | **required even at inference** by FlowMatchingAction (read for gt/mask fields, flow_matching_action.py:812-814) — pass zeros/ones dummies. (Gr00tN1d7 head is lenient.) |

PROPRIO mechanics: located by token-id equality and `masked_scatter`ed with the proprio-encoder output (embedding_utils.py:15-134); the id comes at runtime from `dataset.util.registered_tokens['PROPRIO']` (iron_vla.py:60-65) — **raises unless `update_processor_cache()` was called first**. Padding side: models/ is agnostic for the single-pass action path; must match the training-side transform anyway.

## Key config facts

- PolicyConfig fields consumed: `policy_class`, `use_vlm_backbone`, `hidden_dim`, `num_categories`, `state_dim`, `backbone.{hf_config,input_proprio,model_id,llm_lora}`, `heads.{action_predict,next_token_predict,value_predict,bbox_detr_predict}`, optional `stm`. hf_config is a real HF Qwen3VLConfig carrying extra attrs: `type='Qwen3_VL_Wrapper'`, `model_id`, `attn_implementation`, `stage` ('vla-full-train'|'finetune-frozen-llm'|'freeze-backbone'), `use_causal_mask`, `enable_gradient_checkpointing`, `vision_attention_dropout`.
- FlowMatchingHeadConfig essentials: `action_horizon`, `action_dim`, `state_dim`, `num_categories`, `hidden_dim` (DiT width, 1024), `num_layers` (sequence; [0]=DiT depth), `num_inference_timesteps` (8), `dit_block_type`/`vlm_feature_format`/`vlm_fusion_mode` (active: cross_only/hidden/context), `multi_layer_vlm` + `cross_layer_mapping` ([dit_start, dit_end, vlm_layer], −1=last — this also drives which VLM layers get extracted, via collect_required_vlm_layers), `noise_*`, `loss_type`. Valid combos enforced by utils/config_validation.py (`ALLOW_INVALID_ARCHITECTURE=1` bypass).
- LoRA (vit_condition/dual_stream) nests Linears under `.base` → changes ckpt keys; installed at `IronVLA.__init__` before deserialize, so eval config must match training config exactly. `feature_extraction_norm` is weightless (multi-layer extraction adds no keys).
- head/registry.py imports ALL head modules unconditionally, and iron_vla.py imports `configs.config_schema` + `dataset.util` at module level — these packages must be importable even for action-only use.

## MISSING from former/models — must be copied from the main iron_vla repo

1. `configs/config_schema.py` (PolicyConfig + all head configs) — import-time hard requirement.
2. `dataset/util.py` (`get_processor`, `update_processor_cache`, `registered_tokens`, `get_token_insertion_idx`) — PROPRIO id + tokenizer construction.
3. `dataset/special_tokens.py` — the literal token strings & registration order (token IDs must match the trained ckpt).
4. `dataset/lerobot/transform/backbone_transforms.py` — chat template / prompt assembly / `num_proprio_tokens` / image preprocessing settings (the eval adapter must replicate this exactly).
5. The normalization machinery + stats format (dataset-side; no trace in models/).
6. The actual train config yaml (`vlmact_r01` family) — determines head type, action_horizon, dims, num_categories, padding conventions for LIBERO cloning.
7. For training: `train.py`, `build_optimizer`, the lerobot dataset pipeline; for reference: `IronInferencer`.
Only-if-STM: `util/temporal_sampling.py` (LIBERO config keeps STM off).

## Eval-adapter recipe (grounded)

1. Build PolicyConfig identical to the LIBERO training config; `build_model`; `deserialize(ckpt)`; `.eval().cuda()`.
2. Ensure `update_processor_cache()` ran (populates `registered_tokens['PROPRIO']`).
3. Per replan: two LIBERO cam images (256², **rotated 180°** `[::-1,::-1]` to match dataset orientation) + prompt via the training-side transform → AutoProcessor → batch (+ `state=[1,1,8]` normalized, `embodiment_id`, dummy `action`/`action_mask`) → `forward(compute_loss=False)` → `out[task]['pred'][0]` → denormalize → slice 7 dims → gripper remap (·2−1, sign-invert, optional binarize — same dataset convention as FastWAM) → execute `replan_steps`, repeat.
4. Wrap that in FastWAM's `experiments/libero/eval_libero_single.py` skeleton (replace model-loading + `_predict_action_chunk` + `_obs_to_model_input`); manager/tmux/summarize/vendored-LIBERO stack reused unchanged.

Full reader reports (assembly/backbone/action-head/gaps) archived in the session scratchpad; regenerate by re-reading `former/models` if needed.
