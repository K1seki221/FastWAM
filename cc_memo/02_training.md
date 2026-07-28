# Training pipeline

## Launch chain

`bash scripts/train_zero1.sh <nproc> [hydra overrides…]` → `accelerate launch --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml --num_processes N scripts/train.py "output_dir=./runs/<TASK_BASENAME>/<RUN_ID>" "wandb.name=<TASK_BASENAME>" …` → `train.py` (`@hydra.main(config_name="train")`, registers OmegaConf resolvers at import) → `fastwam.runtime.run_training(cfg)` → `Wan22Trainer.train()`.

- `train_zero2.sh` is identical but uses the ZeRO-2 accelerate config. `accelerate_zero0.yaml` is orphaned (nothing references it).
- The shell script parses `task=<name>` out of the args to derive `TASK_BASENAME`; RUN_ID = timestamp (or export `RUN_ID` yourself; across nodes it is synced via `torch.distributed.TCPStore` on `MASTER_PORT+11`). Caveat: `NNODES`/`NODE_RANK`/`MASTER_ADDR`/`MASTER_PORT` are read but only feed the RUN_ID sync and validation — the `accelerate launch` call passes just `--config_file`/`--num_processes` and the accelerate yaml hardcodes `num_machines: 1`, `machine_rank: 0`, so the scripts as shipped do NOT do a real multi-node launch.
- All paths are relative — **must run from repo root**.
- DS configs (`scripts/ds_configs/ds_zero{1,2}_config.json`): batch-size/grad-accum keys "auto" (zero_optimization block is explicit: no offload, overlap_comm false, bucket sizes 2e8), no fp16/bf16 block — precision comes solely from the trainer's `Accelerator(mixed_precision=cfg.mixed_precision)`; the accelerate yamls deliberately keep `mixed_precision: null`.

## `Wan22Trainer` (src/fastwam/trainer.py)

- **Requires DeepSpeed**: trainer init dereferences `accelerator.state.deepspeed_plugin` unguarded (~:68) — plain `python scripts/train.py` crashes. Always launch via the shell scripts.
- Trainable params: `_apply_dit_only_train_mode` freezes everything, then unfreezes `model.dit` (which for FastWAM is the whole **MoT** = video + action experts, via the `dit = mot` alias) + `proprio_encoder`. VAE/T5 stay frozen. Optimizer `AdamW(betas=(0.9, 0.95))`.
- Model weights are **cast to bf16** (`model_dtype` from `mixed_precision`), not fp32-master + autocast.
- Steps: `max_steps` (if set) wins over `num_epochs`; else total = ceil(len(ds)/(bs·world))·epochs / grad_accum. Warmup hardcoded 5% of total; cosine (`CosineAnnealingLR`, eta_min=lr·0.01) or constant. `save_every`/`eval_every`/`log_every` count **optimizer steps**.
- Dataloader: custom `ResumableEpochSampler` (utils/samplers.py — full randperm seeded `seed+epoch+epoch_offset`; not a distributed sampler, sharding is done by Accelerate). Dataset length consistency is asserted across ranks.
- Logging: rich stdout + optional wandb (`wandb.enabled=false` default; enabling without wandb installed raises). **No EMA, no tensorboard anywhere.**
- `evaluate()` runs on ALL ranks every `eval_every`: random val sample per rank; computes val loss, a full inference rollout (`eval_num_inference_steps`, seed 42), PSNR/SSIM (rollout-vs-GT `psnr_rg`, rollout-vs-VAE-decode `psnr_rd`, decode-vs-GT `psnr_dg`), and action L1/L2 de-normalized through `val_dataset.lerobot_dataset.processor` (requires `proprio` in the sample). Each rank writes `eval/step_XXXXXX_rank_RRR.mp4` (pred / vae_recon / gt stacked vertically, fps 8).
- If `data.val` is null, **val_ds IS train_ds** (same object) — `build_datasets` in runtime.py.

## Run directory layout (`runs/<task>/<run_id>/`)

```
config.yaml                        # resolved cfg (written by all ranks, benign race)
dataset_stats.json                 # normalization stats (written on first run when pretrained_norm_stats null)
checkpoints/weights/step_XXXXXX.pt # plain torch payload {"mot": ..., "step", "torch_dtype"[, "proprio_encoder"]} — main rank only
checkpoints/state/step_XXXXXX/     # accelerate/DeepSpeed save_state shards + trainer_state.json {global_step, epoch, batch_in_epoch}
eval/step_XXXXXX_rank_RRR.mp4
wandb/                             # if enabled
```

## Resume semantics

- `resume=<state dir>` (e.g. `.../checkpoints/state/step_002000`) → full restore: accelerator state + global_step/epoch + dataloader position (`set_epoch_offset` / `set_resume_batch_offset`; batch offset only applies while the sampler's internal epoch == 0). Missing `trainer_state.json` → step parsed from dir name, no dataloader resume.
- `resume=<weights .pt file>` → weights ONLY; optimizer/scheduler/step lost (explicit warning under ZeRO-2).

## Preprocessing scripts (both must be run before first training)

1. **`scripts/preprocess_action_dit_backbone.py`** (argparse, NOT hydra): builds the interpolated ActionDiT backbone `.pt` (see `01_model_architecture.md`). Loads the model yaml with plain OmegaConf — unresolved `${data...}` interpolations fall back (action_dim → 7 with warning). Rerun when ActionDiT/video-DiT arch or scaling policy changes.
2. **`scripts/precompute_text_embeds.py`** (hydra, same `train` config tree; multi-GPU via torchrun, prompts sharded `rank::world_size`): scans `cfg.data` for nodes with `dataset_dirs`, reads `{dir}/meta/tasks.jsonl`, formats through `DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"` (defined in `robot_video_dataset.py`), encodes with T5 bf16, writes `{sha256(prompt)}.t5_len{context_len}.{enc_id}.pt` (payload `{"context": [L,D] bf16, "mask": [L] bool}`) into `text_embedding_cache_dir`. Defaults `overwrite=true` (pass `+overwrite=false` for incremental). Extra key `override_instruction` encodes a single custom prompt.
   - **enc_id trap**: the script derives enc_id from `model_id` ("Wan2.2-TI2V-5B" → `wan22ti2v5b`) but the dataset **hardcodes** `wan22ti2v5b` in the lookup path — changing `model_id` silently breaks cache lookup (dataset raises FileNotFoundError telling you to run the script). Cache also invalidated by changes to task lists, `context_len` (128), or the prompt template.

## Utils (src/fastwam/utils/)

- `config_resolvers.py`: registers `oc.load` (shadowed custom loader), `eval` (**Python builtin eval — arbitrary code exec in configs**), `split`, `max`, `round_up/down`, `sum_shapes`, `max_action_dim`, `max_state_dim`. Idempotent; must be registered before hydra composition (each entrypoint does it). Current yamls don't actually use the custom ones.
- `misc.py`: work-dir registry (`register_work_dir`/`get_work_dir`, default `./runs/`) — this is how the dataset knows where to write `dataset_stats.json`. Imports boto3/termcolor/DTensor at module top (must be installed).
- `pytorch_utils.py`: `set_global_seed` (seeds with `seed + global_rank`), dict helpers, `optimizer_to`, `is_rank0`.
- `samplers.py`: `ResumableEpochSampler` (above). `logging_config.py`: rich logging, non-rank-0 loggers disabled. `video_io.py`: `save_mp4` (libx264, pads odd dims). `video_metrics.py`: `video_psnr`, `video_ssim` ([3,T,H,W] in [0,1]).

## Reference hyperparams (task configs)

All 6 tasks: batch_size 16 (per GPU), num_workers 8, lr 1e-4 cosine, weight_decay 1e-2, grad_accum 1, `model.mot_checkpoint_mixed_attn: false`. LIBERO: 10 epochs, save 2000 / eval 200 (paper: 1 node × 8 GPUs). RoboTwin: 5 epochs, save 2500 / eval 500 (paper: 64 GPUs).
