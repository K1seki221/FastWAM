# Datasets & dataloading

## Stack (outermost → innermost)

1. `RobotVideoDataset` (`src/fastwam/datasets/lerobot/robot_video_dataset.py`) — the `_target_` in `configs/data/*.yaml`; instantiated by `runtime.build_datasets`.
2. `BaseLerobotDataset` (`base_lerobot_dataset.py`) — per-key state/action/image dicts, padding flags, retries, dataset-stats computation, holds the processor.
3. `MultiLeRobotDataset` → `LeRobotDataset` (`lerobot/lerobot_dataset.py`) — **vendored & patched LeRobot v2.1** (parquet per episode + mp4 per camera; `data/chunk-XXX/episode_XXXXXX.parquet`, `videos/chunk-XXX/{video_key}/episode_XXXXXX.mp4`, `meta/{info,tasks,episodes,episodes_stats}.jsonl|json`). Timestamp-sync and version checks are commented out vs upstream. Video decode via torchcodec (approximate seek); any decode error falls back to pyav with only a logged warning.
4. `FastWAMProcessor` (`processors/fastwam_processor.py`) — image transforms, action/state transforms, normalization, dim merging; runs inside `BaseLerobotDataset.__getitem__`.

## Final training sample (what the model sees)

| Key | Shape | Notes |
|---|---|---|
| `video` | [3, 9, H, W] float [-1,1] | 33 raw frames subsampled ×4 → 9; LIBERO 224×448 (2-cam horizontal concat), RoboTwin 384×320 (3-cam mosaic: head 256×320 top, wrists 128×160 each side-by-side bottom) |
| `action` | [32, A] | normalized; LIBERO A=7 (6 delta-EEF + 1 gripper), RoboTwin A=14 (absolute joints) |
| `proprio` | [32, P] | LIBERO P=8 (eef pose 6 + gripper 2), RoboTwin P=14; sliced from 33 states (drops last) |
| `prompt` | str | `DEFAULT_PROMPT.format(task=...)` |
| `context` / `context_mask` | [128, D] / [128] | cached T5 embeds; padded rows zeroed, **mask forced all-ones** (Wan2.2 convention) |
| `image_is_pad` / `action_is_pad` / `proprio_is_pad` | [9] / [32] / **[33]** | note proprio_is_pad off-by-one vs proprio |

Sequence math: `num_frames: 33`, `action_video_freq_ratio: 4` → 32 actions, 9 video frames; asserts `(num_frames−1) % ratio == 0` and `((num_frames−1)/ratio) % 4 == 0`.

## Normalization (two separate "stats" systems!)

- LeRobot's own `meta/episodes_stats.jsonl` → internal only.
- **FastWAM's `dataset_stats.json`** (from `BaseLerobotDataset.get_dataset_stats`: full-episode parquet sweep, per-key `stepwise_*` and `global_*` min/max/q01/q99/mean/std) → drives `LinearNormalizer`.
- First train run (`pretrained_norm_stats: null`): rank 0 computes, saves to `{work_dir}/dataset_stats.json`, broadcasts. Subsequent runs point `pretrained_norm_stats` at that file (avoids the expensive full-dataset sweep). A directly-constructed val dataset raises without stats, but via `runtime.build_datasets` the val instantiation auto-receives the fallback chain `data.val.pretrained_norm_stats` → `data.train.pretrained_norm_stats` → `{work_dir}/dataset_stats.json`.
- Modes: LIBERO `min/max` (→[-1,1]); RoboTwin `z-score` with `pretrained_norm_stats: ./data/robotwin2.0/dataset_stats.json`. States always use `global_*` stats; actions use `stepwise_*` only if `use_stepwise_action_norm` (false here). **`forward` clamps to [-5,5] — not invertible in `backward`.**
- LIBERO `delta_action_dim_mask.default: [T×6, F]` — on padded steps the 6 delta-EEF dims are zeroed BEFORE normalization (gripper untouched).

## LIBERO vs RoboTwin configs (configs/data/)

| | `libero_2cam.yaml` | `robotwin.yaml` |
|---|---|---|
| dirs | 4× `./data/libero_mujoco3.3.2/libero_{spatial,object,goal,10}_no_noops_lerobot` | `./data/robotwin2.0/robotwin2.0` |
| cameras | `image`, `wrist_image` (512²→224²) | `cam_high`, `cam_left_wrist`, `cam_right_wrist` (480×640→240×320) |
| concat | `horizontal` → 224×448 | `robotwin` mosaic → 384×320 |
| action/state dims | 7 / 8 | 14 / 14 |
| norm | min/max, computed at startup | z-score, precomputed stats file |
| val | none (`val_set_proportion: 0.0`) | `val:` node duplicated (`val_set_proportion: 0.01`, `is_training_set: false`) — **edit shared fields in BOTH nodes** |
| text cache | `./data/text_embeds_cache/libero` | `./data/text_embeds_cache/robotwin` |

## Foot-guns

- **Errors are swallowed with random-sample substitution at two levels**: `BaseLerobotDataset.__getitem__` gets 5 total attempts (requested idx + 4 random substitutes, then raises); `RobotVideoDataset.__getitem__` catches any exception and returns one random sample (prints traceback). Corrupt data trains silently.
- Text-embed cache path: `{cache_dir}/{sha256(full formatted prompt)}.t5_len{context_len}.wan22ti2v5b.pt` — enc_id **hardcoded** in `robot_video_dataset.py`; changing `DEFAULT_PROMPT`, `override_instruction`, `context_len`, or model_id invalidates/breaks lookup. Missing file → FileNotFoundError pointing at `scripts/precompute_text_embeds.py`.
- A typo'd `dataset_dirs` path triggers a confusing **HF Hub snapshot_download attempt** (path used as repo_id → 401/404), not "file not found".
- `LEROBOT_HOME` env var set → import-time ValueError (`constants.py`); use `HF_LEROBOT_HOME`.
- Episode-level train/val split with fixed internal seed 42 (not exposed by RobotVideoDataset).
- Window padding: indices clamp at episode edges; frames past the end replicate the last frame with `*_is_pad=True`.
- Images round-trip float[0,1] → uint8 → float (quantization). Image `raw_shape` validation is commented out.
- `during_training=False` would skip video decode entirely, but `RobotVideoDataset` always sets it True.
- `search_dataset_stats_cache_json` in `utils/normalizer.py` is dead/legacy code (uses GitPython, unused).
- Stats aggregation weights episodes equally regardless of length; stats-path actions use `sliding_window_with_replication` (replicates final action row, slightly biasing end-of-episode stats).
