export HF_ENDPOINT=https://hf-mirror.com
BASE=/dataset_rc/ruijie.zhang@xiaopeng.com

for s in 10 goal object spatial; do
  d=$BASE/libero_groot/libero_${s}_no_noops_1.0.0_lerobot
  until hf download IPEC-COMMUNITY/libero_${s}_no_noops_1.0.0_lerobot \
        --repo-type dataset --local-dir $d --max-workers 4; do
    echo "retrying libero_${s} in 60s..."; sleep 60
  done
  cp $BASE/FastWAM/Isaac-GR00T/examples/LIBERO/modality.json $d/meta/
done

# goal patch — only valid once libero_goal is fully downloaded
cp $BASE/FastWAM/Isaac-GR00T/examples/LIBERO/patches/episode_000082.mp4 \
   $BASE/libero_groot/libero_goal_no_noops_1.0.0_lerobot/videos/chunk-000/observation.images.image/episode_000082.mp4

du -sh $BASE/libero_groot/*