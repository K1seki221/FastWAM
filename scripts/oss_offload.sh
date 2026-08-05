#!/usr/bin/env bash
# v2: upload groot_runs checkpoint dirs to OSS, exact per-file verify, then delete local.
# Idempotent: cp -u resumes/skips already-uploaded objects.
set -uo pipefail
CFG=/dataset_rc/ruijie.zhang@xiaopeng.com/codebase/xprobot/iron_vla/data_tools/sync_ckpt/oss_config
SRC=/dataset_rc/ruijie.zhang@xiaopeng.com/projects/groot_runs
DST=oss://xrobot-log/user-upload/fuyao/ruijie-zhang/groot_runs
RUNS="baseline_libero_10 router_libero_10 s1_baseline_libero_all s1_router_libero_all s1_router_rlr5x_libero_all s1_router_rlr10x_libero_all s1_gr1_baseline s1_gr1_router"

for run in $RUNS; do
  d="$SRC/$run"
  [[ -d "$d" ]] || { echo "OFFLOAD_SKIP $run already_gone"; continue; }
  echo "OFFLOAD_START $run files=$(find "$d" -type f | wc -l)"
  if ! ossutil --config-file "$CFG" cp -r -f -u "$d" "$DST/$run/" --jobs 4 --parallel 8 >"/tmp/oss_cp_$run.log" 2>&1; then
    echo "OFFLOAD_FAIL $run cp_error (see /tmp/oss_cp_$run.log)"
    continue
  fi
  echo "OFFLOAD_UPLOADED $run"
  if python3 /tmp/oss_verify.py "$d" "$DST/$run/" "$CFG"; then
    rm -rf "$d"
    echo "OFFLOAD_DELETED $run"
  else
    echo "OFFLOAD_FAIL $run verify_failed (LOCAL KEPT)"
  fi
done
echo "OFFLOAD_DONE all"
