#!/usr/bin/env python
"""One-shot prep for locally downloaded gr1_unified.* dirs.

Provisioning lessons from cc_memo/12 ("gr1_unified provisioning lessons"):
the NVIDIA X-Embodiment-Sim dirs need a dtype fix + stats regeneration before
the GR00T trainer accepts them. Order matters:

1. Completeness check against .remote_manifest.json — abort on missing or
   size-mismatched files. Running repair on a partial download would DROP
   the not-yet-downloaded episodes.
2. info.json dtype fix: observation.state/action declare dtype "object";
   generate_stats only stats "float*" features, so without this the stats
   silently omit state/action and the loader dies with KeyError.
3. repair_lerobot_metadata.py (file-index repair; no-op on complete dirs).
4. Regenerate meta/stats.json + meta/relative_stats.json (shipped stats.json
   predates the dtype fix), then ASSERT observation.state/action landed.
5. H264 fourcc sweep (trainer's decoder cannot handle some AV1).

Usage: env/groot/bin/python scripts/prep_gr1_unified.py [--root DIR] [--skip-completeness]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GROOT = REPO_ROOT / "Isaac-GR00T"
DEFAULT_ROOT = Path("/data/ruijiezhang/gr1_unified")
EMBODIMENT_TAG = "ROBOCASA_GR1_TABLETOP"
FIXED_KEYS = ("observation.state", "action")
FOURCC = {b"avc1": "h264", b"av01": "av1", b"hev1": "hevc", b"hvc1": "hevc"}


def check_completeness(root: Path) -> list[Path]:
    manifest = json.loads((root / ".remote_manifest.json").read_text())
    problems = []
    dirs = []
    for dirname, files in sorted(manifest.items()):
        local_dir = root / dirname
        dirs.append(local_dir)
        for rel, size in files.items():
            p = root / rel
            if not p.is_file():
                problems.append(f"MISSING {rel}")
            # meta/ files are modified by this very script (dtype fix, stats
            # regen, repair) — existence-only so prep stays idempotent.
            elif "/meta/" not in rel and p.stat().st_size != size:
                problems.append(f"SIZE {rel}: local {p.stat().st_size} != remote {size}")
    if problems:
        for line in problems[:20]:
            print(f"  {line}")
        raise SystemExit(
            f"completeness check FAILED: {len(problems)} problem(s) across "
            f"{len(manifest)} dirs — is the download finished?"
        )
    print(f"completeness OK: {sum(len(f) for f in manifest.values())} files in {len(dirs)} dirs")
    return dirs


def fix_dtypes(dataset_dir: Path) -> bool:
    info_path = dataset_dir / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    changed = False
    for key in FIXED_KEYS:
        feat = info["features"].get(key)
        if feat is None:
            raise SystemExit(f"{info_path}: feature {key} missing entirely")
        if feat["dtype"] == "object":
            feat["dtype"] = "float64"
            changed = True
    if changed:
        bak = info_path.with_suffix(".json.bak")
        if not bak.exists():
            shutil.copy2(info_path, bak)
        info_path.write_text(json.dumps(info, indent=4) + "\n")
    return changed


def regen_stats(dataset_dir: Path) -> str:
    # Import inside the worker: gr00t is heavy and ProcessPoolExecutor forks.
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.stats import generate_rel_stats, generate_stats

    (dataset_dir / "meta" / "stats.json").unlink(missing_ok=True)
    (dataset_dir / "meta" / "relative_stats.json").unlink(missing_ok=True)
    generate_stats(dataset_dir)
    generate_rel_stats(dataset_dir, EmbodimentTag.resolve(EMBODIMENT_TAG))

    stats = json.loads((dataset_dir / "meta" / "stats.json").read_text())
    for key in FIXED_KEYS:
        if key not in stats or "mean" not in stats[key]:
            raise RuntimeError(f"{dataset_dir.name}: {key} absent from regenerated stats.json")
    return dataset_dir.name


def probe_codec(mp4: Path) -> tuple[Path, str]:
    with mp4.open("rb") as f:
        head = f.read(1 << 20)
        f.seek(max(0, mp4.stat().st_size - (1 << 20)))
        tail = f.read()
    for cc, name in FOURCC.items():
        if cc in head or cc in tail:
            return mp4, name
    return mp4, "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--skip-completeness", action="store_true")
    args = ap.parse_args()
    root = args.root

    if args.skip_completeness:
        dirs = sorted(p for p in root.glob("gr1_unified.*") if p.is_dir())
    else:
        dirs = check_completeness(root)

    print(f"\n[2/5] dtype fix on {len(dirs)} dirs")
    n_fixed = sum(fix_dtypes(d) for d in dirs)
    print(f"  fixed {n_fixed} info.json (rest already float64)")

    print(f"\n[3/5] repair_lerobot_metadata (no-op stats: regenerated in step 4)")
    cmd = [
        sys.executable,
        str(GROOT / "scripts" / "repair_lerobot_metadata.py"),
        *[str(d) for d in dirs],
        "--embodiment-tag",
        EMBODIMENT_TAG,
        "--no-regenerate-stats",
    ]
    subprocess.run(cmd, check=True, cwd=GROOT)

    print(f"\n[4/5] regenerate stats + relative stats ({len(dirs)} dirs, 6 workers)")
    with ProcessPoolExecutor(max_workers=6) as pool:
        for name in pool.map(regen_stats, dirs):
            print(f"  stats OK: {name}")

    print("\n[5/5] codec sweep")
    mp4s = [p for d in dirs for p in d.rglob("*.mp4")]
    bad = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        for mp4, codec in pool.map(probe_codec, mp4s):
            if codec != "h264":
                bad.append((mp4, codec))
    if bad:
        for mp4, codec in bad[:10]:
            print(f"  NON-H264 ({codec}): {mp4}")
        raise SystemExit(
            f"{len(bad)}/{len(mp4s)} videos are not H264 — run "
            "Isaac-GR00T/examples/SimplerEnv/convert_av1_to_h264.py before training"
        )
    print(f"  all {len(mp4s)} videos H264")
    print("\nPREP COMPLETE — dataset ready for training")


if __name__ == "__main__":
    main()
