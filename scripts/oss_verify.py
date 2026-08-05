"""Exact per-file verification of an OSS-uploaded run dir.
Usage: oss_verify.py <local_dir> <oss_prefix> <config_file>
Prints VERIFY_PASS or VERIFY_FAIL with details; exit 0 only on pass.
Extra remote objects (directory markers) are ignored.
"""
import os, re, subprocess, sys

local_dir, prefix, cfg = sys.argv[1], sys.argv[2].rstrip("/") + "/", sys.argv[3]

out = subprocess.run(
    ["ossutil", "--config-file", cfg, "ls", prefix],
    capture_output=True, text=True, timeout=600,
).stdout
remote = {}
for line in out.splitlines():
    m = re.match(r"\s*\S+ \S+ \+0000 UTC\s+(\d+)\s+\S+\s+\S+\s+(oss://\S+)$", line)
    if m and not m.group(2).endswith("/"):
        remote[m.group(2)] = int(m.group(1))

missing, mismatched, checked = [], [], 0
for root, _, files in os.walk(local_dir):
    for f in files:
        lp = os.path.join(root, f)
        rel = os.path.relpath(lp, local_dir)
        key = prefix + rel
        size = os.path.getsize(lp)
        checked += 1
        if key not in remote:
            missing.append(rel)
        elif remote[key] != size:
            mismatched.append(f"{rel} local={size} remote={remote[key]}")

if not missing and not mismatched and checked > 0:
    print(f"VERIFY_PASS files={checked} remote_objects={len(remote)}")
    sys.exit(0)
print(f"VERIFY_FAIL checked={checked} missing={len(missing)} mismatched={len(mismatched)}")
for x in (missing + mismatched)[:10]:
    print("  " + x)
sys.exit(1)
