#!/usr/bin/env python3
"""Write Mike's hand-timed creampie counts into captions (count field only).

stdin: JSON list of {"asset_id", "creampie_times_seconds": [...], "count": N}. For each asset the
caption's "Separate Creampies | N (~mm:ss, ...)" field is replaced (or added in front), keeping
every other field -- including a trailing "Title" / "Dog" field -- and any generation info
byte-identical. Old descriptions are written to --backup first. Pair with a row in
captioner_hand_counted (the lock) so no recount overwrites it.

    docker cp scripts/apply_hand_counts.py immich_captioner:/tmp/ && \\
    docker exec -i immich_captioner python3 /tmp/apply_hand_counts.py --backup /tmp/b.json < counts.json
"""
import argparse
import json
import sys

import requests

sys.path.insert(0, "/app")
import captioner as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--backup", default="/tmp/hand_counts_backup.json")
args = ap.parse_args()


def fmt(s):
    s = int(s)
    return f"{s // 60:02d}:{s % 60:02d}"


def set_field(caption, count, times):
    parts = [p.strip() for p in (caption or "").split(" | ")]
    if len(parts) >= 2 and parts[0].lower() == "separate creampies":
        parts = parts[2:]
    rest = " | ".join(p for p in parts if p)
    field = f"Separate Creampies | {count} (~{', '.join(times)})" if times else f"Separate Creampies | {count}"
    return f"{field} | {rest}" if rest else field


backup = []
for r in json.load(sys.stdin):
    aid = r["asset_id"]
    a = requests.get(f"{C.IMMICH_URL}/api/assets/{aid}", headers=C.immich_headers(), timeout=30)
    if a.status_code != 200:
        print("MISSING", aid, a.status_code, flush=True)
        continue
    desc = (a.json().get("exifInfo") or {}).get("description") or ""
    cap, gen = C.split_description(desc)
    new_cap = set_field(cap, r["count"], [fmt(t) for t in r["creampie_times_seconds"]])
    new = C.compose_description(new_cap, gen)
    backup.append({"asset_id": aid, "old": desc})
    if new == desc:
        print("same", aid[:8], flush=True)
        continue
    u = requests.put(f"{C.IMMICH_URL}/api/assets/{aid}", headers={**C.immich_headers(), "Content-Type": "application/json"},
                     json={"description": new}, timeout=30)
    print(u.status_code, aid[:8], new_cap.split(" | Breast")[0][:120], flush=True)
json.dump(backup, open(args.backup, "w"))
