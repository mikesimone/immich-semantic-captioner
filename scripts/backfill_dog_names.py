#!/usr/bin/env python3
"""Backfill a "Dog | <name>" field into existing captions of assets in the per-dog albums.

Text-only (no models). The dog comes from the album-identity prefix map (000.001 - Ruby ...
000.010 - Randy, IDENTITY_ALBUM_PREFIX_MAP); an asset in two dog albums gets both names
("Dog | Bella, Ruby"). The field goes before a trailing "Title | ..." field if there is one,
otherwise at the end; everything else (incl. locked counts, generation info) is untouched.
Skips blank captions and ones that already carry a Dog field. Dry run unless --apply.

    docker cp scripts/backfill_dog_names.py immich_captioner:/tmp/ && \\
    docker exec immich_captioner python3 /tmp/backfill_dog_names.py --apply --backup /tmp/dogs_backup.json
"""
import argparse
import collections
import json
import sys

import requests

sys.path.insert(0, "/app")
import captioner as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--apply", action="store_true")
ap.add_argument("--backup", default="/tmp/dog_names_backup.json")
args = ap.parse_args()

conn = C.pg_connect()
with conn.cursor() as cur:
    cur.execute("""SELECT a.id::text, al."albumName", ae.description FROM asset a
                   JOIN album_asset aa ON aa."assetId" = a.id JOIN album al ON al.id = aa."albumId"
                   JOIN asset_exif ae ON ae."assetId" = a.id
                   WHERE a."deletedAt" IS NULL AND al."albumName" ~ '^000\\.0(0[1-9]|10) '""")
    rows = cur.fetchall()

dogs, desc_of = collections.defaultdict(list), {}
for aid, album, desc in rows:
    for name in C._identity_prefix_matches(album):
        if name not in dogs[aid]:
            dogs[aid].append(name)
    desc_of[aid] = desc or ""


def with_dog(caption, names):
    parts = caption.split(" | ")
    if "Dog" in parts[:-1]:
        return caption
    field = f"Dog | {', '.join(sorted(names))}"
    if len(parts) >= 2 and parts[-2] == "Title":
        return " | ".join(parts[:-2] + [field] + parts[-2:])
    return f"{caption} | {field}"


per_dog, changed, skipped, backup, samples = collections.Counter(), 0, 0, [], []
for aid, names in dogs.items():
    cap, gen = C.split_description(desc_of[aid])
    if not cap or not names:
        skipped += 1
        continue
    new_cap = with_dog(cap, names)
    if new_cap == cap:
        skipped += 1
        continue
    backup.append({"asset_id": aid, "old": desc_of[aid]})
    if len(samples) < 3:
        samples.append(new_cap[-140:])
    if args.apply:
        r = requests.put(f"{C.IMMICH_URL}/api/assets/{aid}", headers={**C.immich_headers(), "Content-Type": "application/json"},
                         json={"description": C.compose_description(new_cap, gen)}, timeout=30)
        if r.status_code != 200:
            print("FAIL", aid, r.status_code, flush=True)
            continue
    changed += 1
    for n in names:
        per_dog[n] += 1

json.dump(backup, open(args.backup, "w"))
print(json.dumps({"assets_in_dog_albums": len(dogs), "named": changed, "skipped_blank_or_done": skipped,
                  "per_dog": dict(per_dog), "applied": args.apply, "samples": samples}, indent=1))
