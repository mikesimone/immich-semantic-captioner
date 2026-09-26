#!/usr/bin/env python3
"""Backfill the "Title | <filename>" field into existing VIDEO captions (text-only, no models).

Uses captioner.video_title()/with_title(), so the skip rules match what the captioner writes
for new videos. Only the caption part changes; generation info and every other field (incl.
hand-locked creampie counts) stay byte-identical. Skips blank captions, bare "Please
Categorize", and captions that already carry the same Title. Backs up every changed
description to --backup before writing. Dry run unless --apply.

Run inside the captioner container (it has the captioner code and Postgres/Immich env):
    docker cp scripts/backfill_video_titles.py immich_captioner:/tmp/ && \\
    docker exec immich_captioner python3 /tmp/backfill_video_titles.py --apply --backup /tmp/titles_backup.json
"""
import argparse
import json
import sys

import requests

sys.path.insert(0, "/app")
import captioner as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--apply", action="store_true")
ap.add_argument("--backup", default="/tmp/video_titles_backup.json")
args = ap.parse_args()

conn = C.pg_connect()
with conn.cursor() as cur:
    cur.execute("""SELECT a.id::text, a."originalFileName", ae.description FROM asset a
                   JOIN asset_exif ae ON ae."assetId" = a.id
                   WHERE a.type = 'VIDEO' AND a."deletedAt" IS NULL
                     AND btrim(coalesce(ae.description, '')) <> ''""")
    rows = cur.fetchall()

changed, skipped_name, skipped_other, backup, samples = 0, 0, 0, [], []
for aid, fn, desc in rows:
    cap, gen = C.split_description(desc)
    title = C.video_title(fn)
    if not cap:
        skipped_other += 1
        continue
    if title is None:
        skipped_name += 1
        continue
    new_cap = C.with_title(cap, title)
    if new_cap == cap:
        skipped_other += 1
        continue
    new = C.compose_description(new_cap, gen)
    backup.append({"asset_id": aid, "old": desc})
    if len(samples) < 5:
        samples.append(new_cap[-160:])
    if args.apply:
        r = requests.put(f"{C.IMMICH_URL}/api/assets/{aid}", headers={**C.immich_headers(), "Content-Type": "application/json"},
                         json={"description": new}, timeout=30)
        if r.status_code != 200:
            print("FAIL", aid, r.status_code, flush=True)
            continue
    changed += 1

json.dump(backup, open(args.backup, "w"))
print(json.dumps({"videos_with_captions": len(rows), "titled": changed, "skipped_uninformative_name": skipped_name,
                  "skipped_other": skipped_other, "applied": args.apply, "samples": samples}, indent=1))
