#!/usr/bin/env python3
"""One-off: reset Single Creampie + Multiple Creampie for hand re-sorting.

For every video in those two albums:
  1. clear the description (so the captioner re-queues it),
  2. remove it from Single Creampie / Multiple Creampie ONLY,
  3. unarchive it, so it surfaces in the main timeline.

Sub-album membership (Puta Locura, Czech, Bondage, Slutwife, Gangbang, Lactation,
Internet Titties, ...) is deliberately left untouched -- that curation can't be regenerated
until watermark-based routing exists, and per instruction only Single/Multiple get stripped.

Note the whole 200.000.xxx range now counts as "multiple creampie" to the captioner, so a
video left in e.g. Puta Locura will be treated as a multi once it's re-captioned.

Pass --dry-run to preview. A real run writes an undo file first and aborts if it can't.
"""
import json
import os
import sys
import time

sys.path.insert(0, "/app")
from captioner import (  # noqa: E402
    IMMICH_URL, immich_headers, immich_remove_from_album, SLEEP_SECONDS,
)
import requests

SINGLE = ("Single Creampie", "3a22144e-143c-4f43-a508-8b3f7fadbcb5")
MULTIPLE = ("Multiple Creampie", "e7479905-44b5-42ca-86d0-aaf8fb7c36e3")
CHUNK = 200


def album_video_ids(album_id):
    ids = []
    page = 1
    while True:
        r = requests.post(
            f"{IMMICH_URL}/api/search/metadata",
            headers={**immich_headers(), "Content-Type": "application/json"},
            json={"albumIds": [album_id], "type": "VIDEO", "page": page, "size": 500},
            timeout=60,
        )
        r.raise_for_status()
        data = r.json().get("assets", {})
        items = data.get("items", [])
        if not items:
            break
        ids.extend(a["id"] for a in items)
        nxt = data.get("nextPage")
        if nxt is None:
            break
        page = int(nxt)
        time.sleep(SLEEP_SECONDS)
    return ids


def bulk_put(ids, payload, label):
    for i in range(0, len(ids), CHUNK):
        chunk = ids[i:i + CHUNK]
        r = requests.put(
            f"{IMMICH_URL}/api/assets",
            headers={**immich_headers(), "Content-Type": "application/json"},
            json={"ids": chunk, **payload},
            timeout=60,
        )
        if r.status_code >= 300:
            print(f"[error] {label} chunk {i}: {r.status_code} {r.text}", flush=True)
        else:
            print(f"[ok] {label} {i + len(chunk)}/{len(ids)}", flush=True)
        time.sleep(SLEEP_SECONDS)


def main():
    dry_run = "--dry-run" in sys.argv

    per_album = {name: album_video_ids(aid) for name, aid in (SINGLE, MULTIPLE)}
    for name, ids in per_album.items():
        print(f"[info] {name}: {len(ids)} videos", flush=True)
    all_ids = sorted({a for ids in per_album.values() for a in ids})
    print(f"[info] {len(all_ids)} unique videos to reset", flush=True)

    if dry_run:
        print("[dry-run] would clear descriptions, unarchive, and remove from Single/Multiple only")
        return

    undo_path = os.environ.get("CLEANUP_UNDO", "/undo/creampie_reset_undo.json")
    try:
        os.makedirs(os.path.dirname(undo_path), exist_ok=True)
        with open(undo_path, "w") as fh:
            json.dump({name: ids for name, ids in per_album.items()}, fh, indent=2)
        print(f"[undo] wrote undo list to {undo_path}", flush=True)
    except OSError as e:
        print(f"[fatal] could not write undo file ({e}) -- aborting without changes", flush=True)
        return

    print("\n[step] clearing descriptions", flush=True)
    bulk_put(all_ids, {"description": ""}, "cleared")

    print("\n[step] unarchiving", flush=True)
    bulk_put(all_ids, {"visibility": "timeline"}, "unarchived")

    print("\n[step] removing from Single/Multiple only", flush=True)
    for name, aid in (SINGLE, MULTIPLE):
        for asset_id in per_album[name]:
            immich_remove_from_album(asset_id, aid)
            time.sleep(SLEEP_SECONDS)
        print(f"[ok] removed {len(per_album[name])} from {name}", flush=True)

    print(f"\n[done] reset {len(all_ids)} videos", flush=True)


if __name__ == "__main__":
    main()
