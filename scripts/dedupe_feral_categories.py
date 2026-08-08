#!/usr/bin/env python3
"""One-off: feral-on-human content is its own category and doesn't belong in either the
human Single Creampie album or Furry Stuff.

- Single Creampie: feral videos are *assumed* to have one creampie for captioning purposes,
  which had the side effect of auto-filing every one of them into the human Single Creampie
  album. The captioner no longer does this (see the `not feral` guard in process_candidate);
  this clears the residue.
- Furry Stuff: "feral" specifically means a real, non-anthropomorphic animal -- the opposite
  of furry, which is animal-humanoid characters. Most of this overlap predates the captioner's
  furry auto-filing rather than being caused by it, but it's wrong either way.

Pass --dry-run to preview without mutating anything. A real run first writes an undo file
(album_id -> removed asset ids) so the removals can be replayed back in if the scope turns
out to be wrong -- much of the Furry Stuff overlap is long-standing manual curation rather
than anything the captioner did, so this is deliberately reversible.
"""
import json
import os
import sys
import time

sys.path.insert(0, "/app")
from captioner import IMMICH_URL, immich_headers, immich_remove_from_album, SLEEP_SECONDS  # noqa: E402
import requests

FERAL_ALBUM_IDS = {
    "Feral on Human Video": "46675d67-bcc5-4375-b998-27a976204dcb",
    "Feral on Human Animation": "564d0896-7f0e-4b3b-9cd7-0e32d62f3392",
    "Feral on Human Art": "6897fe81-d85a-480d-aeee-bf11c7813b80",
    "Feral on Feral Animation": "1c78f627-138b-4b60-a0f7-4816125476d0",
}
REMOVE_FROM = {
    "Single Creampie": "3a22144e-143c-4f43-a508-8b3f7fadbcb5",
    "Furry Stuff": "b135f926-dd5b-4230-aa05-32bbdb2cf315",
}


def get_album_asset_ids(album_id: str):
    ids = set()
    page = 1
    while True:
        r = requests.post(
            f"{IMMICH_URL}/api/search/metadata",
            headers={**immich_headers(), "Content-Type": "application/json"},
            json={"albumIds": [album_id], "page": page, "size": 500},
            timeout=60,
        )
        r.raise_for_status()
        data = r.json().get("assets", {})
        items = data.get("items", [])
        if not items:
            break
        ids.update(a["id"] for a in items)
        next_page_raw = data.get("nextPage")
        if next_page_raw is None:
            break
        page = int(next_page_raw)
        time.sleep(SLEEP_SECONDS)
    return ids


def main():
    dry_run = "--dry-run" in sys.argv

    feral_ids = set()
    for name, aid in FERAL_ALBUM_IDS.items():
        ids = get_album_asset_ids(aid)
        print(f"[info] {name}: {len(ids)} assets", flush=True)
        feral_ids |= ids
    print(f"[info] {len(feral_ids)} unique feral assets total", flush=True)

    plan = {}
    for name, album_id in REMOVE_FROM.items():
        target_ids = get_album_asset_ids(album_id)
        overlap = sorted(feral_ids & target_ids)
        plan[name] = {"album_id": album_id, "asset_ids": overlap}
        print(f"\n[info] {name}: {len(target_ids)} assets, feral overlap: {len(overlap)}", flush=True)

    if not dry_run:
        undo_path = os.environ.get("FERAL_CLEANUP_UNDO", "/undo/feral_cleanup_undo.json")
        try:
            os.makedirs(os.path.dirname(undo_path), exist_ok=True)
            with open(undo_path, "w") as fh:
                json.dump(plan, fh, indent=2)
            print(f"\n[undo] wrote undo list to {undo_path}", flush=True)
        except OSError as e:
            # Don't mutate anything if the safety net couldn't be written.
            print(f"[fatal] could not write undo file ({e}) -- aborting without changes", flush=True)
            return

    total_removed = 0
    for name, entry in plan.items():
        album_id = entry["album_id"]
        for asset_id in entry["asset_ids"]:
            if dry_run:
                print(f"[dry-run] would remove {asset_id} from {name}", flush=True)
                continue
            immich_remove_from_album(asset_id, album_id)
            print(f"[ok] removed {asset_id} from {name}", flush=True)
            time.sleep(SLEEP_SECONDS)
        total_removed += len(entry["asset_ids"])

    verb = "would remove" if dry_run else "removed"
    print(f"\n[done] {verb} {total_removed} feral memberships", flush=True)


if __name__ == "__main__":
    main()
