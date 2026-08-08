#!/usr/bin/env python3
"""One-off backfill: anything shot on Ray-Ban Meta glasses belongs in Camspy.

Keyed off EXIF make (default "Meta"), matching the rule the captioner now applies to newly
captioned assets. Existing Camspy members are left alone; this only adds the stragglers.

Pass --dry-run to preview.
"""
import sys
import time

sys.path.insert(0, "/app")
from captioner import (  # noqa: E402
    IMMICH_URL, immich_headers, immich_add_to_album, SLEEP_SECONDS,
    CAMSPY_ALBUM_ID, CAMSPY_EXIF_MAKE,
)
import requests


def search_ids(body):
    ids = {}
    page = 1
    while True:
        r = requests.post(
            f"{IMMICH_URL}/api/search/metadata",
            headers={**immich_headers(), "Content-Type": "application/json"},
            json={**body, "withExif": True, "page": page, "size": 500},
            timeout=60,
        )
        r.raise_for_status()
        data = r.json().get("assets", {})
        items = data.get("items", [])
        if not items:
            break
        for a in items:
            ids[a["id"]] = (a.get("exifInfo") or {}).get("model")
        nxt = data.get("nextPage")
        if nxt is None:
            break
        page = int(nxt)
        time.sleep(SLEEP_SECONDS)
    return ids


def main():
    dry_run = "--dry-run" in sys.argv
    # The search API matches `make` case-sensitively against the stored value.
    shot_on = search_ids({"make": CAMSPY_EXIF_MAKE.capitalize()})
    already = set(search_ids({"albumIds": [CAMSPY_ALBUM_ID]}))
    missing = sorted(set(shot_on) - already)
    print(f"[info] make={CAMSPY_EXIF_MAKE!r}: {len(shot_on)} assets library-wide", flush=True)
    print(f"[info] already in Camspy: {len(already & set(shot_on))}", flush=True)
    print(f"[info] to add: {len(missing)}", flush=True)

    for asset_id in missing:
        if dry_run:
            print(f"[dry-run] would add {asset_id} ({shot_on[asset_id]}) to Camspy", flush=True)
            continue
        immich_add_to_album(asset_id, CAMSPY_ALBUM_ID)
        print(f"[ok] added {asset_id} to Camspy", flush=True)
        time.sleep(SLEEP_SECONDS)

    verb = "would add" if dry_run else "added"
    print(f"[done] {verb} {len(missing)} assets to Camspy", flush=True)


if __name__ == "__main__":
    main()
