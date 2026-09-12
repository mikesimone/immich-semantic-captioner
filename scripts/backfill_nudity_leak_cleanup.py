#!/usr/bin/env python3
"""One-off backfill: re-scan already-captioned assets and strip JoyCaption's leaked
nudity/genital-status commentary from captions on non-adult-content assets (e.g. "his
nipples are small and light-colored" or "no visible cum or sexual activity" on an ordinary
clothed conference photo) -- without regenerating captions.

Reuses strip_false_nudity_leaks()/is_adult_album() from captioner.py so this stays in sync
with the same logic the live pipeline now applies to freshly-generated captions -- it just
re-applies that logic to descriptions written before the fix existed.

Run inside the captioner container (after rebuilding it with the strip_false_nudity_leaks
fix so this imports the patched version):

    docker exec -i immich_captioner python3 - < scripts/backfill_nudity_leak_cleanup.py

Options:

    docker exec -i immich_captioner python3 - --dry-run < scripts/backfill_nudity_leak_cleanup.py
"""
import sys
import time

sys.path.insert(0, "/app")
from captioner import (  # noqa: E402
    IMMICH_URL,
    SLEEP_SECONDS,
    immich_headers,
    immich_update_description,
    get_asset_albums,
    strip_false_nudity_leaks,
)
import requests

DRY_RUN = "--dry-run" in sys.argv[1:]
PAGE_SIZE = 250


def iter_captioned_assets():
    page = 1
    while True:
        r = requests.post(
            f"{IMMICH_URL}/api/search/metadata",
            headers={**immich_headers(), "Content-Type": "application/json"},
            json={"withExif": True, "page": page, "size": PAGE_SIZE},
            timeout=60,
        )
        r.raise_for_status()
        data = r.json().get("assets", {})
        items = data.get("items", [])
        if not items:
            return
        for item in items:
            yield item
        next_page_raw = data.get("nextPage")
        if next_page_raw is None:
            return
        page = int(next_page_raw)
        time.sleep(SLEEP_SECONDS)


def main():
    scanned = 0
    changed = 0
    skipped_adult_album = 0

    for item in iter_captioned_assets():
        exif = item.get("exifInfo") or {}
        desc = exif.get("description") or ""
        if not desc.strip():
            continue
        scanned += 1
        asset_id = item.get("id")

        albums = get_asset_albums(asset_id)
        updated = strip_false_nudity_leaks(desc, albums)
        if updated == desc:
            continue

        verb = "would update" if DRY_RUN else "updating"
        print(f"[{verb}] {asset_id} albums={albums}", flush=True)
        print(f"  before: {desc!r}", flush=True)
        print(f"  after:  {updated!r}", flush=True)
        if not DRY_RUN:
            immich_update_description(asset_id, updated)
            time.sleep(SLEEP_SECONDS)
        changed += 1

    print(f"[done] scanned={scanned} changed={changed} (dry_run={DRY_RUN})", flush=True)


if __name__ == "__main__":
    main()
