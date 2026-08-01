#!/usr/bin/env python3
"""One-off backfill: re-scan already-captioned assets' existing descriptions against the
same caption-text trigger regexes captioner.py applies to freshly-generated captions, and
file any matches into the corresponding album -- without regenerating captions.

Run inside the captioner container so it reuses the exact same regex/album-id definitions
(imported from captioner.py) rather than a second copy that could drift out of sync:

    docker exec -i immich_captioner python3 - < scripts/backfill_album_triggers.py

Add TRIGGERS entries below for future "add existing X to album Y" backfills.
"""
import sys
import time

sys.path.insert(0, "/app")
from captioner import (  # noqa: E402
    IMMICH_URL,
    SLEEP_SECONDS,
    immich_add_to_album,
    immich_headers,
    _LACTATION_TRIGGER_RE,
    _HUCOW_TRIGGER_RE,
    LACTATION_ALBUM_ID,
    HUCOW_ALBUM_ID,
)
import requests

TRIGGERS = [
    ("Lactation", _LACTATION_TRIGGER_RE, LACTATION_ALBUM_ID),
    ("Hucow", _HUCOW_TRIGGER_RE, HUCOW_ALBUM_ID),
]

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
        data = r.json()
        assets = data.get("assets", {})
        items = assets.get("items", [])
        if not items:
            return
        for item in items:
            yield item
        next_page_raw = assets.get("nextPage")
        if next_page_raw is None:
            return
        page = int(next_page_raw)
        time.sleep(SLEEP_SECONDS)


def main():
    dry_run = "--dry-run" in sys.argv
    scanned = 0
    matched_counts = {name: 0 for name, _, _ in TRIGGERS}

    for item in iter_captioned_assets():
        exif = item.get("exifInfo") or {}
        desc = exif.get("description") or ""
        if not desc.strip():
            continue
        scanned += 1

        asset_id = item.get("id")
        for name, trigger_re, album_id in TRIGGERS:
            if trigger_re.search(desc):
                matched_counts[name] += 1
                verb = "would add" if dry_run else "adding"
                print(f"[{name}] {asset_id} matched -> {verb} to album {album_id}: {desc[:120]!r}", flush=True)
                if not dry_run:
                    immich_add_to_album(asset_id, album_id)
                    time.sleep(SLEEP_SECONDS)

    print(f"[done] scanned {scanned} captioned assets", flush=True)
    for name, count in matched_counts.items():
        print(f"[done] {name}: {count} matches filed", flush=True)


if __name__ == "__main__":
    main()
