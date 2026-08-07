#!/usr/bin/env python3
"""One-off: Multiple Creampie is meant to mean "one woman, multiple creampies in a single
continuous encounter" -- it's not a looser bucket for compilations (different women cut
together), feral content (assumed single per policy), or orgies (generally multiple
separate one-man-one-woman couples, not one woman with multiple men). Residual overlap from
before those category boundaries were established gets removed here; add other albums to
EXCLUDE_ALBUMS for future one-off cleanups of the same kind."""
import sys
import time

sys.path.insert(0, "/app")
from captioner import IMMICH_URL, immich_headers, immich_remove_from_album, SLEEP_SECONDS  # noqa: E402
import requests

MULTIPLE_ALBUM_ID = "e7479905-44b5-42ca-86d0-aaf8fb7c36e3"
EXCLUDE_ALBUMS = {
    "Creampie Compilation": "090261d8-dd49-48a0-8c48-9be74f67ee90",
    "Feral on Human Video": "46675d67-bcc5-4375-b998-27a976204dcb",
    "Feral on Human Animation": "564d0896-7f0e-4b3b-9cd7-0e32d62f3392",
    "Orgy": "63f5733b-d244-4f99-a42b-db59e5ef6d15",
}


def get_album_asset_ids(album_id: str):
    ids = []
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
        ids.extend(a["id"] for a in items)
        next_page_raw = data.get("nextPage")
        if next_page_raw is None:
            break
        page = int(next_page_raw)
        time.sleep(SLEEP_SECONDS)
    return set(ids)


def main():
    multiple_ids = get_album_asset_ids(MULTIPLE_ALBUM_ID)
    print(f"[info] Multiple Creampie: {len(multiple_ids)} assets", flush=True)

    total_removed = 0
    for name, album_id in EXCLUDE_ALBUMS.items():
        excl_ids = get_album_asset_ids(album_id)
        overlap = multiple_ids & excl_ids
        print(f"[info] {name}: {len(excl_ids)} assets, overlap with Multiple Creampie: {len(overlap)}", flush=True)
        for asset_id in overlap:
            immich_remove_from_album(asset_id, MULTIPLE_ALBUM_ID)
            print(f"[ok] removed {asset_id} from Multiple Creampie (also in {name})", flush=True)
            time.sleep(SLEEP_SECONDS)
        total_removed += len(overlap)

    print(f"[done] removed {total_removed} category-overlap assets from Multiple Creampie", flush=True)


if __name__ == "__main__":
    main()
