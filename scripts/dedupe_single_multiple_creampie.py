#!/usr/bin/env python3
"""One-off: Single Creampie and Multiple Creampie are meant to be mutually exclusive, but
a miscounting bug (blowjobs/oral counted as creampies, plus classifier jitter counted as
separate rapid-fire events) caused a bunch of Single Creampie videos to also get filed into
Multiple Creampie. Remove the overlap from Multiple Creampie, keeping Single Creampie as
the source of truth (does not touch anything else about the asset)."""
import sys
import time

sys.path.insert(0, "/app")
from captioner import IMMICH_URL, immich_headers, immich_remove_from_album, SLEEP_SECONDS  # noqa: E402
import requests

SINGLE_ALBUM_ID = "3a22144e-143c-4f43-a508-8b3f7fadbcb5"
MULTIPLE_ALBUM_ID = "e7479905-44b5-42ca-86d0-aaf8fb7c36e3"


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
    single_ids = get_album_asset_ids(SINGLE_ALBUM_ID)
    multiple_ids = get_album_asset_ids(MULTIPLE_ALBUM_ID)
    overlap = single_ids & multiple_ids
    print(f"[info] Single Creampie: {len(single_ids)} assets", flush=True)
    print(f"[info] Multiple Creampie: {len(multiple_ids)} assets", flush=True)
    print(f"[info] overlap: {len(overlap)} assets -- removing from Multiple Creampie", flush=True)

    for asset_id in overlap:
        immich_remove_from_album(asset_id, MULTIPLE_ALBUM_ID)
        print(f"[ok] removed {asset_id} from Multiple Creampie", flush=True)
        time.sleep(SLEEP_SECONDS)

    print(f"[done] removed {len(overlap)} overlapping assets from Multiple Creampie", flush=True)


if __name__ == "__main__":
    main()
