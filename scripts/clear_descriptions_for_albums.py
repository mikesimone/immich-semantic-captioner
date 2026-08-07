#!/usr/bin/env python3
"""One-off: clear the description on every asset in every album whose name matches a
given regex, so the captioner picks them back up as uncaptioned candidates and
regenerates fresh captions (e.g. after a captioning-format change).

Run inside the captioner container:

    docker exec -i immich_captioner python3 - '^200\.' < scripts/clear_descriptions_for_albums.py

Optionally pass an asset type filter (VIDEO or IMAGE) as a second argument to only clear
that type -- e.g. to redo videos after a video-captioning fix without touching images that
weren't affected:

    docker exec -i immich_captioner python3 - '^200\.' VIDEO < scripts/clear_descriptions_for_albums.py
"""
import re
import sys
import time

sys.path.insert(0, "/app")
from captioner import IMMICH_URL, immich_headers, immich_list_albums, SLEEP_SECONDS  # noqa: E402
import requests

CHUNK_SIZE = 200


def get_album_asset_ids(album_id: str, asset_type: str = None):
    ids = []
    page = 1
    while True:
        body = {"albumIds": [album_id], "page": page, "size": 500}
        if asset_type:
            body["type"] = asset_type
        r = requests.post(
            f"{IMMICH_URL}/api/search/metadata",
            headers={**immich_headers(), "Content-Type": "application/json"},
            json=body,
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
    return ids


def clear_descriptions(asset_ids):
    ids = list(asset_ids)
    for i in range(0, len(ids), CHUNK_SIZE):
        chunk = ids[i:i + CHUNK_SIZE]
        r = requests.put(
            f"{IMMICH_URL}/api/assets",
            headers={**immich_headers(), "Content-Type": "application/json"},
            json={"ids": chunk, "description": ""},
            timeout=60,
        )
        if r.status_code >= 300:
            print(f"[error] clear chunk {i}-{i+len(chunk)} failed {r.status_code}: {r.text}", flush=True)
        else:
            print(f"[ok] cleared {len(chunk)} descriptions ({i + len(chunk)}/{len(ids)})", flush=True)
        time.sleep(SLEEP_SECONDS)


def main():
    pattern = re.compile(sys.argv[1]) if len(sys.argv) > 1 else re.compile(r"^200\.")
    asset_type = sys.argv[2] if len(sys.argv) > 2 else None
    albums = [a for a in immich_list_albums() if pattern.search(a.get("albumName", ""))]
    print(f"[match] {len(albums)} albums match {pattern.pattern!r} (type={asset_type}):", flush=True)
    for a in albums:
        print(f"  {a['id']} {a['albumName']} ({a.get('assetCount')})", flush=True)

    all_ids = set()
    for a in albums:
        ids = get_album_asset_ids(a["id"], asset_type)
        print(f"[fetch] {a['albumName']}: {len(ids)} assets", flush=True)
        all_ids.update(ids)

    print(f"[total] {len(all_ids)} unique assets across {len(albums)} albums", flush=True)
    clear_descriptions(all_ids)
    print(f"[done] cleared {len(all_ids)} descriptions", flush=True)


if __name__ == "__main__":
    main()
