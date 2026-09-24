#!/usr/bin/env python3
"""One-off: add every animated entry in Furry Stuff (VIDEO-type assets, plus GIFs) to
Anthro Video, if not already a member.

"Animated" = type VIDEO (mp4, webm, ...) or type IMAGE with a .gif extension. Checked:
none of the WEBP/PNG images in Furry Stuff have a nonzero Immich-reported animation
duration, so they're static and excluded.

Pass --dry-run to preview. A real run writes an undo file first and aborts if it can't.
"""
import json
import os
import sys
import time

sys.path.insert(0, "/app")
from captioner import IMMICH_URL, immich_headers, immich_add_to_album, SLEEP_SECONDS  # noqa: E402
import requests

ANTHRO_VIDEO = ("Anthro Video", "9d09367b-1416-4488-996d-6f4caca26ae1")
FURRY_STUFF = ("Furry Stuff", "b135f926-dd5b-4230-aa05-32bbdb2cf315")


def album_assets(album_id):
    out = []
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
        out.extend(items)
        nxt = data.get("nextPage")
        if nxt is None:
            break
        page = int(nxt)
        time.sleep(SLEEP_SECONDS)
    return out


def is_animated(asset):
    if asset.get("type") == "VIDEO":
        return True
    name = asset.get("originalFileName", "")
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    return asset.get("type") == "IMAGE" and ext == "gif"


def main():
    dry_run = "--dry-run" in sys.argv

    furry = album_assets(FURRY_STUFF[1])
    animated = [a for a in furry if is_animated(a)]
    anthro_ids = {a["id"] for a in album_assets(ANTHRO_VIDEO[1])}

    to_add = [a["id"] for a in animated if a["id"] not in anthro_ids]
    print(f"[info] Furry Stuff: {len(furry)} total, {len(animated)} animated, "
          f"{len(anthro_ids)} already in Anthro Video, {len(to_add)} to add", flush=True)

    if dry_run:
        for aid in to_add:
            print(f"[dry-run] would add {aid} to Anthro Video", flush=True)
        return

    undo_path = os.environ.get("CLEANUP_UNDO", "/undo/furry_animated_to_anthro_video_undo.json")
    try:
        os.makedirs(os.path.dirname(undo_path), exist_ok=True)
        with open(undo_path, "w") as fh:
            json.dump({"additions": [{"asset_id": a, "album": ANTHRO_VIDEO[0], "album_id": ANTHRO_VIDEO[1]}
                                      for a in to_add]}, fh, indent=2)
        print(f"[undo] wrote undo list to {undo_path}", flush=True)
    except OSError as e:
        print(f"[fatal] could not write undo file ({e}) -- aborting without changes", flush=True)
        return

    for aid in to_add:
        immich_add_to_album(aid, ANTHRO_VIDEO[1])
        print(f"[ok] added {aid} to Anthro Video", flush=True)
        time.sleep(SLEEP_SECONDS)

    print(f"\n[done] {len(to_add)} additions", flush=True)


if __name__ == "__main__":
    main()
