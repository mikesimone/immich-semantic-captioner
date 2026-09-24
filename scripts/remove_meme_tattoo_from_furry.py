#!/usr/bin/env python3
"""One-off: remove anything from Furry Stuff that also belongs to a "meme" or "tattoo"
album (misfiled there off incidental caption words).

Matched by:
  - album name starts with "020." -- the entire meme numbering range (020.000-020.999,
    including sub-numbered albums like 020.002.001), not just names containing "meme"
  - album name contains "tattoo" (case-insensitive)

Pass --dry-run to preview. A real run writes an undo file first and aborts if it can't.
"""
import json
import os
import sys
import time

sys.path.insert(0, "/app")
from captioner import IMMICH_URL, immich_headers, immich_remove_from_album, immich_list_albums, SLEEP_SECONDS  # noqa: E402
import requests

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


def main():
    dry_run = "--dry-run" in sys.argv

    albums = immich_list_albums()
    matches = [a for a in albums
               if a.get("albumName", "").strip().startswith("020.")
               or "tattoo" in a.get("albumName", "").lower()]
    print(f"[info] matched {len(matches)} meme/tattoo albums:", flush=True)
    for a in matches:
        print(f"       {a['id']}  {a.get('albumName')}", flush=True)

    contaminated = {}  # asset_id -> set of album names it was found in
    for a in matches:
        for asset in album_assets(a["id"]):
            contaminated.setdefault(asset["id"], set()).add(a.get("albumName"))

    furry = {asset["id"] for asset in album_assets(FURRY_STUFF[1])}
    to_remove = sorted(furry & contaminated.keys())
    print(f"[info] Furry Stuff: {len(furry)} members, {len(to_remove)} also in a meme/tattoo album", flush=True)

    if dry_run:
        for aid in to_remove:
            print(f"[dry-run] would remove {aid} from Furry Stuff (also in {sorted(contaminated[aid])})", flush=True)
        return

    undo_path = os.environ.get("CLEANUP_UNDO", "/undo/remove_meme_tattoo_from_furry_undo_2.json")
    try:
        os.makedirs(os.path.dirname(undo_path), exist_ok=True)
        with open(undo_path, "w") as fh:
            json.dump({"removals": [{"asset_id": a, "album": FURRY_STUFF[0], "album_id": FURRY_STUFF[1],
                                      "also_in": sorted(contaminated[a])} for a in to_remove]}, fh, indent=2)
        print(f"[undo] wrote undo list to {undo_path}", flush=True)
    except OSError as e:
        print(f"[fatal] could not write undo file ({e}) -- aborting without changes", flush=True)
        return

    for aid in to_remove:
        immich_remove_from_album(aid, FURRY_STUFF[1])
        print(f"[ok] removed {aid} from Furry Stuff", flush=True)
        time.sleep(SLEEP_SECONDS)

    print(f"\n[done] {len(to_remove)} removals", flush=True)


if __name__ == "__main__":
    main()
