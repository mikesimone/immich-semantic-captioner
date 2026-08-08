#!/usr/bin/env python3
"""One-off: enforce the human/non-human category boundaries.

Rules being enforced:
  - Single Creampie, Multiple Creampie, Bondage Creampie are 100% human -- remove any
    Anthro Video or feral members from them.
  - Anthro Video is 100% non-human cartoon and belongs only in Anthro Video + Furry Stuff --
    so anthro members are also removed from the feral albums, and any anthro member missing
    from Furry Stuff is added to it.
  - Feral belongs in its feral albums and nothing else (handled by
    dedupe_feral_categories.py; re-checked here).

Deliberately keyed off Anthro Video + feral album membership rather than Furry Stuff
membership: Furry Stuff is contaminated with human videos the captioner auto-filed there off
incidental caption words, so it is NOT a reliable "this isn't human" signal (verified: 97 of
106 Furry-Stuff-tagged videos in Single Creampie were imported into the human Single Creampie
folder). Those human-in-Furry-Stuff entries are reported but NOT touched here.

Pass --dry-run to preview. A real run writes an undo file first and aborts if it can't.
"""
import json
import os
import re
import sys
import time

sys.path.insert(0, "/app")
from captioner import IMMICH_URL, immich_headers, immich_remove_from_album, immich_add_to_album, SLEEP_SECONDS  # noqa: E402
import requests

ANTHRO_VIDEO = ("Anthro Video", "9d09367b-1416-4488-996d-6f4caca26ae1")
FURRY_STUFF = ("Furry Stuff", "b135f926-dd5b-4230-aa05-32bbdb2cf315")
FERAL_ALBUMS = {
    "Feral on Human Video": "46675d67-bcc5-4375-b998-27a976204dcb",
    "Feral on Human Animation": "564d0896-7f0e-4b3b-9cd7-0e32d62f3392",
    "Feral on Human Art": "6897fe81-d85a-480d-aeee-bf11c7813b80",
    "Feral on Feral Animation": "1c78f627-138b-4b60-a0f7-4816125476d0",
}
HUMAN_CREAMPIE_ALBUMS = {
    "Single Creampie": "3a22144e-143c-4f43-a508-8b3f7fadbcb5",
    "Multiple Creampie": "e7479905-44b5-42ca-86d0-aaf8fb7c36e3",
    "Bondage Creampie": "dbb4e052-b9c8-4901-969c-02c24c209efc",
}


def album_assets(album_id):
    out = {}
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
        for a in items:
            out[a["id"]] = a.get("originalPath", "")
        nxt = data.get("nextPage")
        if nxt is None:
            break
        page = int(nxt)
        time.sleep(SLEEP_SECONDS)
    return out


def main():
    dry_run = "--dry-run" in sys.argv

    anthro = set(album_assets(ANTHRO_VIDEO[1]))
    furry = album_assets(FURRY_STUFF[1])
    feral = set()
    for name, aid in FERAL_ALBUMS.items():
        feral |= set(album_assets(aid))
    nonhuman = anthro | feral
    print(f"[info] Anthro Video: {len(anthro)}  feral (all): {len(feral)}  Furry Stuff: {len(furry)}", flush=True)

    removals = []   # (asset_id, album_name, album_id)
    additions = []  # (asset_id, album_name, album_id)

    for name, aid in HUMAN_CREAMPIE_ALBUMS.items():
        members = set(album_assets(aid))
        bad = sorted(members & nonhuman)
        print(f"[info] {name}: {len(members)} members, {len(bad)} non-human to remove", flush=True)
        removals += [(a, name, aid) for a in bad]

    for name, aid in FERAL_ALBUMS.items():
        members = set(album_assets(aid))
        bad = sorted(members & anthro)
        if bad:
            print(f"[info] {name}: {len(bad)} anthro members to remove", flush=True)
        removals += [(a, name, aid) for a in bad]

    missing_furry = sorted(anthro - set(furry))
    print(f"[info] Anthro Video members missing from Furry Stuff: {len(missing_furry)} to add", flush=True)
    additions += [(a, FURRY_STUFF[0], FURRY_STUFF[1]) for a in missing_furry]

    # Report-only: human content sitting in Furry Stuff (not touched -- see module docstring).
    human_in_furry = [
        a for a, path in furry.items()
        if a not in nonhuman and re.search(r"/library/admin/200", path or "")
    ]
    print(f"\n[report] {len(human_in_furry)} assets in Furry Stuff were imported under a human "
          f"200.x folder and are not anthro/feral -- NOT touched by this script", flush=True)

    print(f"\n[plan] {len(removals)} removals, {len(additions)} additions", flush=True)
    if dry_run:
        for a, name, _ in removals:
            print(f"[dry-run] would remove {a} from {name}", flush=True)
        for a, name, _ in additions:
            print(f"[dry-run] would add {a} to {name}", flush=True)
        return

    undo_path = os.environ.get("CLEANUP_UNDO", "/undo/human_creampie_cleanup_undo.json")
    try:
        os.makedirs(os.path.dirname(undo_path), exist_ok=True)
        with open(undo_path, "w") as fh:
            json.dump({
                "removals": [{"asset_id": a, "album": n, "album_id": i} for a, n, i in removals],
                "additions": [{"asset_id": a, "album": n, "album_id": i} for a, n, i in additions],
                "report_human_in_furry": human_in_furry,
            }, fh, indent=2)
        print(f"[undo] wrote undo list to {undo_path}", flush=True)
    except OSError as e:
        print(f"[fatal] could not write undo file ({e}) -- aborting without changes", flush=True)
        return

    for a, name, aid in removals:
        immich_remove_from_album(a, aid)
        print(f"[ok] removed {a} from {name}", flush=True)
        time.sleep(SLEEP_SECONDS)
    for a, name, aid in additions:
        immich_add_to_album(a, aid)
        print(f"[ok] added {a} to {name}", flush=True)
        time.sleep(SLEEP_SECONDS)

    print(f"\n[done] {len(removals)} removals, {len(additions)} additions", flush=True)


if __name__ == "__main__":
    main()
