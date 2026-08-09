#!/usr/bin/env python3
"""One-off: everything in the 200.000.00x studio/kink sub-albums is also a multiple creampie.

Actions:
  1. Merge Czech Gloryhole (200.000.007) into Czech Fantasy (200.000.006) -- same thing --
     then delete the now-redundant Gloryhole album (assets are NOT deleted, only the album).
  2. Add every asset from the 200.000.00x sub-albums into 200.000.000 - Multiple Creampie,
     leaving sub-album membership intact.
  3. Archive them, and clear their descriptions so they get re-captioned through the
     multiple-creampie counting path.
  4. Renumber the tail of the series to close the gap left by the merge:
     200.000.008 - Orgy     -> 200.000.007 - Orgy
     200.000.009 - Hentaied -> 200.000.008 - Hentaied

Pass --dry-run to preview. A real run writes an undo file first and aborts if it can't.
"""
import json
import os
import sys
import time

sys.path.insert(0, "/app")
from captioner import IMMICH_URL, immich_headers, immich_add_to_album, SLEEP_SECONDS  # noqa: E402
import requests

MULTIPLE = ("200.000.000 - Multiple Creampie", "e7479905-44b5-42ca-86d0-aaf8fb7c36e3")
CZECH_FANTASY = ("200.000.006 - Czech Fantasy", "c35650ec-1f8b-4bd1-a259-1dc4bedf8c58")
CZECH_GLORYHOLE = ("200.000.007 - Czech Gloryhole", "bb166e0e-1f1b-4be2-8261-b19329c3a460")
SUBALBUMS = {
    "200.000.001 - Puta Locura": "19fb81d6-4f60-4119-b932-618bc0dddddd",
    "200.000.002 - Creampie Squad": "1eb69465-6eed-4d7a-a544-4e04c56c8b1a",
    "200.000.003 - Gangbang Creampie": "21cef040-b4cf-4063-adf3-ceab39487f50",
    "200.000.004 - Slutwife Jessica": "4a51919f-6d90-4171-95d7-a749c2e5564e",
    "200.000.005 - Slutwife Marion": "abcf0fed-f9a9-467b-a2ea-cfac6a9b7982",
    CZECH_FANTASY[0]: CZECH_FANTASY[1],
    CZECH_GLORYHOLE[0]: CZECH_GLORYHOLE[1],
    "200.000.008 - Orgy": "63f5733b-d244-4f99-a42b-db59e5ef6d15",
    "200.000.009 - Hentaied": "44977bc4-3540-4df4-bf38-47d4c7ca7d70",
}
RENAMES = [
    ("63f5733b-d244-4f99-a42b-db59e5ef6d15", "200.000.007 - Orgy"),
    ("44977bc4-3540-4df4-bf38-47d4c7ca7d70", "200.000.008 - Hentaied"),
]
CHUNK = 200


def album_asset_ids(album_id):
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
        nxt = data.get("nextPage")
        if nxt is None:
            break
        page = int(nxt)
        time.sleep(SLEEP_SECONDS)
    return ids


def bulk_put(ids, payload, label):
    for i in range(0, len(ids), CHUNK):
        chunk = ids[i:i + CHUNK]
        r = requests.put(
            f"{IMMICH_URL}/api/assets",
            headers={**immich_headers(), "Content-Type": "application/json"},
            json={"ids": chunk, **payload},
            timeout=60,
        )
        if r.status_code >= 300:
            print(f"[error] {label} chunk {i}: {r.status_code} {r.text}", flush=True)
        else:
            print(f"[ok] {label} {i + len(chunk)}/{len(ids)}", flush=True)
        time.sleep(SLEEP_SECONDS)


def main():
    dry_run = "--dry-run" in sys.argv

    per_album = {name: album_asset_ids(aid) for name, aid in SUBALBUMS.items()}
    for name, ids in per_album.items():
        print(f"[info] {name}: {len(ids)} assets", flush=True)

    gloryhole = per_album[CZECH_GLORYHOLE[0]]
    already_multi = set(album_asset_ids(MULTIPLE[1]))
    all_sub = sorted({a for ids in per_album.values() for a in ids})
    to_add = [a for a in all_sub if a not in already_multi]

    print(f"\n[info] {len(all_sub)} unique assets across the sub-albums", flush=True)
    print(f"[info] {len(to_add)} not yet in Multiple Creampie -> will be added", flush=True)
    print(f"[info] {len(gloryhole)} Czech Gloryhole assets -> merge into Czech Fantasy, then delete that album", flush=True)
    print(f"[info] renames: " + "; ".join(f"{n}" for _, n in RENAMES), flush=True)

    if dry_run:
        print("\n[dry-run] no changes made")
        return

    undo_path = os.environ.get("CLEANUP_UNDO", "/undo/multi_consolidation_undo.json")
    try:
        os.makedirs(os.path.dirname(undo_path), exist_ok=True)
        with open(undo_path, "w") as fh:
            json.dump({"per_album": per_album, "added_to_multiple": to_add,
                       "gloryhole_assets": gloryhole}, fh, indent=2)
        print(f"[undo] wrote undo list to {undo_path}", flush=True)
    except OSError as e:
        print(f"[fatal] could not write undo file ({e}) -- aborting without changes", flush=True)
        return

    print("\n[step] merging Czech Gloryhole into Czech Fantasy", flush=True)
    for a in gloryhole:
        immich_add_to_album(a, CZECH_FANTASY[1])
        time.sleep(SLEEP_SECONDS)
    print(f"[ok] merged {len(gloryhole)} assets", flush=True)

    print("\n[step] adding sub-album assets to Multiple Creampie", flush=True)
    for a in to_add:
        immich_add_to_album(a, MULTIPLE[1])
        time.sleep(SLEEP_SECONDS)
    print(f"[ok] added {len(to_add)}", flush=True)

    print("\n[step] archiving + clearing descriptions so they re-caption as multi", flush=True)
    bulk_put(all_sub, {"visibility": "archive"}, "archived")
    bulk_put(all_sub, {"description": ""}, "cleared")

    print("\n[step] deleting the redundant Czech Gloryhole album", flush=True)
    r = requests.delete(f"{IMMICH_URL}/api/albums/{CZECH_GLORYHOLE[1]}",
                        headers=immich_headers(), timeout=30)
    print(f"[{'ok' if r.status_code < 300 else 'error'}] delete album -> HTTP {r.status_code}", flush=True)

    print("\n[step] renumbering the tail of the series", flush=True)
    for album_id, new_name in RENAMES:
        r = requests.patch(
            f"{IMMICH_URL}/api/albums/{album_id}",
            headers={**immich_headers(), "Content-Type": "application/json"},
            json={"albumName": new_name}, timeout=30,
        )
        print(f"[{'ok' if r.status_code < 300 else 'error'}] -> {new_name} (HTTP {r.status_code})", flush=True)

    print(f"\n[done] consolidated {len(all_sub)} assets into Multiple Creampie", flush=True)


if __name__ == "__main__":
    main()
