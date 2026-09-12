#!/usr/bin/env python3
"""One-off repair for captions that name the same person twice over.

apply_identity_overrides() rewrites a generic noun phrase into the person's name ("a dog"
-> "LydiaDog"). When the model had ALREADY used the name -- which the prompt explicitly
tells it to do -- and then described the subject generically anyway, that substitution left
the name doubled, in two shapes:

    "Illustration of LydiaDog, a dog, running forward."
        -> "Illustration of LydiaDog, LydiaDog, running forward."
    "An anthropomorphic dog character named LydiaDog, drawn in a digital medium."
        -> "LydiaDog character named LydiaDog, drawn in a digital medium."

captioner.py now collapses both at caption time via collapse_name_redundancy(), which this
script reuses so the repair and the live path can never drift apart. It fixes captions
written before that, in place, with no GPU work: the repair is a pure text edit, so
re-captioning these assets would waste video-card time and churn captions that are
otherwise perfectly good.

Only adjacent repeats are collapsed, so a caption that legitimately mentions the person
again in a later sentence is untouched. Generation info in the description is split off and
re-attached byte-for-byte.

Run inside the captioner container (needs the updated captioner.py -- restart the
container first, or the import of split_description will fail):

    docker exec -i immich_captioner python3 - --dry-run < scripts/fix_identity_name_stutter.py
    docker exec -i immich_captioner python3 - < scripts/fix_identity_name_stutter.py

Options:
    --album NAME   Restrict to one album (repeatable). Default: every album that matches a
                   configured identity.
    --dry-run      Print the before/after for each change, write nothing.
"""
import argparse
import sys
import time
from typing import List

sys.path.insert(0, "/app")
from captioner import (  # noqa: E402
    IMMICH_URL,
    SLEEP_SECONDS,
    DRY_RUN as CAPTIONER_DRY_RUN,
    immich_headers,
    immich_update_description,
    split_description,
    compose_description,
    collapse_name_redundancy,
    extract_identities_from_albums,
    _IDENTITY_MAP,
)
import requests  # noqa: E402


def _api(method: str, path: str, **kw):
    r = requests.request(method, f"{IMMICH_URL}{path}",
                         headers={**immich_headers(), "Content-Type": "application/json"},
                         timeout=60, **kw)
    r.raise_for_status()
    return r.json() if r.content else None


def album_assets(album_id: str) -> List[dict]:
    out: List[dict] = []
    page = 1
    while True:
        data = _api("POST", "/api/search/metadata",
                    json={"albumIds": [album_id], "withExif": True, "page": page, "size": 250})
        assets = data.get("assets", {})
        items = assets.get("items", [])
        if not items:
            break
        out.extend(items)
        nxt = assets.get("nextPage")
        if nxt is None:
            break
        page = int(nxt)
        time.sleep(SLEEP_SECONDS)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--album", action="append", default=[])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    dry = args.dry_run or CAPTIONER_DRY_RUN

    albums = _api("GET", "/api/albums")
    if args.album:
        wanted = set(args.album)
        albums = [a for a in albums if a.get("albumName") in wanted]
        missing = wanted - {a.get("albumName") for a in albums}
        if missing:
            raise SystemExit(f"[fatal] no such album(s): {sorted(missing)}")
    else:
        # Only albums that actually map to an identity can have produced a stutter.
        albums = [a for a in albums
                  if extract_identities_from_albums([a.get("albumName") or ""])]

    print(f"[scan] {len(albums)} identity album(s), "
          f"{len(_IDENTITY_MAP)} configured identity token(s)", flush=True)

    fixed = 0
    seen = set()
    for album in albums:
        name = album.get("albumName") or ""
        identities = extract_identities_from_albums([name])
        assets = album_assets(album["id"])
        hits = 0
        for asset in assets:
            asset_id = asset["id"]
            if asset_id in seen:
                continue
            seen.add(asset_id)
            desc = (asset.get("exifInfo") or {}).get("description")
            caption, gen = split_description(desc)
            if not caption:
                continue
            repaired = collapse_name_redundancy(caption, identities)
            if repaired == caption:
                continue
            hits += 1
            fixed += 1
            verb = "would fix" if dry else "fixing"
            print(f"[{verb}] {asset.get('originalFileName', asset_id)}\n"
                  f"  before: {caption[:120]}\n  after : {repaired[:120]}", flush=True)
            if not dry:
                immich_update_description(asset_id, compose_description(repaired, gen))
                time.sleep(SLEEP_SECONDS)
        print(f"[album] {name!r}: {hits} stutter(s) in {len(assets)} asset(s)", flush=True)

    print(f"\n[done] {'would fix' if dry else 'fixed'}={fixed} (dry_run={dry})", flush=True)


if __name__ == "__main__":
    main()
