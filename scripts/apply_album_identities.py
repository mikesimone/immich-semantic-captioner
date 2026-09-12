#!/usr/bin/env python3
"""Backfill identity names into captions using ALBUM membership as the only source.

The gap this closes
-------------------
There were two ways an identity name got into a caption, and both leave holes:

  * captioner.py's album-based injection, which only runs when the asset is captioned. An
    asset captioned BEFORE it was filed into the identity album -- or before that album was
    added to IDENTITY_ALBUM_MAP -- keeps its anonymous caption forever, because a non-empty
    description means "already captioned" and it is never looked at again.
  * apply_people_names.py, which is sourced from Immich's People tags. That cannot help an
    asset with no face tag, and for drawn anthro content most assets have no detectable
    face at all (see scripts/tag_album_people_faces.py) -- 64 assets in
    "300.000.006 - Lydia Dog" had neither a face tag nor the name in their caption, so
    nothing in the system would ever have fixed them.

This script closes that hole from the album side: every asset in an identity-mapped album
whose caption does not name the mapped identity gets the name applied.

Why this does not re-caption
----------------------------
apply_people_names.py clears the description and lets the GPU regenerate, because a NEW
name arriving from face recognition changes what the caption should say throughout. Here
nothing about the image has changed -- the caption is already accurate, it just says
"an anthropomorphic dog girl" where it should say "LydiaDog". That is exactly the rewrite
apply_identity_overrides() performs as a pure text substitution ("a/an/the <noun>" -> the
name, falling back to an IDENTITY_ENSURE_MODE prefix), so this runs the existing logic over
the stored caption instead of burning hours of video-card time regenerating prose that was
already fine.

It also has the side effect of making these assets invisible to apply_people_names.py: once
the name is present, that script's idempotency check skips them, so the hourly job stops
wanting to re-caption the ones that do carry a face tag.

Generation info is split off and re-attached byte-for-byte ([[GEN_INFO_SEPARATOR]]), and the
misfile cleanup is deliberately NOT applied -- this script only ever edits caption text, it
never removes an asset from an album.

Run inside the captioner container:

    docker exec -i immich_captioner python3 - --dry-run < scripts/apply_album_identities.py
    docker exec -i immich_captioner python3 - < scripts/apply_album_identities.py

Options:
    --album NAME   Restrict to one album (repeatable). Default: every album that matches a
                   configured identity.
    --dry-run      Print the before/after for each change, write nothing.
"""
import argparse
import re
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
    immich_apply_tags,
    split_description,
    compose_description,
    apply_identity_overrides,
    extract_identities_from_albums,
    is_identity_authoritative_album,
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


def name_present(name: str, text: str) -> bool:
    return re.search(rf"\b{re.escape(name)}\b", text, re.IGNORECASE) is not None


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
        albums = [a for a in albums
                  if extract_identities_from_albums([a.get("albumName") or ""])]

    print(f"[scan] {len(albums)} identity album(s)\n", flush=True)

    total_fixed = 0
    total_skipped_no_person = 0
    seen = set()

    for album in albums:
        aname = album.get("albumName") or ""
        identities = extract_identities_from_albums([aname])
        authoritative = is_identity_authoritative_album([aname])
        assets = album_assets(album["id"])
        fixed = 0
        already = 0
        no_person = 0
        blank = 0

        for asset in assets:
            asset_id = asset["id"]
            if asset_id in seen:
                continue
            seen.add(asset_id)

            caption, gen = split_description((asset.get("exifInfo") or {}).get("description"))
            if not caption:
                # Uncaptioned (or generation-info only) -- the main captioner still owns
                # this one and will inject the name itself when it gets to it.
                blank += 1
                continue
            if all(name_present(n, caption) for n in identities):
                already += 1
                continue

            updated, implied_tags, misfiled = apply_identity_overrides(caption, [aname])
            if misfiled or updated == caption:
                # apply_identity_overrides found no person to attach the name to and this
                # album is not authoritative, so it declined to inject. Report rather than
                # force it: forcing a name onto a caption with nobody in it is how you get
                # "LydiaDog: a screenshot of a menu".
                no_person += 1
                print(f"[no-person] {asset.get('originalFileName', asset_id)}: "
                      f"{caption[:100]}", flush=True)
                continue

            fixed += 1
            verb = "would fix" if dry else "fixing"
            print(f"[{verb}] {asset.get('originalFileName', asset_id)}\n"
                  f"  before: {caption[:110]}\n  after : {updated[:110]}", flush=True)
            if not dry:
                if immich_update_description(asset_id, compose_description(updated, gen)):
                    if implied_tags:
                        immich_apply_tags(asset_id, implied_tags)
                time.sleep(SLEEP_SECONDS)

        total_fixed += fixed
        total_skipped_no_person += no_person
        flag = " [authoritative]" if authoritative else ""
        print(f"[album] {aname!r}{flag}: {len(assets)} asset(s) -- "
              f"fixed={fixed} already_named={already} no_person={no_person} "
              f"uncaptioned={blank}\n", flush=True)

    print(f"[done] {'would fix' if dry else 'fixed'}={total_fixed} "
          f"no_person={total_skipped_no_person} (dry_run={dry})", flush=True)


if __name__ == "__main__":
    main()
