#!/usr/bin/env python3
"""Hourly reconciliation pass: for assets where Immich's own People (face-recognition)
feature has a named person not yet reflected in the caption, clear the caption so the
main captioner picks the asset back up as an uncaptioned candidate and regenerates it --
this time with the name fed into the prompt (captioner.py's caption_image()/caption_video()
now merge person_names from both album membership AND get_asset_people_names()), so the
model writes it naturally into the prose instead of a mechanical text patch after the fact.

This is deliberately separate from the album-based identity injection in captioner.py
(IDENTITY_ALBUM_MAP) -- that system depends on a human filing a photo into a specifically
named album at caption time, and only covers a fixed roster of names. Immich's People tags
come from automatic face recognition (plus a human naming the cluster afterward), often well
after the asset was already captioned, and cover ordinary photos (friends, coworkers) that
never go anywhere near an album. This pass just decides WHICH already-captioned assets need
a fresh pass; the actual (re)captioning happens in captioner.py's normal polling loop.

Sourced from the People side (GET /api/people, then /api/search/metadata?personIds=...)
rather than scanning every captioned asset in the library -- most assets have zero named
people, so this keeps each hourly run to a handful of targeted API calls instead of an N+1
GET over the whole library.

Idempotent: if a name is already present in the description, the asset is left alone. Once
the model successfully re-captions with the name included, the next hourly run sees it
already present and stops touching that asset. Safe to run every hour with no separate state
tracking.

Run inside the captioner container (reuses captioner.py's Immich helpers/config):

    docker exec -i immich_captioner python3 - < scripts/apply_people_names.py

Options:

    docker exec -i immich_captioner python3 - --dry-run < scripts/apply_people_names.py

  --dry-run   Print what would be cleared for re-captioning, without writing anything.
"""
import re
import sys
import time
from typing import Dict, List

sys.path.insert(0, "/app")
from captioner import (  # noqa: E402
    IMMICH_URL,
    SLEEP_SECONDS,
    DRY_RUN as CAPTIONER_DRY_RUN,
    immich_headers,
    immich_update_description,
    canonical_people_name,
    split_description,
    _PERSON_WORD_RE,
    _IDENTITY_HINTS,
)
import requests

DRY_RUN = CAPTIONER_DRY_RUN or "--dry-run" in sys.argv[1:]

PAGE_SIZE = 250

# Irregular plurals for the nouns IDENTITY_NOUN_HINTS deals in.
_PLURAL_FORM = {"man": "men", "woman": "women", "person": "people", "child": "children"}


def _hint_forms(hint: str) -> List[str]:
    return [hint, _PLURAL_FORM.get(hint, hint + "s")]


def _gender_plausible(name: str, caption: str) -> bool:
    """A name with a known IDENTITY_NOUN_HINTS entry (Lydia=woman/girl/..., Mike=man/guy/...)
    must have one of its hinted nouns appear somewhere in the EXISTING caption before we
    trigger a re-caption for it -- otherwise Immich's People tag disagrees with what the
    photo apparently shows (e.g. a "Lydia" face-match on a caption that only ever uses "he"/
    "his"/"the man" -- a likely face-rec misfire). Forcing captioner.py to re-caption with
    "always refer to them as Lydia" baked into the prompt would make a wrong tag WORSE than
    leaving the caption alone -- it'd bake the wrong name into freshly-generated prose instead
    of just leaving an unrelated mention out. Names with no hint entry (most real People
    names -- "Justin Hang" isn't gendered vocabulary) have no signal to check, so they're
    allowed through: Immich's tag is the only evidence we have for those.
    """
    hints = _IDENTITY_HINTS.get(name)
    if not hints:
        return True
    forms = [f for h in hints for f in _hint_forms(h)]
    return any(re.search(rf"\b{re.escape(f)}\b", caption, re.IGNORECASE) for f in forms)


def fetch_named_people() -> List[dict]:
    r = requests.get(
        f"{IMMICH_URL}/api/people",
        headers=immich_headers(),
        params={"withHidden": "true"},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    people = data.get("people", data if isinstance(data, list) else [])
    return [p for p in people if p.get("name")]


def iter_assets_for_person(person_id: str):
    page = 1
    while True:
        r = requests.post(
            f"{IMMICH_URL}/api/search/metadata",
            headers={**immich_headers(), "Content-Type": "application/json"},
            json={"personIds": [person_id], "withExif": True, "page": page, "size": PAGE_SIZE},
            timeout=60,
        )
        r.raise_for_status()
        data = r.json().get("assets", {})
        items = data.get("items", [])
        if not items:
            return
        for item in items:
            yield item
        next_page_raw = data.get("nextPage")
        if next_page_raw is None:
            return
        page = int(next_page_raw)
        time.sleep(SLEEP_SECONDS)


def name_present(name: str, text: str) -> bool:
    return re.search(rf"\b{re.escape(name)}\b", text, re.IGNORECASE) is not None


def needs_recaption(desc: str, names: List[str]) -> bool:
    missing = [n for n in names if not name_present(n, desc)]
    if not missing:
        return False
    if not _PERSON_WORD_RE.search(desc):
        # No pronoun or person-noun anywhere -- a non-prose stub (structured metadata field,
        # raw OCR) or a face-match on content with no one actually depicted (e.g. illustrated
        # art misidentified against a real person's face). Re-captioning won't fix a bad tag,
        # and would waste GPU time regenerating a caption that was never wrong to begin with.
        return False
    return all(_gender_plausible(n, desc) for n in missing)


def main():
    people = fetch_named_people()
    print(f"[people] {len(people)} named person(s) in Immich", flush=True)
    for p in people:
        print(f"  {p['id']} {p['name']!r} -> {canonical_people_name(p['name'])!r}", flush=True)

    asset_names: Dict[str, List[str]] = {}
    asset_desc: Dict[str, str] = {}
    asset_gen: Dict[str, str] = {}

    for p in people:
        cname = canonical_people_name(p["name"])
        count = 0
        for item in iter_assets_for_person(p["id"]):
            asset_id = item.get("id")
            # Only the caption slot is examined, never the generation-info slot. The
            # generation JSON embeds the raw render prompt, which for these assets contains
            # the character LoRA's trigger token ("lydiadog") -- reading the whole
            # description would see that as "the name is already in the caption" and skip an
            # asset that genuinely needs re-captioning.
            desc, gen = split_description((item.get("exifInfo") or {}).get("description"))
            if gen:
                asset_gen[asset_id] = gen
            if not desc.strip():
                continue
            asset_desc[asset_id] = desc
            names = asset_names.setdefault(asset_id, [])
            if cname not in names:
                names.append(cname)
            count += 1
        print(f"[scan] {p['name']!r}: {count} captioned asset(s)", flush=True)

    print(f"[scan] {len(asset_desc)} unique captioned asset(s) with a named person", flush=True)

    cleared = 0
    skipped = 0
    for asset_id, desc in asset_desc.items():
        names = asset_names[asset_id]
        if not needs_recaption(desc, names):
            skipped += 1
            continue

        # Clearing means "clear the caption", not "clear the description". Where the asset
        # carries generation info (render parameters that arrived in EXIF ImageDescription
        # and exist nowhere else), the description is reset to that JSON alone rather than
        # to an empty string -- which still reads as uncaptioned to the main captioner's
        # candidate scan, so the asset is picked back up exactly as before, minus the data
        # loss.
        gen = asset_gen.get(asset_id)
        replacement = gen or ""
        verb = "would clear" if DRY_RUN else "clearing"
        kept = " (keeping generation info)" if gen else ""
        print(f"[{verb}]{kept} {asset_id} names={names}\n  current: {desc[:160]!r}", flush=True)
        if not DRY_RUN:
            immich_update_description(asset_id, replacement)
            time.sleep(SLEEP_SECONDS)
        cleared += 1

    print(f"[done] cleared_for_recaption={cleared} skipped={skipped} (dry_run={DRY_RUN})", flush=True)


if __name__ == "__main__":
    main()
