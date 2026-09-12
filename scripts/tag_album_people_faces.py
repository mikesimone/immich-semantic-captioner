#!/usr/bin/env python3
"""Assign an Immich Person to the main character of every asset in an album, for albums
whose subjects the stock face detector refuses to see.

Why this exists
---------------
Immich can only attach a Person to an asset that has a FACE row; there is no "tag this
person on this photo" without one. For the anthro/furry albums that is a dead end out of
the box: face detection has already run to completion on all 373 assets of
"300.000.006 - Lydia Dog" and found a face on only 17, because buffalo_l scores a drawn
dog muzzle far below Immich's default minScore of 0.7. The other 356 cannot carry the
Person tag at all, so they never surface under that person.

The detector is not blind to these faces, just unconfident: re-running the SAME model
against the SAME image at minScore 0.05 finds the real face on most of them, and the box
lands within a few percent of where a human drew one by hand (measured: IoU 0.46-0.66
against the three hand-drawn boxes on this album). So this script re-detects at a low
threshold and writes the winner back as a `manual`-sourceType face, which is exactly what
the Immich UI's own "add a face" action creates.

Deliberately NOT done by lowering the server's facialRecognition.minScore: that setting is
global, and re-running detection library-wide at 0.05 would seed thousands of junk faces
across every real person's clusters. This touches one album and creates nothing elsewhere.

Telling a face from a cabinet
-----------------------------
A threshold that low returns plenty of non-faces, and they cannot be filtered on geometry
alone -- a drawer front on one of these assets came back as a 97x150 box, comfortably
larger than a genuine but tightly-cropped 109x144 face on another. Two signals are
combined instead, because measurement showed neither is sufficient alone:

  * detection score -- verified real faces scored 0.25-0.59, verified junk 0.05-0.15, but
    a real face cropped down to one eye and a muzzle scored only 0.09, so a plain
    score cutoff high enough to exclude junk also loses real faces.
  * embedding similarity to faces ALREADY confirmed as this person -- reference
    embeddings are rebuilt by re-detecting the assets a human already tagged and taking
    the box that best overlaps the hand-drawn one. Good faces scored +0.78 against that
    set, junk +0.39 to +0.57. Note buffalo_l is a human-face model, so its embeddings on
    stylised muzzles are only weakly discriminative (the three reference faces agree with
    each other at just +0.34 to +0.55) -- which is why this is a secondary signal used to
    rescue low-score boxes, never the primary one.

So: a box is accepted outright above --score-confident, accepted as PROBABLE if it clears
both --score-floor and --min-similarity, and otherwise rejected and reported for manual
tagging. Every PROBABLE decision is listed separately in the summary precisely because it
is the tier that can be wrong -- spot-check those, not the confident ones.

Where several boxes survive, the largest wins: in this content the male is usually
generated as "faceless male" (it is in the render prompts), so most assets have exactly
one real candidate, and where there are two the main character is the nearer and therefore
larger one.

Existing faces are respected. An asset already carrying a face for the target person is
left alone; an unassigned face, or one in an auto-generated UNNAMED cluster, is reassigned
rather than having a second face invented beside it. A face belonging to a DIFFERENT NAMED
person is never touched -- that would silently steal the photo from someone else's People
page -- it is reported instead.

Run inside the captioner container: the ML service is only reachable on the
immich_default docker network, not from the host.

    docker exec -i immich_captioner python3 - --album "300.000.006 - Lydia Dog" \
        --person "Lydia Dog" --probe --cache /tmp/lydiadog.json \
        < scripts/tag_album_people_faces.py

    docker exec -i immich_captioner python3 - --album "300.000.006 - Lydia Dog" \
        --person "Lydia Dog" --cache /tmp/lydiadog.json \
        < scripts/tag_album_people_faces.py

Options:
    --album NAME        Album to process (exact albumName). Required.
    --person NAME       Immich Person to assign. Must already exist. Required.
    --probe             Detect and report only; write nothing to Immich.
    --cache PATH        Persist detections here and reuse them on later runs. Detection is
                        the slow part (download + GPU per asset), so a --probe pass can be
                        reviewed and then applied without paying for it twice.
    --min-score F       Detector threshold (default 0.05).
    --score-confident F Accept any box at or above this outright (default 0.20).
    --score-floor F     Below this, reject regardless of similarity (default 0.05).
    --min-similarity F  Similarity to the reference faces needed to rescue a low-score
                        box (default 0.60).
    --limit N           Only process the first N assets.
    --dry-run           Print every decision, write nothing.
"""
import argparse
import io
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, "/app")
from captioner import (  # noqa: E402
    IMMICH_URL,
    SLEEP_SECONDS,
    DRY_RUN as CAPTIONER_DRY_RUN,
    immich_headers,
)
import requests  # noqa: E402

ML_URL = "http://immich-machine-learning:3003/predict"
ML_MODEL = "buffalo_l"

# Geometry filters for detection noise: (0,0,0,0) boxes, boxes hanging off the frame at
# negative coordinates, and specks a few pixels across. These are rejected on shape because
# at this threshold their scores are indistinguishable from a real faint detection.
MIN_BOX_FRACTION = 0.03   # box side >= 3% of the corresponding image dimension...
MIN_BOX_PIXELS = 20       # ...and at least this many pixels, for small images
MAX_OOB_FRACTION = 0.25   # tolerate slight clipping at the frame edge, not a mostly-off box


def _api(method: str, path: str, **kw):
    r = requests.request(method, f"{IMMICH_URL}{path}",
                         headers={**immich_headers(), "Content-Type": "application/json"},
                         timeout=60, **kw)
    r.raise_for_status()
    return r.json() if r.content else None


def find_album(name: str) -> dict:
    for a in _api("GET", "/api/albums"):
        if a.get("albumName") == name:
            return a
    raise SystemExit(f"[fatal] no album named {name!r}")


def find_person(name: str) -> dict:
    data = _api("GET", "/api/people", params={"withHidden": "true"})
    people = data.get("people", data if isinstance(data, list) else [])
    matches = [p for p in people if p.get("name") == name]
    if not matches:
        raise SystemExit(f"[fatal] no Immich Person named {name!r} -- create it first")
    if len(matches) > 1:
        raise SystemExit(f"[fatal] {len(matches)} people named {name!r}; disambiguate by hand")
    return matches[0]


def album_assets(album_id: str) -> List[dict]:
    """All assets in an album, via the paginated search API.

    GET /api/albums/{id} does NOT carry an assets array on this Immich version (v3.2.0,
    verified empirically -- the response has assetCount but no assets key at all), and the
    album controller exposes no album-assets route, so the metadata search's albumIds
    filter is the way to enumerate them.
    """
    out: List[dict] = []
    page = 1
    while True:
        data = _api("POST", "/api/search/metadata",
                    json={"albumIds": [album_id], "page": page, "size": 250})
        assets = data.get("assets", {})
        items = assets.get("items", [])
        if not items:
            break
        out.extend(items)
        next_page = assets.get("nextPage")
        if next_page is None:
            break
        # nextPage comes back as a string but "page" must be a number -- same quirk the
        # main captioner's API scan works around.
        page = int(next_page)
        time.sleep(SLEEP_SECONDS)
    return out


def asset_faces(asset_id: str) -> List[dict]:
    return _api("GET", "/api/faces", params={"id": asset_id}) or []


def detect(asset_id: str, filename: str, min_score: float) -> dict:
    """Re-detect one asset. Returns {'w','h','boxes':[{x1,y1,x2,y2,score,embedding}]}."""
    blob = requests.get(f"{IMMICH_URL}/api/assets/{asset_id}/original",
                        headers=immich_headers(), timeout=180).content
    entries = json.dumps({
        "facial-recognition": {
            "detection": {"modelName": ML_MODEL, "options": {"minScore": min_score}},
            "recognition": {"modelName": ML_MODEL},
        }
    })
    r = requests.post(
        ML_URL,
        data={"entries": entries},
        files={"image": (filename, io.BytesIO(blob), "application/octet-stream")},
        timeout=300,
    )
    r.raise_for_status()
    out = r.json()
    boxes = []
    for f in out.get("facial-recognition") or []:
        b = f.get("boundingBox") or {}
        emb = f.get("embedding")
        if isinstance(emb, str):
            try:
                emb = json.loads(emb)
            except ValueError:
                emb = None
        boxes.append({
            "x1": b.get("x1"), "y1": b.get("y1"), "x2": b.get("x2"), "y2": b.get("y2"),
            "score": f.get("score"), "embedding": emb,
        })
    return {"w": int(out.get("imageWidth") or 0), "h": int(out.get("imageHeight") or 0),
            "boxes": boxes}


def _cosine(a: List[float], b: List[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if not na or not nb:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def _iou(box: dict, ref: Tuple[float, float, float, float]) -> float:
    x1, y1 = max(box["x1"], ref[0]), max(box["y1"], ref[1])
    x2, y2 = min(box["x2"], ref[2]), min(box["y2"], ref[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])
    a2 = (ref[2] - ref[0]) * (ref[3] - ref[1])
    return inter / (a1 + a2 - inter) if (a1 + a2 - inter) else 0.0


def _sane(box: dict, w: int, h: int) -> Optional[Dict[str, int]]:
    """Clamp a box to the frame and reject detection noise. None = rejected."""
    try:
        x1, y1, x2, y2 = float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])
    except (KeyError, TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    bw, bh = x2 - x1, y2 - y1
    # A box mostly outside the frame is an artefact of the detector's padding, not a
    # subject standing at the edge.
    oob = max(0.0, -x1) + max(0.0, x2 - w) + max(0.0, -y1) + max(0.0, y2 - h)
    if oob > MAX_OOB_FRACTION * (bw + bh):
        return None
    cx1, cy1 = max(0.0, x1), max(0.0, y1)
    cx2, cy2 = min(float(w), x2), min(float(h), y2)
    cw, ch = cx2 - cx1, cy2 - cy1
    if cw <= 0 or ch <= 0:
        return None
    if cw < max(MIN_BOX_PIXELS, MIN_BOX_FRACTION * w):
        return None
    if ch < max(MIN_BOX_PIXELS, MIN_BOX_FRACTION * h):
        return None
    return {"x": round(cx1), "y": round(cy1), "width": round(cw), "height": round(ch)}


def build_references(person_id: str, assets: List[dict], min_score: float,
                     cache: Dict[str, dict]) -> List[List[float]]:
    """Reference embeddings for the target person, from assets a human already tagged.

    The hand-drawn faces have no stored embedding (Immich does not compute one for a
    `manual` face -- verified: face_search has no row for any of them), so the asset is
    re-detected and the box with the best overlap against the human's box is taken as
    that person's face.
    """
    refs: List[List[float]] = []
    for asset in assets:
        faces = [f for f in asset_faces(asset["id"])
                 if (f.get("person") or {}).get("id") == person_id]
        if not faces:
            continue
        det = _detect_cached(asset, min_score, cache)
        if not det["boxes"]:
            continue
        for f in faces:
            ref = (f["boundingBoxX1"], f["boundingBoxY1"], f["boundingBoxX2"], f["boundingBoxY2"])
            best = max(det["boxes"], key=lambda b: _iou(b, ref))
            if _iou(best, ref) >= 0.2 and best.get("embedding"):
                refs.append(best["embedding"])
                print(f"[reference] {asset.get('originalFileName')} "
                      f"iou={_iou(best, ref):.2f} score={best['score']:.3f}", flush=True)
    return refs


def _detect_cached(asset: dict, min_score: float, cache: Dict[str, dict]) -> dict:
    key = f"{asset['id']}@{min_score}"
    if key not in cache:
        cache[key] = detect(asset["id"], asset.get("originalFileName") or "image", min_score)
    return cache[key]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--album", required=True)
    ap.add_argument("--person", required=True)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--cache")
    ap.add_argument("--min-score", type=float, default=0.05)
    ap.add_argument("--score-confident", type=float, default=0.20)
    ap.add_argument("--score-floor", type=float, default=0.05)
    ap.add_argument("--min-similarity", type=float, default=0.60)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    write = not (args.dry_run or args.probe or CAPTIONER_DRY_RUN)

    cache: Dict[str, dict] = {}
    if args.cache and os.path.exists(args.cache):
        with open(args.cache) as fh:
            cache = json.load(fh)
        print(f"[cache] loaded {len(cache)} detection(s) from {args.cache}", flush=True)

    def save_cache():
        if args.cache:
            tmp = args.cache + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(cache, fh)
            os.replace(tmp, args.cache)

    album = find_album(args.album)
    person = find_person(args.person)
    person_id, person_name = person["id"], person["name"]
    assets = album_assets(album["id"])
    print(f"[album] {album['albumName']!r} -> {len(assets)} asset(s)", flush=True)
    print(f"[person] {person_name!r} id={person_id}", flush=True)
    print(f"[mode] min_score={args.min_score} confident>={args.score_confident} "
          f"floor>={args.score_floor} sim>={args.min_similarity} "
          f"{'WRITING' if write else 'read-only'}\n", flush=True)

    refs = build_references(person_id, assets, args.min_score, cache)
    save_cache()
    if refs:
        pairs = [_cosine(refs[i], refs[j])
                 for i in range(len(refs)) for j in range(i + 1, len(refs))]
        spread = f", self-agreement {min(pairs):+.2f}..{max(pairs):+.2f}" if pairs else ""
        print(f"[reference] {len(refs)} embedding(s){spread}\n", flush=True)
    else:
        print("[reference] none -- low-score boxes cannot be rescued, only "
              f"score>={args.score_confident} will be accepted\n", flush=True)

    stats = {"already": 0, "reassigned": 0, "confident": 0, "probable": 0,
             "rejected": 0, "other_person": 0, "error": 0}
    probable: List[str] = []
    undetected: List[str] = []
    conflicts: List[str] = []

    todo = assets[: args.limit] if args.limit else assets
    for i, asset in enumerate(todo, 1):
        asset_id = asset["id"]
        fname = asset.get("originalFileName", asset_id)
        label = f"[{i}/{len(todo)}] {fname}"
        try:
            faces = asset_faces(asset_id)

            if any((f.get("person") or {}).get("id") == person_id for f in faces):
                stats["already"] += 1
                print(f"{label}: already tagged {person_name}", flush=True)
                continue

            # Prefer adopting a face Immich already found over inventing one: its box is a
            # real detection at full confidence.
            adoptable = [f for f in faces if not (f.get("person") or {}).get("name")]
            blocked = [f for f in faces if (f.get("person") or {}).get("name")]

            if adoptable:
                def area(f):
                    return (max(0, f.get("boundingBoxX2", 0) - f.get("boundingBoxX1", 0))
                            * max(0, f.get("boundingBoxY2", 0) - f.get("boundingBoxY1", 0)))
                face = max(adoptable, key=area)
                print(f"{label}: reassigning existing face {face['id']} -> {person_name}",
                      flush=True)
                if write:
                    # Note the inversion: the PATH carries the target person and the BODY
                    # carries the face being moved (PersonService.reassignFacesById).
                    _api("PUT", f"/api/faces/{person_id}", json={"id": face["id"]})
                    time.sleep(SLEEP_SECONDS)
                stats["reassigned"] += 1
                continue

            if blocked:
                # Every face here belongs to another NAMED person. Reassigning would remove
                # the asset from that person's page, which is not this script's call.
                names = sorted({(f.get("person") or {}).get("name") for f in blocked})
                stats["other_person"] += 1
                conflicts.append(f"{fname} -> {names}")
                print(f"{label}: SKIPPED, face(s) belong to {names}", flush=True)
                continue

            det = _detect_cached(asset, args.min_score, cache)
            if i % 25 == 0:
                save_cache()

            candidates = []
            for b in det["boxes"]:
                geom = _sane(b, det["w"], det["h"])
                if not geom:
                    continue
                score = float(b.get("score") or 0.0)
                sim = (max(_cosine(b["embedding"], r) for r in refs)
                       if refs and b.get("embedding") else None)
                if score >= args.score_confident:
                    tier = "confident"
                elif (score >= args.score_floor and sim is not None
                        and sim >= args.min_similarity):
                    tier = "probable"
                else:
                    continue
                candidates.append((geom, score, sim, tier))

            if not candidates:
                stats["rejected"] += 1
                undetected.append(fname)
                best = max((float(b.get("score") or 0) for b in det["boxes"]), default=0.0)
                print(f"{label}: no acceptable face ({len(det['boxes'])} raw box(es), "
                      f"best score {best:.3f})", flush=True)
                continue

            # Confident boxes outrank probable ones; within a tier, largest wins.
            candidates.sort(key=lambda c: (c[3] == "confident",
                                           c[0]["width"] * c[0]["height"]), reverse=True)
            geom, score, sim, tier = candidates[0]
            simtxt = f" sim={sim:+.2f}" if sim is not None else ""
            extra = f" (+{len(candidates) - 1} other)" if len(candidates) > 1 else ""
            print(f"{label}: {tier.upper()} face {geom} in {det['w']}x{det['h']} "
                  f"score={score:.3f}{simtxt}{extra}", flush=True)
            if write:
                _api("POST", "/api/faces", json={
                    "personId": person_id, "assetId": asset_id,
                    "imageWidth": det["w"], "imageHeight": det["h"], **geom,
                })
                time.sleep(SLEEP_SECONDS)
            stats[tier] += 1
            if tier == "probable":
                probable.append(f"{fname}  score={score:.3f}{simtxt}  {geom}")

        except Exception as e:
            stats["error"] += 1
            print(f"{label}: ERROR {e}", flush=True)

    save_cache()
    print("\n[summary] " + "  ".join(f"{k}={v}" for k, v in stats.items()), flush=True)
    if probable:
        print(f"\n[probable] {len(probable)} asset(s) accepted on the weaker signal -- "
              f"these are the ones worth spot-checking:", flush=True)
        for p in probable:
            print(f"  {p}", flush=True)
    if conflicts:
        print(f"\n[conflicts] {len(conflicts)} asset(s) whose only face belongs to another "
              f"named person -- reassign by hand if that is wrong:", flush=True)
        for c in conflicts:
            print(f"  {c}", flush=True)
    if undetected:
        print(f"\n[undetected] {len(undetected)} asset(s) carry no {person_name} tag and "
              f"need a box drawn by hand:", flush=True)
        for name in undetected:
            print(f"  {name}", flush=True)


if __name__ == "__main__":
    main()
