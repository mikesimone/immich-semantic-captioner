#!/usr/bin/env python3
"""Dry-run the meme-inbox check (meme_signals) against assets with known answers. Changes nothing.

Samples N random stills from album-number prefixes that should be memes and from ones that shouldn't,
runs the same OCR + describe prompt the router uses, and prints a confusion table.

Run with the captioner stopped (it needs the GPU), from the repo root:
  scripts/compose.sh stop immich-captioner
  scripts/compose.sh run --rm -T -v "$PWD/scripts:/scripts:ro" \\
      --entrypoint python immich-captioner /scripts/check_meme_route.py [N]
  scripts/compose.sh up -d immich-captioner
"""
import random, sys
sys.path.insert(0, "/app")
import captioner as c  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
SHOULD = ("020.9",)                         # sorted meme topic albums
SHOULD_NOT = ("000.", "010.", "002.")       # dogs, events/places, Me
random.seed(1111)


def album_assets_fallback(prefixes):
    import json, urllib.request
    out = set()
    for a in c.immich_list_albums():
        if a["albumName"].startswith(prefixes):
            body = json.dumps({"albumIds": [a["id"]], "type": "IMAGE", "size": 200}).encode()
            req = urllib.request.Request(c.IMMICH_URL.rstrip("/") + "/api/search/metadata", data=body,
                                         headers={"x-api-key": c.IMMICH_API_KEY, "Content-Type": "application/json"})
            out.update(x["id"] for x in json.load(urllib.request.urlopen(req))["assets"]["items"])
    return random.sample(sorted(out), min(N, len(out)))


def main():
    ocr_fn = c.load_florence_ocr()
    cap = c.load_joycaption()
    sample = lambda p: album_assets_fallback(p)
    results = {"should": [], "should_not": []}
    for label, prefixes in (("should", SHOULD), ("should_not", SHOULD_NOT)):
        for aid in sample(prefixes):
            thumb = c.immich_get_thumbnail(aid)
            sig = c.meme_signals(thumb, None, ocr_fn(thumb), cap)
            results[label].append(sig["meme"])
            print(f"{label:10} {aid[:8]} meme={sig['meme']!s:5} {sig['why']}", flush=True)
    tp = sum(results["should"]); fn = len(results["should"]) - tp
    fp = sum(results["should_not"]); tn = len(results["should_not"]) - fp
    print(f"\nmemes caught {tp}/{tp+fn}   non-memes wrongly filed {fp}/{fp+tn}")


if __name__ == "__main__":
    main()
