#!/usr/bin/env python3
"""Find (and optionally trash) duplicate videos in the porn albums (200.x).

High-confidence duplicates only: identical checksum, OR the same original filename (ignoring
case, extension and copy markers like "+1" / " (2)") with durations within 1 s. Immich's own
visual duplicate groups (asset.duplicateId) are only LISTED, never trashed -- those are fuzzy.

Per group it keeps, in order: an asset with a hand-count lock (captioner_hand_counted) or an
entry in the porn-classifier eval set, then the higher resolution, then the bigger file, then
the one in more albums. Before trashing the others it adds the kept copy to every album they
were in. Trashing is Immich's soft delete (restorable from the trash for trashDays).

Dry run by default; --apply to act. Run on WOPR with ~/.api-keys sourced (IMMICH_URL,
IMMICH_API_KEY); reads Postgres through `docker exec immich_postgres psql`.
First used 2026-09-26 at Mike's request: 5 pairs trashed.
"""
import argparse
import collections
import json
import os
import re
import subprocess

import requests

EVAL_SET = "/opt/classifier/training-data/eval/creampie-counts-20260925.json"
SQL = """
SELECT DISTINCT a.id, a."originalFileName", coalesce(a.duration, 0), encode(a.checksum, 'hex'),
  coalesce(e."exifImageWidth", 0) * coalesce(e."exifImageHeight", 0), coalesce(e."fileSizeInByte", 0),
  coalesce(a."duplicateId"::text, ''), (SELECT count(*) FROM album_asset x WHERE x."assetId" = a.id)
FROM asset a JOIN album_asset aa ON aa."assetId" = a.id JOIN album al ON al.id = aa."albumId"
LEFT JOIN asset_exif e ON e."assetId" = a.id
WHERE al."albumName" LIKE '200.%' AND a.type = 'VIDEO' AND a."deletedAt" IS NULL
"""


def psql(sql):
    out = subprocess.run(["docker", "exec", "-i", "immich_postgres", "psql", "-U", "postgres", "-d", "immich",
                          "-At", "-F", "\t"], input=sql, capture_output=True, text=True, check=True).stdout
    return [l.split("\t") for l in out.splitlines() if l]


def norm(fn):
    b = re.sub(r"\.[A-Za-z0-9]{2,4}$", "", fn).lower().strip()
    return re.sub(r"(\+\d+|\s*\(\d+\))+$", "", b).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="add albums to the kept copy and trash the rest")
    args = ap.parse_args()
    url = os.environ["IMMICH_URL"].rstrip("/")
    hdr = {"x-api-key": os.environ["IMMICH_API_KEY"], "Content-Type": "application/json"}

    rows = [dict(zip(("id", "name", "dur", "ck", "px", "size", "dup", "albums"), r)) for r in psql(SQL)]
    protected = {r[0] for r in psql("SELECT asset_id::text FROM captioner_hand_counted")}
    if os.path.exists(EVAL_SET):
        protected |= {r["asset_id"] for r in json.load(open(EVAL_SET))}

    groups, seen = [], set()
    by_ck = collections.defaultdict(list)
    by_name = collections.defaultdict(list)
    for r in rows:
        by_ck[r["ck"]].append(r)
        by_name[norm(r["name"])].append(r)
    for g in by_ck.values():
        if len(g) > 1:
            groups.append(g)
    for g in by_name.values():
        g = sorted(g, key=lambda r: int(r["dur"]))
        cl = [[g[0]]]
        for r in g[1:]:
            if int(r["dur"]) - int(cl[-1][-1]["dur"]) <= 1000:
                cl[-1].append(r)
            else:
                cl.append([r])
        groups += [c for c in cl if len(c) > 1]
    uniq = []
    for g in groups:
        key = frozenset(r["id"] for r in g)
        if key not in seen:
            seen.add(key)
            uniq.append(g)

    def rank(r):
        return (r["id"] in protected, int(r["px"]), int(r["size"]), int(r["albums"]))

    for g in uniq:
        g = sorted(g, key=rank, reverse=True)
        keep, drops = g[0], g[1:]
        print(f"KEEP  {keep['id']}  {keep['name'][:70]}")
        for d in drops:
            if d["id"] in protected:
                print(f"  SKIP (protected) {d['id']}")
                continue
            print(f"  TRASH {d['id']}  {d['name'][:70]}")
            if args.apply:
                have = {a["id"] for a in requests.get(f"{url}/api/albums", params={"assetId": keep["id"]}, headers=hdr, timeout=30).json()}
                for a in requests.get(f"{url}/api/albums", params={"assetId": d["id"]}, headers=hdr, timeout=30).json():
                    if a["id"] not in have:
                        requests.put(f"{url}/api/albums/{a['id']}/assets", headers=hdr, json={"ids": [keep["id"]]}, timeout=30)
                requests.delete(f"{url}/api/assets", headers=hdr, json={"ids": [d["id"]], "force": False}, timeout=30).raise_for_status()

    fuzzy = collections.defaultdict(list)
    for r in rows:
        if r["dup"]:
            fuzzy[r["dup"]].append(r)
    for g in fuzzy.values():
        if len(g) > 1:
            print("MAYBE (Immich visual duplicate group, not touched): " +
                  " || ".join(f"{r['id']} {r['name'][:50]} {int(r['dur']) / 1000:.0f}s" for r in g))
    print(f"{len(uniq)} high-confidence group(s){'' if args.apply else ' (dry run)'}")


if __name__ == "__main__":
    main()
