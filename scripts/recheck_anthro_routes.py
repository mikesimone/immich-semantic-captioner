#!/usr/bin/env python3
"""Re-check every upload the router filed as anthro (captioner_routed outcome='anthro')
against _ANTHRO_DESCRIBE_PROMPT, and with --apply send the ones that fail back through
routing: removed from the anthro routing albums, caption blanked, unarchived, and their
captioner_routed row dropped, so the running captioner routes them again from scratch.

Written 2026-10-10 after the triage prompt's ANTHRO line turned out to say Y for nearly
everything (human porn, screenshots, phone photos). Assets that are also in some album the
router doesn't add for anthro were filed by hand since, so they're reported and left alone.

Run with the captioner stopped, from the repo root:
  scripts/compose.sh stop immich-captioner
  scripts/compose.sh run --rm -T -v "$PWD/scripts:/scripts:ro" \
      --entrypoint python immich-captioner /scripts/recheck_anthro_routes.py [--apply] [asset-id ...]
  scripts/compose.sh up -d immich-captioner
"""
import os
import sys
import tempfile

sys.path.insert(0, "/app")
import captioner as c  # noqa: E402

APPLY = "--apply" in sys.argv
ONLY = [a for a in sys.argv[1:] if not a.startswith("--")]
ANTHRO_KEYS = ["furry", "anthro_video", "anthro_sex_video", "human_anthro_video",
               "human_anthro_still", "cow_anthro"]


def frames_for(asset_id: str, is_video: bool):
    if not is_video:
        return [(0.0, c.immich_get_thumbnail(asset_id))]
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        c.immich_download_original(asset_id, path)
        return c.extract_video_frames(path, dense=False)
    finally:
        os.unlink(path)


def main():
    conn = c.pg_connect()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT r.asset_id::text, a.type::text, a."originalFileName"
            FROM captioner_routed r JOIN asset a ON a.id = r.asset_id
            WHERE r.outcome = 'anthro' AND a."deletedAt" IS NULL
            ORDER BY r.routed_at
        """)
        rows = [r for r in cur.fetchall() if not ONLY or r[0] in ONLY]
        cur.execute("""SELECT aa."assetId"::text, aa."albumId"::text, al."albumName"
                       FROM album_asset aa JOIN album al ON al.id = aa."albumId" """)
        albums = {}
        for asset_id, album_id, name in cur.fetchall():
            albums.setdefault(asset_id, []).append((album_id, name))
    anthro_ids = {c.album_id_for(k) for k in ANTHRO_KEYS} - {None}
    caption_detailed = c.load_joycaption()
    kept = requeued = hand_filed = 0
    lydia_dog_id = c.album_id_for("lydia_dog")
    for asset_id, typ, name in rows:
        if any(aid == lydia_dog_id for aid, _ in albums.get(asset_id, [])):
            kept += 1
            print(f"[keep] {asset_id} {name} in Lydia Dog -- not rechecked", flush=True)
            continue
        try:
            frames = frames_for(asset_id, typ == "VIDEO")
        except Exception as e:
            print(f"[error] {asset_id} {name}: {e}", flush=True)
            continue
        raw = c._classify_with_prompt([img for _, img in frames], c._ANTHRO_DESCRIBE_PROMPT,
                                      lambda t: {"anthro": c._parse_anthro_describe(t), "text": t},
                                      caption_detailed, max_new_tokens=60)
        hits = sum(r["anthro"] for r in raw)
        sample = " ".join(raw[len(raw) // 2]["text"].split())[:140] if raw else ""
        if c.frames_majority(raw, "anthro", c.ANTHRO_MIN_FRACTION):
            kept += 1
            print(f"[keep] {asset_id} {name} anthro={hits}/{len(raw)} :: {sample}", flush=True)
            continue
        mine = albums.get(asset_id, [])
        other = [n for aid, n in mine if aid not in anthro_ids]
        if other:
            hand_filed += 1
            print(f"[hand-filed] {asset_id} {name} anthro={hits}/{len(raw)} also in {other} -- left alone :: {sample}", flush=True)
            continue
        requeued += 1
        print(f"[requeue] {asset_id} {name} anthro={hits}/{len(raw)} :: {sample}", flush=True)
        if APPLY:
            for aid, _ in mine:
                c.immich_remove_from_album(asset_id, aid)
            c.immich_update_description(asset_id, "")
            c.immich_unarchive(asset_id)
            with conn.cursor() as cur:
                cur.execute("DELETE FROM captioner_routed WHERE asset_id = %s", (asset_id,))
    print(f"\n[summary] checked={len(rows)} kept={kept} requeue={requeued} hand-filed={hand_filed} "
          f"applied={APPLY}", flush=True)


if __name__ == "__main__":
    main()
