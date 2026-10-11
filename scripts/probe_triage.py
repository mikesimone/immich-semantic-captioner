#!/usr/bin/env python3
"""Ad-hoc: run the upload-routing triage prompt (plus any candidate prompts below) on specific
assets and print each frame's raw answer, so anthro/furry misroutes can be diagnosed. Does not
write back to Immich.

Run with the captioner stopped (it needs the GPU JoyCaption is sitting on), from the repo root:
  scripts/compose.sh stop immich-captioner
  scripts/compose.sh run --rm -T -v "$PWD/scripts:/scripts:ro" \
      --entrypoint python immich-captioner /scripts/probe_triage.py <asset-id> ...
  scripts/compose.sh up -d immich-captioner
"""
import os
import re
import sys
import tempfile

import requests

sys.path.insert(0, "/app")
import captioner as c  # noqa: E402

ASSET_IDS = sys.argv[1:]

# name -> (prompt, max_new_tokens, parser returning True when the frame counts as anthro)
CANDIDATES = {
    "current": (c._TRIAGE_PROMPT, 60, lambda t: c._parse_triage(t)["anthro"]),
}
if hasattr(c, "_ANTHRO_DESCRIBE_PROMPT"):
    CANDIDATES["describe"] = (c._ANTHRO_DESCRIBE_PROMPT, 60, c._parse_anthro_describe)


def frames_for(asset_id: str):
    info = requests.get(f"{c.IMMICH_URL}/api/assets/{asset_id}", headers=c.immich_headers(),
                        timeout=30).json()
    if info.get("type") != "VIDEO":
        return info, [(0.0, c.immich_get_thumbnail(asset_id))]
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        c.immich_download_original(asset_id, path)
        return info, c.extract_video_frames(path, dense=False)
    finally:
        os.unlink(path)


def main():
    caption_detailed = c.load_joycaption()
    for asset_id in ASSET_IDS:
        try:
            info, frames = frames_for(asset_id)
        except Exception as e:
            print(f"\n=== {asset_id} error: {e}", flush=True)
            continue
        print(f"\n=== {asset_id} {info.get('originalFileName', '')} frames={len(frames)}", flush=True)
        for name, (prompt, tokens, is_anthro) in CANDIDATES.items():
            raw = caption_detailed.batch([
                {"pil_image": img, "prompt_override": prompt, "max_new_tokens": tokens, "greedy": True}
                for _, img in frames
            ])
            rows = [{"anthro": bool(is_anthro(t))} for t in raw]
            hits = sum(r["anthro"] for r in rows)
            routed = c.frames_majority(rows, "anthro", c.ANTHRO_MIN_FRACTION)
            print(f"  {name}: anthro={hits}/{len(rows)} -> {'ANTHRO' if routed else 'not anthro'}", flush=True)
            for (ts, _), t in list(zip(frames, raw))[:4]:
                one_line = re.sub(r"\s+", " ", t)[:220]
                print(f"    [{ts:7.1f}] {one_line}", flush=True)


if __name__ == "__main__":
    main()
