#!/usr/bin/env python3
"""Ad-hoc debug: download a video, extract the frame at a specific timestamp, and print the
RAW classifier text the video-signal prompt produces for it -- for diagnosing why a
particular frame is (mis)classified.

Usage: python3 - <asset_id> <timestamp_seconds> [<timestamp_seconds> ...]
"""
import sys
import tempfile
import os

sys.path.insert(0, "/app")
from captioner import (  # noqa: E402
    load_joycaption,
    immich_download_original,
    extract_video_frames,
    _VIDEO_SIGNAL_PROMPT,
    _parse_video_signal,
)
import subprocess
from PIL import Image

asset_id = sys.argv[1]
timestamps = [float(x) for x in sys.argv[2:]]

print("[debug] loading JoyCaption...", flush=True)
caption_detailed = load_joycaption()

fd, video_path = tempfile.mkstemp(suffix=".mp4")
os.close(fd)
immich_download_original(asset_id, video_path)

for ts in timestamps:
    out_path = f"/tmp/frame_debug/{asset_id}_{ts:.2f}.jpg"
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{ts:.2f}", "-i", video_path, "-frames:v", "1", "-q:v", "2", out_path],
        capture_output=True, timeout=60,
    )
    if not os.path.exists(out_path):
        print(f"[debug] {ts}: frame extraction failed", flush=True)
        continue
    img = Image.open(out_path).convert("RGB")
    raw = caption_detailed(img, prompt_override=_VIDEO_SIGNAL_PROMPT, max_new_tokens=100, greedy=True)
    parsed = _parse_video_signal(raw)
    print(f"\n=== {asset_id} @ {ts}s -> {out_path} ===", flush=True)
    print("RAW:", repr(raw), flush=True)
    print("PARSED:", parsed, flush=True)

os.remove(video_path)
