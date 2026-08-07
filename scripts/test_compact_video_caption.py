#!/usr/bin/env python3
"""Ad-hoc test harness for the compact video-porn caption format (caption_video() in
captioner.py). Downloads real sample videos from a handful of albums, runs the new
captioning logic, and prints the result -- does NOT write anything back to Immich
(caption_video() only returns text; only process_candidate()/immich_update_description()
would persist it, and this script never calls those).

Run inside the captioner container (needs the loaded JoyCaption model + GPU):

    docker exec -i immich_captioner python3 - < scripts/test_compact_video_caption.py
"""
import sys

sys.path.insert(0, "/app")
from captioner import (  # noqa: E402
    IMMICH_URL,
    immich_headers,
    load_joycaption,
    caption_video,
    get_asset_albums,
    extract_identities_from_albums,
    is_dense_sampling_album,
    is_compilation_album,
    is_feral_album,
    is_multiple_creampie_album,
)
import requests

ALBUMS = {
    "Internet Titties": "b1c706bd-7d08-41c9-9223-00c090b49317",
    "Single Creampie": "3a22144e-143c-4f43-a508-8b3f7fadbcb5",
    "Feral on Human Video": "46675d67-bcc5-4375-b998-27a976204dcb",
    "Multiple Creampie": "e7479905-44b5-42ca-86d0-aaf8fb7c36e3",
    "Creampie Compilation": "090261d8-dd49-48a0-8c48-9be74f67ee90",
}

N_PER_ALBUM = 5


def get_album_videos(album_id: str, n: int):
    # GET /api/albums/{id} doesn't return an "assets" array on this Immich version
    # (verified empirically) -- use the metadata search endpoint's albumIds filter instead.
    r = requests.post(
        f"{IMMICH_URL}/api/search/metadata",
        headers={**immich_headers(), "Content-Type": "application/json"},
        json={"albumIds": [album_id], "type": "VIDEO", "size": n, "page": 1},
        timeout=60,
    )
    r.raise_for_status()
    return r.json().get("assets", {}).get("items", [])


def main():
    print("[test] loading JoyCaption...", flush=True)
    caption_detailed = load_joycaption()

    for album_name, album_id in ALBUMS.items():
        print(f"\n===== {album_name} =====", flush=True)
        videos = get_album_videos(album_id, N_PER_ALBUM)
        if not videos:
            print("  (no video assets found)", flush=True)
            continue
        for v in videos:
            asset_id = v["id"]
            albums = get_asset_albums(asset_id)
            person_names = extract_identities_from_albums(albums)
            dense = is_dense_sampling_album(albums)
            compilation = is_compilation_album(albums)
            feral = is_feral_album(albums)
            multiple = is_multiple_creampie_album(albums)
            print(f"\n--- {asset_id} (dense={dense}, compilation={compilation}, feral={feral}, multiple={multiple}, person_names={person_names}) ---", flush=True)
            try:
                caption, mode = caption_video(
                    asset_id, caption_detailed, person_names=person_names, dense=dense,
                    compilation=compilation, feral=feral, multiple=multiple,
                )
                print(f"[{mode}] {caption}", flush=True)
            except Exception as e:
                print(f"[error] {asset_id}: {e}", flush=True)


if __name__ == "__main__":
    main()
