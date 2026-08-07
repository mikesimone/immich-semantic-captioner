#!/usr/bin/env python3
"""Ad-hoc: test caption_video() on specific asset IDs (used for spot-checking the Lydia
skip-description rule and the multi-woman prompt fix). Does not write back to Immich."""
import sys

sys.path.insert(0, "/app")
from captioner import (  # noqa: E402
    load_joycaption,
    caption_video,
    get_asset_albums,
    extract_identities_from_albums,
    is_dense_sampling_album,
    is_compilation_album,
    is_feral_album,
    is_multiple_creampie_album,
)

ASSET_IDS = sys.argv[1:]


def main():
    print("[test] loading JoyCaption...", flush=True)
    caption_detailed = load_joycaption()

    for asset_id in ASSET_IDS:
        albums = get_asset_albums(asset_id)
        person_names = extract_identities_from_albums(albums)
        dense = is_dense_sampling_album(albums)
        compilation = is_compilation_album(albums)
        feral = is_feral_album(albums)
        multiple = is_multiple_creampie_album(albums)
        print(f"\n--- {asset_id} (albums={albums}, dense={dense}, compilation={compilation}, feral={feral}, multiple={multiple}, person_names={person_names}) ---", flush=True)
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
