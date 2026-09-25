# Immich Semantic Captioner

![Docker](https://img.shields.io/badge/docker-required-blue)
![GPU Optional](https://img.shields.io/badge/GPU-optional-green)
![Model](https://img.shields.io/badge/model-Florence--2-purple)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

Adds OCR-first, Florence-2 powered semantic descriptions to Immich assets.

This service automatically generates human-readable descriptions for images stored in Immich and writes them into `asset_exif.description`, enabling significantly improved natural-language search.

---

## Quickstart

### Prerequisites

- Immich already running
- Docker
- (Recommended) NVIDIA GPU + NVIDIA Container Toolkit for acceleration
- Immich API key
- Network access from this container to:
  - Immich server (`IMMICH_URL`)
  - Immich Postgres (`PGHOST`, usually `immich_postgres`)

### 1) Clone

```
git clone https://github.com/mikesimone/immich-semantic-captioner.git
cd immich-semantic-captioner
```

### 2) Configure

```
cp .env.example .env
```

Secrets are not kept in `.env`. Export them in the shell that runs compose
(on WOPR they live in `~/.api-keys`, which `scripts/compose.sh` sources):

- `IMMICH_API_KEY`
- `IMMICH_DB_PASSWORD` (Immich's `DB_PASSWORD`; passed to the container as `PGPASSWORD`)

Addresses default to the Immich docker network (`http://immich_server:2283`,
`immich_postgres:5432`, db `immich`, user `postgres`). Override them in `.env`
only if your stack differs, using `CAPTIONER_IMMICH_URL`, `CAPTIONER_PGHOST`,
`CAPTIONER_PGPORT`, `CAPTIONER_PGDATABASE`, `CAPTIONER_PGUSER`. The prefix keeps a
host-side `IMMICH_URL` in your shell from leaking into the container.

Edit `.env` and set:

- USE_API_ONLY (default true): Set to 'true' for fully API-supported operation (recommended for most users). Set to 'false' only if you're comfortable with direct Postgres access and need maximum speed on large libraries.

### 3) Run (GPU mode)

```
scripts/compose.sh up -d immich-captioner
```

`scripts/compose.sh` sources `~/.api-keys` and runs `docker compose`; a bare
`docker compose` works too if the secrets are already exported, and fails with
a clear error if they are not.

### 4) Watch logs

```
docker logs -f immich_captioner
```

---

## CPU-only Mode

This project can run on CPU, but it will be significantly slower.

### Steps

1) Remove the `gpus: all` line from `docker-compose.yml`

2) Set in `.env`:

```
CUDA_BASE_IMAGE=ubuntu:24.04
```

3) Rebuild:

```
scripts/compose.sh build --no-cache immich-captioner
scripts/compose.sh up -d immich-captioner
```

---

## What This Does

For each Immich asset with an empty description:

1. Pulls candidate assets directly from Immich’s Postgres database.
2. Fetches the asset thumbnail via the Immich API.
3. Runs Florence-2:
   - OCR first (for screenshots, memes, documents)
   - Detailed caption fallback (for photos)
4. Cleans watermark and meme boilerplate text.
5. Injects deterministic identity tokens based on album naming convention.
6. Updates the description via Immich’s API.
7. Skips problematic assets via a persistent skip table.

---

## Upload Routing and Album Moves

When Postgres credentials are available (`PG*`, even with `USE_API_ONLY=true`), the captioner
also files new uploads and reacts to albums you file things into. Set `ROUTING_ENABLED=0` to
turn this off.

**New uploads** (in no album, never captioned before) are routed in this order. Each step can
stop processing; a stopped asset still gets its normal caption so it isn't re-queued.

1. **CamSpy**: EXIF make `Meta` (Ray-Ban Meta) or a filename containing `SpyPhoto` goes to
   `400.001`, gets the full narrative caption, and is archived.
2. **Known person**: the asset waits (up to `FACE_WAIT_MAX_SECONDS`) for Immich's face
   recognition. A named person with an album titled after them (`002.000 - Lydia`, `Me`, ...)
   gets filed there and captioned with their name, and stays in the timeline. People without an
   album just get named in the caption. `ROUTING_PERSON_EXCLUDE` (LydiaDog by default) is skipped.
3. **Anthro**: `300.000.000 - Furry Stuff`. Videos also go to `300.001` (and `300.002` for
   anthro/anthro sex or `300.004` for human/anthro sex). Stills go to `300.005` for human/anthro sex,
   `300.000.002` for an anthro cow, and `300.006.000 - Lydia Dog` when the generation info names her
   LoRA, Immich tags her, or a low-threshold face re-detection matches her tagged faces. Archived.
4. **Nudity**: without nudity, the regular caption, left in the timeline.
5. **Porn categories**: solo woman goes to `200.010.000`, plus `200.010.001` if masturbating;
   cow print or horns go to Hucow; lactation goes to `200.010.002`; glory wall goes to
   `200.000.006`; tentacles or the Hentaied logo go to `200.000.008` (archive and stop). Videos
   get the compact porn-ID caption with no creampie count. A studio logo, title card, or filename
   (Slutwife Jessica/Marion, Puta Locura, Creampie Squad, Gangbang/5 Guy Creampie) files the video
   into that studio album plus `200.000.000`, runs the CumCounter, and archives it. Otherwise a
   detected creampie gets the caption `Please Categorize | ...`. It stays in the timeline, and if
   she's restrained it also goes to `200.002.000 - Bondage Creampie`. Anything else filed into a
   porn album is archived, unless a still shows cum in or leaking from a vagina.

**Album moves** are detected by snapshotting album membership every `MOVE_POLL_SECONDS`.
Additions the captioner makes itself, and everything already filed when routing first starts,
don't count.

- Added to `200.000.000 - Multiple Creampie`: the CumCounter replaces the `Separate Creampies`
  field and keeps the other fields. The asset is archived.
- Added to `200.001.000 - Single Creampie`: archived.
- Added to any `100.000.x` album: the CumCounter runs, and archive state is left alone.

The CumCounter samples the video densely and counts a new creampie each time the frames return
to "not inserted, genitals in view" a gap after the previous one. The gap is
`CREAMPIE_GAP_FRACTION` (default 0.1) of the video's length, clamped between
`CREAMPIE_MIN_GAP_SECONDS` (8) and `CREAMPIE_MAX_GAP_SECONDS` (90). It deliberately errs high, because it only runs on content filed as multiples.
Anything in `200.000.000` never reads below `MULTI_CREAMPIE_MIN_COUNT` (default 2).
- A `Please Categorize` caption loses that prefix once the asset is filed anywhere.
- Added to any `300.006.x` (Lydia Dog) album: the still gets the `Lydia Dog` Person tag on its
  main face, or on the whole frame when no face is found.

Routing albums are found by number (see `_DEFAULT_ROUTING_ALBUM_NUMBERS` in `captioner.py`,
overridable with `ROUTING_ALBUM_NUMBERS`), so the text after the number can change freely.

State lives in four tables the captioner creates in the Immich database: `captioner_meta`,
`captioner_routed` (assets already routed, seeded with everything captioned before the first
run), `captioner_face_wait`, and `captioner_album_member`.

---

## Architecture Overview

```
Immich Postgres  ──→  Candidate Selection
        │
        ↓
Immich API  ──→  Thumbnail Fetch
        │
        ↓
Florence-2 (GPU or CPU)
        │
        ↓
Caption Cleanup + Identity Injection
        │
        ↓
Immich API  ──→  Description Update
```

---

## Identity Injection Logic

Album naming convention used by default:

```
NNN(.NNN)* - PersonName [optional text]
```

Examples:

```
002.000 - Lydia
002.002 - Lydia Being a Good Girl
100.000.005 - Jen K
```

Injected identity:

- First name
- Or first name + last initial (if present)

Identity is guaranteed to appear in the caption.

Example:

```
Lydia: The image shows ...
```

### Important

You do **not** need to use numeric prefixes.

The numbering scheme shown above is purely organizational and used for manual sorting.  
Any album naming structure is valid as long as a recognizable person name appears at the start of the album title.

Valid examples without numbers:

```
Lydia
Lydia - Photoshoot
Jen K
Joey Graduation
```

The system extracts the first name token (and optional last initial) and ignores non-person suffix text.

## Configuring Recognized People

You must explicitly define which names should be recognized and injected.

Edit your `.env` file:

IDENTITY_ALBUM_MAP=Lydia:Lydia,Me:Me,Joey:Joey
IDENTITY_NOUN_HINTS=Lydia:woman|girl,Me:man|guy,Joey:boy|child
IDENTITY_ENSURE_MODE=prefix

### IDENTITY_ALBUM_MAP

Maps album title tokens to the canonical name injected into captions.

Format:
MatchToken:CanonicalName

Example:
Album "002.002 - Lydia Being a Good Girl"
→ Injects: Lydia

You can name albums anything you want.
The system only looks for defined match tokens.

### IDENTITY_NOUN_HINTS (Optional)

Used to replace generic phrases like:
- "a woman"
- "the man"
- "a young boy"

If omitted, the identity will still be injected,
but generic noun replacement will be skipped.

### Important

Only names defined in IDENTITY_ALBUM_MAP are injected.
No guessing. No heuristics.

---

## Database Changes

## API-Only Mode (Recommended / Default)

By default, `USE_API_ONLY=true` — the tool scans for uncaptioned assets using paginated calls to Immich's `/search/metadata` endpoint (with `with_exif=true`).

This is fully supported by Immich, requires no DB credentials, and avoids any risk from direct database access.

Trade-off: Slower on very large libraries (full pagination through all assets), but safe and future-proof.

To use direct DB mode (faster candidate discovery):
- Set `USE_API_ONLY=false`
- Ensure valid Postgres credentials are set
- Be aware: Immich does not officially support direct DB reads; schema changes may break this.

Direct DB mode is for advanced/power users only.


If you disable API-only, it creates one additional table if not present:

```
captioner_skip
```

Used to track assets that failed processing.

Upload routing (see above) adds the `captioner_meta`, `captioner_routed`, `captioner_face_wait`
and `captioner_album_member` tables whenever Postgres credentials are set, in either mode. No
other schema changes are made.

---

## Reset All Captions

⚠ **DANGER: This command wipes ALL descriptions in your Immich library!**  
Only run this if you KNOW you want to zero them out and start captioning from scratch (e.g., after changing models or cleanup rules).

⚠ WARNING: This overwrites all descriptions.

```
docker exec -i immich_postgres psql -U postgres -d immich -c "UPDATE asset_exif SET description='';"
```

After running, restart the captioner container to begin re-processing everything.

---

## Performance Notes

- OCR captions are faster than detailed captions.
- GPU strongly recommended for large libraries.
- Uses thumbnails for speed and reduced VRAM usage.
- HuggingFace cache stored in a Docker volume.
- Safe to run continuously; processes only uncaptionsed assets.

---

## Why This Improves Immich Search

Immich’s default ML search relies on embeddings and object detection.

This system:

- Ensures deterministic identity presence in captions.
- Prioritizes OCR for memes, screenshots, and text-heavy images.
- Stores structured descriptions in a first-class searchable field.
- Enables reliable natural-language queries like:

  - "Lydia black dress"
  - "Joey graduation stage"
  - "Me tattoo progress"
  - "funny meme about work"

---
## Important

This project is not affiliated with [Immich](https://github.com/immich-app/immich)
## License

MIT License. See `LICENSE`.




