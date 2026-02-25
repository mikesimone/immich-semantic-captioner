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

Edit `.env` and set:

- `IMMICH_URL`
- `IMMICH_API_KEY`
- Postgres settings (`PGHOST`, `PGPASSWORD`, etc.)

### 3) Run (GPU mode)

```
docker compose up -d immich-captioner
```

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
docker compose build --no-cache immich-captioner
docker compose up -d immich-captioner
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

---

## Database Changes

Creates one additional table if not present:

```
captioner_skip
```

Used to track assets that failed processing.

No other schema changes are made.

---

## Reset All Captions

⚠ WARNING: This overwrites all descriptions.

```
docker exec -i immich_postgres psql -U postgres -d immich -c "UPDATE asset_exif SET description='';"
```

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

## License

MIT License. See `LICENSE`.




