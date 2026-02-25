# Immich Semantic Captioner

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
Immich Postgres → candidate selection
          ↓
Immich API → thumbnail fetch
          ↓
Florence-2 (GPU or CPU)
          ↓
Caption cleanup + identity injection
          ↓
Immich API → description update
```

---

## Identity Injection Logic

Album names must follow:

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

---

## Database Changes

Creates one additional table if not present:

```
captioner_skip
```

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
- GPU strongly recommended.
- Uses thumbnails for speed.
- HuggingFace cache stored in a Docker volume.

---

## License

MIT License. See `LICENSE`.

