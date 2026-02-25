# Immich Semantic Captioner

![Docker](https://img.shields.io/badge/docker-required-blue)
![GPU Optional](https://img.shields.io/badge/GPU-optional-green)
![Model](https://img.shields.io/badge/model-Florence--2-purple)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

Semantic captioning and identity-aware description injection for Immich.

---

## What This Project Does

This system augments Immich search by:

- Running Florence-2 for OCR-first image captioning
- Writing captions into `asset_exif.description`
- Injecting human identity based on album names
- Guaranteeing name presence in descriptions
- Enabling semantic search across reaction memes, personal photos, and structured albums

---

## Why This Is Better Than Default Immich Search

Immich's built-in ML search relies on embeddings and object detection.

This system:

- Forces deterministic identity presence
- Prioritizes OCR (critical for memes)
- Stores results in first-class searchable fields
- Allows identity-based semantic retrieval
- Enables album-driven metadata logic

You can now search:

- “Lydia black dress”
- “Joey graduation stage”
- “Me tattoo progress”
- “Funny meme about work”

And actually get consistent, meaningful results.

---

## Architecture Overview

See included diagram: `architecture_diagram.png`

Processing pipeline:

1. Query Postgres for uncaptionsed assets
2. Fetch thumbnail via Immich API
3. Run Florence-2 caption generation
4. Apply identity overrides
5. Update `asset_exif.description`

---

## Quickstart

### 1. Clone

```bash
git clone https://github.com/mikesimone/immich-semantic-captioner.git
cd immich-semantic-captioner
```

### 2. Configure

Copy environment template:

```bash
cp .env.example .env
```

Fill in:

- IMMICH_URL
- IMMICH_API_KEY
- DB_PASSWORD

### 3. Start

```bash
docker compose up -d --build
```

---

## CPU-Only Mode

If you do not have a GPU:

- Remove `gpus: all` from docker-compose.yml
- Use CPU base image in Dockerfile

Florence-2 will run slower but remains functional.

---

## Database Impact

Creates one optional table:

`captioner_skip`

Used to track assets that failed processing.

No other schema modifications.

---

## License

MIT License

---

## Production Notes

- Designed for private deployments
- Overwrites existing descriptions
- Intended for deterministic metadata enrichment

---


