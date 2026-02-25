
# Immich Semantic Captioner

Adds OCR-first, Florence-2 powered semantic descriptions to Immich assets.

## What It Does

* Automatically captions images using Florence-2
* OCR-first (screenshots & memes searchable by text)
* Injects identity based on album naming convention
* Writes descriptions into `asset_exif.description`
* Skips problematic assets via a skip table
* GPU accelerated (CUDA supported)

## Requirements

* Immich already running
* Docker
* NVIDIA GPU (recommended)
* NVIDIA Container Toolkit

## Setup

1. Copy `.env.example` → `.env`
2. Fill in your Immich URL + API key
3. Run:

```
docker compose up -d immich-captioner
```

## Identity Convention

Albums must follow:

```
NNN(.NNN)* - PersonName [optional text]
```

Example:

```
002.002 - Lydia Being a Good Girl
```

Identity injected: `Lydia`

## Reset Captions

```
docker exec -i immich_postgres psql -U postgres -d immich -c "UPDATE asset_exif SET description='';"
```


