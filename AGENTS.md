# AI Project Context

## Purpose
`immich-semantic-captioner` adds semantic descriptions to Immich assets to improve natural-language search. It performs OCR/caption generation, cleanup, deterministic identity injection, and writes descriptions back to Immich.

## Runtime / deployment
- Canonical production instance runs on WOPR as the `immich_captioner` Docker container.
- API-only mode is the recommended/default integration; direct Postgres mode exists for faster candidate discovery and is more schema-sensitive.
- GPU acceleration is preferred for the production library.
- Secrets and deployment-specific `.env` values remain outside Git.
- WOPR deployment truth and host paths are documented in `mikesimone/Environment`.

## Important invariants
- Identity injection is deterministic and configured; do not add guessy person-identification heuristics.
- Avoid permanent skips for transient Immich thumbnail/API failures; retry behavior matters.
- Direct database access must remain optional and clearly separated from supported API behavior.
- Commands that wipe/reset descriptions are destructive and must remain conspicuously documented.

## Working rules
Read `README.md` before changing caption selection, identity handling, skip/retry behavior, or deployment. Preserve API compatibility where possible. Never commit Immich API keys or database credentials. Update docs when runtime model, identity configuration, schema interaction, or deployment behavior changes.
