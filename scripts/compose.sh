#!/usr/bin/env bash
# docker compose for the captioner, with its secrets (IMMICH_API_KEY, IMMICH_DB_PASSWORD,
# CREAMPIE_SCORER_TOKEN) fetched from the fleet's Infisical vault, not ~/.api-keys.
# Safe from cron/systemd/non-login shells: e.g. scripts/compose.sh up -d immich-captioner
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
. scripts/vault-env.sh
vault_env IMMICH_API_KEY IMMICH_DB_PASSWORD CREAMPIE_SCORER_TOKEN
exec docker compose "$@"
