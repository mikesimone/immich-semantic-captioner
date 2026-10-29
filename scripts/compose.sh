#!/usr/bin/env bash
# docker compose for the captioner. Secrets are not passed in at all any more: the container reads them from the
# RAM-only /run/fleet/captioner (vault-written by fleet-sync, TPM-restored at boot), mounted at /run/secrets.
# Safe from cron/systemd/non-login shells: e.g. scripts/compose.sh up -d immich-captioner
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
sudo -n test -s /run/fleet/captioner/IMMICH_API_KEY || { echo "compose.sh: /run/fleet/captioner is empty - run 'sudo fleet-sync sync' (or 'fleet-sync unseal') first" >&2; exit 1; }
exec docker compose "$@"
