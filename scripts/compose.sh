#!/usr/bin/env bash
# docker compose for the captioner, with secrets loaded from ~/.api-keys.
# Safe from cron/systemd/non-login shells: e.g. scripts/compose.sh up -d immich-captioner
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck source=/dev/null
. "$HOME/.api-keys"
exec docker compose "$@"
