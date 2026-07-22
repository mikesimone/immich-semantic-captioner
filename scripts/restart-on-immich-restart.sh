#!/usr/bin/env bash
# Watches for immich_server (re)starting and restarts immich_captioner in response.
#
# The captioner keeps DB connections and its candidate-fetch loop running across
# an Immich restart, and can poll assets before Immich's own services (thumbnail
# generation, etc.) have caught back up. Restarting it gives it a clean start
# once Immich is confirmed healthy again, instead of running against a
# half-initialized Immich until its own retry logic eventually recovers.
set -euo pipefail

IMMICH_CONTAINER="${IMMICH_CONTAINER:-immich_server}"
CAPTIONER_CONTAINER="${CAPTIONER_CONTAINER:-immich_captioner}"
HEALTH_TIMEOUT_SECONDS="${HEALTH_TIMEOUT_SECONDS:-300}"
HEALTH_POLL_SECONDS="${HEALTH_POLL_SECONDS:-5}"

log() { echo "[$(date -u +%FT%TZ)] $*"; }

wait_for_healthy() {
    local waited=0
    while (( waited < HEALTH_TIMEOUT_SECONDS )); do
        local status
        status="$(docker inspect -f '{{.State.Health.Status}}' "$IMMICH_CONTAINER" 2>/dev/null || echo "unknown")"
        if [[ "$status" == "healthy" ]]; then
            return 0
        fi
        sleep "$HEALTH_POLL_SECONDS"
        waited=$(( waited + HEALTH_POLL_SECONDS ))
    done
    return 1
}

log "Watching for '$IMMICH_CONTAINER' restarts to bounce '$CAPTIONER_CONTAINER'..."

docker events --filter "container=$IMMICH_CONTAINER" --filter "event=start" --format '{{.Time}}' |
while read -r _; do
    log "$IMMICH_CONTAINER started; waiting for it to report healthy..."
    if wait_for_healthy; then
        log "$IMMICH_CONTAINER healthy; restarting $CAPTIONER_CONTAINER"
    else
        log "Timed out waiting for $IMMICH_CONTAINER to become healthy after ${HEALTH_TIMEOUT_SECONDS}s; restarting $CAPTIONER_CONTAINER anyway"
    fi
    docker restart "$CAPTIONER_CONTAINER"
done
