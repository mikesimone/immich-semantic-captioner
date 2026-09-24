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

# Reconcile actual state once before trusting the event stream. The loop below is purely
# event-driven, and on a cold boot docker.service auto-starts immich_server (restart:
# always) at roughly the moment this unit comes up -- if that 'start' event lands before
# `docker events` has attached, it is missed permanently. That matters because the
# captioner runs restart: unless-stopped, so a container that was explicitly stopped (by
# wopr-shutdown, or by hand) is NOT restarted by the daemon at boot: it would sit dead
# until a human noticed the queue had stopped draining. Checking the pair's real state at
# startup closes the race without changing the steady-state behavior.
prime_captioner() {
    local immich_running captioner_running
    immich_running="$(docker inspect -f '{{.State.Running}}' "$IMMICH_CONTAINER" 2>/dev/null || echo false)"
    captioner_running="$(docker inspect -f '{{.State.Running}}' "$CAPTIONER_CONTAINER" 2>/dev/null || echo missing)"
    if [[ "$immich_running" == "true" && "$captioner_running" == "false" ]]; then
        log "startup reconcile: $IMMICH_CONTAINER is up but $CAPTIONER_CONTAINER is down"
        wait_for_healthy || log "$IMMICH_CONTAINER not healthy within ${HEALTH_TIMEOUT_SECONDS}s; starting $CAPTIONER_CONTAINER anyway"
        docker start "$CAPTIONER_CONTAINER" >/dev/null \
            && log "started $CAPTIONER_CONTAINER" \
            || log "WARNING: could not start $CAPTIONER_CONTAINER"
    fi
}

prime_captioner

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
