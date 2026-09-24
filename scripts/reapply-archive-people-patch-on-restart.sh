#!/usr/bin/env bash
# Watches for immich_server (re)starting -- including immich-auto-upgrade.service pulling a
# new image -- and re-applies both archive-people patches against the fresh container: the
# backend query patch (patch_immich_archive_people.py) and the frontend static-asset patch
# (patch_immich_archive_people_frontend.py, fixes the person-detail page itself only showing
# timeline-visibility photos).
#
# Both patches live in the container's writable layer, not the image, so every restart (and
# every upgrade, which recreates the container from a brand-new unpatched image) loses them.
# This is the same watch-and-react pattern as restart-on-immich-restart.sh, just reacting with
# a re-patch instead of a captioner bounce.
#
# Both patches are idempotent and each may issue its own restart the first time it actually
# applies something (the backend patch always restarts to load its change; the frontend patch
# restarts only for the index.html hop of its rename cascade, since that file is served from
# an in-memory copy read at startup, unlike the plain-static chunk files it renames first).
# Either restart re-triggers this same "start" event, but on the next pass each patch finds
# its own target already applied and does nothing further -- so two patches each
# self-restarting once converges in at most a few passes, not an unbounded loop.
set -euo pipefail

IMMICH_CONTAINER="${IMMICH_CONTAINER:-immich_server}"
HEALTH_TIMEOUT_SECONDS="${HEALTH_TIMEOUT_SECONDS:-300}"
HEALTH_POLL_SECONDS="${HEALTH_POLL_SECONDS:-5}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_SCRIPT="$SCRIPT_DIR/patch_immich_archive_people.py"
FRONTEND_PATCH_SCRIPT="$SCRIPT_DIR/patch_immich_archive_people_frontend.py"

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

log "Watching for '$IMMICH_CONTAINER' restarts to re-apply the archive-people patch..."

docker events --filter "container=$IMMICH_CONTAINER" --filter "event=start" --format '{{.Time}}' |
while read -r _; do
    log "$IMMICH_CONTAINER started; waiting for it to report healthy..."
    if wait_for_healthy; then
        log "$IMMICH_CONTAINER healthy; applying archive-people patch"
    else
        log "Timed out waiting for $IMMICH_CONTAINER to become healthy after ${HEALTH_TIMEOUT_SECONDS}s; attempting patch anyway"
    fi
    if python3 "$PATCH_SCRIPT" --container "$IMMICH_CONTAINER"; then
        log "backend patch check/apply completed successfully"
    else
        log "BACKEND PATCH FAILED -- see output above. immich_server is running UNPATCHED (People page will undercount archived-only people again). This needs a human to check whether upstream Immich changed the query this patch targets."
    fi
    if python3 "$FRONTEND_PATCH_SCRIPT" --container "$IMMICH_CONTAINER"; then
        log "frontend patch check/apply completed successfully"
    else
        log "FRONTEND PATCH FAILED -- see output above. The person-detail page may only show timeline-visibility photos again. This needs a human to check whether upstream Immich changed the code this patch targets."
    fi
done
