#!/usr/bin/env bash
# Hourly: convert any AVI in a porn album (album numbers 100.x-500.x) to MP4 in place, because
# Immich's real-time HLS can't stream AVI (Mike, 2026-09-27: "DEATH TO AVI!"). Uses
# convert_avi_to_mp4.sh (albums/tags/captions kept). Originals are kept in $BACKUP_DIR for 14
# days, then pruned. Run by immich-avi-sweep.timer.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Never overlap a conversion already in progress (a manual batch, or a slow previous sweep).
if pgrep -f "convert_avi_to_mp4\.sh" >/dev/null; then echo "$(date -u +%FT%TZ) conversion already running; skipping"; exit 0; fi
export BACKUP_DIR="${BACKUP_DIR:-/mnt/nvme/immich/avi-backup-auto}"
ids=$(docker exec immich_postgres psql -U postgres -d immich -At -c "
  SELECT DISTINCT a.id FROM asset a JOIN album_asset aa ON aa.\"assetId\" = a.id JOIN album al ON al.id = aa.\"albumId\"
  WHERE a.type = 'VIDEO' AND a.\"deletedAt\" IS NULL AND lower(a.\"originalFileName\") LIKE '%.avi'
    AND al.\"albumName\" ~ '^[1-5][0-9][0-9]\.'")
[ -n "$ids" ] && "$HERE/convert_avi_to_mp4.sh" $ids
[ -d "$BACKUP_DIR" ] && find "$BACKUP_DIR" -type f -mtime +14 -delete && find "$BACKUP_DIR" -type d -empty -delete
exit 0
