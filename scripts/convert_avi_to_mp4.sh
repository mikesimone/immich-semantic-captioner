#!/usr/bin/env bash
# Convert unstreamable AVI originals to MP4 in place (same asset row), so Immich's real-time HLS
# player can index and stream them. Mike asked for this 2026-09-26 ("Convert").
#
# Why: HLS works from the ORIGINAL file and needs an asset_keyframe row; AVI has no PTS, so no
# row is ever written (and /opt/immich/patches/reencode-inplace.sh keeps the .avi name, so the
# re-encoded AVIs still got no row -- the keyframe sweep excludes AVIs for that reason).
#
# Per asset: back up the .avi -> encode (NVENC in immich_server) -> verify duration + real packet
# durations -> write <stem>.mp4 next to it -> update asset.originalPath / originalFileName /
# checksum, asset_exif.fileSizeInByte, and the sidecar (<name>.avi.xmp -> <name>.mp4.xmp) ->
# delete the .avi -> queue refresh-metadata -> wait for the asset_keyframe row. The asset row is
# kept, so albums, tags, captions, archive state and faces survive.
#
# Backups: $BACKUP_DIR/<path under the library> -- kept until Mike says they can go.
# Usage: scripts/convert_avi_to_mp4.sh <asset-id> [<asset-id> ...]   (DRY_RUN=1 to only print)
set -uo pipefail
set +u; . "$HOME/.api-keys"; set -u
: "${IMMICH_URL:?}"; : "${IMMICH_API_KEY:?}"
BACKUP_DIR="${BACKUP_DIR:-/mnt/nvme/immich/avi-backup-20260926}"
HOST_ROOT=/mnt/nvme/immich/library   # host side of immich_server's /data
PSQL=(docker exec immich_postgres psql -U postgres -d immich -At -v ON_ERROR_STOP=1)
DEX=(docker exec immich_server)
FF=/usr/lib/jellyfin-ffmpeg/ffmpeg; FP=/usr/lib/jellyfin-ffmpeg/ffprobe
sql() { "${PSQL[@]}" -c "$1"; }
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
ok=0; fail=0

for id in "$@"; do
  path=$(sql "select \"originalPath\" from asset where id='$id' and \"deletedAt\" is null")
  [ -z "$path" ] && { log "SKIP $id: not found/trashed"; fail=$((fail+1)); continue; }
  case "${path,,}" in *.avi) ;; *) log "SKIP $id: not an .avi ($path)"; continue ;; esac
  host="${path/#\/data/$HOST_ROOT}"
  [ -f "$host" ] || { log "SKIP $id: missing on disk"; fail=$((fail+1)); continue; }
  newpath="${path%.*}.mp4"; newhost="${host%.*}.mp4"
  [ -e "$newhost" ] && { newpath="${path%.*} (converted).mp4"; newhost="${host%.*} (converted).mp4"; }
  ref=$(awk -v ms="$(sql "select coalesce(duration,0) from asset where id='$id'")" 'BEGIN{printf "%.3f", ms/1000}')
  bytes=$(stat -c %s "$host")
  maxrate=$(awk -v b="$bytes" -v d="$ref" 'BEGIN{ if(d<=0){print 6000; exit} r=(b*8/d/1000)*1.15; if(r<1200)r=1200; if(r>16000)r=16000; printf "%d", r }')
  log "$id  $(basename "$path")  ref=${ref}s  -> $(basename "$newpath")"
  [ "${DRY_RUN:-0}" = "1" ] && continue

  bdst="$BACKUP_DIR/${host#$HOST_ROOT/}"; mkdir -p "$(dirname "$bdst")"
  cp -p "$host" "$bdst" && [ "$(stat -c %s "$bdst")" = "$bytes" ] || { log "  FAIL backup"; fail=$((fail+1)); continue; }

  tmp="${path%.*}.convert-tmp.mp4"
  if ! "${DEX[@]}" "$FF" -v error -nostdin -fflags +genpts -i "$path" -map 0:v:0 -map "0:a:0?" \
        -c:v h264_nvenc -preset p5 -rc vbr -cq 25 -b:v 0 -maxrate "${maxrate}k" -bufsize "$((maxrate*2))k" \
        -pix_fmt yuv420p -c:a aac -b:a 160k -movflags +faststart "$tmp" -y; then
    log "  FAIL encode"; "${DEX[@]}" rm -f "$tmp"; fail=$((fail+1)); continue; fi
  new=$("${DEX[@]}" "$FP" -v error -show_entries format=duration -of csv=p=0 "$tmp" | tr -d '\r,')
  na=$("${DEX[@]}" "$FP" -v error -select_streams v:0 -read_intervals '%+#5' -show_entries packet=duration -of csv=p=0 "$tmp" | grep -c 'N/A')
  good=$(awk -v a="$ref" -v b="$new" -v na="$na" 'BEGIN{ if(b<=0||na>0){print 0;exit} if(a<=0){print 1;exit} d=(a>b?a-b:b-a); print (d/a<0.02)?1:0 }')
  [ "$good" = "1" ] || { log "  FAIL verify (ref=$ref new=$new na=$na)"; "${DEX[@]}" rm -f "$tmp"; fail=$((fail+1)); continue; }

  "${DEX[@]}" sh -c "mv -f '$tmp' '$newpath' && chown root:root '$newpath' && chmod 644 '$newpath'" || { log "  FAIL move"; fail=$((fail+1)); continue; }
  sha=$(sha1sum "$newhost" | cut -d' ' -f1); nb=$(stat -c %s "$newhost")
  newname=$(basename "$newpath" | sed "s/'/''/g"); np=$(printf '%s' "$newpath" | sed "s/'/''/g")
  sql "update asset set \"originalPath\"='$np', \"originalFileName\"='$newname', checksum=decode('$sha','hex') where id='$id'" >/dev/null
  sql "update asset_exif set \"fileSizeInByte\"=$nb where \"assetId\"='$id'" >/dev/null
  side=$(sql "select path from asset_file where \"assetId\"='$id' and type='sidecar'")
  if [ -n "$side" ]; then
    nside="${newpath}.xmp"
    "${DEX[@]}" sh -c "[ -f '$side' ] && mv -f '$side' '$nside'" && \
      sql "update asset_file set path='$(printf '%s' "$nside" | sed "s/'/''/g")' where \"assetId\"='$id' and type='sidecar'" >/dev/null
  fi
  "${DEX[@]}" rm -f "$path"
  curl -sS -o /dev/null -X POST "$IMMICH_URL/api/assets/jobs" -H "x-api-key: $IMMICH_API_KEY" -H 'Content-Type: application/json' \
    -d "{\"assetIds\":[\"$id\"],\"name\":\"refresh-metadata\"}"
  kf=0; for _ in $(seq 1 36); do sleep 5; kf=$(sql "select count(*) from asset_keyframe where \"assetId\"='$id'"); [ "$kf" != "0" ] && break; done
  if [ "$kf" != "0" ]; then log "  OK  $(awk -v b=$nb 'BEGIN{printf "%.0fMB", b/1048576}') ${new}s, keyframe index present, backup: $bdst"; ok=$((ok+1))
  else log "  CONVERTED but no keyframe row after 3 min (check later), backup: $bdst"; fail=$((fail+1)); fi
done
log "done: ok=$ok fail=$fail"
