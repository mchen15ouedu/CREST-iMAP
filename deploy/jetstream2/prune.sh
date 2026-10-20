#!/usr/bin/env bash
# prune.sh - daily cache housekeeping (crest-worker-prune.timer, runs as `crest`).
# The worker re-downloads every queue file with force_download, so the HF
# download cache only grows; DEM tiles are reused across events and are only
# dropped under disk pressure. Never touches job/session dirs of a running
# worker (preflight.sh clears orphans at service start).
set -euo pipefail
: "${HF_HOME:?}" "${EVENT_DEM_CACHE:?}" "${WORKER_WORK_ROOT:?}"
keep=${HF_CACHE_KEEP_DAYS:-2}
hub=$HF_HOME/hub

if [ -d "$hub" ]; then
  n=$(find "$hub" -type f -path '*/blobs/*' -mtime +"$keep" -print -delete | wc -l)
  find "$hub" -xtype l -delete 2> /dev/null || true # snapshot links to pruned blobs
  echo "prune: removed $n HF cache blob(s) older than $keep d"
fi

free_gb() { df -P -BG "$1" | awk 'NR==2 {gsub("G", "", $4); print $4}'; }
min=${DISK_MIN_FREE_GB:-12}
if [ -d "$EVENT_DEM_CACHE" ] && [ "$(free_gb "$EVENT_DEM_CACHE")" -lt "$min" ]; then
  n=$(find "$EVENT_DEM_CACHE" -type f -atime +"${DEM_KEEP_DAYS:-60}" -print -delete | wc -l)
  echo "prune: disk below $min GB free - removed $n DEM tile(s) unread for ${DEM_KEEP_DAYS:-60} d"
fi

for d in "$HF_HOME" "$EVENT_DEM_CACHE" "$WORKER_WORK_ROOT"; do
  if [ -d "$d" ]; then du -sh "$d" 2> /dev/null || true; fi
done
echo "prune: $(free_gb "$WORKER_WORK_ROOT") GB free under $WORKER_WORK_ROOT"
