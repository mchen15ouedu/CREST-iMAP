#!/usr/bin/env bash
# run-worker.sh - ExecStart of crest-worker.service (runs as the `crest` user).
# Starts the EXISTING headless worker, `python -m crestimap.worker`, pinned to
# the release `current` points at right now (resolved once: a later update
# can never change modules under this process; it needs a restart).
set -euo pipefail

CREST_HOME=${CREST_HOME:-/opt/crest-worker}
REL=$(readlink -e "$CREST_HOME/current") || {
  echo "run-worker: no current release under $CREST_HOME" >&2
  exit 1
}

# The token is read by crestimap.worker from ~/huggingface.txt (a symlink to
# the systemd credential). Never take it from the environment.
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
if [ ! -s "$HOME/huggingface.txt" ]; then
  echo "run-worker: ~/huggingface.txt does not resolve to a non-empty token credential" >&2
  exit 1
fi

: "${EVENT_MAX_CELLS_GPU:?set in /etc/crest-worker/crest-worker.env}"
: "${WORKER_WORK_ROOT:?set in /etc/crest-worker/crest-worker.env}"
: "${EVENT_DEM_CACHE:?set in /etc/crest-worker/crest-worker.env}"
# Provenance engine label (crestimap.worker only setdefault()s "hpc:...").
if [ -z "${EVENT_ENGINE_IDENT:-}" ]; then
  EVENT_ENGINE_IDENT="jetstream2:$(hostname)"
fi
export EVENT_ENGINE_IDENT

# WORKER_EXTRA_ARGS is an ALLOW-list: only `--poll-s N` (30..3600). argparse
# accepts any unique prefix (--on = --once, --crest = --crest-ai, --max = an
# unchecked --max-cells) and the last value wins, so a deny-list cannot keep
# a flag from overriding the checked settings above. Same rule in preflight.sh.
# shellcheck disable=SC2206  # WORKER_EXTRA_ARGS is a plain flag list
extra=(${WORKER_EXTRA_ARGS:-})
i=0
while [ "$i" -lt "${#extra[@]}" ]; do
  a=${extra[$i]} v=
  case $a in
    --poll-s) v=${extra[$((i + 1))]:-} i=$((i + 1)) ;;
    --poll-s=*) v=${a#--poll-s=} ;;
  esac
  if [[ ! $v =~ ^[0-9]{1,4}$ ]] || [ "$v" -lt 30 ] || [ "$v" -gt 3600 ]; then
    echo "run-worker: refusing WORKER_EXTRA_ARGS '${WORKER_EXTRA_ARGS:-}': only '--poll-s N' (30-3600) is allowed (see crest-worker.env)" >&2
    exit 1
  fi
  i=$((i + 1))
done

cd "$REL/CREST-iMAP"
echo "run-worker: release $REL ($(sed -n 's/^FORK_SHA=//p' "$REL/RELEASE.txt" | cut -c1-12) / CREST_AI $(sed -n 's/^CREST_AI_SHA=//p' "$REL/RELEASE.txt" | cut -c1-12)) engine=$EVENT_ENGINE_IDENT"
exec "$CREST_HOME/venv/bin/python" -u -m crestimap.worker \
  --repo "${CREST_DATA_REPO:-vincewin/CREST_data}" \
  --crest-ai "$REL/CREST_AI" \
  --work-root "$WORKER_WORK_ROOT" \
  --device "${EVENT_DEVICE:-cuda}" \
  --max-cells "$EVENT_MAX_CELLS_GPU" \
  --dt-every "${EVENT_DT_EVERY:-1}" \
  --max-sessions "${EVENT_MAX_SESSIONS:-6}" \
  --evict-idle-h "${EVENT_EVICT_IDLE_H:-12}" \
  --sessions --publish "${extra[@]}"
