# shellcheck shell=bash
# shellcheck disable=SC2034  # the variables are used by the scripts sourcing this
# lib.sh - shared helpers for the Jetstream2 CREST-iMAP GPU worker kit.
# Sourced by bootstrap.sh, update.sh, healthcheck.sh and crest-py; never run.
# Runbook: docs/JETSTREAM2_WORKER.md

# ---------------------------------------------------------------- layout --
CREST_HOME=/opt/crest-worker          # venv/ releases/<stamp>/ current previous bin/
CREST_ETC=/etc/crest-worker           # crest-worker.env (no secrets), hf_token (root 0600)
CREST_VAR=/var/lib/crest-worker       # service user's HOME
CREST_USER=crest
CREST_UNIT=crest-worker.service
CREST_ENV_FILE=$CREST_ETC/crest-worker.env
CREST_TOKEN_FILE=$CREST_ETC/hf_token
# systemd LoadCredential= exposes the root-only token file to the service
# (and only to it) at this fixed path while the unit runs.
CREST_CRED_PATH=/run/credentials/$CREST_UNIT/hf_token
VENV=$CREST_HOME/venv
PY=$VENV/bin/python

# ------------------------------------------------------- safety floors --
# Production solver pin: the CPU Spaces' Dockerfile CRESTIMAP_REF and the
# crestimap SHA stamped on every HPC-published manifest. Contains the EF5
# stage-floor fix (4c55c0f6 itself), basin domains (678cbf17), cold re-run
# stamps (2e210d2b) and maxspeed/flowdir (e8652781). Nothing older may run.
FORK_MIN_SHA=4c55c0f6b9e28ab56b7eb4ad2eb328350be8e06f
# Minimum CREST_AI: 939800a added events/pinned.json protection and dropped
# the 30-day/monthly roll-off; 731ab17 is GitHub main == deployed Spaces.
CREST_AI_MIN_SHA=731ab174009a2ba96fe8b81b5e1499e1a20da524
# Solver physics the PI has approved (2026-10-20: draining-time outflow
# limiter + interflow store, fork v2 c5184759; the ZeroGPU worker runs it).
SOLVER_APPROVED_SHA=c518475956c933a9c0fc46962b3825685a54b181
# Files that make up the worker's solve path. A FORK_REF whose copies differ
# from SOLVER_APPROVED_SHA changes solver physics vs the approved code and is
# refused unless the env file sets ALLOW_SOLVER_CHANGE=1 (a user decision).
SOLVER_PATH_FILES=(crestimap/worker.py crestimap/session.py crestimap/event.py
                   crestimap/forcing.py crestimap/solver.py crestimap/tiled.py
                   crestimap/dem.py crestimap/domain.py crestimap/io.py)

# --------------------------------------------------------------- output --
_ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log()  { printf '[%s] %s\n' "$(_ts)" "$*"; }
warn() { printf '[%s] WARNING: %s\n' "$(_ts)" "$*" >&2; }
die()  { printf '[%s] ERROR: %s\n' "$(_ts)" "$*" >&2; exit 1; }

require_root() { [ "$(id -u)" -eq 0 ] || die "run this with sudo"; }

# on_volume_unmounted <path>: true when <path> is meant to live on an attached
# volume (Exosphere mounts them at /media/volume/<name>) but the volume is not
# mounted, so <path> would silently land on the 60 GB root disk.
on_volume_unmounted() {
  local p=$1 t
  case $p in /media/volume/*) ;; *) return 1 ;; esac
  # findmnt -T fails on a path that does not exist yet (a fresh volume has
  # no jobs/ dir): test the deepest existing ancestor instead
  while [ ! -e "$p" ] && [ "$p" != / ]; do p=$(dirname "$p"); done
  t=$(findmnt -n -o TARGET -T "$p" 2> /dev/null || echo /)
  [ "$t" = / ]
}

# ------------------------------------------------------------- env file --
# KEY=VALUE parser with systemd EnvironmentFile= semantics for the simple
# subset this kit uses: whole-line comments, optional surrounding quotes,
# no variable expansion, no inline comments. The file is never `source`d,
# so nothing in it can execute. Keys land in CREST_ENV_KEYS.
CREST_ENV_KEYS=()
load_env_file() {
  local f=$1 line key val
  [ -r "$f" ] || die "cannot read $f"
  while IFS= read -r line || [ -n "$line" ]; do
    line=${line%$'\r'}
    case $line in '' | '#'* | ';'*) continue ;; esac
    if [[ ! $line =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
      warn "ignoring malformed line in $f: $line"
      continue
    fi
    key=${BASH_REMATCH[1]}
    val=${BASH_REMATCH[2]}
    if [[ $val =~ ^\"(.*)\"$ ]] || [[ $val =~ ^\'(.*)\'$ ]]; then
      val=${BASH_REMATCH[1]}
    fi
    export "$key=$val"
    CREST_ENV_KEYS+=("$key")
  done < "$f"
}

# ----------------------------------------------------- run as the worker --
# (cd / first: the caller's cwd, e.g. /home/exouser, may be unreadable to it)
as_svc() { (cd / && runuser -u "$CREST_USER" -- env HOME="$CREST_VAR" "$@"); }
svc_git() { as_svc git "$@"; }

# Clean-environment Python as the worker user (no token, no root env).
# usage: svc_py <cwd> <python args...>
svc_py() {
  local cwd=$1
  shift
  (cd "$cwd" && runuser -u "$CREST_USER" -- env -i PATH=/usr/local/bin:/usr/bin:/bin \
    HOME="$CREST_VAR" LANG=C.UTF-8 PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_IMPLICIT_TOKEN=1 "$PY" "$@")
}

# ------------------------------------------------------------------ GPU --
gpu_wait() {  # gpu_wait <seconds>: nvidia-smi answers (driver loaded)
  local deadline=$((SECONDS + ${1:-120}))
  until nvidia-smi -L > /dev/null 2>&1; do
    [ "$SECONDS" -lt "$deadline" ] || return 1
    sleep 5
  done
}
gpu_mem_mib() { nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -d ' '; }
gpu_driver() { nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | tr -d ' '; }
# gpu_virt_mode: "VGPU" on a partial-GPU flavor (g3.small..g3.large, NVIDIA
# vGPU guest driver), "Pass-Through" on g3.xl, "None" on bare metal.
gpu_virt_mode() {  # (|| true: never trip set -e/pipefail in the callers)
  { nvidia-smi -q 2> /dev/null | sed -n 's/^[[:space:]]*Virtualization Mode[[:space:]]*:[[:space:]]*//p' | head -n1; } || true
}
# gpu_vgpu_license: the "License Status" line under "vGPU Software Licensed
# Product", e.g. "Licensed (Expiry: ...)", "Unlicensed (Restricted)"; empty
# when the driver does not report one. An unlicensed vGPU runs CUDA degraded
# after 20 min and CUDA calls FAIL after 24 h (NVIDIA vGPU licensing guide):
# every job taken then is consumed as failed.json.
gpu_vgpu_license() {  # (grep finds nothing on passthrough: || true)
  { nvidia-smi -q 2> /dev/null | grep -A3 'vGPU Software Licensed Product' |
    sed -n 's/^[[:space:]]*License Status[[:space:]]*:[[:space:]]*//p' | head -n1; } || true
}
# Largest per-device cell budget for a card: 15 % headroom, 1 GiB CUDA
# context, <bytes_per_cell> peak solver bytes per bounding-box cell.
safe_cells() {
  local mib=$1 bpc=$2
  echo $(((mib * 1048576 * 85 / 100 - 1073741824) / bpc))
}

# ------------------------------------------------------------- releases --
# A release is releases/<UTC stamp>/{CREST-iMAP,CREST_AI,RELEASE.txt}: blobless
# sparse git clones (crestimap/ and hf_data/ only) checked out detached at an
# exact commit and never modified afterwards. `current` points at the one the
# service runs; run-worker.sh resolves it once at start, so a later update
# never changes files under a running process.
clone_at() {  # clone_at <url> <dest> <ref> <sparse dir>
  local url=$1 dest=$2 ref=$3 sub=$4 sha
  svc_git clone --quiet --filter=blob:none --no-checkout "$url" "$dest"
  # `init --cone` + `set`: Ubuntu 22.04's git 2.34 rejects `set --cone`
  # (init and set were only unified in git 2.35); this form works on both.
  svc_git -C "$dest" sparse-checkout init --cone
  svc_git -C "$dest" sparse-checkout set "$sub"
  sha=$(svc_git -C "$dest" rev-parse --verify --quiet "origin/$ref^{commit}") ||
    sha=$(svc_git -C "$dest" rev-parse --verify --quiet "$ref^{commit}") ||
    die "ref '$ref' not found in $url"
  svc_git -C "$dest" -c advice.detachedHead=false checkout --quiet --detach "$sha"
}

release_field() { sed -n "s/^$2=//p" "$1" 2> /dev/null | head -n1; }

# hfdata_tree <release dir>: git tree id of the release's hf_data/ (what the
# worker imports). update.sh restarts only when this or the fork SHA changes,
# so a CREST_AI main commit outside hf_data/ does not cost the resident
# sessions (RAM-only) a restart.
hfdata_tree() { svc_git -C "$1/CREST_AI" rev-parse --verify --quiet 'HEAD:hf_data'; }

# make_release: builds a new release dir from FORK_REF / CREST_AI_REF and
# leaves its path in NEW_RELEASE. Verification is separate (verify_release).
NEW_RELEASE=
make_release() {
  local stamp dir
  : "${FORK_REPO:?FORK_REPO missing from $CREST_ENV_FILE}" "${FORK_REF:?}"
  : "${CREST_AI_REPO:?CREST_AI_REPO missing from $CREST_ENV_FILE}" "${CREST_AI_REF:?}"
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  dir=$CREST_HOME/releases/$stamp
  install -d -o "$CREST_USER" -g "$CREST_USER" -m 0755 "$CREST_HOME/releases" "$dir"
  log "release $stamp: fork $FORK_REPO @ $FORK_REF"
  clone_at "$FORK_REPO" "$dir/CREST-iMAP" "$FORK_REF" crestimap
  log "release $stamp: CREST_AI $CREST_AI_REPO @ $CREST_AI_REF (hf_data only)"
  clone_at "$CREST_AI_REPO" "$dir/CREST_AI" "$CREST_AI_REF" hf_data
  NEW_RELEASE=$dir
}

# verify_release <dir> [record]: refuses (exit) on anything that would make the
# worker unsafe to publish with. With `record` (fresh builds only) it writes
# RELEASE.txt; re-verifying an existing release never rewrites it.
verify_release() {
  local d=$1 mode=${2:-} fork_sha ai_sha changed=0
  [ -f "$d/CREST-iMAP/crestimap/worker.py" ] || die "$d: crestimap/worker.py missing"
  [ -f "$d/CREST_AI/hf_data/eventstore.py" ] || die "$d: hf_data/eventstore.py missing"
  fork_sha=$(svc_git -C "$d/CREST-iMAP" rev-parse HEAD)
  ai_sha=$(svc_git -C "$d/CREST_AI" rev-parse HEAD)
  svc_git -C "$d/CREST-iMAP" merge-base --is-ancestor "$FORK_MIN_SHA" "$fork_sha" ||
    die "fork $fork_sha predates the production pin $FORK_MIN_SHA - refusing"
  if ! svc_git -C "$d/CREST-iMAP" diff --quiet "$SOLVER_APPROVED_SHA" "$fork_sha" -- "${SOLVER_PATH_FILES[@]}"; then
    changed=1
    if [ "${ALLOW_SOLVER_CHANGE:-0}" != 1 ]; then
      svc_git -C "$d/CREST-iMAP" diff --stat "$SOLVER_APPROVED_SHA" "$fork_sha" -- "${SOLVER_PATH_FILES[@]}" >&2 || true
      die "FORK_REF $fork_sha changes the solver path vs the approved $SOLVER_APPROVED_SHA (above). That is a physics change: set ALLOW_SOLVER_CHANGE=1 in $CREST_ENV_FILE only after the user approves it."
    fi
    warn "solver path differs from the approved $SOLVER_APPROVED_SHA (ALLOW_SOLVER_CHANGE=1)"
  fi
  svc_git -C "$d/CREST_AI" merge-base --is-ancestor "$CREST_AI_MIN_SHA" "$ai_sha" ||
    die "CREST_AI $ai_sha is older than $CREST_AI_MIN_SHA - its retention code would strip pinned demo events; refusing"
  grep -q '^def pinned_ids' "$d/CREST_AI/hf_data/eventstore.py" ||
    die "REFUSING: $d/CREST_AI/hf_data/eventstore.py lacks 'def pinned_ids' (old code would strip the pinned demo events)"
  grep -q 'EVENT_DEMOTE' "$d/CREST_AI/hf_data/eventstore.py" ||
    die "REFUSING: eventstore.py lacks the KEEP_FULL demotion clamp (EVENT_DEMOTE)"
  if [ "$mode" = record ]; then
    {
      echo "RELEASE_BUILT=$(_ts)"
      echo "FORK_URL=$FORK_REPO"
      echo "FORK_REF=$FORK_REF"
      echo "FORK_SHA=$fork_sha"
      echo "CREST_AI_URL=$CREST_AI_REPO"
      echo "CREST_AI_REF=$CREST_AI_REF"
      echo "CREST_AI_SHA=$ai_sha"
      echo "SOLVER_PATH_CHANGED=$changed"
    } > "$d/RELEASE.txt"
    chown "$CREST_USER:$CREST_USER" "$d/RELEASE.txt"
  elif [ "$(release_field "$d/RELEASE.txt" FORK_SHA)" != "$fork_sha" ] ||
    [ "$(release_field "$d/RELEASE.txt" CREST_AI_SHA)" != "$ai_sha" ]; then
    die "$d/RELEASE.txt does not match its checkouts - rebuild with update.sh"
  fi
  log "verified $(basename "$d"): fork ${fork_sha:0:12}  CREST_AI ${ai_sha:0:12}  pinned_ids present"
}

# selftest_release <dir>: imports the whole worker path as the service user,
# exactly as the service will (release on sys.path, no token).
selftest_release() {
  local d=$1 code
  code=$(
    cat << 'PYEOF'
import os, sys
rel = sys.argv[1]
sys.path[:0] = [os.path.join(rel, "CREST-iMAP"), os.path.join(rel, "CREST_AI")]
import crestimap, crestimap.worker, crestimap.session
assert os.path.realpath(crestimap.__file__).startswith(os.path.realpath(rel)), \
    f"crestimap imported from {crestimap.__file__}, not the release"
from hf_data import eventsim, eventstore, eventqueue, provenance
assert callable(getattr(eventstore, "pinned_ids", None)), "eventstore.pinned_ids missing"
import numpy, rasterio, pyarrow, PIL, requests, torch, huggingface_hub
from huggingface_hub.utils import EntryNotFoundError  # used by pinned_ids
fs = provenance._git_sha(os.path.join(rel, "CREST-iMAP"))
ai = provenance._git_sha(os.path.join(rel, "CREST_AI"))
assert fs and ai, f"provenance cannot read git SHAs (fork={fs}, crest_ai={ai})"
print(f"imports OK: crestimap={fs} crest_ai={ai} torch={torch.__version__} "
      f"(CUDA {torch.version.cuda}) hub={huggingface_hub.__version__} "
      f"rasterio={rasterio.__version__} numpy={numpy.__version__}")
PYEOF
  )
  svc_py "$d/CREST-iMAP" -c "$code" "$d" || die "import self-test failed for $d"
}

# activate_release <dir>: atomically repoint `current` (old one -> `previous`).
activate_release() {
  local d=$1 cur
  cur=$(readlink -e "$CREST_HOME/current" || true)
  if [ -n "$cur" ] && [ "$cur" != "$d" ]; then
    ln -sfn "$cur" "$CREST_HOME/previous"
  fi
  ln -sfn "$d" "$CREST_HOME/current.new"
  mv -Tf "$CREST_HOME/current.new" "$CREST_HOME/current"
  log "current -> $d"
}

# prune_releases [keep]: delete all but the newest <keep> (and current/previous).
prune_releases() {
  local keep=${1:-3} cur prev d n=0
  cur=$(readlink -e "$CREST_HOME/current" || true)
  prev=$(readlink -e "$CREST_HOME/previous" || true)
  while IFS= read -r d; do
    [ -d "$d" ] || continue
    n=$((n + 1))
    if [ "$n" -le "$keep" ] || [ "$d" = "$cur" ] || [ "$d" = "$prev" ]; then
      continue
    fi
    log "pruning old release $d"
    rm -rf -- "$d"
  done < <(printf '%s\n' "$CREST_HOME"/releases/* | sort -r)
}

# ---------------------------------------------------------------- queue --
# own_claim_busy: 0 if THIS host holds a fresh queue claim, 3 if idle,
# anything else = could not tell. Read-only (anonymous).
own_claim_busy() {
  local here
  here=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
  "$here/crest-py" "$here/queue_status.py" --own > /dev/null 2>&1
}
