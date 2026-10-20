#!/usr/bin/env bash
# preflight.sh - ExecStartPre of crest-worker.service (runs as `crest`).
# Exits non-zero (the worker does not start) on anything unsafe. Its output
# is in `journalctl -u crest-worker` (lines prefixed "preflight:").
set -euo pipefail

CREST_HOME=${CREST_HOME:-/opt/crest-worker}
fail() {
  echo "preflight: REFUSING TO START: $*" >&2
  exit 1
}
note() { echo "preflight: $*"; }

# 1. token: present as the systemd credential, never in the environment
[ -z "${HF_TOKEN:-}" ] || fail "HF_TOKEN is set in the environment; remove it (the token comes only from /etc/crest-worker/hf_token)"
if [ -n "${CREDENTIALS_DIRECTORY:-}" ]; then
  [ -s "$CREDENTIALS_DIRECTORY/hf_token" ] || fail "credential hf_token is empty or missing"
  # crestimap.worker reads ~/huggingface.txt; (re)point it at this start's
  # credential dir (systemd owns that path; never hard-code it)
  ln -sfn "$CREDENTIALS_DIRECTORY/hf_token" "$HOME/huggingface.txt" ||
    fail "cannot update $HOME/huggingface.txt"
  [ "$(readlink -f "$HOME/huggingface.txt")" = "$(readlink -f "$CREDENTIALS_DIRECTORY/hf_token")" ] ||
    fail "$HOME/huggingface.txt does not resolve to the token credential"
else
  note "not under systemd (no CREDENTIALS_DIRECTORY): token checks skipped"
fi

# 2. retention policy belongs to the Spaces. eventstore reads these at import
#    and would apply them FROM THE WORKER (EVENT_KEEP_FULL=8 on the HPC once
#    stripped the animations of 25 of 33 listed events).
for v in EVENT_DEMOTE EVENT_KEEP_FULL EVENT_KEEP EVENT_MAX_AGE_D EVENT_LIST_MONTHLY \
  EVENT_RESULTS_MAX_AGE_D EVENT_PINNED_TTL_S SPACE_ID; do
  [ -z "${!v+x}" ] || fail "$v is set; never set retention/Space variables on the worker"
done

# 3. extra worker flags: an ALLOW-list, only `--poll-s N` (30..3600). argparse
#    takes unique prefixes (--on = --once, --crest = --crest-ai, --max-cells
#    past the VRAM check below) and the last value wins, so nothing else may
#    be appended to the command line run-worker.sh builds.
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
    fail "WORKER_EXTRA_ARGS='${WORKER_EXTRA_ARGS:-}': only '--poll-s N' (30-3600) is allowed. Shadow/one-shot runs (--no-publish, --once) hold live claims - use replay_bundle.py; budgets and paths come from the env file."
  fi
  i=$((i + 1))
done

# 4. GPU: wait for the driver (gpu-driver-fix.service runs at boot), then
#    check the cell budget against what the card can hold
deadline=$((SECONDS + 300))
until nvidia-smi -L > /dev/null 2>&1; do
  [ "$SECONDS" -lt "$deadline" ] || fail "nvidia-smi did not answer within 300 s (driver not loaded?)"
  sleep 5
done
# vGPU flavors (g3.large and smaller): an unlicensed vGPU degrades CUDA after
# 20 min and fails every CUDA call after 24 h, which would consume each job as
# failed.json. The license is acquired shortly after boot: wait for it within
# the same deadline; refuse only on an explicit non-"Licensed" status.
vmode=$(nvidia-smi -q 2> /dev/null | sed -n 's/^[[:space:]]*Virtualization Mode[[:space:]]*:[[:space:]]*//p' | head -n1) || vmode=
if [ "$vmode" = VGPU ]; then
  while :; do
    # (grep exits 1 when the section is absent: keep set -e/pipefail quiet)
    lic=$(nvidia-smi -q 2> /dev/null | grep -A3 'vGPU Software Licensed Product' |
      sed -n 's/^[[:space:]]*License Status[[:space:]]*:[[:space:]]*//p' | head -n1) || lic=
    case $lic in
      Licensed*)
        note "vGPU license: $lic"
        break
        ;;
      '')
        note "WARNING: vGPU guest but nvidia-smi reports no License Status; not checked"
        break
        ;;
    esac
    [ "$SECONDS" -lt "$deadline" ] ||
      fail "the vGPU is not licensed ($lic): CUDA would degrade, then fail and consume jobs as failed.json. Ask help@jetstream-cloud.org."
    sleep 10
  done
fi
mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -d ' ')
[[ $mib =~ ^[0-9]+$ ]] || fail "could not read GPU memory (got '$mib')"
cells=${EVENT_MAX_CELLS_GPU:?EVENT_MAX_CELLS_GPU unset}
bpc=${BYTES_PER_CELL_EST:-350}
[[ $cells =~ ^[0-9]+$ && $bpc =~ ^[0-9]+$ && $bpc -gt 0 ]] || fail "EVENT_MAX_CELLS_GPU / BYTES_PER_CELL_EST must be integers"
safe=$(((mib * 1048576 * 85 / 100 - 1073741824) / bpc))
[ "$cells" -le "$safe" ] ||
  fail "EVENT_MAX_CELLS_GPU=$cells exceeds what this ${mib} MiB GPU holds (~$safe at $bpc B/cell, 15 % headroom). A CUDA OOM consumes the job as failed.json - lower it."
note "GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1), ${mib} MiB; budget $cells cells (ceiling ~$safe)"

# 5. code: the release the worker will import must carry pinned_ids
rel=$(readlink -e "$CREST_HOME/current") || fail "no current release (run bootstrap.sh)"
[ -f "$rel/CREST-iMAP/crestimap/worker.py" ] || fail "$rel has no crestimap/worker.py"
grep -q '^def pinned_ids' "$rel/CREST_AI/hf_data/eventstore.py" ||
  fail "$rel/CREST_AI/hf_data/eventstore.py lacks def pinned_ids (would strip pinned demo events)"
[ -s "$rel/RELEASE.txt" ] || fail "$rel/RELEASE.txt missing (release was never verified)"
note "release $(basename "$rel"): fork $(sed -n 's/^FORK_SHA=//p' "$rel/RELEASE.txt" | cut -c1-12), CREST_AI $(sed -n 's/^CREST_AI_SHA=//p' "$rel/RELEASE.txt" | cut -c1-12)"

# 6. disk: data dirs exist (a volume may not be mounted yet) and have room
for d in "${WORKER_WORK_ROOT:?}" "${EVENT_DEM_CACHE:?}" "${HF_HOME:?}"; do
  case $d in
    /media/volume/*)
      # data meant for an attached volume must not land on the root disk
      # (e.g. after an unshelve without re-attaching the volume)
      if [ "$(findmnt -n -o TARGET -T "$d" 2> /dev/null || echo /)" = / ]; then
        fail "$d is under /media/volume/ but no volume is mounted there - re-attach the volume in Exosphere, then: systemctl reset-failed crest-worker && systemctl start crest-worker"
      fi
      ;;
  esac
  if [ ! -d "$d" ] || [ ! -w "$d" ]; then
    fail "$d is missing or not writable (volume not mounted? re-run bootstrap.sh)"
  fi
done
free_gb=$(df -P -BG "$WORKER_WORK_ROOT" | awk 'NR==2 {gsub("G", "", $4); print $4}')
if [ "${free_gb:-0}" -lt "${DISK_MIN_FREE_GB:-12}" ]; then
  echo "preflight: WARNING: only ${free_gb} GB free under $WORKER_WORK_ROOT" >&2
fi

# 7. resident sessions are RAM-only: their scratch dirs from a previous
#    process are useless now (the worker rebuilds from the bundle's init_depth)
find "$WORKER_WORK_ROOT" -mindepth 1 -maxdepth 1 -type d \( -name 'sess_*' -o -name 'job_*' \) \
  -exec rm -rf -- {} + 2> /dev/null || true

note "OK"
