#!/usr/bin/env bash
# healthcheck.sh - READ-ONLY health report for the Jetstream2 worker.
# Changes nothing locally or on Hugging Face (queue reads are anonymous).
#
#   sudo healthcheck.sh               GPU (+ vGPU license), service, update timer,
#                                     code versions, hf_data drift vs GitHub main
#                                     and the deployed Space, queue/claims,
#                                     publishes, pinned demo set, disk, SU, logs
#   sudo healthcheck.sh --since 24h   + soak counters from the journal (gate G4)
#   sudo healthcheck.sh --quick       skip everything that talks to Hugging Face
#                                     or GitHub
# Exit 0 = healthy, 1 = problem(s) listed under "PROBLEMS".
set -euo pipefail
here=$(dirname "$(readlink -f "$0")")
# shellcheck source=lib.sh
. "$here/lib.sh"
require_root
load_env_file "$CREST_ENV_FILE"

SINCE=24h QUICK=0
while [ $# -gt 0 ]; do
  case $1 in
    --since)
      SINCE=${2:?}
      shift
      ;;
    --quick) QUICK=1 ;;
    -h | --help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *) die "unknown option $1" ;;
  esac
  shift
done

problems=()
bad() {
  problems+=("$*")
  echo "  !! $*"
}
ok() { echo "  ok $*"; }
hdr() { printf '\n== %s ==\n' "$*"; }

echo "CREST-iMAP Jetstream2 worker health - $(hostname) - $(_ts)"

hdr "GPU"
if nvidia-smi -L > /dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu,temperature.gpu \
    --format=csv,noheader | sed 's/^/  /'
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2> /dev/null | sed 's/^/  process: /' || true
  vmode=$(gpu_virt_mode)
  lic=$(gpu_vgpu_license)
  if [ "$vmode" = VGPU ]; then
    case $lic in
      Licensed*) ok "vGPU license: $lic" ;;
      '') echo "  -- vGPU guest, but nvidia-smi -q reports no License Status (check: nvidia-smi -q | grep -A2 'vGPU Software Licensed Product')" ;;
      *) bad "vGPU NOT licensed ($lic): CUDA degrades after 20 min and fails after 24 h - every job taken then becomes failed.json. Stop the worker; ask help@jetstream-cloud.org" ;;
    esac
  else
    echo "  -- virtualization mode ${vmode:-?} (license check applies to vGPU flavors only)"
  fi
else
  bad "nvidia-smi does not answer (driver not loaded? after a kernel update: reboot)"
fi

hdr "service"
st=$(systemctl show "$CREST_UNIT" -p ActiveState --value 2> /dev/null || echo unknown)
en=$(systemctl is-enabled "$CREST_UNIT" 2> /dev/null || true)
pid=$(systemctl show "$CREST_UNIT" -p MainPID --value 2> /dev/null || echo 0)
echo "  $CREST_UNIT: $st ($en), MainPID $pid, restarts $(systemctl show "$CREST_UNIT" -p NRestarts --value 2> /dev/null || echo ?), since $(systemctl show "$CREST_UNIT" -p ActiveEnterTimestamp --value 2> /dev/null || echo ?)"
if [ "$st" = active ]; then
  ok "running"
elif [ "$en" = enabled ]; then
  bad "enabled but $st - journalctl -u $CREST_UNIT -n 60 (start-limit-hit? systemctl reset-failed $CREST_UNIT)"
else
  echo "  -- not enabled: expected before gate G2"
fi
echo "  prune timer: $(systemctl is-active crest-worker-prune.timer 2> /dev/null || true)"
ut=$(systemctl is-enabled crest-worker-update.timer 2> /dev/null || true)
echo "  update timer: $ut, next $(systemctl show crest-worker-update.timer -p NextElapseUSecRealtime --value 2> /dev/null || echo ?)"
if [ "$ut" != enabled ] && [ "${AUTO_UPDATE:-1}" = 1 ]; then
  bad "crest-worker-update.timer is $ut but AUTO_UPDATE=1: nothing refreshes hf_data (re-run bootstrap.sh)"
fi
if systemctl is-failed --quiet crest-worker-update.service 2> /dev/null; then
  bad "the last automatic update failed - journalctl -u crest-worker-update -n 60"
fi
if [ -s "$CREST_VAR/updates.log" ]; then
  echo "  last update: $(tail -n1 "$CREST_VAR/updates.log")"
fi
if [ -s "$CREST_VAR/update.blocked" ]; then
  bad "a release failed to stay up and was rolled back; auto-update skips that code ($CREST_VAR/update.blocked, see updates.log) - tell the user"
fi
if [ -e "$CREST_TOKEN_FILE" ]; then
  perm=$(stat -c '%U:%G %a' "$CREST_TOKEN_FILE")
  if [ "$perm" = "root:root 600" ]; then ok "token file $perm"; else bad "token file is $perm (must be root:root 600)"; fi
else
  bad "no token file $CREST_TOKEN_FILE (sudo $CREST_HOME/bin/bootstrap.sh --set-token)"
fi

hdr "code"
cur=$(readlink -e "$CREST_HOME/current" || true)
if [ -z "$cur" ]; then
  bad "no current release"
else
  echo "  current: $cur"
  echo "  fork     $(release_field "$cur/RELEASE.txt" FORK_SHA | cut -c1-12) (ref $(release_field "$cur/RELEASE.txt" FORK_REF | cut -c1-12), solver-path changed: $(release_field "$cur/RELEASE.txt" SOLVER_PATH_CHANGED))"
  echo "  CREST_AI $(release_field "$cur/RELEASE.txt" CREST_AI_SHA | cut -c1-12) (ref $(release_field "$cur/RELEASE.txt" CREST_AI_REF), built $(release_field "$cur/RELEASE.txt" RELEASE_BUILT))"
  if grep -q '^def pinned_ids' "$cur/CREST_AI/hf_data/eventstore.py" 2> /dev/null; then
    ok "eventstore.pinned_ids present"
  else
    bad "current release's eventstore lacks pinned_ids"
  fi
  if [ "$st" = active ] && [ "${pid:-0}" -gt 0 ]; then
    wpid=$(pgrep -P "$pid" -f crestimap.worker 2> /dev/null || true)
    [ -n "$wpid" ] || wpid=$pid
    run_cwd=$(readlink -e "/proc/$wpid/cwd" 2> /dev/null || true)
    if [ "$run_cwd" = "$cur/CREST-iMAP" ]; then
      ok "running process uses the current release"
    else
      bad "running process uses ${run_cwd:-?}, not the current release: restart (sudo $CREST_HOME/bin/update.sh --force)"
    fi
  fi
fi
echo "  kit: $(cat "$CREST_HOME/bin/KIT_VERSION" 2> /dev/null || echo unknown)"
echo "  torch: $(sed -n 's/^torch==//p' "$VENV/pip-freeze.txt" 2> /dev/null || echo ?)  huggingface_hub: $(sed -n 's/^huggingface[-_]hub==//p' "$VENV/pip-freeze.txt" 2> /dev/null || echo ?)"

if [ "$QUICK" != 1 ] && [ -n "$cur" ]; then
  hdr "hf_data drift (release vs GitHub ${CREST_AI_REF:-main} and the deployed Space; read-only)"
  rc=0
  drift=$("$here/crest-py" "$here/queue_status.py" --code-drift 2>&1) || rc=$?
  printf '%s\n' "$drift" | sed 's/^/  /'
  case $rc in
    0) ok "publish-path hf_data in sync" ;;
    1) bad "hf_data drift (above): run sudo $CREST_HOME/bin/update.sh; if the Space differs from GitHub main, tell the user" ;;
    *) echo "  -- drift check incomplete (rc=$rc; network?)" ;;
  esac
fi

if [ "$QUICK" != 1 ]; then
  hdr "queue, publishes, pinned demo set (anonymous, read-only)"
  if "$here/crest-py" "$here/queue_status.py" 2>&1 | sed 's/^/  /'; then :; else bad "queue_status.py failed"; fi
  rc=0
  "$here/crest-py" "$here/queue_status.py" --own > /dev/null 2>&1 || rc=$?
  case $rc in
    0) echo "  this host holds a fresh claim (mid-visit)" ;;
    3) echo "  this host holds no claim (idle or between visits)" ;;
    *) bad "could not determine own claims (rc=$rc)" ;;
  esac
  if [ "$st" = active ]; then
    if "$here/crest-py" "$here/release_claims.py" 2>&1 | grep -q '(process gone)'; then
      bad "claims by DEAD processes of this host exist (release_claims.py lists them)"
    fi
  fi
  if [ -f "$CREST_VAR/pins_baseline.json" ]; then
    if "$here/crest-py" "$here/queue_status.py" --pins check "$CREST_VAR/pins_baseline.json" 2>&1 | sed 's/^/  /'; then
      ok "pinned demo events intact vs baseline"
    else
      bad "pinned demo events changed vs $CREST_VAR/pins_baseline.json - STOP the worker and tell the user"
    fi
  else
    echo "  -- no pins baseline yet ($CREST_VAR/pins_baseline.json is saved at gate G2)"
  fi
fi

hdr "disk"
for d in / "${DATA_ROOT:-/}"; do
  df -h "$d" | awk 'NR==2 {printf "  %s: %s used of %s, %s free (%s)\n", $6, $3, $2, $4, $5}'
done
free_gb=$(df -P -BG "${WORKER_WORK_ROOT:-/}" | awk 'NR==2 {gsub("G", "", $4); print $4}')
if [ "${free_gb:-0}" -lt "${DISK_MIN_FREE_GB:-12}" ]; then bad "only ${free_gb} GB free for jobs"; fi
for d in "${EVENT_DEM_CACHE:-}" "${HF_HOME:-}" "${WORKER_WORK_ROOT:-}"; do
  if [ -n "$d" ] && [ -d "$d" ]; then du -sh "$d" 2> /dev/null | sed 's/^/  /'; fi
done

hdr "SU burn (estimate; authoritative: Exosphere allocation card / ACCESS, 12-24 h behind)"
rate=$(($(nproc) * 2))
up_h=$(awk '{printf "%d", $1 / 3600}' /proc/uptime)
echo "  $(nproc) vCPU x 2 SU = $rate SU/h while ACTIVE ($((rate * 24)) SU/day); stopped 50 %, shelved 0"
echo "  this boot: up ${up_h} h = ~$((rate * up_h)) SU"
if [ -n "${SU_BUDGET_START:-}" ] && start_s=$(date -u -d "$SU_BUDGET_START" +%s 2> /dev/null); then
  used=$(((($(date -u +%s) - start_s) / 3600) * rate))
  left=$((${SU_BUDGET:-200000} - used))
  echo "  if active 24/7 since $SU_BUDGET_START: ~$used SU used, ~$left left = ~$((left / rate / 24)) days at this rate"
fi

hdr "logs (journalctl -u $CREST_UNIT)"
journalctl -u "$CREST_UNIT" -n 25 --no-pager -o short-iso 2> /dev/null | sed 's/^/  /' || true
since_arg=$SINCE
[[ $since_arg =~ ^[0-9]+[smhd]$ ]] && since_arg="-$since_arg"
j=$(journalctl -u "$CREST_UNIT" --since "$since_arg" --no-pager -o cat 2> /dev/null || true)
cnt() { printf '%s\n' "$j" | grep -cE "$1" || true; }
printf '\n  last %s: %s publish OK | %s publish FAILED | %s visit FAILED | %s commit failed | %s lost claim race | %s over-budget skips | %s claims released on stop | %s starts\n' \
  "$SINCE" "$(cnt 'publish OK')" "$(cnt 'publish FAILED')" "$(cnt 'visit FAILED|: FAILED')" \
  "$(cnt 'commit failed')" "$(cnt 'lost claim race')" "$(cnt 'exceeds the .* budget')" \
  "$(cnt 'release_claims: .*: released')" "$(cnt 'run-worker: release')"
if [ "$(cnt 'visit FAILED|: FAILED')" != 0 ]; then bad "failed visits in the last $SINCE (each one consumed its job as failed.json)"; fi
if [ "$(cnt 'CUDA out of memory|OutOfMemoryError')" != 0 ]; then bad "CUDA out-of-memory in the last $SINCE: lower EVENT_MAX_CELLS_GPU"; fi

hdr "summary"
if [ ${#problems[@]} -eq 0 ]; then
  echo "  HEALTHY"
  exit 0
fi
echo "  PROBLEMS (${#problems[@]}):"
printf '   - %s\n' "${problems[@]}"
exit 1
