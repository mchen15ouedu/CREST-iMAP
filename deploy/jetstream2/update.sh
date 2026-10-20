#!/usr/bin/env bash
# update.sh - refresh the worker's code and RESTART it. A running Python
# process keeps the modules it imported (the HPC worker ran weeks-old code
# after every sync), so new code takes effect only through this restart.
#
#   sudo update.sh             new release: fork @ FORK_REF + CREST_AI @ CREST_AI_REF
#                              (main = newest hf_data), verify (pinned_ids, minimum
#                              commits, solver-path guard), import self-test, wait
#                              until the worker holds no claim, stop, switch, start
#   sudo update.sh --now       do not wait for the worker to finish its visit
#   sudo update.sh --force     restart even if the code is unchanged (env file edits)
#   sudo update.sh --check     build + verify only; touch nothing that runs
#   sudo update.sh --rollback  switch back to the previous release and restart
#   --auto                     crest-worker-update.timer (daily): as plain update.sh,
#                              but a busy worker is not an error (retried next run)
#   --wait-min N               how long to wait for an idle worker (default 45)
#
# "Unchanged" = same fork SHA and same hf_data/ tree: a CREST_AI main commit
# outside hf_data/ does not restart the worker (resident sessions are RAM-only).
# A new release that does not stay up is rolled back automatically.
# Kit changes (these scripts, the units) come from re-running bootstrap.sh from
# an updated kit checkout, then `update.sh --force`. Runbook: docs/JETSTREAM2_WORKER.md
set -euo pipefail
shopt -s inherit_errexit 2> /dev/null || true
here=$(dirname "$(readlink -f "$0")")
# shellcheck source=lib.sh
. "$here/lib.sh"
require_root
load_env_file "$CREST_ENV_FILE"

NOW=0 FORCE=0 CHECK=0 ROLLBACK=0 AUTO=0 WAIT_MIN=45
while [ $# -gt 0 ]; do
  case $1 in
    --now) NOW=1 ;;
    --force) FORCE=1 ;;
    --check) CHECK=1 ;;
    --rollback) ROLLBACK=1 ;;
    --auto) AUTO=1 ;;
    --wait-min)
      WAIT_MIN=${2:?}
      shift
      ;;
    -h | --help)
      sed -n '2,22p' "$0"
      exit 0
      ;;
    *) die "unknown option $1" ;;
  esac
  shift
done
[[ $WAIT_MIN =~ ^[0-9]+$ ]] || die "--wait-min takes whole minutes"

if [ "$AUTO" = 1 ] && [ "${AUTO_UPDATE:-1}" != 1 ]; then
  log "--auto: AUTO_UPDATE=${AUTO_UPDATE} in $CREST_ENV_FILE - nothing to do"
  exit 0
fi

# one update at a time (the daily timer and a manual run must not interleave)
exec 9> /run/crest-worker-update.lock
flock -n 9 || die "another update.sh is running"

UPDATES_LOG=$CREST_VAR/updates.log
# "<fork sha> <hf_data tree>" of a release that failed to stay up: --auto skips
# it (no daily restart churn) until a manual update.sh succeeds; healthcheck.sh
# reports the file as a problem.
BLOCK_FILE=$CREST_VAR/update.blocked
record() { echo "$(_ts) $*" >> "$UPDATES_LOG"; }
release_key() { printf '%s %s\n' "$(release_field "$1/RELEASE.txt" FORK_SHA)" "$(hfdata_tree "$1")"; }

wait_idle() {
  local deadline=$((SECONDS + WAIT_MIN * 60)) rc
  while :; do
    rc=0
    own_claim_busy || rc=$?
    case $rc in
      3)
        log "worker holds no claim (between visits) - safe to restart"
        return 0
        ;;
      0) log "worker is mid-visit (holds a fresh claim); waiting ... (--now skips)" ;;
      *) warn "could not read the queue (rc=$rc); retrying" ;;
    esac
    if [ "$SECONDS" -ge "$deadline" ]; then
      return 1
    fi
    sleep 30
  done
}

# restart_into <release dir> [norollback]: stop (claims released by
# ExecStopPost), switch, start. If the new release does not stay up, switch
# back to the release that ran before and start that (unless norollback).
restart_into() {
  local d=$1 mode=${2:-} was old
  was=$(systemctl is-active "$CREST_UNIT" 2> /dev/null || true)
  old=$(readlink -e "$CREST_HOME/current" || true)
  if [ "$was" = active ] && [ "$NOW" != 1 ]; then
    if ! wait_idle; then
      if [ "$AUTO" = 1 ]; then
        log "--auto: worker still busy after $WAIT_MIN min - nothing changed; the next timer run retries"
        record "auto-update deferred (worker busy): $d"
        exit 0
      fi
      die "worker still busy after $WAIT_MIN min - nothing changed. Re-run later, or --now (the visit is lost and re-runs)."
    fi
  fi
  if [ "$was" = active ]; then
    log "stopping $CREST_UNIT"
    systemctl stop "$CREST_UNIT"
  fi
  activate_release "$d"
  if [ "$was" != active ]; then
    log "$CREST_UNIT was not running ($was); left stopped"
    return 0
  fi
  log "starting $CREST_UNIT"
  systemctl start "$CREST_UNIT" || true
  sleep 20
  if systemctl is-active --quiet "$CREST_UNIT"; then
    journalctl -u "$CREST_UNIT" -n 15 --no-pager -o cat | grep -E 'run-worker:|preflight:|up \(P1.6' || true
    return 0
  fi
  journalctl -u "$CREST_UNIT" -n 30 --no-pager -o cat >&2 || true
  if [ "$mode" != norollback ] && [ -n "$old" ] && [ "$old" != "$d" ]; then
    warn "$CREST_UNIT did not stay up on $d - rolling back to $old"
    release_key "$d" > "$BLOCK_FILE"
    systemctl stop "$CREST_UNIT" 2> /dev/null || true
    activate_release "$old"
    systemctl reset-failed "$CREST_UNIT" 2> /dev/null || true
    systemctl start "$CREST_UNIT" || true
    record "FAILED start on $d ($(cat "$BLOCK_FILE")); rolled back to $old; auto-update blocked for it"
    die "new release did not stay up; rolled back to $old (journalctl -u $CREST_UNIT -n 60). Tell the user."
  fi
  die "$CREST_UNIT did not stay up - journalctl -u $CREST_UNIT -n 60"
}

cur=$(readlink -e "$CREST_HOME/current" || true)
[ -n "$cur" ] || die "no current release - run bootstrap.sh first"

if [ "$ROLLBACK" = 1 ]; then
  prev=$(readlink -e "$CREST_HOME/previous") || die "no previous release to roll back to"
  verify_release "$prev"
  selftest_release "$prev"
  restart_into "$prev" norollback
  record "rollback $cur -> $prev"
  exit 0
fi

make_release
new=$NEW_RELEASE
# a built release that never became `current` is removed on any exit
# (verification refusal, --check, unchanged code, busy worker)
trap 'if [ -d "$new" ] && [ "$new" != "$(readlink -e "$CREST_HOME/current" || true)" ]; then rm -rf -- "$new"; fi' EXIT
verify_release "$new" record
selftest_release "$new"

same=0
if [ "$(release_field "$new/RELEASE.txt" FORK_SHA)" = "$(release_field "$cur/RELEASE.txt" FORK_SHA)" ] &&
  [ -n "$(hfdata_tree "$new")" ] && [ "$(hfdata_tree "$new")" = "$(hfdata_tree "$cur")" ]; then
  same=1
fi

if [ "$same" != 1 ] || [ "$CHECK" = 1 ]; then
  # the new hf_data vs the deployed Space (report only: GitHub main may lead
  # the Space for a while during a deploy)
  if ! "$here/crest-py" "$here/queue_status.py" --code-drift --release "$new" 2>&1 | sed 's/^/  /'; then
    warn "hf_data drift or check failure (above): tell the user if it persists"
  fi
fi

if [ "$CHECK" = 1 ]; then
  verdict="differs from current"
  [ "$same" = 1 ] && verdict="same fork and hf_data as current"
  log "--check: the new release builds and verifies ($verdict); nothing activated"
  exit 0
fi

if [ "$same" = 1 ]; then
  if [ "$FORCE" != 1 ]; then
    log "already up to date (fork $(release_field "$cur/RELEASE.txt" FORK_SHA | cut -c1-12), hf_data tree $(hfdata_tree "$cur" | cut -c1-12)); nothing restarted"
    exit 0
  fi
  log "--force: restarting on the unchanged release"
  restart_into "$cur" norollback
  record "forced restart on $cur"
  exit 0
fi

if [ "$AUTO" = 1 ] && [ -s "$BLOCK_FILE" ] && [ "$(release_key "$new")" = "$(cat "$BLOCK_FILE")" ]; then
  log "--auto: this code failed to stay up before ($BLOCK_FILE) - not retried; a manual update.sh tries it again"
  exit 0
fi
log "new code: fork $(release_field "$cur/RELEASE.txt" FORK_SHA | cut -c1-12) -> $(release_field "$new/RELEASE.txt" FORK_SHA | cut -c1-12), CREST_AI $(release_field "$cur/RELEASE.txt" CREST_AI_SHA | cut -c1-12) -> $(release_field "$new/RELEASE.txt" CREST_AI_SHA | cut -c1-12) (hf_data changed)"
restart_into "$new"
rm -f "$BLOCK_FILE"
record "update $cur -> $new fork=$(release_field "$new/RELEASE.txt" FORK_SHA) crest_ai=$(release_field "$new/RELEASE.txt" CREST_AI_SHA)"
prune_releases 3
log "done. Check: sudo $CREST_HOME/bin/healthcheck.sh"
