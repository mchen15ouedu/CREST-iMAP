#!/usr/bin/env bash
# bootstrap.sh - set up a fresh Jetstream2 GPU VM as a CREST-iMAP queue worker.
# Idempotent: safe to re-run (it repairs, never overwrites your env file, and
# NEVER enables or starts the worker - that is validation gate G2).
#
#   sudo ./bootstrap.sh                install / repair everything
#   sudo ./bootstrap.sh --set-token    store the HF token (prompted, hidden)
#   sudo ./bootstrap.sh --check-token  show the stored token's scope (read-only)
#   sudo ./bootstrap.sh --new-release  also build + activate a fresh code release
#                                      (worker must be stopped; else use update.sh)
# Optional overrides (prefix the command): TORCH_INDEX_URL=..., TORCH_SPEC=...,
# FORCE_OS=1, INIT_DATA_ROOT=/media/volume/<name> (used only when bootstrap
# first creates the env file: puts the data dirs on that attached volume).
# Runbook: docs/JETSTREAM2_WORKER.md
set -euo pipefail
shopt -s inherit_errexit 2> /dev/null || true

KIT=$(dirname "$(readlink -f "$0")")
# shellcheck source=lib.sh
. "$KIT/lib.sh"

EXEC_FILES=(bootstrap.sh update.sh healthcheck.sh preflight.sh run-worker.sh prune.sh
  crest-py release_claims.py queue_status.py replay_bundle.py)
DATA_FILES=(lib.sh crest-worker.service crest-worker-prune.service crest-worker-prune.timer
  crest-worker-update.service crest-worker-update.timer crest-worker.env.example README.md)
UNITS=(crest-worker.service crest-worker-prune.service crest-worker-prune.timer
  crest-worker-update.service crest-worker-update.timer)
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}
TORCH_SPEC=${TORCH_SPEC:-torch>=2.4,<2.15}
PIP_DEPS=(numpy rasterio requests pyarrow pillow huggingface_hub pytest)

check_token() {
  require_root
  [ -s "$CREST_TOKEN_FILE" ] || die "no token at $CREST_TOKEN_FILE (sudo $0 --set-token)"
  [ "$(stat -c '%U:%G %a' "$CREST_TOKEN_FILE")" = "root:root 600" ] ||
    die "$CREST_TOKEN_FILE must be root:root 0600 (is $(stat -c '%U:%G %a' "$CREST_TOKEN_FILE"))"
  if [ ! -x "$PY" ] || [ ! -f "$CREST_HOME/bin/queue_status.py" ]; then
    warn "venv/kit not installed yet - run sudo $0 first, then --check-token"
    return 0
  fi
  # root reads the root-only file inside Python and sends it only to the
  # read-only whoami endpoint; the token is never printed
  env -i PATH=/usr/bin:/bin HOME=/root LANG=C.UTF-8 HF_HUB_DISABLE_TELEMETRY=1 \
    "$PY" "$CREST_HOME/bin/queue_status.py" --whoami --token-file "$CREST_TOKEN_FILE" \
    --repo "${CREST_DATA_REPO:-vincewin/CREST_data}"
}

set_token() {
  require_root
  install -d -o root -g root -m 0755 "$CREST_ETC"
  [ -r /dev/tty ] || die "--set-token needs an interactive terminal"
  local tok tmp
  printf 'Paste the fine-grained HF token (input hidden), then press Enter: ' > /dev/tty
  IFS= read -rs tok < /dev/tty
  printf '\n' > /dev/tty
  tok=${tok//[[:space:]]/}
  [ -n "$tok" ] || die "empty input - nothing stored"
  [[ $tok =~ ^hf_[A-Za-z0-9]{20,}$ ]] || warn "input does not look like an hf_... token; storing it anyway"
  tmp=$(mktemp "$CREST_ETC/.hf_token.XXXXXX") # mode 0600
  printf '%s\n' "$tok" > "$tmp"                # builtin printf: never on a command line
  tok=
  unset tok
  chown root:root "$tmp"
  chmod 0600 "$tmp"
  mv -f "$tmp" "$CREST_TOKEN_FILE"
  log "token stored in $CREST_TOKEN_FILE (root:root 0600); checking its scope ..."
  [ -f "$CREST_ENV_FILE" ] && load_env_file "$CREST_ENV_FILE"
  check_token || warn "token scope check did not pass - fix the token before G2"
}

install_all() {
  require_root
  local new_release=${1:-0} drv mib name kit_rev p d u code safe cur vmode lic
  # -------------------------------------------------------------- 0. OS --
  # shellcheck source=/dev/null
  . /etc/os-release
  case "${ID:-}:${VERSION_ID:-}" in
    ubuntu:22.04 | ubuntu:24.04) ;;
    *) [ "${FORCE_OS:-0}" = 1 ] || die "tested on Ubuntu 22.04/24.04 (Featured-Ubuntu24); this is ${PRETTY_NAME:-unknown}. FORCE_OS=1 to try anyway." ;;
  esac

  # ------------------------------------------------------------- 1. GPU --
  command -v nvidia-smi > /dev/null || die "nvidia-smi not found: use a Jetstream2 featured image on a g3.* flavor"
  gpu_wait 120 || die "nvidia-smi does not answer (driver not loaded - after a kernel update, reboot; see the runbook)"
  drv=$(gpu_driver)
  mib=$(gpu_mem_mib)
  name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
  log "GPU: $name, ${mib} MiB, driver $drv"
  [ "${drv%%.*}" -ge 525 ] || die "NVIDIA driver $drv is older than 525: the CUDA 12.6 torch wheels need >= 525"
  [ "$mib" -ge 9500 ] || die "GPU memory ${mib} MiB < 9.5 GB - use g3.large (20 GB) or g3.xl (40 GB)"
  vmode=$(gpu_virt_mode)
  lic=$(gpu_vgpu_license)
  log "GPU virtualization mode: ${vmode:-?}; vGPU license: ${lic:-not reported}"
  if [ "$vmode" = VGPU ]; then
    case $lic in
      Licensed*) ;;
      '') warn "vGPU guest but nvidia-smi -q reports no 'License Status' - check it at gate G0" ;;
      *) warn "the vGPU is NOT licensed ($lic): CUDA degrades after 20 min and fails after 24 h. Re-check in a few minutes (nvidia-smi -q | grep -A2 'vGPU Software Licensed Product'); if it stays unlicensed, ask help@jetstream-cloud.org. preflight refuses to start the worker on an unlicensed vGPU." ;;
    esac
  fi

  # ------------------------------------------------------- 2. packages --
  local pkgs=(git ca-certificates curl python3 python3-venv) missing=()
  for p in "${pkgs[@]}"; do
    dpkg -s "$p" > /dev/null 2>&1 || missing+=("$p")
  done
  if [ ${#missing[@]} -gt 0 ]; then
    log "apt: installing ${missing[*]}"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -q
    apt-get install -y -q --no-install-recommends "${missing[@]}"
  fi
  python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' || die "python3 >= 3.10 required"

  # ------------------------------------------------ 3. user, dirs, config --
  if ! id -u "$CREST_USER" > /dev/null 2>&1; then
    useradd --system --user-group --home-dir "$CREST_VAR" --create-home \
      --shell /usr/sbin/nologin "$CREST_USER"
    log "created system user $CREST_USER (home $CREST_VAR)"
  fi
  install -d -o "$CREST_USER" -g "$CREST_USER" -m 0750 "$CREST_VAR"
  install -d -o root -g root -m 0755 "$CREST_HOME" "$CREST_HOME/bin" "$CREST_ETC"
  install -d -o "$CREST_USER" -g "$CREST_USER" -m 0755 "$CREST_HOME/releases"
  if [ ! -f "$CREST_ENV_FILE" ]; then
    install -o root -g root -m 0644 "$KIT/crest-worker.env.example" "$CREST_ENV_FILE"
    sed -i 's/\r$//' "$CREST_ENV_FILE"
    if [ -n "${INIT_DATA_ROOT:-}" ]; then
      [[ $INIT_DATA_ROOT =~ ^/[A-Za-z0-9._/-]+$ ]] || die "INIT_DATA_ROOT must be a plain absolute path"
      INIT_DATA_ROOT=${INIT_DATA_ROOT%/}
      sed -i "s#=/var/lib/crest-worker/data#=$INIT_DATA_ROOT#" "$CREST_ENV_FILE"
      log "the new env file puts the data dirs under $INIT_DATA_ROOT"
    fi
    log "created $CREST_ENV_FILE from the example - review it (no secrets go there)"
  fi
  grep -qiE '^[[:space:]]*(HF_TOKEN|HUGGING_FACE_HUB_TOKEN)=' "$CREST_ENV_FILE" &&
    die "$CREST_ENV_FILE contains a token line - remove it; the token belongs only in $CREST_TOKEN_FILE"
  load_env_file "$CREST_ENV_FILE"
  : "${DATA_ROOT:?}" "${WORKER_WORK_ROOT:?}" "${EVENT_DEM_CACHE:?}" "${HF_HOME:?}" "${EVENT_MAX_CELLS_GPU:?}"
  case $DATA_ROOT in /home/* | /root/*) die "DATA_ROOT under /home or /root is hidden from the service (ProtectHome=yes)" ;; esac
  for d in "$DATA_ROOT" "$WORKER_WORK_ROOT" "$EVENT_DEM_CACHE" "$HF_HOME" "$DATA_ROOT/replay"; do
    if on_volume_unmounted "$d"; then
      die "$d is under /media/volume/ but no volume is mounted there - attach the volume in Exosphere (it mounts at /media/volume/<name>), then re-run"
    fi
    install -d -o "$CREST_USER" -g "$CREST_USER" -m 0750 "$d"
  done
  # token: crestimap.worker reads ~/huggingface.txt; point it at the systemd
  # credential (exists only while the unit runs, readable only by it)
  ln -sfn "$CREST_CRED_PATH" "$CREST_VAR/huggingface.txt"
  chown -h "$CREST_USER:$CREST_USER" "$CREST_VAR/huggingface.txt"
  if [ -e "$CREST_TOKEN_FILE" ]; then
    chown root:root "$CREST_TOKEN_FILE"
    chmod 0600 "$CREST_TOKEN_FILE"
  fi

  # ------------------------------------------------------- 4. Python env --
  [ -x "$PY" ] || {
    log "creating venv $VENV"
    python3 -m venv "$VENV"
  }
  # --no-cache-dir: the torch wheels alone are several GB, and root's pip
  # cache would stay on the 60 GB root disk (prune.sh never sees it)
  "$PY" -m pip install -q --no-cache-dir --disable-pip-version-check --upgrade pip
  log "pip: '$TORCH_SPEC' from $TORCH_INDEX_URL (CUDA 12.6 wheels run on drivers >= 525)"
  "$PY" -m pip install -q --no-cache-dir --disable-pip-version-check "$TORCH_SPEC" --index-url "$TORCH_INDEX_URL"
  log "pip: ${PIP_DEPS[*]}"
  "$PY" -m pip install -q --no-cache-dir --disable-pip-version-check "${PIP_DEPS[@]}"
  "$PY" -m pip freeze > "$VENV/pip-freeze.txt"
  code=$(
    cat << 'PYEOF'
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
p = torch.cuda.get_device_properties(0)
x = torch.randn(4096, 4096, device="cuda")
s = float((x @ x).abs().sum())
free, total = torch.cuda.mem_get_info()
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}) on {p.name}: "
      f"{total / 2**30:.1f} GiB, sm_{p.major}{p.minor}, matmul ok")
PYEOF
  )
  svc_py / -c "$code" ||
    die "torch cannot use the GPU as $CREST_USER (driver $drv, vGPU license: ${lic:-n/a}). Every CUDA 12.x wheel needs the same driver floor (>= 525.60.13), so another cu12x index will not fix this: see the runbook's troubleshooting row 'torch reports no CUDA'."

  # ----------------------------------------------------- 5. kit scripts --
  if [ "$(readlink -f "$KIT")" != "$(readlink -f "$CREST_HOME/bin")" ]; then
    for p in "${EXEC_FILES[@]}"; do
      install -o root -g root -m 0755 "$KIT/$p" "$CREST_HOME/bin/$p"
    done
    for p in "${DATA_FILES[@]}"; do
      if [ -f "$KIT/$p" ]; then
        install -o root -g root -m 0644 "$KIT/$p" "$CREST_HOME/bin/$p"
      fi
    done
    sed -i 's/\r$//' "$CREST_HOME"/bin/* # in case the kit came from a Windows checkout
  fi
  kit_rev=unknown
  if [ -n "${SUDO_USER:-}" ] && [ -d "$KIT/../../.git" ]; then
    kit_rev=$(runuser -u "$SUDO_USER" -- git -C "$KIT" rev-parse --short=12 HEAD 2> /dev/null || echo unknown)
  fi
  echo "KIT_REV=$kit_rev KIT_INSTALLED=$(_ts)" > "$CREST_HOME/bin/KIT_VERSION"

  # ------------------------------------------------------ 6. code release --
  cur=$(readlink -e "$CREST_HOME/current" || true)
  if [ -n "$cur" ] && [ "$new_release" != 1 ]; then
    log "keeping current release $cur (update.sh refreshes code)"
    verify_release "$cur"
    [ "$(release_field "$cur/RELEASE.txt" FORK_REF)" = "$FORK_REF" ] ||
      warn "env FORK_REF=$FORK_REF differs from the current release's; run update.sh"
    selftest_release "$cur"
  else
    if systemctl is-active --quiet "$CREST_UNIT"; then
      die "crest-worker is running - change code with update.sh, not bootstrap"
    fi
    make_release
    verify_release "$NEW_RELEASE" record
    selftest_release "$NEW_RELEASE"
    activate_release "$NEW_RELEASE"
  fi
  # the release's hf_data publish modules vs GitHub main and the deployed
  # Space (report only; healthcheck.sh flags it as a problem)
  if ! "$CREST_HOME/bin/crest-py" "$CREST_HOME/bin/queue_status.py" --code-drift 2>&1 | sed 's/^/  /'; then
    warn "hf_data drift or check failure (above): tell the user before gate G2"
  fi

  # ---------------------------------------------------------- 7. systemd --
  for u in "${UNITS[@]}"; do
    install -o root -g root -m 0644 "$KIT/$u" "/etc/systemd/system/$u"
    sed -i 's/\r$//' "/etc/systemd/system/$u"
  done
  systemctl daemon-reload
  systemctl enable --now crest-worker-prune.timer > /dev/null
  # daily code refresh (update.sh --auto): newest CREST_AI main hf_data, the
  # fork stays at FORK_REF, restarts only between visits. AUTO_UPDATE=0 opts out.
  if [ "${AUTO_UPDATE:-1}" = 1 ]; then
    systemctl enable --now crest-worker-update.timer > /dev/null
  else
    systemctl disable --now crest-worker-update.timer > /dev/null 2>&1 || true
    warn "AUTO_UPDATE=0: nothing refreshes hf_data; run update.sh yourself (healthcheck flags drift)"
  fi
  if systemctl is-active --quiet "$CREST_UNIT"; then
    warn "crest-worker is running: restart it (sudo $CREST_HOME/bin/update.sh --force) to load changed kit files"
  fi

  # ------------------------------------------------------ 8. network/budget --
  for u in "https://huggingface.co/api/datasets/${CREST_DATA_REPO:-vincewin/CREST_data}" \
    https://prd-tnm.s3.amazonaws.com/ https://github.com/mchen15ouedu/CREST_AI; do
    code=$(curl -sS -o /dev/null -m 20 -w '%{http_code}' "$u" 2> /dev/null || echo unreachable)
    log "  outbound $u -> HTTP $code"
  done
  safe=$(safe_cells "$mib" "${BYTES_PER_CELL_EST:-350}")
  log "cell budget EVENT_MAX_CELLS_GPU=$EVENT_MAX_CELLS_GPU; this card holds ~$safe at ${BYTES_PER_CELL_EST:-350} B/cell"
  [ "$EVENT_MAX_CELLS_GPU" -le "$safe" ] ||
    warn "lower EVENT_MAX_CELLS_GPU in $CREST_ENV_FILE to <= $safe (preflight refuses to start otherwise)"

  echo
  log "bootstrap complete. The worker is installed but NOT enabled or started."
  if [ -s "$CREST_TOKEN_FILE" ]; then
    log "token present. Next: validation gates G0 and G1 (docs/JETSTREAM2_WORKER.md)."
  else
    log "next: store the token:  sudo $CREST_HOME/bin/bootstrap.sh --set-token"
  fi
  log "health:  sudo $CREST_HOME/bin/healthcheck.sh"
}

case ${1:-} in
  '') install_all 0 ;;
  --new-release) install_all 1 ;;
  --set-token) set_token ;;
  --check-token)
    [ -f "$CREST_ENV_FILE" ] && load_env_file "$CREST_ENV_FILE"
    check_token
    ;;
  -h | --help) sed -n '2,14p' "$0" ;;
  *) die "unknown option $1 (see --help)" ;;
esac
