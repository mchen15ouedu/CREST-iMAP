#!/usr/bin/env python3
"""Release queue claims left behind by DEAD worker processes on THIS host.

Why this exists: crestimap.worker installs no signal handler. When systemd
stops the service (KillSignal=SIGINT) the worker unwinds via
KeyboardInterrupt and its finally-block stops the heartbeat thread, but the
`events/queue/<id>.claim` it held stays until it goes stale (600 s). A
crash, SIGKILL or hard reboot leaves the same. crest-worker.service runs this
as ExecStopPost= (the worker has exited) and ExecStartPre= (before the next
one starts).

What it touches - only claims whose `worker` field is "<this hostname>:<pid>"
where <pid> is not a live `crestimap.worker` process, re-read immediately
before release. The release itself is the worker's OWN call,
`Worker._cleanup_queue(ev, queued, success=False)` with publishing off: one
"worker done <id> (claim only)" commit deleting just the .claim. Specs,
forcing bundles, failed.json and other workers' claims are never touched and
the queue entry is not consumed, so the event is re-claimable at once.

  release_claims.py            report only (anonymous reads, no token used)
  release_claims.py --apply    release them (token from the systemd credential)

Never prints the token. Exit status is 0 unless the arguments are wrong.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import socket
import sys
import tempfile


def _utcnow() -> datetime.datetime:
    """Naive UTC now (the queue's %Y-%m-%dT%H:%M:%SZ stamps are naive UTC)."""
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


def _log(msg: str):
    print(f"release_claims: {msg}", flush=True)


def _truststore():
    try:                      # only present on networks with TLS inspection
        import truststore
        truststore.inject_into_ssl()
    except Exception:
        pass


def _release_paths():
    home = os.environ.get("CREST_HOME", "/opt/crest-worker")
    rel = os.path.realpath(os.path.join(home, "current"))
    fork = os.environ.get("CREST_FORK_DIR") or os.path.join(rel, "CREST-iMAP")
    crest_ai = os.environ.get("CREST_AI_DIR") or os.path.join(rel, "CREST_AI")
    return fork, crest_ai


def _live_worker_pids() -> set:
    """PIDs of running `python -m crestimap.worker` processes on this host."""
    pids = set()
    if not os.path.isdir("/proc"):
        return pids
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/cmdline", "rb") as fp:
                argv = fp.read().split(b"\0")
        except OSError:
            continue
        if b"crestimap.worker" in argv:
            pids.add(int(d))
    return pids


def _read_token() -> str | None:
    cands = []
    if os.environ.get("CREDENTIALS_DIRECTORY"):
        cands.append(os.path.join(os.environ["CREDENTIALS_DIRECTORY"], "hf_token"))
    cands += [os.path.expanduser("~/huggingface.txt"), "/etc/crest-worker/hf_token"]
    for p in cands:
        try:
            with open(p, encoding="utf-8") as fp:
                tok = fp.read().strip()
            if tok:
                return tok
        except OSError:
            continue
    return None


def own_dead_claims(repo: str, host: str, qprefix: str) -> list:
    """[(event_id, claim dict)] for claims by host:<dead pid>. Anonymous
    reads: the dataset is public, so no token is needed to look."""
    from huggingface_hub import HfApi, hf_hub_download
    files = HfApi().list_repo_files(repo, repo_type="dataset", token=False)
    live = _live_worker_pids()
    out = []
    for f in sorted(files):
        if not (f.startswith(qprefix + "/") and f.endswith(".claim")):
            continue
        try:
            p = hf_hub_download(repo, f, repo_type="dataset", token=False,
                                force_download=True)
            with open(p, encoding="utf-8") as fp:
                claim = json.load(fp)
        except Exception as e:                     # vanished meanwhile, etc.
            _log(f"could not read {f}: {type(e).__name__}")
            continue
        worker = str(claim.get("worker") or "")
        whost, _, wpid = worker.rpartition(":")
        if whost != host:
            continue                               # another worker's claim
        if wpid.isdigit() and int(wpid) in live:
            _log(f"{f}: held by LIVE worker pid {wpid} - left alone")
            continue
        out.append((os.path.basename(f)[:-len(".claim")], claim))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true",
                    help="release the claims (default: report only)")
    ap.add_argument("--repo", default=os.environ.get("CREST_DATA_REPO",
                                                     "vincewin/CREST_data"))
    ap.add_argument("--host", default=socket.gethostname(),
                    help="host part of the claim ident (default: this host)")
    args = ap.parse_args(argv)

    _truststore()
    fork, crest_ai = _release_paths()
    if not os.path.isdir(os.path.join(fork, "crestimap")):
        _log(f"no crestimap release at {fork} - nothing to do")
        return 0
    sys.path.insert(0, fork)
    try:
        from crestimap import worker as W          # the queue protocol owner
    except Exception as e:
        _log(f"cannot import crestimap.worker ({type(e).__name__}: {e})")
        return 0

    try:
        dead = own_dead_claims(args.repo, args.host, W.QPREFIX)
    except Exception as e:
        _log(f"queue listing failed ({type(e).__name__}: {e}); claims, if any, "
             f"go stale after 600 s")
        return 0
    if not dead:
        _log(f"no claims left by dead {args.host}:* workers")
        return 0
    for ev, c in dead:
        _log(f"{ev}: claim by {c.get('worker')} hb={c.get('hb')} (process gone)")
    if not args.apply:
        _log("report only (pass --apply to release)")
        return 0

    tok = _read_token()
    if not tok:
        _log("no token readable - leaving the claim(s) to go stale (600 s)")
        return 0
    os.environ["HF_TOKEN"] = tok                   # Worker._load_token reads it
    ns = argparse.Namespace(
        repo=args.repo, crest_ai=crest_ai,
        work_root=os.environ.get("WORKER_WORK_ROOT") or tempfile.gettempdir(),
        max_cells=0, devices="", tiles=0, publish=False,  # never consume
        stale_s=600, hb_s=240, device="cpu")
    w = W.Worker(ns)
    for ev, c in dead:
        path = f"{W.QPREFIX}/{ev}.claim"
        cur = w._read_json(path)
        if not cur or cur.get("worker") != c.get("worker"):
            _log(f"{ev}: claim changed or gone since listing - left alone")
            continue
        w._cleanup_queue(ev, cur.get("queued", ""), success=False)
        gone = w._read_json(path) is None
        _log(f"{ev}: {'released' if gone else 'release FAILED (goes stale in 600 s)'}"
             f" at {_utcnow():%Y-%m-%dT%H:%M:%SZ}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
