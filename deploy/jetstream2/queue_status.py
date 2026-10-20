#!/usr/bin/env python3
"""READ-ONLY view of the CREST 2-D event queue for the Jetstream2 worker.

Writes nothing to Hugging Face, ever: it removes HF_TOKEN from its own
environment and disables implicit tokens before huggingface_hub is
imported, so every request is anonymous (vincewin/CREST_data is public)
and any write would be rejected by the server. The only token use is
--whoami, which sends the token file's content to the read-only whoami
endpoint to report the token's scope (never the token itself).

Reuses the queue code instead of re-implementing it:
crestimap.worker.expected_cells/QPREFIX/TS (capacity check, layout) and
hf_data.eventqueue._spec/_claim/_fresh, hf_data.eventstore.load_index/
pinned_ids (index + pinned demo set).

  queue_status.py                       queue + publishers + pinned-demo summary
  queue_status.py --own                 exit 0 if THIS host holds a fresh claim, 3 if not
  queue_status.py --pins save FILE      record the pinned ids index.json lists (+ stored frames)
  queue_status.py --pins check FILE     G2/G4: none of them dropped or stripped since
  queue_status.py --verify-publish ID   G2: a published event's provenance, frames, grid
  queue_status.py --whoami [--token-file F]   G0: fine-grained token scope
  queue_status.py --code-drift [--release DIR]  the release's hf_data vs
                                        CREST_AI_REF on GitHub and vs the deployed
                                        Space (HFDATA_REF_SPACE); exit 0 in sync,
                                        1 publish-path modules differ, 2 unknown
  queue_status.py --json ...            machine-readable output

Paths: the current release (CREST_HOME/current) unless CREST_FORK_DIR /
CREST_AI_DIR point elsewhere.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import json
import os
import re
import socket
import subprocess
import sys
import tempfile

OWN_BUSY, OWN_IDLE = 0, 3


def _utcnow() -> datetime.datetime:
    """Naive UTC now (the queue's %Y-%m-%dT%H:%M:%SZ stamps are naive UTC)."""
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


def _anonymous():
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN_PATH"):
        os.environ.pop(k, None)
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"


def _truststore():
    try:                      # only present on networks with TLS inspection
        import truststore
        truststore.inject_into_ssl()
    except Exception:
        pass


def _paths():
    home = os.environ.get("CREST_HOME", "/opt/crest-worker")
    rel = os.path.realpath(os.path.join(home, "current"))
    fork = os.environ.get("CREST_FORK_DIR") or os.path.join(rel, "CREST-iMAP")
    ai = os.environ.get("CREST_AI_DIR") or os.path.join(rel, "CREST_AI")
    for p in (ai, fork):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    info = {}
    try:
        with open(os.path.join(rel, "RELEASE.txt"), encoding="utf-8") as fp:
            for line in fp:
                k, _, v = line.strip().partition("=")
                if k:
                    info[k] = v
    except OSError:
        pass
    return rel, info


def _ident(host: str) -> str:
    return os.environ.get("EVENT_ENGINE_IDENT") or f"jetstream2:{host}"


def _age_s(stamp: str | None, fmt: str) -> float | None:
    try:
        t = datetime.datetime.strptime(stamp or "", fmt)
    except ValueError:
        return None
    return (_utcnow() - t).total_seconds()


def _fmt_age(s: float | None) -> str:
    if s is None:
        return "?"
    if s < 120:
        return f"{s:.0f}s"
    if s < 7200:
        return f"{s / 60:.0f}m"
    return f"{s / 3600:.1f}h"


# --------------------------------------------------------------------- queue
def queue_rows(files: list, host: str, max_cells: int) -> list:
    from crestimap.worker import QPREFIX, TS, expected_cells
    from hf_data import eventqueue
    q = {f for f in files if f.startswith(QPREFIX + "/")}
    ids = sorted({os.path.basename(f)[:-5] for f in q
                  if f.endswith(".json") and not f.endswith(".failed.json")})
    rows = []
    for ev in ids:
        spec = eventqueue._spec(ev) or {}
        claim = eventqueue._claim(ev) if f"{QPREFIX}/{ev}.claim" in q else None
        dom = spec.get("domain") or {}
        try:
            cells = expected_cells(spec)
        except Exception:
            cells = None
        worker = (claim or {}).get("worker")
        fresh = eventqueue._fresh(claim)
        mine = bool(worker) and str(worker).rpartition(":")[0] == host
        bundle = f"{QPREFIX}/{ev}.forcing.tar.gz" in q
        fits = cells is not None and cells <= max_cells
        rows.append({
            "id": ev, "queued": spec.get("queued"),
            "queued_age_s": _age_s(spec.get("queued"), TS),
            "cold": bool(spec.get("cold")), "dem_res": spec.get("dem_res"),
            "n_bbox": dom.get("n_bbox"), "n_active": dom.get("n_active"),
            "cells": cells, "bundle": bundle, "domain": bool(dom),
            "claim_worker": worker, "claim_hb": (claim or {}).get("hb"),
            "claim_age_s": _age_s((claim or {}).get("hb"), TS),
            "claim_fresh": fresh, "mine": mine, "fits": fits,
            "failed": f"{QPREFIX}/{ev}.failed.json" in q,
            "claimable_here": bundle and bool(dom) and fits and (not fresh or mine),
        })
    return rows


def failed_ids(files: list) -> list:
    from crestimap.worker import QPREFIX
    return sorted(os.path.basename(f)[:-len(".failed.json")] for f in files
                  if f.startswith(QPREFIX + "/") and f.endswith(".failed.json"))


# ---------------------------------------------------------------------- pins
_PNG = re.compile(r"^depth_\d+\.png$")


def stored_pngs(files: list, prefix: str) -> collections.Counter:
    c = collections.Counter()
    for f in files:
        parts = f.split("/")
        if len(parts) == 3 and parts[0] == prefix and _PNG.match(parts[2]):
            c[parts[1]] += 1
    return c


def pin_state(files: list) -> dict:
    from hf_data import eventstore
    pins = eventstore.pinned_ids(force=True)
    if pins is None:
        raise RuntimeError("events/pinned.json could not be read")
    idx = eventstore.load_index()
    if not idx:
        raise RuntimeError("events/index.json could not be read (or is empty)")
    pngs = stored_pngs(files, eventstore.PREFIX)
    listed = {e: {"n_frames": (idx[e] or {}).get("n_frames"),
                  "archive_frames": (idx[e] or {}).get("archive_frames"),
                  "pngs": pngs.get(e, 0)}
              for e in sorted(pins) if e in idx}
    return {"pinned": len(pins), "indexed": len(idx), "listed": listed,
            "taken": _utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}


def pin_check(base: dict, now: dict) -> list:
    problems = []
    for e, b in base["listed"].items():
        n = now["listed"].get(e)
        if n is None:
            problems.append(f"{e}: was listed in index.json, now GONE")
            continue
        if (n["pngs"] or 0) < (b["pngs"] or 0):
            problems.append(f"{e}: stored frames {b['pngs']} -> {n['pngs']} (STRIPPED)")
        if (n["n_frames"] or 0) < (b["n_frames"] or 0):
            problems.append(f"{e}: n_frames {b['n_frames']} -> {n['n_frames']}")
    return problems


# -------------------------------------------------------------------- verify
def verify_publish(repo: str, ev: str, files: list, host: str, rel_info: dict) -> list:
    """[(check, ok, detail)] for one published event."""
    from huggingface_hub import hf_hub_download
    from hf_data import eventstore
    folder = f"{eventstore.PREFIX}/{ev}/"
    have = {f[len(folder):] for f in files if f.startswith(folder)}
    out = []
    if "manifest.json" not in have:
        return [("manifest.json present", False, f"no {folder}manifest.json")]
    p = hf_hub_download(repo, folder + "manifest.json", repo_type="dataset",
                        force_download=True)
    with open(p, encoding="utf-8") as fp:
        man = json.load(fp)
    prov = man.get("provenance") or {}
    want_engine = _ident(host)
    out.append(("provenance.engine", prov.get("engine") == want_engine,
                f"{prov.get('engine')} (want {want_engine})"))
    fsha = (rel_info.get("FORK_SHA") or "")[:12]
    out.append(("provenance.crestimap", bool(fsha) and prov.get("crestimap") == fsha,
                f"{prov.get('crestimap')} (want {fsha or '?'})"))
    asha = (rel_info.get("CREST_AI_SHA") or "")[:12]
    out.append(("provenance.crest_ai", bool(asha) and prov.get("crest_ai") == asha,
                f"{prov.get('crest_ai')} (want {asha or '?'})"))
    frames = man.get("frames") or []
    missing = [f.get("file") for f in frames if f.get("file") not in have] + \
              [f.get("png") for f in frames if f.get("png") and f.get("png") not in have]
    out.append(("frames listed == stored", not missing,
                f"{len(frames)} frames listed, {len(missing)} file(s) missing"
                + (f": {missing[:3]}" if missing else "")))
    md = man.get("maxdepth") or "maxdepth.tif"
    out.append(("maxdepth present", md in have, md))
    gr = man.get("grid") or {}
    dx = gr.get("dx_m")
    out.append(("native resolution (dx_m <= 45)", isinstance(dx, (int, float)) and dx <= 45,
                f"grid {gr.get('ny')}x{gr.get('nx')}, dx {dx} m"))
    dom = man.get("domain") or {}
    out.append(("basin domain (not a rectangle)", bool(dom.get("n_active")),
                f"n_active={dom.get('n_active')} n_bbox={dom.get('n_bbox')}"))
    idx = eventstore.load_index()
    s = idx.get(ev) or {}
    out.append(("index.json entry", s.get("generated") == man.get("generated")
                and s.get("engine") == prov.get("engine"),
                f"generated {s.get('generated')} engine {s.get('engine')}"))
    return out


# ---------------------------------------------------------------- code drift
# The hf_data modules on the worker's publish path (Worker.process_visit calls
# eventsim._update_archive/_make_pngs and eventstore.publish_event; eventqueue
# and provenance are imported by those). Retention lives in eventstore.
CORE_HFDATA = ("eventstore.py", "eventsim.py", "eventqueue.py", "provenance.py")
DRIFT_OK, DRIFT_FOUND, DRIFT_UNKNOWN = 0, 1, 2


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          check=True, timeout=600).stdout


def _hfdata_blobs(repo_dir: str) -> dict:
    """{module.py: git blob id} for hf_data/*.py at HEAD (works in blobless,
    sparse and --no-checkout clones: only tree objects are read)."""
    blobs = {}
    for line in _git("-C", repo_dir, "ls-tree", "HEAD", "hf_data/").splitlines():
        meta, _, path = line.partition("\t")
        parts = meta.split()
        if len(parts) == 3 and parts[1] == "blob" and path.endswith(".py"):
            blobs[path.rpartition("/")[2]] = parts[2]
    return blobs


def _diff(a: dict, b: dict) -> list:
    return sorted(f for f in set(a) | set(b) if a.get(f) != b.get(f))


def code_drift(ai_dir: str, ai_url: str, ai_ref: str, space: str) -> tuple[int, list]:
    """Compare the release's hf_data with (1) the head of ai_ref on GitHub and
    (2) the deployed Space's hf_data (what ZeroGPU and the CPU Spaces publish
    with). Git blob ids are compared, so only trees are fetched, never files."""
    found = unknown = False
    try:
        local_sha = _git("-C", ai_dir, "rev-parse", "HEAD").strip()
        local = _hfdata_blobs(ai_dir)
    except Exception as e:                                   # noqa: BLE001
        return DRIFT_UNKNOWN, [f"?? cannot read the release's CREST_AI checkout {ai_dir}: {e}"]
    lines = [f"release CREST_AI {local_sha[:12]} ({ai_dir})"]
    absent = [f for f in CORE_HFDATA if f not in local]
    if absent:
        return DRIFT_FOUND, lines + [f"!! release lacks hf_data/{', '.join(absent)}"]

    # 1. GitHub: has hf_data moved on ai_ref since this release was built?
    try:
        out = _git("ls-remote", ai_url, f"refs/heads/{ai_ref}").split()
        head = out[0] if out else None
        if head is None:
            lines.append(f"-- GitHub: {ai_ref} is not a branch of {ai_url} (pinned ref); "
                         "freshness not checked")
        elif head == local_sha:
            lines.append(f"ok GitHub: release = {ai_ref} head {head[:12]}")
        else:
            with tempfile.TemporaryDirectory(prefix="crest-drift-") as tmp:
                _git("clone", "--quiet", "--depth", "1", "--filter=blob:none",
                     "--no-checkout", "--branch", ai_ref, ai_url, tmp)
                remote = _hfdata_blobs(tmp)
            changed = _diff(local, remote)
            core = [f for f in changed if f in CORE_HFDATA]
            if core:
                found = True
                lines.append(f"!! GitHub: {ai_ref} head {head[:12]} changed publish-path "
                             f"hf_data/{', '.join(core)} since this release: run "
                             "update.sh (or wait for crest-worker-update.timer)")
            elif changed:
                lines.append(f"-- GitHub: {ai_ref} head {head[:12]} changed only "
                             f"hf_data/{', '.join(changed)} (not on the publish path); "
                             "the daily update picks it up")
            else:
                lines.append(f"ok GitHub: {ai_ref} moved to {head[:12]}, hf_data unchanged")
    except Exception as e:                                   # noqa: BLE001
        unknown = True
        lines.append(f"?? GitHub check failed: {type(e).__name__}: {e}")

    # 2. the deployed Space (public; anonymous listing with blob ids)
    try:
        from huggingface_hub import HfApi
        space_blobs = {}
        for f in HfApi().list_repo_tree(space, repo_type="space", path_in_repo="hf_data"):
            bid = getattr(f, "blob_id", None)
            if bid and f.path.endswith(".py"):
                space_blobs[f.path.rpartition("/")[2]] = bid
        if not space_blobs:
            raise RuntimeError(f"no hf_data/*.py listed in Space {space}")
        changed = _diff(local, space_blobs)
        core = [f for f in changed if f in CORE_HFDATA]
        if core:
            found = True
            lines.append(f"!! Space {space}: hf_data/{', '.join(core)} differ from this "
                         "release - the worker would publish and apply retention with other "
                         "code than the Spaces and ZeroGPU; tell the user (GitHub main and "
                         "the Space are meant to be identical)")
        else:
            lines.append(f"ok Space {space}: publish-path hf_data identical "
                         f"({', '.join(CORE_HFDATA)})")
        other = [f for f in changed if f not in CORE_HFDATA]
        if other:
            lines.append(f"-- Space {space}: other hf_data modules differ (not on the "
                         f"worker's publish path): {', '.join(other)}")
    except Exception as e:                                   # noqa: BLE001
        unknown = True
        lines.append(f"?? Space check failed: {type(e).__name__}: {e}")
    if found:
        return DRIFT_FOUND, lines
    return (DRIFT_UNKNOWN if unknown else DRIFT_OK), lines


# --------------------------------------------------------------------- whoami
def whoami(token_file: str | None, repo: str) -> tuple[bool, list]:
    from huggingface_hub import HfApi
    cands = [token_file] if token_file else []
    if os.environ.get("CREDENTIALS_DIRECTORY"):
        cands.append(os.path.join(os.environ["CREDENTIALS_DIRECTORY"], "hf_token"))
    cands.append("/etc/crest-worker/hf_token")
    tok = None
    for p in cands:
        try:
            with open(p, encoding="utf-8") as fp:
                tok = fp.read().strip() or None
            if tok:
                break
        except OSError:
            continue
    if not tok:
        return False, ["no readable token file (run as root: sudo ...)"]
    info = HfApi().whoami(token=tok)
    del tok
    lines = [f"account: {info.get('name')}"]
    at = ((info.get("auth") or {}).get("accessToken") or {})
    role = at.get("role")
    lines.append(f"token: '{at.get('displayName')}' role={role}")
    ok = role == "fineGrained"
    if not ok:
        lines.append(f"FAIL: token role is {role!r}; create a FINE-GRAINED token")
    fg = at.get("fineGrained") or {}
    scoped = fg.get("scoped") or []
    target_write = False
    for s in scoped:
        ent = s.get("entity") or {}
        perms = s.get("permissions") or []
        name = ent.get("name") or ent.get("_id")
        lines.append(f"  scoped {ent.get('type')} {name}: {', '.join(perms)}")
        writes = [x for x in perms if x.endswith(".write") or "write" in x]
        if name == repo and ent.get("type") in ("dataset", None) and writes:
            target_write = True
        elif writes and ent.get("type") in ("user", "org"):
            ok = False
            lines.append(f"FAIL: account-wide write on {name} covers far more than the one "
                         f"dataset - scope the token to {repo} only")
        elif writes:
            ok = False
            lines.append(f"FAIL: write permission on {name} is not needed - remove it")
    glob = fg.get("global") or []
    if glob:
        lines.append(f"  global: {', '.join(map(str, glob))}")
        if any("write" in str(g) for g in glob):
            ok = False
            lines.append("FAIL: global write permissions are not needed - remove them")
    if not target_write:
        ok = False
        lines.append(f"FAIL: no repository-scoped write permission on dataset {repo}")
    return ok, lines


# ---------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=os.environ.get("CREST_DATA_REPO",
                                                     "vincewin/CREST_data"))
    ap.add_argument("--host", default=socket.gethostname())
    ap.add_argument("--max-cells", type=int,
                    default=int(os.environ.get("EVENT_MAX_CELLS_GPU", "40000000")))
    ap.add_argument("--own", action="store_true")
    ap.add_argument("--pins", nargs=2, metavar=("save|check", "FILE"))
    ap.add_argument("--verify-publish", metavar="EVENT_ID")
    ap.add_argument("--whoami", action="store_true")
    ap.add_argument("--token-file")
    ap.add_argument("--code-drift", action="store_true")
    ap.add_argument("--release", metavar="DIR",
                    help="--code-drift: a release dir other than current")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if args.whoami:
        _truststore()
        ok, lines = whoami(args.token_file, args.repo)
        print("\n".join(lines))
        print("token scope:", "OK" if ok else "NOT OK")
        return 0 if ok else 1

    _anonymous()
    os.environ["CREST_DATA_REPO"] = args.repo      # hf_data reads it at import
    _truststore()
    rel, rel_info = _paths()

    if args.code_drift:
        ai_dir = (os.path.join(args.release, "CREST_AI") if args.release else
                  os.environ.get("CREST_AI_DIR") or os.path.join(rel, "CREST_AI"))
        rc, lines = code_drift(
            ai_dir,
            os.environ.get("CREST_AI_REPO", "https://github.com/mchen15ouedu/CREST_AI.git"),
            os.environ.get("CREST_AI_REF", "main"),
            os.environ.get("HFDATA_REF_SPACE", "vincewin/CREST_demo"))
        print("\n".join(lines))
        print("hf_data drift:", {DRIFT_OK: "none", DRIFT_FOUND: "FOUND",
                                 DRIFT_UNKNOWN: "could not check everything"}[rc])
        return rc

    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(args.repo, repo_type="dataset")

    if args.own:
        rows = queue_rows(files, args.host, args.max_cells)
        mine = [r for r in rows if r["mine"] and r["claim_fresh"]]
        for r in mine:
            print(f"{r['id']}: claimed by {r['claim_worker']} hb {_fmt_age(r['claim_age_s'])} ago")
        if not mine:
            print(f"no fresh claim held by {args.host}")
        return OWN_BUSY if mine else OWN_IDLE

    if args.pins:
        mode, path = args.pins
        now = pin_state(files)
        if mode == "save":
            with open(path, "w", encoding="utf-8") as fp:
                json.dump(now, fp, indent=1)
            print(f"saved {len(now['listed'])} pinned ids listed in index.json "
                  f"({now['pinned']} pinned, {now['indexed']} indexed) -> {path}")
            return 0
        if mode != "check":
            ap.error("--pins takes save or check")
        with open(path, encoding="utf-8") as fp:
            base = json.load(fp)
        problems = pin_check(base, now)
        for p in problems:
            print("  " + p)
        print(f"pinned demo events vs baseline {base.get('taken')}: "
              f"{len(base['listed'])} checked -> {'FAIL' if problems else 'PASS'}")
        return 1 if problems else 0

    if args.verify_publish:
        checks = verify_publish(args.repo, args.verify_publish, files, args.host, rel_info)
        bad = 0
        for name, ok, detail in checks:
            bad += not ok
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        print(f"{args.verify_publish}: {'PASS' if not bad else f'{bad} check(s) FAILED'}")
        return 0 if not bad else 1

    # default: overview
    from hf_data import eventstore
    rows = queue_rows(files, args.host, args.max_cells)
    failed = failed_ids(files)
    idx = eventstore.load_index()
    engines = collections.Counter((s or {}).get("engine") for s in idx.values())
    last = {}
    for e, s in idx.items():
        eng = (s or {}).get("engine")
        if (s or {}).get("generated", "") > last.get(eng, ("", ""))[0]:
            last[eng] = ((s or {}).get("generated", ""), e)
    try:
        pins = pin_state(files)
    except Exception as e:
        pins = {"error": f"{type(e).__name__}: {e}"}
    me = _ident(args.host)
    if args.json:
        print(json.dumps({"host": args.host, "engine": me, "max_cells": args.max_cells,
                          "release": rel_info, "queue": rows, "failed": failed,
                          "engines": dict(engines), "last_by_engine": last,
                          "pins": pins}, indent=1, default=str))
        return 0
    print(f"queue {args.repo} events/queue/  ({len(rows)} spec(s); this host "
          f"{args.host}, budget {args.max_cells / 1e6:.0f} M cells)")
    for r in rows:
        cl = (f"{r['claim_worker']} hb {_fmt_age(r['claim_age_s'])}"
              f"{'' if r['claim_fresh'] else ' STALE'}") if r["claim_worker"] else "unclaimed"
        cells = f"{r['cells'] / 1e6:.1f}M" if r["cells"] else "?"
        act = f"{r['n_active'] / 1e6:.2f}M act" if r["n_active"] else "no domain"
        flags = []
        if not r["bundle"]:
            flags.append("no bundle yet")
        if not r["fits"]:
            flags.append("OVER BUDGET")
        if r["mine"] and r["claim_fresh"]:
            flags.append("MINE")
        if r["claimable_here"] and not r["mine"]:
            flags.append("claimable here")
        print(f"  {r['id']}  {cells} bbox / {act}  queued {_fmt_age(r['queued_age_s'])} ago"
              f"{' cold' if r['cold'] else ''}  {cl}  {' '.join(flags)}")
    if failed:
        print(f"  failed.json: {', '.join(failed)}")
    print(f"index.json: {len(idx)} listed; publishers:")
    for eng, n in engines.most_common():
        g, e = last.get(eng, ("?", "?"))
        mark = "   <- this worker" if eng == me else ""
        print(f"  {n:3d}  {eng}  (latest {e} at {g}){mark}")
    if me not in engines:
        print(f"  (nothing listed from {me} yet)")
    if "error" in pins:
        print(f"pinned demo set: ERROR {pins['error']}")
    else:
        stripped = [e for e, v in pins["listed"].items() if (v["n_frames"] or 0) and not v["pngs"]]
        print(f"pinned demo set: {pins['pinned']} pinned, {len(pins['listed'])} listed in "
              f"index.json, {len(stripped)} listed-but-frameless")
    if rel_info:
        print(f"release: fork {rel_info.get('FORK_SHA', '?')[:12]}  "
              f"CREST_AI {rel_info.get('CREST_AI_SHA', '?')[:12]}  ({rel})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
