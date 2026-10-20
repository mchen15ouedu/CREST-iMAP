#!/usr/bin/env python3
"""G1 offline replay: solve one queue bundle on this GPU, publish nothing.

Downloads `events/queue/<id>.json` + `<id>.forcing.tar.gz` (at HEAD, or at a
past dataset revision for an already-published event) and runs EXACTLY the
worker's new-session visit - `crestimap.session.EventSession(cfg,
resume=True).advance(...)`, then `hf_data.eventsim._update_archive` and
`_make_pngs`, with the same EventConfig fields `Worker.process_visit` uses.
The OSCER brief's rule applies: validation drives EventSession directly,
because `crestimap.worker` writes claims even with --no-publish.

It cannot write to Hugging Face: HF_TOKEN is removed from the environment and
implicit tokens are disabled before huggingface_hub is imported, so every
request is anonymous (the dataset is public). No claim, no heartbeat, no
publish, no queue change. Outputs stay under --out.

Reports: grid vs the spec's native n_bbox (coarsening check), frames, wall
time, GPU-min per (M bbox cells x simulated hour), peak CUDA memory and
bytes per bbox cell (-> BYTES_PER_CELL_EST / EVENT_MAX_CELLS_GPU), and, when
the event is published with the same grid, wet-cell Jaccard and mean |dDepth|
of the last depth frame both runs wrote (informational: the published run
usually continued a resident session) and of the episode maxdepth.

  sudo crest-py replay_bundle.py --event 2026101511_07239500
  sudo crest-py replay_bundle.py --event 2026101123_09426620 --find-revision
  sudo crest-py replay_bundle.py --event ID --revision <dataset commit sha>
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import sys
import tarfile
import time

for _k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN_PATH"):
    os.environ.pop(_k, None)
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

TMIN = "%Y-%m-%dT%H:%MZ"
WET_M = 0.05                                   # wet-cell threshold for Jaccard


def _utcnow() -> datetime.datetime:
    """Naive UTC now (the queue's %Y-%m-%dT%H:%M:%SZ stamps are naive UTC)."""
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


def _log(msg: str):
    print(f"[{_utcnow():%H:%M:%S}Z] {msg}", flush=True)


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
        if p not in sys.path:
            sys.path.insert(0, p)
    return fork, ai


def _live_worker() -> bool:
    if not os.path.isdir("/proc"):
        return False
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                with open(f"/proc/{d}/cmdline", "rb") as fp:
                    if b"crestimap.worker" in fp.read().split(b"\0"):
                        return True
            except OSError:
                continue
    return False


def _bundle_at(api, repo: str, qprefix: str, ev: str, rev: str) -> bool:
    try:
        return all(api.file_exists(repo, f"{qprefix}/{ev}{suffix}", repo_type="dataset",
                                   revision=rev)
                   for suffix in (".json", ".forcing.tar.gz"))
    except Exception:
        return False


def find_revision(api, repo: str, qprefix: str, ev: str) -> str | None:
    """A dataset revision that still holds this event's spec + bundle.

    Fast path: the commit that last wrote events/<ev>/manifest.json. The
    worker publishes BEFORE it consumes the queue entry, so at that commit
    the very bundle behind the published result is still queued.
    Slow fallback: the newest commit titled 'queue <ev> (...)' (the enqueue
    commit, eventqueue.enqueue) - lists the whole history, minutes."""
    try:
        for item in api.list_repo_tree(repo, path_in_repo=f"events/{ev}", repo_type="dataset",
                                       expand=True):
            if item.path.endswith("/manifest.json") and getattr(item, "last_commit", None):
                rev = item.last_commit.oid
                if _bundle_at(api, repo, qprefix, ev, rev):
                    _log(f"publish commit {rev[:12]} ({item.last_commit.date:%Y-%m-%d %H:%MZ}) "
                         f"still holds the bundle")
                    return rev
                _log(f"publish commit {rev[:12]} no longer holds the bundle - scanning history")
    except Exception as e:
        _log(f"events/{ev} not listable ({type(e).__name__}) - scanning history")
    _log("listing dataset commits to find the enqueue revision (slow) ...")
    for c in api.list_repo_commits(repo, repo_type="dataset"):
        if (c.title or "").startswith(f"queue {ev} (") and _bundle_at(api, repo, qprefix, ev,
                                                                      c.commit_id):
            _log(f"enqueue commit {c.commit_id[:12]} at {c.created_at:%Y-%m-%d %H:%MZ}")
            return c.commit_id
    return None


def compare_published(repo: str, ev: str, out_dir: str, manifest: dict) -> dict:
    import numpy as np
    import rasterio
    from huggingface_hub import hf_hub_download
    res = {}
    try:
        p = hf_hub_download(repo, f"events/{ev}/manifest.json", repo_type="dataset",
                            force_download=True)
        with open(p, encoding="utf-8") as fp:
            pub = json.load(fp)
    except Exception:
        return {"published": False}
    res["published"] = True
    res["published_engine"] = (pub.get("provenance") or {}).get("engine")
    res["published_window"] = [pub.get("sim_start"), pub.get("t_end")]
    g0, g1 = pub.get("grid") or {}, manifest.get("grid") or {}
    res["same_grid"] = (g0.get("ny"), g0.get("nx")) == (g1.get("ny"), g1.get("nx"))
    res["published_grid"] = [g0.get("ny"), g0.get("nx"), g0.get("dx_m")]
    if not res["same_grid"]:
        return res

    def stats(pub_name: str, local_name: str, key: str):
        q = hf_hub_download(repo, f"events/{ev}/{pub_name}", repo_type="dataset",
                            force_download=True)
        with rasterio.open(q) as ds:
            a = ds.read(1).astype(np.float32) / 100.0          # cm -> m (io.write_depth)
        with rasterio.open(os.path.join(out_dir, local_name)) as ds:
            b = ds.read(1).astype(np.float32) / 100.0
        wa, wb = a >= WET_M, b >= WET_M
        union = int((wa | wb).sum())
        res[f"{key}_wet_jaccard"] = round(float((wa & wb).sum()) / union, 4) if union else 1.0
        res[f"{key}_mean_abs_ddepth_cm_wet"] = (
            round(float(np.abs(a - b)[wa | wb].mean() * 100), 2) if union else 0.0)
        res[f"{key}_wet_cells_published_replay"] = [int(wa.sum()), int(wb.sum())]

    try:
        # primary: the last depth frame BOTH runs wrote at the same timestamp
        # (same bundle, same window; differences come from the resume state -
        # init_depth with momentum at rest here vs the published run's state)
        common = sorted({f.get("file") for f in manifest.get("frames") or []}
                        & {f.get("file") for f in pub.get("frames") or []} - {None})
        if common:
            res["frame_compared"] = common[-1]
            stats(common[-1], common[-1], "frame")
        # secondary: episode maxdepth. _update_archive merged the PUBLISHED
        # archive's older frames into this run's maxdepth, so this is partly
        # shared by construction - informational only.
        stats(pub.get("maxdepth") or "maxdepth.tif", manifest.get("maxdepth") or "maxdepth.tif",
              "episode_maxdepth")
    except Exception as e:
        res["compare_error"] = f"{type(e).__name__}: {e}"
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--event", required=True)
    ap.add_argument("--revision", default=None,
                    help="dataset commit holding the bundle (default: HEAD)")
    ap.add_argument("--find-revision", action="store_true",
                    help="use the commit that published events/<id>/manifest.json (the "
                         "bundle is still queued there); else scan the history")
    ap.add_argument("--repo", default=os.environ.get("CREST_DATA_REPO",
                                                     "vincewin/CREST_data"))
    ap.add_argument("--out", default=os.path.join(
        os.environ.get("DATA_ROOT", "/var/lib/crest-worker/data"), "replay"))
    ap.add_argument("--max-cells", type=int,
                    default=int(os.environ.get("EVENT_MAX_CELLS_GPU", "40000000")))
    ap.add_argument("--device", default=os.environ.get("EVENT_DEVICE", "cuda"))
    ap.add_argument("--dt-every", type=int,
                    default=int(os.environ.get("EVENT_DT_EVERY", "1")))
    ap.add_argument("--keep", action="store_true", help="keep the extracted bundle")
    ap.add_argument("--force", action="store_true",
                    help="run even though a live worker shares this GPU")
    args = ap.parse_args(argv)

    if _live_worker() and not args.force:
        _log("a crestimap.worker process is running on this host: a replay would compete "
             "for its GPU memory (a CUDA OOM consumes a live job). Stop crest-worker first "
             "or pass --force.")
        return 2

    _truststore()
    _paths()
    from huggingface_hub import HfApi, hf_hub_download
    from crestimap import EventConfig
    from crestimap.session import EventSession
    from crestimap.worker import QPREFIX, expected_cells
    from hf_data import eventsim
    import torch

    api = HfApi()
    ev = args.event
    rev = args.revision
    if rev is None:
        files = set(api.list_repo_files(args.repo, repo_type="dataset"))
        if f"{QPREFIX}/{ev}.forcing.tar.gz" in files and f"{QPREFIX}/{ev}.json" in files:
            rev = "main"
        elif args.find_revision:
            rev = find_revision(api, args.repo, QPREFIX, ev)
        if rev is None:
            _log(f"{ev}: no bundle at HEAD; pass --revision SHA or --find-revision")
            return 2
    root = os.path.join(args.out, ev)
    forcing_dir, out_dir = os.path.join(root, "forcing"), os.path.join(root, "out")
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(forcing_dir)
    _log(f"{ev}: reading the spec at revision {rev[:12]} (anonymous, read-only)")
    with open(hf_hub_download(args.repo, f"{QPREFIX}/{ev}.json", repo_type="dataset",
                              revision=rev), encoding="utf-8") as fp:
        spec = json.load(fp)
    if not spec.get("domain"):
        _log(f"{ev}: spec has no basin domain - the worker never solves it (rectangle)")
        return 2
    cells = expected_cells(spec)
    if cells > args.max_cells:
        _log(f"{ev}: {cells / 1e6:.1f} M cells exceeds the {args.max_cells / 1e6:.0f} M budget; "
             f"the worker would not claim it (never coarsened). Raise --max-cells only if the "
             f"card holds it.")
        return 2
    _log(f"{ev}: downloading the forcing bundle")
    tar = hf_hub_download(args.repo, f"{QPREFIX}/{ev}.forcing.tar.gz", repo_type="dataset",
                          revision=rev)
    with tarfile.open(tar) as tf:
        tf.extractall(forcing_dir)
    t0 = datetime.datetime.strptime(spec["t0"], TMIN)
    t_end = datetime.datetime.strptime(spec["t_end"], TMIN)
    _log(f"{ev}: {len(os.listdir(forcing_dir))} grids; window {spec['t0']} .. {spec['t_end']}; "
         f"{cells / 1e6:.1f} M bbox cells, dem_res {spec.get('dem_res', '1')}")

    cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    if args.device.startswith("cuda") and not cuda:
        _log("CUDA requested but torch.cuda.is_available() is False")
        return 1
    free0 = total = None
    if cuda:
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats()
        free0, total = torch.cuda.mem_get_info()

    # identical to crestimap.worker.Worker.process_visit (new session)
    cfg = EventConfig(
        event_id=ev, bbox=tuple(spec["bbox_basin"]), t0=t0, t_end=t_end,
        sim_start=datetime.datetime.strptime(spec["sim_start"], TMIN),
        ef5_output_dir=forcing_dir, out_dir=out_dir,
        model=spec.get("model", "crest"), dem_res=str(spec.get("dem_res", "1")),
        dem_cache=os.environ.get("EVENT_DEM_CACHE", os.path.join(args.out, "dem_cache")),
        max_cells=args.max_cells, trigger=spec.get("trigger"), device=args.device,
        devices=None, n_tiles=0, dt_every=args.dt_every,
        progress=lambda s: _log(f"{ev}: {s}"))
    t_wall = time.time()
    sess = EventSession(cfg, resume=True)
    ny, nx = sess.grid.z.shape
    n_spec = (spec.get("domain") or {}).get("n_bbox") or cells
    native = abs(ny * nx - n_spec) <= 0.01 * n_spec
    _log(f"{ev}: grid {ny}x{nx} = {ny * nx / 1e6:.2f} M (spec n_bbox {n_spec / 1e6:.2f} M), "
         f"dx {sess.grid.dx:.1f} m -> {'native' if native else 'COARSENED'}")
    if not native:
        _log("FAIL: the DEM was block-meaned (resolution coarsened) - never acceptable")
        return 1
    t_state = sess.state_time()
    manifest = sess.advance(forcing_dir, out_dir, t0, t_end, device=args.device)
    solve_s = time.time() - t_wall

    # exactly the tail of the worker's visit (eventsim.run_one)
    manifest["gauge"] = spec["gauge"]["id"]
    manifest["hydro"] = spec.get("hydro") or []
    manifest["status"] = "active"
    manifest["basin"] = spec.get("basin")
    manifest["huc12s"] = spec.get("huc12s")
    manifest["domain"] = {**{k: v for k, v in (spec.get("domain") or {}).items()
                             if k != "hucs"}, **(manifest.get("domain") or {})}
    eventsim._update_archive(out_dir, manifest, lambda s: _log(f"{ev}: {s}"))
    if not manifest.get("archive_frames"):
        manifest["status"] = "ended"
        manifest["dry"] = True
    eventsim._make_pngs(out_dir, manifest)
    wall_s = time.time() - t_wall
    sim_h = max((t_end - t_state).total_seconds() / 3600.0, 1e-9)

    rep = {
        "event_id": ev, "revision": rev, "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "grid": [ny, nx], "dx_m": round(sess.grid.dx, 2), "n_bbox_spec": n_spec,
        "native_resolution": native,
        "n_active": (manifest.get("domain") or {}).get("n_active"),
        "resumed_from": t_state.strftime(TMIN), "t_end": spec["t_end"],
        "simulated_h": round(sim_h, 2), "frames": len(manifest.get("frames") or []),
        "archive_frames": manifest.get("archive_frames"), "dry": bool(manifest.get("dry")),
        "solve_s": round(solve_s, 1), "wall_s": round(wall_s, 1),
        "gpu_min_per_Mcell_simh": round(solve_s / 60.0 / (ny * nx / 1e6) / sim_h, 3),
    }
    if cuda:
        alloc, resv = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
        ctx = max(total - free0, 0)
        bpc = resv / (ny * nx)
        rep.update({
            "gpu_total_GiB": round(total / 2**30, 2),
            "context_before_GiB": round(ctx / 2**30, 2),
            "peak_allocated_GiB": round(alloc / 2**30, 2),
            "peak_reserved_GiB": round(resv / 2**30, 2),
            "peak_fraction_of_gpu": round((resv + ctx) / total, 3),
            "bytes_per_bbox_cell_reserved": round(bpc, 1),
            "suggested_BYTES_PER_CELL_EST": int(bpc * 1.1) + 1,
            "suggested_EVENT_MAX_CELLS_GPU": int((0.85 * total - max(ctx, 2**30)) / (bpc * 1.1)),
        })
    rep["published_comparison"] = compare_published(args.repo, ev, out_dir, manifest)
    with open(os.path.join(root, "replay_report.json"), "w", encoding="utf-8") as fp:
        json.dump(rep, fp, indent=1)
    if not args.keep:
        shutil.rmtree(forcing_dir, ignore_errors=True)
    print(json.dumps(rep, indent=1))
    _log(f"{ev}: report -> {os.path.join(root, 'replay_report.json')} (nothing was published)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
