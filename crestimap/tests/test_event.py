"""End-to-end event simulation on synthetic data (no network, no EF5):
valley DEM + hourly EF5-format q/runoff/subrunoff grids -> run_event ->
depth frames, maxdepth, manifest. Also unit-checks 3DEP tile naming."""
import datetime
import json
import os
import pathlib
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from crestimap import EventConfig, run_event
from crestimap.dem import tile_name, tiles_for_bbox
from crestimap.io import read_depth

# CRESTIMAP_TEST_DEVICE=cuda exercises run_event's production device path
DEVICE = os.environ.get("CRESTIMAP_TEST_DEVICE", "cpu")

rasterio = None
try:
    import rasterio
    from rasterio.transform import from_origin
except ImportError:
    pass

SEC3 = 1.0 / 1200.0
W, N = -97.5, 35.4
NYC, NXC = 12, 16
T0 = datetime.datetime(2026, 8, 10, 12, 0)


def test_tile_naming():
    assert tile_name(34, -98) == "n35w098"
    ts = tiles_for_bbox(-97.9, 34.2, -97.1, 34.8)
    assert ts == ["n35w098"]
    ts2 = tiles_for_bbox(-98.2, 34.8, -97.1, 35.3)
    assert set(ts2) == {"n35w099", "n35w098", "n36w099", "n36w098"}


def write_synthetic_domain(d, ny, nx, tr, n_units=2):
    """Basin-shaped test domain: an ellipse over the valley (the channel row
    is inside), split into `n_units` HUC-like units along x. Written as
    domain_huc.tif exactly like hf_data/hucdomain.py does for real events."""
    yy, xx = np.mgrid[0:ny, 0:nx]
    ell = ((xx - nx / 2) / (0.48 * nx)) ** 2 + ((yy - ny / 2) / (0.42 * ny)) ** 2 <= 1.0
    lab = np.full((ny, nx), -1, dtype=np.int16)
    for k in range(n_units):
        sel = ell & (xx >= k * nx / n_units) & (xx < (k + 1) * nx / n_units)
        lab[sel] = k
    with rasterio.open(str(d / "domain_huc.tif"), "w", driver="GTiff",
                       height=ny, width=nx, count=1, dtype="int16",
                       crs="EPSG:4326", transform=tr, nodata=-1) as ds:
        ds.write(lab, 1)
    return lab


def _synthetic_event_dir(d):
    """Valley DEM (1/3 of 3" res = 1" cells here for speed) + EF5 grids
    + basin domain raster."""
    # DEM at 1 arc-sec (3x refinement of the 3" EF5 grid): tilted valley
    ny, nx = NYC * 3, NXC * 3
    tr = from_origin(W, N, SEC3 / 3, SEC3 / 3)
    yy, xx = np.mgrid[0:ny, 0:nx]
    z = (0.002 * xx * 30.0                       # downstream tilt along x
         + 0.5 * np.abs(yy - ny / 2) / (ny / 2) * 8.0)  # valley cross-section
    dem_path = str(d / "dem_in.tif")
    with rasterio.open(dem_path, "w", driver="GTiff", height=ny, width=nx,
                       count=1, dtype="float32", crs="EPSG:4326",
                       transform=tr) as ds:
        ds.write(z.astype(np.float32), 1)
    write_synthetic_domain(d, ny, nx, tr)
    # EF5 grids at 3"
    tre = from_origin(W, N, SEC3, SEC3)
    rng = np.random.default_rng(3)
    for hh in range(4):
        t = T0 + datetime.timedelta(hours=hh - 2)   # covers sim_start..t_end
        for kind, base in (("runoff", 25.0), ("subrunoff", 4.0)):
            arr = (base * (0.5 + rng.random((NYC, NXC)))).astype(np.float32)
            with rasterio.open(str(d / f"{kind}.{t:%Y%m%d%H%M}.crest.tif"),
                               "w", driver="GTiff", height=NYC, width=NXC,
                               count=1, dtype="float32", crs="EPSG:4326",
                               transform=tre) as ds:
                ds.write(arr, 1)
        q = np.zeros((NYC, NXC), dtype=np.float32)
        q[NYC // 2, :] = 40.0                       # channel along the valley
        with rasterio.open(str(d / f"q.{t:%Y%m%d%H%M}.crest.tif"), "w",
                           driver="GTiff", height=NYC, width=NXC, count=1,
                           dtype="float32", crs="EPSG:4326",
                           transform=tre) as ds:
            ds.write(q, 1)
    return dem_path


def test_run_event_end_to_end():
    with tempfile.TemporaryDirectory() as td:
        d = pathlib.Path(td)
        dem_path = _synthetic_event_dir(d)
        out = d / "event_out"
        cfg = EventConfig(
            event_id="test_evt", bbox=(W, N - NYC * SEC3, W + NXC * SEC3, N),
            t0=T0, t_end=T0 + datetime.timedelta(hours=1),
            ef5_output_dir=str(d), out_dir=str(out), model="crest",
            sim_start=T0 - datetime.timedelta(hours=1),
            dem_path=dem_path, output_every_s=900.0, device=DEVICE)
        log = []
        cfg.progress = log.append
        man = run_event(cfg)
        # manifest + files
        assert (out / "manifest.json").exists()
        assert json.loads((out / "manifest.json").read_text())["event_id"] == "test_evt"
        assert (out / "maxdepth.tif").exists() and (out / "dem.tif").exists()
        assert len(man["frames"]) >= 7          # 2 h at 15-min cadence
        for fr in man["frames"]:
            assert (out / fr["file"]).exists()
        # water actually accumulated in the valley
        md, _, _ = read_depth(str(out / "maxdepth.tif"))
        assert md.max() > 0.01, f"max depth only {md.max():.4f} m"
        # routed-channel coupling: the q=40 m3/s channel (coarse row NYC//2)
        # must hold rating-depth water (~1 m) through the run
        ch = md[NYC // 2 * 3 - 1: NYC // 2 * 3 + 4, :]
        assert ch.max() > 0.5, f"channel depth only {ch.max():.3f} m"
        assert any("channel coupling" in s for s in log)
        # frames are compact (storage budget: uint16+deflate)
        biggest = max((out / fr["file"]).stat().st_size for fr in man["frames"])
        assert biggest < 200_000, f"frame unexpectedly large: {biggest} B"
        # BASIN, not rectangle: no water anywhere outside the domain raster
        assert man["domain"] and man["domain"]["n_units"] == 2
        with rasterio.open(str(d / "domain_huc.tif")) as ds:
            lab = ds.read(1)
        assert md[lab < 0].max() == 0.0, "water outside the basin domain"
        assert any("basin domain" in s for s in log)
        # channel pre-wet happened
        assert any("channel pre-wet" in s for s in log)


def test_run_event_refuses_rectangle():
    """No domain raster in the bundle -> error, never a rectangular run."""
    with tempfile.TemporaryDirectory() as td:
        d = pathlib.Path(td)
        dem_path = _synthetic_event_dir(d)
        os.remove(str(d / "domain_huc.tif"))
        cfg = EventConfig(
            event_id="test_box", bbox=(W, N - NYC * SEC3, W + NXC * SEC3, N),
            t0=T0, t_end=T0 + datetime.timedelta(minutes=30),
            ef5_output_dir=str(d), out_dir=str(d / "o"), model="crest",
            sim_start=T0, dem_path=dem_path, device=DEVICE)
        try:
            run_event(cfg)
        except FileNotFoundError as e:
            assert "rectangle" in str(e)
        else:
            raise AssertionError("run_event ran without a basin domain")


if __name__ == "__main__":
    if rasterio is None:
        print("SKIP: rasterio not available")
        sys.exit(0)
    for fn in (test_tile_naming, test_run_event_end_to_end,
               test_run_event_refuses_rectangle):
        fn()
        print(f"PASS  {fn.__name__}")
    print("all event tests passed")
