"""Burn tool (tools/burn_channels.py) and DEM burn screen: a valley with
road embankments across the river fails the screen; burned, it passes;
the burn only lowers terrain and its bed falls monotonically downstream."""
import importlib.util
import logging
import os

import numpy as np

from crestimap import hydronet as hn
from crestimap import screen

_TOOL = os.path.join(os.path.dirname(__file__), "..", "..", "tools",
                     "burn_channels.py")
_spec = importlib.util.spec_from_file_location("burn_channels", _TOOL)
burn_channels = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(burn_channels)

DX = 30.0
NY, NX = 61, 121


def _valley_with_roads():
    """Thalweg along row 30 falling toward j=0, three embankments 2 m high
    across the valley (the raw-DEM failure mode)."""
    i, j = np.mgrid[0:NY, 0:NX]
    z = 0.02 * np.abs(i - 30) * DX * 0.1 + 0.002 * DX * j
    for jr in (30, 60, 90):
        z[:, jr] = np.maximum(z[:, jr], z[30, jr] + 2.0)
    stream = np.zeros((NY, NX), bool)
    stream[30, :] = True
    return z, stream


def test_raw_valley_fails_screen():
    z, stream = _valley_with_roads()
    r = screen.screen_dem(z, DX, DX)
    assert not r.passed and r.checked == "main stems"
    assert r.filled_frac > 0.2
    r2 = screen.screen_dem(z, DX, DX, channel_mask=stream)
    # one-cell embankments only pit the cell above each; the fill catches it
    assert not r2.passed and r2.filled_frac > 0.5 and r2.adherence < 1.0


def test_burned_valley_passes_screen():
    z, stream = _valley_with_roads()
    zb, chan, stats = burn_channels.burn(z, DX, DX, stream,
                                         np.full(z.shape, np.nan),
                                         min_area_km2=0.0)
    assert np.all(zb <= z + 1e-12), "burning only lowers terrain"
    assert chan[30].all()
    bed = zb[30]
    assert np.all(np.diff(bed) > 0), "bed rises strictly away from the outlet"
    r = screen.screen_dem(zb, DX, DX, channel_mask=chan)
    assert r.passed, r.summary()
    assert r.adherence >= 0.99
    assert screen.screen_dem(zb, DX, DX).passed


def test_bankfull_geometry_from_area():
    z, stream = _valley_with_roads()
    area = np.where(stream, 100.0, np.nan)          # km2 on every line cell
    zb, chan, stats = burn_channels.burn(z, DX, DX, stream, area,
                                         min_area_km2=0.0, buffer_cells=0)
    assert np.isclose(stats["max_depth_m"], 0.30 * 100 ** 0.213)
    assert np.isclose(stats["max_width_m"], 2.70 * 100 ** 0.352)
    # 2.70*100^0.352 = 13.4 m < one 30 m cell: the channel is the line itself
    assert chan.sum() == stream.sum()


def test_screen_logs_warning(caplog):
    z, _ = _valley_with_roads()
    r = screen.screen_dem(z, DX, DX)
    with caplog.at_level(logging.WARNING, logger="crestimap"):
        r.log("run start:")
    assert any("FAIL" in m and "compromised" in m for m in caplog.messages)


def test_tiny_domain_is_not_judged():
    z = np.add.outer(np.zeros(5), np.arange(5.0))
    r = screen.screen_dem(z, DX, DX)
    assert r.passed and r.note


# ------------------------------------------------ CLI end to end (files)
def _cli(args):
    """Run the tool as users do: its own process (no torch loaded)."""
    import subprocess
    import sys
    p = subprocess.run([sys.executable, _TOOL] + [str(a) for a in args],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
    return p.stdout


def _write_dem(path, z):
    import rasterio
    from rasterio.transform import from_origin
    with rasterio.open(path, "w", driver="GTiff", height=z.shape[0],
                       width=z.shape[1], count=1, dtype="float64",
                       crs="EPSG:32615",
                       transform=from_origin(500000.0, 4000000.0, DX, DX)) as ds:
        ds.write(z, 1)


def _check_outputs(out):
    import rasterio
    tag = screen.read_burn_tag(str(out))
    assert tag and tag["line_cells"] > 0
    with rasterio.open(out) as ds:
        zb = ds.read(1).astype(float)
    with rasterio.open(str(out).replace(".tif", "_channels.tif")) as ds:
        ch = ds.read(1) > 0
    assert ch[30].sum() >= NX - 2
    r = screen.screen_dem(zb, DX, DX, channel_mask=ch)
    assert r.passed, r.summary()


def test_cli_vector_flowlines(tmp_path):
    import json
    z, _ = _valley_with_roads()
    dem = tmp_path / "dem.tif"
    _write_dem(dem, z)
    # the thalweg as a line, through cell centres of row 30, in EPSG:32615
    y = 4000000.0 - (30 + 0.5) * DX
    xs = [500000.0 + (j + 0.5) * DX for j in (NX - 1, 0)]
    fc = {"type": "FeatureCollection",
          "crs": {"type": "name", "properties": {"name": "EPSG:32615"}},
          "features": [{"type": "Feature",
                        "properties": {"TotDASqKm": 5.0, "FType": 460},
                        "geometry": {"type": "LineString",
                                     "coordinates": [[xs[0], y], [xs[1], y]]}},
                       {"type": "Feature",
                        "properties": {"TotDASqKm": 0.0, "FType": 566},
                        "geometry": {"type": "LineString",
                                     "coordinates": [[xs[0], y - 600], [xs[1], y - 600]]}}]}
    lines = tmp_path / "lines.geojson"
    lines.write_text(json.dumps(fc))
    out = tmp_path / "burned.tif"
    _cli(["--dem", dem, "--out", out, "--flowlines", lines,
          "--area-field", "TotDASqKm", "--query", "FType != 566",
          "--min-area-km2", "0"])
    _check_outputs(out)
    import rasterio
    with rasterio.open(str(out).replace(".tif", "_channels.tif")) as ds:
        assert ds.read(1)[30 + 20].sum() == 0, "the coastline was filtered out"


def test_cli_merit_dir_upa(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    z, _ = _valley_with_roads()
    dem = tmp_path / "dem.tif"
    _write_dem(dem, z)
    # coarse "MERIT" grid: 3x the DEM cell, river along its row 10, flowing
    # west (dir 16), upa growing downstream
    n_r, n_c = NY // 3, NX // 3 + 1
    # shifted one fine cell north so coarse row 10's centre is DEM row 30
    tr = from_origin(500000.0, 4000000.0 + DX, 3 * DX, 3 * DX)
    d8 = np.zeros((n_r, n_c), "uint8")
    upa = np.zeros((n_r, n_c), "float32")
    d8[10, :] = 16
    upa[10, :] = np.linspace(10.0, 1.0, n_c)
    for name, arr in (("dir.tif", d8), ("upa.tif", upa)):
        with rasterio.open(tmp_path / name, "w", driver="GTiff", height=n_r,
                           width=n_c, count=1, dtype=arr.dtype,
                           crs="EPSG:32615", transform=tr) as ds:
            ds.write(arr, 1)
    out = tmp_path / "burned.tif"
    _cli(["--dem", dem, "--out", out, "--merit-dir", tmp_path / "dir.tif",
          "--merit-upa", tmp_path / "upa.tif", "--min-area-km2", "0"])
    _check_outputs(out)
