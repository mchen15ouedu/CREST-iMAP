"""DEM burn screen — is the DEM hydrologically conditioned (channels burned)
before a CREST-iMAP run?

Raw bare-earth DEMs (3DEP, SRTM, lidar) carry road embankments, bridge
decks and culvert crossings straight across the river. The 2-D solver
routes on the DEM as given, so each crossing is a dam: water ponds behind
it and the reach downstream starves. Burning the channel network in
(tools/burn_channels.py, or any other method) removes those blockages.

The screen works on any DEM, however it was burned:

* checked cells : the channel mask's cells when one is given (the burn
                  tool writes it); otherwise the DEM's main stems, cells
                  draining >= 20 % of the domain's largest upstream area
                  (and >= 1 km2). Smaller DEM-derived "streams" are not
                  used: in lake, kettle or karst terrain they run through
                  natural closed depressions that no burn removes.
* filled_frac   : share of checked cells sitting in a depression (the
                  Priority-Flood fill raises them by more than tol_m). Every
                  embankment across a river fills the reach behind it.
* adherence     : only with a channel mask: share of channel cells whose
                  steepest descent on the DEM leads to another channel cell
                  or out of the domain. A burned network carries its own
                  water; a raw one spills out of it.
* fill p95 / max: reported, not judged (lakes on the network keep the max
                  at metres even after burning).

Calibration on two 1" 3DEP basins (NHDPlus HR flowlines, burned with
tools/burn_channels.py), raw -> burned:
  Gaylord MI (glacial)  channel filled 39.2 % -> 0.3 %, adherence 51 -> 99 %,
                        main stem filled 43.3 % -> 0.0 %
  Iowa                  channel filled 24.1 % -> 2.4 %, adherence 64 -> 95 %,
                        main stem filled 43.9 % -> 0.2 %

It never stops a run: CRESTiMAP logs the verdict at the start and the end.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import math

import numpy as np

from . import hydronet

log = logging.getLogger("crestimap")

# Thresholds (from the calibration above, with margin both ways)
MAX_FILLED_FRAC = 0.05
MIN_ADHERENCE = 0.90
MAIN_STEM_FRAC = 0.20
BURN_TAG = "CRESTIMAP_BURN"


@dataclasses.dataclass
class BurnScreen:
    passed: bool
    checked: str          # "channel mask" or "main stems"
    stream_km2: float
    n_cells: int
    filled_frac: float
    fill_p95_m: float
    fill_max_m: float
    adherence: float | None
    reasons: list
    note: str = ""

    def summary(self):
        a = "n/a" if self.adherence is None else f"{self.adherence:.1%}"
        head = "DEM burn screen: " + ("PASS" if self.passed else "FAIL")
        where = (self.checked if self.checked == "channel mask" else
                 f"main stems >= {self.stream_km2:.3g} km2")
        body = (f"{self.n_cells} cells checked ({where}); in depressions "
                f"{self.filled_frac:.1%} (limit {MAX_FILLED_FRAC:.0%}); "
                f"channel adherence {a} (limit {MIN_ADHERENCE:.0%}); "
                f"fill p95 {self.fill_p95_m:.2f} m, max {self.fill_max_m:.2f} m")
        out = f"{head} - {body}"
        if self.note:
            out += f" [{self.note}]"
        if not self.passed:
            out += (". " + "; ".join(self.reasons) + ". The DEM does not look "
                    "burned: crossings will dam the river and inundation will "
                    "be wrong. Burn it (tools/burn_channels.py) and re-run; "
                    "results of this run are compromised.")
        return out

    def log(self, when=""):
        msg = (when + " " if when else "") + self.summary()
        (log.info if self.passed else log.warning)(msg)

    def as_dict(self):
        return dataclasses.asdict(self)


def _steepest_to(z, active, dx, dy):
    """Flat index of each cell's strictly-lowest active neighbour (-1 none)."""
    ny, nx = z.shape
    diag = math.hypot(dx, dy)
    zp = np.pad(z, 1, constant_values=np.nan)
    ap = np.pad(active, 1, constant_values=False)
    best = np.zeros((ny, nx))
    tgt = np.full((ny, nx), -1, np.int64)
    ii, jj = np.mgrid[0:ny, 0:nx]
    for k in range(8):
        di, dj = hydronet.DI[k], hydronet.DJ[k]
        d = dx if di == 0 else dy if dj == 0 else diag
        zn = zp[1 + di:1 + di + ny, 1 + dj:1 + dj + nx]
        an = ap[1 + di:1 + di + ny, 1 + dj:1 + dj + nx]
        g = np.where(an, (z - zn) / d, -np.inf)
        upd = g > best
        best = np.where(upd, g, best)
        tgt = np.where(upd, (ii + di) * nx + (jj + dj), tgt)
    return tgt


def screen_dem(z, dx, dy, active=None, channel_mask=None, net=None,
               tol_m=0.05) -> BurnScreen:
    """Screen DEM `z` (ny, nx) [m] with cell sizes dx, dy [m]."""
    z = np.asarray(z, float)
    act = np.ones(z.shape, bool) if active is None else np.asarray(active, bool)
    if net is None:
        net = hydronet.flow_network(z, dx, dy, active=act)
    ch = None
    if channel_mask is not None:
        ch = np.asarray(channel_mask, bool) & act
        if not ch.any():
            ch = None
    if ch is not None:
        cells, checked, thr = ch, "channel mask", 0.0
    else:
        thr = max(1.0, MAIN_STEM_FRAC * float(net.acc_km2.max()))
        cells, checked = act & (net.acc_km2 >= thr), "main stems"
    n = int(cells.sum())
    if n == 0:
        return BurnScreen(True, checked, thr, 0, 0.0, 0.0, 0.0, None, [],
                          note="no stream draining >= 1 km2: domain too small "
                               "to screen")
    fs = (net.zf - z)[cells]
    filled_frac = float((fs > tol_m).mean())
    reasons = []
    if filled_frac > MAX_FILLED_FRAC:
        reasons.append(f"{filled_frac:.1%} of {checked} cells sit in depressions "
                       "(embankments / crossings damming the river)")
    adh = None
    if ch is not None:
        tgt = _steepest_to(z, act, dx, dy).ravel()
        chf = ch.ravel()
        idx = np.flatnonzero(chf)
        t = tgt[idx]
        ok = (net.ds[idx] < 0) | ((t >= 0) & chf[np.maximum(t, 0)])
        adh = float(ok.mean())
        if adh < MIN_ADHERENCE:
            reasons.append(f"only {adh:.1%} of channel cells drain along the "
                           "channel")
    return BurnScreen(not reasons, checked, thr, n, filled_frac,
                      float(np.percentile(fs, 95)), float(fs.max()), adh, reasons)


def read_burn_tag(dem_path):
    """The burn tool's provenance tag of a GeoTIFF, or None."""
    import rasterio
    with rasterio.open(dem_path) as ds:
        tag = ds.tags().get(BURN_TAG)
    return json.loads(tag) if tag else None


def main(argv=None):
    """python -m crestimap.screen dem.tif [--channels channels.tif]"""
    import argparse
    import rasterio
    from .forcing import SolverGrid
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("dem")
    ap.add_argument("--channels", help="channel mask GeoTIFF (same grid)")
    ap.add_argument("--mask", help="basin mask GeoTIFF (same grid, 1 = active)")
    a = ap.parse_args(argv)
    grid = SolverGrid.from_dem(a.dem)
    act = None
    if a.mask:
        with rasterio.open(a.mask) as ds:
            act = ds.read(1) > 0
    ch = None
    if a.channels:
        with rasterio.open(a.channels) as ds:
            ch = ds.read(1) > 0
    tag = read_burn_tag(a.dem)
    print("burn tag:", json.dumps(tag) if tag else "none")
    r = screen_dem(grid.z.cpu().numpy(), grid.dx, grid.dy, active=act,
                   channel_mask=ch)
    print(r.summary())
    return 0 if r.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
