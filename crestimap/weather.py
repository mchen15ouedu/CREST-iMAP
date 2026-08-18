"""Weather forcing for the full CREST-iMAP model: precipitation and PET as
callables t_seconds -> (ny, nx) tensor [m/s] on the solver grid, exactly
the interface `SWESolver.run(rain_fn=...)` and `CRESTiMAP` consume.

Readers
-------
* `tif_stack(dir, pattern, ...)`   — a directory of GeoTIFFs, one per time,
  timestamp in the file name (EF5/MRMS style `precip.202608141200.tif`,
  or any pattern with a `{time}` placeholder + strftime format). Any CRS /
  resolution; regridded to the solver grid ("containing" = every fine cell
  takes its coarse cell's rate, mass-conserving for rates).
* `uniform_series(times, values)`  — one value per time, spatially uniform
  (hyetographs, tests).
* `constant(rate)`                 — e.g. a climatological PET in mm/day.

Units: `units="mm/h"` (rate, default), `"mm"` (accumulation over the
interval to the NEXT file — converted to a rate), `"mm/day"`, `"m/s"`.
All readers hold the value piecewise-constant until the next timestamp and
expose `next_change(t)` so the solver lands exactly on switches.
"""
from __future__ import annotations

import datetime as _dt
import glob
import os
import re

import numpy as np
import torch

from .forcing import GriddedSeriesForcing, regrid_to_solver

_UNIT_TO_MM_PER_H = {"mm/h": 1.0, "mm/hr": 1.0, "mm/day": 1.0 / 24.0,
                     "mm/d": 1.0 / 24.0, "m/s": 3.6e6}


def _to_mm_per_h(arr, units, interval_s=None):
    u = units.lower()
    if u == "mm":
        if not interval_s:
            raise ValueError("units='mm' needs the interval to the next file")
        return arr / (interval_s / 3600.0)
    if u not in _UNIT_TO_MM_PER_H:
        raise ValueError(f"unknown units {units!r}")
    return arr * _UNIT_TO_MM_PER_H[u]


def _pattern_to_regex(pattern: str, time_format: str):
    """'precip.{time}.tif' + '%Y%m%d%H%M' -> compiled regex with a 'time'
    group of the right width."""
    width = len(_dt.datetime(2000, 1, 1).strftime(time_format))
    esc = re.escape(pattern).replace(r"\{time\}", f"(?P<time>.{{{width}}})")
    return re.compile("^" + esc + "$")


def list_stack(directory: str, pattern: str = "{time}.tif",
               time_format: str = "%Y%m%d%H%M"):
    """[(datetime, path), ...] sorted, for files matching pattern."""
    rx = _pattern_to_regex(pattern, time_format)
    out = []
    for p in glob.glob(os.path.join(directory, "*")):
        m = rx.match(os.path.basename(p))
        if not m:
            continue
        try:
            when = _dt.datetime.strptime(m.group("time"), time_format)
        except ValueError:
            continue
        out.append((when, p))
    out.sort()
    return out


def tif_stack(directory: str, grid, t0: _dt.datetime, pattern: str = "{time}.tif",
              time_format: str = "%Y%m%d%H%M", units: str = "mm/h",
              scale: float = 1.0, nodata_fill: float = 0.0, device=None,
              dtype=None) -> GriddedSeriesForcing:
    """Directory of timestamped GeoTIFFs -> forcing callable [m/s].

    grid : SolverGrid (target raster). scale multiplies raw values (e.g.
    0.1 for tenths). Missing/nodata cells become `nodata_fill` mm/h.
    """
    import rasterio
    files = list_stack(directory, pattern, time_format)
    if not files:
        raise FileNotFoundError(f"no files matching {pattern!r} in {directory}")
    times = [t for t, _ in files]
    paths = [p for _, p in files]
    dst_shape = tuple(grid.z.shape)

    def loader(i):
        with rasterio.open(paths[i]) as ds:
            a = ds.read(1).astype(float) * scale
            if ds.nodata is not None:
                a = np.where(a == ds.nodata, np.nan, a)
            a = regrid_to_solver(a, ds.transform, dst_shape, grid.transform,
                                 ds.crs, grid.crs, method="containing")
        a = np.where(np.isfinite(a), a, nodata_fill)
        interval = None
        if i + 1 < len(times):
            interval = (times[i + 1] - times[i]).total_seconds()
        elif i > 0:
            interval = (times[i] - times[i - 1]).total_seconds()
        return np.clip(_to_mm_per_h(a, units, interval), 0.0, None)

    return GriddedSeriesForcing(times, loader, t0, hold=True, device=device,
                                dtype=dtype)


def uniform_series(times, values, grid_shape, t0: _dt.datetime,
                   units: str = "mm/h", device=None, dtype=None):
    """Spatially uniform time series -> forcing callable [m/s]."""
    times = list(times)
    vals = [float(v) for v in values]
    if len(times) != len(vals):
        raise ValueError("times and values must have the same length")

    def loader(i):
        interval = None
        if i + 1 < len(times):
            interval = (times[i + 1] - times[i]).total_seconds()
        rate = _to_mm_per_h(np.array(vals[i]), units, interval)
        return np.full(grid_shape, float(rate))

    return GriddedSeriesForcing(times, loader, t0, hold=True, device=device,
                                dtype=dtype)


class ConstantForcing:
    """Spatially and temporally constant rate (e.g. PET 4 mm/day)."""

    def __init__(self, value, grid_shape, units="mm/day", device=None, dtype=None):
        rate_mm_h = float(_to_mm_per_h(np.array(float(value)), units))
        self._t = torch.full(tuple(grid_shape), rate_mm_h / 3600.0 / 1000.0,
                             dtype=dtype or torch.get_default_dtype(), device=device)

    def __call__(self, t_seconds):
        return self._t

    def next_change(self, t_seconds):
        return float("inf")


def constant(value, grid_shape, units="mm/day", device=None, dtype=None):
    return ConstantForcing(value, grid_shape, units, device, dtype)


def accumulate(fn, t_start: float, t_end: float):
    """Exact time integral of a piecewise-constant forcing over [t_start,
    t_end] -> depth tensor [m]. Walks the forcing's own switch points."""
    t = float(t_start)
    total = None
    while t < t_end - 1e-9:
        rate = fn(t)
        nc = fn.next_change(t) if hasattr(fn, "next_change") else float("inf")
        seg = min(t_end - t, nc if nc > 1e-9 else t_end - t)
        total = rate * seg if total is None else total + rate * seg
        t += seg
    if total is None:
        total = fn(t_start) * 0.0
    return total
