"""Forcing and initial-condition interface to CREST-AI (EF5/CREST).

In the coupled deployment the CREST water balance is NOT run inside this
package (that was v1.x). Instead the CREST-AI pipeline runs EF5/CREST for
the event window and hands over:

  initial conditions : 2-D discharge (Q) and soil-moisture (SM) grids at
                       event start (channel pre-wetting / antecedent state)
  forcing            : gridded surface runoff (fast flow) + subsurface
                       runoff (interflow/baseflow) per timestep, which enter
                       the shallow-water equations as the lateral-inflow
                       source term [m/s]

EF5 writes GeoTIFF output grids; this module regrids them onto the solver
raster and exposes a `rain_fn(t)`-style callable for `SWESolver.run`.

Status: interface is stable, EF5-file plumbing is experimental until wired
to real CREST-AI event output.
"""
from __future__ import annotations

import bisect
import datetime as _dt

import numpy as np
import torch


class GriddedSeriesForcing:
    """Time series of lateral-inflow grids -> callable(t_seconds) -> tensor.

    Parameters
    ----------
    times : sorted list of datetimes (grid validity times).
    loader : callable(index) -> 2-D numpy array in mm/h on the solver grid.
             (Keep loading lazy: events are long, grids are big.)
    t0 : datetime that corresponds to simulation time t = 0 s.
    hold : if True (default) use the most recent grid at or before t
           (piecewise-constant, mass-consistent with EF5 accumulation);
           if False, linearly interpolate between bracketing grids.
    """

    MM_PER_HOUR_TO_M_PER_S = 1.0 / 3600.0 / 1000.0

    def __init__(self, times, loader, t0, hold=True, device=None, dtype=None):
        self.times = list(times)
        self.loader = loader
        self.t0 = t0
        self.hold = hold
        self.device = device
        self.dtype = dtype or torch.get_default_dtype()
        self._cache = {}

    def _grid(self, i):
        if i not in self._cache:
            arr = np.asarray(self.loader(i), dtype=float)
            self._cache = {i: torch.as_tensor(
                arr * self.MM_PER_HOUR_TO_M_PER_S,
                dtype=self.dtype, device=self.device)}  # keep only latest
        return self._cache[i]

    def __call__(self, t_seconds: float) -> torch.Tensor:
        when = self.t0 + _dt.timedelta(seconds=float(t_seconds))
        i = bisect.bisect_right(self.times, when) - 1
        i = max(0, min(i, len(self.times) - 1))
        if self.hold or i == len(self.times) - 1:
            return self._grid(i)
        span = (self.times[i + 1] - self.times[i]).total_seconds()
        w = ((when - self.times[i]).total_seconds() / span) if span > 0 else 0.0
        return (1 - w) * self._grid(i) + w * self._grid(i + 1)

    def next_change(self, t_seconds: float) -> float:
        """Seconds from t until the forcing next switches grids (inf if never).
        SWESolver.run clips dt to this so piecewise-constant forcing is
        integrated exactly (no smearing across hour boundaries)."""
        when = self.t0 + _dt.timedelta(seconds=float(t_seconds))
        i = bisect.bisect_right(self.times, when)
        if i >= len(self.times):
            return float("inf")
        return (self.times[i] - when).total_seconds()


def sum_forcings(*fns):
    """Combine lateral-inflow forcings (surface runoff, subsurface runoff,
    inlet inflow). The combined tensor is rebuilt only when a component
    changes (piecewise-constant per hour), so its identity is stable within
    an hour — TiledSolver's per-tile slice cache keys on id()."""
    fns = [f for f in fns if f is not None]
    state = {"key": None, "val": None}

    def combined(t):
        parts = [f(t) for f in fns]
        key = tuple(id(p) for p in parts)
        if key != state["key"]:
            out = parts[0]
            for p in parts[1:]:
                out = out + p
            state["key"], state["val"] = key, torch.clamp(out, min=0.0)
        return state["val"]

    def next_change(t):
        return min((f.next_change(t) for f in fns if hasattr(f, "next_change")),
                   default=float("inf"))
    combined.next_change = next_change
    return combined


def regrid_to_solver(src: np.ndarray, src_transform, dst_shape, dst_transform,
                     src_crs=None, dst_crs=None, method="containing"):
    """Regrid an EF5 output grid onto the (finer) solver raster.

    method:
      "containing" (default) — every solver cell takes the value of the EF5
        cell that contains it. Runoff rate (mm/h) is intensive, so this
        partitions each coarse cell's water exactly over its footprint:
        mass-conserving by construction for coarse -> fine.
      "bilinear" — smooth field (NOT exactly mass-conserving; for SM/Q
        diagnostics, not for runoff forcing).
    Uses rasterio.warp when CRSs differ; direct index mapping otherwise.
    """
    same_crs = (src_crs is None and dst_crs is None) or (src_crs == dst_crs)
    if method == "containing" and same_crs:
        ny, nx = dst_shape
        rows, cols = np.mgrid[0:ny, 0:nx]
        xs, ys = dst_transform * (cols + 0.5, rows + 0.5)
        inv = ~src_transform
        sc, sr = inv * (xs, ys)
        sr = np.clip(np.floor(sr).astype(int), 0, src.shape[0] - 1)
        sc = np.clip(np.floor(sc).astype(int), 0, src.shape[1] - 1)
        return src[sr, sc].astype(float)
    from rasterio.warp import reproject, Resampling
    dst = np.zeros(dst_shape, dtype=float)
    reproject(source=src.astype(float), destination=dst,
              src_transform=src_transform, dst_transform=dst_transform,
              src_crs=src_crs or "EPSG:4326", dst_crs=dst_crs or "EPSG:4326",
              resampling=(Resampling.nearest if method == "containing"
                          else Resampling.bilinear))
    return dst


# --------------------------------------------------------------------------- #
# EF5 output plumbing (filename convention: <kind>.<YYYYMMDDHHMM>.<model>.tif)
# --------------------------------------------------------------------------- #

_M_PER_DEG_LAT = 111132.0


class SolverGrid:
    """Solver raster derived from a DEM GeoTIFF.

    Holds z (torch tensor), the affine transform + CRS, and metric cell
    sizes. For geographic CRS the cell size is converted to meters at the
    domain's mean latitude (adequate at basin scale); projected DEMs use
    native units.
    """

    def __init__(self, z, transform, crs, dx, dy):
        self.z = z
        self.transform = transform
        self.crs = crs
        self.dx = dx
        self.dy = dy

    @classmethod
    def from_array(cls, z_np, transform, crs, dtype=None, device=None):
        import math
        geographic = crs is None or getattr(crs, "is_geographic", True)
        if geographic:
            lat = transform.f + transform.e * z_np.shape[0] / 2.0
            dy = abs(transform.e) * _M_PER_DEG_LAT
            dx = abs(transform.a) * _M_PER_DEG_LAT * math.cos(math.radians(lat))
        else:
            dx, dy = abs(transform.a), abs(transform.e)
        z = torch.as_tensor(np.asarray(z_np, dtype=float),
                            dtype=dtype or torch.get_default_dtype(), device=device)
        return cls(z, transform, crs, dx, dy)

    @classmethod
    def from_dem(cls, dem_path, dtype=None, device=None, nodata_fill=None):
        import math
        import rasterio
        with rasterio.open(dem_path) as ds:
            arr = ds.read(1).astype(float)
            if ds.nodata is not None:
                fill = nodata_fill if nodata_fill is not None else np.nanmax(
                    np.where(arr == ds.nodata, -np.inf, arr))
                arr = np.where(arr == ds.nodata, fill, arr)
            tr = ds.transform
            crs = ds.crs
            geographic = crs is None or crs.is_geographic
            if geographic:
                lat = tr.f + tr.e * ds.height / 2.0  # center latitude
                dy = abs(tr.e) * _M_PER_DEG_LAT
                dx = abs(tr.a) * _M_PER_DEG_LAT * math.cos(math.radians(lat))
            else:
                dx, dy = abs(tr.a), abs(tr.e)
        z = torch.as_tensor(arr, dtype=dtype or torch.get_default_dtype(),
                            device=device)
        return cls(z, tr, crs, dx, dy)


def parse_ef5_dir(output_dir, kind, model=None):
    """List EF5 gridded outputs of one kind, sorted by time.

    Accepts <kind>.<YYYYMMDD_HHUU>.<model>.tif (EF5's actual
    currentTimeTextOutput format, Simulator.cpp SetNameStr) and the
    underscore-less <YYYYMMDDHHMM> variant. Returns [(datetime, path)].
    """
    import glob
    import os
    import re
    pat = os.path.join(str(output_dir), f"{kind}.*.tif")
    rx = re.compile(rf"{re.escape(kind)}\.(\d{{8}}_?\d{{4}})\.([^.]+)\.tif$")
    out = []
    for p in glob.glob(pat):
        m = rx.search(os.path.basename(p))
        if not m:
            continue
        if model and m.group(2).lower() != model.lower():
            continue
        ts = m.group(1).replace("_", "")
        out.append((_dt.datetime.strptime(ts, "%Y%m%d%H%M"), p))
    out.sort()
    return out


def ef5_forcing(output_dir, grid: SolverGrid, t0, kinds=("runoff", "subrunoff"),
                model=None, method="containing", hold=True,
                device=None, dtype=None):
    """Total lateral-inflow forcing from EF5 gridded runoff output.

    Sums the listed kinds (surface `runoff` + interflow `subrunoff`),
    regrids each hourly mm/h grid onto the solver raster (containing-cell:
    mass-conserving coarse->fine), converts to m/s, and returns a
    callable(t_seconds) for SWESolver.run. EF5 grids and nodata are
    clamped to >= 0.
    """
    import rasterio
    dst_shape = tuple(grid.z.shape)
    fns = []
    for kind in kinds:
        series = parse_ef5_dir(output_dir, kind, model)
        if not series:
            raise FileNotFoundError(
                f"no '{kind}.*.tif' grids in {output_dir} — did the EF5 run "
                f"use output_grids=...|{kind}|... ?")
        times = [t for t, _ in series]
        paths = [p for _, p in series]

        def loader(i, _paths=paths):
            with rasterio.open(_paths[i]) as ds:
                src = ds.read(1).astype(float)
                if ds.nodata is not None:
                    src = np.where(src == ds.nodata, 0.0, src)
                src = np.clip(src, 0.0, None)
                return regrid_to_solver(src, ds.transform, dst_shape,
                                        grid.transform, ds.crs, grid.crs,
                                        method=method)
        fns.append(GriddedSeriesForcing(times, loader, t0, hold=hold,
                                        device=device, dtype=dtype))
    return sum_forcings(*fns) if len(fns) > 1 else fns[0]


def rating_depth(q, n_ch=0.035, slope=1e-3, width_fn=None):
    """Manning-rating water depth [m] for channel discharge q [m3/s]:
    h ~ (n Q / (w sqrt(S)))^(3/5) with a Leopold-Maddock-style width."""
    q = torch.clamp(torch.as_tensor(q), min=0.0)
    if width_fn is None:
        width_fn = lambda qq: torch.clamp(7.2 * qq ** 0.5, min=1.0)
    w = width_fn(q)
    return (n_ch * q / (w * slope ** 0.5)) ** 0.6


class EF5InletInflow:
    """Routed-flow coupling as a BOUNDARY CONDITION.

    EF5's hourly discharge grids enter the 2-D domain ONLY where the EF5
    channel network crosses INTO the active basin — the upstream inlets of a
    trimmed or partial domain. Inside the basin the shallow-water solver
    routes CREST's own runoff grids (ef5_forcing); nothing else adds mass, so
    the map can never hold more water than CREST delivered. A full basin
    (headwaters included) has no inlets and gets no injection at all.

    History: until 2026-09-22 this coupling was a depth FLOOR applied after
    every solver step on every solver cell of every EF5 channel pixel
    (h := max(h, rating_depth(q))). It refilled whatever drained out of a
    channel cell and manufactured water at the channel conveyance rate for
    the whole run — published maps held 10x-1000x the event's outflow volume.
    Never reintroduce a per-step state clamp as a forcing.

    Inlet detection (once, from the network geometry = max q over the
    series): an EF5 pixel OUTSIDE the active mask with q >= q_min whose
    largest-q 8-neighbour is INSIDE the mask and carries more flow (flow goes
    toward larger q, so that neighbour is its downstream cell). Its hourly q
    is injected as a lateral-inflow rate [m/s] at the lowest active solver
    cell of that inside pixel (the channel). The pour point is never an
    inlet: there the outside pixel carries the larger q.

    Callable like ef5_forcing: t_seconds -> (ny, nx) tensor [m/s],
    piecewise-constant per EF5 hour, plus next_change(t).
    """

    def __init__(self, output_dir, grid, t0, active=None, model=None,
                 q_min=5.0, dtype=None, device=None):
        import rasterio
        self.series = parse_ef5_dir(output_dir, "q", model)
        if not self.series:
            raise FileNotFoundError(f"no 'q.*.tif' grids in {output_dir}")
        self.grid = grid
        self.t0 = t0
        self.q_min = float(q_min)
        self.dtype = dtype or torch.get_default_dtype()
        self.device = device
        self.times = [t for t, _ in self.series]
        qs = []
        for k, (_, p) in enumerate(self.series):
            with rasterio.open(p) as ds:
                a = ds.read(1).astype(float)
                if ds.nodata is not None:
                    a = np.where(a == ds.nodata, 0.0, a)
                qs.append(np.clip(a, 0.0, None))
                if k == 0:
                    src_tr, src_crs = ds.transform, ds.crs
        self.q = np.stack(qs)                          # (T, sy, sx) m3/s
        sy, sx = self.q.shape[1:]
        # solver cell -> containing EF5 pixel (same mapping as
        # regrid_to_solver "containing")
        ny, nx = tuple(grid.z.shape)
        rows, cols = np.mgrid[0:ny, 0:nx]
        xs, ys = grid.transform * (cols + 0.5, rows + 0.5)
        same_crs = (src_crs is None and grid.crs is None) or (src_crs == grid.crs)
        if not same_crs:
            from rasterio.warp import transform as _tf
            xs, ys = (np.asarray(v).reshape(ny, nx) for v in
                      _tf(grid.crs or "EPSG:4326", src_crs or "EPSG:4326",
                          xs.ravel(), ys.ravel()))
        inv = ~src_tr
        sc, sr = inv * (xs, ys)
        sr = np.floor(sr).astype(int)
        sc = np.floor(sc).astype(int)
        valid = (sr >= 0) & (sr < sy) & (sc >= 0) & (sc < sx)
        # keep the mapping for the level-based channel pre-wet
        self.block = (np.where(valid, sr, -1), np.where(valid, sc, -1))
        act = (active.detach().cpu().numpy().astype(bool)
               if active is not None else np.ones((ny, nx), bool))
        m = valid & act
        inside = np.zeros((sy, sx), bool)
        inside[sr[m], sc[m]] = True
        qmax = self.q.max(axis=0)
        chan = qmax >= self.q_min
        self.inlets = []            # (src_r, src_c, solver_r, solver_c)
        z = grid.z.detach().cpu().numpy()
        cand = np.argwhere(chan & ~inside)
        for i, j in cand:
            best, bq = None, qmax[i, j]
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ii, jj = i + di, j + dj
                    if 0 <= ii < sy and 0 <= jj < sx and qmax[ii, jj] > bq:
                        best, bq = (ii, jj), qmax[ii, jj]
            if best is None or not inside[best]:
                continue            # downstream is outside too: not our inlet
            cells = m & (sr == best[0]) & (sc == best[1])
            if not cells.any():
                continue
            zc = np.where(cells, z, np.inf)
            r, c = np.unravel_index(int(np.argmin(zc)), zc.shape)
            self.inlets.append((int(i), int(j), int(r), int(c)))
        self._cell_area = float(grid.dx * grid.dy)
        self._cache = {}

    # -- diagnostics ------------------------------------------------------ #
    def total_m3s(self, t_seconds=0.0) -> float:
        i = self._index(t_seconds)
        return float(sum(self.q[i, a, b] for a, b, _, _ in self.inlets))

    def stats(self, t_seconds=0.0):
        return {"inlets": len(self.inlets), "total_m3s": self.total_m3s(t_seconds),
                "peak_m3s": float(max((self.q[:, a, b].max() for a, b, _, _
                                       in self.inlets), default=0.0))}

    # -- forcing interface ------------------------------------------------ #
    def _index(self, t_seconds):
        when = self.t0 + _dt.timedelta(seconds=float(t_seconds))
        i = bisect.bisect_right(self.times, when) - 1
        return max(0, min(i, len(self.times) - 1))

    def _grid(self, i):
        if i not in self._cache:
            g = torch.zeros(tuple(self.grid.z.shape), dtype=self.dtype,
                            device=self.device)
            for a, b, r, c in self.inlets:
                g[r, c] += float(self.q[i, a, b]) / self._cell_area   # m/s
            self._cache = {i: g}                  # keep only the latest hour
        return self._cache[i]

    def __call__(self, t_seconds: float) -> torch.Tensor:
        return self._grid(self._index(t_seconds))

    def next_change(self, t_seconds: float) -> float:
        when = self.t0 + _dt.timedelta(seconds=float(t_seconds))
        i = bisect.bisect_right(self.times, when)
        if i >= len(self.times):
            return float("inf")
        return (self.times[i] - when).total_seconds()


def initial_state_from_ef5(q_grid, sm_grid, dem, dx, dy,
                           bankfull_width_fn=None, block=None, q_min=1.0):
    """Build (h, qx, qy) initial conditions from EF5 2-D Q and SM grids.

    ONE-TIME channel pre-wetting (this is the only place EF5 discharge puts
    water inside the basin): Q [m3/s] on the routed network -> a channel
    stage via a rating approximation h ~ (n Q / (w sqrt(S)))^(3/5). With
    `block` = (sr, sc) solver-cell -> EF5-pixel index arrays (EF5InletInflow
    .block), each EF5 channel pixel is filled to a water LEVEL = lowest bed
    in the pixel + stage, so only the low cells (the channel) get water
    instead of every cell of the 3x3 footprint carrying the full stage
    uphill. Without `block` the stage is applied as a uniform depth (legacy,
    synthetic tests). SM enters the CREST-AI side (it conditions the runoff
    EF5 sends us) — kept here for provenance/diagnostics.
    """
    q = torch.as_tensor(np.asarray(q_grid, dtype=float))
    if bankfull_width_fn is None:
        bankfull_width_fn = lambda qq: torch.clamp(7.2 * qq ** 0.5, min=1.0)  # Leopold-Maddock-ish
    chan = q >= q_min
    w = bankfull_width_fn(torch.clamp(q, min=0.0))
    n_ch, slope = 0.035, 1e-3
    stage = torch.minimum((n_ch * torch.clamp(q, min=0.0) / (w * slope ** 0.5)) ** 0.6,
                          torch.as_tensor(5.0))
    if block is None or dem is None:
        h0 = torch.where(chan, stage, torch.zeros_like(stage))
        return h0, torch.zeros_like(h0), torch.zeros_like(h0)
    z = np.asarray(dem.detach().cpu().numpy() if torch.is_tensor(dem) else dem,
                   dtype=float)
    sr, sc = block
    chan_np = chan.numpy()
    pid = np.where(chan_np & (sr >= 0), sr.astype(np.int64) * (int(sc.max()) + 2)
                   + sc.astype(np.int64), -1)
    h0 = np.zeros_like(z)
    if (pid >= 0).any():
        ids, inv = np.unique(pid[pid >= 0], return_inverse=True)
        zmin = np.full(ids.size, np.inf)
        np.minimum.at(zmin, inv, z[pid >= 0])
        level = zmin[inv] + stage.numpy()[pid >= 0]
        h0[pid >= 0] = np.clip(level - z[pid >= 0], 0.0, None)
    h0 = torch.as_tensor(h0, dtype=q.dtype)
    return h0, torch.zeros_like(h0), torch.zeros_like(h0)
