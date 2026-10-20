"""Flow network from a DEM — D8 directions, flow accumulation and slope.

Used by the subsurface (interflow) router and by the DEM burn screen. The
SWE solver never sees any of this: it routes on the DEM as given.

* Depressions are resolved with Priority-Flood (Barnes et al. 2014): cells
  are flooded inward from the domain border (grid edge, or the edge of the
  basin mask) in elevation order, so every active cell gets a path out.
  The filled surface `zf` is used for directions only.
* D8: steepest descent on `zf` (distance-weighted, dx != dy allowed). A
  cell with no strictly lower active neighbour (a flat or a filled pit)
  follows the flood tree instead, i.e. back along the cell that flooded it.
  Steepest edges strictly descend and flood edges only join equal `zf`, so
  the network is acyclic.
* Border cells with no lower active neighbour drain OUT of the domain
  (downstream index -1): the basin outlet, and nothing else on a basin.

Pure numpy; the flood loop is jitted with numba when it is installed.
"""
from __future__ import annotations

import dataclasses
import heapq
import math

import numpy as np

# D8 neighbour offsets, clockwise from east
DI = (0, 1, 1, 1, 0, -1, -1, -1)
DJ = (1, 1, 0, -1, -1, -1, 0, 1)


def _flood_py(z, active, seed, ny, nx):
    """Priority-Flood over flattened arrays. Returns (zf, from_index)."""
    n = ny * nx
    zf = z.copy()
    done = np.zeros(n, np.bool_)
    frm = np.full(n, -1, np.int64)
    heap = [(0.0, 0, 0)]
    heap.pop()
    count = 0
    for c in range(n):
        if seed[c]:
            heapq.heappush(heap, (zf[c], count, c))
            count += 1
            done[c] = True
    while heap:
        zc, _, c = heapq.heappop(heap)
        ci, cj = c // nx, c % nx
        for k in range(8):
            ni, nj = ci + DI[k], cj + DJ[k]
            if ni < 0 or ni >= ny or nj < 0 or nj >= nx:
                continue
            m = ni * nx + nj
            if done[m] or not active[m]:
                continue
            done[m] = True
            if zf[m] < zc:
                zf[m] = zc
            frm[m] = c
            # FIFO among equal elevations: flats drain along shortest paths
            heapq.heappush(heap, (zf[m], count, m))
            count += 1
    return zf, frm


try:  # pragma: no cover - optional speed-up
    import numba as _nb
    _flood = _nb.njit(cache=True)(_flood_py)
except Exception:  # numba missing or cannot compile
    _flood = _flood_py


@dataclasses.dataclass
class FlowNet:
    """D8 network on an (ny, nx) grid, flattened row-major.

    ds      : downstream flat index, -1 = leaves the domain (or inactive)
    length  : distance to the downstream cell [m]
    slope   : drop on the filled DEM / length, floored at slope_min [-]
    zf      : filled DEM (directions only) (ny, nx)
    active  : bool (ny, nx)
    order   : active flat indices, every cell before its downstream cell
    batches : offsets into `order` of mutually independent batches
    acc_km2 : upstream area including the cell [km2] (ny, nx)
    """
    shape: tuple
    dx: float
    dy: float
    ds: np.ndarray
    length: np.ndarray
    slope: np.ndarray
    zf: np.ndarray
    active: np.ndarray
    order: np.ndarray
    batches: np.ndarray
    acc_km2: np.ndarray

    def accumulate(self, weight):
        """Sum `weight` (ny, nx) over each cell's upstream area (incl. itself)."""
        acc = np.where(self.active, np.asarray(weight, float), 0.0).ravel()
        return _accumulate(acc, self.ds, self.order, self.batches).reshape(self.shape)


def _topo_order(ds, active_flat):
    """Kahn's algorithm in numpy batches. Returns (order, batch offsets):
    every cell comes after all of its upstream cells, and the cells of one
    batch are never upstream of each other."""
    n = ds.size
    indeg = np.bincount(ds[ds >= 0], minlength=n)
    frontier = np.flatnonzero(active_flat & (indeg == 0))
    out, offs = [], [0]
    while frontier.size:
        out.append(frontier)
        offs.append(offs[-1] + frontier.size)
        nxt = ds[frontier]
        nxt = nxt[nxt >= 0]
        np.subtract.at(indeg, nxt, 1)
        cand = np.unique(nxt)
        frontier = cand[indeg[cand] == 0]
    order = np.concatenate(out) if out else np.zeros(0, np.int64)
    if order.size != int(active_flat.sum()):
        raise RuntimeError("flow network has a cycle - this is a bug")
    return order, np.asarray(offs, np.int64)


def _accumulate(acc, ds, order, batches):
    """Add each cell's value into its downstream cell, batch by batch."""
    acc = acc.copy()
    for b in range(batches.size - 1):
        cells = order[batches[b]:batches[b + 1]]
        d = ds[cells]
        keep = d >= 0
        np.add.at(acc, d[keep], acc[cells[keep]])
    return acc


def flow_network(z, dx, dy, active=None, slope_min=1e-3,
                 no_outlet=None) -> FlowNet:
    """Build the D8 network of DEM `z` (ny, nx) with cell sizes dx, dy [m].

    active    : optional bool (ny, nx) basin mask; cells outside are ignored
                and the mask edge is the domain border.
    slope_min : floor on the D8 slope used for interflow travel times. EF5
                floors the DROP at 1 m per cell instead, which at 1"
                (~30 m) cells means a 3.3 % minimum slope.
    no_outlet : optional bool (ny, nx): border cells that may NOT drain out
                of the domain (e.g. river inflow points); they route inward.
    """
    z = np.asarray(z, dtype=np.float64)
    ny, nx = z.shape
    act = np.ones((ny, nx), bool) if active is None else np.asarray(active, bool)
    if not act.any():
        raise ValueError("active mask is empty")
    # seeds: active cells on the grid edge or next to an inactive cell
    pad = np.pad(act, 1, constant_values=False)
    inner = np.ones_like(act)
    for k in range(8):
        inner &= pad[1 + DI[k]:1 + DI[k] + ny, 1 + DJ[k]:1 + DJ[k] + nx]
    seed = act & ~inner
    if no_outlet is not None:
        seed &= ~np.asarray(no_outlet, bool)
        if not seed.any():
            raise ValueError("no_outlet leaves the domain without an outlet")
    zfl, frm = _flood(z.ravel().copy(), act.ravel(), seed.ravel(), ny, nx)
    zf = zfl.reshape(ny, nx)

    # steepest descent on zf among active neighbours
    diag = math.hypot(dx, dy)
    dist = [dx if DI[k] == 0 else dy if DJ[k] == 0 else diag for k in range(8)]
    zp = np.pad(zf, 1, constant_values=np.nan)
    ap = np.pad(act, 1, constant_values=False)
    best = np.zeros((ny, nx))
    best_k = np.full((ny, nx), -1)
    for k in range(8):
        zn = zp[1 + DI[k]:1 + DI[k] + ny, 1 + DJ[k]:1 + DJ[k] + nx]
        an = ap[1 + DI[k]:1 + DI[k] + ny, 1 + DJ[k]:1 + DJ[k] + nx]
        g = np.where(an, (zf - zn) / dist[k], -np.inf)
        upd = g > best
        best = np.where(upd, g, best)
        best_k = np.where(upd, k, best_k)
    ii, jj = np.mgrid[0:ny, 0:nx]
    kk = np.clip(best_k, 0, 7)
    di = np.asarray(DI)[kk]
    dj = np.asarray(DJ)[kk]
    steep = ((ii + di) * nx + (jj + dj)).ravel()
    has = (best_k >= 0).ravel() & act.ravel()
    ds = np.where(has, steep, frm)
    ds = np.where(act.ravel(), ds, -1).astype(np.int64)

    # length / slope to the downstream cell
    flat = np.arange(ny * nx)
    tgt = np.where(ds >= 0, ds, flat)
    ddi = np.abs(tgt // nx - flat // nx)
    ddj = np.abs(tgt % nx - flat % nx)
    length = np.where((ddi == 1) & (ddj == 1), diag,
                      np.where(ddi == 1, dy, dx)).astype(float)
    drop = np.where(ds >= 0, zf.ravel() - zf.ravel()[tgt], 0.0)
    slope = np.maximum(drop / length, slope_min)

    order, batches = _topo_order(ds, act.ravel())
    cell_km2 = dx * dy / 1e6
    acc = _accumulate(np.where(act, cell_km2, 0.0).ravel(), ds, order, batches)
    return FlowNet((ny, nx), float(dx), float(dy), ds, length, slope, zf, act,
                   order, batches, acc.reshape(ny, nx))
