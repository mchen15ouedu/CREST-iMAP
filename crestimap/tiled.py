"""HUC12-partitioned parallel solver — basins as tiles, ghost cells only where
units touch.

User directive 2026-08-18: the whole contributing basin is simulated; when
it is too big for one device, the HYDROLOGIC UNITS are the decomposition —
each device owns a contiguous group of HUC12 units. Because unit boundaries
are ridgelines except where the channel crosses, the halo that actually
carries water is a handful of cells per neighbouring pair; the exchange is
an index_copy of those cells, not a plane of the raster.

Layout
------
labels (ny, nx) int16 : k >= 0 unit index, -1 outside the basin
groups                : units -> G tiles (contiguous runs along the domain's
                        long axis, balanced by owned-cell count; G = number
                        of devices unless n_tiles is given)
tile g                : rect [r0:r1, c0:c1] = bounds of its OWNED cells + a
                        `halo`-cell margin (clipped to the raster);
                        owned  = labels in group g
                        ghost  = other tiles' basin cells within Chebyshev
                                 distance <= halo of an owned cell
                        active = owned | ghost   (rest of the rect is a sink,
                                 exactly like SWESolver(active=...))
Step
----
SSP-RK2 with a halo exchange after EACH stage (halo = 2 covers the MUSCL
+ hydrostatic-reconstruction stencil of one stage). Owned cells therefore
see exactly the values the monolithic masked solve would give them: the
tiled result is bit-identical to SWESolver(active=mask) on the union grid
(tests/test_tiled.py). Global CFL dt = min over tiles.

Devices
-------
One process, any mix of "cuda:i"/"cpu"; tensors live on their tile's
device, halos move with .to(). Multi-node (torch.distributed) is a thin
wrapper on the same tile/halo tables — HPC's follow-up.
"""
from __future__ import annotations

import numpy as np
import torch

from .solver import SWESolver, desing_velocity


# --------------------------------------------------------------------------- #
# partition
# --------------------------------------------------------------------------- #
def plan_groups(labels: np.ndarray, n_tiles: int) -> list[list[int]]:
    """Units -> n_tiles contiguous groups: order units by centroid along the
    domain's long axis, cut into runs balanced by cell count."""
    act = labels >= 0
    units, counts = np.unique(labels[act], return_counts=True)
    units = [int(u) for u in units]
    if n_tiles <= 1 or len(units) <= 1:
        return [units]
    n_tiles = min(n_tiles, len(units))
    rows, cols = np.nonzero(act)
    ny_span = rows.max() - rows.min() + 1
    nx_span = cols.max() - cols.min() + 1
    axis_vals = cols if nx_span >= ny_span else rows
    lab_flat = labels[rows, cols]
    cent = {u: float(axis_vals[lab_flat == u].mean()) for u in units}
    order = sorted(units, key=lambda u: cent[u])
    cnt = {int(u): int(c) for u, c in zip(units, counts)}
    total = sum(cnt.values())
    target = total / n_tiles
    groups, cur, acc = [], [], 0
    for u in order:
        cur.append(u)
        acc += cnt[u]
        if acc >= target * (len(groups) + 1) - 0.5 * cnt[u] and \
                len(groups) < n_tiles - 1:
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    """Chebyshev dilation by r cells (numpy only)."""
    out = mask.copy()
    ny, nx = mask.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy == 0 and dx == 0:
                continue
            src = mask[max(0, -dy):ny - max(0, dy), max(0, -dx):nx - max(0, dx)]
            out[max(0, dy):ny - max(0, -dy), max(0, dx):nx - max(0, -dx)] |= src
    return out


class Tile:
    __slots__ = ("g", "device", "r0", "r1", "c0", "c1", "owned", "active",
                 "solver", "h", "qx", "qy", "md", "h1", "qx1", "qy1",
                 "own_idx", "recv")

    def __init__(self, g, device):
        self.g = g
        self.device = device
        self.recv = []          # [(src_tile_index, idx_here, idx_there)]


class TiledSolver:
    def __init__(self, z, labels, dx, dy, devices, n_manning=0.05, n_tiles=None,
                 halo=2, order=2, bc="open", eps=1e-6, cfl=0.45, dt_max=60.0,
                 dtype=torch.float32, log=lambda s: None):
        """z: (ny, nx) tensor/array (CPU); labels: (ny, nx) int16 numpy."""
        z = torch.as_tensor(np.asarray(z), dtype=dtype)
        self.ny, self.nx = z.shape
        self.z_cpu = z
        self.labels = np.asarray(labels, dtype=np.int16)
        assert self.labels.shape == tuple(z.shape)
        self.dx, self.dy = float(dx), float(dy)
        self.devices = list(devices) or ["cpu"]
        self.halo = int(halo)
        self.dtype = dtype
        self.dt_max = dt_max
        self.eps = eps
        self.cfl = cfl
        self.log = log
        n_tiles = n_tiles or len(self.devices)
        self.groups = plan_groups(self.labels, n_tiles)
        act = self.labels >= 0
        self.active_global = act
        # ---- tiles ------------------------------------------------------- #
        self.tiles: list[Tile] = []
        owner = np.full(self.labels.shape, -1, dtype=np.int16)   # tile index
        for gi, units in enumerate(self.groups):
            t = Tile(gi, self.devices[gi % len(self.devices)])
            own = np.isin(self.labels, units) & act
            owner[own] = gi
            rows, cols = np.nonzero(own)
            t.r0 = max(0, rows.min() - self.halo)
            t.r1 = min(self.ny, rows.max() + 1 + self.halo)
            t.c0 = max(0, cols.min() - self.halo)
            t.c1 = min(self.nx, cols.max() + 1 + self.halo)
            t.owned = own[t.r0:t.r1, t.c0:t.c1]
            self.tiles.append(t)
        self.owner = owner
        n_own = 0
        for t in self.tiles:
            own = t.owned
            near = _dilate(own, self.halo)
            ghost = near & ~own & act[t.r0:t.r1, t.c0:t.c1]
            t.active = own | ghost
            n_own += int(own.sum())
            zt = self.z_cpu[t.r0:t.r1, t.c0:t.c1].to(t.device)
            t.solver = SWESolver(zt, dx=self.dx, dy=self.dy, n_manning=n_manning,
                                 order=order, bc=bc, eps=eps, cfl=cfl,
                                 dt_max=dt_max,
                                 active=torch.as_tensor(t.active, device=t.device))
            t.own_idx = torch.as_tensor(np.flatnonzero(own), device=t.device)
            # ghost sources: owner tile of each ghost cell
            gr, gc = np.nonzero(ghost)
            src_tile = owner[gr + t.r0, gc + t.c0]
            for si in np.unique(src_tile):
                si = int(si)
                if si < 0:
                    continue
                sel = src_tile == si
                # global (row, col) for now; flat local indices are built in
                # the second pass once every tile's rect is known
                t.recv.append([si, gr[sel] + t.r0, gc[sel] + t.c0])
        # second pass: turn global (row, col) of ghosts into flat local
        # indices here and in the source tile
        for t in self.tiles:
            recv = []
            for si, gr, gc in t.recv:
                s = self.tiles[si]
                idx_here = (gr - t.r0) * (t.c1 - t.c0) + (gc - t.c0)
                idx_there = (gr - s.r0) * (s.c1 - s.c0) + (gc - s.c0)
                recv.append((si,
                             torch.as_tensor(idx_here, device=t.device),
                             torch.as_tensor(idx_there, device=s.device)))
            t.recv = recv
        n_rect = sum((t.r1 - t.r0) * (t.c1 - t.c0) for t in self.tiles)
        n_ghost = sum(int(len(r[1])) for t in self.tiles for r in t.recv)
        self.log(f"tiled solver: {len(self.tiles)} tile(s) on "
                 f"{len(self.devices)} device(s); owned {n_own / 1e3:.0f}k cells, "
                 f"tile rects {n_rect / 1e3:.0f}k (x{n_rect / max(1, n_own):.2f}), "
                 f"ghost cells {n_ghost} (halo {self.halo}); units per tile "
                 f"{[len(g) for g in self.groups]}")

    # ------------------------------------------------------------------ #
    # scatter / gather
    # ------------------------------------------------------------------ #
    def scatter(self, h, qx, qy, md=None):
        h = torch.as_tensor(h, dtype=self.dtype)
        qx = torch.as_tensor(qx, dtype=self.dtype)
        qy = torch.as_tensor(qy, dtype=self.dtype)
        md = h.clone() if md is None else torch.as_tensor(md, dtype=self.dtype)
        for t in self.tiles:
            sl = (slice(t.r0, t.r1), slice(t.c0, t.c1))
            t.h = h[sl].to(t.device).clone()
            t.qx = qx[sl].to(t.device).clone()
            t.qy = qy[sl].to(t.device).clone()
            t.md = md[sl].to(t.device).clone()
            t.h, t.qx, t.qy = t.solver.mask_state(t.h, t.qx, t.qy)

    def _gather(self, attr):
        out = torch.zeros((self.ny, self.nx), dtype=self.dtype)
        for t in self.tiles:
            loc = getattr(t, attr).detach()
            own = torch.as_tensor(t.owned, device=loc.device)
            piece = torch.where(own, loc, torch.zeros_like(loc)).cpu()
            out[t.r0:t.r1, t.c0:t.c1] += piece
        return out

    def gather(self):
        return self._gather("h"), self._gather("qx"), self._gather("qy")

    def gather_h(self):
        return self._gather("h")

    def gather_maxdepth(self):
        return self._gather("md")

    # ------------------------------------------------------------------ #
    # halo exchange + stepping
    # ------------------------------------------------------------------ #
    def _exchange(self, names=("h", "qx", "qy")):
        for t in self.tiles:
            for si, idx_here, idx_there in t.recv:
                s = self.tiles[si]
                for nm in names:
                    src = getattr(s, nm).reshape(-1).index_select(0, idx_there)
                    dst = getattr(t, nm).reshape(-1)
                    dst.index_copy_(0, idx_here, src.to(t.device))

    def compute_dt(self):
        dt = self.dt_max
        for t in self.tiles:
            dt = min(dt, t.solver.compute_dt(t.h, t.qx, t.qy))
        return dt

    def _slice_fn(self, fn):
        """Global-tensor callable -> per-tile slice callables with an
        identity cache (piecewise-constant forcing = same object per hour)."""
        if fn is None:
            return [None] * len(self.tiles)
        cache = {}

        def make(t):
            def f(tt):
                g = fn(tt)
                key = (id(g), t.g)
                if key not in cache:
                    cache.clear() if len(cache) > 4 * len(self.tiles) else None
                    cache[key] = g[t.r0:t.r1, t.c0:t.c1].to(t.device)
                return cache[key]
            return f
        return [make(t) for t in self.tiles]

    def step(self, dt, rains):
        """One SSP-RK2 step on every tile with a halo exchange per stage.
        rains: per-tile zero-arg callables returning that tile's lateral
        inflow slice (or None)."""
        # stage 1
        for t, rf in zip(self.tiles, rains):
            rain = rf() if rf is not None else None
            r1 = t.solver._rhs(t.h, t.qx, t.qy, rain)
            t.h1 = torch.clamp(t.h + dt * r1[0], min=0.0)   # noqa (dynamic attrs)
            t.qx1 = t.qx + dt * r1[1]
            t.qy1 = t.qy + dt * r1[2]
        self._exchange(("h1", "qx1", "qy1"))
        # stage 2
        for t, rf in zip(self.tiles, rains):
            rain = rf() if rf is not None else None
            r2 = t.solver._rhs(t.h1, t.qx1, t.qy1, rain)
            hn = torch.clamp(0.5 * t.h + 0.5 * (t.h1 + dt * r2[0]), min=0.0)
            qxn = 0.5 * t.qx + 0.5 * (t.qx1 + dt * r2[1])
            qyn = 0.5 * t.qy + 0.5 * (t.qy1 + dt * r2[2])
            qxn, qyn = t.solver._friction(hn, qxn, qyn, dt)
            dry = hn <= self.eps
            qxn = torch.where(dry, torch.zeros_like(qxn), qxn)
            qyn = torch.where(dry, torch.zeros_like(qyn), qyn)
            t.h, t.qx, t.qy = t.solver.mask_state(hn, qxn, qyn)
        self._exchange(("h", "qx", "qy"))

    def run(self, t_end, rain_fn=None, t0=0.0, callback=None, nudge_fn=None,
            dt_every=1, track_max=True):
        """Integrate the scattered state to t_end (same contract as
        SWESolver.run; callback(t, self) — use gather_h()/gather_maxdepth()
        lazily, e.g. only when a frame is due)."""
        t = t0
        nstep = 0
        held_dt = None
        next_change = getattr(rain_fn, "next_change", None)
        rain_slices = self._slice_fn(rain_fn)
        # nudge (EF5ChannelStage) is (t, h) -> max(h, stage(t)) with a global
        # stage tensor; apply it per tile on the sliced stage
        stage_cache = {}

        def nudge_tile(tt, tile):
            st = nudge_fn._stage(nudge_fn._index(tt))
            key = (id(st), tile.g)
            if key not in stage_cache:
                if len(stage_cache) > 4 * len(self.tiles):
                    stage_cache.clear()
                stage_cache[key] = st[tile.r0:tile.r1, tile.c0:tile.c1].to(tile.device)
            return torch.maximum(tile.h, stage_cache[key])

        while t < t_end - 1e-12:
            rains = [(lambda f=f, tt=t: f(tt)) if f is not None else None
                     for f in rain_slices]
            if dt_every <= 1 or held_dt is None or nstep % dt_every == 0:
                dt = self.compute_dt()
                held_dt = 0.9 * dt
            else:
                dt = held_dt
            if t + dt > t_end:
                dt = t_end - t
            if next_change is not None:
                nc = next_change(t)
                if 1e-9 < nc < dt:
                    dt = nc
            self.step(dt, rains)
            t += dt
            nstep += 1
            if nudge_fn is not None:
                for tile in self.tiles:
                    tile.h = nudge_tile(t, tile)
                    tile.h, tile.qx, tile.qy = tile.solver.mask_state(
                        tile.h, tile.qx, tile.qy)
                self._exchange(("h",))
            if track_max:
                for tile in self.tiles:
                    tile.md = torch.maximum(tile.md, tile.h)
            if callback is not None:
                callback(t, self)
        return self.gather()
