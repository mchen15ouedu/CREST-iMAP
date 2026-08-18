"""Basin-shaped solver domains (hydrologic units, never rectangles).

The CREST-AI side (hf_data/hucdomain.py) builds the event domain as the
union of WBD HUC12 units contributing to the trigger gauge and ships it in
the forcing bundle as `domain_huc.tif` — an int16 label raster on the 3DEP
lattice: k >= 0 = index of the unit, -1 = outside the basin. This module
puts it on the solver grid:

  active : bool (ny, nx)  — cells the solver integrates; everything else is
           a sink (SWESolver(active=...)): water leaving the basin through
           the pour point vanishes = free outflow, nothing crosses a divide.
  labels : int16 (ny, nx) — the tiling units for the parallel tier
           (crestimap.tiled): one HUC12 (or a contiguous group) per device.

Only rasterio + numpy + torch are needed here (the ZeroGPU worker image has
no shapely/scipy); all geometry work happens on the CREST-AI side.
"""
from __future__ import annotations

import os

import numpy as np

DOMAIN_FILE = "domain_huc.tif"
GEOJSON_FILE = "domain.geojson"


def find_domain(ef5_output_dir: str, explicit: str | None = None) -> str | None:
    """Path of the domain label raster for a bundle, or None."""
    if explicit:
        return explicit if os.path.exists(explicit) else None
    p = os.path.join(ef5_output_dir, DOMAIN_FILE)
    return p if os.path.exists(p) else None


def load_labels(path: str, grid):
    """Label raster -> int16 (ny, nx) on the solver grid (nearest cell
    centre; solver cells whose centre falls outside the raster are -1)."""
    import rasterio
    with rasterio.open(path) as ds:
        src = ds.read(1).astype(np.int16)
        tr = ds.transform
        nod = ds.nodata
    if nod is not None:
        src = np.where(src == nod, -1, src).astype(np.int16)
    ny, nx = grid.z.shape
    rows, cols = np.mgrid[0:ny, 0:nx]
    xs, ys = grid.transform * (cols + 0.5, rows + 0.5)
    sc, sr = ~tr * (xs, ys)
    sr = np.floor(sr).astype(int)
    sc = np.floor(sc).astype(int)
    inside = (sr >= 0) & (sr < src.shape[0]) & (sc >= 0) & (sc < src.shape[1])
    out = np.full((ny, nx), -1, dtype=np.int16)
    out[inside] = src[sr[inside], sc[inside]]
    return out


def load_domain(path: str, grid, device=None):
    """(active torch.bool tensor on `device`, labels int16 numpy, info dict)."""
    import torch
    labels = load_labels(path, grid)
    act_np = labels >= 0
    n_active = int(act_np.sum())
    if n_active == 0:
        raise ValueError(f"{os.path.basename(path)} covers no solver cell — "
                         f"domain/DEM grids disagree")
    active = torch.as_tensor(act_np, device=device or grid.z.device)
    units = sorted(int(u) for u in np.unique(labels[act_np]))
    info = {"n_active": n_active, "n_bbox": int(labels.size),
            "n_units": len(units),
            "active_frac": round(n_active / labels.size, 4)}
    return active, labels, info


def unit_cells(labels: np.ndarray) -> dict:
    """{unit k: cell count} for the tiling planner."""
    act = labels >= 0
    ks, ns = np.unique(labels[act], return_counts=True)
    return {int(k): int(n) for k, n in zip(ks, ns)}
