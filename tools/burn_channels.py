"""Burn a river network into a DEM for CREST-iMAP (AGREE-style, with
physically sized channels).

    python tools/burn_channels.py --dem dem.tif --flowlines NHDPlus_H_0801_HU4_GDB.gdb \
        --layer NHDFlowline --vaa-layer NHDPlusFlowlineVAA --out dem_burned.tif

    python tools/burn_channels.py --dem dem.tif --merit-dir dir.tif --merit-upa upa.tif \
        --out dem_burned.tif

Writes <out> (burned DEM) and <out stem>_channels.tif (channel mask, the
cells where CREST-iMAP injects routed interflow), both on the DEM's grid,
tagged CRESTIMAP_BURN with the source and parameters.

Why burn: a bare-earth DEM carries road embankments, bridge decks and
culvert crossings across every river. The 2-D solver routes on the DEM as
given, so each one is a dam. See the README, "Burn the channels first".

Method (Hellweger 1997 AGREE, adapted for hydraulics). AGREE's usual
settings (e.g. ArcHydro: 1000 m sharp drop) only serve to force flow
directions; here the burned DEM IS the hydraulic terrain, so:

1. Rasterize the network onto the DEM grid (line cells carry upstream area
   A [km2]: from the flowline attribute, MERIT upa, or the DEM itself).
2. Line elevation = 3x3 minimum of the DEM (finds the thalweg next to a
   slightly offset line), then forced to decrease downstream (1 mm per
   cell minimum) — this cuts through every crossing.
3. Bankfull geometry from upstream area (Bieger et al. 2015, CONUS):
       width  W = 2.70 A^0.352 [m],  depth D = 0.30 A^0.213 [m]
   bed = line elevation - D, again forced to decrease downstream.
4. Cells within W/2 of the line take the bed elevation of their nearest
   line cell (a rectangular channel, the "sharp drop"); cells within a
   further `buffer` cells ramp linearly back to the DEM (AGREE's smooth
   drop). Terrain is only ever lowered.

Data are NOT downloaded: see the README / user manual for where to get
NHDPlus HR (US) and MERIT Hydro (global; registration required).

Requires: numpy, scipy, rasterio, geopandas (vector input).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import sys

import numpy as np



def _load_hydronet():
    """crestimap/hydronet.py on its own: numpy only. Importing the crestimap
    package would load torch, which the tool does not need (and whose
    bundled OpenMP runtime clashes with MKL numpy in some conda envs)."""
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "crestimap", "hydronet.py")
    spec = importlib.util.spec_from_file_location("crestimap_hydronet", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod            # dataclasses look it up
    spec.loader.exec_module(mod)
    return mod


hydronet = _load_hydronet()

BURN_TAG = "CRESTIMAP_BURN"
_M_PER_DEG_LAT = 111_320.0
EPS_DROP = 1e-3            # minimum bed drop per cell [m]

# MERIT Hydro dir.tif encoding (ESRI-style D8) -> (di, dj)
_MERIT_D8 = {1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1),
             16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1)}


def cell_size_m(transform, crs, height):
    """(dx, dy) in metres (geographic CRS: at the grid's centre latitude)."""
    if crs is None or crs.is_geographic:
        lat = transform.f + transform.e * height / 2.0
        dy = abs(transform.e) * _M_PER_DEG_LAT
        dx = abs(transform.a) * _M_PER_DEG_LAT * math.cos(math.radians(lat))
        return dx, dy
    return abs(transform.a), abs(transform.e)


# ----------------------------------------------------------------- network
def lines_from_vector(path, dem_crs, bounds, layer=None, area_field=None,
                      vaa_layer=None, join_key="NHDPlusID", query=None):
    """[(geometry dict, area_km2 or nan)] from a vector file, clipped to the
    DEM bounds and reprojected to the DEM CRS."""
    import geopandas as gpd
    from shapely.geometry import box, mapping
    gdf = gpd.read_file(path, layer=layer)
    if gdf.crs is None:
        raise ValueError(f"{path}: flowlines have no CRS")
    clip = gpd.GeoSeries([box(*bounds)], crs=dem_crs).to_crs(gdf.crs)
    gdf = gdf[gdf.intersects(clip.iloc[0])]
    if query:                               # e.g. "FType != 566" (coastline)
        gdf = gdf.query(query)
    if vaa_layer:
        import pandas as pd
        vaa = gpd.read_file(path, layer=vaa_layer, ignore_geometry=True)
        cols = {c.lower(): c for c in vaa.columns}
        key = cols.get(join_key.lower())
        fld = cols.get((area_field or "TotDASqKm").lower())
        if key is None or fld is None:
            raise ValueError(f"{vaa_layer}: needs {join_key} and "
                             f"{area_field or 'TotDASqKm'}")
        gkey = {c.lower(): c for c in gdf.columns}[join_key.lower()]
        gdf = gdf.merge(pd.DataFrame({gkey: vaa[key], "_area": vaa[fld]}),
                        on=gkey, how="left")
        area_field = "_area"
    gdf = gdf.to_crs(dem_crs)
    if area_field:
        cols = {c.lower(): c for c in gdf.columns}
        if area_field.lower() not in cols:
            raise ValueError(f"no field {area_field!r} in {path} "
                             f"(have {list(gdf.columns)})")
        area = gdf[cols[area_field.lower()]].astype(float).to_numpy()
    else:
        area = np.full(len(gdf), np.nan)
    out = []
    for geom, a in zip(gdf.geometry, area):
        if geom is not None and not geom.is_empty:
            out.append((mapping(geom), a))
    return out


def lines_from_merit(dir_path, upa_path, dem_crs, bounds, threshold_km2):
    """MERIT Hydro: one segment per stream pixel (upa >= threshold), from
    its centre to its downstream pixel's centre, carrying its upa."""
    import rasterio
    from rasterio.warp import transform_bounds, transform_geom
    from rasterio.windows import from_bounds
    with rasterio.open(upa_path) as su, rasterio.open(dir_path) as sd:
        if su.transform != sd.transform or su.shape != sd.shape:
            raise ValueError("MERIT dir and upa tiles must be on one grid")
        b = transform_bounds(dem_crs, su.crs, *bounds, densify_pts=21)
        win = from_bounds(*b, transform=su.transform).round_offsets().round_lengths()
        win = win.intersection(rasterio.windows.Window(0, 0, su.width, su.height))
        pad = 2
        win = rasterio.windows.Window(max(win.col_off - pad, 0),
                                      max(win.row_off - pad, 0),
                                      win.width + 2 * pad, win.height + 2 * pad)
        upa = su.read(1, window=win, boundless=True, fill_value=0).astype(float)
        d8 = sd.read(1, window=win, boundless=True, fill_value=0)
        tr = su.window_transform(win)
        src_crs = su.crs
    ii, jj = np.nonzero(upa >= threshold_km2)
    out = []
    for i, j in zip(ii, jj):
        step = _MERIT_D8.get(int(d8[i, j]))
        if step is None:
            continue
        i2, j2 = i + step[0], j + step[1]
        x1, y1 = tr * (j + 0.5, i + 0.5)
        x2, y2 = tr * (j2 + 0.5, i2 + 0.5)
        g = {"type": "LineString", "coordinates": [(x1, y1), (x2, y2)]}
        out.append((transform_geom(src_crs, dem_crs, g), upa[i, j]))
    return out


# ----------------------------------------------------------------- burning
def rasterize_lines(lines, shape, transform):
    """(stream bool, area km2 with nan where unknown); larger rivers win."""
    from rasterio.features import rasterize
    if not lines:
        raise ValueError("no flowlines intersect the DEM")
    a = np.array([x[1] for x in lines], float)
    known = np.isfinite(a)
    order = np.argsort(np.where(known, a, -1.0))
    shapes = [(lines[k][0], (a[k] if known[k] else -1.0)) for k in order]
    val = rasterize(shapes, out_shape=shape, transform=transform, fill=-2.0,
                    dtype="float64", all_touched=False)
    stream = val > -2.0
    area = np.where(val >= 0.0, val, np.nan)
    return stream, area


def _enforce_descent(zline, stream, net):
    """Make zline strictly decrease (EPS_DROP) downstream along stream cells."""
    z = zline.ravel().copy()
    s = stream.ravel()
    for b in range(net.batches.size - 1):
        cells = net.order[net.batches[b]:net.batches[b + 1]]
        cells = cells[s[cells]]
        d = net.ds[cells]
        keep = d >= 0
        cells, d = cells[keep], d[keep]
        keep = s[d]
        np.minimum.at(z, d[keep], z[cells[keep]] - EPS_DROP)
    return z.reshape(zline.shape)


def burn(z, dx, dy, stream, area_km2, active=None, depth_a=0.30, depth_b=0.213,
         width_a=2.70, width_b=0.352, buffer_cells=3, min_area_km2=1.0):
    """Burn the rasterized network into z. Returns (burned z, channel mask,
    stats dict)."""
    from scipy import ndimage
    z = np.asarray(z, float)
    act = np.ones(z.shape, bool) if active is None else np.asarray(active, bool)
    stream = stream & act
    # provisional flow directions that follow the rasterized lines. Only the
    # network's outlet may drain out of the domain: other stream cells on
    # the edge are inflows, and a reach dammed by a crossing next to an
    # inflow edge must not drain backwards out of it.
    zt = np.where(stream, z - 1e4, z)
    edge = act & ~ndimage.binary_erosion(act, structure=np.ones((3, 3)),
                                         border_value=0)
    se = stream & edge
    no_out = None
    if se.sum() > 1:
        rank = np.where(np.isfinite(area_km2), area_km2, -np.inf)
        if np.isfinite(rank[se]).any():
            key = np.where(se, rank, -np.inf)
        else:                                   # no areas: the lowest one
            key = np.where(se, -z, -np.inf)
        outlet = np.unravel_index(int(np.argmax(key)), z.shape)
        no_out = se.copy()
        no_out[outlet] = False
    net = hydronet.flow_network(zt, dx, dy, active=act, no_outlet=no_out)
    dem_area = net.acc_km2
    # the network's own upstream area where it has one (it sees beyond the
    # DEM), else what accumulates on the DEM along the burned lines
    area = np.where(stream & np.isfinite(area_km2), area_km2, dem_area)
    area = np.where(stream, area, 0.0)
    stream = stream & (area >= min_area_km2)
    if not stream.any():
        raise ValueError(f"no stream cells with upstream area >= {min_area_km2} km2")
    # 1. line elevation: local thalweg, forced downhill
    zmin = ndimage.minimum_filter(np.where(act, z, np.inf), size=3)
    zline = _enforce_descent(np.where(stream, zmin, z), stream, net)
    # 2. bankfull geometry
    depth = np.where(stream, depth_a * np.power(np.maximum(area, 1e-6), depth_b), 0.0)
    width = np.where(stream, width_a * np.power(np.maximum(area, 1e-6), width_b), 0.0)
    bed = _enforce_descent(np.where(stream, zline - depth, z), stream, net)
    # 3. distance to the nearest line cell and its geometry
    dist, (ni, nj) = ndimage.distance_transform_edt(~stream, sampling=(dy, dx),
                                                    return_indices=True)
    bed_n = bed[ni, nj]
    half_w = 0.5 * width[ni, nj]
    cell = math.sqrt(dx * dy)
    ramp = max(buffer_cells, 0) * cell
    chan = act & (stream | (dist <= half_w))
    zb = np.where(chan, np.minimum(z, bed_n), z)
    if ramp > 0:
        frac = (dist - half_w) / ramp
        buf = act & ~chan & (frac <= 1.0)
        zr = bed_n + np.clip(frac, 0.0, 1.0) * (z - bed_n)
        zb = np.where(buf, np.minimum(z, zr), zb)
    stats = dict(line_cells=int(stream.sum()), channel_cells=int(chan.sum()),
                 max_area_km2=float(area.max()),
                 max_depth_m=float(depth.max()), max_width_m=float(width.max()),
                 lowered_cells=int((zb < z).sum()),
                 mean_cut_m=float((z - zb)[zb < z].mean()) if (zb < z).any() else 0.0)
    return zb, chan, stats


# ----------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Burn a river network into a DEM for CREST-iMAP.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--dem", required=True)
    ap.add_argument("--out", required=True, help="burned DEM GeoTIFF")
    ap.add_argument("--channels-out", help="channel mask GeoTIFF "
                    "(default <out stem>_channels.tif)")
    ap.add_argument("--mask", help="basin mask GeoTIFF on the DEM grid (1 = basin)")
    src = ap.add_argument_group("network (one of)")
    src.add_argument("--flowlines", help="vector file: NHDPlus HR .gdb/.gpkg/.shp, "
                     "MERIT-Basins, any lines")
    src.add_argument("--layer", help="layer in --flowlines (e.g. NHDFlowline)")
    src.add_argument("--area-field", help="upstream-area field [km2] "
                     "(e.g. TotDASqKm); default: derive from the DEM")
    src.add_argument("--query", help='pandas filter on the flowline fields, '
                     'e.g. "FType != 566" to drop NHD coastlines')
    src.add_argument("--vaa-layer", help="NHDPlus HR attribute table joined on "
                     "NHDPlusID (e.g. NHDPlusFlowlineVAA) for TotDASqKm")
    src.add_argument("--merit-dir", help="MERIT Hydro dir.tif tile(s) mosaic")
    src.add_argument("--merit-upa", help="MERIT Hydro upa.tif (km2), same grid")
    src.add_argument("--merit-threshold-km2", type=float, default=1.0,
                     help="MERIT stream definition (default 1 km2)")
    g = ap.add_argument_group("geometry")
    g.add_argument("--min-area-km2", type=float, default=1.0,
                   help="burn only streams draining at least this (default 1)")
    g.add_argument("--depth-a", type=float, default=0.30)
    g.add_argument("--depth-b", type=float, default=0.213)
    g.add_argument("--width-a", type=float, default=2.70)
    g.add_argument("--width-b", type=float, default=0.352)
    g.add_argument("--buffer-cells", type=int, default=3)
    a = ap.parse_args(argv)

    import rasterio
    with rasterio.open(a.dem) as ds:
        z = ds.read(1).astype(float)
        prof = ds.profile.copy()
        nodata = ds.nodata
        tr, crs, bounds = ds.transform, ds.crs, ds.bounds
    act = np.ones(z.shape, bool)
    if nodata is not None:
        act &= z != nodata
    act &= np.isfinite(z)
    if a.mask:
        with rasterio.open(a.mask) as ms:
            act &= ms.read(1) > 0
    dx, dy = cell_size_m(tr, crs, z.shape[0])
    if a.flowlines:
        lines = lines_from_vector(a.flowlines, crs, bounds, layer=a.layer,
                                  area_field=a.area_field, vaa_layer=a.vaa_layer,
                                  query=a.query)
        source = dict(kind="vector", path=os.path.basename(a.flowlines),
                      layer=a.layer, area_field=a.area_field, vaa_layer=a.vaa_layer,
                      query=a.query)
    elif a.merit_dir and a.merit_upa:
        lines = lines_from_merit(a.merit_dir, a.merit_upa, crs, bounds,
                                 a.merit_threshold_km2)
        source = dict(kind="merit", threshold_km2=a.merit_threshold_km2)
    else:
        ap.error("give --flowlines, or --merit-dir and --merit-upa")
    stream, area = rasterize_lines(lines, z.shape, tr)
    zfill = np.where(act, z, np.nanmax(np.where(act, z, -np.inf)))
    zb, chan, stats = burn(zfill, dx, dy, stream, area, active=act,
                           depth_a=a.depth_a, depth_b=a.depth_b,
                           width_a=a.width_a, width_b=a.width_b,
                           buffer_cells=a.buffer_cells,
                           min_area_km2=a.min_area_km2)
    zb = np.where(act, zb, z)
    tag = json.dumps(dict(method="AGREE+bankfull (tools/burn_channels.py)",
                          source=source, depth=[a.depth_a, a.depth_b],
                          width=[a.width_a, a.width_b],
                          buffer_cells=a.buffer_cells,
                          min_area_km2=a.min_area_km2,
                          date=_dt.date.today().isoformat(), **stats))
    prof.update(dtype="float32", compress="deflate")
    with rasterio.open(a.out, "w", **prof) as ds:
        ds.write(zb.astype("float32"), 1)
        ds.update_tags(**{BURN_TAG: tag})
    ch_out = a.channels_out or os.path.splitext(a.out)[0] + "_channels.tif"
    cprof = prof.copy()
    cprof.update(dtype="uint8", nodata=0)
    with rasterio.open(ch_out, "w", **cprof) as ds:
        ds.write(chan.astype("uint8"), 1)
        ds.update_tags(**{BURN_TAG: tag})
    print(f"burned DEM : {a.out}")
    print(f"channels   : {ch_out}")
    print("stats      : " + json.dumps(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
