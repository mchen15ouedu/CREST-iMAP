# CREST-iMAP v2 — user manual (standalone, weather-driven model)

This manual covers the standalone model (`CRESTiMAP.run`): precipitation +
PET in, inundation out. The EF5-forced path used by CREST-AI
(`run_event` / `EventSession`) is documented in the `docs/P1_*` and
`DESIGN_*` briefs.

Workflow:

1. Get a DEM of the **basin** (the domain is always a basin, never a box).
2. Get a river network for it. You download this yourself, see §2.
3. Burn the network into the DEM with `tools/burn_channels.py` (§3).
4. Check the result with the burn screen (§4).
5. Run the model on the burned DEM with its channel mask (§5).

---

## 1. Why burn the channels

A bare-earth DEM (3DEP, SRTM, lidar) keeps every road embankment, bridge
deck and culvert crossing that spans a river. The 2-D solver routes water
over the DEM exactly as given, so each crossing is a dam. Water ponds
upstream of it, the reach below is starved, and depths and extents are
wrong in both places.

Measured on two 1" (~30 m) 3DEP basins, raw -> burned with NHDPlus HR:

| Basin | River cells in a depression | River cells draining along the river | Main-stem cells in a depression |
|---|---|---|---|
| Gaylord, MI (glacial) | 39.2 % -> 0.3 % | 51 % -> 99 % | 43.3 % -> 0.0 % |
| Iowa | 24.1 % -> 2.4 % | 64 % -> 95 % | 43.9 % -> 0.2 % |

The channel network also matters for interflow: CREST's subsurface runoff
is routed underground to the channel cells and enters the river there
(§6).

## 2. Getting a river network (manual download)

CREST-iMAP does not download networks. Get one of these:

**United States: NHDPlus High Resolution.**

1. Open the [National Map Downloader](https://apps.nationalmap.gov/downloader/).
2. Under Data, choose Hydrography -> *NHDPlus High Resolution (NHDPlus HR)*,
   format FileGDB.
3. Draw or upload your basin and download every 4-digit HUC (HU4)
   package it touches. Each is a zipped `NHDPLUS_H_<HU4>_HU4_GDB.gdb`, a few
   hundred MB.
4. Unzip it. The tool reads layer `NHDFlowline` and takes upstream area
   (`TotDASqKm`) from table `NHDPlusFlowlineVAA`, joined on `NHDPlusID`.
   If the basin spans several HU4s, burn once per GDB in sequence (the
   output of one run is the `--dem` of the next), or merge the flowlines
   into one GeoPackage first.

**Global: MERIT Hydro** (Yamazaki et al. 2019).

1. Register at the [MERIT Hydro page](https://global-hydrodynamics.github.io/MERIT_Hydro/)
   (Google Form + license agreement). The download password arrives by
   email.
2. Download the `dir` (flow direction) and `upa` (upstream area, km2)
   tiles covering your basin. They come as 30°x30° archives of 5°x5°
   GeoTIFFs.
3. If the basin spans tiles, mosaic each variable, e.g.
   `gdalbuildvrt dir.vrt dir_*.tif` and `gdalbuildvrt upa.vrt upa_*.tif`.
   The two must be on the same grid.

MERIT Hydro is 3" (~90 m). On a 1" DEM its centrelines can sit a cell or
two off the real channel. The tool takes the lowest DEM cell around each
line cell, which absorbs about one cell of offset. Prefer NHDPlus HR in
the US.

**Anything else.** Any line network with a CRS works: MERIT-Basins
flowlines, national hydrography, your own digitized rivers. Give the
upstream-area field with `--area-field` if there is one. Otherwise upstream
area is accumulated on the DEM, which underestimates it for rivers that
enter the DEM from outside.

## 3. Burning: `tools/burn_channels.py`

```bash
pip install -e ".[tools]"          # rasterio, scipy, geopandas

# NHDPlus HR
python tools/burn_channels.py --dem dem.tif --out dem_burned.tif \
    --flowlines NHDPLUS_H_0710_HU4_GDB.gdb --layer NHDFlowline \
    --vaa-layer NHDPlusFlowlineVAA --query "FType != 566"

# MERIT Hydro
python tools/burn_channels.py --dem dem.tif --out dem_burned.tif \
    --merit-dir dir.vrt --merit-upa upa.vrt --merit-threshold-km2 1
```

Outputs, on the DEM's grid:

* `dem_burned.tif`: the burned DEM.
* `dem_burned_channels.tif`: the channel mask (1 = channel cell), which
  you pass to the model as `channel_mask=`.

Both are tagged `CRESTIMAP_BURN` with the source, the parameters and the
burn statistics; `python -m crestimap.screen` prints the tag.

**Method.** The burn follows AGREE (Hellweger 1997), sized for hydraulics.
AGREE's usual settings (e.g. ArcHydro's 1000 m sharp drop) exist only to
force flow directions. Here the burned DEM *is* the terrain the solver
floods, so the channel must have a physical size.

1. The network is rasterized onto the DEM grid. Each line cell carries
   upstream area A [km2], from the attribute, MERIT `upa`, or the DEM.
2. Line elevation is the minimum DEM value in a 3x3 window around the
   line cell, which finds the thalweg next to a slightly offset line. It
   is then forced to fall at least 1 mm per cell downstream, which cuts
   through every crossing.
3. Bankfull geometry comes from upstream area (Bieger et al. 2015,
   CONUS-wide regressions): width W = 2.70·A^0.352 m, depth
   D = 0.30·A^0.213 m. The bed is the line elevation minus D, again forced
   to fall downstream.
4. Cells within W/2 of the line take their nearest line cell's bed
   elevation: a rectangular channel, AGREE's "sharp drop". Cells within a
   further `--buffer-cells` ramp linearly back to the DEM: AGREE's "smooth
   drop".
5. Terrain is only ever lowered.

Only the network's outlet may drain out of the DEM. That is the
largest-area stream cell on the DEM edge, or the lowest one without area
data. Other stream cells on the edge are inflows, so a reach dammed by a
crossing next to an inflow edge is routed through the network, not out of
the edge.

| Option | Default | Meaning |
|---|---|---|
| `--min-area-km2` | 1.0 | burn only streams draining at least this |
| `--depth-a`, `--depth-b` | 0.30, 0.213 | D = a·A^b [m] |
| `--width-a`, `--width-b` | 2.70, 0.352 | W = a·A^b [m] |
| `--buffer-cells` | 3 | AGREE smooth-drop buffer beyond the channel |
| `--query` | none | pandas filter on flowline fields, e.g. `"FType != 566"` (drop NHD coastlines) |
| `--area-field` | none | upstream-area field [km2]; otherwise from `--vaa-layer` or the DEM |
| `--mask` | none | basin mask GeoTIFF on the DEM grid (1 = basin) |
| `--merit-threshold-km2` | 1.0 | MERIT stream definition |

Regional hydraulic-geometry curves are better than the CONUS ones where
you have them: pass them through `--depth-*` / `--width-*`.

## 4. The burn screen

`crestimap.screen` decides from the DEM alone whether it looks burned, by
any method:

* **Checked cells.** With a channel mask, its cells. Without one, the
  DEM's main stems: cells draining at least 20 % of the largest upstream
  area in the domain, and at least 1 km2. Smaller DEM-derived "streams" are
  not used, because in lake, kettle or karst terrain they run through
  natural closed depressions that no burn removes.
* **In depressions.** The share of checked cells that the Priority-Flood
  fill raises by more than 5 cm. Each crossing fills the reach behind it.
  Limit: 5 %.
* **Channel adherence** (channel mask only). The share of channel cells
  whose steepest descent leads to another channel cell or out of the
  domain. Limit: 90 %.
* **Fill p95 / max.** Reported, not judged. Lakes and reservoirs on the
  network keep the maximum at metres even after burning.

The model screens the DEM when it is built. It **never stops a run**. A
failed screen is logged as a `WARNING` on logger `"crestimap"` at the
start and again at the end of every run, and the report is in
`result["dem_screen"]`. Python shows warnings without any setup. For the
INFO line on a pass, call `logging.basicConfig(level=logging.INFO)`.

Screen a DEM by hand:

```bash
python -m crestimap.screen dem_burned.tif --channels dem_burned_channels.tif
```

Exit code 0 means the DEM passed and 1 means it failed, so the screen can
gate a batch script.

**Limits.** A dam on the river is a real blockage: the reservoir behind it
fills, and a DEM that keeps the dam can fail the screen even when the rest
is burned. Burning through the dam (NHD flowlines pass through
reservoirs) makes the dam a channel. That is right for flood routing past
a full reservoir, and wrong when the reservoir has storage to give. Decide
per basin. Very small domains (no stream draining 1 km2) are not judged.

## 5. Running the model

```python
import logging
from crestimap import CRESTiMAP, CRESTParams, weather
from crestimap.forcing import SolverGrid
from crestimap.subsurface import KWInterflowParams

logging.basicConfig(level=logging.INFO)
grid = SolverGrid.from_dem("dem_burned.tif")
model = CRESTiMAP(grid, CRESTParams(WM=120, B=0.4, IM=0.05, Ksat=12.0),
                  n_manning=0.06, active=basin_mask,       # the basin
                  channel_mask="dem_burned_channels.tif",
                  kw_params=KWInterflowParams(leaki=0.1, under=1.0))
result = model.run(rain, pet, t_end_s=72 * 3600, sm0=0.06, out_dir="run/")
```

| Argument | Default | Meaning |
|---|---|---|
| `interflow` | `"kw"` | route interflow to the channel (§6); `"local"` adds it to its own cell's surface (legacy, for comparison only) |
| `channel_mask` | none | bool array/tensor or GeoTIFF path on the solver grid |
| `th_km2` | none | without a mask: channel = upstream area >= th_km2 |
| `kw_params` | leaki 0.1, under 1.0 | `KWInterflowParams`, EF5 meanings |
| `slope_min` | 1e-3 | D8 slope floor for interflow travel times |
| `screen_dem` | True | run the burn screen |
| `dt_hydro_h` | 1.0 | hydrologic step [h]; keep 1 to reuse EF5's per-step `leaki` |
| `reinfiltration` | True | ponded water re-enters CREST each step |

With `interflow="kw"`, one of `channel_mask` / `th_km2` is required; the
model refuses to start without one.

`result` holds `h, qx, qy` (final state), `sm` (soil water [m]), `ir`
(interflow reservoirs [m]), `max_depth`, `frames`, `dem_screen`, and
`budget` (cumulative per-cell `precip` and `et` [m], plus
`subsurface_exit_m3`).

## 6. Interflow routing

CREST's interflow is routed with the CREST v4 / EF5 kinematic-wave
subsurface scheme (EF5 `KinematicRoute.cpp`). On each non-channel cell,
every hydrologic step:

```
IR   += interflow
leak  = IR · leaki ;  IR -= leak
```

The leak travels down the D8 network within the same step. Each cell
takes `len / (under · sqrt(slope))` seconds to cross. Walking downstream,
the leak stops at the first channel cell. If the step ends between two
non-channel cells, the leak is split between them in proportion to where
it ended. Leak landing on a hillslope cell joins its IR and moves on next
step. Leak reaching a channel cell, and a channel cell's own interflow,
enter the 2-D solver there as lateral inflow. Leak that walks past the
basin outlet leaves underground (`budget["subsurface_exit_m3"]`).

D8 directions come from the DEM (Priority-Flood, `crestimap.hydronet`), so
run on the burned DEM: burning makes every path end in the network.

Differences from EF5 v4.3:

* Leak arriving at a channel cell is injected. In the EF5 v4.3 fork it is
  written into a slot the channel branch never reads, so it appears to be
  lost.
* No groundwater/baseflow bucket (CREST v4's third layer).
* Slope floor `slope_min` instead of EF5's 1 m minimum drop per cell.
  At 1" that EF5 floor would mean a 3.3 % minimum slope.

Gradients reach `leaki` and the IR state. The routing pattern (`under`,
slope, channel mask) is fixed per run.

## 7. Water budget

Over the active (basin) cells:

```
ΔH + ΔSM + ΔIR = ΣP − ΣET − sink outflow − subsurface exit
```

* H: surface water.
* SM: soil water.
* IR: interflow reservoirs.
* ΣP, ΣET: `budget["precip"]` and `budget["et"]`.
* Sink outflow: water the basin mask removes at the pour point.
* Subsurface exit: `budget["subsurface_exit_m3"]`.

The test suite checks this to 1e-9 with rain, PET, a basin mask and a
measured sink.

## 8. Troubleshooting

* **"DEM burn screen: FAIL" in the log.** Burn the DEM (§3) and re-run.
  If it was burned by another method, run the screen with that method's
  channel mask: if adherence passes but depressions fail, look for dams or
  lakes on the network (§4, Limits).
* **`ValueError: interflow='kw' routes interflow to channel cells`.** Pass
  `channel_mask=` (the burn tool writes one) or `th_km2=`.
* **`channel_mask shape ... != grid`.** The mask must be on the solver
  grid. Use the `*_channels.tif` written next to the DEM you load.
* **`OMP: Error #15 ... libiomp5md.dll already initialized`** (Windows
  conda). torch and MKL numpy each bring an OpenMP runtime. The burn tool
  avoids importing torch. For the model, install numpy from the same
  channel as torch, or use a pip-only environment.
