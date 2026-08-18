# CREST-iMAP v2

**Coupled Routing Excess STorage inundation MApping and Prediction —
version 2.** A differentiable, physically complete flood model in PyTorch:
the **CREST water balance** turns weather into runoff on every cell, and a
**well-balanced 2-D shallow-water solver** routes that runoff over the DEM
into inundation. It is also the hydrodynamic engine of the
[CREST-AI](https://github.com/mchen15ouedu/CREST_AI) real-time dashboard.

v2 is a ground-up rewrite. The vendored ANUGA solver of v1.x is replaced by
a modern finite-volume scheme (Audusse 2004 + HLL, 2nd-order MUSCL), and
v1's Cython cell-wise CREST is ported to vectorized, differentiable torch.
You can run the model two ways:

- **Full model, weather-driven** (`CRESTiMAP`) — give it precipitation +
  PET and a DEM; CREST generates runoff and the solver routes it. This is
  the path for downloaded/offline use. See *Quick start* below.
- **Hydrodynamics only** (`run_event` / `SWESolver`) — feed pre-computed
  runoff/discharge grids straight to the solver. This is how CREST-AI runs
  it, with **EF5/CREST upstream** supplying initial conditions (2-D Q, SM)
  and surface+subsurface runoff forcing, so the water balance is done once,
  basin-wide, before the 2-D step.

Both halves are pure torch on one autograd graph, so the coupled model is
differentiable end to end — Manning n **and** the CREST soil parameters
(WM, B, IM, KE, Ksat) calibrate by gradient. CREST-iMAP v1.x (Python 2.7 +
ANUGA) is preserved unchanged on the `master` branch.

## Numerical scheme

- Regular DEM-aligned raster grid, cell-centered states (h, hu, hv).
- Hydrostatic reconstruction of Audusse et al. (2004): well-balanced
  (exact lake-at-rest C-property, machine precision) and
  positivity-preserving, at first and second order.
- HLL flux; second order via MUSCL/minmod reconstruction in
  (h, eta, u, v).
- SSP-RK2 time stepping with adaptive CFL timestep.
- Point-implicit Manning friction (closed form, differentiable).
- Desingularized wet/dry velocities (Kurganov & Petrova 2007) — stable
  fronts, no NaN gradients.
- Everything is torch tensor ops: the entire simulation is
  **differentiable end to end** w.r.t. Manning n, bed elevation, initial
  conditions, and forcing (autograd calibration), and runs unchanged on
  CPU or GPU.

Validated in `crestimap/tests`: exact C-property (wet and dry, both
orders), Stoker dam-break analytic solution, mass conservation to 1e-9,
autograd gradients vs finite differences, and an end-to-end
EF5-grids-to-solver volume balance.

## Installation

Python >= 3.10, torch >= 2.0.

```bash
pip install -e ".[geo]"      # rasterio + requests for DEM/forcing I/O
pip install -e ".[geo,test]" # + pytest
```

## Quick start

**Full model — rain + PET over a real basin.** Point it at a DEM and a
folder of rainfall GeoTIFFs (any CRS/resolution; MRMS, IMERG, gauge grids):

```python
import datetime as dt
from crestimap import CRESTiMAP, CRESTParams, weather
from crestimap.forcing import SolverGrid

grid = SolverGrid.from_dem("dem.tif")             # 3DEP, SRTM, anything
t0   = dt.datetime(2026, 8, 14, 0, 0)
rain = weather.tif_stack("mrms/", grid, t0, units="mm/h",   # or "mm"/"mm/day"
                         pattern="precip.{time}.tif", time_format="%Y%m%d%H%M")
pet  = weather.constant(4.0, grid.z.shape, units="mm/day")  # or a tif_stack

model = CRESTiMAP(grid, CRESTParams(WM=120, B=0.4, IM=0.05, Ksat=12.0),
                  n_manning=0.06)
result = model.run(rain, pet, t_end_s=72*3600, sm0=0.06,     # 60 mm initial soil
                   out_dir="run/", output_every_s=3600)
print(result["max_depth"].max())                  # peak inundation depth [m]
```

Per-cell parameter rasters (WM, B, IM, KE, Ksat) load with
`CRESTParams.from_rasters(grid, WM="wm.tif", ...)`.

**Hydrodynamics only — you already have runoff/discharge grids:**

```python
import torch
from crestimap import SWESolver

z = torch.zeros(200, 200)
h = torch.zeros(200, 200); h[:, :100] = 1.0        # dam break
solver = SWESolver(z, dx=10.0, dy=10.0, n_manning=0.03, order=2, bc="wall")
h, qx, qy = solver.run(h, torch.zeros_like(h), torch.zeros_like(h), t_end=300.0)
```

Make any of `z`, `n_manning`, or the `CRESTParams` fields require grad and
backpropagate through `run()` to calibrate against observed depths.

## Package layout

| Module | Purpose |
|---|---|
| `crestimap/crest.py` | **CREST water balance** — vectorized differentiable VIC-curve one-layer soil (`crest_step`, `CRESTParams`) |
| `crestimap/model.py` | **`CRESTiMAP`** — the coupled weather-driven model (CREST runoff → SWE routing) |
| `crestimap/weather.py` | precipitation / PET forcing readers (`tif_stack`, `uniform_series`, `constant`) |
| `crestimap/solver.py` | the well-balanced SWE solver (`SWESolver`) |
| `crestimap/forcing.py` | EF5/CREST coupling: runoff-grid forcing, initial state from routed discharge, channel-stage coupling |
| `crestimap/dem.py` | on-demand USGS 3DEP DEM tiles (1" ~30 m, 1/3" ~10 m), local cache |
| `crestimap/event.py` | `EventConfig` / `run_event`: one EF5-forced flood event end to end (DEM, forcing, solve, depth frames, manifest) |
| `crestimap/io.py` | compact uint16-centimeter GeoTIFF depth frames |
| `crestimap/analytic.py` | analytic references (Stoker dam break) |
| `crestimap/tests/` | validation suite (`pytest crestimap/tests`) |

## Deployment in CREST-AI

The CREST-AI dashboard triggers an event when its AI nowcast flags a
flood at a USGS gauge that observations confirm: EF5 runs the basin in
nowcast mode with gridded runoff output, then `run_event` simulates 2-D
inundation at the DEM's native resolution and publishes depth frames.
Design and operations documents in [`docs/`](docs):

| Document | Contents |
|---|---|
| [`DESIGN_V2.md`](docs/DESIGN_V2.md) | solver design and v1 -> v2 rationale |
| [`DESIGN_V27_PARALLEL.md`](docs/DESIGN_V27_PARALLEL.md) | full-basin GPU/parallel architecture: job queue, single-GPU worker, multi-GPU subbasin decomposition with ghost-cell halo exchange, degradation ladder |
| [`P1_WORKER_BRIEF.md`](docs/P1_WORKER_BRIEF.md) | executable contract for the HPC GPU event worker |

## Citation

CREST-iMAP v2 (this branch) has no dedicated paper yet; please cite the
original CREST-iMAP papers:

> Li, Z., Chen, M., Gao, S., Luo, X., Gourley, J., Kirstetter, P., Yang,
> T., Kolar, R., McGovern, A., Wen, Y., Rao, B., Yami, T., Hong, Y., 2021.
> CREST-iMAP v1.0: A fully coupled hydrologic-hydraulic modeling framework
> dedicated to flood inundation mapping and prediction. Environmental
> Modelling and Software, 141, 105051.
> https://doi.org/10.1016/j.envsoft.2021.105051

> Li, Z., Chen, M., Gao, S., Wen, Y., Gourley, J. J., Yang, T., Kolar, R.,
> & Hong, Y., 2022. Can re-infiltration process be ignored for flood
> inundation mapping and prediction during extreme storms? A case study in
> Texas Gulf Coast region. Environmental Modelling & Software, 155,
> 105450. https://doi.org/10.1016/j.envsoft.2022.105450
