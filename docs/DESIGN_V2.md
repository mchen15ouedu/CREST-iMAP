# CREST-iMAP v2 — design

Successor to the ANUGA-based CREST-iMAP v1.x (Li et al. 2021, EMS; Chen et
al. 2021, JHM). Goals, in order:

1. **Stability/accuracy**: replace the solver with a well-balanced,
   positivity-preserving finite-volume scheme — Audusse et al. (2004)
   hydrostatic reconstruction + HLL, MUSCL/minmod second order. Full
   momentum is retained, so trans/supercritical flow and shocks are
   captured (the known weak spot of local-inertial solvers such as
   LISFLOOD-FP and Inunda).
2. **Differentiability**: the solver is pure PyTorch; autograd runs through
   the entire simulation (verified against finite differences). Gradient
   targets: Manning n fields, bathymetry corrections, forcing. Long events
   backprop via gradient checkpointing (`checkpoint_every`).
3. **Two usable halves, one graph**: CREST-iMAP v2 keeps the CREST water
   balance (`crest.py`, `model.py`) AND the hydrodynamic solver, so a
   downloaded copy is a complete weather-driven flood model — precipitation
   + PET in, inundation out (`CRESTiMAP.run`). v1's cell-wise CREST
   (`crest_simp.model`, the VIC-curve one-layer soil) is ported from Cython
   to vectorized, differentiable PyTorch (`crest_step`) rather than retired;
   only the vendored ANUGA solver is gone. In the **CREST-AI real-time
   deployment** the CREST balance is instead computed UPSTREAM by EF5, which
   streams initial conditions (2-D Q, SM) and lateral-inflow forcing
   (surface + subsurface runoff grids) into `run_event`; that path simply
   skips the in-module CREST — it is a deployment wiring choice, not a
   removed capability. Because both halves are pure torch, autograd reaches
   the CREST soil parameters (WM, B, IM, KE, Ksat) and the Manning field in
   one backward pass, so the whole coupled model calibrates by gradient.

   *Fork strategy.* This repo (`mchen15ouedu/CREST-iMAP`, branch `v2`) is
   the full model for general users. If a hydrodynamics-only build is ever
   wanted for CREST-AI (to shave the CREST import/'`model.py`' surface off
   the worker image), it lives in a SEPARATE fork — the public model stays
   whole.

## Numerical scheme

- Regular DEM-aligned raster grid; states (h, hu, hv) cell-centered.
- Hydrostatic reconstruction at faces: `zf = max(zL, zR)`,
  `h* = max(0, h + z - zf)`; per-side pressure corrections
  `g/2 (h^2 - h*^2)` keep the scheme well-balanced; second order
  reconstructs (h, eta = h + z, u, v) with minmod so the C-property is
  exact at both orders (verified to 1e-12, wet and partially dry).
- HLL flux with Einfeldt-type wave-speed bounds; branchless formulation
  (safe for autograd).
- SSP-RK2; CFL-adaptive dt, detached from the graph.
- Desingularized velocities (Kurganov & Petrova 2007) at wet/dry fronts.
- Point-implicit Manning friction (closed form).
- Lateral inflow source term = rainfall excess / EF5 runoff [m/s].

## Validation (crestimap/tests)

| test | result |
|---|---|
| lake at rest, irregular bed, orders 1+2 | residual < 1e-12 |
| lake at rest with dry hills | residual < 1e-12 |
| Stoker wet-bed dam break (2nd order, 800 cells) | rel. L1 < 3% |
| closed-basin mass balance with inflow | error < 1e-10 |
| autograd vs central FD (Manning n, bed scale) | rel. diff < 1e-4 |

Next validation tier: UK EA benchmark cases, then Hurricane Harvey
(Brays Bayou) against the v1.x results and USGS HWMs.

## CREST-AI integration (event-triggered)

- Trigger: nowcast heatmap "flood" warning level -> basin selection ->
  domain bbox + window (spin-up before t0 through nowcast horizon).
- EF5 event run produces 2-D Q/SM initial-condition grids and
  surface+subsurface runoff forcing grids.
- Compute placement: basin-scale (~1e6 cells at 10 m) runs in minutes on
  a GPU; event-triggered dedicated GPU Space (or chunked ZeroGPU with
  state checkpoints) — not CONUS-continuous.
- Outputs: depth + max-depth rasters at 15-min cadence, served through the
  existing CREST-AI 2-D frame pipeline.

## Repository layout

- `crestimap/` — v2 package (Python 3, PyTorch).
- v1.x legacy (Python 2 + vendored ANUGA, `cresthh/` etc.) was removed
  from this branch 2026-08-12; it remains unchanged on `master`.
