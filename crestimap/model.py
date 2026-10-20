"""CREST-iMAP — the full coupled model: CREST water balance on every cell
generates runoff from weather, and the well-balanced SWE solver routes that
runoff over the DEM as 2-D inundation. This is the standalone entry point
for users who drive the model with precipitation + PET (no EF5 upstream).

    from crestimap import CRESTiMAP, CRESTParams
    from crestimap.forcing import SolverGrid
    from crestimap import weather

    grid  = SolverGrid.from_dem("dem.tif")
    rain  = weather.tif_stack("mrms/", grid, t0, units="mm/h")
    pet   = weather.constant(4.0, grid.z.shape, units="mm/day")
    model = CRESTiMAP(grid, CRESTParams(WM=120, B=0.4, IM=0.05, Ksat=12),
                      n_manning=0.06)
    result = model.run(rain, pet, t_end_s=72*3600, out_dir="run/")

Coupling (per hydrologic substep dt_h, default 1 h — the CREST cell model
is a bucket, not a PDE, so it wants hourly steps, while the SWE solver sub-
steps on its own CFL dt inside each):

  1. CREST_step(P, PET, SM) -> overland + interflow depths, updated SM.
  2. That runoff depth / dt_h becomes the lateral-inflow rate handed to
     SWESolver.run(rain_fn=...) for the next dt_h of hydrodynamic routing.
  3. Surface water sitting on a cell is offered back to CREST with the next
     step's rain (re-infiltration, v1 behaviour; toggle reinfiltration=False),
     so ET and infiltration act on ponded water. What CREST returns refills
     the standing water first (momentum scaled with depth, velocity kept) and
     only the excess enters as lateral inflow.

Interflow. `route_interflow=True` (default) adds CREST's interflow to the
routed surface inflow, matching the EF5-forced deployment where runoff +
subrunoff are summed. `route_interflow=False` puts it in a per-cell linear
interflow reservoir S instead: each step S += interflow, then
S·(1 - exp(-dt_h/interflow_k_h)) is released to the surface inflow.
`interflow_k_h=None` (default) never releases — the store is only tracked.
Either way S is returned, so no water leaves the books.

Water budget. Over the active cells (the whole grid without a mask), per
run:  ΔH + ΔSM + ΔS = ΣP - ΣET - sink outflow, with H the surface water,
SM soil water, S the interflow store, ΣP / ΣET the cumulative precipitation
and actual ET in result["budget"]. Sink outflow is what the solver's basin
mask (or an open boundary) removes. Cells outside the mask still run CREST,
but none of their runoff is routed; they are outside the budget.

The hydrologic and hydrodynamic halves are both pure torch, so the whole
coupled run stays differentiable end to end — autograd reaches the CREST
soil parameters (WM, B, IM, KE, Ksat) and the Manning field together.
"""
from __future__ import annotations

import datetime as _dt
import os

import torch

from .crest import CRESTParams, crest_step
from .solver import SWESolver


class CRESTiMAP:
    def __init__(self, grid, crest_params: CRESTParams | None = None,
                 n_manning=0.05, order=2, bc="wall", active=None,
                 dt_hydro_h=1.0, reinfiltration=True, route_interflow=True,
                 interflow_k_h=None, device=None, dtype=None):
        self.grid = grid
        self.device = device or grid.z.device
        self.dtype = dtype or grid.z.dtype
        z = grid.z.to(self.device, self.dtype)
        self.mask = None if active is None else torch.as_tensor(
            active, dtype=torch.bool, device=self.device)
        self.solver = SWESolver(z, dx=grid.dx, dy=grid.dy, n_manning=n_manning,
                                order=order, bc=bc, active=self.mask)
        cp = (crest_params or CRESTParams()).to(z)
        self.params = cp
        self.dt_hydro = float(dt_hydro_h) * 3600.0
        self.reinfiltration = reinfiltration
        self.route_interflow = route_interflow
        # linear-reservoir time constant [h] of the interflow store; scalar
        # or (ny, nx), may require grad. None = hold (track only).
        self.interflow_k_h = None if interflow_k_h is None else torch.as_tensor(
            interflow_k_h, dtype=self.dtype, device=self.device)
        self.shape = tuple(z.shape)

    def run(self, precip_fn, pet_fn, t_end_s, sm0=None, h0=None, qx0=None,
            qy0=None, interflow0=None, out_dir=None, output_every_s=3600.0,
            t0_dt=None, progress=None):
        """Integrate the coupled model to t_end_s.

        precip_fn, pet_fn : callables t_seconds -> (ny, nx) tensor [m/s]
                            (see crestimap.weather).
        sm0 : initial soil water [m]; scalar or (ny, nx). Default 0.5·WM/1000.
        interflow0 : initial interflow store [m] (route_interflow=False only
                     ever fills it). Default 0.
        Returns dict(h, qx, qy, sm, interflow, max_depth, frames, budget)
        with tensors; budget = dict(precip, et) cumulative per-cell depths
        [m]. If out_dir is given, writes depth_<t>.tif frames +
        max_depth.tif.
        """
        ny, nx = self.shape
        z0 = torch.zeros(self.shape, dtype=self.dtype, device=self.device)
        if sm0 is None:
            sm = 0.5 * self.params.WM / 1000.0 + z0
        else:
            sm = torch.as_tensor(sm0, dtype=self.dtype, device=self.device) + z0
        h = z0.clone() if h0 is None else torch.as_tensor(
            h0, dtype=self.dtype, device=self.device).clone()
        qx = z0.clone() if qx0 is None else torch.as_tensor(qx0, dtype=self.dtype, device=self.device).clone()
        qy = z0.clone() if qy0 is None else torch.as_tensor(qy0, dtype=self.dtype, device=self.device).clone()
        if self.mask is not None:
            h, qx, qy = self.solver.mask_state(h, qx, qy)
        store = z0.clone() if interflow0 is None else torch.as_tensor(
            interflow0, dtype=self.dtype, device=self.device) + z0
        cum_p = z0.clone()
        cum_et = z0.clone()

        frames = []
        max_depth = h.clone()
        t = 0.0
        next_out = output_every_s
        while t < t_end_s - 1e-6:
            dt_h = min(self.dt_hydro, t_end_s - t)
            # --- CREST water balance over this hydrologic step ---
            P = precip_fn(t)                       # m/s
            PET = pet_fn(t)                         # m/s
            precip_mm = P * dt_h * 1000.0
            surf_mm = (h * 1000.0) if self.reinfiltration else 0.0
            precip_mm = precip_mm + surf_mm
            pet_mm = PET * dt_h * 1000.0
            sm_new, overland_mm, interflow_mm, et_mm = crest_step(
                precip_mm, pet_mm, sm * 1000.0, self.params, dt_h)
            sm = sm_new / 1000.0
            cum_p = cum_p + P * dt_h
            cum_et = cum_et + et_mm / 1000.0
            if self.route_interflow:
                runoff_mm = overland_mm + interflow_mm
            else:
                # linear interflow reservoir: fill, then release a fraction
                store = store + interflow_mm / 1000.0
                runoff_mm = overland_mm
                if self.interflow_k_h is not None:
                    rel = store * (1.0 - torch.exp(-(dt_h / 3600.0)
                                                   / self.interflow_k_h))
                    if self.mask is not None:   # nowhere to route it to
                        rel = torch.where(self.mask, rel, torch.zeros_like(rel))
                    store = store - rel
                    runoff_mm = runoff_mm + rel * 1000.0
            if self.reinfiltration:
                # Operator split: what CREST returns refills the standing water
                # first, so the surface loses only what ET and infiltration
                # took; momentum scales with depth (velocity kept).  Zeroing h
                # while keeping qx/qy and re-adding all of it as a source over
                # the step would leave momentum on dry cells and empty every
                # channel and lake once per step.  Only the excess over the
                # offered water enters as a source.
                keep_mm = torch.minimum(runoff_mm, surf_mm)
                h_new = keep_mm / 1000.0
                ratio = torch.where(h > 0, h_new / torch.clamp(h, min=1e-12),
                                    torch.zeros_like(h))
                if self.mask is not None:
                    h_new = torch.where(self.mask, h_new, h)
                    ratio = torch.where(self.mask, ratio, torch.ones_like(ratio))
                h, qx, qy = h_new, qx * ratio, qy * ratio
                runoff_mm = runoff_mm - keep_mm
            inflow_rate = (runoff_mm / 1000.0) / dt_h    # m/s, constant over dt_h
            if self.mask is not None:
                inflow_rate = torch.where(self.mask, inflow_rate,
                                          torch.zeros_like(inflow_rate))

            def rain_fn(_t, _r=inflow_rate):
                return _r
            rain_fn.next_change = lambda _t: float("inf")

            # --- hydrodynamic routing of that runoff for dt_h ---
            h, qx, qy = self.solver.run(h, qx, qy, t_end=dt_h, rain_fn=rain_fn)
            t += dt_h
            max_depth = torch.maximum(max_depth, h)
            if progress:
                progress(f"t={t/3600:.1f}h wet={(h>0.01).sum().item()} "
                         f"max={max_depth.max().item():.2f}m")
            if out_dir and t >= next_out - 1e-6:
                frames.append(self._write_frame(out_dir, h, t, t0_dt))
                next_out += output_every_s
        result = {"h": h, "qx": qx, "qy": qy, "sm": sm, "interflow": store,
                  "max_depth": max_depth, "frames": frames,
                  "budget": {"precip": cum_p, "et": cum_et}}
        if out_dir:
            self._write_frame(out_dir, max_depth, None, t0_dt, name="max_depth")
        return result

    def _write_frame(self, out_dir, depth, t_s, t0_dt, name=None):
        from . import io as iomod
        os.makedirs(out_dir, exist_ok=True)
        if name is None:
            if t0_dt is not None and t_s is not None:
                stamp = (t0_dt + _dt.timedelta(seconds=t_s)).strftime("%Y%m%d%H%M")
            else:
                stamp = f"{int((t_s or 0)):08d}"
            name = f"depth_{stamp}"
        path = os.path.join(out_dir, name + ".tif")
        iomod.write_depth(path, depth.detach().cpu().numpy(),
                          self.grid.transform, self.grid.crs)
        return path
