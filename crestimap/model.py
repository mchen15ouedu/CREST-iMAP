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

Interflow (default `interflow="kw"`): CREST's subsurface runoff is routed
to the CHANNEL with the CREST v4 / EF5 kinematic-wave subsurface scheme
(crestimap.subsurface): each non-channel cell fills an interflow reservoir
IR, leaks IR*leaki per step, and the leak travels down the D8 network at
under*sqrt(slope) until it reaches a channel cell, where it is injected
into the 2-D solver as lateral inflow. D8 directions and upstream area are
derived from the DEM (crestimap.hydronet). Channel cells come from
`channel_mask` (tools/burn_channels.py writes one next to the burned DEM)
or, without one, from `th_km2` (upstream area threshold); one of the two
is required. `interflow="local"` keeps the old behaviour (interflow added
to the surface of its own cell) for comparison only.

Burn the channels first. The solver routes on the DEM as given, and a raw
DEM dams every river at each road crossing. The DEM is screened when the
model is built (crestimap.screen); a failed screen is logged as a WARNING
at the start and at the end of every run (logger "crestimap") and is in
result["dem_screen"]. The run is not stopped.

Water budget. Over the active cells, per run:
  dH + dSM + dIR = sum P - sum ET - sink outflow - subsurface exit,
with H the surface water, SM soil water, IR the interflow reservoirs,
sum P / sum ET cumulative precipitation and actual ET (result["budget"]),
subsurface exit the interflow that leaves past the outlet underground
(budget "subsurface_exit_m3") and sink outflow what the solver's basin
mask (or an open boundary) removes. Cells outside the mask still run CREST
but none of their water is routed; they are outside the budget.

The hydrologic and hydrodynamic halves are both pure torch, so the whole
coupled run stays differentiable end to end — autograd reaches the CREST
soil parameters (WM, B, IM, KE, Ksat) and the Manning field together.
"""
from __future__ import annotations

import datetime as _dt
import logging
import os

import numpy as np
import torch

from . import hydronet, screen
from .crest import CRESTParams, crest_step
from .solver import SWESolver
from .subsurface import KWInterflow, KWInterflowParams

log = logging.getLogger("crestimap")


class CRESTiMAP:
    def __init__(self, grid, crest_params: CRESTParams | None = None,
                 n_manning=0.05, order=2, bc="wall", active=None,
                 dt_hydro_h=1.0, reinfiltration=True, interflow="kw",
                 channel_mask=None, th_km2=None,
                 kw_params: KWInterflowParams | None = None, slope_min=1e-3,
                 screen_dem=True, device=None, dtype=None):
        """channel_mask : bool (ny, nx) array/tensor or GeoTIFF path on the
                       solver grid (tools/burn_channels.py *_channels.tif).
        th_km2       : without a channel mask, channel = upstream area >= th.
        kw_params    : KWInterflowParams(leaki, under), EF5 meanings.
        slope_min    : D8 slope floor for interflow travel times.
        screen_dem   : run the DEM burn screen (crestimap.screen)."""
        if interflow not in ("kw", "local"):
            raise ValueError(f"interflow must be 'kw' or 'local', not {interflow!r}")
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
        self.interflow = interflow
        self.kw_params = kw_params or KWInterflowParams()
        self.shape = tuple(z.shape)

        z_np = grid.z.detach().cpu().numpy().astype(float)
        act_np = None if self.mask is None else self.mask.cpu().numpy()
        self.net = hydronet.flow_network(z_np, grid.dx, grid.dy, active=act_np,
                                         slope_min=slope_min)
        ch = self._load_mask(channel_mask)
        self.dem_screen = (screen.screen_dem(z_np, grid.dx, grid.dy,
                                             active=act_np, channel_mask=ch,
                                             net=self.net)
                           if screen_dem else None)
        if ch is None and th_km2 is not None:
            ch = self.net.acc_km2 >= float(th_km2)
        if interflow == "kw" and (ch is None or not (ch & self.net.active).any()):
            raise ValueError(
                "interflow='kw' routes interflow to channel cells: pass "
                "channel_mask= (tools/burn_channels.py writes <dem>_channels.tif) "
                "or th_km2= (upstream-area threshold), or use interflow='local'")
        self.channel = ch
        self._routers = {}

    def _load_mask(self, m):
        if m is None:
            return None
        if isinstance(m, (str, os.PathLike)):
            import rasterio
            with rasterio.open(m) as ds:
                m = ds.read(1) > 0
        m = m.detach().cpu().numpy() if torch.is_tensor(m) else np.asarray(m)
        if m.shape != self.shape:
            raise ValueError(f"channel_mask shape {m.shape} != grid {self.shape}")
        return m.astype(bool)

    def _router(self, dt_s):
        key = round(float(dt_s), 6)
        if key not in self._routers:
            self._routers[key] = KWInterflow(self.net, self.channel,
                                             self.kw_params, dt_s,
                                             device=self.device, dtype=self.dtype)
        return self._routers[key]

    def run(self, precip_fn, pet_fn, t_end_s, sm0=None, h0=None, qx0=None,
            qy0=None, ir0=None, out_dir=None, output_every_s=3600.0,
            t0_dt=None, progress=None):
        """Integrate the coupled model to t_end_s.

        precip_fn, pet_fn : callables t_seconds -> (ny, nx) tensor [m/s]
                            (see crestimap.weather).
        sm0 : initial soil water [m]; scalar or (ny, nx). Default 0.5·WM/1000.
        ir0 : initial interflow reservoir [m] (EF5 ISU, in m). Default 0.
        Returns dict(h, qx, qy, sm, ir, max_depth, frames, budget,
        dem_screen); budget = dict(precip, et) cumulative per-cell depths
        [m] + subsurface_exit_m3. If out_dir is given, writes depth_<t>.tif
        frames + max_depth.tif.
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
        ir = z0.clone() if ir0 is None else torch.as_tensor(
            ir0, dtype=self.dtype, device=self.device) + z0
        cum_p = z0.clone()
        cum_et = z0.clone()
        sub_exit = torch.zeros((), dtype=self.dtype, device=self.device)
        if self.dem_screen is not None:
            self.dem_screen.log("run start:")

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
            if self.interflow == "kw":
                ir, to_channel, exited = self._router(dt_h).step(
                    ir, interflow_mm / 1000.0)
                runoff_mm = overland_mm + to_channel * 1000.0
                sub_exit = sub_exit + exited * (self.grid.dx * self.grid.dy)
            else:
                runoff_mm = overland_mm + interflow_mm
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
        result = {"h": h, "qx": qx, "qy": qy, "sm": sm, "ir": ir,
                  "max_depth": max_depth, "frames": frames,
                  "budget": {"precip": cum_p, "et": cum_et,
                             "subsurface_exit_m3": sub_exit},
                  "dem_screen": self.dem_screen}
        if self.dem_screen is not None:
            self.dem_screen.log("run end:")
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
