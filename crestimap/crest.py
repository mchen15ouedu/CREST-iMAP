"""CREST water balance — the hydrologic half of CREST-iMAP.

One-layer CREST cell model (variable-infiltration-curve soil, EF5 lineage;
Wang et al. 2011, Li et al. 2021 for the CREST-iMAP coupling), ported from
CREST-iMAP v1's `crest_simp.pyx` to vectorized PyTorch: every cell of the
raster is updated at once, on CPU or GPU, and the whole update is built
from `torch.where` selections so it is differentiable w.r.t. the soil
parameters, the states and the forcing (autograd calibration of WM, B, IM,
KE, Ksat exactly like Manning n in the hydrodynamic half).

Per hydrologic step (mm units inside, m/m s at the interface):

  precip  = P·dt (+ surface water offered for re-infiltration)
  adjPET  = PET·dt·KE
  wet step (precip > adjPET):
      precipSoil   = (precip - adjPET)(1 - IM);  precipImperv = the rest
      A            = Wmaxm (1 - (1 - SM/WM)^(1/(1+B))),  Wmaxm = WM(1+B)
      infiltration = WM[(1 - A/Wmaxm)^(1+B) - (1 - (A+precipSoil)/Wmaxm)^(1+B)]
      R            = precipSoil - infiltration          (saturation excess)
      interflow    = min(R, (SM+Wo)/WM/2 · Ksat · dt_h)
      overland     = R - interflow + precipImperv
  dry step (precip <= adjPET):
      overland = 0; ET draws the soil down by (adjPET - precip)·SM/WM
  any SM above WM leaves as interflow ("interflowExcess").

Water balance closes exactly per cell:
  precip_in = actET + (SM_new - SM_old) + overland + interflow.

`crest_step` is the pure math; `CRESTParams` bundles the five parameter
fields (scalars or (ny, nx) tensors, any mix).
"""
from __future__ import annotations

import dataclasses

import torch


@dataclasses.dataclass
class CRESTParams:
    """CREST soil parameters (EF5 names/units). Each may be a float or an
    (ny, nx) tensor; tensors may require_grad for calibration.

    WM   : soil max water capacity [mm]
    B    : variable-infiltration-curve exponent [-]
    IM   : impervious area fraction [0..1]  (EF5 gives it in %; divide by 100)
    KE   : PET adjustment factor [-]
    Ksat : soil saturated hydraulic conductivity [mm/h] (interflow rate)
    """
    WM: object = 120.0
    B: object = 0.5
    IM: object = 0.05
    KE: object = 1.0
    Ksat: object = 10.0

    def to(self, like: torch.Tensor) -> "CRESTParams":
        """Broadcast every field to a tensor on `like`'s device/dtype."""
        def _t(v):
            t = torch.as_tensor(v, dtype=like.dtype, device=like.device)
            return t
        return CRESTParams(_t(self.WM), _t(self.B), _t(self.IM), _t(self.KE),
                           _t(self.Ksat))

    @classmethod
    def from_rasters(cls, grid, WM=None, B=None, IM=None, KE=None, Ksat=None,
                     im_percent=False):
        """Fields as GeoTIFF paths (regridded onto the solver grid) or
        numbers. `im_percent=True` divides IM by 100 (EF5 convention)."""
        import numpy as np
        from .forcing import regrid_to_solver

        def _load(v):
            if v is None:
                return None
            if isinstance(v, (int, float)):
                return float(v)
            import rasterio
            with rasterio.open(v) as ds:
                a = ds.read(1).astype(float)
                if ds.nodata is not None:
                    a = np.where(a == ds.nodata, np.nan, a)
                out = regrid_to_solver(a, ds.transform, tuple(grid.z.shape),
                                       grid.transform, ds.crs, grid.crs,
                                       method="bilinear")
            out = np.where(np.isfinite(out), out, np.nanmedian(out))
            return torch.as_tensor(out, dtype=grid.z.dtype, device=grid.z.device)

        d = {}
        for k, v in (("WM", WM), ("B", B), ("IM", IM), ("KE", KE), ("Ksat", Ksat)):
            lv = _load(v)
            if lv is not None:
                d[k] = lv
        p = cls(**d)
        if im_percent:
            p.IM = p.IM / 100.0
        return p


def crest_step(precip_mm, pet_mm, sm_mm, params: CRESTParams, dt_s: float):
    """One CREST water-balance step for every cell (all tensors broadcast).

    precip_mm : water offered to the soil this step [mm] (rain + any surface
                water re-infiltrating)
    pet_mm    : potential ET this step [mm] (before KE)
    sm_mm     : soil water at step start [mm]
    dt_s      : step length [s]

    Returns (sm_new_mm, overland_mm, interflow_mm, act_et_mm).
    """
    p = params
    WM = torch.where(p.WM > 0, p.WM, torch.full_like(p.WM, 100.0))
    B = torch.where(p.B < 0, torch.ones_like(p.B), p.B)
    IM = torch.clamp(p.IM, 0.0, 0.999999)
    Ksat = torch.where(p.Ksat < 0, torch.ones_like(p.Ksat), p.Ksat)
    KE = p.KE
    SM = torch.clamp(sm_mm, min=0.0)
    precip = precip_mm
    adj_pet = pet_mm * KE
    step_h = float(dt_s) / 3600.0

    # water above capacity leaves as interflow in both branches
    interflow_excess = torch.clamp(SM - WM, min=0.0)
    SM = torch.minimum(SM, WM)

    # ---------------- wet branch: precip > adjPET ----------------
    avail = precip - adj_pet
    precip_soil = torch.clamp(avail, min=0.0) * (1.0 - IM)
    precip_imperv = torch.clamp(avail, min=0.0) - precip_soil
    wmaxm = WM * (1.0 + B)
    frac = torch.clamp(1.0 - SM / WM, min=0.0)
    A = wmaxm * (1.0 - frac ** (1.0 / (1.0 + B)))
    saturating = (precip_soil + A) >= wmaxm
    R_sat = torch.clamp(precip_soil - (WM - SM), min=0.0)
    inner = torch.clamp(1.0 - (A + precip_soil) / wmaxm, min=0.0)
    infil = WM * ((1.0 - A / wmaxm) ** (1.0 + B) - inner ** (1.0 + B))
    infil = torch.minimum(torch.clamp(infil, min=0.0), precip_soil)
    R_unsat = torch.clamp(precip_soil - infil, min=0.0)
    Wo_unsat = SM + infil
    R = torch.where(saturating, R_sat, R_unsat)
    Wo_wet = torch.where(saturating, WM, Wo_unsat)
    # SM == WM (already full): everything is excess
    full = SM >= WM
    R = torch.where(full, precip_soil, R)
    Wo_wet = torch.where(full, WM, Wo_wet)
    tem_x = (SM + Wo_wet) / WM / 2.0 * Ksat * step_h
    interflow_wet = torch.minimum(R, tem_x)
    overland_wet = R - interflow_wet + precip_imperv
    et_wet = adj_pet
    interflow_wet = interflow_wet + interflow_excess

    # ---------------- dry branch: precip <= adjPET ----------------
    excess_et = torch.clamp(adj_pet - precip, min=0.0) * SM / WM
    excess_et = torch.minimum(excess_et, SM)
    Wo_dry = SM - excess_et
    et_dry = excess_et + precip
    interflow_dry = interflow_excess

    wet = precip > adj_pet
    sm_new = torch.where(wet, Wo_wet, Wo_dry)
    overland = torch.where(wet, overland_wet, torch.zeros_like(overland_wet))
    interflow = torch.where(wet, interflow_wet, interflow_dry)
    act_et = torch.where(wet, et_wet, et_dry)
    return sm_new, overland, interflow, act_et


def crest_step_si(precip_rate, pet_rate, sm_m, params: CRESTParams, dt_s: float,
                  surface_in_m=None):
    """SI-unit convenience wrapper: rates in m/s, storages in m.

    surface_in_m : optional surface water depth offered for re-infiltration
                   (CREST-iMAP v1 behaviour); it is added to the step's
                   precipitation before the balance.
    Returns (sm_new_m, overland_m, interflow_m, act_et_m) — depths per step.
    """
    precip_mm = precip_rate * float(dt_s) * 1000.0
    if surface_in_m is not None:
        precip_mm = precip_mm + surface_in_m * 1000.0
    pet_mm = pet_rate * float(dt_s) * 1000.0
    sm, ov, inf, et = crest_step(precip_mm, pet_mm, sm_m * 1000.0, params, dt_s)
    return sm / 1000.0, ov / 1000.0, inf / 1000.0, et / 1000.0
