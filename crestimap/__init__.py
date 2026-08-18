"""CREST-iMAP v2 — differentiable coupled hydrologic–hydrodynamic flood model.

Python-3 successor to the ANUGA-based CREST-iMAP v1.x, in two halves that
you can use together or apart:

  * CREST water balance (`crest.py`, `model.py`) — the one-layer VIC-curve
    soil model on every cell turns precipitation + PET into surface and
    subsurface runoff. Drive the full model straight from weather:

        from crestimap import CRESTiMAP, CRESTParams, weather
        from crestimap.forcing import SolverGrid
        grid  = SolverGrid.from_dem("dem.tif")
        rain  = weather.tif_stack("mrms/", grid, t0, units="mm/h")
        pet   = weather.constant(4.0, grid.z.shape, units="mm/day")
        CRESTiMAP(grid, CRESTParams(WM=120, B=0.4)).run(rain, pet, 72*3600,
                                                        out_dir="run/")

  * Hydrodynamics (`solver.py`, `event.py`) — a well-balanced,
    positivity-preserving 2-D shallow-water solver (Audusse et al. 2004
    hydrostatic reconstruction + HLL), pure PyTorch and differentiable.

In the real-time CREST-AI deployment the CREST water balance is run
UPSTREAM by EF5 instead, which supplies initial conditions (2-D Q, SM) and
lateral-inflow forcing (surface + subsurface runoff grids) to `run_event`;
that path skips the in-module CREST. Both halves share the same torch
autograd graph, so a coupled run is differentiable w.r.t. the CREST soil
parameters (WM, B, IM, KE, Ksat) and the Manning field together.

CREST-iMAP v1.x (Python 2.7 + ANUGA) is preserved on the `master` branch.
"""
from .solver import SWESolver, desing_velocity, minmod
from .analytic import stoker_dambreak
from .event import EventConfig, run_event
from .crest import CRESTParams, crest_step, crest_step_si
from .model import CRESTiMAP
from . import weather

__version__ = "2.0.0.dev0"
__all__ = ["SWESolver", "desing_velocity", "minmod", "stoker_dambreak",
           "EventConfig", "run_event", "CRESTParams", "crest_step",
           "crest_step_si", "CRESTiMAP", "weather"]
