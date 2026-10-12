"""Re-infiltration coupling tests: standing water re-enters the CREST balance
each hydrologic step (v1 behaviour), so a closed pond loses water to
infiltration and ET, the surface + soil budget closes, and no momentum is
left on cells that the exchange dries out."""
import numpy as np
import torch
from rasterio.transform import from_origin
from rasterio.crs import CRS

from crestimap import CRESTiMAP, weather
from crestimap.crest import CRESTParams
from crestimap.forcing import SolverGrid

torch.set_default_dtype(torch.float64)

NY = NX = 31
DX = 30.0
CELL = DX * DX


def _bowl():
    """Closed parabolic depression (walls all round) with a 0.5 m pond."""
    i, j = np.mgrid[0:NY, 0:NX]
    z = 0.002 * ((i - NY // 2) ** 2 + (j - NX // 2) ** 2)
    grid = SolverGrid.from_array(z, from_origin(500000.0, 4000000.0, DX, DX),
                                 CRS.from_epsg(32615))
    h0 = np.clip(0.5 - z, 0.0, None)
    return grid, torch.as_tensor(h0)


def _run(reinfiltration, pet_mm_day, hours=24, sm0=0.02):
    grid, h0 = _bowl()
    zero = weather.constant(0.0, grid.z.shape, units="mm/h")
    pet = weather.constant(pet_mm_day, grid.z.shape, units="mm/day")
    m = CRESTiMAP(grid, CRESTParams(WM=80.0, B=0.3, IM=0.0, Ksat=5.0, KE=1.0),
                  n_manning=0.05, reinfiltration=reinfiltration)
    res = m.run(zero, pet, t_end_s=hours * 3600.0, sm0=sm0, h0=h0)
    return h0, res


def test_pond_persists_without_reinfiltration():
    h0, res = _run(False, pet_mm_day=5.0)
    v0, v1 = float(h0.sum()) * CELL, float(res["h"].sum()) * CELL
    assert abs(v1 - v0) / v0 < 1e-6, "one-way coupling: a closed pond cannot lose water"


def test_reinfiltration_drains_closed_pond():
    h0, res = _run(True, pet_mm_day=5.0)
    v0, v1 = float(h0.sum()) * CELL, float(res["h"].sum()) * CELL
    assert v1 < 0.97 * v0, "re-infiltration must remove ponded water"


def test_surface_plus_soil_budget_closes_without_et():
    # PET = 0: every litre that leaves the pond must be in the soil
    sm0 = 0.02
    h0, res = _run(True, pet_mm_day=0.0, sm0=sm0)
    before = float(h0.sum()) * CELL + sm0 * NY * NX * CELL
    after = float(res["h"].sum()) * CELL + float(res["sm"].sum()) * CELL
    assert abs(after - before) / before < 1e-9, (before, after)


def test_no_momentum_on_dried_cells():
    _, res = _run(True, pet_mm_day=5.0)
    dry = res["h"] <= 0.0
    assert torch.all(res["qx"][dry] == 0.0) and torch.all(res["qy"][dry] == 0.0)
    assert torch.isfinite(res["h"]).all()


if __name__ == "__main__":
    for fn in (test_pond_persists_without_reinfiltration,
               test_reinfiltration_drains_closed_pond,
               test_surface_plus_soil_budget_closes_without_et,
               test_no_momentum_on_dried_cells):
        fn()
        print("ok", fn.__name__)
