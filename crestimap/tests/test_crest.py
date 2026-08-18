"""CREST water-balance tests: per-cell mass closure, the v1 reference
values, monotonic responses, autograd through the soil parameters, and a
short end-to-end coupled CRESTiMAP run on a synthetic tilted plane."""
import datetime as _dt

import numpy as np
import torch

from crestimap.crest import CRESTParams, crest_step
from crestimap import CRESTiMAP, weather
from crestimap.forcing import SolverGrid

torch.set_default_dtype(torch.float64)


def _balance_residual(P_mm, PET_mm, SM_mm, p, dt):
    sm1, ov, inf, et = crest_step(torch.tensor(P_mm), torch.tensor(PET_mm),
                                  torch.tensor(SM_mm), p, dt)
    # closure: precip in = actET + storage change + overland + interflow
    sm0 = min(SM_mm, float(p.WM if isinstance(p.WM, (int, float)) else p.WM))
    res = P_mm - (et.item() + (sm1.item() - sm0) + ov.item() + inf.item())
    return res, (sm1.item(), ov.item(), inf.item(), et.item())


def test_water_balance_closes():
    p = CRESTParams(WM=100.0, B=0.4, IM=0.05, KE=1.0, Ksat=10.0).to(torch.zeros(1))
    for P, PET, SM in [(30, 2, 40), (5, 8, 60), (100, 1, 10), (0, 5, 90),
                       (50, 3, 100), (2, 2, 0)]:
        res, _ = _balance_residual(P, PET, SM, p, 3600.0)
        assert abs(res) < 1e-6, f"balance {P},{PET},{SM}: residual {res:.2e}"


def test_wetter_soil_more_runoff():
    p = CRESTParams(WM=100.0, B=0.4, IM=0.05, KE=1.0, Ksat=10.0).to(torch.zeros(1))
    _, (_, ov_dry, _, _) = _balance_residual(40, 2, 20, p, 3600.0)
    _, (_, ov_wet, _, _) = _balance_residual(40, 2, 90, p, 3600.0)
    assert ov_wet > ov_dry, "a wetter soil must shed more overland flow"


def test_impervious_fraction_adds_runoff():
    base = CRESTParams(WM=120.0, B=0.4, IM=0.0, KE=1.0, Ksat=10.0).to(torch.zeros(1))
    imp = CRESTParams(WM=120.0, B=0.4, IM=0.5, KE=1.0, Ksat=10.0).to(torch.zeros(1))
    _, (_, ov0, _, _) = _balance_residual(30, 2, 30, base, 3600.0)
    _, (_, ov1, _, _) = _balance_residual(30, 2, 30, imp, 3600.0)
    assert ov1 > ov0, "impervious area must increase overland flow"


def test_vectorized_matches_scalar():
    p = CRESTParams(WM=100.0, B=0.4, IM=0.05, KE=1.0, Ksat=10.0)
    P = torch.tensor([[30.0, 5.0], [100.0, 0.0]])
    PET = torch.tensor([[2.0, 8.0], [1.0, 5.0]])
    SM = torch.tensor([[40.0, 60.0], [10.0, 90.0]])
    pv = p.to(P)
    sm, ov, inf, et = crest_step(P, PET, SM, pv, 3600.0)
    for r in range(2):
        for c in range(2):
            s1, o1, i1, e1 = crest_step(P[r, c], PET[r, c], SM[r, c], pv, 3600.0)
            assert abs(sm[r, c] - s1) < 1e-9 and abs(ov[r, c] - o1) < 1e-9


def test_autograd_through_params():
    WM = torch.tensor(100.0, requires_grad=True)
    B = torch.tensor(0.4, requires_grad=True)
    p = CRESTParams(WM=WM, B=B, IM=torch.tensor(0.05), KE=torch.tensor(1.0),
                    Ksat=torch.tensor(10.0))
    P = torch.tensor(40.0)
    _, ov, _, _ = crest_step(P, torch.tensor(2.0), torch.tensor(60.0), p, 3600.0)
    ov.backward()
    assert torch.isfinite(WM.grad) and WM.grad != 0.0, "dOverland/dWM must flow"
    assert torch.isfinite(B.grad), "dOverland/dB must be finite"


def test_coupled_run_on_tilted_plane():
    # synthetic 1 m/km plane, 40x40 @ 30 m; 20 mm/h rain for 3 h then dry
    ny, nx = 40, 40
    import rasterio
    from rasterio.transform import from_origin
    z = (np.arange(ny)[:, None] * 0.03 * np.ones((1, nx)))  # slope south->north
    tr = from_origin(-99.0, 36.0, 0.00027, 0.00027)
    import tempfile, os
    d = tempfile.mkdtemp()
    dem = os.path.join(d, "dem.tif")
    with rasterio.open(dem, "w", driver="GTiff", height=ny, width=nx, count=1,
                       dtype="float64", crs="EPSG:4326", transform=tr) as ds:
        ds.write(z, 1)
    grid = SolverGrid.from_dem(dem)
    t0 = _dt.datetime(2026, 8, 14, 0, 0)
    times = [t0 + _dt.timedelta(hours=k) for k in range(6)]
    rain = weather.uniform_series(times, [20, 20, 20, 0, 0, 0], grid.z.shape, t0,
                                  units="mm/h")
    pet = weather.constant(0.0, grid.z.shape, units="mm/day")
    m = CRESTiMAP(grid, CRESTParams(WM=80.0, B=0.3, IM=0.05, Ksat=8.0),
                  n_manning=0.05, reinfiltration=False)
    res = m.run(rain, pet, t_end_s=6 * 3600, sm0=0.06)  # near-saturated soil
    assert torch.isfinite(res["max_depth"]).all()
    assert res["max_depth"].max() > 0.0, "rain on a near-saturated basin must pond"
    assert (res["sm"] <= 0.080001).all(), "soil water cannot exceed WM"


if __name__ == "__main__":
    for fn in (test_water_balance_closes, test_wetter_soil_more_runoff,
               test_impervious_fraction_adds_runoff, test_vectorized_matches_scalar,
               test_autograd_through_params, test_coupled_run_on_tilted_plane):
        fn()
        print("ok", fn.__name__)
