"""Re-infiltration and water-budget tests for the weather-driven CRESTiMAP.

Standing water re-enters the CREST balance each hydrologic step (v1
behaviour), so a closed pond loses water to infiltration and ET; the state
handed to the solver carries no momentum on cells the exchange dried; and
surface + soil + interflow store + ET + sink outflow closes against rain,
with interflow routed or held in its own store.

Each simulation runs once per session (`functools.lru_cache`) and is shared
by every test that reads it. The solver is instrumented per instance:
`solver.run` records the state it is handed each hydrologic step and
`solver.mask_state` measures the water the basin mask sinks, so the budget
is checked against an outflow the model does not compute itself."""
import datetime as _dt
import functools

import numpy as np
import torch
from rasterio.transform import from_origin
from rasterio.crs import CRS

from crestimap import CRESTiMAP, weather
from crestimap.crest import CRESTParams
from crestimap.forcing import SolverGrid

torch.set_default_dtype(torch.float64)

DX = 30.0
CELL = DX * DX
TR = from_origin(500000.0, 4000000.0, DX, DX)
CRS_ = CRS.from_epsg(32615)
TOL = 1e-9


def _instrument(m):
    """Record every state handed to solver.run and the volume [m3] the mask
    sinks, without touching the solver module."""
    solver = m.solver
    handed, returned, sunk = [], [], [0.0]
    run, mask_state = solver.run, solver.mask_state

    def run_spy(h, qx, qy, *a, **k):
        handed.append((h.clone(), qx.clone(), qy.clone()))
        out = run(h, qx, qy, *a, **k)
        returned.append(tuple(x.clone() for x in out))
        return out

    def mask_spy(h, qx, qy):
        if solver.active is not None:
            sunk[0] += float(h[~solver.active].sum()) * CELL
        return mask_state(h, qx, qy)

    solver.run, solver.mask_state = run_spy, mask_spy
    return handed, returned, sunk


# ---------------------------------------------------------------- closed pond
NB = 31


def _bowl():
    """Closed parabolic depression (walls all round) with a 0.5 m pond."""
    i, j = np.mgrid[0:NB, 0:NB]
    z = 0.002 * ((i - NB // 2) ** 2 + (j - NB // 2) ** 2)
    h0 = np.clip(0.5 - z, 0.0, None)
    return SolverGrid.from_array(z, TR, CRS_), torch.as_tensor(h0)


@functools.lru_cache(maxsize=None)
def _pond(reinfiltration, hours):
    grid, h0 = _bowl()
    zero = weather.constant(0.0, grid.z.shape, units="mm/h")
    pet = weather.constant(5.0, grid.z.shape, units="mm/day")
    m = CRESTiMAP(grid, CRESTParams(WM=80.0, B=0.3, IM=0.0, Ksat=5.0, KE=1.0),
                  n_manning=0.05, reinfiltration=reinfiltration)
    handed, returned, sunk = _instrument(m)
    res = m.run(zero, pet, t_end_s=hours * 3600.0, sm0=0.02, h0=h0)
    return dict(h0=h0, sm0=0.02, res=res, handed=handed, returned=returned,
                sunk=sunk[0])


def test_pond_persists_without_reinfiltration():
    r = _pond(False, 6)
    v0, v1 = float(r["h0"].sum()) * CELL, float(r["res"]["h"].sum()) * CELL
    assert abs(v1 - v0) / v0 < 1e-6, "one-way coupling: a closed pond cannot lose water"


def test_reinfiltration_drains_closed_pond():
    r = _pond(True, 24)
    v0, v1 = float(r["h0"].sum()) * CELL, float(r["res"]["h"].sum()) * CELL
    assert v1 < 0.97 * v0, "re-infiltration must remove ponded water"


def test_no_momentum_handed_to_solver_on_dried_cells():
    # The solver zeroes q where h <= eps after its own step, so the FINAL
    # state cannot show this; the state handed to solver.run each step can.
    r = _pond(True, 24)
    assert len(r["handed"]) == 24
    dried = 0
    for k, (h, qx, qy) in enumerate(r["handed"]):
        dry = h <= 0.0
        assert torch.all(qx[dry] == 0.0) and torch.all(qy[dry] == 0.0), k
        if k:
            prev_h = r["handed"][k - 1][0]
            dried += int((dry & (prev_h > 0.0)).sum())
    assert dried > 0, "the exchange must actually dry some cells"


def test_exchange_keeps_velocity_on_wet_cells():
    # momentum scales with depth: on cells wet both before and after the
    # exchange, the velocity handed to the solver equals the velocity the
    # solver returned the step before
    r = _pond(True, 24)
    checked = 0
    for k in range(1, len(r["handed"])):
        h0, qx0, qy0 = r["returned"][k - 1]
        h1, qx1, qy1 = r["handed"][k]
        wet = (h0 > 1e-3) & (h1 > 1e-3)
        for q0, q1 in ((qx0, qx1), (qy0, qy1)):
            u0, u1 = q0[wet] / h0[wet], q1[wet] / h1[wet]
            assert torch.allclose(u1, u0, rtol=1e-10, atol=1e-14), k
        checked += int(wet.sum())
    assert checked > 0
    assert any(not torch.equal(r["handed"][k][0], r["returned"][k - 1][0])
               for k in range(1, len(r["handed"]))), "the exchange must change h"


def test_pond_budget_closes_with_et():
    r = _pond(True, 24)
    res = r["res"]
    before = float(r["h0"].sum()) + r["sm0"] * NB * NB
    after = float((res["h"] + res["sm"] + res["interflow"]).sum()
                  + res["budget"]["et"].sum() - res["budget"]["precip"].sum())
    assert float(res["budget"]["et"].sum()) > 0.0, "PET must act"
    assert r["sunk"] == 0.0
    assert abs(after - before) / before < TOL, (before, after)


# ------------------------------------------- rain on a plane draining to a sink
NY, NX = 15, 21
HOURS = 6


def _plane():
    """0.1 % plane falling toward column 0; column 0 lies outside the basin
    mask, so everything that reaches it leaves through the sink.

    Kept mild on purpose: at order 2 / CFL 0.45 a thin sheet on a 1 % slope
    goes negative in the second RK stage and the h >= 0 clamp in
    SWESolver.step creates ~10 % mass in 30 steps. That is a solver defect,
    not a coupling one; this test checks the coupling's books."""
    _, j = np.mgrid[0:NY, 0:NX]
    z = 0.001 * DX * j.astype(float)
    active = np.ones((NY, NX), bool)
    active[:, 0] = False
    return SolverGrid.from_array(z, TR, CRS_), torch.as_tensor(active)


@functools.lru_cache(maxsize=None)
def _rain(route_interflow, interflow_k_h=None):
    grid, active = _plane()
    t0 = _dt.datetime(2026, 8, 14)
    times = [t0 + _dt.timedelta(hours=k) for k in range(HOURS)]
    rain = weather.uniform_series(times, [30, 30, 30, 0, 0, 0], grid.z.shape,
                                  t0, units="mm/h")
    pet = weather.constant(6.0, grid.z.shape, units="mm/day")
    m = CRESTiMAP(grid, CRESTParams(WM=80.0, B=0.3, IM=0.05, Ksat=8.0, KE=1.0),
                  n_manning=0.05, active=active, reinfiltration=True,
                  route_interflow=route_interflow, interflow_k_h=interflow_k_h)
    handed, _, sunk = _instrument(m)
    sm0 = 0.06
    res = m.run(rain, pet, t_end_s=HOURS * 3600.0, sm0=sm0)
    return dict(active=active, sm0=sm0, res=res, handed=handed, sunk=sunk[0])


def _plane_budget(r):
    a, res = r["active"], r["res"]
    n = int(a.sum())
    before = r["sm0"] * n * CELL
    store = (res["h"] + res["sm"] + res["interflow"])[a].sum() * CELL
    et = res["budget"]["et"][a].sum() * CELL
    p = res["budget"]["precip"][a].sum() * CELL
    after = float(store + et - p) + r["sunk"]
    return before, after, float(p)


def _assert_closes(r):
    before, after, p = _plane_budget(r)
    assert r["sunk"] > 0.01 * p, "water must leave through the mask sink"
    assert float(r["res"]["budget"]["et"].sum()) > 0.0
    assert abs(after - before) / (before + p) < TOL, (before, after, p)


def test_budget_closes_rain_pet_mask_sink():
    _assert_closes(_rain(True))
    assert float(_rain(True)["res"]["interflow"].abs().max()) == 0.0


def test_budget_closes_interflow_held():
    r = _rain(False)
    assert float(r["res"]["interflow"][r["active"]].sum()) > 0.0, \
        "route_interflow=False must keep interflow in its store"
    _assert_closes(r)
    # held interflow is not routed: less leaves than when it is routed
    assert r["sunk"] < _rain(True)["sunk"]


def test_budget_closes_interflow_reservoir():
    r = _rain(False, 2.0)
    held = _rain(False)
    a = r["active"]
    s, s_held = float(r["res"]["interflow"][a].sum()), float(held["res"]["interflow"][a].sum())
    assert 0.0 < s < s_held, "the linear reservoir must release part of the store"
    _assert_closes(r)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
