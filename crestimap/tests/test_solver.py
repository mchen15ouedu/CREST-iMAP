"""Validation suite for the v2 solver.

Run:  python -m pytest crestimap/tests -q     (or run this file directly)

1. Well-balance (C-property): lake at rest over irregular bathymetry stays
   at rest to machine precision, orders 1 and 2, including dry hills.
2. Stoker dam break: second-order solution matches the analytic profile.
3. Mass conservation: closed basin + lateral inflow gains exactly the
   injected volume.
4. Gradients: d(loss)/d(Manning n) and d(loss)/d(bed scale) from autograd
   match central finite differences.
5. Draining-time limiter: a thin sheet on a slope at cfl 0.45 conserves
   mass (the depth clamp used to create ~10 %), lake at rest over steep
   terrain stays at rest, gradients and checkpointing work through it.
"""
import math
import os
import sys
import pathlib

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from crestimap import SWESolver, stoker_dambreak

torch.set_default_dtype(torch.float64)
# CRESTIMAP_TEST_DEVICE=cuda runs the whole suite on the GPU (P1 gate 1)
DEVICE = os.environ.get("CRESTIMAP_TEST_DEVICE", "cpu")
if DEVICE != "cpu":
    torch.set_default_device(DEVICE)


def _bumpy_bed(ny, nx, amp=0.3):
    y, x = torch.meshgrid(torch.linspace(0, 1, ny), torch.linspace(0, 1, nx),
                          indexing="ij")
    return (amp * torch.sin(3 * math.pi * x) * torch.cos(2 * math.pi * y)
            + 0.2 * amp * torch.sin(9.7 * x + 5.1 * y))


def test_well_balanced_wet():
    for order in (1, 2):
        z = _bumpy_bed(24, 31)
        eta0 = 1.0
        h = torch.clamp(eta0 - z, min=0.0)
        s = SWESolver(z, dx=2.0, dy=2.0, order=order, bc="wall")
        qx = torch.zeros_like(h)
        qy = torch.zeros_like(h)
        h2, qx2, qy2 = s.run(h.clone(), qx, qy, t_end=5.0)
        umax = max(qx2.abs().max().item(), qy2.abs().max().item())
        detamax = ((h2 + z) - eta0).abs().max().item()
        assert umax < 1e-12, f"order {order}: residual momentum {umax:.3e}"
        assert detamax < 1e-12, f"order {order}: surface drift {detamax:.3e}"


def test_well_balanced_dry_hills():
    """Partially dry lake at rest: hills poke above the surface."""
    for order in (1, 2):
        z = _bumpy_bed(24, 31, amp=1.4)  # peaks above eta0
        eta0 = 1.0
        h = torch.clamp(eta0 - z, min=0.0)
        s = SWESolver(z, dx=2.0, dy=2.0, order=order, bc="wall")
        h2, qx2, qy2 = s.run(h.clone(), torch.zeros_like(h), torch.zeros_like(h),
                             t_end=5.0)
        umax = max(qx2.abs().max().item(), qy2.abs().max().item())
        assert umax < 1e-12, f"order {order}: dry-hill residual {umax:.3e}"


def test_stoker_dambreak():
    nx, ny = 800, 4
    L = 40.0
    dx = L / nx
    x = (np.arange(nx) + 0.5) * dx - L / 2
    hl, hr = 1.0, 0.2
    z = torch.zeros(ny, nx)
    h = torch.where(torch.as_tensor(x) < 0, hl, hr).expand(ny, nx).clone()
    s = SWESolver(z, dx=dx, dy=dx, order=2, bc="open", n_manning=0.0, cfl=0.4)
    t_end = 1.5
    h2, qx2, _ = s.run(h, torch.zeros_like(h), torch.zeros_like(h), t_end=t_end)
    href, uref = stoker_dambreak(x, t_end, hl, hr)
    hsim = h2[ny // 2].cpu().numpy()
    rel_l1 = np.abs(hsim - href).mean() / href.mean()
    # middle-state depth (sample well inside the plateau)
    plateau = (x > 1.0) & (x < 3.0)
    hm_err = abs(hsim[plateau].mean() - href[plateau].mean()) / href[plateau].mean()
    assert rel_l1 < 0.03, f"dam break L1 error {rel_l1:.4f}"
    assert hm_err < 0.02, f"middle state error {hm_err:.4f}"


def test_mass_conservation_with_inflow():
    z = _bumpy_bed(20, 20, amp=0.2)
    h = torch.clamp(0.5 - z, min=0.0)
    dx = dy = 5.0
    s = SWESolver(z, dx=dx, dy=dy, order=2, bc="wall")
    # spatially VARYING inflow: drives real dynamics + boundary-adjacent
    # gradients, which is what exposes any wall-face mass leak
    rate = 1e-4 * (1.0 + torch.rand(20, 20))
    rain = lambda t: rate
    v0 = h.sum().item() * dx * dy
    t_end = 20.0
    h2, _, _ = s.run(h, torch.zeros_like(h), torch.zeros_like(h),
                     t_end=t_end, rain_fn=rain)
    v1 = h2.sum().item() * dx * dy
    v_in = rate.sum().item() * t_end * dx * dy
    assert abs((v1 - v0) - v_in) / v_in < 1e-10, \
        f"mass error {(v1 - v0) - v_in:.3e} of {v_in:.3e}"


def _grad_loss(n_scalar, zscale, nsteps=25):
    """Fixed-dt run returning a scalar loss; used for autograd vs FD."""
    ny, nx = 12, 16
    base = _bumpy_bed(ny, nx, amp=0.15)
    z = base * zscale
    h = torch.clamp(0.8 - z, min=0.0) + 0.05
    s = SWESolver(z, dx=2.0, dy=2.0, order=2, bc="wall", n_manning=n_scalar)
    qx = torch.zeros_like(h)
    qy = torch.zeros_like(h)
    # slosh: tilt the surface so there is real dynamics + friction
    h = h + 0.05 * torch.linspace(-1, 1, nx).expand(ny, nx)
    dt = 0.05
    rain = torch.full_like(h, 2e-5)
    for _ in range(nsteps):
        h, qx, qy, _ = s.step(h, qx, qy, dt=dt, rain=rain)
    w = torch.linspace(0.5, 1.5, nx).expand(ny, nx)
    return (h * w).sum()


def test_gradients_match_fd():
    for name, val in (("manning", 0.05), ("zscale", 1.0)):
        p = torch.tensor(val, requires_grad=True)
        if name == "manning":
            loss = _grad_loss(p, torch.tensor(1.0))
        else:
            loss = _grad_loss(torch.tensor(0.05), p)
        loss.backward()
        g_auto = p.grad.item()
        d = 1e-6 if name == "manning" else 1e-6
        args = lambda v: (torch.tensor(v), torch.tensor(1.0)) if name == "manning" \
            else (torch.tensor(0.05), torch.tensor(v))
        lp = _grad_loss(*args(val + d)).item()
        lm = _grad_loss(*args(val - d)).item()
        g_fd = (lp - lm) / (2 * d)
        denom = max(abs(g_fd), 1e-8)
        rel = abs(g_auto - g_fd) / denom
        assert rel < 1e-4, f"{name}: autograd {g_auto:.6e} vs FD {g_fd:.6e} (rel {rel:.2e})"


def test_basin_mask_sink_and_balance():
    """Basin domain (active mask): (a) lake at rest inside an irregular
    basin stays at rest and NO water ever appears outside it; (b) with a
    high divide (mask boundary on dry, high ground) mass inside is exactly
    conserved — the sink only acts where water actually leaves; (c) water
    placed outside the basin is removed on the first step."""
    ny, nx = 30, 40
    z = _bumpy_bed(ny, nx, amp=0.3)
    yy, xx = torch.meshgrid(torch.arange(ny, dtype=torch.get_default_dtype()),
                            torch.arange(nx, dtype=torch.get_default_dtype()),
                            indexing="ij")
    basin = ((xx - nx / 2) / (0.4 * nx)) ** 2 + ((yy - ny / 2) / (0.4 * ny)) ** 2 <= 1.0
    # ring of high ground just outside the basin (the divide)
    z = torch.where(basin, z, z + 5.0)
    eta0 = 1.0
    h = torch.where(basin, torch.clamp(eta0 - z, min=0.0), torch.zeros_like(z))
    s = SWESolver(z, dx=2.0, dy=2.0, order=2, bc="open", active=basin)
    h2, qx2, qy2 = s.run(h.clone(), torch.zeros_like(h), torch.zeros_like(h),
                         t_end=5.0)
    assert h2[~basin].abs().max().item() == 0.0, "water outside the basin"
    assert ((h2 + z) - eta0)[basin & (h > 0)].abs().max().item() < 1e-12
    assert max(qx2.abs().max().item(), qy2.abs().max().item()) < 1e-12
    # (b) inflow inside the basin, divide keeps it in: exact conservation
    rate = torch.where(basin, torch.full_like(z, 1e-4), torch.zeros_like(z))
    v0 = h.sum().item() * 4.0
    h3, _, _ = s.run(h.clone(), torch.zeros_like(h), torch.zeros_like(h),
                     t_end=20.0, rain_fn=lambda t: rate)
    v_in = rate.sum().item() * 20.0 * 4.0
    v1 = h3.sum().item() * 4.0
    assert abs((v1 - v0) - v_in) / v_in < 1e-10, \
        f"masked mass error {(v1 - v0) - v_in:.3e} of {v_in:.3e}"
    # (c) water outside is sunk immediately
    hx = h.clone()
    hx[0, 0] = 3.0
    h4, _, _, _ = s.step(hx, torch.zeros_like(h), torch.zeros_like(h), dt=0.01)
    assert h4[0, 0].item() == 0.0



def _slope_sheet(axis, ny=15, nx=21, d=30.0, slope=0.01):
    """1 cm sheet at rest on a uniform 1 % slope (30 m cells)."""
    i = torch.arange(nx if axis == 1 else ny, dtype=torch.get_default_dtype())
    z = slope * d * i
    z = z.unsqueeze(0).expand(ny, nx) if axis == 1 else z.unsqueeze(1).expand(ny, nx)
    return z.clone(), torch.full((ny, nx), 0.01)


def test_thin_sheet_on_slope_conserves_mass():
    """Regression (2026-10-20): dt comes from the start-of-step state, where
    a sheet at rest only has c = sqrt(gh) (dt ~ 43 s); gravity accelerates
    it by g*S*dt ~ 4 m/s inside the step, the stage-2 Courant number hits
    ~8 and the h >= 0 clamp used to CREATE ~10 % of the water in 30 steps.
    The draining-time limiter must hold mass to round-off at cfl 0.45."""
    for axis in (1, 0):
        z, h0 = _slope_sheet(axis)
        for limit in (True, False):
            s = SWESolver(z, dx=30.0, dy=30.0, n_manning=0.05, order=2,
                          bc="wall", cfl=0.45, drain_limit=limit)
            h, qx, qy = h0.clone(), torch.zeros_like(h0), torch.zeros_like(h0)
            v0 = h.sum().item()
            for _ in range(30):
                h, qx, qy, _ = s.step(h, qx, qy)
            rel = (h.sum().item() - v0) / v0
            if limit:
                assert abs(rel) < 1e-10, f"axis {axis}: mass change {rel:.3e}"
                assert h.min().item() >= 0.0
            else:   # the fixture really exercises the old failure
                assert rel > 0.05, f"axis {axis}: unlimited gain only {rel:.3e}"


def test_well_balanced_steep_terrain():
    """Lake at rest over steep, rough 30 m terrain (cell-to-cell slopes of
    tens of percent, half the domain dry) stays at rest with the limiter
    active, orders 1 and 2, wall and open boundaries and a basin mask."""
    ny, nx = 26, 34
    g = torch.Generator().manual_seed(7)
    y, x = torch.meshgrid(torch.arange(ny, dtype=torch.get_default_dtype()),
                          torch.arange(nx, dtype=torch.get_default_dtype()),
                          indexing="ij")
    z = (1.2 * x + 8.0 * torch.sin(0.9 * y) * torch.cos(0.7 * x)
         + 6.0 * torch.rand(ny, nx, generator=g))
    eta0 = 20.0
    h = torch.clamp(eta0 - z, min=0.0)
    assert 0.2 < (h > 0).double().mean().item() < 0.8
    basin = ((x - nx / 2) / (0.45 * nx)) ** 2 + ((y - ny / 2) / (0.45 * ny)) ** 2 <= 1.0
    # the masked run needs a divide: a lake touching the sink is not at rest
    z_div = torch.where(basin, z, z + 60.0)
    for order in (1, 2):
        for bc, active, zz in (("wall", None, z), ("open", basin, z_div)):
            s = SWESolver(zz, dx=30.0, dy=30.0, order=order, bc=bc,
                          active=active, n_manning=0.05)
            h0 = torch.clamp(eta0 - zz, min=0.0)
            wet = h0 > 0
            h2, qx2, qy2 = s.run(h0.clone(), torch.zeros_like(h), torch.zeros_like(h),
                                 t_end=600.0)
            umax = max(qx2.abs().max().item(), qy2.abs().max().item())
            deta = ((h2 + zz) - eta0)[wet].abs().max().item()
            assert umax < 1e-12, f"order {order} {bc}: momentum {umax:.3e}"
            assert deta < 1e-12, f"order {order} {bc}: surface drift {deta:.3e}"
            assert (h2[~wet] == 0).all(), f"order {order} {bc}: dry cells wetted"


def test_gradients_through_drain_limiter():
    """Autograd matches FD on the slope-sheet case at a fixed 43 s dt,
    where the limiter is active (theta < 1) in most cells."""
    z, h0 = _slope_sheet(1, ny=8, nx=12)

    def loss(n, limit=True):
        s = SWESolver(z, dx=30.0, dy=30.0, n_manning=n, order=2, bc="wall",
                      drain_limit=limit)
        h, qx, qy = h0.clone(), torch.zeros_like(h0), torch.zeros_like(h0)
        for _ in range(8):
            h, qx, qy, _ = s.step(h, qx, qy, dt=43.0)
        return (h * torch.linspace(0.5, 1.5, h.shape[1])).sum(), h

    # the fixture really needs the limiter: unlimited, the clamp adds water
    _, hu = loss(torch.tensor(0.05), limit=False)
    assert hu.sum().item() > 1.01 * h0.sum().item()
    p = torch.tensor(0.05, requires_grad=True)
    loss(p)[0].backward()
    d = 1e-6
    g_fd = (loss(torch.tensor(0.05 + d))[0].item() - loss(torch.tensor(0.05 - d))[0].item()) / (2 * d)
    rel = abs(p.grad.item() - g_fd) / max(abs(g_fd), 1e-8)
    assert torch.isfinite(p.grad), "non-finite gradient through the limiter"
    assert rel < 1e-4, f"autograd {p.grad.item():.6e} vs FD {g_fd:.6e} (rel {rel:.2e})"


def test_checkpoint_and_dt_every_with_limiter():
    """checkpoint_every and dt_every keep working: identical forward state
    and gradient with and without activation checkpointing."""
    z, h0 = _slope_sheet(1, ny=8, nx=12)

    def run(ck):
        n = torch.tensor(0.05, requires_grad=True)
        s = SWESolver(z, dx=30.0, dy=30.0, n_manning=n, order=2, bc="wall")
        h = h0.clone().requires_grad_(True)
        h2, _, _ = s.run(h, torch.zeros_like(h0), torch.zeros_like(h0),
                         t_end=600.0, checkpoint_every=ck, dt_every=3)
        h2.sum().backward()
        return h2.detach(), n.grad.item()

    ha, ga = run(0)
    hb, gb = run(4)
    assert torch.equal(ha, hb)
    assert abs(ga - gb) <= 1e-12 * max(1.0, abs(ga))
    assert abs(ha.sum().item() - h0.sum().item()) / h0.sum().item() < 1e-10


if __name__ == "__main__":
    for fn in (test_well_balanced_wet, test_well_balanced_dry_hills,
               test_stoker_dambreak, test_mass_conservation_with_inflow,
               test_gradients_match_fd, test_basin_mask_sink_and_balance,
               test_thin_sheet_on_slope_conserves_mass,
               test_well_balanced_steep_terrain,
               test_gradients_through_drain_limiter,
               test_checkpoint_and_dt_every_with_limiter):
        fn()
        print(f"PASS  {fn.__name__}")
    print("all tests passed")
