"""HUC12-tiled parallel solver gates (CPU; CRESTIMAP_TEST_DEVICE=cuda or a
comma list "cuda:0,cuda:1" spreads tiles across GPUs).

Gate T1  equivalence: TiledSolver over 1, 3 and 5 tiles reproduces the
         monolithic SWESolver(active=mask) solve on every basin cell
         (bit-identical on CPU: identical stencils, identical dt).
Gate T2  halo economy: ghost cells are a small fraction of owned cells for
         watershed-like units (boundaries are mostly dry ridgelines).
Gate T3  the sink still works per tile: no water outside the basin.
"""
import math
import os
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from crestimap import SWESolver
from crestimap.tiled import TiledSolver, plan_groups

DEV = os.environ.get("CRESTIMAP_TEST_DEVICE", "cpu")
DEVICES = DEV.split(",") if "," in DEV else [DEV]


def _basin(ny=60, nx=90, n_units=6, seed=1):
    """Synthetic 'watershed': a valley DEM sloping to the west, an
    elliptical basin, split into n_units bands along the valley."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:ny, 0:nx]
    z = (0.02 * xx + 0.4 * np.abs(yy - ny / 2) / (ny / 2) * 3.0
         + 0.05 * rng.random((ny, nx)))
    ell = ((xx - nx / 2) / (0.47 * nx)) ** 2 + ((yy - ny / 2) / (0.42 * ny)) ** 2 <= 1.0
    lab = np.full((ny, nx), -1, dtype=np.int16)
    for k in range(n_units):
        sel = ell & (xx >= k * nx / n_units) & (xx < (k + 1) * nx / n_units)
        lab[sel] = k
    return torch.as_tensor(z, dtype=torch.float32), lab


def _forcing(ny, nx, lab, seed=2):
    rng = np.random.default_rng(seed)
    r = torch.as_tensor(rng.random((ny, nx)) * 4e-5, dtype=torch.float32)
    r = torch.where(torch.as_tensor(lab >= 0), r, torch.zeros_like(r))
    return r


def _mono(z, lab, rain, t_end, order):
    act = torch.as_tensor(lab >= 0)
    dev = DEVICES[0]
    s = SWESolver(z.to(dev), dx=30.0, dy=30.0, n_manning=0.05, order=order,
                  bc="open", active=act.to(dev))
    h = torch.zeros_like(z).to(dev)
    # channel pre-wet along the valley centre inside the basin
    h[z.shape[0] // 2, :] = 0.5
    h, qx, qy = s.mask_state(h, torch.zeros_like(h), torch.zeros_like(h))
    rd = rain.to(dev)
    h2, qx2, qy2 = s.run(h, qx, qy, t_end=t_end, rain_fn=lambda t: rd)
    return h2.cpu(), qx2.cpu(), qy2.cpu()


def _tiled(z, lab, rain, t_end, order, n_tiles):
    ts = TiledSolver(z, lab, dx=30.0, dy=30.0, devices=DEVICES, n_manning=0.05,
                     n_tiles=n_tiles, order=order, bc="open", log=print)
    h = torch.zeros_like(z)
    h[z.shape[0] // 2, :] = 0.5
    ts.scatter(h, torch.zeros_like(h), torch.zeros_like(h))
    h2, qx2, qy2 = ts.run(t_end=t_end, rain_fn=lambda t: rain)
    return ts, h2, qx2, qy2


def test_T1_tiled_matches_monolithic():
    z, lab = _basin()
    rain = _forcing(*z.shape, lab)
    for order in (1, 2):
        hm, qxm, qym = _mono(z, lab, rain, t_end=600.0, order=order)
        for n_tiles in (1, 3, 5):
            ts, ht, qxt, qyt = _tiled(z, lab, rain, 600.0, order, n_tiles)
            act = torch.as_tensor(lab >= 0)
            dh = (ht - hm)[act].abs().max().item()
            dq = max((qxt - qxm)[act].abs().max().item(),
                     (qyt - qym)[act].abs().max().item())
            print(f"  order {order} tiles {n_tiles}: max|dh| {dh:.3e}, "
                  f"max|dq| {dq:.3e}, wet {(hm > 0.02).sum().item()}")
            tol = 0.0 if DEVICES == ["cpu"] else 1e-5
            assert dh <= tol and dq <= tol, \
                f"tiled != monolithic (order {order}, {n_tiles} tiles)"


def test_T2_halo_is_small():
    # realistic scale: ~90k basin cells (one HUC12 at 30 m is ~100k), 8 units
    z, lab = _basin(ny=240, nx=360, n_units=8)
    ts = TiledSolver(z, lab, dx=30.0, dy=30.0, devices=DEVICES, n_tiles=4,
                     log=print)
    n_own = int((lab >= 0).sum())
    n_ghost = sum(int(len(r[1])) for t in ts.tiles for r in t.recv)
    frac = n_ghost / n_own
    print(f"  ghost/owned = {n_ghost}/{n_own} = {frac:.3f}")
    assert frac < 0.10, f"halo too large: {frac:.3f}"
    groups = plan_groups(lab, 4)
    assert len(groups) == 4 and sorted(sum(groups, [])) == list(range(8))


def test_T3_no_water_outside_basin():
    z, lab = _basin()
    rain = _forcing(*z.shape, lab) * 5
    ts, ht, _, _ = _tiled(z, lab, rain, 900.0, 2, 3)
    assert ht[torch.as_tensor(lab < 0)].abs().max().item() == 0.0
    md = ts.gather_maxdepth()
    assert md[torch.as_tensor(lab < 0)].abs().max().item() == 0.0
    assert md.max().item() >= ht.max().item()


if __name__ == "__main__":
    for fn in (test_T1_tiled_matches_monolithic, test_T2_halo_is_small,
               test_T3_no_water_outside_basin):
        fn()
        print(f"PASS  {fn.__name__}")
    print("all tiled tests passed")
