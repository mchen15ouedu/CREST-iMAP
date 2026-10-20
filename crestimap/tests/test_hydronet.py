"""D8 flow network from a DEM: one outlet, acyclic, full accumulation,
depressions drained, basin mask respected, numba == pure python."""
import numpy as np

from crestimap import hydronet as hn

DX = 30.0
CELL_KM2 = DX * DX / 1e6


def _valley():
    i, j = np.mgrid[0:21, 0:31]
    z = 0.05 * np.abs(i - 10) + 0.01 * j          # thalweg row 10, falls to j=0
    z[10, 15] -= 2.0                                # a pit on the thalweg
    z[10, 20] += 3.0                                # an embankment across it
    return z


def _path(net, c):
    out = []
    while c >= 0:
        out.append(c)
        c = net.ds[c]
        assert len(out) <= net.ds.size, "cycle"
    return out


def test_single_outlet_and_full_accumulation():
    z = _valley()
    net = hn.flow_network(z, DX, DX)
    assert (net.ds < 0).sum() == 1
    out = int(np.flatnonzero(net.ds < 0)[0])
    assert out == 10 * 31 + 0, "the outlet is the low end of the thalweg"
    assert np.isclose(net.acc_km2.max(), z.size * CELL_KM2)
    assert np.isclose(net.acc_km2.ravel()[out], z.size * CELL_KM2)


def test_every_cell_reaches_the_outlet_through_pit_and_embankment():
    z = _valley()
    net = hn.flow_network(z, DX, DX)
    for c in range(z.size):
        assert net.ds[_path(net, c)[-1]] < 0
    # the pit is filled for directions only
    assert net.zf[10, 15] > z[10, 15]
    assert np.all(net.zf >= z)


def test_order_puts_upstream_first():
    net = hn.flow_network(_valley(), DX, DX)
    pos = np.empty(net.ds.size, int)
    pos[net.order] = np.arange(net.order.size)
    has = net.ds >= 0
    assert np.all(pos[has] < pos[net.ds[has]])


def test_basin_mask_drains_through_its_lowest_edge_only():
    z = _valley()
    i, j = np.mgrid[0:21, 0:31]
    act = (np.abs(i - 10) + 0.6 * np.abs(j - 15)) < 12
    net = hn.flow_network(z, DX, DX, active=act)
    exits = np.flatnonzero((net.ds < 0) & act.ravel())
    assert exits.size == 1
    assert np.isclose(net.acc_km2.max(), act.sum() * CELL_KM2)
    assert np.all(net.ds[~act.ravel()] < 0)
    assert np.all(net.acc_km2[~act] == 0)
    ok = net.ds >= 0
    assert act.ravel()[net.ds[ok]].all(), "flow never enters inactive cells"


def test_flat_drains_without_cycles():
    z = np.zeros((15, 15))
    z[7, 0] = -1.0                                 # one low cell on the edge
    net = hn.flow_network(z, DX, DX)
    for c in range(z.size):
        assert net.ds[_path(net, c)[-1]] < 0


def test_numba_matches_python():
    z = _valley()
    ny, nx = z.shape
    act = np.ones(z.size, bool)
    seed = np.zeros((ny, nx), bool)
    seed[0, :] = seed[-1, :] = seed[:, 0] = seed[:, -1] = True
    a = hn._flood(z.ravel().copy(), act, seed.ravel(), ny, nx)
    b = hn._flood_py(z.ravel().copy(), act, seed.ravel(), ny, nx)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_accumulate_weights():
    net = hn.flow_network(_valley(), DX, DX)
    ones = np.ones(net.shape)
    assert np.allclose(net.accumulate(ones) * CELL_KM2, net.acc_km2)
