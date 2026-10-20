"""KW interflow router (EF5 scheme): analytic travel distance on a uniform
slope, mass conservation, injection only on channel cells, gradients."""
import numpy as np
import torch

from crestimap import hydronet as hn
from crestimap.subsurface import KWInterflow, KWInterflowParams, _walk

torch.set_default_dtype(torch.float64)

DX = 30.0
S = 0.01                     # slope
UNDER = 1.0                  # speed = under*sqrt(S) = 0.1 m/s -> 300 s per cell
TAU = DX / (UNDER * np.sqrt(S))
NX = 30


def _strip():
    """One row falling toward j=0; j=0 is the channel (and the outlet)."""
    z = (S * DX * np.arange(NX, dtype=float))[None, :]
    net = hn.flow_network(z, DX, DX, slope_min=1e-6)
    ch = np.zeros((1, NX), bool)
    ch[0, 0] = True
    return net, ch


def test_travel_distance_matches_ef5_rule():
    net, ch = _strip()
    assert np.allclose(net.slope[1:], S)
    T = 3500.0                                   # 11.67 cell crossings
    dst0, w0, dst1, w1 = _walk(net, ch, UNDER, T)
    m = int(np.ceil(T / TAU))                    # 12 crossings
    for k in range(1, NX):
        if (k - 1) * TAU < T:                    # EF5: arrives at the channel
            assert dst0[k] == 0 and w0[k] == 1.0 and w1[k] == 0.0, k
        else:                                    # T runs out between cells
            far, near = k - m, k - m + 1
            frac = (T - (m - 1) * TAU) / TAU
            assert dst0[k] == far and dst1[k] == near, k
            assert np.isclose(w0[k], frac) and np.isclose(w1[k], 1 - frac), k


def _router(net, ch, leaki=0.3, T=3500.0):
    return KWInterflow(net, ch, KWInterflowParams(leaki=leaki, under=UNDER), T)


def test_mass_is_conserved_and_only_channels_receive():
    net, ch = _strip()
    r = _router(net, ch)
    rng = np.random.default_rng(1)
    ir = torch.as_tensor(rng.random((1, NX)) * 0.01)
    inter = torch.as_tensor(rng.random((1, NX)) * 0.002)
    ir = torch.where(torch.as_tensor(ch), torch.zeros_like(ir), ir)
    before = float(ir.sum() + inter.sum())
    total_ch = 0.0
    for _ in range(20):
        ir, to_ch, ex = r.step(ir, inter)
        assert torch.all(to_ch[~torch.as_tensor(ch)] == 0.0)
        assert torch.all(ir[torch.as_tensor(ch)] == 0.0)
        total_ch += float(to_ch.sum())
        assert float(ex) == 0.0, "the outlet is a channel cell here"
    after = float(ir.sum()) + total_ch
    assert np.isclose(after, before + 19 * float(inter.sum()), rtol=1e-12)


def test_leak_past_the_outlet_exits():
    net, _ = _strip()
    none = np.zeros((1, NX), bool)
    none[0, NX - 1] = True                        # channel at the TOP: never reached
    r = _router(net, none, leaki=1.0)
    ir = torch.zeros((1, NX))
    ir[0, 3] = 1.0
    ir2, to_ch, ex = r.step(ir, torch.zeros((1, NX)))
    assert float(ex) == 1.0 and float(to_ch.sum()) == 0.0
    assert float(ir2.sum()) == 0.0


def test_gradient_reaches_leaki():
    net, ch = _strip()
    leaki = torch.tensor(0.2, requires_grad=True)
    r = KWInterflow(net, ch, KWInterflowParams(leaki=leaki, under=UNDER), 3500.0)
    ir = torch.full((1, NX), 0.01)
    _, to_ch, _ = r.step(ir, torch.zeros((1, NX)))
    to_ch.sum().backward()
    assert leaki.grad is not None and float(leaki.grad) > 0.0
