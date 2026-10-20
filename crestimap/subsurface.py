"""Interflow routing to the channel — the CREST v4 / EF5 kinematic-wave
subsurface scheme (EF5 KinematicRoute.cpp, RouteInt + InitializeRouting),
ported to vectorized torch on a CREST-iMAP grid.

Per hydrologic step of length T, on every active NON-channel cell:

    IR   += interflow                (CREST interflow of the cell)
    leak  = IR * leaki ;  IR -= leak

and the leak travels down the D8 network within the same step: each cell
takes  len / (under * sqrt(slope))  seconds to cross. Walking downstream,
the leak stops at the first channel cell, or, when T runs out between two
non-channel cells, is split between them in proportion to where T ran out
(the farther cell gets (T - t_prev) / (t_cur - t_prev)). Leak that walks off
the domain (past the basin outlet) exits. Leak landing on a non-channel
cell joins that cell's IR at the end of the step (it leaks again next
step); leak reaching a channel cell, and a channel cell's own interflow,
are injected into the 2-D solver there as lateral inflow over the step.

The two destinations and weights depend only on the DEM, the channel mask,
`under` and T, so they are precomputed once; each step is two index_add_.
Gradients reach `leaki` (and the IR state); the routing pattern (`under`,
slope, channel mask) is fixed — it is discrete.

Deliberate differences from EF5 v4.3 (user fork):
* Leak arriving at a channel cell is injected. In the fork, leaks are added
  to the target's BASEFLOW slot, which the channel branch never reads and
  then zeroes, so subsurface water reaching a channel appears to be lost.
* No baseflow / groundwater bucket (CREST v4's third layer) — not ported.
* Slope floor: `slope_min` (see hydronet.flow_network) instead of EF5's
  1 m minimum drop per cell.
* Equal-area cells on the solver grid, so EF5's area ratio is 1.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import torch


@dataclasses.dataclass
class KWInterflowParams:
    """EF5 kinematic-wave subsurface parameters (scalars or (ny, nx)).

    leaki : fraction of the interflow reservoir leaked per hydrologic step
            [0..1] (EF5 clamps to this range). EF5 is hourly, so values
            calibrated there mean per hour: keep dt_hydro_h = 1 to reuse them.
    under : interflow speed multiplier; speed = under * sqrt(slope) [m/s]
    Defaults are CREST-AI's control defaults (under=1.0, leaki=0.1).
    """
    leaki: object = 0.1
    under: object = 1.0


def _walk(net, channel, under, T):
    """EF5 InitializeRouting for every cell at once (numpy).

    Returns dst0, w0, dst1, w1 as flat arrays over all cells: destination
    indices (-1 = leaves the domain) and fractions of the cell's leak.
    """
    n = net.ds.size
    ds = net.ds
    under = np.broadcast_to(np.asarray(under, float), net.shape).ravel()
    speed = np.maximum(under, 1e-12) * np.sqrt(net.slope)
    nex = net.length / speed                         # seconds to cross
    ch = channel.ravel()

    cur = np.arange(n)                               # currentNode
    prev = np.full(n, -1)                            # previousNode
    t_cur = np.zeros(n)
    t_prev = np.zeros(n)
    walking = net.active.ravel() & ~ch
    while walking.any():
        idx = np.flatnonzero(walking)
        c = cur[idx]
        t_prev[idx] = t_cur[idx]
        prev[idx] = c
        t_cur[idx] = t_cur[idx] + nex[c]
        cur[idx] = ds[c]                             # -1 when it leaves
        nc = cur[idx]
        still = (t_cur[idx] < T) & (nc >= 0)
        still[still] = ~ch[nc[still]]
        walking[idx] = still
    w0 = np.ones(n)
    w1 = np.zeros(n)
    mid = (cur >= 0) & ~ch[np.maximum(cur, 0)] & ((t_cur - t_prev) > 0)
    w0[mid] = (T - t_prev[mid]) / (t_cur[mid] - t_prev[mid])
    w1[mid] = 1.0 - w0[mid]
    dst0 = np.where(cur >= 0, cur, -1)
    dst1 = np.where(w1 > 0, prev, -1)
    return dst0, w0, dst1, w1


class KWInterflow:
    """Precomputed interflow router for one grid / channel mask / step."""

    def __init__(self, net, channel, params: KWInterflowParams | None, dt_s,
                 device=None, dtype=torch.float64):
        p = params or KWInterflowParams()
        channel = np.asarray(channel, bool) & net.active
        self.shape = net.shape
        self.dt_s = float(dt_s)
        dst0, w0, dst1, w1 = _walk(net, channel, p.under, self.dt_s)
        src = np.flatnonzero(net.active.ravel() & ~channel.ravel())
        kw = dict(device=device)
        self.src = torch.as_tensor(src, dtype=torch.long, **kw)
        self.dst0 = torch.as_tensor(dst0[src], dtype=torch.long, **kw)
        self.dst1 = torch.as_tensor(dst1[src], dtype=torch.long, **kw)
        self.w0 = torch.as_tensor(w0[src], dtype=dtype, **kw)
        self.w1 = torch.as_tensor(w1[src], dtype=dtype, **kw)
        self.channel = torch.as_tensor(channel, **kw)
        self.active = torch.as_tensor(net.active, **kw)
        self.source = self.active & ~self.channel
        leaki = torch.as_tensor(p.leaki, dtype=dtype, device=device)
        self.leaki = torch.clamp(leaki, 0.0, 1.0)

    def step(self, ir, interflow):
        """One hydrologic step. ir, interflow : (ny, nx) depths [m].

        Returns (ir_new, to_channel, exited): to_channel is the (ny, nx)
        depth [m] to inject on channel cells over this step; exited is the
        depth-sum [m·cells] that left the domain underground.
        """
        zero = torch.zeros_like(interflow)
        ir = torch.where(self.source, ir + interflow, zero)
        leak = ir * self.leaki
        ir = ir - leak
        flat = leak.reshape(-1)[self.src]
        arrive = torch.zeros(ir.numel(), dtype=ir.dtype, device=ir.device)
        exited = torch.zeros((), dtype=ir.dtype, device=ir.device)
        for dst, w in ((self.dst0, self.w0), (self.dst1, self.w1)):
            ok = dst >= 0
            amt = flat * w
            arrive = arrive.index_add(0, dst[ok], amt[ok])
            exited = exited + amt[~ok].sum()
        arrive = arrive.reshape(self.shape)    # dst1 = -1 only where w1 = 0
        own = torch.where(self.channel, interflow, zero)
        to_channel = torch.where(self.channel, arrive, zero) + own
        ir = ir + torch.where(self.source, arrive, zero)
        return ir, to_channel, exited
