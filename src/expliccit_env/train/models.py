"""Architectures by name, and the stages around them inside a policy graph.

A policy graph is: message -> Scaler (columns, non-finite values, scaling)
-> net1 -> LCF beliefs (scaled) -> net2 -> a. Everything a deployed policy
needs is a buffer or a parameter, so the exported ONNX graph is complete.

The "expliccit-v1" modules use expliCCit's layer layout and parameter names,
so its released checkpoints load into them unchanged.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

type F32 = npt.NDArray[np.float32]
_BIG = 3.0e38


class Scaler(nn.Module):
    """x -> (clip(x, lo, hi) - center) / spread on chosen columns, with any
    non-finite value replaced by 0 first (as expliCCit's environment does)."""

    cols: torch.Tensor
    lo: torch.Tensor
    hi: torch.Tensor
    center: torch.Tensor
    spread: torch.Tensor

    def __init__(self, cols: tuple[int, ...], lo: F32, hi: F32, center: F32, spread: F32) -> None:
        super().__init__()
        self.register_buffer("cols", torch.tensor(cols, dtype=torch.long))
        for name, v in (("lo", lo), ("hi", hi), ("center", center), ("spread", spread)):
            self.register_buffer(name, torch.tensor(v, dtype=torch.float32))

    @staticmethod
    def fit(cols: tuple[int, ...], x: F32, kind: Literal["robust", "minmax", "standard"]) -> Scaler:
        """Fit on rows x (already restricted to ``cols``); non-finite values
        are ignored, and a column without any finite value maps to itself."""
        # An all-NaN column is expected (a label undefined on these rows):
        # it gets centre 0 and spread 1 below, so numpy's warnings are noise.
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            x64 = np.where(np.isfinite(x), x, np.nan).astype(np.float64)
            n = x64.shape[1]
            lo, hi = np.full(n, -_BIG), np.full(n, _BIG)
            if kind == "minmax":  # sklearn's MinMaxScaler, feature_range (0, 1)
                center = np.nanmin(x64, axis=0)
                spread = np.nanmax(x64, axis=0) - center
            else:
                if kind == "robust":
                    lo, hi = np.nanquantile(x64, 0.001, axis=0), np.nanquantile(x64, 0.999, axis=0)
                clipped = np.clip(x64, lo, hi)
                center, spread = np.nanmean(clipped, axis=0), np.nanstd(clipped, axis=0)
        center = np.where(np.isfinite(center), center, 0.0)
        spread = np.where(np.isfinite(spread) & (spread > 0), spread, 1.0)
        lo, hi = np.where(np.isfinite(lo), lo, -_BIG), np.where(np.isfinite(hi), hi, _BIG)
        f = np.float32
        return Scaler(cols, lo.astype(f), hi.astype(f), center.astype(f), spread.astype(f))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.index_select(1, self.cols)
        # |x| < 3e38 is false for NaN and both infinities, and exports as
        # Abs, Less, Where, which tract runs (IsInf and IsNaN do not analyse).
        x = torch.where(torch.abs(x) < _BIG, x, torch.zeros_like(x))
        return (torch.minimum(torch.maximum(x, self.lo), self.hi) - self.center) / self.spread

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.spread + self.center


class _Residual(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear1, self.bn1 = nn.Linear(width, width), nn.BatchNorm1d(width)
        self.linear2, self.bn2 = nn.Linear(width, width), nn.BatchNorm1d(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.bn2(self.linear2(torch.relu(self.bn1(self.linear1(x)))))
        return torch.relu(h + x)


class ExpliccitNet1(nn.Module):
    """expliCCit's encoder: a widening layer, two residual blocks, a tapering
    stack (w, w/2, w/4, w/8 with batch norm), one linear head per LCF."""

    def __init__(
        self, inputs: int, lcfs: int, width: int = 512, dropout: float = 0.3, per_lcf: int = 1
    ) -> None:
        super().__init__()
        w = width

        def layer(a: int, b: int, drop: bool) -> list[nn.Module]:
            return [
                nn.Linear(a, b),
                nn.BatchNorm1d(b),
                nn.ReLU(),
                *([nn.Dropout(dropout)] if drop else []),
            ]

        self.initial = nn.Sequential(*layer(inputs, w, True))
        self.res1, self.res2 = _Residual(w), _Residual(w)
        self.hidden = nn.Sequential(
            *layer(w, w // 2, True), *layer(w // 2, w // 4, True), *layer(w // 4, w // 8, False)
        )
        self.heads = nn.ModuleList([nn.Linear(w // 8, per_lcf) for _ in range(lcfs)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.hidden(self.res2(self.res1(self.initial(x))))
        return torch.cat([head(h) for head in self.heads], dim=1)


def _softplus(x: torch.Tensor) -> torch.Tensor:
    # relu(x) + log(1 + exp(-|x|)): stable, and only ops tract runs.
    return torch.relu(x) + torch.log(1 + torch.exp(-torch.abs(x)))


class IntervalNet1(nn.Module):
    """net1 reporting what the observations support: for each LCF a centre
    and an interval. The base network gives three numbers per LCF (m, a, b);
    the bounds are m - softplus(a) - c and m + softplus(b) + c, so they never
    cross the centre, and c is the conformal correction fitted on held-out
    paths. Output: the K centres, then the K lower bounds, then the K upper
    bounds, all in the LCF scaler's units."""

    widen: torch.Tensor

    def __init__(self, base: nn.Module, lcfs: int) -> None:
        super().__init__()
        self.base, self.k = base, lcfs
        self.register_buffer("widen", torch.zeros(lcfs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.base(x).reshape(-1, self.k, 3)
        m, a, b = o[:, :, 0], o[:, :, 1], o[:, :, 2]
        lo = torch.minimum(m - _softplus(a) - self.widen, m)
        hi = torch.maximum(m + _softplus(b) + self.widen, m)
        return torch.cat([m, lo, hi], dim=1)


class ExpliccitActor(nn.Module):
    """LCF beliefs -> a in [-1, 1] (two hidden ReLU layers, tanh)."""

    def __init__(self, lcfs: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(lcfs, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Tanh(),
        )  # fmt: skip

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


class Critic(nn.Module):
    """(LCF beliefs, a) -> Q."""

    def __init__(self, lcfs: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(lcfs + 1, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )  # fmt: skip

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, a], dim=1)).squeeze(1)


class Value(nn.Module):
    """LCF beliefs -> V (implicit Q-learning's state value)."""

    def __init__(self, lcfs: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(lcfs, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )  # fmt: skip

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s).squeeze(1)


NET1 = {"expliccit-v1": ExpliccitNet1}
NET2 = {"expliccit-v1": ExpliccitActor}


class PolicyGraph(nn.Module):
    """The deployed policy under contract sage-v1: message -> a.

    With ``as_deployed`` it repeats what expliCCit's deployed controller does
    between the networks (beliefs back to physical units, clamped at 0, the
    first six rounded to integers, scaled again) and rounds a to three
    decimals before the guest exponentiates it; otherwise net2 sees net1's
    output directly, as in training."""

    def __init__(
        self,
        scaler: Scaler,
        net1: nn.Module,
        lcf_scaler: Scaler,
        net2: nn.Module,
        action_max: float,
        *,
        as_deployed: bool = False,
    ) -> None:
        super().__init__()
        self.scaler, self.net1, self.lcf_scaler, self.net2 = scaler, net1, lcf_scaler, net2
        self.action_max = action_max
        self.as_deployed = as_deployed
        # Round a to this many decimals (as_deployed implies 3).
        self.action_decimals: int | None = None

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        z = self.net1(self.scaler(obs))
        if self.as_deployed:
            c = torch.relu(self.lcf_scaler.inverse(z))
            c = torch.cat([torch.round(c[:, :6]), c[:, 6:]], dim=1)
            z = (c - self.lcf_scaler.center) / self.lcf_scaler.spread
        a = self.net2(z) * self.action_max
        decimals = 3 if self.as_deployed else self.action_decimals
        if decimals is None:
            return a
        scale = float(10**decimals)
        return torch.round(a * scale) / scale
