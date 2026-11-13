"""The networks by name: net1's encoders, net2's offline actor and
critics, and the scaler in front of them. graph.py assembles them into a
policy; beliefs.py reads net1's output.

The "expliccit-v1" modules use expliCCit's layer layout and parameter names,
so its released checkpoints load into them unchanged.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
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
        # One column at a time: float64 copies of the whole table (several
        # at once, for the quantiles and the clipped values) took 9 GB on
        # net1 v6's 4.4 million training rows and ran a 16 GB runner out.
        n = x.shape[1]
        lo, hi = np.full(n, -_BIG), np.full(n, _BIG)
        center, spread = np.zeros(n), np.ones(n)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for j in range(n):
                c = x[:, j].astype(np.float64)
                c[~np.isfinite(c)] = np.nan
                if kind == "minmax":  # sklearn's MinMaxScaler, feature_range (0, 1)
                    center[j] = np.nanmin(c)
                    spread[j] = np.nanmax(c) - center[j]
                    continue
                if kind == "robust":
                    lo[j], hi[j] = np.nanquantile(c, 0.001), np.nanquantile(c, 0.999)
                np.clip(c, lo[j], hi[j], out=c)
                center[j], spread[j] = np.nanmean(c), np.nanstd(c)
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


class BinNet1(nn.Module):
    """net1 reporting, for each LCF, a probability distribution over fixed
    bins of that quantity, in report units. ``bounds[j]`` holds LCF j's
    B + 1 bin boundaries: bin i is [bounds[j, i], bounds[j, i + 1]). For
    classifying a value, the first bin also takes everything below and the
    last everything above; the outer boundaries only serve to read
    quantiles. Output: the K distributions in LCF order, each summing to 1.
    With several members (an ensemble, each trained on its own), the
    distribution is their average, which widens where they disagree.
    """

    bounds: torch.Tensor

    def __init__(self, members: Sequence[nn.Module], lcfs: int, bins: int) -> None:
        super().__init__()
        self.members = nn.ModuleList(members)
        self.k, self.b = lcfs, bins
        self.register_buffer("bounds", torch.zeros(lcfs, bins + 1))

    def _probs(self, x: torch.Tensor) -> torch.Tensor:
        p = [torch.softmax(m(x).reshape(-1, self.k, self.b), dim=2) for m in self.members]
        return p[0] if len(p) == 1 else torch.stack(p).mean(dim=0)

    def log_probs(self, x: torch.Tensor) -> torch.Tensor:
        """[n, K, B] log-probabilities."""
        if len(self.members) == 1:
            return torch.log_softmax(self.members[0](x).reshape(-1, self.k, self.b), dim=2)
        return torch.log(self._probs(x).clamp(min=1e-12))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._probs(x).reshape(-1, self.k * self.b)

    def bin_of(self, y: torch.Tensor) -> torch.Tensor:
        """[n, K] bin index of each value: the number of inner boundaries at
        or below it. A NaN's bin means nothing (mask it)."""
        inner = self.bounds[:, 1:-1].contiguous()
        return torch.searchsorted(inner, y.t().contiguous(), right=True).t()


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
