"""Mahak's active-learning driver (``mahak_online.py``) with Tier 1 as its oracle.

Mahak (HotNets'23) maps one metric of a congestion control over a grid of
(buffer, mRTT, bandwidth, capacity step), asking its oracle for the point
where a Gaussian process is least certain. This is a port of the driver:

- each axis is ``np.arange(min, max + step, step)``, so it can pass ``max``,
  and the step axis is rounded to two decimals, as Mahak formats it;
- features are the grid min-max scaled, then rotated by a full PCA;
- the model is scikit-learn's GaussianProcessRegressor with
  RBF(length_scale=0.1, bounds (1e-60, 1e5)), 10 optimizer restarts,
  ``normalize_y`` and ``random_state=100``, refitted on every label so far;
- the first point is drawn by ``choice`` on a legacy RandomState (Mahak calls
  the global ``np.random.choice``, so seed s here is Mahak under
  ``np.random.seed(s)``), then ``budget`` points each at the largest
  predictive standard deviation (first index on ties);
- the map is the posterior mean at every grid point.

One deliberate difference: Mahak's oracle crashes on a failed run. Here a
failed or undefined query is recorded with its reason, its point leaves the
candidate pool, and it never becomes a label.
"""

from __future__ import annotations

import csv
import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt

# scikit-learn ships no type information; its objects are typed Any below.
from sklearn.decomposition import PCA  # pyright: ignore[reportMissingTypeStubs]
from sklearn.gaussian_process import (  # pyright: ignore[reportMissingTypeStubs]
    GaussianProcessRegressor,
)
from sklearn.gaussian_process.kernels import RBF  # pyright: ignore[reportMissingTypeStubs]
from sklearn.preprocessing import MinMaxScaler  # pyright: ignore[reportMissingTypeStubs]

from expliccit_env.episode import EpisodeOk, EpisodeOutcome
from expliccit_env.mahimahi import MahakMetric
from expliccit_env.spec import (
    DropTail,
    EpisodeSpec,
    MahakMrtt,
    MahakStep,
    PathSpec,
    StockCC,
    StockName,
)

type F64 = npt.NDArray[np.float64]

# Mahak's CSV columns, in its grid order.
COLUMNS: Final = ("buffer_size", "mRTT", "BW", "bw_step")
# helper.py runs every query for 20 s.
DURATION_MS: Final = 20_000


@dataclass(frozen=True, slots=True)
class Axis:
    """One of Mahak's ``min,max,step`` arguments."""

    lo: int | float
    hi: int | float
    step: int | float

    @staticmethod
    def parse(text: str, kind: type[int] | type[float]) -> Axis:
        lo, hi, step = (kind(x) for x in text.split(","))
        if step <= 0 or lo > hi:
            raise ValueError(f"axis {text}: need min <= max and step > 0")
        return Axis(lo, hi, step)

    def values(self) -> npt.NDArray[Any]:
        return np.arange(self.lo, self.hi + self.step, self.step)


@dataclass(frozen=True, slots=True)
class MahakSpace:
    buffer: Axis  # packets
    mrtt: Axis  # Mahak's mRTT: one-way delay, ms
    bw: Axis  # Mbit/s
    change: Axis  # capacity ratio after 12 s

    @staticmethod
    def parse(buffer: str, mrtt: str, bw: str, change: str) -> MahakSpace:
        return MahakSpace(
            Axis.parse(buffer, int),
            Axis.parse(mrtt, int),
            Axis.parse(bw, int),
            Axis.parse(change, float),
        )

    def grid(self) -> F64:
        """Every point, one row each, columns as in ``COLUMNS``."""
        change = [float(f"{e:.2f}") for e in self.change.values()]
        axes = (self.buffer.values(), self.mrtt.values(), self.bw.values(), change)
        return np.array(list(itertools.product(*axes)), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class MahakPoint:
    buffer_pkts: int
    mrtt_ms: int
    bw_mbps: int
    step: float

    @staticmethod
    def of(row: F64) -> MahakPoint:
        return MahakPoint(int(row[0]), int(row[1]), int(row[2]), float(row[3]))


@dataclass(frozen=True, slots=True)
class Failure:
    reason: str


type Oracle = Callable[[MahakPoint], float | Failure]


@dataclass(frozen=True, slots=True)
class Query:
    index: int  # row of the grid
    point: MahakPoint
    result: float | Failure


@dataclass(frozen=True, slots=True)
class MahakMap:
    grid: F64
    queries: tuple[Query, ...]
    mean: F64  # posterior mean per grid row; NaN if no query succeeded


def _features(grid: F64) -> F64:
    scaler: Any = MinMaxScaler()
    pca: Any = PCA()
    return cast("F64", pca.fit_transform(scaler.fit_transform(grid)))


class _Gp:
    def __init__(self, x: F64, y: Sequence[float]) -> None:
        self._gp: Any = GaussianProcessRegressor(
            kernel=RBF(length_scale=0.1, length_scale_bounds=(1e-60, 1e5)),
            n_restarts_optimizer=10,
            normalize_y=True,
            random_state=100,
        )
        self._gp.fit(x, np.asarray(y, dtype=np.float64))

    def mean(self, x: F64) -> F64:
        return cast("F64", self._gp.predict(x))

    def std(self, x: F64) -> F64:
        return cast("F64", self._gp.predict(x, return_std=True)[1])


def explore(space: MahakSpace, oracle: Oracle, *, budget: int, seed: int) -> MahakMap:
    """Run Mahak: one random query, then ``budget`` uncertainty queries."""
    grid = space.grid()
    features = _features(grid)
    alive = np.ones(grid.shape[0], dtype=bool)
    rs = np.random.RandomState(seed)  # Mahak draws from the legacy global RNG
    queries: list[Query] = []
    for _ in range(budget + 1):
        if not alive.any():
            break
        labelled = [q for q in queries if not isinstance(q.result, Failure)]
        if labelled:
            gp = _Gp(features[[q.index for q in labelled]], [_value(q) for q in labelled])
            i = int(np.argmax(np.where(alive, gp.std(features), -np.inf)))
        else:
            i = int(rs.choice(np.flatnonzero(alive), size=1)[0])
        point = MahakPoint.of(grid[i])
        result = oracle(point)
        if isinstance(result, float) and not math.isfinite(result):
            result = Failure(f"non-finite value {result}")
        if isinstance(result, Failure):
            alive[i] = False
        queries.append(Query(i, point, result))
    labelled = [q for q in queries if not isinstance(q.result, Failure)]
    mean = (
        _Gp(features[[q.index for q in labelled]], [_value(q) for q in labelled]).mean(features)
        if labelled
        else np.full(grid.shape[0], np.nan)
    )
    return MahakMap(grid, tuple(queries), mean)


def _value(q: Query) -> float:
    assert not isinstance(q.result, Failure)
    return q.result


def write_csv(m: MahakMap, metric: MahakMetric, path: Path) -> None:
    """Mahak's ``predictions/Mahak_<cc>.csv`` layout (pandas index column first)."""
    with path.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["", *COLUMNS, metric])
        for i, (row, v) in enumerate(zip(m.grid, m.mean, strict=True)):
            p = MahakPoint.of(row)
            value = "" if math.isnan(v) else repr(float(v))
            w.writerow([i, p.buffer_pkts, p.mrtt_ms, p.bw_mbps, f"{p.step:.2f}", value])


# --- the Tier 1 oracle -------------------------------------------------------


def mahak_spec(p: MahakPoint, cc: StockName, seed: int) -> EpisodeSpec:
    """The episode helper.py's ``oracle_query`` runs for one point."""
    return EpisodeSpec(
        path=PathSpec(
            trace=MahakStep(bw_mbps=p.bw_mbps, step=p.step),
            delay=MahakMrtt(mahak_compat_mrtt_ms=p.mrtt_ms),
            buffer=DropTail(packets=p.buffer_pkts),
        ),
        cc=StockCC(name=cc),
        duration_ms=DURATION_MS,
        seed=seed,
    )


def metric_value(outcome: EpisodeOutcome, metric: MahakMetric) -> float | Failure:
    """The metric Mahak's oracle returns, or why there is none."""
    if not isinstance(outcome, EpisodeOk):
        return Failure(str(outcome.row.get("failure")))
    value = outcome.row.get("mahak_delay" if metric == "delay" else metric)
    if not isinstance(value, float):
        return Failure(f"metric undefined: {outcome.row.get('metrics_undefined')}")
    return value
