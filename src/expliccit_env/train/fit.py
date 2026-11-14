"""Fitting net1 (supervised) and net2 offline on a frozen net1 (cloning,
TD3+BC, IQL or expliCCit's objective), the held-out report, the policy
graph of an offline net2, and expliCCit's released checkpoints as our
modules."""

# torch's own annotations leave many signatures partly unknown (from_numpy,
# manual_seed, onnx.export); every other strict check stays on.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from expliccit_env.policy import EXPLICCIT_OBS_COLS, OBS_DIM
from expliccit_env.train import beliefs
from expliccit_env.train.beliefs import Beliefs, Bins, Interval, MeansView, PoolBins
from expliccit_env.train.data import Split, Steps
from expliccit_env.train.graph import (
    Deployed,
    EveryStep,
    Front,
    MeasuredBdp,
    PolicyGraph,
    Reads,
    TruthColumns,
    WindowTarget,
)
from expliccit_env.train.lcfs import columns as lcf_columns
from expliccit_env.train.lcfs import value as lcf_value
from expliccit_env.train.models import (
    NET1,
    NET2,
    BinNet1,
    Critic,
    ExpliccitActor,
    ExpliccitNet1,
    IntervalNet1,
    Scaler,
    Value,
)
from expliccit_env.train.score import utility
from expliccit_env.train.spec import EXPLICCIT_LCFS, LCF_UNITS, Inputs, Net1Spec, Net2Spec
from expliccit_env.train.spec import Bins as BinsOutput
from expliccit_env.train.spec import Interval as IntervalOutput
from expliccit_env.train.spec import Point as PointOutput

type F32 = npt.NDArray[np.float32]
_EVAL_BATCH: Final = 65_536


@dataclass(frozen=True, slots=True)
class Net1:
    """A trained net1: its input scaler, network and LCF scaler, the LCFs it
    reports, and how to read its output (``kind``, beliefs.py)."""

    scaler: Scaler
    net: nn.Module
    lcf_scaler: Scaler  # targets in report units <-> net1's output space
    lcfs: tuple[str, ...]
    kind: Beliefs = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", beliefs.of(self.net, self.lcf_scaler).eval())

    @torch.no_grad()
    def beliefs(self, obs: F32) -> F32:
        """net1's output z, for each row."""
        self.net.eval()
        out = [
            self.net(self.scaler(torch.from_numpy(obs[i : i + _EVAL_BATCH])))
            for i in range(0, obs.shape[0], _EVAL_BATCH)
        ]
        return torch.cat(out).numpy() if out else np.zeros((0, self.width), np.float32)

    @property
    def width(self) -> int:
        """Numbers per row that net1 hands net2."""
        return self.kind.width

    def distributions(self, obs: F32) -> F32:
        """[n, K, B] bin probabilities (distribution net1 only)."""
        if not isinstance(self.kind, Bins):
            raise ValueError("only a distribution net1 states bin probabilities")
        return self.beliefs(obs).reshape(-1, self.kind.k, self.kind.b)

    @torch.no_grad()
    def predict(self, obs: F32) -> F32:
        """LCFs in report units (LCF_UNITS): each belief's mean, the
        estimate squared error rewards."""
        means = [
            self.kind.means(torch.from_numpy(self.beliefs(obs[i : i + _EVAL_BATCH]))).numpy()
            for i in range(0, obs.shape[0], _EVAL_BATCH)
        ]
        return np.concatenate(means) if means else np.zeros((0, self.kind.k), np.float32)

    @torch.no_grad()
    def bounds(self, obs: F32, level: float = 0.9) -> tuple[F32, F32]:
        """Lower and upper bounds in report units: an interval net1's, or a
        distribution net1's central interval holding ``level`` of its mass;
        a point net1 states none."""
        if isinstance(self.kind, Bins):
            tail = (1 - level) / 2
            q = quantiles(self.distributions(obs), self.kind.bounds.numpy(), (tail, 1 - tail))
            return q[:, :, 0], q[:, :, 1]
        if isinstance(self.kind, Interval):
            lo, hi = self.kind.bounds(torch.from_numpy(self.beliefs(obs)))
            return lo.numpy(), hi.numpy()
        raise ValueError("a point net1 states no bounds")


def quantiles(p: F32, bounds: F32, qs: tuple[float, ...]) -> F32:
    """[n, K, len(qs)] quantiles of binned distributions [n, K, B] with bin
    boundaries [K, B + 1], the mass spread evenly within each bin. Rows go
    in batches, so memory does not grow with n * K * B."""
    out = np.empty((*p.shape[:2], len(qs)), np.float32)
    for i in range(0, p.shape[0], _QUANTILE_BATCH):
        out[i : i + _QUANTILE_BATCH] = _quantiles(p[i : i + _QUANTILE_BATCH], bounds, qs)
    return out


_QUANTILE_BATCH: Final = 8192


def _quantiles(p: F32, bounds: F32, qs: tuple[float, ...]) -> F32:
    p64 = p.astype(np.float64)
    upto = np.cumsum(p64, axis=2)
    out = np.empty((*p.shape[:2], len(qs)))
    last = p.shape[2] - 1
    for j, q in enumerate(qs):
        crossed = upto >= q * upto[:, :, -1:] - 1e-12
        i = np.where(crossed.any(axis=2), crossed.argmax(axis=2), last)[:, :, None]
        mass = np.take_along_axis(p64, i, axis=2)[:, :, 0]
        below = np.take_along_axis(upto, i, axis=2)[:, :, 0] - mass
        frac = np.clip((q * upto[:, :, -1] - below) / np.where(mass > 0, mass, 1), 0, 1)
        lo = np.take_along_axis(np.broadcast_to(bounds[:, :-1], p.shape), i, axis=2)[:, :, 0]
        hi = np.take_along_axis(np.broadcast_to(bounds[:, 1:], p.shape), i, axis=2)[:, :, 0]
        out[:, :, j] = lo + frac * (hi - lo)
    return out.astype(np.float32)


_TAUS: Final = tuple((j + 0.5) / 50 for j in range(50))


def crps(p: F32, bounds: F32, y: F32) -> F32:
    """[n, K] continuous ranked probability score of binned distributions
    against true values (report units; NaN where y is): twice the mean
    pinball loss over 50 evenly spaced quantile levels. It is proper, works
    on the distribution's CDF rather than its bins, so it compares
    distributions over different bins, and equals the absolute error for a
    point forecast."""
    q = quantiles(p, bounds, _TAUS)
    tau = np.array(_TAUS)
    u = y[:, :, None] - q
    return (2 * (u * (tau - (u < 0))).mean(axis=2)).astype(np.float32)


def _bin_bounds(y: F32, bins: int) -> F32:
    """Per LCF, B + 1 bin boundaries fitted to the training targets. A
    quantity within [0, 1] gets B equal bins. Any other gets bin 0 below
    the 0.1% quantile of its positive values (zero and near zero),
    log-spaced bins up to its 99.9% quantile, and one more log step as the
    last bin, which also takes everything above."""
    out = np.empty((y.shape[1], bins + 1), np.float64)
    for j in range(y.shape[1]):
        v = y[:, j][np.isfinite(y[:, j])].astype(np.float64)
        pos = v[v > 0]
        if v.size and v.min() >= 0 and v.max() <= 1:
            out[j] = np.linspace(0.0, 1.0, bins + 1)
            continue
        lo = float(np.quantile(pos, 0.001)) if pos.size else 1.0
        hi = float(np.quantile(v, 0.999)) if v.size else lo * 10
        if hi <= lo:  # a constant quantity: any small range will do
            hi = lo * 2
        inner = np.geomspace(lo, hi, bins - 1)
        step = inner[1] / inner[0]
        out[j] = [min(0.0, float(v.min()) if v.size else 0.0), *inner, hi * step]
    return out.astype(np.float32)


def targets(steps: Steps, lcfs: tuple[str, ...]) -> F32:
    return np.stack([steps.labels[n] * np.float32(LCF_UNITS[n][0]) for n in lcfs], axis=1)


def _metrics(pred: F32, true: F32) -> dict[str, float]:
    ok = np.isfinite(true) & np.isfinite(pred)
    p, t = pred[ok].astype(np.float64), true[ok].astype(np.float64)
    if t.size == 0:
        return {"n": 0}
    var = float(t.var())
    mse = float(((p - t) ** 2).mean())
    return {
        "n": int(t.size),
        "mae": float(np.abs(p - t).mean()),
        "rmse": mse**0.5,
        "r2": 1 - mse / var if var > 0 else float("nan"),
    }


def _baselines(steps: Steps, rows: npt.NDArray[np.bool_]) -> dict[str, F32]:
    """Classical estimators from the message itself, in report units."""
    o = steps.obs[rows]
    queue_ms = o[:, 2] * 100 - o[:, 77]  # srtt minus min RTT
    return {
        "capacity_bps": o[:, 74] * 100,  # max delivered rate over 2 s
        "receive_rate_bps": o[:, 71] * 100,  # delivered rate over the last 100 steps
        "queue_delay_mean_ns": queue_ms,
        "queue_delay_ns": queue_ms,
    }


def report_net1(net1: Net1, steps: Steps, split: Split | None = None) -> dict[str, Any]:
    """Error per LCF in report units, next to the classical estimators."""
    rows = np.ones(steps.obs.shape[0], bool) if split is None else steps.rows(split)
    pred, true = net1.predict(steps.obs[rows]), targets(steps, net1.lcfs)[rows]
    base = _baselines(steps, rows)
    out: dict[str, Any] = {}
    intervals = isinstance(net1.kind, Interval)
    lo, hi = net1.bounds(steps.obs[rows]) if intervals else (pred, pred)
    busy = steps.labels["busy_fraction"][rows]
    dist = _report_distributions(net1, steps, rows) if isinstance(net1.kind, Bins) else {}

    def part(column: str) -> F32:
        """net1's estimate of a label column, in the column's own units."""
        return pred[:, net1.lcfs.index(column)] / np.float32(LCF_UNITS[column][0])

    for k, name in enumerate(net1.lcfs):
        out[name] = {"unit": LCF_UNITS[name][1], "net1": _metrics(pred[:, k], true[:, k])}
        if name in dist:
            out[name]["distribution"] = dist[name]
        parts = lcf_columns([name])
        if parts != {name} and parts <= set(net1.lcfs):
            # A composite whose parts net1 also states: the formula on the
            # parts' estimates, to say whether its own head adds anything.
            derived = lcf_value(name, part) * np.float32(LCF_UNITS[name][0])
            out[name]["derived"] = _metrics(derived, true[:, k])
        if name in base:
            out[name]["baseline"] = _metrics(base[name], true[:, k])
        if intervals:
            ok = np.isfinite(true[:, k])
            width = hi[:, k] - lo[:, k]
            inside = (lo[:, k] <= true[:, k]) & (true[:, k] <= hi[:, k])
            out[name]["interval"] = {
                "coverage": float(inside[ok].mean()) if ok.any() else float("nan"),
                "mean_width": float(width[ok].mean()) if ok.any() else float("nan"),
                # Identifiability: how wide the interval is when the link is
                # kept busy against when it is not.
                "mean_width_busy": _mean_where(width, ok & (busy >= 0.95)),
                "mean_width_idle": _mean_where(width, ok & (busy < 0.8)),
            }
    return out


def _mean_where(x: npt.NDArray[np.floating[Any]], rows: npt.NDArray[np.bool_]) -> float:
    return float(x[rows].mean()) if rows.any() else float("nan")


_LEVELS: Final = (0.5, 0.8, 0.9, 0.95)


def _regime(
    m: npt.NDArray[np.bool_],
    nll: npt.NDArray[np.float64],
    hit90: npt.NDArray[np.float64],
    width90: npt.NDArray[np.float32],
) -> dict[str, float]:
    return {
        "n": int(m.sum()),
        "log_loss": _mean_where(nll, m),
        "coverage_90": _mean_where(hit90, m),
        "width_90": _mean_where(width90, m),
    }


def _report_distributions(
    net1: Net1, steps: Steps, rows: npt.NDArray[np.bool_]
) -> dict[str, dict[str, Any]]:
    """Honesty and sharpness of a distribution net1, per LCF: the log score
    (nats) against a forecast that ignores the measurements (the training
    rows' bin frequencies), the information that difference amounts to,
    the coverage of central sets at several levels, the entropy and the
    90% interval's width; then the same split by whether the link was kept
    busy (busy fraction at least 0.95) or not (below 0.8).

    net1 states probabilities of bins, so coverage is tested on bins: with
    the randomized probability integral transform u = P(below the true
    bin) + V P(true bin), V uniform, u is uniform exactly when the bin
    probabilities are right, and a central set holds the truth when u lies
    within it. The width is the interpolated interval a human reads, which
    is accurate to one bin."""
    kind = net1.kind
    assert isinstance(kind, Bins)
    bounds = kind.bounds.numpy()
    p = net1.distributions(steps.obs[rows])  # float32; one LCF at a time in float64
    y = targets(steps, net1.lcfs)
    assert isinstance(net1.net, BinNet1)
    with torch.no_grad():  # the network's bin_of, whose memory does not grow with the bins
        idx = net1.net.bin_of(torch.from_numpy(np.nan_to_num(y))).numpy()
    train = steps.rows("train") if steps.rows("train").any() else np.ones_like(rows)
    qs = sorted({q for lv in _LEVELS for q in ((1 - lv) / 2, (1 + lv) / 2)})
    qv = quantiles(p, bounds, tuple(qs))
    busy = steps.labels["busy_fraction"][rows]
    rng = np.random.default_rng(0)
    out: dict[str, dict[str, Any]] = {}
    for k, name in enumerate(net1.lcfs):
        ok = np.isfinite(y[rows, k])
        if not ok.any():
            continue
        i = idx[rows, k]
        pk64 = p[:, k].astype(np.float64)
        seen = np.isfinite(y[train, k])
        clim = np.bincount(idx[train, k][seen], minlength=kind.b) + 1.0
        clim = clim / clim.sum()
        p_true = np.take_along_axis(pk64, i[:, None], 1)[:, 0]
        nll = -np.log(np.clip(p_true, 1e-12, None))
        below = np.take_along_axis(np.cumsum(pk64, 1), i[:, None], 1)[:, 0] - p_true
        u = below + rng.uniform(size=p_true.size) * p_true
        ent = -(pk64 * np.log2(np.clip(pk64, 1e-12, None))).sum(axis=1)

        inside = {lv: np.abs(u - 0.5) <= lv / 2 for lv in _LEVELS}
        width90 = qv[:, k, qs.index((1 + 0.9) / 2)] - qv[:, k, qs.index((1 - 0.9) / 2)]
        cov = {str(lv): _mean_where(inside[lv].astype(np.float64), ok) for lv in _LEVELS}
        clim_nll = _mean_where(-np.log(clim[i]), ok)
        t = np.where(ok, y[rows, k], 0).astype(np.float32)[:, None]
        pk, bk = p[:, k : k + 1], bounds[k : k + 1]
        score = crps(pk, bk, t)[:, 0]
        clim_p = np.broadcast_to(clim.astype(np.float32), pk.shape)
        clim_crps = _mean_where(crps(clim_p, bk, t)[:, 0], ok)
        median_mae = _mean_where(np.abs(quantiles(pk, bk, (0.5,))[:, 0, 0] - t[:, 0]), ok)
        hit90 = inside[0.9].astype(np.float64)
        out[name] = {
            "crps": _mean_where(score, ok),
            "crps_skill": 1 - _mean_where(score, ok) / clim_crps if clim_crps > 0 else float("nan"),
            "median_mae": median_mae,
            "log_loss": _mean_where(nll, ok),
            "log_loss_without_measurements": clim_nll,
            "information_bits": (clim_nll - _mean_where(nll, ok)) / np.log(2),
            "coverage": cov,
            "coverage_error": float(np.mean([abs(cov[str(lv)] - lv) for lv in _LEVELS])),
            "entropy_bits": _mean_where(ent, ok),
            "width_90": _mean_where(width90, ok),
            "busy": _regime(ok & (busy >= 0.95), nll, hit90, width90),
            "idle": _regime(ok & (busy < 0.8), nll, hit90, width90),
        }
    return out


def _masked_mse(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return ((pred - y) ** 2 * mask).sum() / mask.sum().clamp(min=1)


def _masked_pinball(
    pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, coverage: float
) -> torch.Tensor:
    """Pinball loss of (centres, lower, upper) at quantiles 0.5 and the
    interval's two tails."""
    k = y.shape[1]
    tail = (1 - coverage) / 2
    loss = torch.zeros(())
    for j, q in enumerate((0.5, tail, 1 - tail)):
        e = y - pred[:, j * k : (j + 1) * k]
        loss = loss + (torch.maximum(q * e, (q - 1) * e) * mask).sum() / mask.sum().clamp(min=1)
    return loss


def _calibrate(
    net: IntervalNet1, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, coverage: float
) -> None:
    """Conformalized quantile regression (Romano et al., 2019): widen (or
    narrow) each LCF's interval so that it covers the held-out rows at the
    requested rate."""
    net.eval()
    with torch.no_grad():
        out = net(x)
    k = y.shape[1]
    lo, hi = out[:, k : 2 * k], out[:, 2 * k :]
    score = torch.maximum(lo - y, y - hi)
    widen = torch.zeros(k)
    for j in range(k):
        e = score[:, j][mask[:, j] > 0]
        if e.numel():
            level = min(1.0, coverage * (1 + 1 / e.numel()))
            widen[j] = torch.quantile(e, level)
    net.widen.copy_(widen)


def fit_net1(spec: Net1Spec, steps: Steps, seed: int) -> tuple[Net1, dict[str, Any]]:
    torch.manual_seed(seed)
    tr, va = steps.rows("train"), steps.rows("val")
    y = targets(steps, spec.lcfs)
    cols = list(spec.obs_cols)
    scaler = Scaler.fit(spec.obs_cols, steps.obs[np.ix_(np.flatnonzero(tr), cols)], spec.scaling)
    lcf_kind = "minmax" if spec.scaling == "minmax" else "standard"
    lcf_scaler = Scaler.fit(tuple(range(len(spec.lcfs))), y[tr], lcf_kind)

    def tensors(rows: npt.NDArray[np.bool_]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Scaled in batches into one tensor: memory for the inputs, not for
        # copies of the whole message.
        idx = np.flatnonzero(rows)
        x = torch.empty(idx.size, len(cols))
        with torch.no_grad():
            for i in range(0, idx.size, _EVAL_BATCH):
                b = idx[i : i + _EVAL_BATCH]
                x[i : i + b.size] = scaler(torch.from_numpy(steps.obs[b]))
            t = lcf_scaler(torch.from_numpy(y[rows]))
        mask = torch.from_numpy(np.isfinite(y[rows]).astype(np.float32))
        return x, t, mask

    (xt, yt, mt), (xv, yv, mv) = tensors(tr), tensors(va)
    net = _net1_module(spec)
    log: dict[str, Any]
    if isinstance(net, BinNet1):  # spec.output.kind == "bins"
        net.bounds.copy_(torch.from_numpy(_bin_bounds(y[tr], net.b)))
        with torch.no_grad():  # targets become bin indices
            yt = net.bin_of(torch.from_numpy(np.nan_to_num(y[tr]))).float()
            yv = net.bin_of(torch.from_numpy(np.nan_to_num(y[va]))).float()
        # Each member is trained on its own, with its own batch order.
        runs = []
        for i, member in enumerate(net.members):
            one = BinNet1([member], net.k, net.b)
            one.bounds.copy_(net.bounds)
            runs.append(_train(one, spec, (xt, yt, mt), (xv, yv, mv), seed + i))
        ensemble = _val_loss(net, spec, xv, yv, mv)
        log = {"val_loss": runs[0], "members_val_loss": runs, "ensemble_val_loss": ensemble}
    else:
        log = {"val_loss": _train(net, spec, (xt, yt, mt), (xv, yv, mv), seed)}
        if (
            isinstance(net, IntervalNet1)
            and isinstance(spec.output, IntervalOutput)
            and xv.shape[0]
        ):
            _calibrate(net, xv, yv, mv, spec.output.coverage)
    log["epochs"] = len(log["val_loss"])
    return Net1(scaler, net, lcf_scaler, tuple(spec.lcfs)), log


def _loss(
    net: nn.Module, spec: Net1Spec, x: torch.Tensor, y: torch.Tensor, m: torch.Tensor
) -> torch.Tensor:
    """The training objective: the log score of the true bin for a
    distribution, the pinball loss for an interval, else squared error."""
    match spec.output:
        case BinsOutput():
            assert isinstance(net, BinNet1)
            lp = net.log_probs(x).gather(2, y.long().unsqueeze(2)).squeeze(2)
            return -(lp * m).sum() / m.sum().clamp(min=1)
        case IntervalOutput(coverage=coverage):
            return _masked_pinball(net(x), y, m, coverage)
        case PointOutput():
            return _masked_mse(net(x), y, m)


@torch.no_grad()
def _val_loss(
    net: nn.Module, spec: Net1Spec, x: torch.Tensor, y: torch.Tensor, m: torch.Tensor
) -> float:
    """_loss over all rows, in batches so memory does not grow with the
    validation set. Each _loss is a masked sum over its rows' mask sum, so
    weighting a batch by its mask sum gives the loss over all rows."""
    if not x.shape[0]:
        return float("nan")
    net.eval()
    total = 0.0
    for i in range(0, x.shape[0], _EVAL_BATCH):
        b = slice(i, i + _EVAL_BATCH)
        total += float(_loss(net, spec, x[b], y[b], m[b])) * float(m[b].sum())
    return total / max(float(m.sum()), 1.0)


def _train(
    net: nn.Module,
    spec: Net1Spec,
    train: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    val: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    seed: int,
) -> list[float]:
    """Adam with early stopping on the validation loss; leaves ``net`` at its
    best epoch and returns the validation loss per epoch. Without any
    validation target it runs every epoch and keeps the last."""
    rng = np.random.default_rng(seed)
    (xt, yt, mt), (xv, yv, mv) = train, val
    judged = bool(mv.sum() > 0)
    opt = torch.optim.AdamW(net.parameters(), lr=spec.lr, weight_decay=spec.weight_decay)
    best, best_state, stale = float("inf"), copy.deepcopy(net.state_dict()), 0
    history: list[float] = []
    for _ in range(spec.epochs):
        net.train()
        order = torch.from_numpy(rng.permutation(xt.shape[0]))
        for i in range(0, xt.shape[0] - 1, spec.batch_size):  # BatchNorm needs 2+ rows
            b = order[i : i + spec.batch_size]
            if b.numel() < 2:
                continue
            loss = _loss(net, spec, xt[b], yt[b], mt[b])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        val_loss = _val_loss(net, spec, xv, yv, mv)
        history.append(val_loss)
        if not judged:
            best_state = copy.deepcopy(net.state_dict())
        elif val_loss < best:
            best, best_state, stale = val_loss, copy.deepcopy(net.state_dict()), 0
        else:
            stale += 1
            if stale >= spec.patience:
                break
    net.load_state_dict(best_state)
    return history


@dataclass(frozen=True, slots=True)
class Transitions:
    s: torch.Tensor
    a: torch.Tensor
    r: torch.Tensor
    s2: torch.Tensor

    def parts(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.s, self.a, self.r, self.s2


def _consecutive(steps: Steps) -> npt.NDArray[np.bool_]:
    """[n - 1]: whether row t + 1 is the step right after row t, in its
    episode (not when a step stride dropped the steps between)."""
    return (steps.episode[:-1] == steps.episode[1:]) & (steps.step[1:] == steps.step[:-1] + 1)


def transitions(
    steps: Steps, states: F32, actions: F32, spec: Net2Spec, split: Split
) -> tuple[Transitions, float]:
    """(s_t, a_t, r, s_t+n) over n = spec.n_step consecutive steps, r the
    discounted sum of their n rewards; a (logged_actions) scaled to [-1, 1]
    and clipped. Returns the fraction of actions that needed clipping."""
    n = spec.n_step
    # links[t]: how many of rows 0..t follow their predecessor as the next step.
    links = np.concatenate(([0], np.cumsum(_consecutive(steps))))
    ahead = np.zeros(links.size, bool)  # rows t .. t + n are consecutive steps
    if links.size > n:
        ahead[: links.size - n] = links[n:] - links[:-n] == n
    i = np.flatnonzero(steps.rows(split) & ahead)
    reward = steps.reward if spec.reward == "sage" else utility(steps.labels, spec)
    first = 1 if spec.reward_alignment == "next" else 0
    r = sum(spec.discount**k * reward[i + first + k] for k in range(n))
    r = np.asarray(r, np.float32)
    keep = np.isfinite(actions[i]) & np.isfinite(r)
    i, r = i[keep], r[keep]
    a = actions[i] / spec.action_max
    clipped = float(np.mean(np.abs(a) > 1)) if a.size else 0.0
    t = Transitions(
        torch.from_numpy(states[i]),
        torch.from_numpy(np.clip(a, -1, 1).reshape(-1, 1)),
        torch.from_numpy((r * spec.reward_scale).astype(np.float32)),
        torch.from_numpy(states[i + n]),
    )
    return t, clipped


def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for t, s in zip(target.parameters(), source.parameters(), strict=True):
            t.mul_(1 - tau).add_(s, alpha=tau)


def _policy_stats(pi: torch.Tensor) -> dict[str, float]:
    """The actor's output on a batch: its mean, spread and saturation."""
    return {
        "pi_mean": float(pi.mean()),
        "pi_std": float(pi.std()),
        "sat": float((pi.abs() > 0.98).float().mean()),
    }


class Learner:
    """Actor, critics and their targets, updated one batch at a time. Each
    update returns its losses and diagnostics (docs/training.md)."""

    def __init__(self, spec: Net2Spec, k: int) -> None:
        self.spec = spec
        self.actor = NET2[spec.arch](k, spec.hidden)
        n_critics = {"td3bc": 2, "expliccit-v1": 1, "bc": 0, "iql": 2}[spec.algo]
        self.critics = [Critic(k, spec.hidden) for _ in range(n_critics)]
        self.value = Value(k, spec.hidden) if spec.algo == "iql" else None
        self.actor_t = copy.deepcopy(self.actor)
        self.critics_t = [copy.deepcopy(c) for c in self.critics]
        # expliCCit's learning rates for its objective, the spec's otherwise.
        theirs = spec.algo == "expliccit-v1"
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=1e-4 if theirs else spec.lr)
        critic_params = [p for c in self.critics for p in c.parameters()]
        critic_params += list(self.value.parameters()) if self.value is not None else []
        self.opt_c = (
            torch.optim.Adam(critic_params, lr=1e-3 if theirs else spec.lr)
            if critic_params
            else None
        )

    def _critic_loss(self, s: torch.Tensor, a: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        qs = [c(s, a) for c in self.critics]
        loss = torch.stack([((q - y) ** 2).mean() for q in qs]).sum()
        if self.spec.algo == "expliccit-v1":  # random actions pushed below the data's
            a_rand = torch.empty_like(a).uniform_(-1, 1)
            loss = loss + self.spec.cql_alpha * (self.critics[0](s, a_rand).mean() - qs[0].mean())
        return loss

    def update(self, t: Transitions, step: int, *, actor: bool = True) -> dict[str, float]:
        """One gradient step on the critics and (every policy_delay steps,
        unless ``actor`` is False) on the actor."""
        spec, (s, a, r, s2) = self.spec, (t.s, t.a, t.r, t.s2)
        if self.opt_c is None:  # bc
            pi = self.actor(s)
            bc = ((pi - a) ** 2).mean()
            self.opt_a.zero_grad()
            bc.backward()
            self.opt_a.step()
            return {"actor": float(bc.detach()), "bc": float(bc.detach())} | _policy_stats(
                pi.detach()
            )
        if spec.algo == "iql":
            return self._iql(t, actor=actor)
        with torch.no_grad():
            a2 = self.actor_t(s2)
            if spec.algo == "td3bc":
                noise = torch.randn_like(a2) * spec.policy_noise
                a2 = (a2 + noise.clamp(-spec.noise_clip, spec.noise_clip)).clamp(-1, 1)
            q2 = torch.stack([c(s2, a2) for c in self.critics_t]).min(dim=0).values
            y = r + spec.discount**spec.n_step * q2
            # Target actions outside what the batch's data did: where the
            # critic's target is extrapolation.
            lo, hi = torch.quantile(a, 0.005), torch.quantile(a, 0.995)
            ood = float(((a2 < lo) | (a2 > hi)).float().mean())
        critic_loss = self._critic_loss(s, a, y)
        self.opt_c.zero_grad()
        critic_loss.backward()
        self.opt_c.step()
        out = {"critic": float(critic_loss.detach()), "ood": ood}
        if actor and (spec.algo == "expliccit-v1" or step % spec.policy_delay == 0):
            pi = self.actor(s)
            q = self.critics[0](s, pi)
            bc = ((pi - a) ** 2).mean()
            lam = (spec.alpha / q.abs().mean().detach()) if spec.algo == "td3bc" else 1.0
            actor_loss = -lam * q.mean() + (bc if spec.algo == "td3bc" else 0.0)
            # Per row, the pull of the critic against the pull of the data.
            (dq,) = torch.autograd.grad(q.sum(), pi, retain_graph=True)
            self.opt_a.zero_grad()
            actor_loss.backward()
            self.opt_a.step()
            out |= {
                "actor": float(actor_loss.detach()),
                "bc": float(bc.detach()),
                "q": float(q.mean().detach()),
                "q_pull": float((lam * dq).abs().mean()),
                "bc_pull": float((2 * (pi - a)).abs().mean().detach()),
            } | _policy_stats(pi.detach())
            _soft_update(self.actor_t, self.actor, spec.tau)
            for c, ct in zip(self.critics, self.critics_t, strict=True):
                _soft_update(ct, c, spec.tau)
        return out

    def _iql(self, t: Transitions, *, actor: bool) -> dict[str, float]:
        spec, (s, a, r, s2) = self.spec, t.parts()
        assert self.value is not None and self.opt_c is not None
        with torch.no_grad():
            qt = torch.stack([c(s, a) for c in self.critics_t]).min(dim=0).values
            y = r + spec.discount**spec.n_step * self.value(s2)
        v = self.value(s)
        u = qt - v
        value_loss = (torch.abs(spec.expectile - (u < 0).float()) * u**2).mean()
        critic_loss = torch.stack([((c(s, a) - y) ** 2).mean() for c in self.critics]).sum()
        self.opt_c.zero_grad()
        (critic_loss + value_loss).backward()
        self.opt_c.step()
        for c, ct in zip(self.critics, self.critics_t, strict=True):
            _soft_update(ct, c, spec.tau)
        out = {
            "critic": float(critic_loss.detach()),
            "value": float(value_loss.detach()),
            "v": float(v.mean().detach()),
        }
        if actor:
            with torch.no_grad():
                adv = qt - v.detach()
                w = torch.exp(spec.beta * adv).clamp(max=spec.max_weight)
            pi = self.actor(s)
            bc = ((pi - a) ** 2).sum(dim=1)
            actor_loss = (w * bc).mean()
            self.opt_a.zero_grad()
            actor_loss.backward()
            self.opt_a.step()
            out |= {
                "actor": float(actor_loss.detach()),
                "bc": float(bc.mean().detach()),
                "adv_std": float(adv.std()),
                "w_mean": float(w.mean()),
                "w_capped": float((w >= spec.max_weight).float().mean()),
            } | _policy_stats(pi.detach())
        return out


def window_target(net1: Net1) -> WindowTarget:
    """log2 of the BDP net1 believes in, from any kind of net1 that reports
    capacity and one-way delay."""
    missing = {"capacity_bps", "one_way_delay_ns"} - set(net1.lcfs)
    if missing:
        raise ValueError(f"a window target needs net1's {sorted(missing)}")
    return WindowTarget(
        net1.kind, net1.lcfs.index("capacity_bps"), net1.lcfs.index("one_way_delay_ns")
    )


type View = Literal["bins", "means"]


def view_of(net1: Net1, view: View, input_bins: int | None = None) -> nn.Module | None:
    """What turns net1's output into net2's input: all of it ("bins"; for a
    distribution net1, pooled into input_bins per LCF when set), or each
    belief's standardized mean ("means")."""
    if view == "means":
        if input_bins is not None:
            raise ValueError("input_bins applies to the bins view only")
        return MeansView(net1.kind)
    if input_bins is None:
        return None
    if not isinstance(net1.kind, Bins):
        raise ValueError(f"input_bins {input_bins} pools a distribution net1's bins")
    return PoolBins(net1.kind.k, net1.kind.b, input_bins)


_READS: Final[dict[str, Reads]] = {
    "net1": "beliefs",
    "truth": "beliefs",
    "raw": "measurements",
    "net1+raw": "both",
}


def front(
    net1: Net1,
    *,
    inputs: Inputs = "net1",
    view: View = "bins",
    input_bins: int | None = None,
    truth: frozenset[str] = frozenset(),
    faults: dict[str, tuple[str, float]] | None = None,
    deployed: tuple[bool, ...] | None = None,
    traffic: nn.Module | None = None,
) -> Front:
    """net2's front end on net1 (graph.Front): the one module both the
    exported graph and training read net2's input through. ``inputs``
    "truth" takes every LCF from the truth, ``truth`` names some to
    take (a patched net1); ``faults`` (parse_faults) makes beliefs wrong on
    purpose; ``deployed`` names the LCFs expliCCit's deployed controller
    rounds."""
    if inputs == "truth":
        truth = frozenset(net1.lcfs)
    if inputs in ("raw", "net1+raw") and truth:
        raise ValueError(f'inputs "{inputs}" take nothing from the truth')
    missing = sorted(truth - set(net1.lcfs))
    if missing:
        raise ValueError(f"cannot take {missing} from the truth: net1 reports {list(net1.lcfs)}")
    unknown = sorted(n for n in truth if not TruthColumns.known(n))
    if unknown:
        raise ValueError(f"cannot take {unknown} from the truth: not known at the step")
    fault = None
    if faults:
        fault = net1.kind.fault(
            [faults[n][1] if faults.get(n, ("", 0.0))[0] == "scale" else 1.0 for n in net1.lcfs],
            [faults[n][1] if faults.get(n, ("", 0.0))[0] == "pin" else 0.0 for n in net1.lcfs],
            [faults.get(n, ("", 0.0))[0] == "pin" for n in net1.lcfs],
        )
    reads = _READS[inputs]
    return Front(
        net1.scaler,
        net1.net,
        net1.kind,
        None if reads == "measurements" else view_of(net1, view, input_bins),
        reads=reads,
        take=[n in truth for n in net1.lcfs] if truth else None,
        fault=fault,
        deployed=None if deployed is None else Deployed(net1.kind, deployed),
        traffic=traffic,
    ).eval()


def truth_columns(net1: Net1) -> TruthColumns:
    """The contract's truth input -> net1's LCFs in report units."""
    return TruthColumns(net1.lcfs)


def raw_width(net1: Net1) -> int:
    """Numbers per row in the measurements net1 reads (its input scaler)."""
    with torch.no_grad():
        return int(net1.scaler(torch.zeros(1, OBS_DIM)).shape[1])


def input_width(f: Front) -> int:
    """Numbers per row in net2's input through front ``f``."""
    with torch.no_grad():
        return int(f(torch.zeros(1, OBS_DIM))[1].shape[1])


def net2_width(net1: Net1, spec: Net2Spec) -> int:
    """Numbers per row in net2's input."""
    return input_width(net2_front(net1, spec))


def net2_front(net1: Net1, spec: Net2Spec) -> Front:
    return front(net1, inputs=spec.inputs, view=spec.view, input_bins=spec.input_bins)


def oracle(net1: Net1, z: torch.Tensor, y: F32) -> torch.Tensor:
    """net1's output z with every LCF taken from the truth y (report units)
    where it is a number, as an oracle's front end builds it."""
    take = torch.ones(net1.kind.k, dtype=torch.bool)
    with torch.no_grad():
        return net1.kind.patch(z, torch.from_numpy(y), take)


@torch.no_grad()
def net2_view(net1: Net1, spec: Net2Spec, steps: Steps) -> tuple[F32, F32]:
    """Per step, through the front end the exported graph uses: net2's input
    and log2 of the BDP its window is aimed at (NaN unless spec.action is
    "bdp"). In batches, so memory follows net2's input."""
    f = net2_front(net1, spec)
    target = _target(net1, spec.inputs) if spec.action == "bdp" else None
    y = targets(steps, net1.lcfs) if f.uses_truth else None
    states, log2_bdp = [], []
    for i in range(0, steps.obs.shape[0], _EVAL_BATCH):
        obs = torch.from_numpy(steps.obs[i : i + _EVAL_BATCH])
        yb = None if y is None else torch.from_numpy(y[i : i + _EVAL_BATCH])
        z, x, traffic = f(obs, yb)
        nan = torch.full((x.shape[0],), float("nan"))
        log2_bdp.append(nan if target is None else target(obs, z, traffic))
        states.append(x)
    if not states:
        return np.zeros((0, input_width(f)), np.float32), np.zeros(0, np.float32)
    return torch.cat(states).numpy(), torch.cat(log2_bdp).numpy()


def _target(net1: Net1, inputs: Inputs) -> nn.Module:
    """The window target of net2 on these inputs: net1's believed BDP, or
    BBR's estimate for the black box."""
    return MeasuredBdp() if inputs == "raw" else window_target(net1)


def logged_actions(steps: Steps, spec: Net2Spec, log2_bdp: F32) -> F32:
    """The action each step took, in net2's terms: the contract's a, or with
    action "bdp" log2 of the next step's window over this step's believed
    BDP (NaN where the next row is not the next step)."""
    if spec.action == "multiplier":
        return steps.action
    nxt = np.full(steps.cwnd.shape, np.nan, np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        nxt[:-1] = np.where(_consecutive(steps), np.log2(steps.cwnd[1:]), np.nan)
    return (nxt - log2_bdp).astype(np.float32)


def fit_net2(
    spec: Net2Spec, steps: Steps, net1: Net1, seed: int
) -> tuple[nn.Module, dict[str, Any]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    states, log2_bdp = net2_view(net1, spec, steps)
    actions = logged_actions(steps, spec, log2_bdp)
    data, clipped = transitions(steps, states, actions, spec, "train")
    val, _ = transitions(steps, states, actions, spec, "val")
    width = states.shape[1]
    del states  # every row's input; the transitions hold what training needs
    if data.s.shape[0] == 0:
        raise ValueError("no training transitions")
    learner = Learner(spec, width)
    log: dict[str, list[float]] = {}
    for step in range(spec.steps):
        b = torch.from_numpy(rng.integers(0, data.s.shape[0], spec.batch_size))
        batch = Transitions(data.s[b], data.a[b], data.r[b], data.s2[b])
        for name, v in learner.update(batch, step).items():
            log.setdefault(name, []).append(v)
    actor = learner.actor.eval()
    with torch.no_grad():
        val_bc = float(((actor(val.s) - val.a) ** 2).mean()) if val.s.shape[0] else float("nan")
    tail = max(1, spec.steps // 10)
    return actor, {
        "transitions": int(data.s.shape[0]),
        "actions_clipped": clipped,
        "final": {
            name: float(np.mean(v[-tail:])) if v else float("nan") for name, v in log.items()
        },
        "val_bc_mse": val_bc,
    }


def save_net1(net1: Net1, spec: Net1Spec, path: Path) -> None:
    """The whole estimator: its spec, input and output scalers, weights."""
    torch.save(
        {
            "spec": spec.model_dump(mode="json"),
            "scaler": net1.scaler.state_dict(),
            "lcf_scaler": net1.lcf_scaler.state_dict(),
            "net": net1.net.state_dict(),
        },
        path,
    )


def load_net1(path: Path) -> tuple[Net1, Net1Spec]:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    spec = Net1Spec.model_validate(saved["spec"])
    k, f = len(spec.lcfs), len(spec.obs_cols)
    zeros = np.zeros((2, f), np.float32)
    scaler = Scaler.fit(spec.obs_cols, zeros, "standard")
    scaler.load_state_dict(saved["scaler"])
    lcf_scaler = Scaler.fit(tuple(range(k)), np.zeros((2, k), np.float32), "standard")
    lcf_scaler.load_state_dict(saved["lcf_scaler"])
    net = _net1_module(spec)
    net.load_state_dict(saved["net"])
    return Net1(scaler, net.eval(), lcf_scaler, tuple(spec.lcfs)), spec


def _net1_module(spec: Net1Spec) -> nn.Module:
    k, f, arch = len(spec.lcfs), len(spec.obs_cols), NET1[spec.arch]
    match spec.output:
        case PointOutput():
            return arch(f, k, spec.width, spec.dropout)
        case IntervalOutput():
            return IntervalNet1(arch(f, k, spec.width, spec.dropout, 3), k)
        case BinsOutput(bins=b, members=n):
            return BinNet1([arch(f, k, spec.width, spec.dropout, b) for _ in range(n)], k, b)


def policy_graph(
    net1: Net1,
    actor: nn.Module,
    action_max: float,
    action_decimals: int | None = None,
    *,
    truth: frozenset[str] = frozenset(),
    action: Literal["multiplier", "bdp"] = "multiplier",
    view: View = "bins",
    input_bins: int | None = None,
    inputs: Inputs = "net1",
    faults: dict[str, tuple[str, float]] | None = None,
    deployed: tuple[bool, ...] | None = None,
) -> PolicyGraph:
    """The exported policy of a net2 asked at every step (graph.EveryStep):
    ``front``'s options, then net2, then the window multiplier or, with
    action "bdp", a window aimed at the believed BDP. ``deployed`` also
    rounds the action to three decimals, as expliCCit's controller does."""
    f = front(
        net1,
        inputs=inputs,
        view=view,
        input_bins=input_bins,
        truth=truth,
        faults=faults,
        deployed=deployed,
    )
    decimals = 3 if deployed is not None else action_decimals
    return PolicyGraph(
        f,
        EveryStep(actor, action_max, decimals),
        _target(net1, inputs) if action == "bdp" else None,
        truth_columns(net1) if f.uses_truth else None,
    ).eval()


def parse_faults(texts: list[str], lcfs: tuple[str, ...]) -> dict[str, tuple[str, float]]:
    """``LCF*F`` (scale the belief by F) or ``LCF=V`` (pin it to V, in report
    units) -> {LCF: ("scale" or "pin", number)}: a wrong encoder head in the
    exported policy (R2's injected faults, beliefs.Beliefs.fault)."""
    out: dict[str, tuple[str, float]] = {}
    for t in texts:
        op = "*" if "*" in t else "=" if "=" in t else None
        if op is None:
            raise ValueError(f"fault {t!r}: write LCF*FACTOR or LCF=VALUE")
        name, number = t.split(op, 1)
        if name not in lcfs:
            raise ValueError(f"fault {t!r}: {name} is not among net1's LCFs {list(lcfs)}")
        out[name] = ("scale" if op == "*" else "pin", float(number))
    return out


# --- expliCCit's released checkpoints -----------------------------------------


def _sklearn_minmax(path: Path, cols: tuple[int, ...]) -> Scaler:
    """sklearn's MinMaxScaler from its parameters (x * scale_ + min_)."""
    p = json.loads(path.read_text())
    scale = np.array(p["scale_"], np.float64)
    center = -np.array(p["min_"], np.float64) / scale
    n = scale.size
    big = np.full(n, 3.0e38, np.float32)
    f = np.float32
    return Scaler(cols, -big, big, center.astype(f), (1 / scale).astype(f))


def load_expliccit(models: Path) -> tuple[Net1, nn.Module]:
    """expliCCit's net1 and actor (sage_rl/offline/models: best_model.pth,
    actor.pth and the scaler JSON files), loaded without running pickles."""
    scaler = _sklearn_minmax(models / "x_scaler_params.json", EXPLICCIT_OBS_COLS)
    lcf_scaler = _sklearn_minmax(models / "y_scaler_params.json", tuple(range(len(EXPLICCIT_LCFS))))
    net = ExpliccitNet1(len(EXPLICCIT_OBS_COLS), len(EXPLICCIT_LCFS))
    net.load_state_dict(
        torch.load(models / "best_model.pth", map_location="cpu", weights_only=True)
    )
    actor = ExpliccitActor(len(EXPLICCIT_LCFS))
    actor.load_state_dict(torch.load(models / "actor.pth", map_location="cpu", weights_only=True))
    return Net1(scaler, net.eval(), lcf_scaler, EXPLICCIT_LCFS), actor.eval()
