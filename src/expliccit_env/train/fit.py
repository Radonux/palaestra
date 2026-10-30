"""Fitting net1 (supervised) and net2 (offline RL on frozen net1), the
held-out report, and expliCCit's released checkpoints as our modules."""

# torch's own annotations leave many signatures partly unknown (from_numpy,
# manual_seed, onnx.export); every other strict check stays on.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from expliccit_env.policy import EXPLICCIT_OBS_COLS
from expliccit_env.train.data import Split, Steps
from expliccit_env.train.models import (
    NET1,
    NET2,
    Critic,
    ExpliccitActor,
    ExpliccitNet1,
    IntervalNet1,
    PolicyGraph,
    Scaler,
    Value,
)
from expliccit_env.train.spec import EXPLICCIT_LCFS, LCF_UNITS, Net1Spec, Net2Spec

type F32 = npt.NDArray[np.float32]
_EVAL_BATCH: Final = 65_536


@dataclass(frozen=True, slots=True)
class Net1:
    scaler: Scaler
    net: nn.Module
    lcf_scaler: Scaler  # targets in report units <-> net1's output space
    lcfs: tuple[str, ...]
    intervals: bool = False  # output: centres, then lower and upper bounds

    @torch.no_grad()
    def beliefs(self, obs: F32) -> F32:
        """net1's output (scaled LCFs), for each row."""
        self.net.eval()
        out = [
            self.net(self.scaler(torch.from_numpy(obs[i : i + _EVAL_BATCH])))
            for i in range(0, obs.shape[0], _EVAL_BATCH)
        ]
        return torch.cat(out).numpy() if out else np.zeros((0, len(self.lcfs)), np.float32)

    def predict(self, obs: F32) -> F32:
        """LCFs in report units (LCF_UNITS): the centres."""
        z = self.beliefs(obs)[:, : len(self.lcfs)]
        return self.lcf_scaler.inverse(torch.from_numpy(np.ascontiguousarray(z))).numpy()

    def bounds(self, obs: F32) -> tuple[F32, F32]:
        """Lower and upper bounds in report units (intervals only)."""
        k, z = len(self.lcfs), self.beliefs(obs)
        inv = self.lcf_scaler.inverse
        lo = inv(torch.from_numpy(np.ascontiguousarray(z[:, k : 2 * k]))).numpy()
        hi = inv(torch.from_numpy(np.ascontiguousarray(z[:, 2 * k :]))).numpy()
        return lo, hi


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
    lo, hi = net1.bounds(steps.obs[rows]) if net1.intervals else (pred, pred)
    busy = steps.labels["busy_fraction"][rows]
    for k, name in enumerate(net1.lcfs):
        out[name] = {"unit": LCF_UNITS[name][1], "net1": _metrics(pred[:, k], true[:, k])}
        if name in base:
            out[name]["baseline"] = _metrics(base[name], true[:, k])
        if net1.intervals:
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


def _mean_where(x: F32, rows: npt.NDArray[np.bool_]) -> float:
    return float(x[rows].mean()) if rows.any() else float("nan")


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
    rng = np.random.default_rng(seed)
    tr, va = steps.rows("train"), steps.rows("val")
    y = targets(steps, spec.lcfs)
    scaler = Scaler.fit(spec.obs_cols, steps.obs[tr][:, spec.obs_cols], spec.scaling)
    lcf_kind = "minmax" if spec.scaling == "minmax" else "standard"
    lcf_scaler = Scaler.fit(tuple(range(len(spec.lcfs))), y[tr], lcf_kind)

    def tensors(rows: npt.NDArray[np.bool_]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            x = scaler(torch.from_numpy(steps.obs[rows]))
            t = lcf_scaler(torch.from_numpy(y[rows]))
        mask = torch.from_numpy(np.isfinite(y[rows]).astype(np.float32))
        return x, t, mask

    (xt, yt, mt), (xv, yv, mv) = tensors(tr), tensors(va)
    net = _net1_module(spec)

    def loss_of(pred: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        if spec.intervals:
            return _masked_pinball(pred, y, m, spec.coverage)
        return _masked_mse(pred, y, m)

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
            loss = loss_of(net(xt[b]), yt[b], mt[b])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            val = float(loss_of(net(xv), yv, mv)) if xv.shape[0] else float("nan")
        history.append(val)
        if val < best:
            best, best_state, stale = val, copy.deepcopy(net.state_dict()), 0
        else:
            stale += 1
            if stale >= spec.patience:
                break
    net.load_state_dict(best_state)
    if isinstance(net, IntervalNet1) and xv.shape[0]:
        _calibrate(net, xv, yv, mv, spec.coverage)
    net1 = Net1(scaler, net, lcf_scaler, tuple(spec.lcfs), spec.intervals)
    return net1, {"val_loss": history, "epochs": len(history)}


@dataclass(frozen=True, slots=True)
class Transitions:
    s: torch.Tensor
    a: torch.Tensor
    r: torch.Tensor
    s2: torch.Tensor

    def parts(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.s, self.a, self.r, self.s2


def transitions(
    steps: Steps, states: F32, spec: Net2Spec, split: Split
) -> tuple[Transitions, float]:
    """(s_t, a_t, r, s_t+1) within episodes; a scaled to [-1, 1] and clipped.
    Returns the fraction of actions that needed clipping."""
    i = np.flatnonzero(steps.rows(split)[:-1] & (steps.episode[:-1] == steps.episode[1:]))
    reward = steps.reward if spec.reward == "sage" else utility(steps, spec)
    r = reward[i + 1] if spec.reward_alignment == "next" else reward[i]
    keep = np.isfinite(steps.action[i]) & np.isfinite(r)
    i, r = i[keep], r[keep]
    a = steps.action[i] / spec.action_max
    clipped = float(np.mean(np.abs(a) > 1)) if a.size else 0.0
    t = Transitions(
        torch.from_numpy(states[i]),
        torch.from_numpy(np.clip(a, -1, 1).reshape(-1, 1)),
        torch.from_numpy((r * spec.reward_scale).astype(np.float32)),
        torch.from_numpy(states[i + 1]),
    )
    return t, clipped


def utility(steps: Steps, spec: Net2Spec) -> F32:
    """Per step: receive rate over capacity, minus the delay and loss terms,
    from ground truth. An empty window (no departures) costs no delay."""
    lab = steps.labels
    with np.errstate(divide="ignore", invalid="ignore"):
        used = lab["receive_rate_bps"] / lab["capacity_bps"]
        delay = np.nan_to_num(lab["queue_delay_mean_ns"] / lab["base_rtt_ns"])
    loss = np.nan_to_num(lab["drop_fraction"])
    r = used - spec.delay_weight * delay - spec.loss_weight * loss
    return np.where(np.isfinite(used), r, np.nan).astype(np.float32)


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
    update returns its losses and diagnostics (docs/training.md, "Tuning")."""

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
            y = r + spec.discount * q2
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
            y = r + spec.discount * self.value(s2)
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


def fit_net2(
    spec: Net2Spec, steps: Steps, net1: Net1, seed: int
) -> tuple[nn.Module, dict[str, Any]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if spec.inputs == "net1":
        states = net1.beliefs(steps.obs)
    else:
        with torch.no_grad():
            z = net1.lcf_scaler(torch.from_numpy(targets(steps, net1.lcfs))).numpy()
        states = np.where(np.isfinite(z), z, 0).astype(np.float32)
    data, clipped = transitions(steps, states, spec, "train")
    if data.s.shape[0] == 0:
        raise ValueError("no training transitions")
    learner = Learner(spec, states.shape[1])
    log: dict[str, list[float]] = {}
    for step in range(spec.steps):
        b = torch.from_numpy(rng.integers(0, data.s.shape[0], spec.batch_size))
        batch = Transitions(data.s[b], data.a[b], data.r[b], data.s2[b])
        for name, v in learner.update(batch, step).items():
            log.setdefault(name, []).append(v)
    actor = learner.actor.eval()
    val, _ = transitions(steps, states, spec, "val")
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
    return Net1(scaler, net.eval(), lcf_scaler, tuple(spec.lcfs), spec.intervals), spec


def _net1_module(spec: Net1Spec) -> nn.Module:
    k, f = len(spec.lcfs), len(spec.obs_cols)
    if not spec.intervals:
        return NET1[spec.arch](f, k, spec.width, spec.dropout)
    return IntervalNet1(NET1[spec.arch](f, k, spec.width, spec.dropout, 3), k)


def policy_graph(
    net1: Net1, actor: nn.Module, action_max: float, action_decimals: int | None = None
) -> PolicyGraph:
    graph = PolicyGraph(net1.scaler, net1.net, net1.lcf_scaler, actor, action_max)
    graph.action_decimals = action_decimals
    return graph.eval()


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
