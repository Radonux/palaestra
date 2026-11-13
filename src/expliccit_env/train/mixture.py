"""The links a closed-loop learner trains on (ppo.py): the traffic each
link gets (alone, a stock competitor or copies of the policy, traffic that
hides the capacity, a rising capacity), the K rollouts of one link, and
the other flows' counts the value network may read."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import numpy.typing as npt

from expliccit_env.spec import (
    MAX_CROSS,
    STOCK_NAMES,
    AgentCC,
    Condition,
    EpisodeSpec,
    OnnxPolicy,
    PathSpec,
    PeriodicLevels,
    StockName,
    TcpCross,
    Trace,
    UdpCross,
)
from expliccit_env.train.spec import PPOSpec


def agent(policy: Path, sigma: float, noise_seed: int | None, hold: int = 1) -> AgentCC:
    sha = hashlib.sha256(policy.read_bytes()).hexdigest()
    return AgentCC(
        policy=OnnxPolicy(path=policy, sha256=sha),
        noise_sigma=sigma,
        noise_seed=noise_seed,
        noise_hold_steps=hold,
    )


BROAD_START_MS: Final = 10_000
BROAD_SECOND: Final[tuple[StockName, ...]] = ("cubic", "bbr", "reno")
HELD_OUT: Final[frozenset[StockName]] = frozenset({"bbr3", "westwood"})
# The Internet's common controllers more often, as examples/net1_broad.py.
DIVERSE_WEIGHT: Final[dict[StockName, float]] = {
    n: {"cubic": 6.0, "bbr": 4.0, "reno": 2.0}.get(n, 1.0) for n in STOCK_NAMES if n not in HELD_OUT
}


def diverse(rng: np.random.Generator) -> StockName:
    names = list(DIVERSE_WEIGHT)
    w = np.array([DIVERSE_WEIGHT[n] for n in names])
    return names[int(rng.choice(len(names), p=w / w.sum()))]


def stock_window(rng: np.random.Generator, duration_ms: int) -> TcpCross:
    """A stock flow's lifetime under competition "dynamic", as
    examples/net1_broad.py's window: from 0 s (35%) or a uniform time in
    [1 s, 60% of the episode], and 30% of the time leaving at a uniform
    time at least 5 s later; the controller is drawn from DIVERSE_WEIGHT."""
    cc = diverse(rng)
    start = 0 if rng.random() < 0.35 else int(rng.uniform(1000, 0.6 * duration_ms))
    if rng.random() < 0.3 and start + 5000 < duration_ms:
        return TcpCross(cc=cc, start_ms=start, stop_ms=int(rng.uniform(start + 5000, duration_ms)))
    return TcpCross(cc=cc, start_ms=start)


def competition(
    rng: np.random.Generator,
    copies: tuple[int, ...],
    copy: AgentCC | None,
    competition: Literal["at-start", "broad", "diverse", "dynamic"] = "at-start",
    duration_ms: int = 30_000,
    *,
    copies_share: float | None = None,
) -> list[TcpCross | UdpCross]:
    """Cubic or BBR, or n - 1 copies of the policy (n in copies) from 0 s;
    with competition "broad", stock flows start within BROAD_START_MS and a
    second one joins half the time; with "diverse", as "broad" with each
    stock flow drawn from DIVERSE_WEIGHT; with "dynamic", each stock flow
    gets a lifetime anywhere in the episode (stock_window)."""
    if copies_share is None:
        choices = ["cubic", "bbr", *(f"copies{n}" for n in copies if copy is not None)]
        pick = choices[int(rng.integers(len(choices)))]
    elif copy is not None and copies and rng.random() < copies_share:
        pick = f"copies{copies[int(rng.integers(len(copies)))]}"
    else:
        pick = ("cubic", "bbr")[int(rng.integers(2))]
    if not pick.startswith("copies"):
        stock: StockName = "cubic" if pick == "cubic" else "bbr"
        if competition == "at-start":
            return [TcpCross(cc=stock, start_ms=0)]
        if competition == "dynamic":
            first = stock_window(rng, duration_ms)
            return [first, stock_window(rng, duration_ms)] if rng.random() < 0.5 else [first]
        if competition == "diverse":
            stock = diverse(rng)
        flows: list[TcpCross | UdpCross] = [
            TcpCross(cc=stock, start_ms=int(rng.integers(0, BROAD_START_MS + 1)))
        ]
        if rng.random() < 0.5:
            second = (
                diverse(rng)
                if competition == "diverse"
                else BROAD_SECOND[int(rng.integers(len(BROAD_SECOND)))]
            )
            flows.append(TcpCross(cc=second, start_ms=int(rng.integers(0, BROAD_START_MS + 1))))
        return flows
    assert copy is not None
    n = int(pick.removeprefix("copies"))
    return [TcpCross(cc=copy, start_ms=0) for _ in range(n - 1)]


def info_traffic(
    rng: np.random.Generator, trace: Trace, duration_ms: int
) -> list[TcpCross | UdpCross]:
    """Traffic under which the flow cannot see the capacity part of the
    time, one of three alike: a stock flow (DIVERSE_WEIGHT) from 0 s that
    leaves at a uniform time in [20%, 60%] of the episode; UDP at a uniform
    0.3 to 0.7 of the link's mean capacity, on and off for uniform 1 to 4 s
    each; or a stock TCP flow on and off for uniform 2 to 6 s each."""
    kind = int(rng.integers(3))
    if kind == 0:
        stop = int(rng.uniform(0.2, 0.6) * duration_ms)
        return [TcpCross(cc=diverse(rng), start_ms=0, stop_ms=max(stop, 1))]
    lo, hi = (1000, 4000) if kind == 1 else (2000, 6000)
    on, off = int(rng.uniform(lo, hi)), int(rng.uniform(lo, hi))
    starts = range(int(rng.uniform(0, off)), duration_ms - 1, on + off)
    windows = [(t, min(t + on, duration_ms)) for t in starts][:MAX_CROSS]
    if kind == 1:
        rate = round(float(rng.uniform(0.3, 0.7)) * mean_mbps(trace), 3)
        return [UdpCross(rate_mbps=rate, start_ms=a, stop_ms=b) for a, b in windows]
    cc = diverse(rng)
    return [TcpCross(cc=cc, start_ms=a, stop_ms=b) for a, b in windows]


def mean_mbps(trace: Trace) -> float:
    ts = trace.timestamps()  # one 1500-byte packet per entry, in ms
    return 12.0 * ts.size / float(ts[-1])


def rising(rng: np.random.Generator, path: PathSpec) -> PathSpec:
    """The link with its capacity alternating between half and twice its
    mean, each level held a uniform 1 to 5 s."""
    m = mean_mbps(path.trace)
    trace = PeriodicLevels(levels_mbps=[m / 2, 2 * m], level_ms=int(rng.uniform(1000, 5000)))
    return path.model_copy(update={"trace": trace})


def group_episodes(
    arm: tuple[PathSpec, Condition],
    policy: Path,
    spec: PPOSpec,
    rng: np.random.Generator,
    draw: float,
) -> list[EpisodeSpec]:
    """K rollouts of one link: the same seed, different noise seeds. The
    uniform ``draw`` picks the link's traffic: alone below spec.alone, then
    competition (competition) for the next spec.compete, else the paths'
    own condition, unless spec.info takes the next share (info_traffic). Then
    spec.rise of the links get a rising capacity (rising)."""
    path, cond = arm
    seed, base = int(rng.integers(0, 2**63)), int(rng.integers(0, 2**62))
    template = agent(policy, spec.noise_sigma, None, spec.noise_hold_steps)
    if spec.rise and rng.random() < spec.rise:
        path = rising(rng, path)
    if draw < spec.alone:
        cross: list[TcpCross | UdpCross] = []
    elif draw < spec.alone + spec.compete:
        cross = competition(
            rng,
            spec.copies,
            template,
            spec.competition,
            spec.duration_ms,
            copies_share=spec.copies_share,
        )
    elif draw < spec.alone + spec.compete + spec.info:
        cross = info_traffic(rng, path.trace, spec.duration_ms)
    else:
        cross = cond.flows(path.trace)

    def own(c: TcpCross | UdpCross, noise: int) -> TcpCross | UdpCross:
        if isinstance(c, TcpCross) and isinstance(c.cc, AgentCC):
            return c.model_copy(update={"cc": c.cc.model_copy(update={"noise_seed": noise})})
        return c

    return [
        EpisodeSpec(
            path=path.model_copy(update={"loss": cond.loss}),
            cross=[own(c, base + k + 7919 * (i + 1)) for i, c in enumerate(cross)],
            cc=template.model_copy(update={"noise_seed": base + k}),
            duration_ms=spec.duration_ms,
            seed=seed,
            steps=True,
        )
        for k in range(spec.group)
    ]


# The kinds of other flow on the link, in the critic's counts.
RIVALS: Final = ("copy", "tcp", "udp")


def rivals_of(e: EpisodeSpec) -> npt.NDArray[np.float64]:
    """An episode's cross flows as rows: kind (index in RIVALS), start
    and stop in seconds."""

    def kind(c: TcpCross | UdpCross) -> int:
        if isinstance(c, UdpCross):
            return 2
        return 0 if isinstance(c.cc, AgentCC) else 1

    rows = [
        (kind(c), c.start_ms / 1000, (e.duration_ms if c.stop_ms is None else c.stop_ms) / 1000)
        for c in e.cross
    ]
    return np.array(rows, np.float64).reshape(-1, 3)


def realized(
    rivals: npt.NDArray[np.float64], connected_ms: list[float | None]
) -> npt.NDArray[np.float64]:
    """An episode's cross flows as they ran (the episodes table's
    cross_connected_ms): a TCP flow starts when its handshake completed,
    not at its planned start, and one that never connected is no rival.
    Stops are exact, since the guest aborts a flow at its stop."""
    if len(connected_ms) != len(rivals):
        raise ValueError(f"{len(connected_ms)} connect times for {len(rivals)} cross flows")
    out = rivals.copy()
    keep = np.ones(len(rivals), bool)
    for i, t in enumerate(connected_ms):
        if rivals[i, 0] == RIVALS.index("udp"):
            continue
        if t is None:
            keep[i] = False
        else:
            out[i, 1] = t / 1000
    return out[keep]


def batch(
    arms: list[tuple[PathSpec, Condition]], policy: Path, spec: PPOSpec, rng: np.random.Generator
) -> tuple[list[EpisodeSpec], dict[str, int], dict[str, npt.NDArray[np.float64]]]:
    """One iteration's rollouts: links_per_iteration links, K each, each
    episode's link index (its group), and its cross flows (rivals_of)."""
    eps: list[EpisodeSpec] = []
    groups: dict[str, int] = {}
    rivals: dict[str, npt.NDArray[np.float64]] = {}
    for j in range(spec.links_per_iteration):
        arm = arms[int(rng.integers(len(arms)))]
        for e in group_episodes(arm, policy, spec, rng, float(rng.random())):
            groups[e.episode_id()] = j
            rivals[e.episode_id()] = rivals_of(e)
            eps.append(e)
    return eps, groups, rivals


def active(rivals: npt.NDArray[np.float64], times: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
    """At each time (s into the episode), how many cross flows of each
    kind in RIVALS are active, over 4."""
    on = (rivals[None, :, 1] <= times[:, None]) & (times[:, None] < rivals[None, :, 2])
    kinds = rivals[:, 0].astype(int)
    counts = [(on & (kinds == k)).sum(axis=1) for k in range(len(RIVALS))]
    return (np.stack(counts, axis=1) / 4).astype(np.float32)
