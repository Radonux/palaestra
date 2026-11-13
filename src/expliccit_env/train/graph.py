"""The deployed policy under contract sage-v1 (policy.py), built from parts.

A policy graph is three parts in a row:

1. ``Front``: the step message, and for an oracle or a patched net1 the
   true LCFs, -> net1's beliefs z, net2's input x and, where a traffic head
   is attached, its probabilities of the other traffic. It is net1 (with
   any LCFs taken from the truth, injected faults, or expliCCit's deployed
   rounding) followed by net2's view; or, for the black box, the scaled
   measurements alone; or both side by side.
2. A target (optional): log2 of the BDP net2's window is aimed at, from
   net1's beliefs (``WindowTarget`` and its probing variants) or from the
   measurements (``MeasuredBdp``). With one, net2's output u is the next
   window as a log2 multiple of it, and the graph needs the sender's cwnd.
3. A decision: ``EveryStep`` asks net2 at every step (expliCCit's and
   Sage's contract, an action a on the window multiplier or a window);
   ``recurrent.Held`` asks a recurrent net2 once per min RTT and holds its
   decision.

Training computes net2's inputs with the same Front and target modules
that are exported (fit.net2_view, ppo._trajectory), so what net2 learns
from is what it reads in the guest.
"""

# torch's own annotations leave many signatures partly unknown; every other
# strict check stays on.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal, Protocol, cast

import torch
from torch import nn

from expliccit_env.policy import OBS_COLUMNS, TRUTH_COLUMNS
from expliccit_env.train import lcfs
from expliccit_env.train.beliefs import Beliefs, Bins, Point
from expliccit_env.train.models import Scaler

A_LIMIT: Final = 7.99  # the contract's |a| <= 8
PACKET_BITS: Final = 1500 * 8  # Mahimahi's packets, in the BDP

type Reads = Literal["beliefs", "measurements", "both"]


class Front(nn.Module):
    """obs [n, 78] (and y [n, K], the true LCFs in report units, where some
    are taken from the truth) -> (z, x, traffic).

    ``reads``: net2 reads net1's beliefs through ``view`` ("beliefs"), the
    measurements net1 reads, scaled by its input scaler ("measurements",
    the black box; z is then those measurements), or both, the view's
    output first ("both"). ``take``: which LCFs come from y when it is
    given. ``fault``: a wrong belief injected after that. ``deployed``:
    expliCCit's deployed rounding of a point net1's beliefs. ``traffic``: a
    head on the scaled measurements stating the other traffic's class."""

    take: torch.Tensor | None

    def __init__(
        self,
        scaler: Scaler,
        net1: nn.Module,
        kind: Beliefs,
        view: nn.Module | None = None,
        *,
        reads: Reads = "beliefs",
        take: Sequence[bool] | None = None,
        fault: nn.Module | None = None,
        deployed: nn.Module | None = None,
        traffic: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.scaler, self.net1, self.kind, self.view = scaler, net1, kind, view
        self.reads: Reads = reads
        self.fault, self.deployed, self.traffic = fault, deployed, traffic
        if take is not None and not any(take):
            take = None
        self.register_buffer("take", None if take is None else torch.tensor(take, dtype=torch.bool))

    @property
    def uses_truth(self) -> bool:
        return self.take is not None

    def beliefs(self, obs: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """net1's output after the truth, faults and rounding."""
        z = self.net1(self.scaler(obs))
        if self.take is not None and y is not None:
            z = self.kind.patch(z, y, self.take)
        if self.fault is not None:
            z = self.fault(z)
        if self.deployed is not None:
            z = self.deployed(z)
        return z

    def forward(
        self, obs: torch.Tensor, y: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        m = self.scaler(obs)
        if self.reads == "measurements":
            return m, m, None
        z = self.beliefs(obs, y)
        x = z if self.view is None else self.view(z)
        if self.reads == "both":
            x = torch.cat([x, m], dim=1)
        return z, x, None if self.traffic is None else self.traffic(m)


class Deployed(nn.Module):
    """expliCCit's deployed controller between its networks: a point net1's
    beliefs back to report units, clamped at 0, the ``rounded`` LCFs rounded
    to integers, standardized again."""

    rounded: torch.Tensor

    def __init__(self, kind: Beliefs, rounded: Sequence[bool]) -> None:
        super().__init__()
        if not isinstance(kind, Point):
            raise ValueError("expliCCit's deployed rounding applies to a point net1")
        self.kind = kind
        self.register_buffer("rounded", torch.tensor(rounded, dtype=torch.bool))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        c = torch.relu(self.kind.means(z))
        c = torch.where(self.rounded, torch.round(c), c)
        return (c - self.kind.center) / self.kind.spread


class TruthColumns(nn.Module):
    """The contract's "truth" input (policy.TRUTH_COLUMNS, label units) ->
    net1's LCFs in report units: a label is its column, a composite its
    formula over the columns (lcfs.value), as the labels compute it. An LCF
    the guest cannot know at the step (``known``) is NaN, which a patch
    reads as no truth."""

    units: torch.Tensor

    def __init__(self, names: Sequence[str]) -> None:
        super().__init__()
        self.names = tuple(names)
        units = [lcfs.LCF_UNITS[n][0] for n in names]
        self.register_buffer("units", torch.tensor(units, dtype=torch.float32))

    @staticmethod
    def known(name: str) -> bool:
        return lcfs.columns([name]) <= set(TRUTH_COLUMNS)

    def forward(self, truth: torch.Tensor) -> torch.Tensor:
        def column(name: str) -> lcfs.F32:
            return cast("lcfs.F32", truth[:, TRUTH_COLUMNS.index(name)])

        nan = torch.full_like(truth[:, 0], float("nan"))
        ys = [
            cast("torch.Tensor", lcfs.value(n, column)) if self.known(n) else nan
            for n in self.names
        ]
        return torch.stack(ys, dim=1) * self.units


@dataclass(frozen=True, slots=True)
class Seen:
    """What a decision sees at a step: the message, net2's input x, net1's
    beliefs z, the traffic head's probabilities, log2 of the target BDP
    (lb) and the sender's window, the last three where the graph has them."""

    obs: torch.Tensor
    x: torch.Tensor
    z: torch.Tensor
    traffic: torch.Tensor | None
    lb: torch.Tensor | None
    cwnd: torch.Tensor | None


class Decision(Protocol):
    """What PolicyGraph needs of its decision (EveryStep, recurrent.Held)."""

    @property
    def state_dim(self) -> int: ...


class WindowTarget(nn.Module):
    """log2 of the believed BDP in packets: the mean capacity (Mbit/s) times
    twice the mean one-way delay (ms), at least one packet. Any kind of
    net1 with both LCFs."""

    def __init__(self, kind: Beliefs, capacity: int, one_way: int) -> None:
        super().__init__()
        self.kind, self.capacity, self.one_way = kind, capacity, one_way

    def bdp(self, z: torch.Tensor) -> torch.Tensor:
        mean = self.kind.means(z)
        bdp = mean[:, self.capacity] * 1e6 * mean[:, self.one_way] * 2e-3 / PACKET_BITS
        return torch.log2(torch.clamp(bdp, min=1.0))

    def forward(
        self, obs: torch.Tensor, z: torch.Tensor, traffic: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.bdp(z)


class SpreadTarget(nn.Module):
    """WindowTarget plus k times net1's stated spread of capacity past s0:
    log2 BDP + k relu(sd - s0), sd the standard deviation of log2 capacity
    under a distribution net1's belief. Where the flow keeps the link busy,
    net1's spread is small (median 0.14 on validation) and this is
    WindowTarget; where it cannot see the capacity, the spread widens and
    the window probes above the believed BDP in proportion. With
    ``copies_gated``, the lift is scaled by 1 - P(copies) from the traffic
    head, so the flow does not probe among copies of itself, where a push
    only deepens the queue."""

    log_mid: torch.Tensor

    def __init__(
        self, base: WindowTarget, k: float, s0: float, *, copies_gated: bool = False
    ) -> None:
        super().__init__()
        if not isinstance(base.kind, Bins):
            raise ValueError("a spread target needs a distribution net1")
        self.base, self.k, self.s0, self.copies_gated = base, k, s0, copies_gated
        self.kind: Bins = base.kind
        mid = self.kind.mid[base.capacity]
        self.register_buffer("log_mid", torch.log2(torch.clamp(mid, min=1e-3)))

    def forward(
        self, obs: torch.Tensor, z: torch.Tensor, traffic: torch.Tensor | None = None
    ) -> torch.Tensor:
        p = self.kind.probs(z)[:, self.base.capacity, :]
        mean = (p * self.log_mid).sum(dim=1, keepdim=True)
        sd = torch.sqrt(torch.clamp((p * (self.log_mid - mean) ** 2).sum(dim=1), min=0.0))
        lift = self.k * torch.relu(sd - self.s0)
        if self.copies_gated:
            if traffic is None:
                raise ValueError("a copies-gated target needs the traffic head")
            lift = lift * (1 - traffic[:, _COPIES])
        return self.base.bdp(z) + lift


class IdleTarget(nn.Module):
    """WindowTarget plus k times how idle net1 believes the link is, past
    i0: log2 BDP + k relu(1 - busy - i0), busy the mean of its busy-fraction
    belief. The arm on means that probes where the flow cannot fill the
    link, beside SpreadTarget, which probes on the stated spread."""

    def __init__(self, base: WindowTarget, busy: int, k: float, i0: float) -> None:
        super().__init__()
        self.base, self.busy, self.k, self.i0 = base, busy, k, i0

    def forward(
        self, obs: torch.Tensor, z: torch.Tensor, traffic: torch.Tensor | None = None
    ) -> torch.Tensor:
        busy = self.base.kind.means(z)[:, self.busy]
        return self.base.bdp(z) + self.k * torch.relu(1 - busy - self.i0)


# BBR's bottleneck bandwidth: the kernel's per-ACK delivery rate
# (tcpi_delivery_rate), its maximum over the last 200 steps (2 s). Not
# max_delivered_x100mbps: that is a 1 s average, which trails a growing
# window by about 0.5 s, so a window aimed just above it grows by 19% per
# 0.5 s instead of per RTT (round 19's first black box started 2.2 below
# the rule and collapsed).
MAX_DELIVERED: Final = OBS_COLUMNS.index("delivery_rate_x100mbps_mid_max")
MIN_RTT_MS: Final = OBS_COLUMNS.index("min_rtt_ms")
_COPIES: Final = 1  # traffic.CLASSES.index("copies")


class MeasuredBdp(nn.Module):
    """The black box's target: log2 of BBR's BDP estimate in packets, the
    maximum delivery rate over the last 2 s times the minimum RTT, at least
    one packet."""

    def forward(
        self, obs: torch.Tensor, z: torch.Tensor, traffic: torch.Tensor | None = None
    ) -> torch.Tensor:
        bits = obs[:, MAX_DELIVERED] * 1e8 * obs[:, MIN_RTT_MS] * 1e-3
        # Not nan_to_num: its IsInf is an op the guest's tract cannot run.
        bits = torch.where(torch.abs(bits) < 3.0e38, bits, torch.zeros_like(bits))
        return torch.log2(torch.clamp(bits / PACKET_BITS, min=1.0))


class EveryStep(nn.Module):
    """net2 asked at every step: a = net2(x) * action_max, or with a target
    the window log2 BDP + u, as a = that - log2 cwnd within the contract's
    bound. A net2 with memory declares ``state_dim`` and maps (x, state) to
    (u, next state). ``decimals`` rounds a (expliCCit's deployed controller
    uses 3)."""

    def __init__(self, net2: nn.Module, action_max: float, decimals: int | None = None) -> None:
        super().__init__()
        self.net2, self.action_max, self.decimals = net2, action_max, decimals

    @property
    def state_dim(self) -> int:
        return int(getattr(self.net2, "state_dim", 0))

    def forward(
        self, seen: Seen, state: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if state is None:
            a = self.net2(seen.x) * self.action_max
        else:
            u, state = self.net2(seen.x, state)
            a = u * self.action_max
        if seen.lb is not None:
            if seen.cwnd is None:
                raise ValueError("a window target needs the cwnd input")
            a = seen.lb.unsqueeze(1) + a - torch.log2(torch.clamp(seen.cwnd, min=1.0))
            a = torch.clamp(a, -A_LIMIT, A_LIMIT)
        if self.decimals is None:
            return a, state
        scale = float(10**self.decimals)
        return torch.round(a * scale) / scale, state


class PolicyGraph(nn.Module):
    """Front, target, decision (module docstring); ``truth`` maps the
    contract's truth input to the true LCFs the front takes."""

    def __init__(
        self,
        front: Front,
        decide: nn.Module,
        target: nn.Module | None = None,
        truth: TruthColumns | None = None,
    ) -> None:
        super().__init__()
        if front.uses_truth != (truth is not None):
            raise ValueError("a front that takes LCFs from the truth needs the truth's columns")
        self.front, self.decide, self.target, self.truth = front, decide, target, truth

    @property
    def uses_truth(self) -> bool:
        return self.truth is not None

    @property
    def state_dim(self) -> int:
        """S of the contract's "state" input, 0 for a net2 without memory."""
        return cast(Decision, self.decide).state_dim

    def beliefs(self, obs: torch.Tensor, truth: torch.Tensor | None = None) -> torch.Tensor:
        y = None if self.truth is None or truth is None else self.truth(truth)
        return self.front.beliefs(obs, y)

    def step(
        self,
        obs: torch.Tensor,
        state: torch.Tensor | None,
        truth: torch.Tensor | None = None,
        cwnd: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """a and, for a net2 with memory, its next state."""
        y = None if self.truth is None or truth is None else self.truth(truth)
        z, x, traffic = self.front(obs, y)
        lb = None if self.target is None else self.target(obs, z, traffic)
        return self.decide(Seen(obs, x, z, traffic, lb, cwnd), state)

    def forward(
        self,
        obs: torch.Tensor,
        truth: torch.Tensor | None = None,
        cwnd: torch.Tensor | None = None,
    ) -> torch.Tensor:
        state = torch.zeros(obs.shape[0], self.state_dim) if self.state_dim else None
        return self.step(obs, state, truth, cwnd)[0]
