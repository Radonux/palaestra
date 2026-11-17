"""What the recurrent net2 computes (ppo.py trains it).

Its front end and window target (graph.py), which training also reads
its inputs through; the head that turns the actor's output into u (u
itself, a residual on the share rule, or a gate that mixes rules by the
student's beliefs), the same in every learner and in the exported graph;
the rules' teachers DAgger labels decisions with; and the exported graph.
"""

# torch's own annotations leave many signatures partly unknown (from_numpy,
# manual_seed); every other strict check stays on.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

from typing import Final, Literal

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from expliccit_env.train import fit
from expliccit_env.train.graph import Front, MeasuredBdp, PolicyGraph, SpreadTarget
from expliccit_env.train.recurrent import ConcatActor, EnsembleActor, Held, RecurrentActor
from expliccit_env.train.share import ACTION_MAX as SHARE_ACTION_MAX
from expliccit_env.train.share import CoopRule, ShareRule
from expliccit_env.train.spec import Gate, Gate3, PPOSpec, U
from expliccit_env.train.traffic import CLASSES

type F32 = npt.NDArray[np.float32]


P0: Final = 0.02  # a gate student's beliefs at the start


def policy_u(
    spec: PPOSpec,
    a: torch.Tensor,
    share: torch.Tensor | None = None,
    coop: torch.Tensor | None = None,
    base: torch.Tensor | None = None,
) -> torch.Tensor:
    """The policy's mean u from its actor's output a in [-1, 1]: the one
    head every learner trains and the exported graph runs, so a policy
    taught by one learner continues under another unchanged. Student "u":
    a times action_max, plus the base rule's u where one is given. "gate"
    and "gate3": beliefs (a + 1) / 2 past their thresholds (a [..., 2] for
    "gate3"), mixing the share rule's u and the cooperative window's."""
    match spec.student:
        case U():
            u = a * spec.action_max
            return u if base is None else u + base
        case Gate(threshold=t):
            if share is None:
                raise ValueError('student "gate" needs the share rule\'s u at each decision')
            return gated_u(sharpen((a + 1) / 2, t), share, spec.u_rule)
        case Gate3(threshold=t, threshold_q=tq, copies=copies):
            if share is None or coop is None:
                raise ValueError('student "gate3" needs the rules\' u at each decision')
            q_t = t if tq is None else tq
            p, q = sharpen((a[..., 0] + 1) / 2, t), sharpen((a[..., 1] + 1) / 2, q_t)
            return gated3_u(p, q, share, coop, spec.u_rule, copies=copies)


def gated_u(p: torch.Tensor, share: torch.Tensor, u_rule: float) -> torch.Tensor:
    """A "gate" student's u: the fixed rule's, moved toward the share
    rule's by the believed probability p of a stock TCP flow."""
    return u_rule + p * (share - u_rule)


def gated3_u(
    p: torch.Tensor,
    q: torch.Tensor,
    share: torch.Tensor,
    coop: torch.Tensor,
    u_rule: float,
    *,
    copies: Literal["coop", "rule"] = "coop",
) -> torch.Tensor:
    """A "gate3" student's u: toward the share rule by p, the belief in a
    stock TCP flow, and otherwise toward the cooperative window by q, the
    belief in copies; with copies "rule", q vetoes the push instead."""
    if copies == "rule":
        return u_rule + p * (1 - q) * (share - u_rule)
    return u_rule + p * (share - u_rule) + (1 - p) * q * (coop - u_rule)


def sharpen(p: torch.Tensor, t: float) -> torch.Tensor:
    return torch.clamp((p - t) / (1 - t), 0.0, 1.0) if t > 0 else p


class Rules(nn.Module):
    """net1's beliefs z -> [n, 2]: the share rule's u and the cooperative
    window's u (train/share.py), both on net1's standardized means, from any
    kind of net1. What the gate students mix, and what the rule teachers
    read."""

    def __init__(self, net1: fit.Net1, spec: PPOSpec) -> None:
        super().__init__()
        self.kind = net1.kind
        self.share = ShareRule(net1, spec.u_rule, spec.share_k)
        self.coop = CoopRule(net1, spec.u_rule)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        m = self.kind.scaled_means(z)
        return torch.cat([self.share(m), self.coop(m)], dim=1) * SHARE_ACTION_MAX


class Head(nn.Module):
    """A Held decision's head (recurrent.py): the actor's output a in [-1, 1]
    (with net1's beliefs z and the traffic head's probabilities) -> u, by
    policy_u, so the exported graph sets the u every learner trained. With
    ``traffic_q`` (or ``traffic_p``), a "gate3" student's belief in copies
    (or in a stock flow) comes from the traffic head instead
    (train/traffic.py)."""

    def __init__(
        self,
        net1: fit.Net1,
        spec: PPOSpec,
        *,
        traffic_q: bool = False,
        traffic_p: bool = False,
    ) -> None:
        super().__init__()
        if (traffic_q or traffic_p) and spec.student.kind != "gate3":
            raise ValueError('the traffic head stands in for a "gate3" student\'s beliefs only')
        self.spec, self.traffic_q, self.traffic_p = spec, traffic_q, traffic_p
        needs = spec.student.kind != "u" or spec.base == "share"
        self.rules = Rules(net1, spec) if needs else None

    def forward(
        self, a: torch.Tensor, z: torch.Tensor, traffic: torch.Tensor | None
    ) -> torch.Tensor:
        if self.traffic_q or self.traffic_p:
            if traffic is None:
                raise ValueError("this head reads the traffic head's probabilities")
            p = 2 * traffic[:, CLASSES.index("stock")] - 1 if self.traffic_p else a[:, 0]
            a = torch.stack([p, 2 * traffic[:, CLASSES.index("copies")] - 1], dim=1)
        if self.rules is None:
            return policy_u(self.spec, a)
        r = self.rules(z)
        if self.spec.student.kind == "u":  # base "share": a residual on the share rule
            return policy_u(self.spec, a, base=r[:, 0])
        return policy_u(self.spec, a, r[:, 0], r[:, 1])


class SpreadHead(nn.Module):
    """A "gate3" student's Head plus k relu(sd - s0) (1 - p): net1's stated
    spread of capacity (graph.SpreadTarget.spread) past s0, times k, in
    proportion to the student's belief 1 - p that no stock TCP flow is
    active. With k < 0 a wide belief lowers the window when the flow is
    alone, where it means capacity is overstated, or among copies, where it
    means a shared link; beside a buffer-filling flow, where it does
    neither, p keeps the window (exp-rule-spread-v7-val-287)."""

    def __init__(self, head: Head, net1: fit.Net1, spec: PPOSpec, k: float, s0: float) -> None:
        super().__init__()
        if not isinstance(spec.student, Gate3) or head.traffic_p:
            raise ValueError('the spread is scaled by a "gate3" student\'s own belief p')
        self.head, self.k, self.threshold = head, k, spec.student.threshold
        self.spread = SpreadTarget(fit.window_target(net1), 1.0, s0)

    def forward(
        self, a: torch.Tensor, z: torch.Tensor, traffic: torch.Tensor | None
    ) -> torch.Tensor:
        no_stock = 1 - sharpen((a[:, 0] + 1) / 2, self.threshold)
        wide = torch.relu(self.spread.spread(z) - self.spread.s0)
        return self.head(a, z, traffic) + self.k * wide * no_stock


def teacher3_u(share_u: F32, coop_u: F32, counts: F32, u_rule: float) -> F32:
    """Student "gate3"'s teacher: the share rule's u while a stock TCP flow
    is active, else the cooperative window's while a copy is, else the
    fixed rule's."""
    rest = np.where(counts[:, 0] > 0, coop_u, np.float32(u_rule))
    return np.where(counts[:, 1] > 0, share_u, rest).astype(np.float32)


def teacher_u(share_u: F32, counts: F32, u_rule: float) -> F32:
    """The teacher of students "u" and "gate" at each decision: the share
    rule's u while a stock TCP flow is active (counts from ``active``), the
    fixed rule's otherwise."""
    return np.where(counts[:, 1] > 0, share_u, np.float32(u_rule)).astype(np.float32)


def front(net1: fit.Net1, spec: PPOSpec, *, traffic: nn.Module | None = None) -> Front:
    """The actor's front end (graph.Front), shared by training and the
    exported graph: net1's beliefs ("net1"), the truth's ("truth", an
    oracle), or the measurements alone ("raw", the black box)."""
    return fit.front(
        net1,
        inputs=spec.inputs,
        view=spec.view,
        input_bins=spec.input_bins if spec.view == "bins" else None,
        traffic=traffic,
    )


def target(net1: fit.Net1, spec: PPOSpec) -> nn.Module:
    """The BDP the window is aimed at: net1's belief, or BBR's estimate for
    the black box."""
    return MeasuredBdp() if spec.inputs == "raw" else fit.window_target(net1)


def width(net1: fit.Net1, spec: PPOSpec) -> int:
    return fit.input_width(front(net1, spec))


def new_actor(width: int, spec: PPOSpec) -> RecurrentActor:
    """The spec's actor as it starts: the fixed rule (or, on base "share",
    the share rule), or a gate student believing P0 everywhere."""
    actor = RecurrentActor(
        width, spec.hidden, memory=spec.memory, outputs=2 if spec.student.kind == "gate3" else 1
    )
    if spec.student.kind != "u":  # p = (tanh(head) + 1) / 2 = P0 everywhere
        actor.clone_rule(2 * P0 - 1, 1.0)
    else:
        actor.clone_rule(spec.u_rule if spec.base == "fixed" else 0.0, spec.action_max)
    return actor


def policy_graph(
    net1: fit.Net1,
    actor: RecurrentActor | EnsembleActor | ConcatActor,
    spec: PPOSpec,
    truth: frozenset[str] = frozenset(),
    *,
    traffic: nn.Module | None = None,
    head: nn.Module | None = None,
) -> PolicyGraph:
    """The exported net2: the spec's front end (the LCFs in ``truth`` from
    the guest's live truth instead of net1, all of them for inputs
    "truth"; ``traffic`` a traffic head beside net1), its window target,
    and a Held decision with the spec's Head (or ``head``)."""
    f = front(net1, spec, traffic=traffic)
    if truth:
        if spec.inputs != "net1":
            raise ValueError(f'inputs "{spec.inputs}" take no LCF from the truth one by one')
        f = fit.front(
            net1,
            view=spec.view,
            input_bins=spec.input_bins if spec.view == "bins" else None,
            truth=truth,
            traffic=traffic,
        )
    decide = Held(actor, head if head is not None else Head(net1, spec), spec.action_max)
    columns = fit.truth_columns(net1) if f.uses_truth else None
    return PolicyGraph(f, decide, target(net1, spec), columns).eval()
