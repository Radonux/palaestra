"""Continuing an earlier run (PPOSpec.init_from) and teaching from one
(PPOSpec.teacher_from): the checks that the earlier run computes the same
function here, what it starts from, and an actor on the truth as DAgger's
teacher."""

# torch's own annotations leave many signatures partly unknown (from_numpy,
# manual_seed); every other strict check stays on.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from expliccit_env.train import actors, fit
from expliccit_env.train.recurrent import RecurrentActor
from expliccit_env.train.spec import PPOSpec

type F32 = npt.NDArray[np.float32]


# What an actor's function depends on besides its weights, and what a value
# network's inputs do: a run continued from another (init_from) must match.
ACTOR_FIELDS: Final = (
    "view",
    "input_bins",
    "hidden",
    "memory",
    "student",
    "base",
    "inputs",
    "traffic_head",
    "action_max",
)
CRITIC_FIELDS: Final = ("critic_truth", "critic_flows", "critic_hidden")


def same_net1(a: fit.Net1, b: fit.Net1) -> bool:
    def parts(n: fit.Net1) -> list[dict[str, torch.Tensor]]:
        return [n.scaler.state_dict(), n.lcf_scaler.state_dict(), n.net.state_dict()]

    return a.lcfs == b.lcfs and all(
        x.keys() == y.keys() and all(torch.equal(x[k], y[k]) for k in x)
        for x, y in zip(parts(a), parts(b), strict=True)
    )


def earlier_run(run: Path, net1: fit.Net1) -> PPOSpec:
    """The spec of an earlier run (its output directory), which must have
    used this net1."""
    spec = PPOSpec.parse((run / "ppo_spec.json").read_text())
    if not same_net1(fit.load_net1(run / "net1.pt")[0], net1):
        raise ValueError(f"{run} used another net1 than {spec.net1_from}'s")
    return spec


def mismatch(a: PPOSpec, b: PPOSpec, fields: tuple[str, ...]) -> list[str]:
    """The fields in which a and b differ; for "student", only its kind
    counts (a gate's thresholds may change between runs)."""

    def key(s: PPOSpec, f: str) -> object:
        return s.student.kind if f == "student" else getattr(s, f)

    return [f for f in fields if key(a, f) != key(b, f)]


@dataclass(frozen=True, slots=True)
class PolicyTeacher:
    """A recurrent actor on the true LCFs (PPOSpec.teacher_from) as
    DAgger's teacher."""

    actor: RecurrentActor
    view: nn.Module | None
    action_max: float

    @staticmethod
    def load(run: Path, net1: fit.Net1) -> PolicyTeacher:
        t = earlier_run(run, net1)
        if t.inputs != "truth" or t.student.kind != "u" or t.base != "fixed":
            raise ValueError(f'{run}: a teacher reads the truth (inputs "truth"), student "u"')
        net = RecurrentActor(actors.width(net1, t), t.hidden, memory=t.memory)
        net.load_state_dict(torch.load(run / "actor.pt", weights_only=True))
        view = fit.view_of(net1, t.view, t.input_bins if t.view == "bins" else None)
        return PolicyTeacher(net.eval(), view, t.action_max)

    @torch.no_grad()
    def window(self, net1: fit.Net1, z: torch.Tensor, y: F32, set_window: F32) -> F32:
        """[T] log2 of the window the teacher would set at each decision of
        one flow, given net1's output z, the true LCFs y (report units) and
        the window the flow set after each decision (log2 packets): it reads
        the truth's bins and, as its previous u, the flow's last window on
        the true BDP."""
        zt = fit.oracle(net1, z, y)
        x = zt if self.view is None else self.view(zt)
        lb = fit.window_target(net1).bdp(zt).numpy()
        prev = np.concatenate([[0.0], (set_window - lb)[:-1]]).astype(np.float32)
        inp = torch.cat([x, torch.from_numpy(prev / self.action_max).unsqueeze(1)], dim=1)
        u = self.actor.sequence(inp.unsqueeze(0))[0].numpy() * self.action_max
        return (lb + u).astype(np.float32)


def init_paths(spec: PPOSpec, source: PPOSpec | None) -> tuple[Path | None, Path | None]:
    """The saved actor and value network the run starts from: init_actor and
    init_critic, or init_from's (``source`` is that run's spec), its value
    network only for "ppo" and where it has one that reads the same inputs."""
    if spec.init_from is None:
        return spec.init_actor, spec.init_critic
    if source is None:
        raise ValueError("init_from needs the earlier run's spec (earlier_run)")
    if bad := mismatch(spec, source, ACTOR_FIELDS):
        raise ValueError(f"init_from {spec.init_from} differs in {', '.join(bad)}")
    critic = spec.init_from / "critic.pt"
    keep = (
        spec.algo == "ppo"
        and source.algo != "td3bc"
        and not mismatch(spec, source, CRITIC_FIELDS)
        and critic.exists()
    )
    return spec.init_from / "actor.pt", critic if keep else None
