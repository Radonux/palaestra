"""A rollout's recorded steps -> each agent flow's decisions (Trajectory):
the actor's inputs through the exported front end, the realized u, the
reward between decisions, the value network's inputs and, for DAgger,
the teacher's labels."""

# torch's own annotations leave many signatures partly unknown (from_numpy,
# manual_seed); every other strict check stays on.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn

from expliccit_env.policy import OBS_DIM
from expliccit_env.train import fit, score
from expliccit_env.train.actors import Rules, front, target, teacher3_u, teacher_u
from expliccit_env.train.graph import Front
from expliccit_env.train.handoff import PolicyTeacher
from expliccit_env.train.lcfs import Labels, columns
from expliccit_env.train.mixture import active, realized
from expliccit_env.train.recurrent import OBS_MIN_RTT, decision_steps
from expliccit_env.train.spec import LCF_UNITS, Gate3, Net2Spec, PPOSpec

type F32 = npt.NDArray[np.float32]
# Per-step reward: the validation tables' log utility (delay weight 0.5,
# loss weight 4).
REWARD: Net2Spec = Net2Spec(reward="log_utility")
STEP_S: Final = 0.01  # the guest's step interval
UTILITY_LABELS = ("receive_rate_bps", "capacity_bps", "queue_delay_mean_ns", "base_rtt_ns")


@dataclass(frozen=True, slots=True)
class Trajectory:
    """One flow's decisions in one rollout."""

    group: tuple[int, int]  # (link of the iteration, flow: 0 observed, i + 1 cross i)
    inp: F32  # [T, width + 1] the actor's inputs: net1's view, previous u / max
    critic: F32  # [T, C] the value network's inputs
    u: F32  # [T] realized u of each decision
    reward: F32  # [T] the block's summed per-step reward after the decision
    gamma: F32  # [T] exp(-dt / tau) to the next decision
    base: F32 | None = None  # [T] the rule's u the actor's output adds to (base "share")
    teacher: F32 | None = None  # [T] the privileged teacher's u (algo "dagger")
    tcp: F32 | None = None  # [T] 1 while a stock TCP flow is active (algo "dagger")
    share: F32 | None = None  # [T] the share rule's u (algo "dagger", gate students)
    copies: F32 | None = None  # [T] 1 while a copy of the policy is active (algo "dagger")
    coop: F32 | None = None  # [T] the cooperative window's u (algo "dagger", gate students)


def _read(path: Path, eids: set[str], lcfs: tuple[str, ...]) -> pa.Table | None:
    if not path.exists():
        return None
    labels = {f"label_{n}" for n in (*UTILITY_LABELS, "drop_fraction", *columns(lcfs))}
    names = pq.read_schema(path).names
    cols = [c for c in names if c in {"episode_id", "flow", "step", "t_ns", "obs", "cwnd"} | labels]
    t = pq.read_table(path, columns=cols)
    keep = np.isin(np.asarray(t["episode_id"].to_pylist()), list(eids))
    return t.filter(pa.array(keep))


def _labels(t: pa.Table) -> Labels:
    return Labels(
        {
            c.removeprefix("label_"): np.asarray(t[c].to_numpy(zero_copy_only=False), np.float32)
            for c in t.column_names
            if c.startswith("label_")
        }
    )


def _by_step(t: pa.Table) -> pa.Table:
    order = np.argsort(np.asarray(t["step"].to_numpy(zero_copy_only=False)), kind="stable")
    return t.take(pa.array(order, pa.int64()))


def team_utility(utilities: list[npt.NDArray[np.float32]]) -> list[npt.NDArray[np.float32]]:
    """Each flow's per-step utility replaced by the flows' mean at the same
    step (copies start together, so step k is the same instant for all);
    a step only some flows reached averages those."""
    n = max(u.size for u in utilities)
    stack = np.stack([np.pad(u, (0, n - u.size), constant_values=np.nan) for u in utilities])
    mean = np.nanmean(stack, axis=0).astype(np.float32)
    return [mean[: u.size] for u in utilities]


@torch.no_grad()
def trajectory(
    t: pa.Table,
    utility: npt.NDArray[np.float32],
    net1: fit.Net1,
    *,
    front: Front,
    target: nn.Module,
    rules: Rules,
    spec: PPOSpec,
    group: tuple[int, int],
    rivals: npt.NDArray[np.float64],
    teacher: PolicyTeacher | None = None,
) -> Trajectory | None:
    """One flow's decisions; ``t`` is its steps in step order, ``utility``
    the reward of each of those steps, ``rivals`` its episode's cross
    flows (rivals_of) and ``teacher`` DAgger's teacher where it is a policy
    on the truth (PPOSpec.teacher_from). The actor's inputs and the window
    target come from the exported graph's own front end and target."""
    n = t.num_rows
    flat = cast("pa.FixedSizeListArray[Any, Any]", t["obs"].combine_chunks()).flatten()
    obs = np.asarray(flat, np.float32).reshape(n, OBS_DIM)
    cwnd = np.asarray(t["cwnd"].to_numpy(zero_copy_only=False), np.float32)
    t_s = np.asarray(t["t_ns"].to_numpy(zero_copy_only=False), np.float64) / 1e9
    labels = _labels(t)
    d = decision_steps(obs[:, OBS_MIN_RTT])
    d = d[d + 1 < n]  # a decision's realized window is the next step's
    if d.size < 2:
        return None
    y = np.stack([labels[k][d] * np.float32(LCF_UNITS[k][0]) for k in net1.lcfs], axis=1)
    o = torch.from_numpy(obs[d])
    z, xt, traffic = front(o, torch.from_numpy(y) if front.uses_truth else None)
    x = xt.numpy()
    lb = target(o, z, traffic).numpy()
    u = (np.log2(np.maximum(cwnd[d + 1], 1.0)) - lb).astype(np.float32)
    # The rules' u on the actor's beliefs, as its head reads them in the graph.
    ruled = rules(z).numpy().astype(np.float32) if spec.inputs != "raw" else None
    base = ruled[:, 0] if ruled is not None and spec.base == "share" else None
    prev = np.concatenate([[0.0], u[:-1]]).astype(np.float32)
    # Each step's reward weighs 1 - exp(-10 ms / tau), so a return is a
    # discounted mean utility, of order 1 whatever tau is.
    r = utility * -math.expm1(-STEP_S / spec.tau_s)
    ends = np.append(d[1:], n - 1)
    cum = np.concatenate([[0.0], np.cumsum(r)])
    reward = (cum[ends + 1] - cum[d + 1]).astype(np.float32)  # steps d + 1 .. next decision
    gamma = np.exp(-(t_s[ends] - t_s[d]) / spec.tau_s).astype(np.float32)
    frac = (np.arange(d.size) / d.size).astype(np.float32)[:, None]
    inp = np.concatenate([x, (prev / spec.action_max)[:, None]], axis=1).astype(np.float32)
    parts = [x, (prev / spec.action_max)[:, None], frac]
    if spec.critic_truth:
        ys = net1.lcf_scaler(torch.from_numpy(y)).numpy()
        parts.insert(0, np.where(np.isfinite(ys), ys, 0.0).astype(np.float32))
    counts = active(rivals, d * STEP_S)
    if spec.critic_flows:
        parts.append(counts)
    critic = np.concatenate(parts, axis=1).astype(np.float32)
    if spec.algo != "dagger" and spec.student.kind == "u":
        return Trajectory(group, inp, critic, u, reward, gamma, base)
    # The teachers read net1's beliefs, also for the black box ("raw"), whose
    # label is then the teacher's window on its own measured BDP.
    z1 = z if spec.inputs == "net1" else torch.from_numpy(net1.beliefs(obs[d]))
    if ruled is None or spec.inputs != "net1":
        ruled = rules(z1).numpy().astype(np.float32)
    share_u, coop_u = ruled[:, 0], ruled[:, 1]
    if spec.algo != "dagger":  # a gate student continued by PPO
        return Trajectory(group, inp, critic, u, reward, gamma, share=share_u, coop=coop_u)
    tcp = (counts[:, 1] > 0).astype(np.float32)
    copies = (counts[:, 0] > 0).astype(np.float32)
    if teacher is not None:  # the window it would set, on the student's BDP
        label = (teacher.window(net1, z1, y, u + lb) - lb).astype(np.float32)
    else:
        label = (
            teacher3_u(share_u, coop_u, counts, spec.u_rule)
            if isinstance(spec.student, Gate3) and spec.student.copies == "coop"
            else teacher_u(share_u, counts, spec.u_rule)
        )
        if spec.inputs == "raw":
            label = (label + fit.window_target(net1).bdp(z1).numpy() - lb).astype(np.float32)
    return Trajectory(
        group, inp, critic, u, reward, gamma, base, label, tcp, share_u, copies, coop_u
    )


def from_rollouts(
    rollouts: Path,
    groups: dict[str, int],
    net1: fit.Net1,
    spec: PPOSpec,
    rivals: dict[str, npt.NDArray[np.float64]],
    *,
    teacher: PolicyTeacher | None = None,
) -> list[Trajectory]:
    """Every agent flow of every ok rollout episode."""
    fe, tgt, rules = front(net1, spec), target(net1, spec), Rules(net1, spec).eval()
    out: list[Trajectory] = []
    flows_of: dict[str, list[tuple[int, pa.Table]]] = {}
    connected: dict[str, list[float | None]] = {}
    for shard in sorted(rollouts.glob("shards/*")):
        # A shard whose episodes all failed has no column of results.
        eps = [
            r for r in pq.read_table(shard / "episodes.parquet").to_pylist() if r["status"] == "ok"
        ]
        ok = {str(r["episode_id"]) for r in eps}
        connected |= {str(r["episode_id"]): r["cross_connected_ms"] for r in eps}
        for name, flow in (("steps.parquet", False), ("cross_steps.parquet", True)):
            t = _read(shard / name, ok, net1.lcfs)
            if t is None or t.num_rows == 0:
                continue
            eids = np.asarray(t["episode_id"].to_pylist())
            flows = np.asarray(t["flow"].to_pylist()) + 1 if flow else np.zeros(len(eids), int)
            for e in np.unique(eids):
                for f in np.unique(flows[eids == e]):
                    rows = np.flatnonzero((eids == e) & (flows == f))
                    flows_of.setdefault(str(e), []).append(
                        (int(f), _by_step(t.take(pa.array(rows, pa.int64()))))
                    )
    # Only agent flows record steps, so an episode with several is copies.
    for e, fl in flows_of.items():
        utils = [np.nan_to_num(score.utility(_labels(t), REWARD)) for _, t in fl]
        if spec.copies_reward == "team" and len(fl) > 1:
            utils = team_utility(utils)
        for (f, t), u in zip(fl, utils, strict=True):
            tr = trajectory(
                t,
                u,
                net1,
                front=fe,
                target=tgt,
                rules=rules,
                spec=spec,
                group=(groups[e], f),
                rivals=realized(rivals[e], connected[e]),
                teacher=teacher,
            )
            if tr is not None:
                out.append(tr)
    return out
