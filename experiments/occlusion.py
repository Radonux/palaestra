"""How much of a learned net2's decisions each group of LCFs carries.

Replays a trained policy (its PPOSpec, net1 and actor at one iteration,
as its training run saved them) over recorded steps, once as it
is and once per group of LCFs with that group's beliefs replaced by their
mean over the same steps (occlusion). The steps keep the observations
(``campaign run --tables steps`` without experiments/slim_steps.py), and
each episode is replayed with the window the sender had at each step, so
every variant sees the same history of messages and windows; only what
net1 reports for the occluded LCFs differs, from the first step on, and
it reaches the decision through the actor (and through the window target
and the rules where they read an occluded LCF). Reported per group: the
mean absolute change of the log2 window a decision sets, and the share of
decisions it moves by more than 0.1 (7%).

Groups: the five composites net1 v10 and neto1 v3 added back
(COMPOSITES5), expliCCit's seven LCFs, the rest of the policy's LCFs, all
of them, and each LCF alone.

Usage: python experiments/occlusion.py OUT --policy NAME=SPEC.json,NET1.pt,ACTOR.pt ...
           --data DIR ... [--controllers NAME=CONTROLLER ...] [--max-episodes 40]
A policy's episodes are those whose controller is the one given for its
name in --controllers (agent:HASH, as episodes.parquet names it); without
one, every episode under --data is replayed.
"""

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn

from expliccit_env.policy import OBS_DIM
from expliccit_env.train import fit
from expliccit_env.train.actors import new_actor, policy_graph
from expliccit_env.train.actors import width as actor_width
from expliccit_env.train.beliefs import Bins
from expliccit_env.train.spec import EXPLICCIT_LCFS, PPOSpec

COMPOSITES5 = (
    "departure_rate_bps",
    "available_capacity_bps",
    "overload_bps",
    "bdp_pkts",
    "cross_arrival_rate_bps",
)
MOVED = 0.1  # log2 window: a 7% change


class Occlude(nn.Module):
    """Row v of the batch keeps net1's output z except where mask[v] holds,
    which takes the reference."""

    def __init__(self, mask: torch.Tensor, ref: torch.Tensor) -> None:
        super().__init__()
        self.mask, self.ref = mask, ref

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.where(self.mask, self.ref, z)


def load(spec_json: Path, net1_pt: Path, actor_pt: Path) -> tuple[fit.Net1, nn.Module, PPOSpec]:
    """A trained net2 as its run saved it: ppo_spec.json, net1.pt and an
    iteration's actor.pt."""
    spec = PPOSpec.model_validate_json(spec_json.read_text())
    net1, _ = fit.load_net1(net1_pt)
    actor = new_actor(actor_width(net1, spec), spec)
    actor.load_state_dict(torch.load(actor_pt, weights_only=True))
    return net1, actor, spec


type Episode = tuple[np.ndarray, np.ndarray, np.ndarray]  # obs [T, OBS_DIM], cwnd [T], cwnd_set [T]


def episodes(dirs: list[Path], controller: str | None, cap: int) -> list[Episode]:
    """The steps of up to ``cap`` ok episodes of ``controller``, in order."""
    out: list[Episode] = []
    for d in dirs:
        for shard in sorted(d.rglob("steps.parquet")):
            ep = pq.read_table(
                shard.parent / "episodes.parquet", columns=["episode_id", "controller", "status"]
            )
            keep = sorted(
                e["episode_id"]
                for e in ep.to_pylist()
                if e["status"] == "ok" and (controller is None or e["controller"] == controller)
            )
            if not keep:
                continue
            t = pq.read_table(shard, columns=["episode_id", "step", "obs", "cwnd", "cwnd_set"])
            ids = np.asarray(t["episode_id"].to_numpy(zero_copy_only=False)).astype(str)
            lists = cast("pa.FixedSizeListArray[Any, Any]", t["obs"].combine_chunks())
            flat = lists.flatten()  # pyright: ignore[reportUnknownMemberType]
            obs = np.asarray(flat, np.float32).reshape(-1, OBS_DIM)
            cwnd = np.asarray(t["cwnd"].to_numpy(), np.float32)
            cwnd_set = np.asarray(t["cwnd_set"].to_numpy(), np.float32)
            step = np.asarray(t["step"].to_numpy())
            for e in keep:
                rows = np.flatnonzero(ids == e)
                rows = rows[np.argsort(step[rows])]
                out.append((obs[rows], cwnd[rows], cwnd_set[rows]))
                if len(out) >= cap:
                    return out
    return out


def groups(lcfs: tuple[str, ...]) -> dict[str, list[int]]:
    idx = {n: i for i, n in enumerate(lcfs)}
    g = {
        "five composites": [idx[n] for n in COMPOSITES5 if n in idx],
        "expliCCit's seven": [idx[n] for n in EXPLICCIT_LCFS if n in idx],
    }
    g["the rest"] = [
        i for i in range(len(lcfs)) if i not in g["five composites"] + g["expliCCit's seven"]
    ]
    g["all"] = list(range(len(lcfs)))
    g |= {n: [i] for n, i in idx.items()}
    return {k: v for k, v in g.items() if v}


def occlusion(
    net1: fit.Net1, actor: nn.Module, spec: PPOSpec, eps: list[Episode]
) -> dict[str, dict[str, float]]:
    """Per variant, over decisions: the change of u (the learned step away
    from the believed BDP) and of the log2 window (u plus the believed
    BDP, which moves only where an occluded LCF feeds the target)."""
    graph = policy_graph(net1, actor, spec)  # pyright: ignore[reportArgumentType]
    kind = net1.kind
    per = kind.b if isinstance(kind, Bins) else 1
    gs = groups(net1.lcfs)
    names = ["none", *gs]
    v = len(names)
    with torch.no_grad():
        ref = torch.from_numpy(np.concatenate([net1.beliefs(o) for o, _, _ in eps]).mean(0))
    mask = torch.zeros(v, ref.numel(), dtype=torch.bool)
    for k, n in enumerate(names[1:], 1):
        for i in gs[n]:
            mask[k, i * per : (i + 1) * per] = True
    graph.front.fault = Occlude(mask, ref)
    n_h = (
        graph.state_dim - 5
    )  # Held's state: GRU, prev u, steps left, window, BDP at decision, flag
    us: list[np.ndarray] = []
    windows: list[np.ndarray] = []
    gap: list[float] = []  # |log2 cwnd the guest set - window replayed|, no occlusion
    with torch.no_grad():
        for obs, cwnd, cwnd_set in eps:
            state = torch.zeros(v, graph.state_dim)
            for t in range(obs.shape[0]):
                o = torch.from_numpy(np.repeat(obs[t : t + 1], v, 0))
                _, state = graph.step(o, state, cwnd=torch.full((v, 1), float(cwnd[t])))
                assert state is not None
                if state[0, -1] > 0.5:  # this step decided
                    w = state[:, n_h + 2]
                    us.append((w - state[:, n_h + 3]).numpy())
                    windows.append(w.numpy())
                    if cwnd_set[t] > 0:
                        gap.append(abs(float(np.log2(cwnd_set[t])) - float(w[0])))
    u, w = np.stack(us), np.stack(windows)  # [decisions, variants]
    du, dw = np.abs(u - u[:, :1]), np.abs(w - w[:, :1])
    out = {
        n: {
            "u": float(du[:, k].mean()),
            "u_moved": float((du[:, k] > MOVED).mean()),
            "window": float(dw[:, k].mean()),
        }
        for k, n in enumerate(names)
        if k
    }
    return out | {
        "decisions": {
            "n": float(u.shape[0]),
            "episodes": float(len(eps)),
            "u_mean": float(u[:, 0].mean()),
            "u_sd": float(u[:, 0].std()),
            "replay_gap_median": float(np.median(gap)) if gap else float("nan"),
        }
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument(
        "--policy", action="append", required=True, help="NAME=SPEC.json,NET1.pt,ACTOR.pt"
    )
    p.add_argument("--data", type=Path, nargs="+", required=True)
    p.add_argument("--controllers", action="append", default=[], help="NAME=CONTROLLER")
    p.add_argument("--max-episodes", type=int, default=40)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ctl = dict(x.split("=", 1) for x in a.controllers)
    report: dict[str, dict[str, dict[str, float]]] = {}
    for item in a.policy:
        name, rest = item.split("=", 1)
        spec_json, net1_pt, actor_pt = (Path(x) for x in rest.split(","))
        net1, actor, spec = load(spec_json, net1_pt, actor_pt)
        eps = episodes(a.data, ctl.get(name), a.max_episodes)
        if not eps:
            raise SystemExit(f"{name}: no episodes under {a.data}")
        report[name] = occlusion(net1, actor, spec, eps)
        print(
            name,
            json.dumps(
                {k: v for k, v in report[name].items() if " " in k or k in ("all", "decisions")}
            ),
            flush=True,
        )
    (a.out / "occlusion.json").write_text(json.dumps(report, indent=1))
    rows = ["five composites", "expliCCit's seven", "the rest", "all"]
    lines = [
        "Occlusion over each policy's decisions: mean |change in u| with the group's"
        " beliefs at their mean over the same steps, the share of decisions whose u moved"
        " by more than 0.1 in brackets, and the mean |change in the log2 window| after"
        " the slash (u plus the believed BDP, which moves where the group feeds the"
        " target). Replay gap: median |log2 of the window the guest set - the replayed"
        " window| without occlusion.",
        "",
        "| policy | decisions | u mean (sd) | replay gap | " + " | ".join(rows) + " |",
        "|---|---|---|---|" + "---|" * len(rows),
    ]
    for name, r in report.items():
        dec = r["decisions"]
        cells = [
            f"{r[g]['u']:.3f} ({r[g]['u_moved']:.2f}) / {r[g]['window']:.3f}" if g in r else ""
            for g in rows
        ]
        lines.append(
            f"| {name} | {int(dec['n'])} | {dec['u_mean']:+.3f} ({dec['u_sd']:.3f}) | "
            + f"{dec['replay_gap_median']:.3f} | "
            + " | ".join(cells)
            + " |"
        )
    lines += ["", "Largest single LCFs (mean |change in u|):", ""]
    for name, r in report.items():
        single = sorted(
            ((v["u"], k) for k, v in r.items() if k not in rows and k != "decisions"), reverse=True
        )
        lines.append(f"- {name}: " + ", ".join(f"{k} {m:.3f}" for m, k in single[:8]))
    (a.out / "occlusion.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
