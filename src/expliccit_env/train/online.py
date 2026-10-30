"""Online (closed-loop) training of net2 on a frozen net1.

docs/methodology.md, section 4. Each iteration exports the current policy,
runs a batch of episodes over the training paths with seeded exploration
noise, adds their transitions (reward from ground truth) to a replay
buffer, and updates net2 off-policy. The noise-free policy is evaluated on
held-out paths every few iterations.
"""

# torch's own annotations leave many signatures partly unknown (from_numpy,
# manual_seed); every other strict check stays on.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from expliccit_env import campaign
from expliccit_env.spec import AgentCC, CampaignSpec, EpisodeSpec, OnnxPolicy, PathSpec, paths_of
from expliccit_env.train import data, fit
from expliccit_env.train.export import to_onnx
from expliccit_env.train.spec import Net2Spec
from expliccit_env.uml import UmlBuild


class OnlineSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    spec_version: Literal[1] = 1
    # A campaign whose sweep lists the training paths (its ccs are ignored).
    paths_from: Path
    # A campaign whose sweep lists held-out evaluation paths.
    eval_from: Path | None = None
    net1_from: Path  # a training run's net1.pt, kept frozen
    init_from: Path | None = None  # a training run's net2.pt to start from
    # Teacher step data to put in the replay buffer before the first update.
    offline_data: list[Path] = Field(default_factory=list[Path])
    # Only these controllers' episodes from offline_data; all when unset.
    offline_controllers: frozenset[str] | None = None
    net2: Net2Spec = Net2Spec(algo="td3bc", reward="utility", discount=0.99)
    iterations: PositiveInt = 10
    episodes_per_iteration: PositiveInt = 16
    duration_ms: PositiveInt = 20_000
    noise_sigma: Annotated[float, Field(ge=0, le=2)] = 0.05
    updates_per_iteration: PositiveInt = 2000
    # Critic-only updates before the first iteration, on whatever the replay
    # buffer holds (offline_data), so the actor does not follow a blank critic.
    critic_warmup: Annotated[int, Field(ge=0)] = 0
    eval_every: PositiveInt = 5
    replay_capacity: PositiveInt = 2_000_000
    seed: Annotated[int, Field(ge=0)] = 0

    @classmethod
    def parse(cls, text: str) -> OnlineSpec:
        return cls.model_validate_json(text)


class _Replay:
    """The newest ``capacity`` transitions."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.t: fit.Transitions | None = None

    def add(self, t: fit.Transitions) -> None:
        if self.t is not None:
            t = fit.Transitions(
                *(torch.cat([a, b]) for a, b in zip(self.t.parts(), t.parts(), strict=True))
            )
        self.t = fit.Transitions(*(x[-self.capacity :] for x in t.parts()))

    def sample(self, rng: np.random.Generator, n: int) -> fit.Transitions:
        assert self.t is not None
        b = torch.from_numpy(rng.integers(0, self.t.s.shape[0], n))
        return fit.Transitions(*(x[b] for x in self.t.parts()))

    def __len__(self) -> int:
        return 0 if self.t is None else int(self.t.s.shape[0])


def _transitions_from(
    dirs: list[Path],
    net1: fit.Net1,
    spec: Net2Spec,
    controllers: frozenset[str] | None = None,
) -> fit.Transitions:
    steps = data.load(dirs, seed=0, val=0.0, test=0.0, controllers=controllers)
    t, _ = fit.transitions(steps, net1.beliefs(steps.obs), spec, "train")
    return t


def _agent(policy: Path, sigma: float) -> AgentCC:
    sha = hashlib.sha256(policy.read_bytes()).hexdigest()
    return AgentCC(policy=OnnxPolicy(path=policy, sha256=sha), noise_sigma=sigma)


def _run(
    out: Path, episodes: list[EpisodeSpec], build: UmlBuild, workers: int, scratch: Path | None
) -> dict[str, Any]:
    campaign.write_episodes(out, episodes, {"name": out.name})
    for _ in campaign.run_shard(
        out,
        build,
        campaign.Shard(0, 1),
        workers=workers,
        scratch=scratch,
        tables=frozenset({"steps"}),
    ):
        pass
    return campaign.summary(out)


def run_online(
    spec: OnlineSpec, out: Path, build: UmlBuild, *, workers: int, scratch: Path | None = None
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    (out / "online_spec.json").write_text(spec.model_dump_json(indent=2))
    torch.manual_seed(spec.seed)
    rng = np.random.default_rng(spec.seed)
    net1, net1_spec = fit.load_net1(spec.net1_from)
    fit.save_net1(net1, net1_spec, out / "net1.pt")
    k = len(net1.lcfs) * (3 if net1.intervals else 1)
    learner = fit.Learner(spec.net2, k)
    if spec.init_from is not None:
        state = torch.load(spec.init_from, map_location="cpu", weights_only=True)
        learner.actor.load_state_dict(state)
        learner.actor_t.load_state_dict(state)
    replay = _Replay(spec.replay_capacity)
    if spec.offline_data:
        replay.add(_transitions_from(spec.offline_data, net1, spec.net2, spec.offline_controllers))
    for u in range(spec.critic_warmup if len(replay) else 0):
        learner.update(replay.sample(rng, spec.net2.batch_size), u, actor=False)
    train_paths: list[PathSpec] = paths_of(
        CampaignSpec.parse(spec.paths_from.read_text()).sweep, rng
    )
    eval_paths: list[PathSpec] = (
        paths_of(CampaignSpec.parse(spec.eval_from.read_text()).sweep, rng)
        if spec.eval_from
        else []
    )
    history: list[dict[str, Any]] = []

    def export(path: Path) -> Path:
        graph = fit.policy_graph(
            net1, learner.actor, spec.net2.action_max, spec.net2.action_decimals
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(to_onnx(graph))
        learner.actor.train()
        return path

    def evaluate(it_dir: Path, policy: Path) -> dict[str, Any]:
        cc = _agent(policy, 0.0)
        episodes = [
            EpisodeSpec(path=p, cc=cc, duration_ms=spec.duration_ms, seed=7 + i)
            for i, p in enumerate(eval_paths)
        ]
        return next(iter(_run(it_dir / "eval", episodes, build, workers, scratch).values()))

    for it in range(spec.iterations):
        it_dir = out / f"iter{it:03d}"
        policy = export(it_dir / "policy.onnx")
        cc = _agent(policy, spec.noise_sigma)
        picks = rng.integers(0, len(train_paths), spec.episodes_per_iteration)
        episodes = [
            EpisodeSpec(
                path=train_paths[i],
                cc=cc,
                duration_ms=spec.duration_ms,
                seed=int(rng.integers(0, 2**63)),
                steps=True,
            )
            for i in picks
        ]
        rollout = next(iter(_run(it_dir / "rollouts", episodes, build, workers, scratch).values()))
        new = _transitions_from([it_dir / "rollouts"], net1, spec.net2)
        replay.add(new)
        logs: dict[str, list[float]] = {}
        for u in range(spec.updates_per_iteration):
            for name, v in learner.update(replay.sample(rng, spec.net2.batch_size), u).items():
                logs.setdefault(name, []).append(v)
        entry: dict[str, Any] = {
            "iteration": it,
            "replay": len(replay),
            "new_transitions": int(new.s.shape[0]),
            "mean_reward_new": float(new.r.mean()) if new.s.shape[0] else float("nan"),
            "rollouts": rollout,
            "losses": {name: float(np.mean(v[-200:])) for name, v in logs.items()},
        }
        if eval_paths and ((it + 1) % spec.eval_every == 0 or it + 1 == spec.iterations):
            entry["eval"] = evaluate(it_dir, export(it_dir / "policy-updated.onnx"))
        history.append(entry)
        (out / "history.json").write_text(json.dumps(history, indent=2, default=float))
    torch.save(learner.actor.state_dict(), out / "net2.pt")
    export(out / "policy.onnx")
    return {"iterations": history}
