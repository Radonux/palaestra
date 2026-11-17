"""A recurrent net2 trained in closed loop on a frozen net1, by PPO,
TD3+BC or DAgger (PPOSpec.algo).

The actor (recurrent.py) reads net1's output (pooled bins, or means) and
its own previous u through a GRU, and holds each decision for
clamp(min RTT, 10 ms, 50 ms). It answers u, the window's log2 ratio to the
believed BDP, and starts as the fixed rule (u_rule everywhere) unless the
spec sets base or init_actor. Each iteration exports the policy, runs each
of a few training links K times (the same seed, so the same path and loss
draws, with different exploration noise), and learns from those rollouts.
A share of the links (compete) gets competition in place of its
condition: Cubic or BBRv1, or copies of the policy itself (PPOSpec
competition sets when the stock flows start and how many there are).

- "ppo": the clipped policy gradient on each iteration's rollouts. The
  reward is the per-step log utility from ground truth (score.utility),
  per flow. Copies of the policy each learn from their own utility or,
  with copies_reward "team", from the copies' mean: a copy that learns
  from its own gains by pushing harder than the others, which lowers what
  they all get. Returns are discounted by elapsed time, gamma =
  exp(-dt / tau), and each decision's advantage has its group's mean at the
  same decision taken off, on top of the value function: the spread
  between links dominates the returns, and the K rollouts of a link share
  it.
- "td3bc": TD3+BC off a replay of the last few iterations' rollouts, with
  the same reward.
- "dagger": DAgger from a privileged teacher, a rule that knows which
  other flows share the link. Every decision the actor makes is labelled
  with the teacher's u, so it learns from a label at each decision rather
  than from a return. The student (PPOSpec.student) learns u itself ("u"),
  its belief that a stock TCP flow shares the link ("gate"), or that
  belief and its belief that copies do ("gate3"); a gate student's u moves
  from the fixed rule toward the teacher's rules by its beliefs.

The critics (PPO's value network, TD3+BC's Q networks) see the actor's
inputs, the true LCFs (critic_truth) and, with critic_flows, how many
flows of each kind share the link. They are used in training only and
never exported.
"""

# torch's own annotations leave many signatures partly unknown (from_numpy,
# manual_seed); every other strict check stays on.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from expliccit_env.spec import AgentCC, Condition, EpisodeSpec, OnnxPolicy, PathSpec
from expliccit_env.train import fit, handoff, learners, mixture, rollouts, score
from expliccit_env.train.actors import Head, policy_graph
from expliccit_env.train.actors import width as actor_width
from expliccit_env.train.export import to_onnx
from expliccit_env.train.handoff import earlier_run
from expliccit_env.train.mixture import RIVALS
from expliccit_env.train.recurrent import RecurrentActor
from expliccit_env.train.spec import PPOSpec
from expliccit_env.train.trajectories import REWARD, from_rollouts
from expliccit_env.uml import UmlBuild

# What callers (the CLI, the experiments) take from here.
__all__ = ["Head", "PPOSpec", "earlier_run", "policy_graph", "run_ppo"]


def _export(path: Path, net1: fit.Net1, actor: RecurrentActor, spec: PPOSpec) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(to_onnx(policy_graph(net1, actor, spec)))
    actor.train()
    return path


def _scores(d: Path) -> dict[tuple[str, str], float]:
    """Mean log utility per (link, condition) of an evaluation run."""
    per = score.episode_scores([d], REWARD)
    return {(k[1], k[2]): float(np.mean(v)) for k, v in per.items()}


def _evaluate(
    d: Path,
    policy: Path,
    arms: list[tuple[PathSpec, Condition]],
    spec: PPOSpec,
    run: Any,
) -> dict[tuple[str, str], float]:
    cc = AgentCC(
        policy=OnnxPolicy(path=policy, sha256=hashlib.sha256(policy.read_bytes()).hexdigest())
    )
    eps = [
        EpisodeSpec(
            path=p.model_copy(update={"loss": c.loss}),
            cross=c.flows(p.trace),
            cc=cc,
            duration_ms=spec.duration_ms,
            seed=7 + i,
            steps=True,
        )
        for i, (p, c) in enumerate(arms)
    ]
    run(d, eps)
    out = _scores(d)
    if not spec.keep_rollouts:
        shutil.rmtree(d / "shards", ignore_errors=True)
    return out


def _paired(
    pol: dict[tuple[str, str], float], rule: dict[tuple[str, str], float]
) -> dict[str, Any]:
    keys = sorted(pol.keys() & rule.keys())
    diff = np.array([pol[k] - rule[k] for k in keys])
    by: dict[str, list[float]] = {}
    for k, v in zip(keys, diff.tolist(), strict=True):
        by.setdefault(k[1], []).append(v)
    return {
        "log_utility": float(np.mean([pol[k] for k in keys])) if keys else math.nan,
        "minus_rule": float(diff.mean()) if keys else math.nan,
        "pairs": len(keys),
        "minus_rule_by_condition": {c: float(np.mean(v)) for c, v in sorted(by.items())},
    }


def _keep_net1(spec: PPOSpec, out: Path) -> fit.Net1:
    """The frozen net1, saved in the run as net1.pt, with the traffic head
    the actor reads (inputs "net1+traffic") beside it as traffic_head.pt."""
    net1, net1_spec = fit.load_net1(spec.net1_from)
    fit.save_net1(net1, net1_spec, out / "net1.pt")
    if spec.traffic_head is not None:
        shutil.copyfile(spec.traffic_head, out / "traffic_head.pt")
    return net1


def run_ppo(
    spec: PPOSpec, out: Path, build: UmlBuild, *, workers: int, scratch: Path | None = None
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    (out / "ppo_spec.json").write_text(spec.model_dump_json(indent=2))
    torch.manual_seed(spec.seed)
    rng = np.random.default_rng(spec.seed)
    net1 = _keep_net1(spec, out)
    width = actor_width(net1, spec)
    critic_width = (
        width
        + 2
        + (len(net1.lcfs) if spec.critic_truth else 0)
        + (len(RIVALS) if spec.critic_flows else 0)
    )
    source = None if spec.init_from is None else earlier_run(spec.init_from, net1)
    learner = learners.make_learner(spec, width, critic_width, source)
    teacher = (
        None if spec.teacher_from is None else handoff.PolicyTeacher.load(spec.teacher_from, net1)
    )
    train_arms = rollouts.arms(spec.paths_from, rng)
    eval_arms = rollouts.arms(spec.eval_from, rng) if spec.eval_from else []

    def run(d: Path, eps: list[EpisodeSpec]) -> dict[str, Any]:
        return rollouts.run(d, eps, build, workers, scratch)

    history: list[dict[str, Any]] = []
    rule_scores: dict[tuple[str, str], float] = {}
    if eval_arms and spec.rule_policy is not None:
        rule_scores = _evaluate(out / "rule-eval", spec.rule_policy, eval_arms, spec, run)
    if eval_arms:
        start = _export(out / "initial" / "policy.onnx", net1, learner.actor, spec)
        pol = _evaluate(out / "initial" / "eval", start, eval_arms, spec, run)
        history.append({"iteration": -1, "eval": _paired(pol, rule_scores)})
        (out / "history.json").write_text(json.dumps(history, indent=2, default=float))
    began_all, slowest = time.monotonic(), 0.0
    for it in range(spec.iterations):
        began = time.monotonic()
        if spec.time_budget_s is not None and began - began_all + slowest > spec.time_budget_s:
            break
        it_dir = out / f"iter{it:03d}"
        policy = _export(it_dir / "policy.onnx", net1, learner.actor, spec)
        eps, groups, rivals = mixture.batch(train_arms, policy, spec, rng)
        summary = next(iter(run(it_dir / "rollouts", eps).values()))
        trs = from_rollouts(it_dir / "rollouts", groups, net1, spec, rivals, teacher=teacher)
        entry: dict[str, Any] = {"iteration": it, "rollouts": summary}
        if trs:
            entry["learner"] = learner.update(trs, rng, actor=it >= spec.critic_warmup)
        if not spec.keep_rollouts:
            shutil.rmtree(it_dir / "rollouts" / "shards", ignore_errors=True)
        if eval_arms and ((it + 1) % spec.eval_every == 0 or it + 1 == spec.iterations):
            updated = _export(it_dir / "policy-updated.onnx", net1, learner.actor, spec)
            pol = _evaluate(it_dir / "eval", updated, eval_arms, spec, run)
            entry["eval"] = _paired(pol, rule_scores)
            torch.save(learner.actor.state_dict(), it_dir / "actor.pt")
        entry["wall_s"] = time.monotonic() - began
        slowest = max(slowest, entry["wall_s"])
        history.append(entry)
        (out / "history.json").write_text(json.dumps(history, indent=2, default=float))
    torch.save(learner.actor.state_dict(), out / "actor.pt")
    torch.save(learner.critic.state_dict(), out / "critic.pt")
    _export(out / "policy.onnx", net1, learner.actor, spec)
    started = handoff.init_paths(spec, source)
    return {
        "iterations": history,
        "started_from": {
            k: None if v is None else str(v)
            for k, v in zip(("actor", "critic"), started, strict=True)
        },
    }
