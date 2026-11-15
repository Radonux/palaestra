"""End to end: every kind of net1 with every kind of net2 trains, exports,
and runs.

net1 reports single values (expliCCit's design), intervals, or a
distribution over bins (palaestra's). net2 reads either all of net1's
output or only its means, and is either expliCCit's memoryless actor
trained offline (on the window multiplier, or on a window aimed at the
believed BDP) or the recurrent actor taught by DAgger and continued by
PPO. Every combination must keep working: a refactor that silently drops
one, in particular a net2 that reads distributions, fails here.

The host tests train on synthetic steps and check that the exported ONNX
graph computes what the trained modules do. The UML tests run the whole
pipeline through the CLI, as experiments do: a data campaign, net1 and
an offline net2, the policy in the guest, then DAgger and PPO in closed
loop and the final policy in the guest again.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import onnx
import pyarrow as pa
import pytest

torch = pytest.importorskip("torch")

from onnx.reference import ReferenceEvaluator  # noqa: E402

from expliccit_env.policy import OBS_DIM, TRUTH_DIM  # noqa: E402
from expliccit_env.train import actors, data, fit, handoff, learners, ppo, trajectories  # noqa: E402
from expliccit_env.train.export import to_onnx  # noqa: E402
from expliccit_env.train.graph import PolicyGraph  # noqa: E402
from expliccit_env.train.spec import Net1Spec, Net2Spec  # noqa: E402

from .test_train import synthetic  # noqa: E402

type Kind = Literal["point", "interval", "bins"]
type View = Literal["bins", "means"]

# net1's output kinds, as Net1Spec fields. "bins" is the distribution.
KINDS: dict[Kind, dict[str, Any]] = {
    "point": {"output": {"kind": "point"}},
    "interval": {"output": {"kind": "interval"}},
    "bins": {"output": {"kind": "bins", "bins": 16}},
}
VIEWS: tuple[View, ...] = ("bins", "means")  # "bins": all of net1's output


def _net1(kind: Kind, steps: data.Steps, tmp: Path) -> fit.Net1:
    """A small net1 of the kind, saved and loaded back as a run would."""
    spec = Net1Spec.model_validate({"width": 32, "epochs": 2, "batch_size": 256} | KINDS[kind])
    net1, _ = fit.fit_net1(spec, steps, 0)
    fit.save_net1(net1, spec, tmp / "net1.pt")
    loaded, _ = fit.load_net1(tmp / "net1.pt")
    obs = steps.obs[:64]
    assert np.array_equal(net1.beliefs(obs), loaded.beliefs(obs))
    return loaded


def _onnx_matches(graph: PolicyGraph, obs: np.ndarray, *, decisions: int = 12) -> np.ndarray:
    """Run the exported graph step by step, carrying its state and the
    window it sets, beside the torch graph; they must agree. Returns the
    actions."""
    model = onnx.load_from_string(to_onnx(graph))
    names = [i.name for i in model.graph.input]
    run = ReferenceEvaluator(model)
    state = np.zeros((1, graph.state_dim), np.float32)
    cwnd = np.full((1, 1), 40.0, np.float32)
    truth = np.full((1, TRUTH_DIM), np.nan, np.float32)
    actions = []
    for row in obs[:decisions]:
        o = row[None, :].astype(np.float32)
        feed = {"obs": o, "state": state, "cwnd": cwnd, "truth": truth}
        got = run.run(None, {n: feed[n] for n in names})
        with torch.no_grad():
            want, nxt = graph.step(
                torch.from_numpy(o),
                torch.from_numpy(state) if graph.state_dim else None,
                torch.from_numpy(truth) if graph.uses_truth else None,
                torch.from_numpy(cwnd) if graph.target is not None else None,
            )
        assert np.allclose(got[0], want.numpy(), atol=1e-4), (got[0], want)
        a = float(got[0].reshape(-1)[0])
        assert np.isfinite(a) and abs(a) < 8
        actions.append(a)
        if graph.state_dim:
            assert nxt is not None and np.allclose(got[1], nxt.numpy(), atol=1e-4)
            state = got[1].astype(np.float32)
        cwnd = np.clip(cwnd * 2**a, 1, 1e6).astype(np.float32)
    return np.asarray(actions)


@pytest.mark.parametrize("view", VIEWS)
@pytest.mark.parametrize("action", ["multiplier", "bdp"])
@pytest.mark.parametrize("kind", list(KINDS))
def test_an_offline_net2_on_every_net1_trains_and_exports(
    kind: Kind, action: Literal["multiplier", "bdp"], view: View, tmp_path: Path
) -> None:
    torch.manual_seed(0)
    steps = synthetic(n_episodes=10, steps=120)
    net1 = _net1(kind, steps, tmp_path)
    spec = Net2Spec(
        algo="td3bc", steps=20, batch_size=64, hidden=16, action=action, view=view, action_max=2.0
    )
    actor, log = fit.fit_net2(spec, steps, net1, 0)
    assert log["transitions"] > 0 and np.isfinite(log["final"]["critic"])
    graph = fit.policy_graph(net1, actor, spec.action_max, action=action, view=view)
    _onnx_matches(graph, steps.obs)


def test_a_loaded_net1_is_not_scored_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Scoring a distribution net1 holds rows x LCFs x bins probabilities,
    which does not fit beside a net2 dataset; its own run already did it."""
    from expliccit_env.train import run  # noqa: PLC0415
    from expliccit_env.train.spec import TrainSpec  # noqa: PLC0415

    steps = synthetic(n_episodes=10, steps=120)
    _net1("bins", steps, tmp_path)
    monkeypatch.setattr(data, "load", lambda *_a, **_k: steps)
    net2 = {"algo": "td3bc", "steps": 20, "batch_size": 64, "hidden": 16, "action": "bdp"}
    spec = {"data": [str(tmp_path)], "net2": net2 | {"action_max": 2.0}}
    loaded = run.train(
        TrainSpec.model_validate(spec | {"net1_from": str(tmp_path / "net1.pt")}), tmp_path / "a"
    )
    assert "test" not in loaded["net1"] and loaded["net2"]["transitions"] > 0
    assert (tmp_path / "a" / "policy.onnx").stat().st_size > 0
    net1 = {"width": 32, "epochs": 1, "batch_size": 256}
    trained = run.train(TrainSpec.model_validate(spec | {"net1": net1}), tmp_path / "b")
    assert trained["net1"]["test"]


def _flow(steps: data.Steps, episode: int) -> pa.Table:
    """One synthetic episode as a rollout's step table (the window 20)."""
    rows = steps.episode == episode
    obs = steps.obs[rows]
    n = obs.shape[0]
    return pa.table(
        {
            "step": pa.array(np.arange(n, dtype=np.int32)),
            "t_ns": pa.array((np.arange(n) * 10_000_000).astype(np.uint64)),
            "obs": pa.FixedSizeListArray.from_arrays(pa.array(obs.reshape(-1)), OBS_DIM),
            "cwnd": pa.array(np.full(n, 20, np.uint32)),
            **{f"label_{k}": pa.array(v[rows].astype(np.float64)) for k, v in steps.labels.items()},
        }
    )


@pytest.mark.parametrize("view", VIEWS)
@pytest.mark.parametrize("kind", list(KINDS))
def test_a_recurrent_net2_on_every_net1_is_taught_continued_and_exported(
    kind: Kind, view: View, tmp_path: Path
) -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    steps = synthetic(n_episodes=10, steps=120)
    net1 = _net1(kind, steps, tmp_path)
    common: dict[str, Any] = {
        "paths_from": Path("x"),
        "net1_from": tmp_path / "net1.pt",
        "view": view,
        "input_bins": None,
        "hidden": 8,
        "critic_hidden": 16,
        "updates_per_iteration": 5,
        "batch_trajectories": 4,
        "epochs": 1,
    }
    taught_spec = ppo.PPOSpec.model_validate(
        common | {"algo": "dagger", "student": {"kind": "gate3"}}
    )
    width = actors.width(net1, taught_spec)
    critic_width = width + 2 + len(net1.lcfs)

    def decisions(spec: ppo.PPOSpec) -> list[trajectories.Trajectory]:
        front, target, rules = (
            actors.front(net1, spec),
            actors.target(net1, spec),
            actors.Rules(net1, spec),
        )
        out = []
        for e in range(4):
            tr = trajectories.trajectory(
                _flow(steps, e),
                rng.normal(size=120).astype(np.float32),
                net1,
                front=front,
                target=target,
                rules=rules,
                spec=spec,
                group=(e, 0),
                rivals=np.array([[1, 0.0, 0.6]]),  # Cubic for the first 0.6 s
            )
            assert tr is not None
            out.append(tr)
        return out

    taught = learners.make_learner(taught_spec, width, critic_width)
    logs = taught.update(decisions(taught_spec), rng)
    assert np.isfinite(logs["imitation_loss"]) and np.isfinite(logs["value_loss"])
    run = tmp_path / "taught"
    run.mkdir()
    (run / "ppo_spec.json").write_text(taught_spec.model_dump_json())
    fit.save_net1(net1, fit.load_net1(tmp_path / "net1.pt")[1], run / "net1.pt")
    torch.save(taught.actor.state_dict(), run / "actor.pt")
    torch.save(taught.critic.state_dict(), run / "critic.pt")

    rl_spec = taught_spec.model_copy(update={"algo": "ppo", "init_from": run, "anchor": 1.0})
    source = handoff.earlier_run(run, net1)
    rl = learners.make_learner(rl_spec, width, critic_width, source)
    start = ppo.policy_graph(net1, rl.actor, rl_spec)
    assert to_onnx(start) == to_onnx(ppo.policy_graph(net1, taught.actor, taught_spec))
    logs = rl.update(decisions(rl_spec), rng)
    assert np.isfinite(logs["policy_loss"]) and logs["kl_to_start"] >= 0
    _onnx_matches(ppo.policy_graph(net1, rl.actor, rl_spec), steps.obs)


# --- In UML, through the CLI ---------------------------------------------------

_PATHS = [
    {
        "trace": {"kind": "periodic_levels", "levels_mbps": [mbps], "level_ms": 1000},
        "delay": {"mahak_compat_mrtt_ms": ms},
        "buffer": {"packets": pkts},
    }
    for mbps, ms, pkts in ((24, 10, 100), (12, 20, 60), (48, 5, 200))
]


def _cli(*args: str | Path) -> None:
    from expliccit_env import cli  # noqa: PLC0415

    assert cli.main([str(a) for a in args]) == 0, args


def _guest_run(build: Path, policy: Path, out: Path) -> np.ndarray:
    """One episode of the policy alone on the first path; its applied actions."""
    import pyarrow.parquet as pq  # noqa: PLC0415

    spec = {
        "path": _PATHS[0],
        "cc": {
            "kind": "agent",
            "policy": {
                "path": str(policy),
                "sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
            },
        },
        "duration_ms": 3000,
        "seed": 3,
        "steps": True,
    }
    (out.parent / f"{out.name}.json").write_text(json.dumps(spec))
    _cli("episode", out.parent / f"{out.name}.json", out, "--build", build)
    (steps,) = out.rglob("steps.parquet")
    applied = pq.read_table(steps, columns=["applied_action"])["applied_action"].to_numpy()
    assert applied.size > 100 and np.isfinite(applied).all()
    assert np.ptp(applied) > 0  # the policy acts on what it reads
    return applied


@pytest.mark.uml
@pytest.mark.parametrize("view", VIEWS)
@pytest.mark.parametrize("kind", list(KINDS))
def test_the_pipeline_runs_for_every_net1_in_uml(
    build: object, kind: Kind, view: View, tmp_path: Path
) -> None:
    from expliccit_env.uml import UmlBuild  # noqa: PLC0415

    assert isinstance(build, UmlBuild)
    root = build.root
    # Teacher data: Cubic and BBR on three links, steps with ground truth.
    camp = {
        "name": "e2e",
        "ccs": [{"kind": "stock", "name": "cubic"}, {"kind": "stock", "name": "bbr"}],
        "sweep": {"kind": "paths", "paths": _PATHS},
        "conditions": [{}],
        "duration_ms": 3000,
        "seed": 1,
        "steps": True,
    }
    (tmp_path / "camp.json").write_text(json.dumps(camp))
    _cli("campaign", "plan", tmp_path / "camp.json", tmp_path / "data")
    _cli(
        "campaign", "run", tmp_path / "data", "--build", root, "--workers", "2", "--tables", "steps"
    )

    # net1, and expliCCit's offline net2 on it, then that policy in the guest.
    train = {
        "data": [str(tmp_path / "data")],
        "split_by": "row",
        "val_fraction": 0.2,
        "test_fraction": 0.2,
        "net1": {"width": 32, "epochs": 2, "batch_size": 256, **KINDS[kind]},
        "net2": {
            "algo": "expliccit-v1",
            "steps": 30,
            "batch_size": 64,
            "hidden": 16,
            "view": view,
            "action": "bdp",
            "action_max": 2.0,
        },
    }
    (tmp_path / "train.json").write_text(json.dumps(train))
    _cli("train", "run", tmp_path / "train.json", tmp_path / "offline")
    report = json.loads((tmp_path / "offline" / "report.json").read_text())
    assert report["net1"]["test"]["capacity_bps"]["net1"]["n"] > 0
    _guest_run(root, tmp_path / "offline" / "policy.onnx", tmp_path / "offline-run")

    # The recurrent net2 on the same net1: taught by DAgger, continued by PPO.
    (tmp_path / "paths.json").write_text(json.dumps(camp | {"ccs": camp["ccs"][:1]}))
    base = {
        "paths_from": str(tmp_path / "paths.json"),
        "net1_from": str(tmp_path / "offline" / "net1.pt"),
        "view": view,
        "input_bins": None,
        "hidden": 8,
        "critic_hidden": 16,
        "iterations": 1,
        "links_per_iteration": 1,
        "group": 2,
        "duration_ms": 3000,
        "epochs": 1,
        "critic_warmup": 0,
        "updates_per_iteration": 5,
        "compete": 0.5,
        "copies": [2],
    }
    stages = {
        "teach": base | {"algo": "dagger", "student": {"kind": "gate3"}},
        "rl": base
        | {
            "algo": "ppo",
            "student": {"kind": "gate3"},
            "init_from": str(tmp_path / "teach"),
            "anchor": 1.0,
        },
    }
    for name, spec in stages.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(spec))
        _cli(
            "train",
            "ppo",
            tmp_path / f"{name}.json",
            tmp_path / name,
            "--build",
            root,
            "--workers",
            "2",
        )
        history = json.loads((tmp_path / name / "history.json").read_text())
        assert history[-1]["rollouts"]["ok"] == 2 and history[-1]["learner"]["decisions"] > 100
    _guest_run(root, tmp_path / "rl" / "policy.onnx", tmp_path / "rl-run")
