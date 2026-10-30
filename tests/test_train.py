"""The trainer (src/expliccit_env/train): needs the train extra (PyTorch)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import pytest

torch = pytest.importorskip("torch")

from expliccit_env.policy import EXPLICCIT_OBS_COLS, OBS_DIM  # noqa: E402
from expliccit_env.train import data, fit  # noqa: E402
from expliccit_env.train.export import to_onnx  # noqa: E402
from expliccit_env.train.models import ExpliccitActor, ExpliccitNet1, PolicyGraph, Scaler  # noqa: E402
from expliccit_env.train.online import OnlineSpec, run_online  # noqa: E402
from expliccit_env.train.spec import Net1Spec, Net2Spec  # noqa: E402


def synthetic(n_episodes: int = 40, steps: int = 200, seed: int = 0) -> data.Steps:
    """Episodes whose 'capacity' is a noisy linear function of two columns."""
    rng = np.random.default_rng(seed)
    n = n_episodes * steps
    obs = rng.normal(size=(n, OBS_DIM)).astype(np.float32)
    obs[::97, 68] = np.inf  # Sage's message can hold inf and NaN
    obs[::89, 3] = np.nan
    cap = (3 * obs[:, 74] - 2 * obs[:, 71] + 50).astype(np.float32) * 1e6
    labels = {name: np.full(n, np.nan, np.float32) for name in Net1Spec().lcfs}
    labels["capacity_bps"] = cap + rng.normal(0, 1e4, n).astype(np.float32)
    labels["one_way_delay_ns"] = np.full(n, 2e7, np.float32)
    episode = np.repeat(np.arange(n_episodes), steps)
    action = rng.uniform(-0.2, 0.2, n).astype(np.float32)
    split = np.where(episode % 5 == 0, "test", np.where(episode % 5 == 1, "val", "train"))
    return data.Steps(
        obs=obs,
        labels=labels,
        action=action,
        reward=(-np.abs(action)).astype(np.float32),
        episode=episode.astype(np.int64),
        episode_ids=tuple(str(i) for i in range(n_episodes)),
        split=split,
    )


def test_expliccit_shapes_match_its_released_checkpoints() -> None:
    def params(m: torch.nn.Module) -> int:
        return sum(p.numel() for p in m.parameters())

    assert params(ExpliccitNet1(len(EXPLICCIT_OBS_COLS), 7)) == 1_265_927
    assert params(ExpliccitActor(7)) == 68_097


def test_scaler_ignores_and_zeroes_non_finite_values() -> None:
    x = np.array([[1.0, np.nan], [3.0, 5.0], [np.inf, 7.0]], np.float32)
    s = Scaler.fit((0, 1), x, "minmax")
    assert np.allclose(s.center.numpy(), [1.0, 5.0]) and np.allclose(s.spread.numpy(), [2.0, 2.0])
    obs = torch.zeros(1, OBS_DIM)
    obs[0, 0], obs[0, 1] = float("inf"), float("nan")
    assert torch.equal(s(obs), torch.tensor([[-0.5, -2.5]]))  # non-finite -> 0, then scaled


def test_net1_learns_and_net2_runs() -> None:
    steps = synthetic()
    spec = Net1Spec(width=64, epochs=30, patience=5, batch_size=256, lcfs=("capacity_bps",))
    net1, log = fit.fit_net1(spec, steps, seed=0)
    report = fit.report_net1(net1, steps, "test")["capacity_bps"]
    assert report["net1"]["r2"] > 0.9, (report, log)
    actor, net2_log = fit.fit_net2(Net2Spec(hidden=32, steps=200, batch_size=64), steps, net1, 0)
    assert net2_log["transitions"] > 0 and np.isfinite(net2_log["final"]["critic"])
    with torch.no_grad():
        a = actor(torch.randn(16, 1))
    assert torch.all(a.abs() <= 1)


def test_interval_net1_covers_held_out_rows_at_the_requested_rate() -> None:
    steps = synthetic(n_episodes=60)
    spec = Net1Spec(
        width=64, epochs=30, patience=5, batch_size=256, lcfs=("capacity_bps",), intervals=True
    )
    net1, _ = fit.fit_net1(spec, steps, seed=0)
    iv = fit.report_net1(net1, steps, "test")["capacity_bps"]["interval"]
    assert 0.83 < iv["coverage"] < 0.97, iv
    lo, hi = net1.bounds(steps.obs[:100])
    assert np.all(lo <= hi)


def test_utility_reward_prices_unused_capacity_delay_and_loss() -> None:
    steps = synthetic(n_episodes=1, steps=3)
    lab = steps.labels
    lab["capacity_bps"] = np.array([10e6, 10e6, 10e6], np.float32)
    lab["receive_rate_bps"] = np.array([10e6, 5e6, 10e6], np.float32)
    lab["queue_delay_mean_ns"] = np.array([0, 0, 20e6], np.float32)
    lab["base_rtt_ns"] = np.full(3, 40e6, np.float32)
    lab["drop_fraction"] = np.array([np.nan, 0.0, 0.01], np.float32)
    r = fit.utility(steps, Net2Spec(reward="utility", delay_weight=0.5, loss_weight=4.0))
    assert np.allclose(r, [1.0, 0.5, 1.0 - 0.25 - 0.04])


def _bandit(n: int = 4096) -> fit.Transitions:
    """One step per transition, two logged actions, the larger one better."""
    g = torch.Generator().manual_seed(0)
    s = torch.randn(n, 3, generator=g)
    a = (torch.randint(0, 2, (n, 1), generator=g).float() - 0.5).clone()
    return fit.Transitions(s, a, a.squeeze(1).clone(), torch.randn(n, 3, generator=g))


def test_iql_prefers_the_better_logged_action_where_cloning_averages() -> None:
    t = _bandit()
    means = {}
    for algo in ("iql", "bc"):
        torch.manual_seed(0)
        learner = fit.Learner(Net2Spec(algo=algo, hidden=32, discount=0.0, beta=10.0), 3)
        rng = np.random.default_rng(0)
        for step in range(1500):
            b = torch.from_numpy(rng.integers(0, t.s.shape[0], 256))
            log = learner.update(fit.Transitions(*(x[b] for x in t.parts())), step)
        with torch.no_grad():
            means[algo] = float(learner.actor(t.s).mean())
        assert {"pi_mean", "pi_std", "sat"} <= log.keys()
    assert means["iql"] > 0.25 and abs(means["bc"]) < 0.1, means


def test_td3bc_reports_where_its_critic_extrapolates() -> None:
    t = _bandit(512)
    near_zero = fit.Transitions(t.s, t.a / 50, t.r, t.s2)  # logged actions +-0.01
    learner = fit.Learner(Net2Spec(hidden=32, batch_size=512), 3)
    log = learner.update(near_zero, 0)
    assert {"ood", "q", "q_pull", "bc_pull", "sat"} <= log.keys()
    # Target smoothing noise of 0.2 leaves actions logged near 0, as the
    # teachers' are, almost everywhere.
    assert log["ood"] > 0.8, log


def test_load_keeps_only_the_named_controllers(tmp_path: Path) -> None:
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    shard = tmp_path / "shards" / "0"
    shard.mkdir(parents=True)
    eps = [("e0", "bbr", 3), ("e1", "cubic", 2), ("e2", "agent:abc", 4)]
    pq.write_table(
        pa.table(
            {
                "episode_id": [e for e, _, _ in eps],
                "status": ["ok"] * 3,
                "spec_json": [json.dumps({"path": {"bw": i}}) for i in range(3)],
                "controller": [c for _, c, _ in eps],
            }
        ),
        shard / "episodes.parquet",
    )
    ids = [e for e, _, n in eps for _ in range(n)]
    obs = pa.FixedSizeListArray.from_arrays(
        pa.array(np.zeros(len(ids) * OBS_DIM, np.float32)), OBS_DIM
    )
    pq.write_table(
        pa.table(
            {
                "episode_id": ids,
                "obs": obs,
                "action": np.zeros(len(ids), np.float32),
                "reward": np.zeros(len(ids), np.float32),
                "label_capacity_bps": np.ones(len(ids), np.float32),
            }
        ),
        shard / "steps.parquet",
    )
    steps = data.load([tmp_path], seed=0, val=0.0, test=0.0, controllers=frozenset({"bbr"}))
    assert steps.episode_ids == ("e0",) and steps.obs.shape == (3, OBS_DIM)
    assert data.load([tmp_path], seed=0, val=0.0, test=0.0).obs.shape[0] == 9
    with pytest.raises(ValueError, match="no steps"):
        data.load([tmp_path], seed=0, val=0.0, test=0.0, controllers=frozenset({"vegas"}))


def test_as_deployed_rounds_beliefs_and_actions() -> None:
    lcf = Scaler.fit(
        tuple(range(7)), np.random.default_rng(0).normal(size=(50, 7)).astype(np.float32), "minmax"
    )
    scaler = Scaler.fit(EXPLICCIT_OBS_COLS, np.zeros((4, 70), np.float32), "minmax")
    g = PolicyGraph(
        scaler, ExpliccitNet1(70, 7, width=64).eval(), lcf, ExpliccitActor(7).eval(), 1.0
    )
    g.as_deployed = True
    with torch.no_grad():
        a = g(torch.randn(8, OBS_DIM)).numpy()
    assert np.array_equal(a, np.round(a * 1000) / 1000)


def test_export_is_a_sage_v1_graph() -> None:
    scaler = Scaler.fit(EXPLICCIT_OBS_COLS, np.zeros((4, 70), np.float32), "robust")
    lcf = Scaler.fit(tuple(range(7)), np.zeros((4, 7), np.float32), "standard")
    graph = PolicyGraph(
        scaler, ExpliccitNet1(70, 7, width=64).eval(), lcf, ExpliccitActor(7).eval(), 1.0
    )
    model = onnx.load_from_string(to_onnx(graph))
    (inp,) = model.graph.input
    (out,) = model.graph.output
    dims = [[d.dim_value for d in v.type.tensor_type.shape.dim] for v in (inp, out)]
    assert (inp.name, out.name, dims) == ("obs", "a", [[1, OBS_DIM], [1, 1]])
    assert not {"IsInf", "IsNaN"} & {n.op_type for n in model.graph.node}  # tract lacks them


def test_split_keeps_a_path_together() -> None:
    def spec(cc: str, seed: int, bw: int) -> str:
        return json.dumps({"path": {"bw": bw}, "cc": cc, "seed": seed, "duration_ms": 1})

    same = {
        data._split_of(data._path_key(spec(cc, s, 10)), 0, 0.3, 0.3)
        for cc in "ab"
        for s in range(20)
    }
    assert len(same) == 1
    many = {data._split_of(data._path_key(spec("a", 0, bw)), 0, 0.3, 0.3) for bw in range(200)}
    assert many == {"train", "val", "test"}


@pytest.mark.uml
def test_exported_graph_runs_in_the_guest(build: object, tmp_path: Path) -> None:
    from expliccit_env.episode import EpisodeOk, run_episode  # noqa: PLC0415
    from expliccit_env.spec import EpisodeSpec  # noqa: PLC0415
    from expliccit_env.uml import UmlBuild  # noqa: PLC0415

    assert isinstance(build, UmlBuild)
    torch.manual_seed(0)
    scaler = Scaler.fit(
        EXPLICCIT_OBS_COLS,
        np.random.default_rng(0).normal(size=(64, 70)).astype(np.float32),
        "robust",
    )
    lcf = Scaler.fit(tuple(range(7)), np.zeros((4, 7), np.float32), "standard")
    graph = PolicyGraph(
        scaler, ExpliccitNet1(70, 7, width=64).eval(), lcf, ExpliccitActor(7).eval(), 0.05
    )
    policy = tmp_path / "policy.onnx"
    policy.write_bytes(to_onnx(graph))
    spec = EpisodeSpec.parse(
        json.dumps(
            {
                "path": {
                    "trace": {"kind": "mahak_step", "bw_mbps": 20, "step": 1.0},
                    "delay": {"mahak_compat_mrtt_ms": 10},
                    "buffer": {"packets": 100},
                },
                "cc": {
                    "kind": "agent",
                    "policy": {
                        "path": str(policy),
                        "sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
                    },
                },
                "duration_ms": 2000,
                "seed": 1,
            }
        )
    )
    out = run_episode(spec, build, tmp_path / "w", timeout_s=300)
    assert isinstance(out, EpisodeOk), out.row
    applied = out.tables["steps"]["applied_action"].to_numpy()
    assert applied.size > 100 and np.all(np.abs(applied) <= 0.05)


@pytest.mark.uml
def test_online_loop_runs_an_iteration(build: object, tmp_path: Path) -> None:
    from expliccit_env.uml import UmlBuild  # noqa: PLC0415

    assert isinstance(build, UmlBuild)
    steps = synthetic()
    steps.labels["drop_fraction"] = np.zeros_like(steps.obs[:, 0])
    net1, _ = fit.fit_net1(Net1Spec(width=64, epochs=1, batch_size=256), steps, 0)
    fit.save_net1(net1, Net1Spec(width=64), tmp_path / "net1.pt")
    path = {
        "trace": {"kind": "periodic_levels", "levels_mbps": [24], "level_ms": 1000},
        "delay": {"mahak_compat_mrtt_ms": 10},
        "buffer": {"packets": 100},
    }
    camp = {
        "name": "p",
        "ccs": [{"kind": "stock", "name": "cubic"}],
        "sweep": {"kind": "paths", "paths": [path]},
    }
    (tmp_path / "paths.json").write_text(json.dumps(camp))
    spec = OnlineSpec(
        paths_from=tmp_path / "paths.json",
        eval_from=tmp_path / "paths.json",
        net1_from=tmp_path / "net1.pt",
        net2=Net2Spec(hidden=16, batch_size=64, reward="utility"),
        iterations=1,
        episodes_per_iteration=2,
        duration_ms=3000,
        updates_per_iteration=20,
    )
    report = run_online(spec, tmp_path / "out", build, workers=2)
    (entry,) = report["iterations"]
    assert entry["rollouts"]["ok"] == 2 and entry["new_transitions"] > 400
    assert entry["eval"]["ok"] == 1
    assert (tmp_path / "out" / "policy.onnx").exists()
