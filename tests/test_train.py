"""The trainer (src/expliccit_env/train): needs the train extra (PyTorch)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import pytest

torch = pytest.importorskip("torch")

from expliccit_env.policy import EXPLICCIT_OBS_COLS, OBS_DIM, TRUTH_COLUMNS, TRUTH_DIM  # noqa: E402
from expliccit_env.train import data, fit  # noqa: E402
from expliccit_env.train.export import to_onnx  # noqa: E402
from expliccit_env.train.models import (  # noqa: E402
    ExpliccitActor,
    ExpliccitNet1,
    IntervalNet1,
    Scaler,
)
from expliccit_env.train.score import utility  # noqa: E402
from expliccit_env.train.spec import (  # noqa: E402
    EXPLICCIT_LCFS,
    LCF_UNITS,
    Bins,
    Interval,
    Net1Spec,
    Net2Spec,
    Point,
)


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
        cwnd=rng.integers(4, 400, n).astype(np.float32),
        episode=episode.astype(np.int64),
        step=np.tile(np.arange(steps), n_episodes).astype(np.int64),
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
        width=64, epochs=30, patience=5, batch_size=256, lcfs=("capacity_bps",), output=Interval()
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
    r = utility(steps.labels, Net2Spec(reward="utility", delay_weight=0.5, loss_weight=4.0))
    assert np.allclose(r, [1.0, 0.5, 1.0 - 0.25 - 0.04])
    lab["receive_rate_bps"][0] = 0.0  # starved: the log is floored at 0.01
    r = utility(steps.labels, Net2Spec(reward="log_utility", delay_weight=0.5, loss_weight=4.0))
    assert np.allclose(r, [np.log(0.01), np.log(0.5), -0.25 - 0.04])


def _bandit(n: int = 4096) -> fit.Transitions:
    """One step per transition, two logged actions, the larger one better."""
    g = torch.Generator().manual_seed(0)
    s = torch.randn(n, 3, generator=g)
    a = (torch.randint(0, 2, (n, 1), generator=g).float() - 0.5).clone()
    return fit.Transitions(s, a, a.squeeze(1).clone(), torch.randn(n, 3, generator=g))


def test_n_step_transitions_sum_the_rewards_of_consecutive_steps() -> None:
    steps = synthetic(n_episodes=2, steps=6)  # episode 0 is in the test split
    steps.step[3:6] += 1  # episode 0's steps: 0, 1, 2, 4, 5, 6 (a gap after row 2)
    states = np.arange(12, dtype=np.float32).reshape(-1, 1)
    spec = Net2Spec(n_step=2, discount=0.5, reward_scale=1.0)
    t, _ = fit.transitions(steps, states, np.zeros(12, np.float32), spec, "test")
    # Starts 1 and 2 would span the gap, 4 and 5 the episode's end.
    assert t.s[:, 0].tolist() == [0.0, 3.0] and t.s2[:, 0].tolist() == [2.0, 5.0]
    r = steps.reward
    assert np.allclose(t.r.numpy(), [r[1] + 0.5 * r[2], r[4] + 0.5 * r[5]])


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
                "step": [k for _, _, n in eps for k in range(n)],
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
    # Every second step of each episode: steps 0 and 2 of e0, 0 of e1, 0 and 2 of e2.
    assert data.load([tmp_path], seed=0, val=0.0, test=0.0, stride=2).obs.shape[0] == 5
    # With pairs, every third step and the one after it: 0 and 1 of each
    # episode, and step 3 of e2 (its successor would be step 4, past the end).
    paired = data.load([tmp_path], seed=0, val=0.0, test=0.0, stride=3, pairs=True)
    assert paired.step.tolist() == [0, 1, 0, 1, 0, 1, 3]


def test_load_orders_rows_whatever_the_shard_order(tmp_path: Path) -> None:
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    def shard(d: Path, rows: list[tuple[str, int]]) -> None:
        d.mkdir(parents=True)
        eps = sorted({e for e, _ in rows})
        pq.write_table(
            pa.table(
                {
                    "episode_id": eps,
                    "status": ["ok"] * len(eps),
                    "spec_json": [json.dumps({"path": {"bw": i}}) for i in range(len(eps))],
                    "controller": ["bbr"] * len(eps),
                }
            ),
            d / "episodes.parquet",
        )
        obs = np.array([[ord(e[-1]), k] + [0] * (OBS_DIM - 2) for e, k in rows], np.float32)
        pq.write_table(
            pa.table(
                {
                    "episode_id": [e for e, _ in rows],
                    "step": [k for _, k in rows],
                    "obs": pa.FixedSizeListArray.from_arrays(pa.array(obs.ravel()), OBS_DIM),
                    "action": np.zeros(len(rows), np.float32),
                    "reward": np.zeros(len(rows), np.float32),
                }
            ),
            d / "steps.parquet",
        )

    rows = [(e, k) for e in ("a", "b", "c") for k in range(4)]
    shard(tmp_path / "x" / "s", rows)
    shard(tmp_path / "y" / "s", rows[::-1])  # as workers that finished in another order
    x, y = (data.load([tmp_path / d], seed=0, val=0.3, test=0.3, split_by="row") for d in "xy")
    assert x.episode_ids == y.episode_ids == ("a", "b", "c")
    assert np.array_equal(x.obs, y.obs) and np.array_equal(x.split, y.split)
    assert x.step.tolist() == [0, 1, 2, 3] * 3 and x.episode.tolist() == [0] * 4 + [1] * 4 + [2] * 4


def test_as_deployed_rounds_beliefs_and_actions() -> None:
    torch.manual_seed(0)
    net1 = _net1()
    actor = ExpliccitActor(7).eval()
    rounded = (True,) * 6 + (False,)
    g = fit.policy_graph(net1, actor, 1.0, deployed=rounded)
    obs = torch.randn(8, OBS_DIM)
    with torch.no_grad():
        a = g(obs).numpy()
        z = g.beliefs(obs)
        c = z * net1.lcf_scaler.spread + net1.lcf_scaler.center
    assert np.array_equal(a, np.round(a * 1000) / 1000)
    assert torch.allclose(c[:, :6], torch.round(c[:, :6]), atol=1e-3) and (c >= -1e-4).all()
    with pytest.raises(ValueError, match="point net1"):
        fit.policy_graph(_net1(intervals=True), ExpliccitActor(21), 1.0, deployed=rounded)


def test_export_is_a_sage_v1_graph() -> None:
    graph = fit.policy_graph(_net1(), ExpliccitActor(7).eval(), 1.0)
    model = onnx.load_from_string(to_onnx(graph))
    (inp,) = model.graph.input
    (out,) = model.graph.output
    dims = [[d.dim_value for d in v.type.tensor_type.shape.dim] for v in (inp, out)]
    assert (inp.name, out.name, dims) == ("obs", "a", [[1, OBS_DIM], [1, 1]])
    assert not {"IsInf", "IsNaN"} & {n.op_type for n in model.graph.node}  # tract lacks them


def _net1(intervals: bool = False) -> fit.Net1:
    rng = np.random.default_rng(0)
    scaler = Scaler.fit(EXPLICCIT_OBS_COLS, rng.normal(size=(64, 70)).astype(np.float32), "robust")
    y = (rng.uniform(size=(64, 7)) * [5e6, 50, 3e7, 100, 1, 2e7, 2e7]).astype(np.float32)
    units = np.array([LCF_UNITS[n][0] for n in EXPLICCIT_LCFS], np.float32)
    lcf = Scaler.fit(tuple(range(7)), y * units, "standard")
    base = ExpliccitNet1(70, 7, width=64, per_lcf=3 if intervals else 1)
    net = IntervalNet1(base, 7) if intervals else base
    return fit.Net1(scaler, net.eval(), lcf, EXPLICCIT_LCFS)


def test_truth_columns_are_expliccits_lcfs() -> None:
    # First, so a policy exported with a seven-wide truth input still reads them.
    assert TRUTH_COLUMNS[:7] == EXPLICCIT_LCFS
    assert len(set(TRUTH_COLUMNS)) == TRUTH_DIM


def test_truth_gives_every_lcf_but_the_one_in_hindsight() -> None:
    # A composite of the truth is the composite of the labels: one formula.
    from expliccit_env.train import lcfs  # noqa: PLC0415
    from expliccit_env.train.graph import TruthColumns  # noqa: PLC0415

    rng = np.random.default_rng(0)
    truth = rng.uniform(0.5, 2.0, size=(64, TRUTH_DIM)).astype(np.float32)
    truth[:8, TRUTH_COLUMNS.index("busy_fraction")] = np.nan
    truth[8:16, TRUTH_COLUMNS.index("active_flows")] = 0
    truth[16:24, TRUTH_COLUMNS.index("queue_delay_mean_ns_10ms")] = np.nan
    names = [n for n in LCF_UNITS if n != "queue_delay_ns"]
    assert len(names) == len(LCF_UNITS) - 1
    labels = lcfs.Labels({n: truth[:, k] for k, n in enumerate(TRUTH_COLUMNS)})
    want = np.stack([labels[n] * np.float32(LCF_UNITS[n][0]) for n in names], axis=1)
    got = TruthColumns(names)(torch.from_numpy(truth)).numpy()
    assert np.array_equal(got, want, equal_nan=True)
    assert not TruthColumns.known("queue_delay_ns")
    late = TruthColumns(["queue_delay_ns", "queue_pkts"])(torch.from_numpy(truth))
    assert late[:, 0].isnan().all() and late[:, 1].isfinite().all()


@pytest.mark.parametrize("intervals", [False, True])
def test_patched_graph_takes_the_named_lcfs_from_the_truth(intervals: bool) -> None:
    torch.manual_seed(0)
    net1 = _net1(intervals)
    k = 21 if intervals else 7
    actor = ExpliccitActor(k).eval()
    obs = torch.randn(5, OBS_DIM)
    truth = torch.rand(5, TRUTH_DIM)
    truth[:, :7] *= torch.tensor([5e6, 50, 3e7, 100, 1, 2e7, 2e7])
    truth[0, 0] = float("nan")  # no departure in the window
    cap = TRUTH_COLUMNS.index("capacity_bps")
    patched = fit.policy_graph(net1, actor, 1.0, truth=frozenset({"capacity_bps"}))
    with torch.no_grad():
        z = net1.net(net1.scaler(obs))
        units = torch.tensor([LCF_UNITS[n][0] for n in EXPLICCIT_LCFS])
        zt = (truth[:, :7] * units - net1.lcf_scaler.center) / net1.lcf_scaler.spread
        for block in range(k // 7):  # the centre and, for intervals, both bounds
            z[:, block * 7 + cap] = zt[:, cap]
        assert torch.equal(patched(obs, truth), actor(z))
        oracle = fit.policy_graph(net1, actor, 1.0, truth=frozenset(EXPLICCIT_LCFS))
        other = torch.randn(5, OBS_DIM)
        assert torch.equal(oracle(obs, truth)[1:], oracle(other, truth)[1:])
        # A NaN truth keeps net1's own belief, so row 0 still reads the message.
        assert not torch.equal(oracle(obs, truth)[0], oracle(other, truth)[0])
        assert torch.isfinite(oracle(obs, truth)).all()
    model = onnx.load_from_string(to_onnx(oracle))
    assert [i.name for i in model.graph.input] == ["obs", "truth"]
    assert not {"IsInf", "IsNaN"} & {n.op_type for n in model.graph.node}
    with pytest.raises(ValueError, match="cannot take"):
        fit.policy_graph(net1, actor, 1.0, truth=frozenset({"queue_delay_ns"}))


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


def test_link_split_keeps_a_link_together_across_conditions() -> None:
    def spec(bw: int, loss: float, cross: list[str]) -> str:
        path = {"trace": {"bw": bw}, "loss": {"downlink": loss}}
        return json.dumps({"path": path, "cross": cross, "cc": "a", "seed": 0})

    variants = [spec(10, loss, cross) for loss in (0.0, 0.01) for cross in ([], ["udp"])]
    assert len({data._path_key(v, "link") for v in variants}) == 1
    assert len({data._path_key(v) for v in variants}) == 4


def test_row_split_scatters_an_episode_at_the_fractions() -> None:
    split = data._row_split(42, 0.16, 0.2, 100_000)  # pyright: ignore[reportPrivateUsage]
    share = {s: float((split == s).mean()) for s in ("train", "val", "test")}
    assert share == pytest.approx({"train": 0.64, "val": 0.16, "test": 0.2}, abs=0.01)
    # Neighbouring rows (one episode's consecutive steps) part ways.
    assert (split[1:] != split[:-1]).mean() > 0.4
    assert np.array_equal(split, data._row_split(42, 0.16, 0.2, 100_000))  # pyright: ignore[reportPrivateUsage]


@pytest.mark.uml
def test_exported_graph_runs_in_the_guest(build: object, tmp_path: Path) -> None:
    from expliccit_env.episode import EpisodeOk, run_episode  # noqa: PLC0415
    from expliccit_env.spec import EpisodeSpec  # noqa: PLC0415
    from expliccit_env.uml import UmlBuild  # noqa: PLC0415

    assert isinstance(build, UmlBuild)
    torch.manual_seed(0)
    graph = fit.policy_graph(_net1(), ExpliccitActor(7).eval(), 0.05)
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


def test_quantiles_read_a_binned_distribution() -> None:
    bounds = np.linspace(0, 10, 11, dtype=np.float32)[None, :]  # 10 bins of width 1
    p = np.full((1, 1, 10), 0.1, np.float32)  # uniform on [0, 10]
    q = fit.quantiles(p, bounds, (0.05, 0.5, 0.95))[0, 0]
    assert np.allclose(q, [0.5, 5.0, 9.5], atol=1e-5)
    one = np.zeros((1, 1, 10), np.float32)
    one[0, 0, 3] = 1.0  # all mass in [3, 4)
    assert np.allclose(fit.quantiles(one, bounds, (0.25, 0.75))[0, 0], [3.25, 3.75], atol=1e-5)


def test_bins_cover_fractions_evenly_and_rates_on_a_log_scale() -> None:
    rng = np.random.default_rng(0)
    y = np.stack([rng.uniform(0, 1, 5000), rng.lognormal(3, 1, 5000)], 1).astype(np.float32)
    y[:500, 1] = 0  # an idle link: zero receive rate
    b = fit._bin_bounds(y, 16)
    assert np.allclose(b[0], np.linspace(0, 1, 17))
    assert b[1, 0] == 0 and np.all(np.diff(b[1]) > 0)
    ratios = b[1, 2:-1] / b[1, 1:-2]
    assert np.allclose(ratios, ratios[0], rtol=1e-4)  # log-spaced


def _noisy(n_episodes: int = 60, sd_mbps: float = 2.0) -> data.Steps:
    """synthetic() with capacity noise net1 cannot predict, of known spread."""
    steps = synthetic(n_episodes=n_episodes)
    rng = np.random.default_rng(1)
    o = steps.obs
    cap = (3 * o[:, 74] - 2 * o[:, 71] + 50 + rng.normal(0, sd_mbps, o.shape[0])) * 1e6
    steps.labels["capacity_bps"] = cap.astype(np.float32)
    return steps


def test_distribution_net1_is_accurate_and_honest() -> None:
    steps = _noisy()
    spec = Net1Spec(
        width=64,
        epochs=30,
        patience=5,
        batch_size=256,
        lcfs=("capacity_bps",),
        output=Bins(bins=32),
    )
    net1, _ = fit.fit_net1(spec, steps, seed=0)
    r = fit.report_net1(net1, steps, "test")["capacity_bps"]
    d = r["distribution"]
    # The noise caps R^2 at 13 / (13 + 4) = 0.76, and RMSE at its sd, 2.
    assert r["net1"]["r2"] > 0.65 and r["net1"]["rmse"] < 2.5, r
    assert d["crps_skill"] > 0.3, d  # far better than ignoring the measurements
    # Ignoring the measurements leaves sd 4.1, the noise sd 2: at most about
    # log2(4.1 / 2) = 1.04 bits to gain.
    assert 0.6 < d["information_bits"] < 1.1, d
    assert abs(d["coverage"]["0.9"] - 0.9) < 0.05 and abs(d["coverage"]["0.5"] - 0.5) < 0.07, d
    # The 90% interval is about as wide as the noise allows: 2 x 1.645 sd.
    assert 4.0 < d["width_90"] < 10.0, d
    lo, hi = net1.bounds(steps.obs[:100])
    assert np.all(lo <= hi)
    assert net1.width == 32


def test_distribution_net1_saves_patches_and_feeds_net2(tmp_path: Path) -> None:
    torch.manual_seed(0)
    steps = _noisy(n_episodes=20)
    spec = Net1Spec(width=64, epochs=2, batch_size=256, lcfs=EXPLICCIT_LCFS, output=Bins(bins=16))
    steps.labels.update(
        {n: np.full(steps.obs.shape[0], np.nan, np.float32) for n in EXPLICCIT_LCFS}
        | {"capacity_bps": steps.labels["capacity_bps"]}
    )
    net1, _ = fit.fit_net1(spec, steps, seed=0)
    fit.save_net1(net1, spec, tmp_path / "net1.pt")
    again, _ = fit.load_net1(tmp_path / "net1.pt")
    obs = steps.obs[:8]
    assert np.array_equal(again.beliefs(obs), net1.beliefs(obs))
    actor, log = fit.fit_net2(
        Net2Spec(hidden=32, steps=50, batch_size=64, algo="bc"), steps, net1, 0
    )
    assert log["transitions"] > 0
    graph = fit.policy_graph(net1, actor, 0.1, truth=frozenset({"capacity_bps"}))
    truth = torch.full((8, TRUTH_DIM), float("nan"))
    cap = TRUTH_COLUMNS.index("capacity_bps")
    truth[:4, cap] = 37e6  # in report units 37 Mbit/s
    with torch.no_grad():
        z = net1.net(net1.scaler(torch.from_numpy(obs))).reshape(8, 7, 16)
        want = z.clone()
        b = int(net1.kind.bin_of(torch.full((1, 7), 37.0))[0, cap])  # pyright: ignore[reportAttributeAccessIssue]
        want[:4, cap] = torch.nn.functional.one_hot(torch.tensor(b), 16).float()
        assert torch.equal(graph(torch.from_numpy(obs), truth), actor(want.reshape(8, -1)) * 0.1)
    model = onnx.load_from_string(to_onnx(graph))
    assert [i.name for i in model.graph.input] == ["obs", "truth"]
    assert not {"IsInf", "IsNaN"} & {n.op_type for n in model.graph.node}


def _bdp_net1() -> tuple[data.Steps, fit.Net1]:
    """A distribution net1 on synthetic(): capacity about 50 Mbit/s, one-way
    delay 20 ms (a BDP near 167 packets); the other LCFs unknown."""
    steps = _noisy(n_episodes=20)
    keep = {"capacity_bps", "one_way_delay_ns"}
    for n in EXPLICCIT_LCFS:
        if n not in keep:
            steps.labels[n] = np.full(steps.obs.shape[0], np.nan, np.float32)
    spec = Net1Spec(width=64, epochs=2, batch_size=256, lcfs=EXPLICCIT_LCFS, output=Bins(bins=16))
    return steps, fit.fit_net1(spec, steps, seed=0)[0]


def test_bdp_action_logs_and_sets_the_window_relative_to_the_believed_bdp() -> None:
    torch.manual_seed(0)
    steps, net1 = _bdp_net1()
    spec = Net2Spec(action="bdp", input_bins=4, action_max=3.0, algo="bc", hidden=32, steps=50)
    states, log2_bdp = fit.net2_view(net1, spec, steps)
    # Each LCF's 16 bins pooled into 4, still a distribution.
    assert states.shape == (steps.obs.shape[0], 7 * 4)
    assert np.allclose(states.reshape(-1, 7, 4).sum(axis=2), 1, atol=1e-5)
    mean = net1.predict(steps.obs)
    bdp = mean[:, 2] * 1e6 * mean[:, 5] * 2e-3 / (1500 * 8)  # capacity, one-way delay
    assert np.allclose(log2_bdp, np.log2(bdp), atol=1e-4) and 6 < np.median(log2_bdp) < 8.5
    # The logged action: log2 of the next step's window over this step's
    # BDP; none at an episode's end or where the next step is missing.
    steps.step[5] += 1
    logged = fit.logged_actions(steps, spec, log2_bdp)
    t = np.array([0, 1, 3])
    assert np.allclose(logged[t], np.log2(steps.cwnd[t + 1]) - log2_bdp[t])
    assert np.isnan(logged[4]) and np.isnan(logged[199])
    actor, log = fit.fit_net2(spec, steps, net1, 0)
    assert log["transitions"] > 0
    # The graph inverts it: the window it sets is the believed BDP times 2^u.
    graph = fit.policy_graph(net1, actor, 3.0, action="bdp", input_bins=4)
    obs = torch.from_numpy(steps.obs[:8])
    cwnd = torch.tensor([[1.0], [10.0], [100.0], [150.0], [300.0], [1000.0], [5000.0], [0.0]])
    with torch.no_grad():
        a = graph(obs, cwnd=cwnd)[:, 0].numpy()
        u = actor(torch.from_numpy(states[:8]))[:, 0].numpy() * 3.0
    assert np.all(np.abs(a) <= 7.99)
    applied = np.log2(np.maximum(cwnd[:, 0].numpy(), 1)) + a - log2_bdp[:8]
    assert np.allclose(applied, u, atol=1e-4)
    model = onnx.load_from_string(to_onnx(graph))
    assert [i.name for i in model.graph.input] == ["obs", "cwnd"]
    assert not any(m.training for m in graph.modules())  # export leaves it as it was
    # A point net1 aims the window at the BDP of its single values.
    point = _net1()
    pg = fit.policy_graph(point, ExpliccitActor(7).eval(), 1.0, action="bdp")
    with torch.no_grad():
        v = point.predict(obs.numpy())
        cap, owd = EXPLICCIT_LCFS.index("capacity_bps"), EXPLICCIT_LCFS.index("one_way_delay_ns")
        bdp = np.log2(np.maximum(v[:, cap] * 1e6 * v[:, owd] * 2e-3 / 12_000, 1.0))
        assert pg.target is not None
        assert np.allclose(pg.target(obs, pg.beliefs(obs), None).numpy(), bdp, atol=1e-4)
    # The single-value view: each distribution's mean, standardized as
    # net1's targets are, with the same window target.
    means = Net2Spec(action="bdp", view="means", action_max=3.0, algo="bc", hidden=32, steps=50)
    states_m, log2_bdp_m = fit.net2_view(net1, means, steps)
    want = net1.lcf_scaler(torch.from_numpy(net1.predict(steps.obs))).numpy()
    assert states_m.shape == (steps.obs.shape[0], 7) and np.allclose(states_m, want, atol=1e-4)
    assert np.array_equal(log2_bdp_m, log2_bdp)
    actor_m, _ = fit.fit_net2(means, steps, net1, 0)
    graph_m = fit.policy_graph(net1, actor_m, 3.0, action="bdp", view="means")
    with torch.no_grad():
        a_m = graph_m(obs, cwnd=cwnd)[:, 0].numpy()
        u_m = actor_m(torch.from_numpy(states_m[:8]))[:, 0].numpy() * 3.0
    applied_m = np.log2(np.maximum(cwnd[:, 0].numpy(), 1)) + a_m - log2_bdp[:8]
    assert np.allclose(applied_m, u_m, atol=1e-4)
    with pytest.raises(ValueError, match="bins view"):
        Net2Spec(view="means", input_bins=4)


class _Counter(torch.nn.Module):
    """A net2 with memory: u is a tenth of its state, which counts steps."""

    state_dim = 1

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return 0.1 * state + 0 * x[:, :1], state + 1


def test_a_net2_with_memory_takes_its_state_and_returns_the_next() -> None:
    from onnx.reference import ReferenceEvaluator  # noqa: PLC0415

    _, net1 = _bdp_net1()
    graph = fit.policy_graph(net1, _Counter(), 1.0, action="bdp", view="means")
    model = onnx.load_from_string(to_onnx(graph))
    assert [i.name for i in model.graph.input] == ["obs", "state", "cwnd"]
    assert [o.name for o in model.graph.output] == ["a", "next_state"]
    obs, cwnd = np.zeros((1, OBS_DIM), np.float32), np.full((1, 1), 100.0, np.float32)
    run, state = ReferenceEvaluator(model), np.zeros((1, 1), np.float32)
    for k in range(3):
        a, nxt = run.run(None, {"obs": obs, "state": state, "cwnd": cwnd})
        with torch.no_grad():
            want, _ = graph.step(
                torch.from_numpy(obs), torch.from_numpy(state), cwnd=torch.from_numpy(cwnd)
            )
        assert np.allclose(a, want.numpy(), atol=1e-5) and nxt.tolist() == [[k + 1.0]]
        state = nxt
    # Called without a state, it starts from zeros, as the guest does.
    with torch.no_grad():
        first = graph.step(torch.from_numpy(obs), torch.zeros(1, 1), cwnd=torch.from_numpy(cwnd))
        assert torch.equal(graph(torch.from_numpy(obs), cwnd=torch.from_numpy(cwnd)), first[0])


def test_an_oracle_net2_on_a_distribution_net1_sees_the_true_bins() -> None:
    steps, net1 = _bdp_net1()
    states, _ = fit.net2_view(net1, Net2Spec(inputs="truth"), steps)
    z = states.reshape(-1, 7, 16)
    cap = net1.net.bin_of(torch.from_numpy(fit.targets(steps, net1.lcfs)))[:, 2].numpy()
    assert np.array_equal(z[:, 2].argmax(axis=1), cap) and np.all(z[:, 2].max(axis=1) == 1)
    beliefs = net1.beliefs(steps.obs).reshape(-1, 7, 16)
    assert np.array_equal(z[:, 0], beliefs[:, 0])  # unknown truth: net1's own belief


def test_crps_matches_known_values() -> None:
    bounds = np.linspace(0, 1, 101, dtype=np.float32)[None, :]  # 100 bins on [0, 1]
    uniform = np.full((1, 1, 100), 0.01, np.float32)
    # Uniform on [0, 1] scored at y: (y^3 + (1 - y)^3) / 3, so 1/12 at 0.5.
    score = fit.crps(uniform, bounds, np.array([[0.5]], np.float32))[0, 0]
    assert abs(score - 1 / 12) < 2e-3
    point = np.zeros((1, 1, 100), np.float32)
    point[0, 0, 20] = 1.0  # all mass in [0.20, 0.21)
    score = fit.crps(point, bounds, np.array([[0.7]], np.float32))[0, 0]
    assert abs(score - 0.495) < 6e-3  # a point forecast scores its absolute error


def test_an_ensemble_averages_its_members_and_round_trips(tmp_path: Path) -> None:
    torch.manual_seed(0)
    steps = _noisy(n_episodes=20)
    spec = Net1Spec(
        width=64, epochs=2, batch_size=256, lcfs=("capacity_bps",), output=Bins(bins=16, members=3)
    )
    net1, log = fit.fit_net1(spec, steps, seed=0)
    assert len(log["members_val_loss"]) == 3 and np.isfinite(log["ensemble_val_loss"])
    obs = steps.obs[:32]
    with torch.no_grad():
        x = net1.scaler(torch.from_numpy(obs))
        each = [torch.softmax(m(x).reshape(-1, 1, 16), dim=2) for m in net1.net.members]
    assert np.allclose(net1.distributions(obs), torch.stack(each).mean(0).numpy(), atol=1e-6)
    assert not np.allclose(each[0].numpy(), each[1].numpy())  # trained apart
    fit.save_net1(net1, spec, tmp_path / "net1.pt")
    again, _ = fit.load_net1(tmp_path / "net1.pt")
    assert np.array_equal(again.beliefs(obs), net1.beliefs(obs))


def test_expliccits_released_models_load_and_act() -> None:
    # The vendored weights (third_party/expliccit-models) load without
    # pickles and give finite beliefs and an action in [-1, 1].
    models = Path(__file__).resolve().parents[1] / "third_party" / "expliccit-models"
    net1, actor = fit.load_expliccit(models)
    obs = np.zeros((4, OBS_DIM), np.float32)
    z = net1.beliefs(obs)
    assert z.shape == (4, len(EXPLICCIT_LCFS)) and np.isfinite(z).all()
    a = actor(torch.from_numpy(z)).detach().numpy()
    assert a.shape == (4, 1) and np.all(np.abs(a) <= 1)


@pytest.mark.parametrize("inputs", ["raw", "net1+raw"])
def test_a_net2_on_the_measurements_sees_them_in_training_and_in_the_graph(inputs: str) -> None:
    torch.manual_seed(0)
    net1 = _net1()
    steps = synthetic(n_episodes=2, steps=50)
    spec = Net2Spec(inputs=inputs)  # pyright: ignore[reportArgumentType]
    states, _ = fit.net2_view(net1, spec, steps)
    raw = fit.raw_width(net1)
    assert raw == len(EXPLICCIT_OBS_COLS)
    assert states.shape[1] == fit.net2_width(net1, spec) == raw + (7 if inputs == "net1+raw" else 0)
    actor = ExpliccitActor(states.shape[1], 32).eval()
    graph = fit.policy_graph(net1, actor, 1.0, inputs=spec.inputs)
    obs = torch.from_numpy(steps.obs[:8])
    with torch.no_grad():
        assert torch.allclose(graph(obs), actor(torch.from_numpy(states[:8])), atol=1e-5)


@pytest.mark.parametrize("intervals", [False, True])
def test_an_injected_fault_scales_or_pins_one_belief_in_report_units(intervals: bool) -> None:
    torch.manual_seed(0)
    net1 = _net1(intervals)
    k = 7
    actor = ExpliccitActor(net1.width, 32).eval()
    faults = fit.parse_faults(["capacity_bps*0.7", "busy_fraction=1"], net1.lcfs)
    assert faults == {"capacity_bps": ("scale", 0.7), "busy_fraction": ("pin", 1.0)}
    obs = torch.from_numpy(synthetic(1, 8).obs)
    with torch.no_grad():
        clean = fit.policy_graph(net1, actor, 1.0).beliefs(obs)
        bad = fit.policy_graph(net1, actor, 1.0, faults=faults).beliefs(obs)
    reps = net1.width // k
    center, spread = net1.lcf_scaler.center.repeat(reps), net1.lcf_scaler.spread.repeat(reps)
    v0, v1 = clean * spread + center, bad * spread + center
    cap, busy = EXPLICCIT_LCFS.index("capacity_bps"), EXPLICCIT_LCFS.index("busy_fraction")
    for r in range(reps):
        assert torch.allclose(v1[:, r * k + cap], 0.7 * v0[:, r * k + cap], rtol=1e-4)
        assert torch.allclose(v1[:, r * k + busy], torch.ones(8), atol=1e-5)
    others = [i for i in range(net1.width) if i % k not in (cap, busy)]
    assert torch.allclose(bad[:, others], clean[:, others], atol=1e-6)
    with pytest.raises(ValueError, match="not among"):
        fit.parse_faults(["nonsense*2"], net1.lcfs)


def test_net1s_output_is_one_declared_kind() -> None:
    assert Net1Spec().output == Point()
    spec = Net1Spec.model_validate({"output": {"kind": "bins", "bins": 32, "members": 3}})
    assert spec.output == Bins(bins=32, members=3)
    with pytest.raises(ValueError, match="Extra inputs"):  # an interval has no bins
        Net1Spec.model_validate({"output": {"kind": "interval", "bins": 16}})


def test_net1_without_validation_rows_keeps_its_last_epoch_not_its_first() -> None:
    steps = synthetic(n_episodes=10, steps=100)
    steps.split[:] = "train"
    spec = Net1Spec(width=32, epochs=5, batch_size=256, lcfs=("capacity_bps",))
    torch.manual_seed(0)
    untrained = fit._net1_module(spec)  # pyright: ignore[reportPrivateUsage]
    torch.manual_seed(0)
    net1, log = fit.fit_net1(spec, steps, 0)
    assert log["epochs"] == 5  # no early stop on a missing validation loss
    changed = any(
        not torch.equal(a, b)
        for a, b in zip(
            untrained.state_dict().values(), net1.net.state_dict().values(), strict=True
        )
    )
    assert changed
