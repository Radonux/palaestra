"""The path's true state, computed live in the guest for oracle and patched
policies (policy.TRUTH_COLUMNS), against the host's labels (needs
EXPLICCIT_UML_DIR)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from expliccit_env.episode import EpisodeOk, run_episode
from expliccit_env.policy import TRUTH_COLUMNS, TRUTH_DIM
from expliccit_env.spec import EpisodeSpec
from expliccit_env.steps import load_steps
from expliccit_env.uml import UmlBuild
from tests import onnx_policies
from tests.test_agent import agent_spec, run_raw

pytestmark = pytest.mark.uml


def test_live_truth_equals_the_labels(build: UmlBuild, tmp_path: Path) -> None:
    # Every code path at once: a capacity step (20 -> 4 Mbit/s at 12 s) into
    # a 60-packet buffer, so the queue fills and drops; random loss on both
    # directions; UDP and TCP cross traffic, so departures of other flows
    # count in the queueing delay but not in the receive rate.
    spec = EpisodeSpec.parse(
        json.dumps(
            {
                "path": {
                    "trace": {"kind": "mahak_step", "bw_mbps": 20, "step": 0.2},
                    "delay": {"base_rtt_ms": 30},
                    "buffer": {"packets": 60},
                    "loss": {"downlink": 0.01, "uplink": 0.005},
                },
                "cc": {"kind": "stock", "name": "bbr"},
                "duration_ms": 14_000,
                "seed": 4,
                "steps": True,
                "cross": [
                    {"kind": "udp_cbr", "rate_mbps": 3, "start_ms": 2000, "stop_ms": 5000},
                    {"kind": "tcp", "cc": "cubic", "start_ms": 4000},
                ],
            }
        )
    )
    out = run_episode(spec, build, tmp_path, timeout_s=300, keep_raw=True)
    assert isinstance(out, EpisodeOk), out.row
    live = load_steps(tmp_path / spec.episode_id() / "raw" / "steps.bin")["truth"]
    table = out.tables["steps"]
    assert table.num_rows == live.shape[0] > 1300
    # Every stored label but the one known only in hindsight.
    stored = {c.removeprefix("label_") for c in table.column_names if c.startswith("label_")}
    assert stored - set(TRUTH_COLUMNS) == {"queue_delay_ns"} and set(TRUTH_COLUMNS) <= stored
    for k, name in enumerate(TRUTH_COLUMNS):
        want = table[f"label_{name}"].to_numpy().astype(np.float64)
        got = live[:, k]
        bad = ~((got == want) | (np.isnan(got) & np.isnan(want)))
        assert not bad.any(), (name, int(bad.sum()), got[bad][:5], want[bad][:5])
    # The episode exercised what it should: drops, a full window of
    # departures, the capacity step, cross traffic in the delay.
    q = live[:, TRUTH_COLUMNS.index("queue_pkts")]
    cap = live[:, TRUTH_COLUMNS.index("capacity_bps")]
    assert q.max() >= 55 and np.isfinite(live[:, 0]).mean() > 0.9
    assert cap.max() > 15e6 and cap.min() < 5e6

    def col(name: str) -> np.ndarray:
        return live[:, TRUTH_COLUMNS.index(name)]

    assert col("active_flows").max() == 3 and col("flow_queue_pkts").max() > 20
    assert np.nanmax(col("drop_fraction")) > 0 and np.nanmax(col("random_drop_fraction")) > 0
    assert (
        np.nanmax(col("ack_queue_delay_mean_ns")) > 0
        and col("cross_traffic_share_10ms").max() > 0.1
    )
    assert col("queue_min_pkts").max() > 20 and (col("capacity_ahead_bps") != cap).any()


@pytest.mark.parametrize("width", [7, TRUTH_DIM])
def test_policy_reads_the_truth(build: UmlBuild, tmp_path: Path, width: int) -> None:
    # a = 0.05 - 0.002 * (true queue): grow while the queue is short, back
    # off as it builds. Every action must be that function of the truth the
    # step recorded, computed in float32 as the graph computes it. Width 7 is
    # a policy exported when the truth held expliCCit's seven: it reads the
    # same columns, the truth's first.
    col = TRUTH_COLUMNS.index("queue_pkts")
    spec = agent_spec(tmp_path, onnx_policies.on_truth(col, -0.002, 0.05, width), "hold")
    outcome, _ = run_raw(build, tmp_path, spec)
    live = load_steps(tmp_path / "work" / outcome.episode_id / "raw" / "steps.bin")["truth"]
    q = live[:, col].astype(np.float32)
    want = (np.float32(-0.002) * q + np.float32(0.05)).astype(np.float64)
    assert np.array_equal(outcome.tables["steps"]["applied_action"].to_numpy(), want)
    assert q.max() > 25  # it filled past the point where it backs off


def test_exported_oracle_runs_in_the_guest(build: UmlBuild, tmp_path: Path) -> None:
    # A net2 on the true LCFs, exported as the trainer exports it, acts in
    # the guest exactly as torch computes it from the truth each step recorded.
    torch = pytest.importorskip("torch")
    from expliccit_env.train import fit  # noqa: PLC0415
    from expliccit_env.train.export import to_onnx  # noqa: PLC0415
    from expliccit_env.train.models import ExpliccitActor  # noqa: PLC0415
    from tests.test_train import _net1  # noqa: PLC0415

    torch.manual_seed(0)
    net1 = _net1()
    graph = fit.policy_graph(net1, ExpliccitActor(7).eval(), 0.05, truth=frozenset(net1.lcfs))
    outcome, _ = run_raw(build, tmp_path, agent_spec(tmp_path, to_onnx(graph), "hold"))
    steps = load_steps(tmp_path / "work" / outcome.episode_id / "raw" / "steps.bin")
    obs = torch.from_numpy(steps["obs"].astype(np.float32))
    truth = torch.from_numpy(steps["truth"].astype(np.float32))
    with torch.no_grad():
        want = graph(obs, truth).numpy().ravel()
    applied = outcome.tables["steps"]["applied_action"].to_numpy()
    assert applied.size > 100 and np.isfinite(applied).all()
    assert np.allclose(applied, want, rtol=0, atol=1e-6)


# The newer labels, a 10 ms window, and composites the graph computes from
# the truth's columns with the labels' formulas.
_NEWER = ("active_flows", "queue_min_pkts_10ms", "fair_share_bps", "normalized_inflight")


@pytest.mark.parametrize("extra", [(), _NEWER])
def test_exported_distribution_policy_runs_in_the_guest(
    build: UmlBuild, tmp_path: Path, extra: tuple[str, ...]
) -> None:
    # A distribution net1 whose busy-fraction belief (and ``extra``) is
    # patched from the truth: the guest must act as torch computes from the
    # recorded steps.
    torch = pytest.importorskip("torch")
    from expliccit_env.train import fit  # noqa: PLC0415
    from expliccit_env.train.export import to_onnx  # noqa: PLC0415
    from expliccit_env.train.models import ExpliccitActor  # noqa: PLC0415
    from expliccit_env.train.spec import EXPLICCIT_LCFS, Bins, Net1Spec  # noqa: PLC0415
    from tests.test_train import synthetic  # noqa: PLC0415

    torch.manual_seed(0)
    steps = synthetic(n_episodes=10)
    rng = np.random.default_rng(0)
    lcfs = (*EXPLICCIT_LCFS, *extra)
    steps.labels.update({n: rng.uniform(0, 1, steps.obs.shape[0]).astype(np.float32) for n in lcfs})
    spec = Net1Spec(width=64, epochs=1, batch_size=256, output=Bins(bins=16), lcfs=lcfs)
    net1, _ = fit.fit_net1(spec, steps, seed=0)
    graph = fit.policy_graph(
        net1, ExpliccitActor(net1.width).eval(), 0.05, truth=frozenset({"busy_fraction", *extra})
    )
    outcome, _ = run_raw(build, tmp_path, agent_spec(tmp_path, to_onnx(graph), "hold"))
    steps_bin = load_steps(tmp_path / "work" / outcome.episode_id / "raw" / "steps.bin")
    obs = torch.from_numpy(steps_bin["obs"].astype(np.float32))
    truth = torch.from_numpy(steps_bin["truth"].astype(np.float32))
    with torch.no_grad():
        want = graph(obs, truth).numpy().ravel()
    applied = outcome.tables["steps"]["applied_action"].to_numpy()
    assert applied.size > 100 and np.isfinite(applied).all()
    assert np.allclose(applied, want, rtol=0, atol=1e-6)
