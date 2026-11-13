"""Several policies on one bottleneck, each driving its own socket, next to
stock and UDP flows (needs EXPLICCIT_UML_DIR)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from expliccit_env.episode import EpisodeOk, run_episode
from expliccit_env.labels import CROSS_PORT_BASE, MAIN_SPORT
from expliccit_env.policy import TRUTH_COLUMNS
from expliccit_env.records import Records, load_records
from expliccit_env.spec import (
    AgentCC,
    DropTail,
    EpisodeSpec,
    MahakMrtt,
    MahakStep,
    OnnxPolicy,
    PathSpec,
    TcpCross,
    UdpCross,
)
from expliccit_env.steps import load_steps
from expliccit_env.uml import UmlBuild
from tests import onnx_policies
from tests.test_agent import sage_window

pytestmark = pytest.mark.uml


def agent(tmp: Path, raw: bytes, mode: str = "hold") -> AgentCC:
    sha = hashlib.sha256(raw).hexdigest()
    path = tmp / f"{sha[:16]}.onnx"
    path.write_bytes(raw)
    return AgentCC(mode=mode, policy=OnnxPolicy(path=path, sha256=sha))  # type: ignore[arg-type]


def link(bw_mbps: float, buffer: int) -> PathSpec:
    return PathSpec(
        trace=MahakStep(bw_mbps=bw_mbps, step=1.0),
        delay=MahakMrtt(mahak_compat_mrtt_ms=10),
        buffer=DropTail(packets=buffer),
    )


def mixed(tmp: Path) -> EpisodeSpec:
    """The observed agent holds 10 packets; cross flow 0 is an agent that
    grows by 2^0.05 a step, flow 1 an agent_cubic that halves the kernel's
    window every two steps, then a Cubic flow and UDP."""
    return EpisodeSpec(
        path=link(30, 300),
        cc=agent(tmp, onnx_policies.constant(0.0)),
        cross=[
            TcpCross(cc=agent(tmp, onnx_policies.constant(0.05))),
            TcpCross(cc=agent(tmp, onnx_policies.constant(-0.5), "cubic"), start_ms=500),
            TcpCross(cc="cubic", start_ms=1000),
            UdpCross(rate_mbps=2),
        ],
        duration_ms=4000,
        seed=11,
    )


def run(build: UmlBuild, tmp: Path, spec: EpisodeSpec) -> tuple[EpisodeOk, Records, Path]:
    out = run_episode(spec, build, tmp / "work", timeout_s=600, keep_raw=True)
    assert isinstance(out, EpisodeOk), out.row
    raw = tmp / "work" / out.episode_id / "raw"
    return out, load_records(raw), raw


def flow_steps(out: EpisodeOk, flow: int) -> pa.Table:
    t = out.tables["cross_steps"]
    return t.filter(pa.array(t["flow"].to_numpy() == flow))


def test_each_agent_drives_only_its_own_socket(build: UmlBuild, tmp_path: Path) -> None:
    out, rec, _ = run(build, tmp_path, mixed(tmp_path))
    row = out.row
    # Flow 1 shrinks to 4 packets behind a full queue, so its ACK state
    # moves about once a round trip, and a step waits for that.
    assert row["cross_steps"][2:] == [0, 0] and row["cross_steps"][0] > 250
    assert row["cross_steps"][1] > 50
    assert row["cross_actions"] == row["cross_steps"]
    acts = rec.agent_action
    assert set(acts["sport"].tolist()) == {MAIN_SPORT, CROSS_PORT_BASE, CROSS_PORT_BASE + 1}
    for sport, a in ((MAIN_SPORT, 0.0), (CROSS_PORT_BASE, 0.05)):
        mine = acts[acts["sport"] == sport]
        assert mine["cwnd"].tolist() == sage_window([a] * mine.size), sport
    for sport in (MAIN_SPORT, CROSS_PORT_BASE, CROSS_PORT_BASE + 1):
        mine = acts[acts["sport"] == sport]
        assert mine["action_seq"].tolist() == list(range(1, mine.size + 1))
    # agent_cubic: each step's window is the kernel's window it read times
    # 2^a, rounded down and at least 4, and Cubic grows it in between.
    cubic = flow_steps(out, 1)
    read, wrote = cubic["cwnd"].to_numpy(), cubic["cwnd_set"].to_numpy()
    ratio = 2.0 ** cubic["applied_action"].to_numpy()
    assert np.array_equal(wrote, np.maximum(4, np.floor(ratio * read + 5e-3)))
    # The share of steps at the floor moves with packet timing (0.49 to 0.56
    # for guests whose syscall timing differs by 256 ns), so the bound is loose.
    assert np.mean(wrote == 4) > 0.3 and read.max() > 4
    assert acts[acts["sport"] == MAIN_SPORT].size == row["agent_actions"]
    grow = flow_steps(out, 0)
    assert grow.num_rows == row["cross_steps"][0]
    assert set(grow["applied_action"].to_numpy().tolist()) == {float(np.float32(0.05))}
    # Hold keeps that socket at exactly what its own policy set: the window
    # each step reads is the one the previous step wrote.
    read, wrote = grow["cwnd"].to_numpy()[1:], grow["cwnd_set"].to_numpy()[:-1]
    assert np.mean(read == wrote) > 0.95
    # The observed socket stayed at 10 while flow 0 grew past it.
    assert set(out.tables["steps"]["cwnd"].to_numpy()[3:].tolist()) == {10}
    assert wrote.max() > 100


def test_a_multi_agent_episode_is_reproducible(build: UmlBuild, tmp_path: Path) -> None:
    spec = mixed(tmp_path)
    a, ra, _ = run(build, tmp_path / "a", spec)
    b, rb, _ = run(build, tmp_path / "b", spec)
    for field in ("t_ns", "sport", "cwnd", "pacing_rate"):
        assert np.array_equal(ra.agent_action[field], rb.agent_action[field]), field
    dep_a, dep_b = ra.link_depart, rb.link_depart
    assert np.array_equal(dep_a["t_ns"], dep_b["t_ns"])
    assert np.array_equal(dep_a["sport"], dep_b["sport"])
    for col in ("t_ns", "cwnd", "applied_action"):
        assert a.tables["cross_steps"][col].equals(b.tables["cross_steps"][col]), col


def test_many_agents_share_the_bottleneck(build: UmlBuild, tmp_path: Path) -> None:
    # Ten copies of one policy that holds 30 packets, on a 20 Mbit/s link
    # with a 20 ms RTT (a BDP of 33 packets): the same window and RTT give
    # each the same share. Each flow's live truth is its own.
    pol = agent(tmp_path, onnx_policies.window_target(30.0))
    spec = EpisodeSpec(
        path=link(20, 400),
        cc=pol,
        cross=[TcpCross(cc=pol) for _ in range(9)],
        duration_ms=6000,
        seed=12,
    )
    out, _, raw = run(build, tmp_path, spec)
    assert min(out.row["cross_steps"]) > 300
    rates = [out.tables["steps"]["label_receive_rate_bps"].to_numpy()[-100:].mean()]
    for i in range(9):
        s = flow_steps(out, i)
        assert np.all(s["cwnd_set"].to_numpy()[3:] == 30)
        rates.append(s["label_receive_rate_bps"].to_numpy()[-100:].mean())
        live = load_steps(raw / f"steps-cross{i}.bin")["truth"]
        for k, name in enumerate(TRUTH_COLUMNS):
            want = s[f"label_{name}"].to_numpy().astype(np.float64)
            got = live[:, k]
            assert np.all((got == want) | (np.isnan(got) & np.isnan(want))), (i, name)
    r = np.array(rates)
    jain = r.sum() ** 2 / (r.size * (r**2).sum())
    assert jain > 0.95, r
    assert 0.8 * 20e6 < r.sum() < 1.01 * 20e6, r.sum()


def test_a_cross_flow_has_its_own_rtt(build: UmlBuild, tmp_path: Path) -> None:
    # Three flows that hold 10 packets each on a 30 Mbit/s link with a 20 ms
    # RTT (a BDP of 50 packets), so nothing queues: each flow's min RTT is
    # its base RTT, and cross flow 0's is 30 ms longer than the others'.
    pol = agent(tmp_path, onnx_policies.constant(0.0))
    spec = EpisodeSpec(
        path=link(30, 300),
        cc=pol,
        cross=[TcpCross(cc=pol, extra_rtt_ms=30), TcpCross(cc=pol)],
        duration_ms=3000,
        seed=13,
    )
    out, _, raw = run(build, tmp_path, spec)
    min_rtt = {"main": out.tables["steps"]["obs"].to_numpy(zero_copy_only=False)}
    for i in range(2):
        min_rtt[i] = flow_steps(out, i)["obs"].to_numpy(zero_copy_only=False)
    seen = {k: min(o[-1] for o in v) for k, v in min_rtt.items()}  # min_rtt_ms, column 77
    assert 20 <= seen["main"] <= 22 and 20 <= seen[1] <= 22, seen
    assert 50 <= seen[0] <= 52, seen
    # Its labels and live truth carry its own base RTT, and its one-way
    # delay is half of that; the others keep the path's.
    longer = flow_steps(out, 0)
    assert set(longer["label_base_rtt_ns"].to_numpy().tolist()) == {50_000_000}
    assert set(longer["label_one_way_delay_ns"].to_numpy().tolist()) == {25_000_000}
    assert set(flow_steps(out, 1)["label_base_rtt_ns"].to_numpy().tolist()) == {20_000_000}
    live = load_steps(raw / "steps-cross0.bin")["truth"]
    assert set(live[:, TRUTH_COLUMNS.index("one_way_delay_ns")].tolist()) == {25e6}
