"""Closed-loop agent controllers in UML (needs EXPLICCIT_UML_DIR)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from expliccit_env.episode import EpisodeOk, run_episode
from expliccit_env.policy import FEATURES
from expliccit_env.records import Records, load_records
from expliccit_env.spec import (
    AgentCC,
    DropTail,
    EpisodeSpec,
    MahakMrtt,
    MahakStep,
    OnnxPolicy,
    PathSpec,
)
from expliccit_env.uml import UmlBuild
from tests import onnx_policies

pytestmark = pytest.mark.uml


def agent_spec(tmp: Path, policy: bytes, mode: str, interval_ms: int = 20) -> EpisodeSpec:
    path = tmp / "policy.onnx"
    path.write_bytes(policy)
    cc = AgentCC(
        mode=mode,
        interval_ms=interval_ms,  # type: ignore[arg-type]
        policy=OnnxPolicy(path=path, sha256=hashlib.sha256(policy).hexdigest()),
    )
    return EpisodeSpec(
        path=PathSpec(
            trace=MahakStep(bw_mbps=20, step=1.0),
            delay=MahakMrtt(mahak_compat_mrtt_ms=10),
            buffer=DropTail(packets=200),
        ),
        cc=cc,
        duration_ms=3000,
        seed=3,
    )


def run_raw(build: UmlBuild, tmp: Path, spec: EpisodeSpec) -> tuple[EpisodeOk, Records]:
    outcome = run_episode(spec, build, tmp / "work", keep_raw=True)
    assert isinstance(outcome, EpisodeOk), outcome.row
    return outcome, load_records(tmp / "work" / outcome.episode_id / "raw")


def test_hold_agent_follows_constant_policy(build: UmlBuild, tmp_path: Path) -> None:
    outcome, rec = run_raw(
        build, tmp_path, agent_spec(tmp_path, onnx_policies.constant(40), "hold")
    )
    acts = rec.agent_action
    assert acts.size == 3000 // 20 - 1  # every 20 ms strictly before the end
    assert set(acts["cwnd"].tolist()) == {40} and set(acts["pacing_rate"].tolist()) == {0}
    obs = rec.tcp_obs
    after = obs["t_ns"] > acts["t_ns"][0]
    # obs shows the state before each ACK is processed, so skip the first one.
    assert np.all(obs["snd_cwnd"][after][1:] == 40)
    assert np.all(obs["snd_cwnd"][~after] == 10)  # initial window held until then
    assert outcome.row["agent_actions"] == acts.size


def test_policy_sees_tcp_info(build: UmlBuild, tmp_path: Path) -> None:
    # cwnd' = 0.5 * cwnd + 20: the loop converges to 40 only if the policy
    # really reads the socket's cwnd back through TCP_INFO.
    w = np.zeros((len(FEATURES), 2))
    w[FEATURES.index("cwnd_pkts"), 0] = 0.5
    _, rec = run_raw(
        build,
        tmp_path,
        agent_spec(tmp_path, onnx_policies.linear(w, np.array([20.0, 0.0])), "hold"),
    )
    cwnd = rec.agent_action["cwnd"].tolist()
    assert cwnd[0] == 25  # from the initial window of 10
    assert len(set(cwnd)) > 3  # it moves with what TCP_INFO reports
    assert set(cwnd[-10:]) == {40}  # the fixed point of c = 0.5 c + 20


def test_reno_agent_grows_between_actions(build: UmlBuild, tmp_path: Path) -> None:
    _, rec = run_raw(
        build, tmp_path, agent_spec(tmp_path, onnx_policies.constant(30), "reno", interval_ms=200)
    )
    obs, acts = rec.tcp_obs, rec.agent_action
    assert set(acts["cwnd"].tolist()) == {30}
    between = (obs["t_ns"] > acts["t_ns"][3]) & (obs["t_ns"] < acts["t_ns"][4])
    assert obs["snd_cwnd"][between].max() > 30  # Reno grew it again


def test_bad_policy_output_fails_the_episode(build: UmlBuild, tmp_path: Path) -> None:
    outcome = run_episode(
        agent_spec(tmp_path, onnx_policies.constant(1.0), "hold"), build, tmp_path / "w"
    )
    assert not isinstance(outcome, EpisodeOk)
    assert "outside [2" in str(outcome.row["failure"])
