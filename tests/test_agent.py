"""Closed-loop sage-v1 policies driving tcp_agent in UML (needs EXPLICCIT_UML_DIR)."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from expliccit_env.episode import EpisodeOk, run_episode
from expliccit_env.policy import OBS_DIM
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


def agent_spec(tmp: Path, policy: bytes, mode: str) -> EpisodeSpec:
    path = tmp / "policy.onnx"
    path.write_bytes(policy)
    cc = AgentCC(
        mode=mode,  # type: ignore[arg-type]
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


def sage_window(actions: list[float]) -> list[int]:
    """Sage's accumulator: from 10 packets, times 2^a, rounded up, in [4, 20000]."""
    precise, out = 10.0, []
    for a in actions:
        precise = min(precise * 2.0 ** float(np.float32(a)), 2147483647.0)
        cwnd = min(math.ceil(precise), 2147483647)
        if cwnd < 4:
            cwnd, precise = 4, 4.0
        if cwnd > 20000:
            cwnd, precise = 20000, 20000.0
        out.append(cwnd)
    return out


def steps_of(outcome: EpisodeOk) -> pa.Table:
    return outcome.tables["steps"]


def test_hold_agent_follows_sage_window(build: UmlBuild, tmp_path: Path) -> None:
    outcome, rec = run_raw(
        build, tmp_path, agent_spec(tmp_path, onnx_policies.constant(0.1), "hold")
    )
    acts, steps = rec.agent_action, steps_of(outcome)
    n = steps.num_rows
    assert 250 <= n <= 300  # about one step per 10 ms over 3 s
    assert acts.size == n == outcome.row["agent_actions"]
    assert acts["cwnd"].tolist() == sage_window([0.1] * n)
    assert set(steps["applied_action"].to_numpy().tolist()) == {float(np.float32(0.1))}
    # The next step reports the action taken as Sage's column 76.
    assert np.all(steps["action"].to_numpy()[:-1] == 0.1)
    assert np.all(acts["pacing_rate"][5:] > 0)  # paced once srtt is known
    # Hold keeps the socket at exactly what the policy set.
    obs = rec.tcp_obs
    k = np.searchsorted(acts["t_ns"], obs["t_ns"], side="right") - 1
    seen = obs["snd_cwnd"][1:][k[1:] >= 0]
    want = acts["cwnd"][k[1:][k[1:] >= 0]]
    assert np.mean(seen == want) > 0.95


def test_policy_reads_the_observation(build: UmlBuild, tmp_path: Path) -> None:
    # a = 0.5 r - 0.35 with r = Sage's column 70 (max delivered rate times min
    # RTT over the window). Every action must be that function of the very
    # message the step table recorded, and the window must react to it.
    w = np.zeros(OBS_DIM)
    w[70] = 0.5
    outcome, rec = run_raw(
        build, tmp_path, agent_spec(tmp_path, onnx_policies.linear(w, -0.35), "hold")
    )
    steps = steps_of(outcome)
    obs = np.asarray(steps["obs"].combine_chunks().flatten()).reshape(-1, OBS_DIM)
    want = np.float32(0.5) * obs[:, 70].astype(np.float32) + np.float32(-0.35)
    assert np.array_equal(steps["applied_action"].to_numpy(), want.astype(np.float64))
    cwnd = rec.agent_action["cwnd"]
    assert len(set(cwnd.tolist())) > 5 and cwnd[-1] > cwnd.min()


def test_recurrent_state_round_trips(build: UmlBuild, tmp_path: Path) -> None:
    outcome, _ = run_raw(build, tmp_path, agent_spec(tmp_path, onnx_policies.counter(1e-3), "hold"))
    applied = steps_of(outcome)["applied_action"].to_numpy()
    expect = np.array(
        [float(np.float32(np.float32(k) * np.float32(1e-3))) for k in range(applied.size)]
    )
    assert np.array_equal(applied, expect)


def test_reno_agent_grows_between_actions(build: UmlBuild, tmp_path: Path) -> None:
    _, rec = run_raw(build, tmp_path, agent_spec(tmp_path, onnx_policies.constant(0.0), "reno"))
    obs, acts = rec.tcp_obs, rec.agent_action
    assert set(acts["cwnd"].tolist()) == {10}
    between = (obs["t_ns"] > acts["t_ns"][3]) & (obs["t_ns"] < acts["t_ns"][4])
    assert obs["snd_cwnd"][between].max() > 10  # Reno grew it before the next action


def test_bad_policy_output_fails_the_episode(build: UmlBuild, tmp_path: Path) -> None:
    outcome = run_episode(
        agent_spec(tmp_path, onnx_policies.constant(20.0), "hold"), build, tmp_path / "w"
    )
    assert not isinstance(outcome, EpisodeOk)
    assert "exceeds" in str(outcome.row["failure"])


def test_exploration_noise_is_seeded_and_has_the_requested_spread(
    build: UmlBuild, tmp_path: Path
) -> None:
    base = agent_spec(tmp_path, onnx_policies.constant(0.0), "hold")
    noisy = base.model_copy(update={"cc": base.cc.model_copy(update={"noise_sigma": 0.1})})
    runs = [
        steps_of(run_raw(build, tmp_path / name, spec)[0])["applied_action"].to_numpy()
        for name, spec in (
            ("a", noisy),
            ("b", noisy),
            ("c", noisy.model_copy(update={"seed": 4})),
        )
    ]
    a, b, c = runs
    assert np.array_equal(a, b)  # same episode seed, same noise
    assert not np.array_equal(a[:50], c[:50])
    assert abs(float(a.mean())) < 0.02 and 0.08 < float(a.std()) < 0.12
