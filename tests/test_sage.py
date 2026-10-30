"""Sage's step (contract sage-v1) in UML: the guest's port against Sage's own
code, and the step training table. Needs EXPLICCIT_UML_DIR; the comparison
also needs EXPLICCIT_SAGE_DIR (a Sage checkout at sage_ref.SAGE_COMMIT) and g++.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from expliccit_env.episode import EpisodeOk, run_episode
from expliccit_env.policy import OBS_DIM, PREV_ACTION_COL
from expliccit_env.spec import EpisodeSpec
from expliccit_env.steps import load_steps
from expliccit_env.uml import UmlBuild
from tests import sage_ref

pytestmark = pytest.mark.uml

# Columns the port does not take from Sage's sage.cc: 1 and 75 (Sage's
# nominal capacity and reward, zeroed here), 76 (filled by the step loop) and
# 77 (expliCCit's addition, checked in test_step_table).
_NOT_COMPARED = {1, 75, PREV_ACTION_COL, 77}
_INT_COLS = {8, 9}  # printed with %d and %u


def run(build: UmlBuild, tmp: Path, cc: str, loss: float) -> tuple[EpisodeOk, Path]:
    spec = EpisodeSpec.parse(
        json.dumps(
            {
                "path": {
                    "trace": {"kind": "mahak_step", "bw_mbps": 24, "step": 0.5},
                    "delay": {"base_rtt_ms": 40},
                    "buffer": {"packets": 80},
                    "loss": {"downlink": loss},
                },
                "cc": {"kind": "stock", "name": cc},
                "duration_ms": 14_000,
                "seed": 9,
                "steps": True,
            }
        )
    )
    out = run_episode(spec, build, tmp, timeout_s=300, keep_raw=True)
    assert isinstance(out, EpisodeOk), out.row
    return out, tmp / spec.episode_id() / "raw"


@pytest.mark.skipif(
    "EXPLICCIT_SAGE_DIR" not in os.environ or not sage_ref.available(),
    reason="set EXPLICCIT_SAGE_DIR to a Sage checkout at sage_ref.SAGE_COMMIT; needs g++",
)
@pytest.mark.parametrize(("cc", "loss"), [("cubic", 0.0), ("bbr", 0.01)])
def test_step_matches_sage_cc(build: UmlBuild, tmp_path: Path, cc: str, loss: float) -> None:
    out, raw = run(build, tmp_path, cc, loss)
    steps = load_steps(raw / "steps.bin")
    exe = sage_ref.build(Path(os.environ["EXPLICCIT_SAGE_DIR"]), tmp_path)
    base_ns = int(out.row["base_ns"])  # type: ignore[call-overload]
    readings = [(int(s["t_us"]), bytes(s["info"])) for s in steps]
    theirs = sage_ref.run(exe, readings, start_us=base_ns // 1000, base_ms=base_ns // 1_000_000)
    assert len(theirs) == steps.size  # every recorded step moved the ACK state
    for k, (fields, ours) in enumerate(zip(theirs, steps["obs"], strict=True)):
        assert len(fields) == OBS_DIM - 1, k
        for c in range(OBS_DIM - 1):
            if c in _NOT_COMPARED:
                continue
            mine = str(int(ours[c])) if c in _INT_COLS else f"{ours[c]:.7f}"
            assert mine == fields[c], (k, c, mine, fields[c])


def test_step_table(build: UmlBuild, tmp_path: Path) -> None:
    out, raw = run(build, tmp_path, "cubic", 0.0)
    steps = load_steps(raw / "steps.bin")
    table = out.tables["steps"]
    assert table.num_rows == steps.size == out.row["n_steps"]
    obs = np.asarray(table["obs"].combine_chunks().flatten()).reshape(-1, OBS_DIM)
    # Column 76 under a stock controller: log2 of the window's change since
    # the previous step (from 10 packets), rounded as sage.cc prints it.
    info = np.frombuffer(steps["info"].tobytes(), np.uint8).reshape(-1, 224)
    cwnd = info[:, 80:84].copy().view("<u4").ravel().astype(np.float64)
    min_rtt = info[:, 148:152].copy().view("<u4").ravel().astype(np.float64)
    assert np.array_equal(obs[:, 77], min_rtt / 1000.0)  # expliCCit's column
    ratio = cwnd / np.concatenate(([10.0], cwnd[:-1]))
    x = np.log2(ratio.astype(np.float32)) * np.float32(1000)
    rounded = np.sign(x) * np.floor(np.abs(x) + np.float32(0.5))  # C round(): half away from 0
    assert np.array_equal(obs[:, PREV_ACTION_COL], rounded.astype(np.float64) / 1000.0)
    # The action taken from a step is the next step's column 76.
    act = table["action"].to_numpy()
    assert np.array_equal(act[:-1], obs[1:, PREV_ACTION_COL]) and np.isnan(act[-1])
    # Labels at step times are the ground truth: 24 then 12 Mbit/s.
    t_s = obs[:, 0]
    cap = table["label_capacity_bps"].to_numpy()
    assert np.allclose(cap[(t_s > 1) & (t_s < 11.9)], 24e6, rtol=0.02)
    assert np.allclose(cap[t_s > 12.2], 12e6, rtol=0.02)
    busy = table["label_busy_fraction"].to_numpy()
    assert np.nanmin(busy) >= 0 and np.nanmax(busy) <= 1
    reward = table["reward"].to_numpy()
    assert np.isfinite(reward).all() and reward.max() > 10  # Sage's scale tops out at 25
