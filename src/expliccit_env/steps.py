"""Sage's steps (contract sage-v1): decoding steps.bin and the training table.

The guest writes one fixed-size record per step (guest/src/step.rs): the
step's CLOCK_MONOTONIC time in us, the TCP_INFO prefix it was computed from,
Sage's 77-value message, the action a policy took after it (NaN under a stock
controller) and the window that action set (0 if none). ``step_table`` joins
each step with the ground-truth labels at its instant, Sage's reward
computed from the exact capacity, and the action taken from it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from expliccit_env import labels
from expliccit_env import mahimahi as mm
from expliccit_env.policy import OBS_DIM, PREV_ACTION_COL
from expliccit_env.records import Records

INFO_LEN: Final = 224
STEP: Final = np.dtype(
    [
        ("t_us", "<u8"),
        ("info", f"V{INFO_LEN}"),
        ("obs", "<f8", (OBS_DIM,)),
        ("applied_action", "<f8"),
        ("cwnd_set", "<u4"),
        ("pad", "<u4"),
    ]
)
type Steps = npt.NDArray[np.void]
type F64 = npt.NDArray[np.float64]

# Message columns the reward reads (policy.OBS_COLUMNS).
_RTT_RATE, _LOSS_X100MBPS, _DELIVERED_X100MBPS = 66, 67, 71


class StepError(ValueError):
    pass


def load_steps(path: Path) -> Steps:
    raw = path.read_bytes()
    if len(raw) % STEP.itemsize:
        raise StepError(f"{path.name}: {len(raw)} bytes is not a whole number of steps")
    steps = np.frombuffer(raw, dtype=STEP)
    if steps.size and np.any(np.diff(steps["t_us"].astype(np.int64)) <= 0):
        raise StepError(f"{path.name}: step times are not increasing")
    return steps


def sage_reward(obs: F64, capacity_mbps: F64) -> F64:
    """Sage's single-flow reward (sage.cc), with the link's exact capacity
    where Sage uses the nominal rate from its command line:
    sign * 25 (rate - 2 loss)^2 / capacity^2 * min(1 if >= 0.8, min_rtt/srtt),
    the rate capped at 0.9 capacity when it exceeds it, the sign negative
    when the rate is below twice the loss rate. NaN where capacity is 0."""
    rate = obs[:, _DELIVERED_X100MBPS] * 100.0
    loss = obs[:, _LOSS_X100MBPS] * 100.0
    rtt_rate = obs[:, _RTT_RATE]
    rtt_term = np.where(rtt_rate >= 0.8, 1.0, rtt_rate)
    sign = np.where(rate < 2 * loss, -1.0, 1.0)
    rate = np.where(rate > capacity_mbps, capacity_mbps * 0.9, rate)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = sign * 25 * (rate - 2 * loss) ** 2 / capacity_mbps**2 * rtt_term
    return np.where(capacity_mbps > 0, r, np.nan)


def step_table(
    steps: Steps,
    rec: Records,
    trace: mm.I64,
    k: labels.PathConstants,
    *,
    duration_ms: int,
) -> pa.Table:
    """One row per step: observation, labels at the step, reward, and the
    action taken from it (the next step's column 76, as Sage's
    gen_offline_data.py derives it; NaN on the last step, ``done``)."""
    n = steps.size
    t_ns = steps["t_us"].astype(np.int64) * 1000
    obs = np.ascontiguousarray(steps["obs"], dtype=np.float64).reshape(n, OBS_DIM)
    lab = labels.labels_at(rec, trace, k, t_ns, duration_ms=duration_ms)
    action = np.append(obs[1:, PREV_ACTION_COL], np.nan)
    capacity_mbps = cast("F64", lab["capacity_bps"]) / 1e6
    # pyarrow-stubs leave FixedSizeListArray's type parameters unknown.
    obs_col = cast(
        "pa.Array[Any]",
        pa.FixedSizeListArray.from_arrays(pa.array(obs.reshape(-1)), OBS_DIM),  # pyright: ignore[reportUnknownMemberType]
    )
    cols: dict[str, pa.Array[Any]] = {
        "step": pa.array(np.arange(n, dtype=np.int32)),
        "t_ns": pa.array(t_ns.astype(np.uint64)),
        "obs": obs_col,
        "action": pa.array(action),
        "reward": pa.array(sage_reward(obs, capacity_mbps)),
        "done": pa.array(np.arange(n) == n - 1),
        "applied_action": pa.array(np.asarray(steps["applied_action"], dtype=np.float64)),
        "cwnd_set": pa.array(np.asarray(steps["cwnd_set"], dtype=np.uint32)),
        **{f"label_{name}": pa.array(v) for name, v in lab.items()},
    }
    return pa.Table.from_pydict(cols)
