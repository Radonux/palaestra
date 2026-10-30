"""The plan handed to the guest runner (``plan.json`` plus trace files).

Mirrors ``guest/src/plan.rs``, which re-validates everything because the
guest does not trust its input either.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

PLAN_VERSION: Final = 4
U32: Final = 2**32
MAX_TRACE_BYTES: Final = 65_000


@dataclass(frozen=True, slots=True)
class LinkPlan:
    """One direction: delay line plus trace-driven droptail link."""

    delay_ns: int
    trace: npt.NDArray[np.uint8]  # opportunities per ms (mahimahi.dense_counts)
    limit_pkts: int
    limit_bytes: int
    loss: int = 0  # i.i.d. drop probability times 2^32

    def __post_init__(self) -> None:
        if not 0 <= self.loss < U32:
            raise ValueError("loss must be a u32 (probability times 2^32)")
        if self.delay_ns < 0:
            raise ValueError("delay_ns must be nonnegative")
        if not (2 <= self.trace.size <= MAX_TRACE_BYTES) or self.trace[-1] == 0:
            raise ValueError(f"trace must have 2..{MAX_TRACE_BYTES} entries and end nonzero")
        if (
            self.limit_pkts < 0
            or self.limit_bytes < 0
            or (self.limit_pkts == 0 and self.limit_bytes == 0)
        ):
            raise ValueError("droptail needs a positive packet or byte limit")


@dataclass(frozen=True, slots=True)
class AgentPlan:
    """An ONNX policy (sage-v1 contract, see policy.py) run every step, with
    Gaussian exploration noise of ``noise_sigma`` added to each action."""

    policy_onnx: bytes
    noise_sigma: float = 0.0
    noise_seed: int = 0

    def __post_init__(self) -> None:
        if not self.policy_onnx:
            raise ValueError("agent needs a policy")
        if not 0.0 <= self.noise_sigma <= 2.0 or not 0 <= self.noise_seed < 2**64:
            raise ValueError("noise_sigma must be in [0, 2] and noise_seed a u64")


@dataclass(frozen=True, slots=True)
class TcpCrossPlan:
    cc: str
    start_ms: int
    stop_ms: int


@dataclass(frozen=True, slots=True)
class UdpCrossPlan:
    rate_bps: int
    start_ms: int
    stop_ms: int


type CrossPlan = TcpCrossPlan | UdpCrossPlan


@dataclass(frozen=True, slots=True)
class GuestPlan:
    duration_ms: int
    cc: str
    downlink: LinkPlan
    uplink: LinkPlan
    start_offset_ms: int = 10
    obs_mark: int = 7
    ring_mib: int = 64
    sysctls: tuple[tuple[str, str], ...] = ()
    agent: AgentPlan | None = None
    cross: tuple[CrossPlan, ...] = ()
    # Record Sage's steps (steps.bin); always on under a policy.
    steps: bool = False

    def __post_init__(self) -> None:
        if (self.cc in ("agent", "agent_reno")) != (self.agent is not None):
            raise ValueError("cc agent/agent_reno and an agent plan go together")

    def write(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        links: dict[str, dict[str, int | str]] = {}
        for name, link in (("downlink", self.downlink), ("uplink", self.uplink)):
            trace_file = f"{name}.trace"
            (directory / trace_file).write_bytes(link.trace.tobytes())
            links[name] = {
                "delay_ns": link.delay_ns,
                "trace_file": trace_file,
                "limit_pkts": link.limit_pkts,
                "limit_bytes": link.limit_bytes,
                "loss": link.loss,
            }
        plan: dict[str, object] = {
            "plan_version": PLAN_VERSION,
            "duration_ms": self.duration_ms,
            "start_offset_ms": self.start_offset_ms,
            "cc": self.cc,
            "obs_mark": self.obs_mark,
            "ring_mib": self.ring_mib,
            **links,
            "sysctls": [list(s) for s in self.sysctls],
            "agent": None,
            "cross": [_cross_json(c) for c in self.cross],
            "steps": self.steps,
        }
        if self.agent is not None:
            (directory / "policy.onnx").write_bytes(self.agent.policy_onnx)
            plan["agent"] = {
                "policy_file": "policy.onnx",
                "noise_sigma": self.agent.noise_sigma,
                "noise_seed": self.agent.noise_seed,
            }
        (directory / "plan.json").write_text(json.dumps(plan, indent=2))


def _cross_json(c: CrossPlan) -> dict[str, int | str]:
    match c:
        case TcpCrossPlan(cc, start, stop):
            return {"kind": "tcp", "cc": cc, "start_ms": start, "stop_ms": stop}
        case UdpCrossPlan(rate, start, stop):
            return {"kind": "udp_cbr", "rate_bps": rate, "start_ms": start, "stop_ms": stop}
