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

PLAN_VERSION: Final = 1
MAX_TRACE_BYTES: Final = 65_000


@dataclass(frozen=True, slots=True)
class LinkPlan:
    """One direction: delay line plus trace-driven droptail link."""

    delay_ns: int
    trace: npt.NDArray[np.uint8]  # opportunities per ms (mahimahi.dense_counts)
    limit_pkts: int
    limit_bytes: int

    def __post_init__(self) -> None:
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
class GuestPlan:
    duration_ms: int
    cc: str
    downlink: LinkPlan
    uplink: LinkPlan
    start_offset_ms: int = 10
    obs_mark: int = 7
    ring_mib: int = 64
    sysctls: tuple[tuple[str, str], ...] = ()

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
        }
        (directory / "plan.json").write_text(json.dumps(plan, indent=2))
