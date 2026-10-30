"""Declarative episode and campaign specs (the user-facing input format).

``EpisodeSpec.parse`` and ``CampaignSpec.parse`` are the trust boundary for
spec files: everything past them is a validated, frozen value.

Delay, prominently: Mahak's "mRTT" is passed straight to ``mm-delay``,
which delays each direction by it, so Mahak's base RTT is TWICE its mRTT.
``{"mahak_compat_mrtt_ms": 20}`` reproduces Mahak (20 ms each way, 40 ms
base RTT); ``{"base_rtt_ms": 40}`` says the same thing explicitly. A delay
is exactly one of the two; giving both is a parse error.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Annotated, Final, Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

from expliccit_env import mahimahi as mm
from expliccit_env.guest_plan import AgentPlan, GuestPlan, LinkPlan

SPEC_VERSION: Final = 1
U64: Final = 2**64


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class MahakMrtt(_Frozen):
    """Mahak semantics: this many ms of delay in EACH direction."""

    mahak_compat_mrtt_ms: PositiveInt

    @property
    def one_way_ns(self) -> int:
        return self.mahak_compat_mrtt_ms * 1_000_000


class BaseRtt(_Frozen):
    """Explicit base (propagation) RTT, split evenly across the directions."""

    base_rtt_ms: PositiveInt

    @property
    def one_way_ns(self) -> int:
        return self.base_rtt_ms * 500_000


type Delay = MahakMrtt | BaseRtt


class MahakStep(_Frozen):
    """Mahak's two-level trace: bw_mbps for 12 s, then step * bw_mbps."""

    kind: Literal["mahak_step"] = "mahak_step"
    bw_mbps: PositiveInt
    step: Annotated[float, Field(gt=0)]

    def timestamps(self) -> mm.I64:
        return mm.mahak_trace(self.bw_mbps, self.step)


class TraceFile(_Frozen):
    """A Mahimahi trace file, pinned by content hash for provenance."""

    kind: Literal["mahimahi_file"] = "mahimahi_file"
    path: Path
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

    def timestamps(self) -> mm.I64:
        raw = self.path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.sha256:
            raise ValueError(f"{self.path}: sha256 does not match the spec")
        return mm.parse_trace(raw.decode("ascii"))


type Trace = Annotated[MahakStep | TraceFile, Field(discriminator="kind")]


class DropTail(_Frozen):
    """Mahimahi droptail limits; 0 means no limit, at least one is set."""

    packets: Annotated[int, Field(ge=0)] = 0
    bytes: Annotated[int, Field(ge=0)] = 0

    @model_validator(mode="after")
    def _some_limit(self) -> Self:
        if self.packets == 0 and self.bytes == 0:
            raise ValueError("droptail needs a packet or byte limit")
        return self


class PathSpec(_Frozen):
    trace: Trace
    # None: the uplink uses the downlink's trace, as Mahak does.
    uplink_trace: Trace | None = None
    delay: Delay
    buffer: DropTail


class StockCC(_Frozen):
    """An unmodified in-kernel congestion control."""

    kind: Literal["stock"] = "stock"
    name: Literal["cubic", "bbr", "reno"]


class OnnxPolicy(_Frozen):
    """An exported policy file, pinned by content hash (contract: policy.py)."""

    path: Path
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

    def read(self) -> bytes:
        raw = self.path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.sha256:
            raise ValueError(f"{self.path}: sha256 does not match the spec")
        return raw


class AgentCC(_Frozen):
    """A policy driving tcp_agent. "hold" is Sage-style (the policy sets cwnd
    and pacing outright); "reno" is Orca-style (the policy sets cwnd, Reno
    evolves it until the next action; Orca proper uses Cubic as the base)."""

    kind: Literal["agent"] = "agent"
    mode: Literal["hold", "reno"] = "hold"
    policy: OnnxPolicy
    interval_ms: PositiveInt = 20

    @property
    def name(self) -> str:
        return "agent" if self.mode == "hold" else "agent_reno"


type CongestionControl = Annotated[StockCC | AgentCC, Field(discriminator="kind")]


class EpisodeSpec(_Frozen):
    spec_version: Literal[1] = 1
    path: PathSpec
    cc: CongestionControl
    duration_ms: PositiveInt = 20_000
    seed: Annotated[int, Field(ge=0, lt=U64)] = 0
    sysctls: dict[str, str] = Field(default_factory=dict[str, str])

    @classmethod
    def parse(cls, text: str) -> EpisodeSpec:
        return cls.model_validate_json(text)

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    def episode_id(self) -> str:
        """Content hash of the spec: same spec, same id (and, on the seeded
        kernel, the same bytes)."""
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()[:20]

    def guest_plan(self) -> GuestPlan:
        down_ts = self.path.trace.timestamps()
        up_ts = (self.path.uplink_trace or self.path.trace).timestamps()
        delay = self.path.delay.one_way_ns
        buf = self.path.buffer

        def link(ts: mm.I64) -> LinkPlan:
            return LinkPlan(delay, mm.dense_counts(ts), buf.packets, buf.bytes)

        agent = (
            AgentPlan(self.cc.policy.read(), self.cc.interval_ms)
            if isinstance(self.cc, AgentCC)
            else None
        )
        return GuestPlan(
            duration_ms=self.duration_ms,
            cc=self.cc.name,
            downlink=link(down_ts),
            uplink=link(up_ts),
            sysctls=tuple(sorted(self.sysctls.items())),
            agent=agent,
        )


# --- campaigns ----------------------------------------------------------------


class MahakGrid(_Frozen):
    """Cartesian product over Mahak's four parameters."""

    kind: Literal["mahak_grid"] = "mahak_grid"
    bw_mbps: list[PositiveInt]
    step: list[Annotated[float, Field(gt=0)]]
    mahak_compat_mrtt_ms: list[PositiveInt]
    buffer_pkts: list[PositiveInt]

    def points(self, rng: np.random.Generator) -> list[tuple[int, float, int, int]]:
        del rng
        return list(
            itertools.product(self.bw_mbps, self.step, self.mahak_compat_mrtt_ms, self.buffer_pkts)
        )


class Range(_Frozen):
    lo: Annotated[float, Field(gt=0)]
    hi: Annotated[float, Field(gt=0)]
    log: bool = True

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.lo > self.hi:
            raise ValueError("range needs lo <= hi")
        return self

    def sample(self, rng: np.random.Generator, n: int) -> list[float]:
        if self.log:
            return np.exp(rng.uniform(np.log(self.lo), np.log(self.hi), n)).tolist()
        return rng.uniform(self.lo, self.hi, n).tolist()


class MahakSample(_Frozen):
    """n points drawn (log-)uniformly from Mahak's search space; bandwidth,
    mRTT and buffer are rounded to integers as Mahak's oracle_query does."""

    kind: Literal["mahak_sample"] = "mahak_sample"
    n: PositiveInt
    bw_mbps: Range
    step: Range
    mahak_compat_mrtt_ms: Range
    buffer_pkts: Range

    def points(self, rng: np.random.Generator) -> list[tuple[int, float, int, int]]:
        bw, st, mrtt, buf = (
            r.sample(rng, self.n)
            for r in (self.bw_mbps, self.step, self.mahak_compat_mrtt_ms, self.buffer_pkts)
        )
        return [
            (max(1, round(b)), s, max(1, round(m)), max(1, round(q)))
            for b, s, m, q in zip(bw, st, mrtt, buf, strict=True)
        ]


type Sweep = Annotated[MahakGrid | MahakSample, Field(discriminator="kind")]


class CampaignSpec(_Frozen):
    spec_version: Literal[1] = 1
    name: Annotated[str, Field(pattern=r"^[A-Za-z0-9_.-]{1,64}$")]
    ccs: list[CongestionControl]
    sweep: Sweep
    replicates: PositiveInt = 1
    duration_ms: PositiveInt = 20_000
    seed: Annotated[int, Field(ge=0, lt=U64)] = 0
    sysctls: dict[str, str] = Field(default_factory=dict[str, str])

    @classmethod
    def parse(cls, text: str) -> CampaignSpec:
        return cls.model_validate_json(text)

    def episodes(self) -> list[EpisodeSpec]:
        """Deterministic expansion: points x ccs x replicates, each with its
        own seed drawn from the campaign seed."""
        rng = np.random.default_rng(self.seed)
        points = self.sweep.points(rng)
        out: list[EpisodeSpec] = []
        for (bw, step, mrtt, buf), cc, _ in itertools.product(
            points, self.ccs, range(self.replicates)
        ):
            out.append(
                EpisodeSpec(
                    path=PathSpec(
                        trace=MahakStep(bw_mbps=bw, step=step),
                        delay=MahakMrtt(mahak_compat_mrtt_ms=mrtt),
                        buffer=DropTail(packets=buf),
                    ),
                    cc=cc,
                    duration_ms=self.duration_ms,
                    seed=int(rng.integers(0, U64, dtype=np.uint64)),
                    sysctls=self.sysctls,
                )
            )
        return out
