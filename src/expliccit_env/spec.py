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
import math
from pathlib import Path
from typing import Annotated, Final, Literal, Self, get_args

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from expliccit_env import mahimahi as mm
from expliccit_env.guest_plan import (
    AgentPlan,
    CrossPlan,
    GuestPlan,
    LinkPlan,
    TcpCrossPlan,
    UdpCrossPlan,
)

SPEC_VERSION: Final = 1
# Sender buffer limit (EpisodeSpec._send_buffer). Linux grows a socket's send
# buffer to k * cwnd * per_mss (tcp_sndbuf_expand): k is 2, or 3 for BBR, and
# per_mss the memory one segment may take, roundup_pow_of_two(MSS 1460 +
# MAX_TCP_HEADER 256 + skb_shared_info 320) + sk_buff 256 = 2304 bytes in this
# build. The limit grants that for a window of SEND_BUFFER_WINDOWS times what
# the path holds: stock controllers reached at most 2.15 times (slow start's
# overshoot, BBR's two BDPs), so 4 leaves them twice that (docs/validation.md).
# ponytail: per_mss read from the UML build's headers; recompute it if the
# kernel config or MTU changes.
SEND_BUFFER_PER_SEGMENT: Final = 3 * 2304
SEND_BUFFER_WINDOWS: Final = 4
SEND_BUFFER_MAX: Final = 64 * 2**20  # Sage's and ccBench's setup scripts
TCP_INIT_CWND: Final = 10
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


class PeriodicLevels(_Frozen):
    """Capacity cycling through levels, each held ``level_ms`` (for example
    expliCCit's case-study link, 24 and 48 Mbit/s alternating every 7 s)."""

    kind: Literal["periodic_levels"] = "periodic_levels"
    levels_mbps: Annotated[list[Annotated[float, Field(gt=0, le=3000)]], Field(min_length=1)]
    level_ms: Annotated[int, Field(gt=0, le=65_000)]

    def timestamps(self) -> mm.I64:
        return mm.periodic_trace(self.levels_mbps, self.level_ms)


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


type Trace = Annotated[MahakStep | PeriodicLevels | TraceFile, Field(discriminator="kind")]


class DropTail(_Frozen):
    """Mahimahi droptail limits; 0 means no limit, at least one is set."""

    packets: Annotated[int, Field(ge=0)] = 0
    bytes: Annotated[int, Field(ge=0)] = 0

    @model_validator(mode="after")
    def _some_limit(self) -> Self:
        if self.packets == 0 and self.bytes == 0:
            raise ValueError("droptail needs a packet or byte limit")
        return self


type Probability = Annotated[float, Field(ge=0.0, lt=1.0)]


class Loss(_Frozen):
    """i.i.d. random loss per direction (Mahimahi's mm-loss), applied before
    the delay line and the link."""

    downlink: Probability = 0.0
    uplink: Probability = 0.0


class PathSpec(_Frozen):
    trace: Trace
    # None: the uplink uses the downlink's trace, as Mahak does.
    uplink_trace: Trace | None = None
    delay: Delay
    buffer: DropTail
    loss: Loss = Loss()


type StockName = Literal[
    "cubic", "bbr", "bbr3", "reno",
    # Further teachers, built as modules (kernel/uml.config).
    "bic", "cdg", "highspeed", "htcp", "hybla", "illinois", "lp", "nv", "scalable",
    "vegas", "veno", "westwood", "yeah",
]  # fmt: skip
STOCK_NAMES: Final = get_args(StockName.__value__)


class StockCC(_Frozen):
    """An in-kernel congestion control: Cubic, BBRv1 (``bbr``), Reno and the
    other controllers Linux 7.0 ships (built as modules here), or Google's
    BBRv3 (``bbr3``: it needs the core patches imported under
    third_party/linux)."""

    kind: Literal["stock"] = "stock"
    name: StockName


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
    """A sage-v1 policy (policy.py) driving tcp_agent after every step.
    "hold" is Sage-style (the window and pacing stay as the policy set
    them); "pure" is Sage's own controller on Linux 7.0's
    stack (the same window, paced by the module at every delivery, as
    Sage's "pure" with DeepCC pacing does); "rate" is Aurora-style (each
    action multiplies a sending rate the kernel paces at, and each step
    lasts half the last step's mean RTT); "cubic" is Orca-style (each action
    multiplies the kernel's current window, and Cubic evolves it until the
    next action)."""

    kind: Literal["agent"] = "agent"
    mode: Literal["hold", "pure", "rate", "cubic"] = "hold"
    policy: OnnxPolicy
    # Exploration (online training): Gaussian noise on each action, seeded
    # from the episode's seed. 0 runs the policy as it is.
    noise_sigma: Annotated[float, Field(ge=0, le=2)] = 0.0
    # Keep each noise draw for this many steps: with a policy that sets an
    # absolute window target, held noise keeps the window at the perturbed
    # level long enough for its effect to show.
    noise_hold_steps: Annotated[int, Field(ge=1, le=1000)] = 1
    # The noise stream's own seed, so several rollouts of one episode (the
    # same seed, so the same path and loss draws) explore differently; unset,
    # it derives from the episode's seed. Left out of the canonical form when
    # unset, so specs without it keep their ids.
    noise_seed: Annotated[int, Field(ge=0, lt=U64)] | None = None
    # The step period: Sage's 10 ms, or the policy's own (Orca's 20 ms). Left
    # out of the canonical form at 10, so specs without it keep their ids.
    period_ms: Annotated[int, Field(ge=10, le=1000)] = 10

    @model_serializer(mode="wrap")
    def _omit_unset_seed(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        out: dict[str, object] = handler(self)
        if self.noise_seed is None:
            out.pop("noise_seed", None)
        if self.period_ms == 10:
            out.pop("period_ms", None)
        return out

    @property
    def name(self) -> str:
        names = {
            "hold": "agent",
            "pure": "agent_pure",
            "rate": "agent_rate",
            "cubic": "agent_cubic",
        }
        return names[self.mode]

    @property
    def label(self) -> str:
        """The controller in tables: its mode, the policy's hash, and its
        period when not Sage's."""
        period = "" if self.period_ms == 10 else f"@{self.period_ms}ms"
        return f"{self.name}:{self.policy.sha256[:12]}{period}"


type CongestionControl = Annotated[StockCC | AgentCC, Field(discriminator="kind")]


class TcpCross(_Frozen):
    """Bulk TCP through the same bottleneck, active over [start_ms, stop_ms)
    of trace time (stop: episode end), with its own congestion control: a
    stock one by name, or an agent driven by its own policy (any number of
    flows, each with its own policy, state and exploration noise).

    ``extra_rtt_ms`` lengthens this flow's base RTT beyond the path's: its
    ACKs wait that long on the receiver's side before entering the path
    (sch_flowdelay), as on an access link of its own. Left out of the
    canonical form when 0, so specs without it keep their ids."""

    kind: Literal["tcp"] = "tcp"
    cc: StockName | AgentCC
    start_ms: Annotated[int, Field(ge=0)] = 0
    stop_ms: PositiveInt | None = None
    extra_rtt_ms: Annotated[int, Field(ge=0, le=10_000)] = 0

    @model_serializer(mode="wrap")
    def _omit_zero_extra(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        out: dict[str, object] = handler(self)
        if not self.extra_rtt_ms:
            out.pop("extra_rtt_ms", None)
        return out

    @property
    def cc_name(self) -> str:
        """The kernel's name for the flow's congestion control."""
        return self.cc if isinstance(self.cc, str) else self.cc.name

    @property
    def label(self) -> str:
        return self.cc if isinstance(self.cc, str) else self.cc.label


class UdpCross(_Frozen):
    """Constant bit rate UDP (1500-byte IP packets), rate counted at the IP
    layer, active over [start_ms, stop_ms) of trace time (stop: episode end)."""

    kind: Literal["udp_cbr"] = "udp_cbr"
    rate_mbps: Annotated[float, Field(gt=0, le=10_000)]
    start_ms: Annotated[int, Field(ge=0)] = 0
    stop_ms: PositiveInt | None = None


type CrossFlow = Annotated[TcpCross | UdpCross, Field(discriminator="kind")]


class UdpShare(_Frozen):
    """Constant bit rate UDP at a share of the path's mean capacity, for a
    campaign condition: each sweep point turns it into a UdpCross, so one
    condition loads slow and fast links alike."""

    kind: Literal["udp_share"] = "udp_share"
    share: Annotated[float, Field(gt=0, lt=1)]
    start_ms: Annotated[int, Field(ge=0)] = 0
    stop_ms: PositiveInt | None = None

    def on(self, trace: Trace) -> UdpCross:
        ts = trace.timestamps()  # one 1500-byte packet per entry, in ms
        mean_mbps = 12.0 * ts.size / float(ts[-1])
        rate = round(self.share * mean_mbps, 3)
        return UdpCross(rate_mbps=rate, start_ms=self.start_ms, stop_ms=self.stop_ms)


type ConditionFlow = Annotated[TcpCross | UdpCross | UdpShare, Field(discriminator="kind")]
# guest/src/plan.rs MAX_CROSS: every flow runs sender to receiver.
MAX_CROSS: Final = 64


class EpisodeSpec(_Frozen):
    spec_version: Literal[1] = 1
    path: PathSpec
    cc: CongestionControl
    duration_ms: PositiveInt = 20_000
    seed: Annotated[int, Field(ge=0, lt=U64)] = 0
    sysctls: dict[str, str] = Field(default_factory=dict[str, str])
    cross: Annotated[list[CrossFlow], Field(max_length=MAX_CROSS)] = Field(
        default_factory=list[CrossFlow]
    )
    # Record Sage's steps (training data); always on under a policy.
    steps: bool = False

    @model_validator(mode="after")
    def _cross_windows(self) -> Self:
        for c in self.cross:
            stop = self.duration_ms if c.stop_ms is None else c.stop_ms
            if not c.start_ms < stop <= self.duration_ms:
                raise ValueError(f"cross flow window [{c.start_ms}, {stop}) not inside the episode")
        return self

    @classmethod
    def parse(cls, text: str) -> EpisodeSpec:
        return cls.model_validate_json(text)

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    def episode_id(self) -> str:
        """Content hash of the spec: same spec, same id (and, on the seeded
        kernel, the same bytes)."""
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()[:20]

    def _send_buffer(self) -> str:
        """tcp_wmem for the senders: what Linux's own autotuning asks for a
        window of SEND_BUFFER_WINDOWS times what the path holds (its peak
        capacity over 1 s times the longest flow's base RTT, plus the buffer:
        the bandwidth-delay product at the RTT of a full queue), at most the
        setups' 64 MB. Stock controllers never reach it; a window far beyond
        the path, which a real sender's NIC and CPU would stop and which costs
        nothing in virtual time, does (docs/validation.md, "TCP buffer
        limits")."""
        counts = mm.dense_counts(self.path.trace.timestamps()).astype(np.int64)
        cycle = np.tile(counts[1:], -(-2000 // max(counts.size - 1, 1)))
        peak = int(np.convolve(cycle, np.ones(1000, np.int64), "valid").max())  # packets/s
        extra = max((c.extra_rtt_ms for c in self.cross if isinstance(c, TcpCross)), default=0)
        rtt_s = (2 * self.path.delay.one_way_ns + extra * 1_000_000) / 1e9
        buf = self.path.buffer
        queue = min(n for n in (buf.packets, -(-buf.bytes // mm.PACKET_SIZE)) if n)
        segments = max(TCP_INIT_CWND, SEND_BUFFER_WINDOWS * (math.ceil(peak * rtt_s) + queue))
        cap = min(SEND_BUFFER_MAX, SEND_BUFFER_PER_SEGMENT * segments)
        return f"4096 65536 {cap}"

    def guest_plan(self) -> GuestPlan:
        down_ts = self.path.trace.timestamps()
        up_ts = (self.path.uplink_trace or self.path.trace).timestamps()
        delay = self.path.delay.one_way_ns
        buf = self.path.buffer

        def link(ts: mm.I64, loss: float) -> LinkPlan:
            return LinkPlan(
                delay, mm.dense_counts(ts), buf.packets, buf.bytes, u32_probability(loss)
            )

        # Each agent's noise stream: the observed flow's from the episode
        # seed, cross flow i's mixed with i so that no two flows share one.
        noise_seed = self.seed ^ 0x5EED_0F_A6E47

        def agent_plan(cc: AgentCC, seed: int) -> AgentPlan:
            own = seed if cc.noise_seed is None else cc.noise_seed
            return AgentPlan(
                cc.policy.read(), cc.noise_sigma, own, cc.noise_hold_steps, cc.period_ms
            )

        def cross(i: int, c: TcpCross | UdpCross) -> CrossPlan:
            stop = self.duration_ms if c.stop_ms is None else c.stop_ms
            match c:
                case TcpCross(cc=AgentCC() as a):
                    seed = noise_seed ^ ((i + 1) * 0x9E37_79B9_7F4A_7C15 % U64)
                    plan = agent_plan(a, seed)
                    return TcpCrossPlan(c.cc_name, c.start_ms, stop, plan, c.extra_rtt_ms * 10**6)
                case TcpCross():
                    return TcpCrossPlan(c.cc_name, c.start_ms, stop, None, c.extra_rtt_ms * 10**6)
                case UdpCross():
                    return UdpCrossPlan(round(c.rate_mbps * 1e6), c.start_ms, stop)

        agent = agent_plan(self.cc, noise_seed) if isinstance(self.cc, AgentCC) else None
        return GuestPlan(
            duration_ms=self.duration_ms,
            cc=self.cc.name,
            downlink=link(down_ts, self.path.loss.downlink),
            uplink=link(up_ts, self.path.loss.uplink),
            sysctls=tuple(
                sorted(({"net.ipv4.tcp_wmem": self._send_buffer()} | self.sysctls).items())
            ),
            agent=agent,
            cross=tuple(cross(i, c) for i, c in enumerate(self.cross)),
            steps=self.steps or agent is not None,
        )


def u32_probability(p: float) -> int:
    """sch_mmlink's loss encoding: probability times 2^32, below 2^32."""
    return min(round(p * 2**32), 2**32 - 1)


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


class PathList(_Frozen):
    """An explicit list of paths (any trace kind, delay, buffer, uplink).
    A condition's loss replaces each path's own."""

    kind: Literal["paths"] = "paths"
    paths: Annotated[list[PathSpec], Field(min_length=1)]


def paths_of(sweep: MahakGrid | MahakSample | PathList, rng: np.random.Generator) -> list[PathSpec]:
    if isinstance(sweep, PathList):
        return list(sweep.paths)
    return [
        PathSpec(
            trace=MahakStep(bw_mbps=bw, step=step),
            delay=MahakMrtt(mahak_compat_mrtt_ms=mrtt),
            buffer=DropTail(packets=buf),
        )
        for bw, step, mrtt, buf in sweep.points(rng)
    ]


type Sweep = Annotated[MahakGrid | MahakSample | PathList, Field(discriminator="kind")]


class Condition(_Frozen):
    """Random loss and cross traffic for one arm of a campaign; every sweep
    point runs under every condition."""

    loss: Loss = Loss()
    cross: Annotated[list[ConditionFlow], Field(max_length=MAX_CROSS)] = Field(
        default_factory=list[ConditionFlow]
    )

    def flows(self, trace: Trace) -> list[CrossFlow]:
        return [c.on(trace) if isinstance(c, UdpShare) else c for c in self.cross]


class CampaignSpec(_Frozen):
    spec_version: Literal[1] = 1
    name: Annotated[str, Field(pattern=r"^[A-Za-z0-9_.-]{1,64}$")]
    ccs: list[CongestionControl]
    sweep: Sweep
    conditions: Annotated[list[Condition], Field(min_length=1)] = Field(
        default_factory=lambda: [Condition()]
    )
    replicates: PositiveInt = 1
    # Run only this many of the ccs on each sweep point (the same ones under
    # every condition), drawn with the campaign's seed: breadth over paths
    # instead of every controller on every path, for estimator data.
    ccs_per_point: PositiveInt | None = None
    # Likewise, run only this many of the conditions on each sweep point:
    # many conditions over many links without their full product.
    conditions_per_point: PositiveInt | None = None
    steps: bool = False
    duration_ms: PositiveInt = 20_000
    seed: Annotated[int, Field(ge=0, lt=U64)] = 0
    sysctls: dict[str, str] = Field(default_factory=dict[str, str])

    @model_validator(mode="after")
    def _subset(self) -> CampaignSpec:
        if self.ccs_per_point is not None and self.ccs_per_point > len(self.ccs):
            raise ValueError(f"ccs_per_point {self.ccs_per_point} exceeds the {len(self.ccs)} ccs")
        n = len(self.conditions)
        if self.conditions_per_point is not None and self.conditions_per_point > n:
            raise ValueError(f"conditions_per_point {self.conditions_per_point} exceeds {n}")
        return self

    @classmethod
    def parse(cls, text: str) -> CampaignSpec:
        return cls.model_validate_json(text)

    def episodes(self) -> list[EpisodeSpec]:
        """Deterministic expansion: points x conditions x ccs (or each
        point's subsets of them) x replicates, each with its own seed drawn
        from the campaign seed."""
        rng = np.random.default_rng(self.seed)
        paths = paths_of(self.sweep, rng)
        out: list[EpisodeSpec] = []
        for path in paths:
            ccs = self.ccs
            if self.ccs_per_point is not None:
                pick = rng.choice(len(self.ccs), self.ccs_per_point, replace=False)
                ccs = [self.ccs[int(i)] for i in np.sort(pick)]
            conds = self.conditions
            if self.conditions_per_point is not None:
                pick = rng.choice(len(conds), self.conditions_per_point, replace=False)
                conds = [conds[int(i)] for i in np.sort(pick)]
            for cond, cc, _ in itertools.product(conds, ccs, range(self.replicates)):
                out.append(
                    EpisodeSpec(
                        path=path.model_copy(update={"loss": cond.loss}),
                        cc=cc,
                        duration_ms=self.duration_ms,
                        seed=int(rng.integers(0, U64, dtype=np.uint64)),
                        sysctls=self.sysctls,
                        cross=cond.flows(path.trace),
                        steps=self.steps,
                    )
                )
        return out
