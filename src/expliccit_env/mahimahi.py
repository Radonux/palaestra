"""Mahimahi and Mahak semantics, ported exactly.

Sources (pinned):
  - Mahak helper.py ``trace_generator``, github.com/Ppazhooh/mahak @ 5077583.
  - Mahimahi ``src/frontend/link_queue.cc`` and ``drop_tail_packet_queue.hh``
    (third_party/mahimahi), whose semantics sch_mmlink implements in the
    kernel; ``LinkOracle`` is an independent Python port used to check the
    kernel's records event by event.
  - Mahak ``mm-th-delay`` (third_party/mahimahi/scripts/mm-th-delay), ported
    as ``mm_th_delay`` and tested against the Perl script.

Units: Mahimahi time is integer milliseconds; sizes are Mahimahi bytes (the
IPv4 total length plus the 4-byte TUN header); a delivery opportunity is
1504 bytes.
"""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Final, Literal

import numpy as np
import numpy.typing as npt

from expliccit_env.records import DROP_QUEUE_FULL, Records

PACKET_SIZE: Final = 1504
LEVEL_MS: Final = 12000

type I64 = npt.NDArray[np.int64]


def mahak_trace(bw_mbps: int, step: float) -> I64:
    """Mahak's two-level step trace: ``bw`` Mbit/s for 12 s, then ``step * bw``.

    Port of ``trace_generator(mm_bw, mm_step, mm_step * mm_bw)`` as Mahak's
    ``oracle_query`` calls it (bandwidth already cast to int there). Keeps
    every float operation in Mahak's order, including numpy's linspace
    rounding and the truncation of ``'%d' % ts``, so the timestamps are
    bit-identical to the trace files Mahak wrote.
    """
    end_bw = step * bw_mbps
    ts1 = np.linspace(0, LEVEL_MS, num=int(float(bw_mbps) * 1000), endpoint=False)
    ts2 = np.linspace(0, LEVEL_MS, num=int(float(end_bw) * 1000), endpoint=False) + LEVEL_MS
    return np.concatenate((ts1, ts2)).astype(np.int64)


def parse_trace(text: str) -> I64:
    """A Mahimahi trace file: one nondecreasing ms timestamp per line, last > 0."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if not lines:
        raise ValueError("trace: no valid timestamps found")
    if not all(line.isdigit() and line.isascii() for line in lines):
        raise ValueError("trace: every line must be a nonnegative integer")
    return validate_trace(np.array([int(x) for x in lines], dtype=np.int64))


def validate_trace(ts: I64) -> I64:
    if ts.size == 0:
        raise ValueError("trace: no valid timestamps found")
    if np.any(np.diff(ts) < 0) or ts[0] < 0:
        raise ValueError("trace: timestamps must be monotonically nondecreasing and nonnegative")
    if ts[-1] == 0:
        raise ValueError("trace: trace must last for a nonzero amount of time")
    return ts


def dense_counts(ts: I64) -> npt.NDArray[np.uint8]:
    """Encode a trace as sch_mmlink's TCA_MMLINK_TRACE: opportunities per ms.

    Entry i counts the timestamps equal to i; the array ends at the last
    timestamp, whose index is the repeat period, exactly like Mahimahi's
    ``base_timestamp_ += schedule_.back()``.
    """
    counts = np.bincount(validate_trace(ts), minlength=int(ts[-1]) + 1)
    if counts.max() > 255:
        raise ValueError(
            f"trace has {counts.max()} opportunities in one ms; the kernel ABI allows 255"
        )
    return counts.astype(np.uint8)


def opportunity_times(ts: I64, end_ms: int) -> I64:
    """Every delivery opportunity at ms in [0, end_ms), the trace repeating
    with period ts[-1] (one entry per opportunity, so ms may repeat)."""
    period = int(ts[-1])
    times = np.concatenate([ts + k * period for k in range(end_ms // period + 2)])
    return times[times < end_ms]


# --- LinkQueue oracle -------------------------------------------------------


@dataclass(slots=True)
class OracleEvents:
    """What a LinkQueue logs, in emission order, plus per-ms opportunity use.

    arrivals: (ms, bytes, queued packets, queued bytes) as found on arrival
    drops: (ms, bytes)
    departures: (ms, arrival_ms, bytes, queued packets, queued bytes) after
    opportunities: ms -> [count, bytes used]
    """

    arrivals: list[tuple[int, int, int, int]] = field(
        default_factory=list[tuple[int, int, int, int]]
    )
    drops: list[tuple[int, int]] = field(default_factory=list[tuple[int, int]])
    departures: list[tuple[int, int, int, int, int]] = field(
        default_factory=list[tuple[int, int, int, int, int]]
    )
    opportunities: dict[int, list[int]] = field(default_factory=dict[int, list[int]])


class LinkOracle:
    """Line-by-line Python port of Mahimahi's LinkQueue with DropTail.

    A test oracle: it runs a Python loop per packet and per opportunity, so it
    is for checking short episodes, never for producing datasets.
    """

    def __init__(self, schedule: Sequence[int], limit_pkts: int, limit_bytes: int) -> None:
        if not schedule or schedule[-1] == 0:
            raise ValueError("schedule must be nonempty and last a nonzero time")
        if limit_pkts == 0 and limit_bytes == 0:
            raise ValueError("Dropping queue must have a byte or packet limit.")
        self.schedule = list(schedule)
        self.limit_pkts = limit_pkts
        self.limit_bytes = limit_bytes
        self.next_delivery = 0
        self.base = 0
        self.queue: deque[tuple[int, int]] = deque()  # (arrival ms, bytes)
        self.queue_bytes = 0
        self.transit: tuple[int, int] | None = None
        self.transit_left = 0
        self.events = OracleEvents()

    def _next_delivery_time(self) -> int:
        return self.schedule[self.next_delivery] + self.base

    def _good_with(self, size_bytes: int, size_packets: int) -> bool:
        ok = True
        if self.limit_bytes:
            ok &= size_bytes <= self.limit_bytes
        if self.limit_pkts:
            ok &= size_packets <= self.limit_pkts
        return ok

    def rationalize(self, now: int) -> None:
        while self._next_delivery_time() <= now:
            t = self._next_delivery_time()
            left = PACKET_SIZE
            # use_a_delivery_opportunity(): log it, then advance with wraparound.
            self.next_delivery = (self.next_delivery + 1) % len(self.schedule)
            if self.next_delivery == 0:
                self.base += self.schedule[-1]
            while left > 0:
                if not self.transit_left:
                    if not self.queue:
                        break
                    self.transit = self.queue.popleft()
                    self.transit_left = self.transit[1]
                    self.queue_bytes -= self.transit[1]
                amount = min(left, self.transit_left)
                self.transit_left -= amount
                left -= amount
                if self.transit_left == 0 and self.transit is not None:
                    arrival, size = self.transit
                    self.events.departures.append(
                        (t, arrival, size, len(self.queue), self.queue_bytes)
                    )
            opp = self.events.opportunities.setdefault(t, [0, 0])
            opp[0] += 1
            opp[1] += PACKET_SIZE - left

    def read_packet(self, now: int, size: int) -> None:
        if size > PACKET_SIZE:
            raise ValueError("packet size is greater than maximum")
        self.rationalize(now)
        self.events.arrivals.append((now, size, len(self.queue), self.queue_bytes))
        if self._good_with(self.queue_bytes + size, len(self.queue) + 1):
            self.queue.append((now, size))
            self.queue_bytes += size
        else:
            self.events.drops.append((now, size))


def replay(
    schedule: Sequence[int],
    arrivals: Iterable[tuple[int, int]],
    limit_pkts: int,
    limit_bytes: int,
    until_ms: int,
) -> OracleEvents:
    """Feed (ms, bytes) arrivals through a LinkOracle, then run it to until_ms."""
    link = LinkOracle(schedule, limit_pkts, limit_bytes)
    for ms, size in arrivals:
        link.read_packet(ms, size)
    link.rationalize(until_ms)
    return link.events


# --- mm-th-delay ------------------------------------------------------------


class Kind(IntEnum):
    ARRIVAL = 0  # "+"
    OPPORTUNITY = 1  # "#"
    DEPARTURE = 2  # "-"
    DROP = 3  # "d"


_KIND_OF: Final = {"+": Kind.ARRIVAL, "#": Kind.OPPORTUNITY, "-": Kind.DEPARTURE, "d": Kind.DROP}
_SYMBOL: Final = {k: s for s, k in _KIND_OF.items()}


@dataclass(frozen=True, slots=True)
class MmLog:
    """A Mahimahi link log, events in file order.

    ``nbytes`` is the third field as Perl reads it (packets, for drops) and
    ``delay`` the fourth (-1 where absent).
    """

    base_timestamp: int
    ts: I64
    kind: npt.NDArray[np.uint8]
    nbytes: I64
    delay: I64

    def __post_init__(self) -> None:
        n = self.ts.size
        if not (self.kind.size == self.nbytes.size == self.delay.size == n):
            raise ValueError("MmLog columns differ in length")


_BASE_RE: Final = re.compile(r"^# base timestamp: (\d+)")


def _parse_event(parts: list[str], base: int) -> tuple[int, Kind, int, int]:
    """One event line, validated in mm-th-delay's order."""
    if len(parts) < 3:
        raise ValueError("Format: timestamp event_type num_bytes [delay]")
    t, sym, nb = parts[0], parts[1], parts[2]
    if not (t.isascii() and t.isdigit()):
        raise ValueError(f"Invalid timestamp: {t}")
    if not (nb.isascii() and nb.isdigit()):
        raise ValueError(f"Invalid byte count: {nb}")
    if sym not in _KIND_OF:
        raise ValueError(f"Unknown event type: {sym}")
    kind = _KIND_OF[sym]
    d = -1
    has_fourth = len(parts) >= 4 and parts[3].isascii() and parts[3].isdigit()
    if kind is Kind.DEPARTURE:
        if not has_fourth:
            raise ValueError("Departure format: timestamp - num_bytes delay")
        d = int(parts[3])
        if int(t) - base - d < 0:
            raise ValueError(f"Invalid timestamp and delay: ts={t}, delay={d}")
    elif kind is Kind.DROP and has_fourth:
        d = int(parts[3])
    return int(t), kind, int(nb), d


def parse_mm_log(text: str) -> MmLog:
    """Parse a log with mm-th-delay's validation. Stricter in one way: the
    delay field must be an integer, as Mahimahi always writes it.

    ponytail: Python loop per line (~1 us); fine for fixtures and Tier 2
    reference logs, vectorize if logs ever reach campaign scale.
    """
    base: int | None = None
    ts: list[int] = []
    kinds: list[int] = []
    nbytes: list[int] = []
    delays: list[int] = []
    for line in text.splitlines():
        if m := _BASE_RE.match(line):
            if base is not None:
                raise ValueError("base timestamp multiply defined")
            base = int(m.group(1))
            continue
        if line.startswith("#"):
            continue
        if base is None:
            raise ValueError("logfile is missing base timestamp")
        t, kind, nb, d = _parse_event(line.split(), base)
        ts.append(t)
        kinds.append(kind)
        nbytes.append(nb)
        delays.append(d)
    if base is None:
        raise ValueError("logfile is missing base timestamp")
    return MmLog(
        base,
        np.array(ts, np.int64),
        np.array(kinds, np.uint8),
        np.array(nbytes, np.int64),
        np.array(delays, np.int64),
    )


def format_mm_log(log: MmLog) -> str:
    """Render a log in Mahimahi's format (inverse of parse_mm_log)."""
    out = [f"# base timestamp: {log.base_timestamp}"]
    for t, k, nb, d in zip(
        log.ts.tolist(), log.kind.tolist(), log.nbytes.tolist(), log.delay.tolist(), strict=True
    ):
        sym = _SYMBOL[Kind(k)]
        out.append(
            f"{t} {sym} {nb} {d}"
            if k == Kind.DEPARTURE or (k == Kind.DROP and d >= 0)
            else f"{t} {sym} {nb}"
        )
    return "\n".join(out) + "\n"


@dataclass(frozen=True, slots=True)
class MmThDelay:
    """What mm-th-delay computes; the first two are what it prints."""

    utilization: float
    mean_delay_ms: float
    average_capacity_mbps: float
    average_ingress_mbps: float
    average_throughput_mbps: float
    duration_s: float

    def perl_output(self) -> str:
        """The script's STDERR: Perl stringifies NVs with %.15g."""
        return f"{self.utilization:.15g}\n{self.mean_delay_ms:.15g}"


@dataclass(frozen=True, slots=True)
class MmThDelayUndefined:
    """Inputs on which the Perl script dies instead of printing."""

    reason: str


def mm_th_delay(log: MmLog) -> MmThDelay | MmThDelayUndefined:
    """Port of mm-th-delay's statistics (the gnuplot output is not ported).

    Same operation order as the script so results match to the last bit:
    integer bit sums, float duration from the first line's and the maximum
    timestamp, averages as (sum / duration) / 1e6, utilization as the ratio
    of averages, mean delay as the integer delay sum over the count.
    """
    if log.ts.size == 0:
        return MmThDelayUndefined("Must have at least one event")
    rel = log.ts - log.base_timestamp
    duration = (int(rel.max()) - int(rel[0])) / 1000.0
    bits = log.nbytes * 8
    cap = int(bits[log.kind == Kind.OPPORTUNITY].sum())
    ing = int(bits[log.kind == Kind.ARRIVAL].sum())
    dep_mask = log.kind == Kind.DEPARTURE
    dep = int(bits[dep_mask].sum())
    if duration == 0:
        return MmThDelayUndefined("Illegal division by zero (duration)")
    avg_cap = (cap / duration) / 1000000.0
    avg_ing = (ing / duration) / 1000000.0
    avg_thr = (dep / duration) / 1000000.0
    delays = log.delay[dep_mask]
    if delays.size == 0:
        return MmThDelayUndefined("Must have at least one departure event")
    if avg_cap == 0:
        return MmThDelayUndefined("Illegal division by zero (capacity)")
    return MmThDelay(
        utilization=avg_thr / avg_cap,
        mean_delay_ms=int(delays.sum()) / delays.size,
        average_capacity_mbps=avg_cap,
        average_ingress_mbps=avg_ing,
        average_throughput_mbps=avg_thr,
        duration_s=duration,
    )


type MahakMetric = Literal["utilization", "first_power", "delay"]


def mahak_metric(result: MmThDelay, metric: MahakMetric, one_way_ms: float) -> float:
    """helper.py ``oracle_query``'s metric. Mahak divides by its "mRTT", which
    is the one-way delay; pass ``mahak_compat_mrtt_ms`` for Mahak specs."""
    mrtt = float(one_way_ms)
    match metric:
        case "utilization":
            return result.utilization
        case "delay":
            return result.mean_delay_ms / mrtt
        case "first_power":
            # numpy division, as in Mahak: 0 delay gives inf (or nan), not an error.
            with np.errstate(divide="ignore", invalid="ignore"):
                return float(
                    np.divide(
                        np.float64(result.utilization), np.float64(result.mean_delay_ms / mrtt)
                    )
                )


def mm_log_from_records(rec: Records, link_id: int, trace: I64, window_ms: int) -> MmLog:
    """The Mahimahi log one link would have written for trace ms [0, window_ms).

    Opportunities come from the trace (Mahimahi logs every one, even on an
    idle link; the kernel only reports the ones it processed). Arrivals,
    drops and departures come from the kernel's records. Lines are ordered by
    ms, opportunities first within a ms, which only affects the script's
    first timestamp and therefore neither printed metric.
    """
    opp = opportunity_times(trace, window_ms)
    arr = rec.link(rec.link_arrival, link_id)
    arr = arr[arr["ms"] < window_ms]
    drp = rec.link(rec.link_drop, link_id)
    drp = drp[(drp["ms"] < window_ms) & (drp["reason"] == DROP_QUEUE_FULL)]
    dep = rec.link(rec.link_depart, link_id)
    dep = dep[dep["ms"] < window_ms]
    ts = np.concatenate([opp, arr["ms"], drp["ms"], dep["ms"]]).astype(np.int64)
    kind = np.concatenate(
        [
            np.full(opp.size, Kind.OPPORTUNITY),
            np.full(arr.size, Kind.ARRIVAL),
            np.full(drp.size, Kind.DROP),
            np.full(dep.size, Kind.DEPARTURE),
        ]
    ).astype(np.uint8)
    nbytes = np.concatenate(
        [
            np.full(opp.size, PACKET_SIZE),
            arr["bytes"],
            np.ones(drp.size),
            dep["bytes"],
        ]
    ).astype(np.int64)
    delay = np.concatenate(
        [
            np.full(opp.size, -1),
            np.full(arr.size, -1),
            drp["bytes"],
            dep["ms"].astype(np.int64) - dep["arrival_ms"].astype(np.int64),
        ]
    ).astype(np.int64)
    # Within a ms, opportunities first: the secondary key is False for them.
    order = np.lexsort((kind != Kind.OPPORTUNITY, ts))
    return MmLog(0, ts[order], kind[order], nbytes[order], delay[order])
