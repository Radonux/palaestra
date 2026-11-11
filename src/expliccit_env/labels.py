"""Ground-truth path labels (candidate LCFs), as pure functions of an
episode's spec-derived constants and its records.

``labels_at`` gives every label at any instants: the step rows net1 trains
on (steps.py) and, through ``labels_table``, one row per TCP_OBS record. A
new candidate LCF added here applies to new episodes, and to stored ones
whose shard kept its link tables (dataset.read_episode gives their
records). All work is vectorized: O((n_obs + n_events) log n_events).

Definitions (link 1 is the data-direction bottleneck):
  capacity_bps        trace delivery-opportunity bits/s over the trailing
                      ``window_ms`` before the sample (1504-byte opportunities,
                      i.e. IP plus Mahimahi's 4-byte header)
  queue_pkts/bytes    bottleneck queue after the last link event at or before
                      the sample, excluding the packet in transit (Mahimahi's
                      droptail definition)
  queue_delay_ns      exact queueing delay (departure instant minus arrival
                      instant) of the latest data packet that reached the
                      bottleneck at or before the sample and was not dropped;
                      NaN if it had not departed by the end of the episode
  base_rtt_ns         two-way propagation delay (both delay lines)
  one_way_delay_ns    one delay line: what Mahak and expliCCit call mRTT
  buffer_pkts/bytes   droptail limits (0 = unlimited)
  random_loss_rate    the data direction's configured i.i.d. loss probability
  cross_traffic_share bytes that cross traffic (every flow but the observed
                      one) took from the bottleneck over the trailing
                      ``window_ms``, as a fraction of the trace's capacity
                      over the same ms; in [0, 1], NaN where the capacity is 0

Windowed labels (trailing ``window_ms`` of whole trace ms before the
sample, as for capacity_bps). ``labels_at`` gives each of them, with
capacity_bps and cross_traffic_share, once per window in ``windows_ms``:
under its own name for the first window (100 ms, expliCCit's) and as
``<name>_<w>ms`` for the others (``label_name``).
  queue_delay_mean_ns mean queueing delay of the packets (all flows) that left
                      the bottleneck in the window; NaN if none did
  free_buffer_pkts    buffer_pkts minus queue_pkts; NaN without a packet limit
  buffer_fill         queue over its limit (the larger of the packet and the
                      byte fraction); 1 means the next arrival is dropped
  busy_fraction       bottleneck bytes used over the bytes the trace offered
                      in the window; NaN where the capacity is 0
  drop_fraction       the observed flow's packets the full queue dropped,
                      over those that reached the bottleneck; NaN if none
  receive_rate_bps    the observed flow's bits leaving the bottleneck per
                      second (link bytes, as capacity_bps counts them); the
                      receiver sees them one propagation delay later
  arrival_rate_bps    bits of every flow reaching the bottleneck queue per
                      second, those the full queue then dropped included
  flow_arrival_rate_bps  the same for the observed flow alone
  queue_min_pkts      the smallest queue_pkts at any instant of the window
                      (the standing queue); NaN for an empty window
  active_flows        flows with a packet reaching the bottleneck in the
                      window
  random_drop_fraction  the observed flow's packets the random loss dropped
                      before the bottleneck, over those it sent toward it
                      (dropped or arrived); NaN if none
  ack_queue_delay_mean_ns  queue_delay_mean_ns of the ACK direction's link:
                      delay the sender sees that the bottleneck queue did
                      not cause
  capacity_ahead_bps  capacity_bps of the ``window_ms`` that start with the
                      sample's ms: the one label that looks forward

Other instant labels:
  flow_queue_pkts     the observed flow's packets in queue_pkts
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from expliccit_env import mahimahi as mm
from expliccit_env.records import DROP_QUEUE_FULL, DROP_RANDOM, Records, Table

DOWNLINK: Final = 1
UPLINK: Final = 2
NS_PER_MS: Final = 1_000_000
# Windows of the windowed labels, in trace ms. The first is expliCCit's and
# the guest's live truth (guest/src/truth.rs); its labels keep plain names.
WINDOWS_MS: Final = (100, 10)
WINDOWED: Final = (
    "capacity_bps",
    "capacity_ahead_bps",
    "queue_delay_mean_ns",
    "ack_queue_delay_mean_ns",
    "queue_min_pkts",
    "busy_fraction",
    "receive_rate_bps",
    "arrival_rate_bps",
    "flow_arrival_rate_bps",
    "active_flows",
    "drop_fraction",
    "random_drop_fraction",
    "cross_traffic_share",
)
# The observed flow's sender port (guest/src/traffic.rs SENDER_ADDR); cross
# flow i sends from CROSS_PORT_BASE + i (guest/src/cross.rs).
MAIN_SPORT: Final = 5001
CROSS_PORT_BASE: Final = 6000

type F64 = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PathConstants:
    base_ns: int  # trace ms 0, CLOCK_MONOTONIC
    one_way_ns: int
    buffer_pkts: int
    buffer_bytes: int
    loss_downlink: float = 0.0


def label_name(name: str, window_ms: int) -> str:
    """A windowed label's column name for one window."""
    return name if window_ms == WINDOWS_MS[0] else f"{name}_{window_ms}ms"


def _window_ms(base_ns: int, t_ns: mm.I64, window_ms: int, end_ms: int) -> tuple[mm.I64, mm.I64]:
    """Each sample's trailing window as trace ms [lo, hi)."""
    hi = np.clip((t_ns - base_ns) // NS_PER_MS, 0, end_ms)
    return np.clip(hi - window_ms, 0, None), hi


def capacity_bps(trace: mm.I64, base_ns: int, t_ns: mm.I64, window_ms: int, end_ms: int) -> F64:
    """Trailing-window capacity from the trace, at each sample time."""
    per_ms = np.bincount(mm.opportunity_times(trace, end_ms + 1), minlength=end_ms + 1)
    cum = np.concatenate(([0], np.cumsum(per_ms)))  # cum[k] = opportunities at ms < k
    lo, hi = _window_ms(base_ns, t_ns, window_ms, end_ms)
    return (cum[hi] - cum[lo]) * (mm.PACKET_SIZE * 8) / (window_ms / 1000.0)


def cross_traffic_share(
    rec: Records,
    trace: mm.I64,
    base_ns: int,
    t_ns: mm.I64,
    *,
    window_ms: int,
    end_ms: int,
    sport: int = MAIN_SPORT,
) -> F64:
    """The share of the bottleneck capacity every flow but ``sport`` used in
    each sample's trailing window. Departures use opportunities, so it lies
    in [0, 1]."""
    dep = rec.link(rec.link_depart, DOWNLINK)
    cross = dep[dep["sport"] != sport]
    cum = np.concatenate(([0], np.cumsum(cross["bytes"].astype(np.int64))))
    lo, hi = _window_ms(base_ns, t_ns, window_ms, end_ms)
    ms = cross["ms"].astype(np.int64)  # nondecreasing: departures are FIFO
    used = cum[np.searchsorted(ms, hi)] - cum[np.searchsorted(ms, lo)]
    cap = capacity_bps(trace, base_ns, t_ns, window_ms, end_ms) * (window_ms / 1000.0) / 8
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(cap > 0, used / cap, np.nan)


def _queue_after_events(rec: Records, link_id: int) -> tuple[mm.I64, mm.I64, mm.I64]:
    """(event times, queue packets after, queue bytes after), time-ordered.

    Ties at one instant: a departure on an opportunity precedes an arrival in
    that instant (opportunities are used before arrivals join), and a drop
    follows its own arrival (it has the larger sequence number).
    """
    arr = rec.link(rec.link_arrival, link_id)
    drp = rec.link(rec.link_drop, link_id)
    # Only a full queue drops an arrival; every other drop happens before the
    # link (random loss, delay line) and leaves the queue unchanged.
    drp = drp[drp["reason"] == DROP_QUEUE_FULL]
    dep = rec.link(rec.link_depart, link_id)
    t = np.concatenate([dep["t_ns"], arr["t_ns"], drp["t_ns"]]).astype(np.int64)
    rank = np.concatenate([np.zeros(dep.size), np.ones(arr.size), np.full(drp.size, 2)])
    seq = np.concatenate([dep["seq"], arr["seq"], drp["seq"]]).astype(np.int64)
    q_pkts = np.concatenate([dep["qlen_pkts"], arr["qlen_pkts"] + 1, drp["qlen_pkts"]]).astype(
        np.int64
    )
    q_bytes = np.concatenate(
        [dep["qlen_bytes"], arr["qlen_bytes"] + arr["bytes"].astype(np.int64), drp["qlen_bytes"]]
    ).astype(np.int64)
    order = np.lexsort((seq, rank, t))
    return t[order], q_pkts[order], q_bytes[order]


def _accepted(rec: Records, link_id: int) -> Table:
    """The arrivals the full queue did not drop, in order: with FIFO, the
    i-th of them is the i-th departure."""
    arr = rec.link(rec.link_arrival, link_id)
    drp = rec.link(rec.link_drop, link_id)
    drp = drp[drp["reason"] == DROP_QUEUE_FULL]
    seq0 = rec.meta["seq"][0]
    arr_seq: mm.I64 = (arr["seq"] - seq0).astype(np.int64)
    # A drop's arrival is the last arrival emitted before it on this link.
    drp_seq: mm.I64 = (drp["seq"] - seq0).astype(np.int64)
    dropped_idx: npt.NDArray[np.intp] = np.searchsorted(arr_seq, drp_seq) - 1
    keep = np.ones(arr.size, dtype=bool)
    keep[dropped_idx] = False
    return arr[keep]


def _accepted_delays(rec: Records, link_id: int) -> tuple[mm.I64, F64]:
    """Arrival instants of non-dropped packets and their exact queueing delays."""
    dep = rec.link(rec.link_depart, link_id)
    acc_t = _accepted(rec, link_id)["t_ns"].astype(np.int64)
    delay = np.full(acc_t.size, np.nan)
    n = min(acc_t.size, dep.size)
    delay[:n] = dep["t_ns"][:n].astype(np.int64) - acc_t[:n]
    return acc_t, delay


def _flow_queue(rec: Records, t: mm.I64, q_pkts: mm.I64, sport: int) -> mm.I64:
    """One flow's packets in the queue at each instant. Like queue_pkts it
    leaves out the packet in transit, which has left the queue but not yet
    departed: the head of the accepted arrivals, when the packets that
    arrived and have not departed are one more than the queue."""
    acc, dep = _accepted(rec, DOWNLINK), rec.link(rec.link_depart, DOWNLINK)
    departed = _count_at(dep["t_ns"], t)
    transit = _count_at(acc["t_ns"], t) - departed > q_pkts
    head = acc["sport"][np.clip(departed, 0, max(acc.size - 1, 0))] if acc.size else np.zeros(1)
    own = acc["sport"] == sport
    waiting = _count_at(acc["t_ns"][own], t) - _count_at(dep["t_ns"][dep["sport"] == sport], t)
    return waiting - (transit & (head == sport))


def _flow_arrivals_and_drops(rec: Records, sport: int = MAIN_SPORT) -> tuple[Table, Table]:
    """One flow's packets that reached the bottleneck, and those the full
    queue dropped (other drops happen before the link)."""
    arr, drp = rec.link(rec.link_arrival, DOWNLINK), rec.link(rec.link_drop, DOWNLINK)
    arr, drp = arr[arr["sport"] == sport], drp[drp["sport"] == sport]
    return arr, drp[drp["reason"] == DROP_QUEUE_FULL]


def drop_fraction(rec: Records, duration_ms: int) -> float:
    """The observed flow's packets the full queue dropped over those that
    reached the bottleneck in the episode; NaN if none reached it."""
    arr, drp = _flow_arrivals_and_drops(rec)
    n = int((arr["ms"] < duration_ms).sum())
    return int((drp["ms"] < duration_ms).sum()) / n if n else float("nan")


def _window_sums(ms: mm.I64, values: F64, lo: mm.I64, hi: mm.I64) -> tuple[F64, mm.I64]:
    """Sum and count of ``values`` whose (nondecreasing) ``ms`` lies in [lo, hi)."""
    cum = np.concatenate(([0.0], np.cumsum(values)))
    a, b = np.searchsorted(ms, lo), np.searchsorted(ms, hi)
    return cum[b] - cum[a], (b - a).astype(np.int64)


def _queue_min(
    queue: tuple[mm.I64, mm.I64], base_ns: int, hi: mm.I64, *, window_ms: int, end_ms: int
) -> F64:
    """The smallest queue at any instant of trace ms [hi - window_ms, hi),
    from the queue after each event; NaN for an empty window. O(end_ms *
    window_ms), which is small beside the events."""
    ev_t, ev_pkts = queue
    starts = base_ns + np.arange(end_ms, dtype=np.int64) * NS_PER_MS
    # The queue as each ms begins, then the least any event in the ms left.
    i = np.searchsorted(ev_t, starts, side="left") - 1
    per_ms = np.where(i >= 0, ev_pkts[np.clip(i, 0, None)], 0) if ev_t.size else np.zeros(end_ms)
    per_ms = per_ms.astype(np.float64)
    ms = (ev_t - base_ns) // NS_PER_MS
    inside = (ms >= 0) & (ms < end_ms)
    np.minimum.at(per_ms, ms[inside], ev_pkts[inside])
    padded = np.concatenate([np.full(window_ms, np.inf), per_ms])
    least = np.lib.stride_tricks.sliding_window_view(padded, window_ms).min(axis=1)[hi]
    return np.where(np.isfinite(least), least, np.nan)


def _count_at(t_sorted: npt.NDArray[np.generic], t: mm.I64) -> mm.I64:
    """How many of the (nondecreasing) instants are at or before each ``t``."""
    return np.searchsorted(t_sorted.astype(np.int64), t, side="right").astype(np.int64)


def labels_at(
    rec: Records,
    trace: mm.I64,
    k: PathConstants,
    t: mm.I64,
    *,
    duration_ms: int,
    windows_ms: tuple[int, ...] = WINDOWS_MS,
    sport: int = MAIN_SPORT,
) -> dict[str, npt.NDArray[np.generic]]:
    """Every label at the instants ``t`` (CLOCK_MONOTONIC ns, nondecreasing)
    for the flow with sender port ``sport``: its rates, drops, queue share
    and competing share; the rest are the path's. Windowed labels come once
    per window, named by ``label_name``."""
    n = t.size
    ev_t, ev_pkts, ev_bytes = _queue_after_events(rec, DOWNLINK)
    i = np.searchsorted(ev_t, t, side="right") - 1
    q_pkts = (
        np.where(i >= 0, ev_pkts[np.clip(i, 0, None)], 0) if ev_t.size else np.zeros(n, np.int64)
    )
    q_bytes = (
        np.where(i >= 0, ev_bytes[np.clip(i, 0, None)], 0) if ev_t.size else np.zeros(n, np.int64)
    )
    acc_t, acc_delay = _accepted_delays(rec, DOWNLINK)
    j = np.searchsorted(acc_t, t, side="right") - 1
    qdelay = (
        np.where(j >= 0, acc_delay[np.clip(j, 0, None)], np.nan)
        if acc_t.size
        else np.full(n, np.nan)
    )
    fills = [q_pkts / k.buffer_pkts] if k.buffer_pkts else []
    fills += [q_bytes / k.buffer_bytes] if k.buffer_bytes else []
    out: dict[str, npt.NDArray[np.generic]] = {
        "queue_pkts": q_pkts.astype(np.int64),
        "queue_bytes": q_bytes.astype(np.int64),
        "flow_queue_pkts": _flow_queue(rec, t, q_pkts, sport),
        "queue_delay_ns": qdelay,
        "free_buffer_pkts": (
            k.buffer_pkts - q_pkts.astype(np.float64) if k.buffer_pkts else np.full(n, np.nan)
        ),
        "buffer_fill": np.maximum.reduce(fills),
        "base_rtt_ns": np.full(n, 2 * k.one_way_ns, dtype=np.int64),
        # Mahak's and expliCCit's "mRTT" (named for what it is: one way).
        "one_way_delay_ns": np.full(n, k.one_way_ns, dtype=np.int64),
        "buffer_pkts": np.full(n, k.buffer_pkts, dtype=np.int64),
        "buffer_bytes": np.full(n, k.buffer_bytes, dtype=np.int64),
        "random_loss_rate": np.full(n, k.loss_downlink),
    }
    for w in windows_ms:
        windowed = _windowed(
            rec,
            trace,
            k,
            t,
            queue=(ev_t, ev_pkts),
            acc_delay=acc_delay,
            window_ms=w,
            duration_ms=duration_ms,
            sport=sport,
        )
        assert tuple(windowed) == WINDOWED
        out |= {label_name(name, w): v for name, v in windowed.items()}
    return out


def _windowed(
    rec: Records,
    trace: mm.I64,
    k: PathConstants,
    t: mm.I64,
    *,
    queue: tuple[mm.I64, mm.I64],
    acc_delay: F64,
    window_ms: int,
    duration_ms: int,
    sport: int,
) -> dict[str, npt.NDArray[np.generic]]:
    """The labels in WINDOWED, in that order, for one window."""
    lo, hi = _window_ms(k.base_ns, t, window_ms, duration_ms)
    cap = capacity_bps(trace, k.base_ns, t, window_ms, duration_ms)
    window_s = window_ms / 1000.0
    ahead_end = duration_ms + window_ms
    per_ms = np.bincount(mm.opportunity_times(trace, ahead_end + 1), minlength=ahead_end + 1)
    cum = np.concatenate(([0], np.cumsum(per_ms)))

    def ms_of(table: Table) -> mm.I64:
        return table["ms"].astype(np.int64)

    def byte_sum(table: Table) -> F64:
        return _window_sums(ms_of(table), table["bytes"].astype(np.float64), lo, hi)[0]

    def count(table: Table) -> mm.I64:
        return _window_sums(ms_of(table), np.ones(table.size), lo, hi)[1]

    dep = rec.link(rec.link_depart, DOWNLINK)
    # FIFO: the i-th departure carries the i-th accepted arrival's delay.
    delay_sum, delay_n = _window_sums(ms_of(dep), acc_delay[: dep.size], lo, hi)
    ack_dep = rec.link(rec.link_depart, UPLINK)
    ack_delay = _accepted_delays(rec, UPLINK)[1][: ack_dep.size]
    ack_sum, ack_n = _window_sums(ms_of(ack_dep), ack_delay, lo, hi)
    opp = rec.link(rec.link_opp, DOWNLINK)
    used, _ = _window_sums(ms_of(opp), opp["used_bytes"].astype(np.float64), lo, hi)
    every = rec.link(rec.link_arrival, DOWNLINK)
    arr, drp = _flow_arrivals_and_drops(rec, sport)
    arrivals, drops = count(arr), count(drp)
    lost = rec.link(rec.link_drop, DOWNLINK)
    random = count(lost[(lost["sport"] == sport) & (lost["reason"] == DROP_RANDOM)])
    flows = np.zeros(t.size, np.int64)
    for port in np.unique(every["sport"]):
        flows += count(every[every["sport"] == port]) > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "capacity_bps": cap,
            "capacity_ahead_bps": (cum[hi + window_ms] - cum[hi]) * (mm.PACKET_SIZE * 8) / window_s,
            "queue_delay_mean_ns": np.where(delay_n > 0, delay_sum / delay_n, np.nan),
            "ack_queue_delay_mean_ns": np.where(ack_n > 0, ack_sum / ack_n, np.nan),
            "queue_min_pkts": _queue_min(
                queue, k.base_ns, hi, window_ms=window_ms, end_ms=duration_ms
            ),
            "busy_fraction": np.where(cap > 0, used * 8 / (cap * window_s), np.nan),
            "receive_rate_bps": byte_sum(dep[dep["sport"] == sport]) * 8 / window_s,
            "arrival_rate_bps": byte_sum(every) * 8 / window_s,
            "flow_arrival_rate_bps": byte_sum(arr) * 8 / window_s,
            "active_flows": flows,
            "drop_fraction": np.where(arrivals > 0, drops / arrivals, np.nan),
            "random_drop_fraction": np.where(
                arrivals + random > 0, random / (arrivals + random), np.nan
            ),
            "cross_traffic_share": cross_traffic_share(
                rec, trace, k.base_ns, t, window_ms=window_ms, end_ms=duration_ms, sport=sport
            ),
        }


def labels_table(
    rec: Records, trace: mm.I64, k: PathConstants, duration_ms: int, window_ms: int = WINDOWS_MS[0]
) -> pa.Table:
    """Labels row for row with the sender observations (TCP_OBS), for one
    window, under plain names: this table has a row per ACK, so it leaves
    the other windows to the steps."""
    t = rec.tcp_obs["t_ns"].astype(np.int64)
    cols = labels_at(rec, trace, k, t, duration_ms=duration_ms, windows_ms=(window_ms,))
    plain = {label_name(name, window_ms): name for name in WINDOWED}
    return pa.Table.from_pydict(
        {
            "obs_index": pa.array(np.arange(t.size, dtype=np.int64)),
            "t_ns": pa.array(rec.tcp_obs["t_ns"]),  # uint64, the join key with obs
            **{plain.get(name, name): pa.array(v) for name, v in cols.items()},
        }
    )


def capacity_cv(trace: mm.I64, duration_ms: int, window_ms: int = 1000) -> float:
    """Capacity volatility over the episode: coefficient of variation of the
    capacity in consecutive windows."""
    per_ms = np.bincount(mm.opportunity_times(trace, duration_ms), minlength=duration_ms)
    w = per_ms[: duration_ms // window_ms * window_ms].reshape(-1, window_ms).sum(axis=1)
    return float(w.std() / w.mean()) if w.size and w.mean() > 0 else float("nan")
