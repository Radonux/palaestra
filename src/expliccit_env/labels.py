"""Ground-truth path labels (candidate LCFs), as pure functions of an
episode's spec-derived constants and its records.

Every label is aligned to the sender observations (one value per TCP_OBS
row), so (obs, labels) is a supervised pair for net1. New candidate LCFs are
added here and recomputed from stored data; nothing requires re-running an
episode. All work is vectorized: O((n_obs + n_events) log n_events).

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

Windowed labels (expliCCit's candidate LCFs; trailing ``window_ms`` of whole
trace ms before the sample, as for capacity_bps):
  queue_delay_mean_ns mean queueing delay of the packets (all flows) that left
                      the bottleneck in the window; NaN if none did
  free_buffer_pkts    buffer_pkts minus queue_pkts; NaN without a packet limit
  buffer_fill         queue over its limit (the larger of the packet and the
                      byte fraction); 1 means the next arrival is dropped
  busy_fraction       bottleneck bytes used over the bytes the trace offered
                      in the window; NaN where the capacity is 0
  receive_rate_bps    the observed flow's bits leaving the bottleneck per
                      second (link bytes, as capacity_bps counts them); the
                      receiver sees them one propagation delay later
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from expliccit_env import mahimahi as mm
from expliccit_env.records import DROP_QUEUE_FULL, Records

DOWNLINK: Final = 1
UPLINK: Final = 2
NS_PER_MS: Final = 1_000_000
# The observed flow's sender port (guest/src/traffic.rs SENDER_ADDR); cross
# flows use other ports.
MAIN_SPORT: Final = 5001

type F64 = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PathConstants:
    base_ns: int  # trace ms 0, CLOCK_MONOTONIC
    one_way_ns: int
    buffer_pkts: int
    buffer_bytes: int
    loss_downlink: float = 0.0


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
    rec: Records, trace: mm.I64, base_ns: int, t_ns: mm.I64, *, window_ms: int, end_ms: int
) -> F64:
    """Cross traffic's share of the bottleneck capacity in each sample's
    trailing window. Departures use opportunities, so it lies in [0, 1]."""
    dep = rec.link(rec.link_depart, DOWNLINK)
    cross = dep[dep["sport"] != MAIN_SPORT]
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


def _accepted_delays(rec: Records, link_id: int) -> tuple[mm.I64, F64]:
    """Arrival instants of non-dropped packets and their exact queueing delays.

    FIFO: the i-th accepted arrival is the i-th departure.
    """
    arr = rec.link(rec.link_arrival, link_id)
    drp = rec.link(rec.link_drop, link_id)
    drp = drp[drp["reason"] == DROP_QUEUE_FULL]
    dep = rec.link(rec.link_depart, link_id)
    seq0 = rec.meta["seq"][0]
    arr_seq: mm.I64 = (arr["seq"] - seq0).astype(np.int64)
    # A drop's arrival is the last arrival emitted before it on this link.
    drp_seq: mm.I64 = (drp["seq"] - seq0).astype(np.int64)
    dropped_idx: npt.NDArray[np.intp] = np.searchsorted(arr_seq, drp_seq) - 1
    keep = np.ones(arr.size, dtype=bool)
    keep[dropped_idx] = False
    acc_t = arr["t_ns"][keep].astype(np.int64)
    delay = np.full(acc_t.size, np.nan)
    n = min(acc_t.size, dep.size)
    delay[:n] = dep["t_ns"][:n].astype(np.int64) - acc_t[:n]
    return acc_t, delay


def _window_sums(ms: mm.I64, values: F64, lo: mm.I64, hi: mm.I64) -> tuple[F64, mm.I64]:
    """Sum and count of ``values`` whose (nondecreasing) ``ms`` lies in [lo, hi)."""
    cum = np.concatenate(([0.0], np.cumsum(values)))
    a, b = np.searchsorted(ms, lo), np.searchsorted(ms, hi)
    return cum[b] - cum[a], (b - a).astype(np.int64)


def labels_at(
    rec: Records,
    trace: mm.I64,
    k: PathConstants,
    t: mm.I64,
    *,
    duration_ms: int,
    window_ms: int = 100,
) -> dict[str, npt.NDArray[np.generic]]:
    """Every label at the instants ``t`` (CLOCK_MONOTONIC ns, nondecreasing)."""
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
    lo, hi = _window_ms(k.base_ns, t, window_ms, duration_ms)
    cap = capacity_bps(trace, k.base_ns, t, window_ms, duration_ms)
    window_s = window_ms / 1000.0
    dep = rec.link(rec.link_depart, DOWNLINK)
    dep_ms = dep["ms"].astype(np.int64)
    # FIFO: the i-th departure carries the i-th accepted arrival's delay.
    dep_delay = acc_delay[: dep.size]
    delay_sum, delay_n = _window_sums(dep_ms, dep_delay, lo, hi)
    main = dep["sport"] == MAIN_SPORT
    recv_bytes, _ = _window_sums(dep_ms[main], dep["bytes"][main].astype(np.float64), lo, hi)
    opp = rec.link(rec.link_opp, DOWNLINK)
    used, _ = _window_sums(opp["ms"].astype(np.int64), opp["used_bytes"].astype(np.float64), lo, hi)
    fills = [q_pkts / k.buffer_pkts] if k.buffer_pkts else []
    fills += [q_bytes / k.buffer_bytes] if k.buffer_bytes else []
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "capacity_bps": cap,
            "queue_pkts": q_pkts.astype(np.int64),
            "queue_bytes": q_bytes.astype(np.int64),
            "queue_delay_ns": qdelay,
            "queue_delay_mean_ns": np.where(delay_n > 0, delay_sum / delay_n, np.nan),
            "free_buffer_pkts": (
                k.buffer_pkts - q_pkts.astype(np.float64) if k.buffer_pkts else np.full(n, np.nan)
            ),
            "buffer_fill": np.maximum.reduce(fills),
            "busy_fraction": np.where(cap > 0, used * 8 / (cap * window_s), np.nan),
            "receive_rate_bps": recv_bytes * 8 / window_s,
            "base_rtt_ns": np.full(n, 2 * k.one_way_ns, dtype=np.int64),
            # Mahak's and expliCCit's "mRTT" (named for what it is: one way).
            "one_way_delay_ns": np.full(n, k.one_way_ns, dtype=np.int64),
            "buffer_pkts": np.full(n, k.buffer_pkts, dtype=np.int64),
            "buffer_bytes": np.full(n, k.buffer_bytes, dtype=np.int64),
            "random_loss_rate": np.full(n, k.loss_downlink),
            "cross_traffic_share": cross_traffic_share(
                rec, trace, k.base_ns, t, window_ms=window_ms, end_ms=duration_ms
            ),
        }


def labels_table(
    rec: Records, trace: mm.I64, k: PathConstants, duration_ms: int, window_ms: int = 100
) -> pa.Table:
    """Labels row for row with the sender observations (TCP_OBS)."""
    t = rec.tcp_obs["t_ns"].astype(np.int64)
    cols = labels_at(rec, trace, k, t, duration_ms=duration_ms, window_ms=window_ms)
    return pa.Table.from_pydict(
        {
            "obs_index": pa.array(np.arange(t.size, dtype=np.int64)),
            "t_ns": pa.array(rec.tcp_obs["t_ns"]),  # uint64, the join key with obs
            **{name: pa.array(v) for name, v in cols.items()},
        }
    )


def capacity_cv(trace: mm.I64, duration_ms: int, window_ms: int = 1000) -> float:
    """Capacity volatility over the episode: coefficient of variation of the
    capacity in consecutive windows."""
    per_ms = np.bincount(mm.opportunity_times(trace, duration_ms), minlength=duration_ms)
    w = per_ms[: duration_ms // window_ms * window_ms].reshape(-1, window_ms).sum(axis=1)
    return float(w.std() / w.mean()) if w.size and w.mean() > 0 else float("nan")
