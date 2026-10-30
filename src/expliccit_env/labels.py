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
  buffer_pkts/bytes   droptail limits (0 = unlimited)
  random_loss_rate    0: no random-loss stage yet
  cross_traffic_share 0: no cross traffic yet
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

type F64 = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PathConstants:
    base_ns: int  # trace ms 0, CLOCK_MONOTONIC
    one_way_ns: int
    buffer_pkts: int
    buffer_bytes: int


def capacity_bps(trace: mm.I64, base_ns: int, t_ns: mm.I64, window_ms: int, end_ms: int) -> F64:
    """Trailing-window capacity from the trace, at each sample time."""
    per_ms = np.bincount(mm.opportunity_times(trace, end_ms + 1), minlength=end_ms + 1)
    cum = np.concatenate(([0], np.cumsum(per_ms)))  # cum[k] = opportunities at ms < k
    ms = np.clip((t_ns - base_ns) // NS_PER_MS, 0, end_ms)
    lo = np.clip(ms - window_ms, 0, None)
    return (cum[ms] - cum[lo]) * (mm.PACKET_SIZE * 8) / (window_ms / 1000.0)


def _queue_after_events(rec: Records, link_id: int) -> tuple[mm.I64, mm.I64, mm.I64]:
    """(event times, queue packets after, queue bytes after), time-ordered.

    Ties at one instant: a departure on an opportunity precedes an arrival in
    that instant (opportunities are used before arrivals join), and a drop
    follows its own arrival (it has the larger sequence number).
    """
    arr = rec.link(rec.link_arrival, link_id)
    drp = rec.link(rec.link_drop, link_id)
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


def labels_table(
    rec: Records, trace: mm.I64, k: PathConstants, duration_ms: int, window_ms: int = 100
) -> pa.Table:
    t = rec.tcp_obs["t_ns"].astype(np.int64)
    ev_t, ev_pkts, ev_bytes = _queue_after_events(rec, DOWNLINK)
    i = np.searchsorted(ev_t, t, side="right") - 1
    q_pkts = (
        np.where(i >= 0, ev_pkts[np.clip(i, 0, None)], 0)
        if ev_t.size
        else np.zeros(t.size, np.int64)
    )
    q_bytes = (
        np.where(i >= 0, ev_bytes[np.clip(i, 0, None)], 0)
        if ev_t.size
        else np.zeros(t.size, np.int64)
    )
    acc_t, acc_delay = _accepted_delays(rec, DOWNLINK)
    j = np.searchsorted(acc_t, t, side="right") - 1
    qdelay = (
        np.where(j >= 0, acc_delay[np.clip(j, 0, None)], np.nan)
        if acc_t.size
        else np.full(t.size, np.nan)
    )
    n = t.size
    return pa.Table.from_pydict(
        {
            "obs_index": pa.array(np.arange(n, dtype=np.int64)),
            "t_ns": pa.array(rec.tcp_obs["t_ns"]),  # uint64, the join key with obs
            "capacity_bps": pa.array(capacity_bps(trace, k.base_ns, t, window_ms, duration_ms)),
            "queue_pkts": pa.array(q_pkts.astype(np.int64)),
            "queue_bytes": pa.array(q_bytes.astype(np.int64)),
            "queue_delay_ns": pa.array(qdelay),
            "base_rtt_ns": pa.array(np.full(n, 2 * k.one_way_ns, dtype=np.int64)),
            "buffer_pkts": pa.array(np.full(n, k.buffer_pkts, dtype=np.int64)),
            "buffer_bytes": pa.array(np.full(n, k.buffer_bytes, dtype=np.int64)),
            "random_loss_rate": pa.array(np.zeros(n)),
            "cross_traffic_share": pa.array(np.zeros(n)),
        }
    )


def capacity_cv(trace: mm.I64, duration_ms: int, window_ms: int = 1000) -> float:
    """Capacity volatility over the episode: coefficient of variation of the
    capacity in consecutive windows."""
    per_ms = np.bincount(mm.opportunity_times(trace, duration_ms), minlength=duration_ms)
    w = per_ms[: duration_ms // window_ms * window_ms].reshape(-1, window_ms).sum(axis=1)
    return float(w.std() / w.mean()) if w.size and w.mean() > 0 else float("nan")
