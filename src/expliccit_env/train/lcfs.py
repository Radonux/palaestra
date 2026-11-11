"""The LCFs net1 can learn, by name: label columns (labels.labels_at) and
composites, each a fixed formula over label columns of one window.

A composite is never stored. ``Labels`` computes it from the stored columns
the first time it is asked for, so adding one needs no new data, and an LCF
set that holds a composite beside the columns it is made of is redundant by
construction: choose the set with that in mind. A windowed name follows
labels.label_name: plain for the first window, ``<name>_<w>ms`` otherwise.

Where a composite comes from is in docs/lcf-concepts.md.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Final

import numpy as np
import numpy.typing as npt

from expliccit_env.labels import WINDOWED, WINDOWS_MS, label_name
from expliccit_env.mahimahi import PACKET_SIZE

type F32 = npt.NDArray[np.float32]
type Get = Callable[[str], F32]  # a label column of the composite's window

# Label columns, with the factor to the units net1's errors are reported in.
_COLUMNS: Final = {
    "queue_delay_mean_ns": (1e-6, "ms"),
    "queue_delay_ns": (1e-6, "ms"),
    "ack_queue_delay_mean_ns": (1e-6, "ms"),
    "queue_pkts": (1.0, "pkts"),
    "queue_bytes": (1.0, "bytes"),
    "queue_min_pkts": (1.0, "pkts"),
    "flow_queue_pkts": (1.0, "pkts"),
    "capacity_bps": (1e-6, "Mbit/s"),
    "capacity_ahead_bps": (1e-6, "Mbit/s"),
    "free_buffer_pkts": (1.0, "pkts"),
    "buffer_pkts": (1.0, "pkts"),
    "buffer_fill": (1.0, ""),
    "busy_fraction": (1.0, ""),
    "base_rtt_ns": (1e-6, "ms"),
    "one_way_delay_ns": (1e-6, "ms"),
    "receive_rate_bps": (1e-6, "Mbit/s"),
    "arrival_rate_bps": (1e-6, "Mbit/s"),
    "flow_arrival_rate_bps": (1e-6, "Mbit/s"),
    "active_flows": (1.0, "flows"),
    "random_loss_rate": (1.0, ""),
    "random_drop_fraction": (1.0, ""),
    "cross_traffic_share": (1.0, ""),
}


def _ratio(a: F32, b: F32) -> F32:
    """a / b, NaN where b is 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / b, np.float32(np.nan)).astype(np.float32)


def _used(g: Get) -> F32:
    """The busy fraction, 0 where the link offered nothing (there it is NaN)."""
    return np.nan_to_num(g("busy_fraction"))


def _queue_in_bdps(g: Get) -> F32:
    """The queue as a fraction of the bandwidth-delay product."""
    return _ratio(g("queue_bytes") * 8, g("capacity_bps") * g("base_rtt_ns") * np.float32(1e-9))


# name -> (formula, factor to report units, unit). C is capacity_bps, U the
# busy fraction, y the arrival rate, q the queue, R the base RTT.
COMPOSITES: Final[dict[str, tuple[Callable[[Get], F32], float, str]]] = {
    # Used = C U, what the link sent (RFC 5136; HPCC's txRate).
    "departure_rate_bps": (lambda g: g("capacity_bps") * _used(g), 1e-6, "Mbit/s"),
    # A = C (1 - U), the capacity no flow used (RFC 5136).
    "available_capacity_bps": (
        lambda g: g("capacity_bps") * (1 - _used(g)),
        1e-6,
        "Mbit/s",
    ),
    # y - C: positive while the queue grows (XCP's spare bandwidth, negated).
    "excess_demand_bps": (lambda g: g("arrival_rate_bps") - g("capacity_bps"), 1e-6, "Mbit/s"),
    # z = y / C, which keeps growing under overload where U stops at 1 (the
    # OSU scheme's load factor with the target at capacity). TIMELY's queue
    # growth rate (y - C) / C is this less 1.
    "load_factor": (lambda g: _ratio(g("arrival_rate_bps"), g("capacity_bps")), 1.0, ""),
    # C R in 1504-byte packets: the data in flight on a full link with an
    # empty queue (CCmatic).
    "bdp_pkts": (
        lambda g: g("capacity_bps") * g("base_rtt_ns") * np.float32(1e-9 / (8 * PACKET_SIZE)),
        1.0,
        "pkts",
    ),
    # q / mu, the wait of a packet that arrives now (PIE), mu the rate out.
    "littles_delay_ns": (
        lambda g: _ratio(
            g("queue_bytes") * np.float32(8e9), g("capacity_bps") * g("busy_fraction")
        ),
        1e-6,
        "ms",
    ),
    # U_j = q / (C R) + txRate / C: 1 on a full link with an empty queue (HPCC).
    "normalized_inflight": (lambda g: _queue_in_bdps(g) + g("busy_fraction"), 1.0, ""),
    # (q + C R) y / (C^2 R): PowerTCP's power over its value on a link that is
    # just full.
    "normalized_power": (
        lambda g: (1 + _queue_in_bdps(g)) * _ratio(g("arrival_rate_bps"), g("capacity_bps")),
        1.0,
        "",
    ),
    # The other flows' arrival rate (Dovrolis 2001's cross-traffic rate).
    "cross_arrival_rate_bps": (
        lambda g: g("arrival_rate_bps") - g("flow_arrival_rate_bps"),
        1e-6,
        "Mbit/s",
    ),
    # C / N, the processor-sharing rate (RCP).
    "fair_share_bps": (lambda g: _ratio(g("capacity_bps"), g("active_flows")), 1e-6, "Mbit/s"),
    # The flow's packets into the bottleneck over those out of it: above 1
    # while its queue or its losses build (Aurora's sending ratio, at the link).
    "sending_ratio": (
        lambda g: _ratio(g("flow_arrival_rate_bps"), g("receive_rate_bps")),
        1.0,
        "",
    ),
    # (R + queueing delay) / R (Aurora's latency ratio, Remy's RTT ratio, on
    # the true delays).
    "rtt_ratio": (
        lambda g: 1 + _ratio(np.nan_to_num(g("queue_delay_mean_ns")), g("base_rtt_ns")),
        1.0,
        "",
    ),
}


def _names(name: str, windowed: bool) -> list[tuple[str, int]]:
    """(column name, window) for each window the name comes in."""
    return [(label_name(name, w), w) for w in (WINDOWS_MS if windowed else WINDOWS_MS[:1])]


# Every LCF name -> (factor, unit).
LCF_UNITS: Final[dict[str, tuple[float, str]]] = {
    column: unit for name, unit in _COLUMNS.items() for column, _ in _names(name, name in WINDOWED)
} | {
    column: (factor, unit)
    for name, (_, factor, unit) in COMPOSITES.items()
    for column, _ in _names(name, True)
}
_COMPOSITE_OF: Final = {
    column: (COMPOSITES[name][0], w) for name in COMPOSITES for column, w in _names(name, True)
}


def _getter(get: Get, window_ms: int) -> Get:
    return lambda name: get(label_name(name, window_ms) if name in WINDOWED else name)


def columns(lcfs: Iterable[str]) -> set[str]:
    """The stored label columns these LCFs are read or computed from."""
    out: set[str] = set()

    def note(name: str) -> F32:
        out.update(columns([name]))
        return np.zeros(1, np.float32)

    for name in lcfs:
        if name in _COMPOSITE_OF:
            formula, w = _COMPOSITE_OF[name]
            formula(_getter(note, w))
        else:
            out.add(name)
    return out


class Labels(dict[str, F32]):
    """Label columns by name; a composite is computed on first use and kept."""

    def __missing__(self, name: str) -> F32:
        if name not in _COMPOSITE_OF:
            raise KeyError(f"no label {name}: not in the data, and not a composite")
        formula, w = _COMPOSITE_OF[name]
        self[name] = value = formula(_getter(self.__getitem__, w)).astype(np.float32)
        return value
