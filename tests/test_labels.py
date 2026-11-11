"""Ground-truth labels against their definitions, on a link the Mahimahi
oracle replays (no UML): records are built from the oracle's events."""

from __future__ import annotations

import numpy as np
import pytest

from expliccit_env import labels
from expliccit_env import mahimahi as mm
from expliccit_env import records as r
from expliccit_env.train.lcfs import COMPOSITES, LCF_UNITS, Labels, columns
from expliccit_env.train.spec import EXPLICCIT_LCFS, Net1Spec

BASE_NS = 7_000_000_000
MS = labels.NS_PER_MS
DURATION_MS = 400
BUFFER = 12
PORTS = (labels.MAIN_SPORT, 6000, 6001)


def _link(link_id: int, trace: mm.I64, arrivals: list[tuple[int, int, int]], seq0: int):
    """One link's records from (ms, bytes, sport) arrivals in time order.
    Within a ms departures come first, then arrivals 1 us apart, each drop
    at its arrival's instant: the qdisc's order."""
    ev = mm.replay(trace.tolist(), [(ms, size) for ms, size, _ in arrivals], BUFFER, 0, DURATION_MS)
    drops = set()
    left = list(ev.drops)
    for i, (ms, size, _) in enumerate(arrivals):  # a drop is of the arrival it follows
        found_pkts = ev.arrivals[i][2]
        if left and left[0] == (ms, size) and found_pkts == BUFFER:
            drops.add(i)
            left.pop(0)
    assert not left
    accepted = [a for i, a in enumerate(arrivals) if i not in drops]
    rows: list[tuple[int, int, str, dict[str, int]]] = []  # (t_ns, order, table, fields)
    for i, ((ms, size, sport), (_, _, pkts, nbytes)) in enumerate(
        zip(arrivals, ev.arrivals, strict=True)
    ):
        t = BASE_NS + ms * MS + 1000 * (1 + i % 900)
        f = {"ms": ms, "bytes": size, "sport": sport, "qlen_pkts": pkts, "qlen_bytes": nbytes}
        rows.append((t, 2 * i + 1, "link_arrival", f))
        if i in drops:
            rows.append((t, 2 * i + 2, "link_drop", f | {"reason": r.DROP_QUEUE_FULL}))
    for i, (ms, arrival_ms, size, pkts, nbytes) in enumerate(ev.departures):
        f = {"ms": ms, "arrival_ms": arrival_ms, "bytes": size, "sport": accepted[i][2]}
        rows.append(
            (BASE_NS + ms * MS, -1, "link_depart", f | {"qlen_pkts": pkts, "qlen_bytes": nbytes})
        )
    rows.sort(key=lambda row: row[:2])
    out: dict[str, list[dict[str, int]]] = {"link_arrival": [], "link_drop": [], "link_depart": []}
    for k, (t, _, table, f) in enumerate(rows):
        out[table].append(f | {"t_ns": t, "seq": seq0 + k, "link_id": link_id})
    opp = [
        {"ms": ms, "count": c, "used_bytes": u, "link_id": link_id, "t_ns": BASE_NS + ms * MS}
        for ms, (c, u) in sorted(ev.opportunities.items())
    ]
    return out | {"link_opp": opp}, seq0 + len(rows)


def _table(rows: list[dict[str, int]], dtype: np.dtype[np.void]) -> r.Table:
    t = np.zeros(len(rows), dtype)
    for name in {name for row in rows for name in row}:
        t[name] = [row.get(name, 0) for row in rows]
    return t


@pytest.fixture(scope="module")
def episode() -> tuple[r.Records, mm.I64, labels.PathConstants]:
    rng = np.random.default_rng(3)
    trace = mm.periodic_trace([6, 18], 100)  # Mbit/s levels, 100 ms each
    # The observed flow sends throughout, one cross flow in the middle, one
    # late; bursts overflow the 12-packet buffer on the slow level.
    data = sorted(
        [(int(ms), 1504, PORTS[0]) for ms in rng.integers(0, DURATION_MS, 420)]
        + [(int(ms), 1504, PORTS[1]) for ms in rng.integers(120, 260, 150)]
        + [(int(ms), 700, PORTS[2]) for ms in rng.integers(300, 330, 20)]
    )
    acks = sorted((int(ms), 40, 5001) for ms in rng.integers(0, DURATION_MS, 300))
    down, seq = _link(labels.DOWNLINK, trace, data, 1)
    up, seq = _link(labels.UPLINK, mm.periodic_trace([12], 100), acks, seq)
    # Random loss before the bottleneck: drop records without an arrival.
    lost = [
        {"ms": ms, "t_ns": BASE_NS + ms * MS + 5, "bytes": 1504, "sport": PORTS[0]}
        | {"reason": r.DROP_RANDOM, "link_id": labels.DOWNLINK, "seq": seq + k}
        for k, ms in enumerate(sorted(rng.integers(0, DURATION_MS, 25).tolist()))
    ]
    drops = sorted(down["link_drop"] + up["link_drop"] + lost, key=lambda row: row["t_ns"])
    assert sum(d["reason"] == r.DROP_QUEUE_FULL for d in down["link_drop"]) > 5
    meta = np.zeros(1, r.META)
    rec = r.Records(
        meta=meta,
        link_arrival=_table(down["link_arrival"] + up["link_arrival"], r.LINK),
        link_drop=_table(drops, r.LINK),
        link_depart=_table(down["link_depart"] + up["link_depart"], r.LINK),
        link_opp=_table(down["link_opp"] + up["link_opp"], r.OPP),
        link_stats=np.zeros(0, r.STATS),
        tcp_obs=np.zeros(0, r.OBS),
        agent_action=np.zeros(0, r.AGENT),
    )
    return rec, trace, labels.PathConstants(BASE_NS, 10 * MS, BUFFER, 0)


def _samples() -> mm.I64:
    rng = np.random.default_rng(4)
    return BASE_NS + np.sort(rng.integers(0, DURATION_MS * MS, 300)).astype(np.int64)


def test_new_labels_follow_their_definitions(episode) -> None:
    rec, trace, k = episode
    t = _samples()
    lab = labels.labels_at(rec, trace, k, t, duration_ms=DURATION_MS)
    arr, dep = rec.link(rec.link_arrival, 1), rec.link(rec.link_depart, 1)
    drp = rec.link(rec.link_drop, 1)
    full = drp[drp["reason"] == r.DROP_QUEUE_FULL]
    lost = drp[drp["reason"] == r.DROP_RANDOM]
    up_arr, up_dep = rec.link(rec.link_arrival, 2), rec.link(rec.link_depart, 2)
    assert rec.link(rec.link_drop, 2).size == 0  # so ACK i departs as ACK i

    # The queue after every bottleneck event, as its records state it.
    ev_t = np.concatenate([x["t_ns"] for x in (dep, arr, full)]).astype(np.int64)
    ev_q = np.concatenate([dep["qlen_pkts"], arr["qlen_pkts"] + 1, full["qlen_pkts"]])
    order = np.lexsort((np.concatenate([x["seq"] for x in (dep, arr, full)]), ev_t))
    ev_t, ev_q = ev_t[order], ev_q[order].astype(np.int64)

    for w in labels.WINDOWS_MS:
        col = {name: lab[labels.label_name(name, w)] for name in labels.WINDOWED}
        for i, ns in enumerate(t.tolist()):
            hi = (ns - BASE_NS) // MS
            lo = max(hi - w, 0)

            def inside(table: r.Table, lo: int = lo, hi: int = hi):
                return table[(table["ms"] >= lo) & (table["ms"] < hi)]

            own = inside(arr)[inside(arr)["sport"] == PORTS[0]]
            assert col["arrival_rate_bps"][i] == inside(arr)["bytes"].sum() * 8 / (w / 1000)
            assert col["flow_arrival_rate_bps"][i] == own["bytes"].sum() * 8 / (w / 1000)
            assert col["active_flows"][i] == len(set(inside(arr)["sport"].tolist()))
            sent = own.size + inside(lost).size
            want = inside(lost).size / sent if sent else np.nan
            assert np.array_equal(col["random_drop_fraction"][i], want, equal_nan=True)
            left = (up_dep["ms"] >= lo) & (up_dep["ms"] < hi)
            waits = (up_dep["t_ns"][left] - up_arr["t_ns"][: up_dep.size][left]).astype(np.int64)
            want = waits.mean() if waits.size else np.nan
            assert np.array_equal(col["ack_queue_delay_mean_ns"][i], want, equal_nan=True)
            # The queue changes only at events: its least value in the window
            # is the least of its value at the start and after each event.
            start, end = BASE_NS + lo * MS, BASE_NS + hi * MS
            before = ev_q[ev_t < start]
            seen = [int(before[-1]) if before.size else 0]
            seen += ev_q[(ev_t >= start) & (ev_t < end)].tolist()
            want = min(seen) if hi > lo else np.nan
            assert np.array_equal(col["queue_min_pkts"][i], want, equal_nan=True)
            if hi + w <= DURATION_MS:
                later = labels.capacity_bps(trace, BASE_NS, np.array([ns + w * MS]), w, DURATION_MS)
                assert col["capacity_ahead_bps"][i] == later[0]
    # Each flow's share of the queue: together the whole queue, and each
    # its packets that arrived and have not left, less one while its packet
    # is in transit (the 700-byte packets share an opportunity).
    shares = [
        labels.labels_at(rec, trace, k, t, duration_ms=DURATION_MS, sport=port)["flow_queue_pkts"]
        for port in PORTS
    ]
    assert np.array_equal(lab["queue_pkts"], sum(shares)) and lab["queue_pkts"].max() == BUFFER
    assert np.array_equal(shares[0], lab["flow_queue_pkts"])
    transit = np.zeros(t.size, bool)
    for port, share in zip(PORTS, shares, strict=True):

        def n(table: r.Table, port: int = port) -> mm.I64:
            own = table["t_ns"][table["sport"] == port].astype(np.int64)
            return np.searchsorted(own, t, side="right")

        held = n(arr) - n(full) - n(dep)
        assert np.all((share == held) | (share == held - 1)) and np.all(share >= 0)
        transit |= share < held
    assert transit.any()
    assert np.nanmax(lab["queue_min_pkts"]) > 0 and np.nanmax(lab["active_flows"]) == 3
    assert np.nanmax(lab["ack_queue_delay_mean_ns"]) > 0


def test_each_window_names_its_labels(episode) -> None:
    rec, trace, k = episode
    t = _samples()
    both = labels.labels_at(rec, trace, k, t, duration_ms=DURATION_MS)
    assert labels.WINDOWS_MS == (100, 10)
    for w in labels.WINDOWS_MS:
        one = labels.labels_at(rec, trace, k, t, duration_ms=DURATION_MS, windows_ms=(w,))
        assert set(one) <= set(both)
        for name, v in one.items():
            assert np.array_equal(v, both[name], equal_nan=True), name
    assert "busy_fraction" in both and "busy_fraction_10ms" in both
    assert "queue_pkts_10ms" not in both  # instant labels have no window
    # A shorter window sees the capacity steps a longer one averages.
    assert np.ptp(both["capacity_bps_10ms"]) > np.ptp(both["capacity_bps"])


def test_composites_are_formulas_over_one_windows_labels(episode) -> None:
    rec, trace, k = episode
    raw = labels.labels_at(rec, trace, k, _samples(), duration_ms=DURATION_MS)
    lab = Labels({name: v.astype(np.float32) for name, v in raw.items()})
    stored = set(lab)

    def same(name: str, want) -> None:
        assert np.allclose(lab[name], want, equal_nan=True, rtol=1e-5), name

    for w, s in ((100, ""), (10, "_10ms")):
        cap, busy, y = (
            lab[f"capacity_bps{s}"],
            lab[f"busy_fraction{s}"],
            lab[f"arrival_rate_bps{s}"],
        )
        bdp_bits = cap * lab["base_rtt_ns"] * 1e-9
        with np.errstate(divide="ignore", invalid="ignore"):
            q_bdp = np.where(bdp_bits > 0, lab["queue_bytes"] * 8 / bdp_bits, np.nan)
            load = np.where(cap > 0, y / cap, np.nan)
            wait = np.where(cap * busy > 0, lab["queue_bytes"] * 8e9 / (cap * busy), np.nan)
            share = np.where(lab[f"active_flows{s}"] > 0, cap / lab[f"active_flows{s}"], np.nan)
        same(f"departure_rate_bps{s}", cap * np.nan_to_num(busy))
        same(f"available_capacity_bps{s}", cap * (1 - np.nan_to_num(busy)))
        same(f"excess_demand_bps{s}", y - cap)
        same(f"load_factor{s}", load)
        same(f"bdp_pkts{s}", bdp_bits / (8 * 1504))
        same(f"littles_delay_ns{s}", wait)
        same(f"normalized_inflight{s}", q_bdp + busy)
        same(f"normalized_power{s}", (1 + q_bdp) * load)
        same(f"cross_arrival_rate_bps{s}", y - lab[f"flow_arrival_rate_bps{s}"])
        same(f"fair_share_bps{s}", share)
        assert lab[f"departure_rate_bps{s}"].dtype == np.float32
        # Overload: the load factor passes 1 where the busy fraction cannot.
        assert np.nanmax(lab[f"load_factor{s}"]) > 1 >= np.nanmax(busy), w
    # What leaves the link is what the flows receive: the departure rate is
    # the observed flow's receive rate plus the others', to within the one
    # packet that an opportunity before the window began to carry.
    slack = 1504 * 8 / 0.1
    assert np.all(lab["departure_rate_bps"] + slack >= lab["receive_rate_bps"])
    assert not np.isnan(lab["departure_rate_bps"]).any()
    assert np.nanmin(lab["sending_ratio"]) > 0 and np.nanmin(lab["rtt_ratio"]) >= 1

    # Every composite comes in every window, is computed once and kept, and
    # names the stored columns it reads.
    assert all(lab[name].shape == lab["queue_pkts"].shape for name in LCF_UNITS)
    assert set(lab) - stored == {f"{c}{s}" for c in COMPOSITES for s in ("", "_10ms")}
    assert lab["load_factor"] is lab["load_factor"]
    assert columns(["load_factor_10ms"]) == {"arrival_rate_bps_10ms", "capacity_bps_10ms"}
    assert columns(["normalized_inflight", "queue_pkts"]) == {
        "queue_bytes",
        "capacity_bps",
        "base_rtt_ns",
        "busy_fraction",
        "queue_pkts",
    }
    assert columns(LCF_UNITS) <= stored
    with pytest.raises(KeyError, match="nope"):
        lab["nope"]
    # Data from before a label existed: the error names the missing column.
    with pytest.raises(KeyError, match="arrival_rate_bps"):
        Labels({"capacity_bps": raw["capacity_bps"].astype(np.float32)})["load_factor"]

    # net1 takes any of them by name, and nothing else.
    assert set(EXPLICCIT_LCFS) <= set(LCF_UNITS) and LCF_UNITS["capacity_bps"] == (1e-6, "Mbit/s")
    assert LCF_UNITS["queue_min_pkts_10ms"] == (1.0, "pkts") and "queue_pkts_10ms" not in LCF_UNITS
    assert Net1Spec(lcfs=("load_factor_10ms", "queue_min_pkts")).lcfs[0] == "load_factor_10ms"
    with pytest.raises(ValueError, match="unknown LCF"):
        Net1Spec(lcfs=("queue_pkts_10ms",))
