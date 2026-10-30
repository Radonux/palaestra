"""Laws of the Mahimahi/Mahak ports, and equality with the Perl original."""

from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from expliccit_env import mahimahi as mm
from tests.perl_ref import run_perl

REPO = Path(__file__).parents[1]

traces = (
    st.lists(st.integers(0, 400), min_size=1, max_size=200)
    .map(lambda xs: np.array(sorted(xs), dtype=np.int64))
    .filter(lambda t: t[-1] > 0)
)


@given(traces)
def test_dense_counts_round_trip(ts: mm.I64) -> None:
    counts = mm.dense_counts(ts)
    assert counts.size == ts[-1] + 1 and counts[-1] > 0
    assert np.array_equal(np.repeat(np.arange(counts.size), counts), ts)


@pytest.mark.parametrize(("bw", "step"), [(10, 1.0), (70, 0.2), (100, 8.0), (5, 0.1), (13, 0.7)])
def test_mahak_trace_levels(bw: int, step: float) -> None:
    ts = mm.mahak_trace(bw, step)
    n1, n2 = int(float(bw) * 1000), int(float(step * bw) * 1000)
    assert ts.size == n1 + n2
    assert np.all(ts[:n1] < mm.LEVEL_MS) and np.all(ts[n1:] >= mm.LEVEL_MS)
    assert np.all(np.diff(ts) >= 0)


@pytest.mark.skipif(
    "EXPLICCIT_MAHAK_DIR" not in os.environ,
    reason="set EXPLICCIT_MAHAK_DIR to a Mahak checkout at 5077583",
)
@pytest.mark.parametrize(("bw", "step"), [(10, 1.0), (70, 0.2), (100, 8.0), (7, 0.3), (45, 1.7)])
def test_mahak_trace_matches_helper_py(bw: int, step: float, tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "mahak_helper", Path(os.environ["EXPLICCIT_MAHAK_DIR"]) / "helper.py"
    )
    assert spec is not None and spec.loader is not None
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    os.chdir(tmp_path)
    (tmp_path / "traces").mkdir()
    name = helper.trace_generator(bw, step, step * bw)
    assert np.array_equal(
        mm.parse_trace((tmp_path / "traces" / name).read_text()), mm.mahak_trace(bw, step)
    )


@st.composite
def arrival_runs(draw: st.DrawFn) -> tuple[mm.I64, list[tuple[int, int]], int, int]:
    ts = draw(traces)
    times = sorted(draw(st.lists(st.integers(0, 1500), max_size=150)))
    sizes = draw(
        st.lists(st.sampled_from([56, 64, 1504, 1004, 1]), min_size=len(times), max_size=len(times))
    )
    limit_pkts = draw(st.integers(0, 12))
    limit_bytes = draw(st.integers(0 if limit_pkts else 1, 20000))
    return ts, list(zip(times, sizes, strict=True)), limit_pkts, limit_bytes


@settings(max_examples=300)
@given(arrival_runs())
def test_link_oracle_laws(run: tuple[mm.I64, list[tuple[int, int]], int, int]) -> None:
    ts, arrivals, limit_pkts, limit_bytes = run
    until = 2000
    ev = mm.replay(ts.tolist(), arrivals, limit_pkts, limit_bytes, until)
    # Every arrival is logged, then either dropped, departed, or still held.
    assert len(ev.arrivals) == len(arrivals)
    held = len(arrivals) - len(ev.drops) - len(ev.departures)
    assert held >= 0
    # FIFO, never in the arrival ms, departures only on opportunity ms.
    dep_arrivals = [a for _, a, _, _, _ in ev.departures]
    assert dep_arrivals == sorted(dep_arrivals)
    assert all(t > a for t, a, _, _, _ in ev.departures)
    assert all(t in ev.opportunities for t, *_ in ev.departures)
    # Droptail never lets the queue exceed its limits.
    for _, _, q_pkts, q_bytes in ev.arrivals:
        assert q_pkts <= (limit_pkts or q_pkts) and q_bytes <= (limit_bytes or q_bytes)
    # Opportunities processed are exactly the trace's, through `until`.
    assert sum(c for c, _ in ev.opportunities.values()) == mm.opportunity_times(ts, until + 1).size
    assert all(0 <= used <= c * mm.PACKET_SIZE for c, used in ev.opportunities.values())
    delivered = sum(b for _, _, b, _, _ in ev.departures)
    used = sum(u for _, u in ev.opportunities.values())
    assert delivered <= used <= delivered + mm.PACKET_SIZE * held


@st.composite
def mm_logs(draw: st.DrawFn) -> mm.MmLog:
    base = draw(st.integers(0, 10**6))
    n = draw(st.integers(1, 60))
    ts, kind, nbytes, delay = [], [], [], []
    for _ in range(n):
        k = draw(st.sampled_from(list(mm.Kind)))
        t = draw(st.integers(0, 3000))
        ts.append(base + t)
        kind.append(k)
        nbytes.append(mm.PACKET_SIZE if k is mm.Kind.OPPORTUNITY else draw(st.integers(0, 1504)))
        delay.append(
            draw(st.integers(0, t))
            if k is mm.Kind.DEPARTURE
            else draw(st.integers(0, 3000))
            if k is mm.Kind.DROP
            else -1
        )
    return mm.MmLog(
        base,
        np.array(ts, np.int64),
        np.array(kind, np.uint8),
        np.array(nbytes, np.int64),
        np.array(delay, np.int64),
    )


@pytest.mark.skipif(shutil.which("perl") is None, reason="needs perl")
@settings(max_examples=250, deadline=None)
@given(log=mm_logs())
def test_mm_th_delay_matches_perl(log: mm.MmLog, tmp_path_factory: pytest.TempPathFactory) -> None:
    tmp = tmp_path_factory.mktemp("perl")
    perl = run_perl(log, tmp)
    match mm.mm_th_delay(log):
        case mm.MmThDelay() as r:
            want = r.perl_output()
            assert perl.startswith(want), (perl, want)
            # The script may die afterwards (gnuplot binning); never mid-number.
            assert perl[len(want) : len(want) + 1] not in set("0123456789.e+-")
        case mm.MmThDelayUndefined():
            first = perl.split("\n", 1)[0]
            assert not first.replace(".", "", 1).replace("e-", "", 1).isdigit(), perl


@given(mm_logs())
def test_mm_log_format_round_trip(log: mm.MmLog) -> None:
    back = mm.parse_mm_log(mm.format_mm_log(log))
    for f in ("ts", "kind", "nbytes"):
        assert np.array_equal(getattr(back, f), getattr(log, f))
    dep = log.kind == mm.Kind.DEPARTURE
    assert np.array_equal(back.delay[dep], log.delay[dep])


def test_mahak_first_power() -> None:
    r = mm.MmThDelay(0.9, 3.0, 10.0, 11.0, 9.0, 20.0)
    assert mm.mahak_metric(r, "utilization", 10) == 0.9
    assert mm.mahak_metric(r, "delay", 10) == 0.3
    assert mm.mahak_metric(r, "first_power", 10) == 0.9 / (3.0 / 10.0)


def test_periodic_trace_has_exact_rates_and_period() -> None:
    ts = mm.periodic_trace([24.0, 48.0], 7000)
    assert int(ts[-1]) == 14_000  # Mahimahi's period: the whole cycle
    assert int((ts <= 7000).sum()) == 24_000_000 * 7 // 12_000
    assert int((ts > 7000).sum()) == 48_000_000 * 7 // 12_000
    assert np.all(np.diff(ts) >= 0) and int(ts[0]) >= 1
