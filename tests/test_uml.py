"""End-to-end checks of Tier 1: real kernel, sch_mmlink, guest runner.

Needs a built UML tree (scripts/build-uml.sh, then build-guest.sh) named by
EXPLICCIT_UML_DIR; skipped otherwise.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from expliccit_env import mahimahi as mm
from expliccit_env.guest_plan import GuestPlan, LinkPlan
from expliccit_env.records import Records, load_records
from expliccit_env.uml import UmlBuild, run_guest
from tests.perl_ref import run_perl

pytestmark = pytest.mark.uml


@pytest.fixture(scope="module")
def build() -> UmlBuild:
    path = os.environ.get("EXPLICCIT_UML_DIR")
    if not path:
        pytest.skip("set EXPLICCIT_UML_DIR to a built UML tree")
    return UmlBuild.at(Path(path))


def episode(
    build: UmlBuild,
    tmp: Path,
    *,
    cc: str,
    bw: int,
    step: float,
    mrtt_ms: int,
    buffer: int,
    duration_ms: int,
    seed: int = 1,
) -> tuple[Records, mm.I64, dict[str, object]]:
    trace = mm.mahak_trace(bw, step)
    link = LinkPlan(
        delay_ns=mrtt_ms * 1_000_000, trace=mm.dense_counts(trace), limit_pkts=buffer, limit_bytes=0
    )
    GuestPlan(duration_ms=duration_ms, cc=cc, downlink=link, uplink=link).write(tmp / "in")
    run = run_guest(build, tmp / "in", tmp / "out", seed=seed, timeout_s=300)
    result = json.loads((tmp / "out" / "result.json").read_text())
    assert run.returncode == 0 and result["status"] == "ok", result
    return load_records(tmp / "out"), trace, result


def assert_matches_oracle(rec: Records, trace: mm.I64, buffer: int) -> None:
    for link_id in (1, 2):
        arr = rec.link(rec.link_arrival, link_id)
        dep = rec.link(rec.link_depart, link_id)
        drp = rec.link(rec.link_drop, link_id)
        opp = rec.link(rec.link_opp, link_id)
        last = int(opp["ms"].max())
        ev = mm.replay(
            trace.tolist(),
            zip(arr["ms"].tolist(), arr["bytes"].tolist(), strict=True),
            buffer,
            0,
            last,
        )
        cols = ("ms", "bytes", "qlen_pkts", "qlen_bytes")
        assert list(zip(*(arr[c].tolist() for c in cols), strict=True)) == ev.arrivals
        cols = ("ms", "arrival_ms", "bytes", "qlen_pkts", "qlen_bytes")
        assert list(zip(*(dep[c].tolist() for c in cols), strict=True)) == ev.departures
        assert list(zip(drp["ms"].tolist(), drp["bytes"].tolist(), strict=True)) == ev.drops
        kernel_opp = {
            int(m): [int(c), int(u)]
            for m, c, u in zip(opp["ms"], opp["count"], opp["used_bytes"], strict=True)
        }
        assert kernel_opp == ev.opportunities


@pytest.mark.parametrize("cc", ["cubic", "bbr"])
def test_kernel_link_matches_mahimahi_oracle(build: UmlBuild, tmp_path: Path, cc: str) -> None:
    # 14 s crosses Mahak's 12 s capacity step (20 -> 4 Mbit/s) into a small buffer.
    rec, trace, result = episode(
        build, tmp_path, cc=cc, bw=20, step=0.2, mrtt_ms=10, buffer=30, duration_ms=14000
    )
    episode_report = result["episode"]
    assert isinstance(episode_report, dict)
    assert episode_report["cc_verified"] == cc
    assert episode_report["ring_dropped"] == 0 and episode_report["softnet_dropped"] == 0
    stats = rec.link_stats
    assert stats.size == 2
    for f in ("drops_delay", "drops_oversize", "drops_gso", "early"):
        assert int(stats[f].sum()) == 0, f
    assert rec.link_drop.size > 0, "the step should overflow the buffer"
    assert_matches_oracle(rec, trace, 30)


def test_ack_delay_line_is_exact(build: UmlBuild, tmp_path: Path) -> None:
    delay_ms = 25
    rec, _, _ = episode(
        build, tmp_path, cc="cubic", bw=10, step=1.0, mrtt_ms=delay_ms, buffer=100, duration_ms=3000
    )
    up, obs = rec.link(rec.link_depart, 2), rec.tcp_obs
    start = int(np.flatnonzero(up["tcp_ack"] == obs["ack_seq"][0])[0])
    seg = up[start : start + obs.size]
    assert np.array_equal(seg["tcp_ack"], obs["ack_seq"])
    lag = obs["t_ns"].astype(np.int64) - seg["t_ns"].astype(np.int64) - delay_ms * 1_000_000
    # UML's clock has 256 ns ticks and process-context clock reads advance it
    # (docs/m0-report.md, F4); the link model itself uses exact instants.
    assert lag.min() >= 0 and lag.max() < 2000


def test_same_seed_is_bit_identical(build: UmlBuild, tmp_path: Path) -> None:
    kw = {"cc": "bbr", "bw": 12, "step": 1.0, "mrtt_ms": 15, "buffer": 40, "duration_ms": 2000}
    episode(build, tmp_path / "a", seed=7, **kw)  # type: ignore[arg-type]
    episode(build, tmp_path / "b", seed=7, **kw)  # type: ignore[arg-type]
    episode(build, tmp_path / "c", seed=8, **kw)  # type: ignore[arg-type]
    files = ("link_arrival", "link_depart", "link_drop", "link_opp", "link_stats", "tcp_obs")
    read = lambda run, f: (tmp_path / run / "out" / f"{f}.bin").read_bytes()  # noqa: E731
    assert all(read("a", f) == read("b", f) for f in files)
    assert any(read("a", f) != read("c", f) for f in files)


@pytest.mark.skipif(shutil.which("perl") is None, reason="needs perl")
def test_mahak_metrics_agree_with_perl(build: UmlBuild, tmp_path: Path) -> None:
    rec, trace, _ = episode(
        build, tmp_path, cc="cubic", bw=30, step=0.5, mrtt_ms=20, buffer=200, duration_ms=14000
    )
    log = mm.mm_log_from_records(rec, 1, trace, 14000)
    r = mm.mm_th_delay(log)
    assert isinstance(r, mm.MmThDelay)
    assert run_perl(log, tmp_path).startswith(r.perl_output())
    assert 0.5 < r.utilization <= 1.0
    assert r.mean_delay_ms >= 1.0
