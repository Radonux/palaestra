"""End-to-end checks of Tier 1: real kernel, sch_mmlink, guest runner.

Needs a built UML tree (scripts/build-uml.sh, then build-guest.sh) named by
EXPLICCIT_UML_DIR; skipped otherwise.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from expliccit_env import campaign, labels
from expliccit_env import mahimahi as mm
from expliccit_env.cli import main
from expliccit_env.episode import EpisodeOk, run_episode
from expliccit_env.guest_plan import GuestPlan, LinkPlan
from expliccit_env.records import DROP_QUEUE_FULL, DROP_RANDOM, Records, load_records
from expliccit_env.spec import CampaignSpec, EpisodeSpec
from expliccit_env.uml import UmlBuild, run_guest
from tests.perl_ref import run_perl

pytestmark = pytest.mark.uml


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
        drp = drp[drp["reason"] == DROP_QUEUE_FULL]  # random loss happens before the link
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


@pytest.mark.parametrize("cc", ["cubic", "bbr", "bbr3"])
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


def test_loss_and_cross_traffic(build: UmlBuild, tmp_path: Path) -> None:
    # 20 Mbit/s; 5 Mbit/s of CBR UDP over [2 s, 8 s); Cubic cross traffic from
    # 6 s; 1% random loss on data, 0.5% on ACKs.
    spec = EpisodeSpec.parse(
        json.dumps(
            {
                "path": {
                    "trace": {"kind": "mahak_step", "bw_mbps": 20, "step": 1.0},
                    "delay": {"base_rtt_ms": 20},
                    "buffer": {"packets": 60},
                    "loss": {"downlink": 0.01, "uplink": 0.005},
                },
                "cc": {"kind": "stock", "name": "bbr"},
                "duration_ms": 10_000,
                "seed": 5,
                "cross": [
                    {"kind": "udp_cbr", "rate_mbps": 5, "start_ms": 2000, "stop_ms": 8000},
                    {"kind": "tcp", "cc": "cubic", "start_ms": 6000},
                ],
            }
        )
    )
    out = run_episode(spec, build, tmp_path / "a", timeout_s=300, keep_raw=True)
    assert isinstance(out, EpisodeOk), out.row
    raw = tmp_path / "a" / spec.episode_id() / "raw"
    rec = load_records(raw)
    # Seeded loss and cross-flow threads keep the episode reproducible.
    run_episode(spec, build, tmp_path / "b", timeout_s=300, keep_raw=True)
    again = tmp_path / "b" / spec.episode_id() / "raw"
    for f in ("link_arrival", "link_depart", "link_drop", "link_opp", "link_stats", "tcp_obs"):
        assert (raw / f"{f}.bin").read_bytes() == (again / f"{f}.bin").read_bytes(), f
    trace = spec.path.trace.timestamps()
    assert_matches_oracle(rec, trace, 60)

    # Random drops are Bernoulli trials on every packet entering each link.
    for link_id, p in ((1, 0.01), (2, 0.005)):
        drp = rec.link(rec.link_drop, link_id)
        k = int((drp["reason"] == DROP_RANDOM).sum())
        n = k + rec.link(rec.link_arrival, link_id).size
        assert abs(k - n * p) < 5 * (n * p * (1 - p)) ** 0.5, (link_id, k, n)
    assert out.row["drops_random"] == int(rec.link_stats["drops_random"].sum()) > 0

    # Every flow crossed the bottleneck; only the observed one is observed.
    sports = set(rec.link(rec.link_depart, 1)["sport"].tolist())
    assert {labels.MAIN_SPORT, 6000, 6001} <= sports
    assert set(rec.tcp_obs["sport"].tolist()) == {labels.MAIN_SPORT}
    udp_sent, tcp_sent = out.row["cross_bytes_sent"]  # type: ignore[misc]
    assert udp_sent == 2500 * 1472  # one 1500-byte packet per 2.4 ms for 6 s
    assert tcp_sent > 0

    lab = out.tables["labels"]
    t_ms = (np.asarray(lab["t_ns"]).astype(np.int64) - out.row["base_ns"]) // 1_000_000  # type: ignore[operator]
    share = np.asarray(lab["cross_traffic_share"])
    assert np.all(np.asarray(lab["random_loss_rate"]) == 0.01)
    assert np.all((share >= 0) & (share <= 1))
    assert np.all(share[t_ms < 2000] == 0)
    # UDP alone: 5 Mbit/s of 1504-byte units on 20 Mbit/s, less 1% loss.
    udp_only = share[(t_ms >= 2200) & (t_ms < 6000)]
    assert udp_only.size and abs(float(np.median(udp_only)) - 0.25) < 0.02
    # Cubic adds its share (small: 1% random loss throttles it, not BBR).
    with_tcp = share[(t_ms >= 6500) & (t_ms < 8000)]
    assert float(np.median(with_tcp)) > float(np.median(udp_only)) + 0.01
    assert float(np.median(share[t_ms >= 8200])) > 0


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


def test_campaign_shards_and_tables(build: UmlBuild, tmp_path: Path) -> None:
    spec = CampaignSpec.parse(
        (Path(__file__).parents[1] / "examples" / "campaign-smoke.json").read_text()
    ).model_copy(update={"duration_ms": 3000})
    assert campaign.plan(spec, tmp_path) == 8
    for k in range(2):
        outcomes = list(campaign.run_shard(tmp_path, build, campaign.Shard(k, 2), workers=2))
        assert len(outcomes) == 4
    assert campaign.status(tmp_path)["episodes_done"] == 8
    for shard in (tmp_path / "shards").iterdir():
        eps = pq.read_table(shard / "episodes.parquet")
        assert set(eps["status"].to_pylist()) == {"ok"}
        obs, lab = pq.read_table(shard / "obs.parquet"), pq.read_table(shard / "labels.parquet")
        assert obs.num_rows == lab.num_rows == sum(eps["n_obs"].to_pylist())
        assert obs["t_ns"].equals(lab["t_ns"])


def test_cli_episode_and_mahak_query(build: UmlBuild, tmp_path: Path) -> None:
    spec = json.loads((Path(__file__).parents[1] / "examples" / "episode-mahak.json").read_text())
    spec["duration_ms"] = 2000
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    assert (
        main(
            [
                "episode",
                str(tmp_path / "spec.json"),
                str(tmp_path / "out"),
                "--build",
                str(build.root),
            ]
        )
        == 0
    )
    assert (tmp_path / "out" / "DONE").exists()
    args = [
        "mahak-query",
        "--build",
        str(build.root),
        "--cc",
        "cubic",
        "--metric",
        "first_power",
        "--buffer",
        "50",
        "--mrtt",
        "10",
        "--bw",
        "10",
        "--step",
        "1.0",
        "--scratch",
        str(tmp_path),
    ]
    assert main(args) == 0
