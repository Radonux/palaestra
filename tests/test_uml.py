"""End-to-end checks in UML: real kernel, sch_mmlink, guest runner.

Needs a built UML tree (scripts/build-uml.sh, then build-guest.sh) named by
EXPLICCIT_UML_DIR; skipped otherwise.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from expliccit_env import campaign, labels
from expliccit_env import mahimahi as mm
from expliccit_env.cli import main
from expliccit_env.dataset import read_episode
from expliccit_env.episode import EpisodeOk, run_episode
from expliccit_env.guest_plan import GuestPlan, LinkPlan
from expliccit_env.records import DROP_QUEUE_FULL, DROP_RANDOM, Records, load_records
from expliccit_env.spec import CampaignSpec, EpisodeSpec
from expliccit_env.uml import UmlBuild, run_guest
from tests import onnx_policies
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
    # One flow: its drop fraction is what the qdisc's own counters say.
    down = stats[stats["link_id"] == labels.DOWNLINK]
    assert labels.drop_fraction(rec, 10**9) == int(down["drops_queue"][0]) / int(
        down["arrivals"][0]
    )
    assert np.isnan(labels.drop_fraction(rec, 0))


def test_a_trace_longer_than_a_netlink_attribute_holds(build: UmlBuild, tmp_path: Path) -> None:
    # 80 s of trace (two 40 s levels) is beyond the 64 KiB one netlink
    # attribute holds; the qdisc reads it from the plan's trace file.
    trace = mm.periodic_trace([12, 24], 40_000)
    counts = mm.dense_counts(trace)
    assert counts.size > 65_535
    link = LinkPlan(delay_ns=10_000_000, trace=counts, limit_pkts=50, limit_bytes=0)
    GuestPlan(duration_ms=3000, cc="cubic", downlink=link, uplink=link).write(tmp_path / "in")
    run = run_guest(build, tmp_path / "in", tmp_path / "out", seed=1, timeout_s=300)
    result = json.loads((tmp_path / "out" / "result.json").read_text())
    assert run.returncode == 0 and result["status"] == "ok", result
    assert_matches_oracle(load_records(tmp_path / "out"), trace, 50)


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
    # (docs/validation.md, F4); the link model itself uses exact instants.
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


def test_a_flow_can_fill_a_deep_buffer_on_a_long_fat_path(build: UmlBuild, tmp_path: Path) -> None:
    # 96 Mbit/s with a 160 ms RTT (a BDP of 1,280 packets) and a buffer of 4
    # BDP: a full queue is 640 ms. Reno's slow start fills it, which needs
    # 9.6 MB in flight. Under Linux's own limits in a 1 GiB guest (a 4 MB
    # send buffer) the queue never passed about 210 ms; the guest now sets
    # Sage's and ccBench's 64 MB limits (guest/src/main.rs, TCP_SYSCTLS).
    spec = EpisodeSpec.parse(
        json.dumps(
            {
                "path": {
                    "trace": {"kind": "periodic_levels", "levels_mbps": [96], "level_ms": 1000},
                    "delay": {"mahak_compat_mrtt_ms": 80},
                    "buffer": {"packets": 5120},
                },
                "cc": {"kind": "stock", "name": "reno"},
                "duration_ms": 4000,
                "seed": 3,
            }
        )
    )
    out = run_episode(spec, build, tmp_path / "work", timeout_s=600)
    assert isinstance(out, EpisodeOk), out.row
    delay_max = out.tables["flow_seconds"]["delay_max_ms"].to_numpy()
    assert np.nanmax(delay_max) > 600, delay_max


def test_a_window_far_beyond_the_path_is_bounded(build: UmlBuild, tmp_path: Path) -> None:
    # An agent that grows its window 2^0.05 a step on 50 Mbit/s, 10 ms RTT
    # and 26 packets never sees delay rise, so its window passes anything the
    # path holds. Sending costs no virtual time, so only the send buffer
    # stops it: at the setups' 64 MB the link sees 54 arrivals per departure,
    # most of them dropped; sized to the path (EpisodeSpec._send_buffer, 1.9
    # MB here) it sees about 9.
    raw = onnx_policies.constant(0.05)
    sha = hashlib.sha256(raw).hexdigest()
    pol = tmp_path / "grow.onnx"
    pol.write_bytes(raw)
    base = {
        "path": {
            "trace": {"kind": "periodic_levels", "levels_mbps": [50], "level_ms": 1000},
            "delay": {"mahak_compat_mrtt_ms": 5},
            "buffer": {"packets": 26},
        },
        "cc": {"kind": "agent", "mode": "hold", "policy": {"path": str(pol), "sha256": sha}},
        "duration_ms": 4000,
        "seed": 2,
    }
    ratio = {}
    for name, sysctls in (("path", {}), ("64mb", {"net.ipv4.tcp_wmem": "4096 65536 67108864"})):
        spec = EpisodeSpec.parse(json.dumps(base | {"sysctls": sysctls}))
        out = run_episode(spec, build, tmp_path / name, timeout_s=600, keep_raw=True)
        assert isinstance(out, EpisodeOk), out.row
        rec = load_records(tmp_path / name / spec.episode_id() / "raw")
        ratio[name] = rec.link(rec.link_arrival, 1).size / rec.link(rec.link_depart, 1).size
    assert ratio["path"] < 15 and ratio["64mb"] > 30, ratio


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

    # Per flow and second: the bytes add up to each flow's total, and the
    # delays to Mahimahi's mean delay over every departure.
    fs = out.tables["flow_seconds"].to_pydict()
    assert fs["second"][:10] == list(range(10)) and set(fs["flow"]) == {0, 1, 2}
    per_flow = [
        sum(b for f, b in zip(fs["flow"], fs["bytes"], strict=True) if f == k) for k in range(3)
    ]
    assert per_flow == out.row["flow_bytes"]
    udp_seconds = [
        s for f, s, n in zip(fs["flow"], fs["second"], fs["packets"], strict=True) if f == 1 and n
    ]
    assert udp_seconds[0] == 2 and udp_seconds[-1] in (7, 8)
    mean = sum(fs["delay_sum_ms"]) / sum(fs["packets"])
    assert abs(mean - out.row["mean_delay_ms"]) < 1e-9  # type: ignore[operator]
    # Per flow over the episode: the mean is the per-second sums' mean, and
    # the 95th percentile lies between the mean and the largest delay.
    for k in range(3):
        sums = [d for f, d in zip(fs["flow"], fs["delay_sum_ms"], strict=True) if f == k]
        n = sum(p for f, p in zip(fs["flow"], fs["packets"], strict=True) if f == k)
        top = max(
            d
            for f, d in zip(fs["flow"], fs["delay_max_ms"], strict=True)
            if f == k and not math.isnan(d)
        )
        mean_k = out.row["flow_delay_mean_ms"][k]  # type: ignore[index]
        assert abs(mean_k - sum(sums) / n) < 1e-9
        assert 0 <= out.row["flow_delay_p95_ms"][k] <= top  # type: ignore[index,operator]

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
        # A stored episode's records give its stored labels again, bit for bit.
        for eid in eps["episode_id"].to_pylist():
            ep = read_episode(shard, eid)
            again = labels.labels_table(ep.rec, ep.trace, ep.consts, ep.spec.duration_ms)
            stored = lab.filter(pc.equal(lab["episode_id"], eid)).drop_columns(["episode_id"])
            assert again.column_names == stored.column_names
            for name in again.column_names:  # bytes: NaN equals NaN
                assert again[name].to_numpy().tobytes() == stored[name].to_numpy().tobytes(), name


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


def test_cli_mahak_driver(build: UmlBuild, tmp_path: Path) -> None:
    # Mahak's flag spellings, a 2 x 2 x 1 x 2 grid, 3 queries after the first.
    args = ["mahak", "--build", str(build.root), "--out", str(tmp_path / "map"), "--seed", "1"]
    args += ["-cc", "bbr", "-budget", "3", "-metric", "first_power", "-buffer", "20,60,40"]
    args += ["-mRTT", "5,10,5", "-bw", "10,10,5", "-change", "0.5,1,0.5"]
    assert main(args) == 0
    out = tmp_path / "map"
    queries = [json.loads(line) for line in (out / "queries.jsonl").read_text().splitlines()]
    assert len(queries) == 4
    assert all(q["failure"] is None and q["first_power"] > 0 for q in queries)
    rows = (out / "Mahak_bbr.csv").read_text().splitlines()
    assert rows[0] == ",buffer_size,mRTT,BW,bw_step,first_power" and len(rows) == 1 + 8
    # Each distinct queried point is one ok episode in the dataset shard.
    eps = pq.read_table(out / "shards" / "00000-of-00001" / "episodes.parquet")
    assert sorted(eps["episode_id"].to_pylist()) == sorted({q["episode_id"] for q in queries})
    assert set(eps["status"].to_pylist()) == {"ok"}
