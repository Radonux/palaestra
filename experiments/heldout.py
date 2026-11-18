"""Write a campaign on the held-out links of examples/campaign-net1-varied.json.

The links are split by link (the path without its loss), seed 1, 15%
validation and 15% test, as net1 v3's training data was. The campaign runs
the given policies and stock controllers on one split's links under one
family of conditions. Episode seeds are drawn in order from the campaign
seed, so changing the controller list changes every episode's seed; the
pairing is by link and condition.

Families: "training" is that campaign's three conditions (clean, 1% loss,
3 Mbit/s UDP plus a Cubic flow from 8 s). The others are conditions it never
had: "new" (a Cubic flow from 0 s, a BBRv1 flow from 4 s, Cubic flows from
2 s and 10 s, 0.1% and 3% random loss, 1% loss on the ACK path),
"cubic-first" (a Cubic flow from 0 s alone), "deep-long" (the training
conditions on the same links with a 2.5 times longer one-way delay, at most
80 ms, and a buffer of 4 to 16 BDP drawn with seed 31), and
"held-competitors" (a BBRv3 flow from 0 s, a Westwood flow from 0 s: the
competitors net1 v4's training data held out), "stock-all" (one flow from
0 s of each other stock controller, STOCK_ALL), and "dynamic", for 120 s
episodes, where competitors arrive and leave mid-episode (DYNAMIC), so that
a controller must notice each change and the others' reaction to it.
Two families hold links where the flow cannot keep the link busy, so that
what it does not know has value: "info", the links' own capacity under a
Cubic or a BBRv1 flow that leaves at 10 s, UDP at half the link's mean
capacity on for 2 s and off for 2 s, and Cubic flows on for 3 s and off for
3 s (INFO); and "rise", the flow alone on each link with its capacity
alternating between half and twice its first level every 2 s, a fourfold
rise every 4 s. "cellular" replays real cellular traces (Mahimahi's,
third_party/mahimahi/traces: AT&T, T-Mobile and Verizon, LTE, UMTS and
EVDO, measured while driving or still): each link keeps its one-way delay
and its buffer in BDPs, and takes one of the eight traces from an offset
that differs from link to link, with the matching uplink trace for the
ACKs; alone and against a Cubic flow from 0 s. The slices are written
under OUT_DIR/traces. "case-studies" is not a split's links but the two
links of expliCCit's case studies, alone: 24 and 48 Mbit/s alternating
every 7 s with a 40 ms one-way delay and a 1000-packet buffer, and 24
Mbit/s halving to 12 at 40 s with 20 ms and 300 packets. "aurora-range"
is not a split's links either: 27 links drawn as PCC-RL's simulator draws
Aurora's training links (1.2 to 6 Mbit/s, 50 to 500 ms one way, buffers of
2 to 2981 packets), clean and with 1% and 3% random loss, so a controller
trained there can be seen on its own ground.

Usage: python experiments/heldout.py OUT_DIR --split test --duration-ms 60000
           --seed 13 [--family training] --stock all --policy NAME=PATH[@MODE] ...

Writes OUT_DIR/campaign.json and OUT_DIR/aliases.json (agent:SHA12 -> NAME,
for `expliccit score --alias`).
"""

import argparse
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from expliccit_env.spec import (
    STOCK_NAMES,
    AgentCC,
    CampaignSpec,
    EpisodeSpec,
    StockCC,
    StockName,
)
from expliccit_env.train import data

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples/campaign-net1-varied.json"
HELD_COMPETITORS: list[dict[str, Any]] = [
    {"cross": [{"kind": "tcp", "cc": cc, "start_ms": 0}]} for cc in ("bbr3", "westwood")
]
# One flow from 0 s of each stock controller the held-out family does not
# hold: the variety a classifier of "another TCP flow" needs to learn the
# class rather than two of its members (train/traffic.py).
STOCK_ALL: list[dict[str, Any]] = [
    {"cross": [{"kind": "tcp", "cc": cc, "start_ms": 0}]}
    for cc in STOCK_NAMES
    if cc not in {c["cross"][0]["cc"] for c in HELD_COMPETITORS}
]
NEW: list[dict[str, Any]] = [
    {"cross": [{"kind": "tcp", "cc": "cubic", "start_ms": 0}]},
    {"cross": [{"kind": "tcp", "cc": "bbr", "start_ms": 4000}]},
    {"cross": [{"kind": "tcp", "cc": "cubic", "start_ms": t} for t in (2000, 10000)]},
    {"loss": {"downlink": 0.001}},
    {"loss": {"downlink": 0.03}},
    {"loss": {"uplink": 0.01}},
]


def split_links(split: str) -> dict[str, dict[str, Any]]:
    """The links (trace, delay, buffer) of one split of SOURCE, by link key,
    in campaign order. The split hashes each planned episode's canonical
    spec, defaults included."""
    paths: dict[str, dict[str, Any]] = {}
    for e in CampaignSpec.parse(SOURCE.read_text()).episodes():
        sj = e.canonical_json()
        if data._split_of(data.link_key(sj), 1, 0.15, 0.15) == split:  # pyright: ignore[reportPrivateUsage]
            path = json.loads(sj)["path"]
            paths.setdefault(data.link_key(sj), {k: path[k] for k in ("trace", "delay", "buffer")})
    return paths


def keep_only(
    eps: list[EpisodeSpec],
    only: str,
    aliases: dict[str, str],
    tested: Callable[[EpisodeSpec], StockName | StockCC | AgentCC] = lambda e: e.cc,
) -> list[EpisodeSpec]:
    """The episodes whose controller under test (``tested``; the observed
    flow's by default) is named in ``only`` (comma-separated stock names
    and policy NAMEs; empty keeps all). Planning every controller and
    keeping some leaves the random stream, so each kept episode has the
    seed, walk and start times a plan of all of them gives it."""
    names = {n for n in only.split(",") if n}
    if not names:
        return eps

    def name(cc: StockName | StockCC | AgentCC) -> str:
        if isinstance(cc, AgentCC):
            return aliases[cc.label]
        return cc if isinstance(cc, str) else cc.name

    return [e for e in eps if name(tested(e)) in names]


def controllers(stock: str, policies: list[str]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """The stock controllers (comma-separated, or all) and the policies
    (NAME=PATH, run as "hold" agents, or NAME=PATH@MODE, or
    NAME=PATH@MODE:PERIOD_MS), and each agent's label (spec.AgentCC.label)
    -> NAME."""
    names = list(STOCK_NAMES) if stock == "all" else [n for n in stock.split(",") if n]
    ccs: list[dict[str, Any]] = [{"kind": "stock", "name": n} for n in names]
    aliases: dict[str, str] = {}
    for name, rest in (x.split("=", 1) for x in policies):
        path, _, how = rest.partition("@")
        mode, _, period = how.partition(":")
        sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        cc: dict[str, Any] = {
            "kind": "agent",
            "mode": mode or "hold",
            "policy": {"path": str(Path(path).resolve()), "sha256": sha},
        }
        if period:
            cc["period_ms"] = int(period)
        ccs.append(cc)
        aliases[AgentCC.model_validate(cc).label] = name
    return ccs, aliases


def deep_long(paths: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each link with a 2.5 times longer one-way delay (at most 80 ms) and a
    buffer of 4 to 16 BDP (log-uniform, seed 31, clipped to 20 to 8000)."""
    rng = np.random.default_rng(31)
    out = []
    for p in paths:
        ow = min(int(p["delay"]["mahak_compat_mrtt_ms"] * 2.5), 80)
        bdp = p["trace"]["levels_mbps"][0] * 1e6 * 2 * ow / 1e3 / 12000
        buf = int(np.clip(bdp * np.exp(rng.uniform(np.log(4), np.log(16))), 20, 8000))
        out.append(p | {"delay": {"mahak_compat_mrtt_ms": ow}, "buffer": {"packets": buf}})
    return out


# For 120 s episodes: arrivals and departures throughout, from the common
# controllers and one delay-based one (Vegas), none of them held out.
DYNAMIC: list[dict[str, Any]] = [
    {"cross": [{"kind": "tcp", "cc": "cubic", "start_ms": 30000, "stop_ms": 90000}]},
    {"cross": [{"kind": "tcp", "cc": "bbr", "start_ms": 0, "stop_ms": 60000}]},
    {"cross": [{"kind": "tcp", "cc": "reno", "start_ms": 20000}]},
    {"cross": [{"kind": "tcp", "cc": "cubic", "start_ms": t} for t in (0, 60000)]},
    {"cross": [{"kind": "tcp", "cc": "vegas", "start_ms": 15000}]},
    {
        "cross": [
            {"kind": "tcp", "cc": "bbr", "start_ms": 40000},
            {"kind": "tcp", "cc": "cubic", "start_ms": 40000, "stop_ms": 100000},
        ]
    },
]


def info(duration_ms: int) -> list[dict[str, Any]]:
    """Family "info": competitors that leave and senders that switch on and
    off, all through the episode."""
    leave = min(10000, duration_ms // 2)
    return [
        {"cross": [{"kind": "tcp", "cc": "cubic", "start_ms": 0, "stop_ms": leave}]},
        {"cross": [{"kind": "tcp", "cc": "bbr", "start_ms": 0, "stop_ms": leave}]},
        {
            "cross": [
                {"kind": "udp_share", "share": 0.5, "start_ms": t, "stop_ms": t + 2000}
                for t in range(2000, duration_ms - 1999, 4000)
            ]
        },
        {
            "cross": [
                {"kind": "tcp", "cc": "cubic", "start_ms": t, "stop_ms": t + 3000}
                for t in range(3000, duration_ms - 2999, 6000)
            ]
        },
    ]


def rise(paths: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Family "rise": each link's capacity alternates between half and
    twice its first level every 2 s."""
    out = []
    for p in paths:
        c = p["trace"]["levels_mbps"][0]
        trace = {"kind": "periodic_levels", "levels_mbps": [c / 2, 2 * c], "level_ms": 2000}
        out.append(p | {"trace": trace})
    return out


CELLULAR_TRACES: tuple[str, ...] = (
    "ATT-LTE-driving",
    "ATT-LTE-driving-2016",
    "TMobile-LTE-driving",
    "TMobile-LTE-short",
    "TMobile-UMTS-driving",
    "Verizon-EVDO-driving",
    "Verizon-LTE-driving",
    "Verizon-LTE-short",
)
CELLULAR_DIR = ROOT / "third_party/mahimahi/traces"
CELLULAR: list[dict[str, Any]] = [{}, {"cross": [{"kind": "tcp", "cc": "cubic", "start_ms": 0}]}]


def trace_slice(src: Path, start_ms: int, length_ms: int, dest: Path) -> dict[str, Any]:
    """A Mahimahi trace from start_ms on, length_ms long, as a pinned trace
    file: each line is the ms of one delivery opportunity, shifted so the
    slice starts at 0 (Mahimahi replays a trace from its start when it ends)."""
    ts = np.array([int(x) for x in src.read_text().split()], np.int64)
    kept = ts[(ts > start_ms) & (ts <= start_ms + length_ms)] - start_ms
    if kept.size < 2:
        raise ValueError(f"{src.name} from {start_ms} ms: fewer than two opportunities")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("".join(f"{t}\n" for t in kept))
    sha = hashlib.sha256(dest.read_bytes()).hexdigest()
    return {"kind": "mahimahi_file", "path": str(dest.resolve()), "sha256": sha}


def mean_mbps(trace: Path) -> float:
    ts = [int(x) for x in trace.read_text().split()]
    return 12.0 * len(ts) / ts[-1]  # one 1500-byte packet per entry, in ms


def cellular(paths: list[dict[str, Any]], duration_ms: int, out: Path) -> list[dict[str, Any]]:
    """Family "cellular": link i takes trace i mod 8 from an offset of
    (i // 8) * 37 s (modulo what the trace leaves), its delay, and its
    buffer in BDPs at the trace's mean capacity."""
    links = []
    for i, p in enumerate(paths):
        name = CELLULAR_TRACES[i % len(CELLULAR_TRACES)]
        down, up = CELLULAR_DIR / f"{name}.down", CELLULAR_DIR / f"{name}.up"
        span = int(down.read_text().split()[-1]) - duration_ms
        start = ((i // len(CELLULAR_TRACES)) * 37_000) % max(span, 1)
        ow = p["delay"]["mahak_compat_mrtt_ms"]
        bdp = p["trace"]["levels_mbps"][0] * 1e6 * 2 * ow / 1e3 / 12000
        in_bdp = p["buffer"]["packets"] / bdp
        cap = mean_mbps(down)
        packets = int(np.clip(round(in_bdp * cap * 1e6 * 2 * ow / 1e3 / 12000), 10, 8000))
        sliced = out / "traces"
        links.append(
            p
            | {
                "trace": trace_slice(
                    down, start, duration_ms + 5000, sliced / f"{name}-{start}.down"
                ),
                "uplink_trace": trace_slice(
                    up, start, duration_ms + 5000, sliced / f"{name}-{start}.up"
                ),
                "buffer": {"packets": packets},
            }
        )
    return links


CASE_STUDIES: list[dict[str, Any]] = [
    {
        "trace": {"kind": "periodic_levels", "levels_mbps": [24, 48], "level_ms": 7000},
        "delay": {"mahak_compat_mrtt_ms": 40},
        "buffer": {"packets": 1000},
    },
    {
        "trace": {"kind": "periodic_levels", "levels_mbps": [24, 12], "level_ms": 40000},
        "delay": {"mahak_compat_mrtt_ms": 20},
        "buffer": {"packets": 300},
    },
]


def aurora_range(n: int = 27) -> list[dict[str, Any]]:
    """n links drawn as PCC-RL's simulator (64bea6e, network_sim.py) draws
    Aurora's training links: a constant 100 to 500 packets of 1500 bytes a
    second (1.2 to 6 Mbit/s), 50 to 500 ms one way, and a buffer of
    1 + int(e^U(0, 8)) packets, all uniform (seed 31). Its random loss, up
    to 5%, comes from the conditions (AURORA_CONDITIONS)."""
    rng = np.random.default_rng(31)
    out = []
    for _ in range(n):
        pps = rng.uniform(100, 500)
        out.append(
            {
                "trace": {
                    "kind": "periodic_levels",
                    "levels_mbps": [round(pps * 0.012, 3)],
                    "level_ms": 1000,
                },
                "delay": {"mahak_compat_mrtt_ms": int(rng.uniform(50, 500))},
                "buffer": {"packets": 1 + int(np.exp(rng.uniform(0, 8)))},
            }
        )
    return out


AURORA_CONDITIONS: list[dict[str, Any]] = [
    {},
    {"loss": {"downlink": 0.01}},
    {"loss": {"downlink": 0.03}},
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--split", choices=["train", "val", "test"], required=True)
    p.add_argument("--duration-ms", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument(
        "--family",
        default="training",
        choices=[
            "training",
            "new",
            "cubic-first",
            "deep-long",
            "held-competitors",
            "stock-all",
            "dynamic",
            "info",
            "rise",
            "cellular",
            "case-studies",
            "aurora-range",
        ],
    )
    p.add_argument("--stock", default="all", help="comma-separated stock controllers, or all")
    p.add_argument(
        "--policy",
        action="append",
        default=[],
        help="NAME=PATH[@MODE[:PERIOD_MS]] of an ONNX policy (mode hold, pure, rate or cubic)",
    )
    a = p.parse_args()
    source = json.loads(SOURCE.read_text())
    paths = split_links(a.split)
    ccs, aliases = controllers(a.stock, a.policy)
    family = a.family
    links = list(paths.values())
    conditions = source["conditions"]
    if family == "deep-long":
        links = deep_long(links)
    elif family == "held-competitors":
        conditions = HELD_COMPETITORS
    elif family == "stock-all":
        conditions = STOCK_ALL
    elif family == "dynamic":
        conditions = DYNAMIC
    elif family == "info":
        conditions = info(a.duration_ms)
    elif family == "rise":
        links, conditions = rise(links), [{}]
    elif family == "cellular":
        links, conditions = cellular(links, a.duration_ms, a.out), CELLULAR
    elif family == "case-studies":
        links, conditions = CASE_STUDIES, [{}]
    elif family == "aurora-range":
        links, conditions = aurora_range(), AURORA_CONDITIONS
    elif family != "training":
        conditions = NEW if family == "new" else NEW[:1]
    campaign = {
        "name": f"{a.split}-{family}-{a.duration_ms // 1000}s",
        "ccs": ccs,
        "sweep": {"kind": "paths", "paths": links},
        "conditions": conditions,
        "duration_ms": a.duration_ms,
        "seed": a.seed,
        "steps": True,
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "campaign.json").write_text(json.dumps(campaign, indent=1) + "\n")
    (a.out / "aliases.json").write_text(json.dumps(aliases, indent=1) + "\n")
    print(f"{len(links)} {a.split} links, {len(ccs)} controllers, {len(conditions)} conditions")


if __name__ == "__main__":
    main()
