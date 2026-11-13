"""expliCCit's case study: its released policy, as deployed, on its 24 and
48 Mbit/s link, alone and with some of its beliefs replaced by the guest's
live truth, from the weights in third_party/expliccit-models.

The link: capacity alternating 24 and 48 Mbit/s every 7 s, one-way delay
40 ms, a 1000-packet buffer, 45 s. It has no randomness, so seeds give the
same run, but the policy's utilization flips between collapse and about
0.5 on small changes (a few TCP settings moved it from 0.47 to 0.04 in a
local check). So next to the exact link, each policy also runs on the eight
links around it, one-way delay 39 to 41 ms by buffer 980 to 1020 packets,
and the report gives the spread over all nine.

Usage: python experiments/expliccit_case.py plan OUT
       python experiments/expliccit_case.py report OUT
plan imports the eight policies, and two without some of the deployed
rounding (ROUNDING), into OUT/policies and plans OUT/c-case;
report appends the results to OUT/summary.md: per policy, utilization and
mean queueing delay (from flow_seconds), and for the policy as deployed,
its net1's median beliefs against the truth along its own runs.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from ccbench_report import capacity_mbps
from heldout import controllers

from expliccit_env import campaign
from expliccit_env.policy import TRUTH_COLUMNS
from expliccit_env.spec import EpisodeSpec

MODELS = Path(__file__).resolve().parents[1] / "third_party" / "expliccit-models"
SECONDS = 45
PATCHES = {  # name -> LCFs taken from the truth
    "none": (),
    "capacity": ("capacity_bps",),
    "busy fraction": ("busy_fraction",),
    "queue": ("queue_pkts",),
    "mean queueing delay": ("queue_delay_mean_ns",),
    "free buffer": ("free_buffer_pkts",),
    "busy fraction, queue, mean queueing delay and free buffer": (
        "busy_fraction",
        "queue_pkts",
        "queue_delay_mean_ns",
        "free_buffer_pkts",
    ),
    "all seven": TRUTH_COLUMNS[:7],
}
# The deployed controller rounds net1's first six LCFs to integers, busy
# fraction among them (a share, so 0 or 1): the policy with no LCF rounded,
# and with every rounding but busy fraction's. name -> LCFs left unrounded.
ROUNDING = {
    "none, no LCF rounded": (
        "queue_delay_mean_ns",
        "queue_pkts",
        "capacity_bps",
        "free_buffer_pkts",
        "busy_fraction",
        "one_way_delay_ns",
    ),
    "none, busy fraction not rounded": ("busy_fraction",),
}
LINKS = [(ow, buf) for ow in (39, 40, 41) for buf in (980, 1000, 1020)]  # (40, 1000) is expliCCit's


def path(ow: int, buf: int) -> dict[str, object]:
    return {
        "trace": {"kind": "periodic_levels", "levels_mbps": [24, 48], "level_ms": 7000},
        "delay": {"mahak_compat_mrtt_ms": ow},
        "buffer": {"packets": buf},
    }


def plan(out: Path) -> None:
    from expliccit_env.train import run  # noqa: PLC0415 (needs the train extra)

    policies = []
    for i, (name, truth) in enumerate(PATCHES.items()):
        d = out / "policies" / f"p{i}"
        run.import_expliccit(MODELS, d, [], as_deployed=True, truth=frozenset(truth))
        policies.append(f"{name}={d / 'policy.onnx'}")
    for i, (name, unrounded) in enumerate(ROUNDING.items()):
        d = out / "policies" / f"r{i}"
        run.import_expliccit(MODELS, d, [], as_deployed=True, unrounded=frozenset(unrounded))
        policies.append(f"{name}={d / 'policy.onnx'}")
    ccs, aliases = controllers("", policies)
    eps = [
        EpisodeSpec.model_validate(
            {
                "path": path(ow, buf),
                "cc": cc,
                "duration_ms": SECONDS * 1000,
                "seed": 1,
                "steps": True,
            }
        )
        for cc in ccs
        for ow, buf in LINKS
    ]
    campaign.write_episodes(out / "c-case", eps, {"name": "c-case"})
    (out / "aliases.json").write_text(json.dumps(aliases, indent=1) + "\n")
    print(f"c-case: {len(eps)} episodes")


def report(out: Path) -> None:
    from expliccit_env.train import data, fit  # noqa: PLC0415

    aliases: dict[str, str] = json.loads((out / "aliases.json").read_text())
    camp = out / "c-case"
    who: dict[str, str] = {}
    exact: set[str] = set()
    rows: dict[str, list[tuple[float, float, bool]]] = {}
    for shard in sorted(camp.glob("shards/*")):
        eps = pq.read_table(
            shard / "episodes.parquet", columns=["episode_id", "spec_json", "status"]
        )
        fs = pq.read_table(shard / "flow_seconds.parquet").to_pydict()
        for eid, sj, status in zip(*(eps[c].to_pylist() for c in eps.column_names), strict=True):
            if status != "ok":
                continue
            spec = EpisodeSpec.model_validate_json(sj)
            name = aliases[spec.cc.label]  # pyright: ignore[reportAttributeAccessIssue]
            who[eid] = name
            is_exact = spec.path.delay.one_way_ns == 40_000_000 and spec.path.buffer.packets == 1000
            if is_exact:
                exact.add(eid)
            mine = [
                i
                for i, (e, f) in enumerate(zip(fs["episode_id"], fs["flow"], strict=True))
                if e == eid and f == 0
            ]
            sent = sum(fs["bytes"][i] for i in mine) * 8 / SECONDS / 1e6
            delay = sum(fs["delay_sum_ms"][i] for i in mine) / max(
                sum(fs["packets"][i] for i in mine), 1
            )
            rows.setdefault(name, []).append(
                (sent / capacity_mbps(spec, 0, SECONDS), delay, is_exact)
            )
    lines = [
        "",
        "## expliCCit's case study",
        "",
        "| LCFs from the truth | utilization, expliCCit's link | median over 9 links | range"
        " | mean queueing delay (ms), expliCCit's link |",
        "|---|---|---|---|---|",
    ]
    for name in [*PATCHES, *ROUNDING]:
        r = rows.get(name, [])
        if len(r) < len(LINKS):
            lines.append(f"| {name} | {len(r)} of {len(LINKS)} runs ok | | | |")
            continue
        u = np.array([x[0] for x in r])
        ex = next(x for x in r if x[2])
        spread = f"{u.min():.3f} to {u.max():.3f}"
        lines.append(f"| {name} | {ex[0]:.3f} | {np.median(u):.3f} | {spread} | {ex[1]:.2f} |")
    # The deployed policy's beliefs along its own runs, against the truth.
    steps = data.load([camp], seed=0, val=0.0, test=0.0)
    ids = np.array(steps.episode_ids)[steps.episode]
    mine = np.array([who.get(e) == "none" and e in exact for e in ids])
    net1, _ = fit.load_expliccit(MODELS)
    belief = np.maximum(net1.predict(steps.obs[mine]), 0)  # clamped at 0, as deployed
    truth = fit.targets(steps, net1.lcfs)[mine]
    # The first two 7 s segments, by step count (steps are evenly spaced).
    step = steps.step[mine]
    early = step < step.max() * 14 / SECONDS
    lines += [
        "",
        "Medians along the deployed policy's run on expliCCit's link (report units):",
        "",
        "| LCF | released net1 | truth |",
        "|---|---|---|",
    ]
    cap = net1.lcfs.index("capacity_bps")
    for k, lcf in enumerate(net1.lcfs):
        if k == cap:
            for level in (24, 48):
                sel = early & (np.abs(truth[:, cap] - level) < 1)
                b, t = np.median(belief[sel, k]), np.median(truth[sel, k])
                lines.append(f"| {lcf}, first {level} Mbit/s segment | {b:.3g} | {t:.3g} |")
        else:
            ok = np.isfinite(truth[:, k])
            lines.append(
                f"| {lcf} | {np.median(belief[ok, k]):.3g} | {np.median(truth[ok, k]):.3g} |"
            )
    with (out / "summary.md").open("a") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    {"plan": plan, "report": report}[sys.argv[1]](Path(sys.argv[2]))
