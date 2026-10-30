"""Command line: ``expliccit {episode,campaign,mahak-query} ...``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from expliccit_env import campaign
from expliccit_env.dataset import ShardWriter
from expliccit_env.episode import Build, EpisodeOk, run_episode
from expliccit_env.native import NativeBuild
from expliccit_env.spec import (
    CampaignSpec,
    DropTail,
    EpisodeSpec,
    MahakMrtt,
    MahakStep,
    PathSpec,
    StockCC,
)
from expliccit_env.uml import UmlBuild


def _build(a: argparse.Namespace) -> Build:
    return NativeBuild.at(Path(a.build)) if a.tier == "native" else UmlBuild.at(Path(a.build))


def _episode(a: argparse.Namespace) -> int:
    spec = EpisodeSpec.parse(Path(a.spec).read_text())
    out = Path(a.out)
    with ShardWriter(out) as w:
        outcome = run_episode(
            spec, _build(a), out / "work", keep_raw=a.keep_raw, timeout_s=a.timeout
        )
        w.add(outcome)
    print(json.dumps(outcome.row, indent=2, default=str))
    return 0 if isinstance(outcome, EpisodeOk) else 1


def _plan(a: argparse.Namespace) -> int:
    n = campaign.plan(CampaignSpec.parse(Path(a.spec).read_text()), Path(a.out))
    print(f"{n} episodes planned in {a.out}")
    return 0


def _run(a: argparse.Namespace) -> int:
    failed = 0
    for o in campaign.run_shard(
        Path(a.out),
        _build(a),
        campaign.Shard.parse(a.shard),
        workers=a.workers,
        scratch=Path(a.scratch) if a.scratch else None,
        keep_raw=a.keep_raw,
        timeout_s=a.timeout,
    ):
        ok = isinstance(o, EpisodeOk)
        failed += not ok
        print(o.episode_id, "ok" if ok else f"FAILED: {o.row.get('failure')}", flush=True)
    return 1 if failed else 0


def _slurm(a: argparse.Namespace) -> int:
    out = Path(a.out).resolve()
    script = campaign.slurm_script(
        out,
        UmlBuild.at(Path(a.build)),
        shards=a.shards,
        cpus_per_task=a.cpus_per_task,
        time=a.time,
        partition=a.partition,
        account=a.account,
        expliccit=Path(sys.argv[0]).resolve(),
    )
    (out / "slurm").mkdir(exist_ok=True)
    path = out / "slurm" / "array.sbatch"
    path.write_text(script)
    print(f"wrote {path}; submit with: sbatch {path}")
    return 0


def _status(a: argparse.Namespace) -> int:
    print(json.dumps(campaign.status(Path(a.out)), indent=2))
    return 0


def _mahak_query(a: argparse.Namespace) -> int:
    """Drop-in for Mahak's helper.oracle_query(X, metric, CC) on Tier 1."""
    spec = EpisodeSpec(
        path=PathSpec(
            trace=MahakStep(bw_mbps=a.bw, step=a.step),
            delay=MahakMrtt(mahak_compat_mrtt_ms=a.mrtt),
            buffer=DropTail(packets=a.buffer),
        ),
        cc=StockCC(name=a.cc),
        seed=a.seed,
    )
    outcome = run_episode(spec, _build(a), Path(a.scratch), timeout_s=a.timeout)
    value = outcome.row.get(a.metric if a.metric != "delay" else "mahak_delay")
    if not isinstance(outcome, EpisodeOk) or value is None:
        print(json.dumps(outcome.row, default=str), file=sys.stderr)
        return 1
    print(value)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="expliccit")
    sub = p.add_subparsers(required=True)

    def common(q: argparse.ArgumentParser) -> None:
        q.add_argument(
            "--build", required=True, help="UML build dir (build-uml.sh + build-guest.sh)"
        )
        q.add_argument(
            "--timeout", type=float, default=1800.0, help="wall-clock seconds per episode"
        )

    e = sub.add_parser("episode", help="run one EpisodeSpec into a one-shard dataset")
    e.add_argument("spec")
    e.add_argument("out")
    e.add_argument("--keep-raw", action="store_true")
    common(e)
    e.set_defaults(fn=_episode)

    c = sub.add_parser("campaign").add_subparsers(required=True)
    cp = c.add_parser("plan", help="expand a CampaignSpec into episodes.jsonl")
    cp.add_argument("spec")
    cp.add_argument("out")
    cp.set_defaults(fn=_plan)
    cr = c.add_parser("run", help="run one shard of a planned campaign")
    cr.add_argument("out")
    cr.add_argument("--shard", default="0/1", help="INDEX/COUNT")
    cr.add_argument("--workers", type=int, default=1, help="parallel UML guests (one core each)")
    cr.add_argument("--scratch", help="node-local scratch for raw records")
    cr.add_argument("--keep-raw", action="store_true")
    common(cr)
    cr.set_defaults(fn=_run)
    cs = c.add_parser("slurm", help="write an sbatch array script for a planned campaign")
    cs.add_argument("out")
    cs.add_argument("--shards", type=int, required=True)
    cs.add_argument("--cpus-per-task", type=int, default=8)
    cs.add_argument("--time", default="04:00:00")
    cs.add_argument("--partition")
    cs.add_argument("--account")
    cs.add_argument("--build", required=True)
    cs.set_defaults(fn=_slurm)
    st = c.add_parser("status")
    st.add_argument("out")
    st.set_defaults(fn=_status)

    m = sub.add_parser("mahak-query", help="Mahak oracle_query on Tier 1: print one metric")
    m.add_argument("--cc", required=True, choices=["cubic", "bbr", "reno"])
    m.add_argument("--metric", required=True, choices=["utilization", "first_power", "delay"])
    m.add_argument("--buffer", type=int, required=True, help="packets")
    m.add_argument("--mrtt", type=int, required=True, help="Mahak mRTT: ONE-WAY delay, ms")
    m.add_argument("--bw", type=int, required=True, help="Mbit/s")
    m.add_argument("--step", type=float, required=True, help="capacity ratio at 12 s")
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--scratch", default="/tmp")
    common(m)
    m.set_defaults(fn=_mahak_query)

    a = p.parse_args(argv)
    return int(a.fn(a))


if __name__ == "__main__":
    sys.exit(main())
