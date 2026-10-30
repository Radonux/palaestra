"""Command line: ``expliccit {episode,campaign,mahak-query,mahak} ...``."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import tempfile
from pathlib import Path

from pydantic import ValidationError

from expliccit_env import campaign, mahak
from expliccit_env.dataset import ShardWriter
from expliccit_env.episode import Build, EpisodeOk, run_episode
from expliccit_env.native import NativeBuild
from expliccit_env.spec import CampaignSpec, EpisodeSpec
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
    point = mahak.MahakPoint(a.buffer, a.mrtt, a.bw, a.step)
    spec = mahak.mahak_spec(point, a.cc, a.seed)
    outcome = run_episode(spec, _build(a), Path(a.scratch), timeout_s=a.timeout)
    match mahak.metric_value(outcome, a.metric):
        case mahak.Failure(reason):
            print(reason, json.dumps(outcome.row, default=str), file=sys.stderr)
            return 1
        case value:
            print(value)
            return 0


def _mahak(a: argparse.Namespace) -> int:
    """Mahak's mahak_online.py, with every query a Tier 1 episode."""
    space = mahak.MahakSpace.parse(a.buffer, a.mrtt, a.bw, a.change)
    out, build = Path(a.out), _build(a)
    out.mkdir(parents=True, exist_ok=True)
    # The oracle is a pure function of (point, seed), so a repeated point
    # reuses its result instead of writing a duplicate episode.
    seen: dict[mahak.MahakPoint, tuple[str | None, float | mahak.Failure]] = {}
    with (
        tempfile.TemporaryDirectory(prefix="expliccit-", dir=a.scratch) as tmp,
        ShardWriter(out / "shards" / campaign.Shard(0, 1).name()) as writer,
    ):

        def oracle(p: mahak.MahakPoint) -> float | mahak.Failure:
            if p not in seen:
                try:
                    spec = mahak.mahak_spec(p, a.cc, a.seed)
                except ValidationError as e:
                    seen[p] = (None, mahak.Failure(f"spec: {e}"))
                else:
                    outcome = run_episode(spec, build, Path(tmp), timeout_s=a.timeout)
                    writer.add(outcome)
                    seen[p] = (outcome.episode_id, mahak.metric_value(outcome, a.metric))
            return seen[p][1]

        m = mahak.explore(space, oracle, budget=a.budget, seed=a.seed)
    mahak.write_csv(m, a.metric, out / f"Mahak_{a.cc}.csv")
    with (out / "queries.jsonl").open("w") as f:
        for q in m.queries:
            row = {
                "grid_index": q.index,
                **dataclasses.asdict(q.point),
                "episode_id": seen[q.point][0],
                a.metric: None if isinstance(q.result, mahak.Failure) else q.result,
                "failure": q.result.reason if isinstance(q.result, mahak.Failure) else None,
            }
            f.write(json.dumps(row) + "\n")
    failed = sum(isinstance(q.result, mahak.Failure) for q in m.queries)
    print(
        f"{len(m.queries)} queries ({failed} failed) over {len(m.grid)} grid points; map in {out}"
    )
    return 0 if failed < len(m.queries) else 1


def _common(q: argparse.ArgumentParser) -> None:
    q.add_argument(
        "--build",
        required=True,
        help="build-uml.sh + build-guest.sh dir, or tier2-setup.sh dir",
    )
    q.add_argument(
        "--tier",
        choices=["uml", "native"],
        default="uml",
        help="uml: Tier 1, virtual time; native: Tier 2, real time, needs root",
    )
    q.add_argument("--timeout", type=float, default=1800.0, help="wall-clock seconds per episode")


def _mahak_query_args(m: argparse.ArgumentParser) -> None:
    m.add_argument("--cc", required=True, choices=["cubic", "bbr", "bbr3", "reno"])
    m.add_argument("--metric", required=True, choices=["utilization", "first_power", "delay"])
    m.add_argument("--buffer", type=int, required=True, help="packets")
    m.add_argument("--mrtt", type=int, required=True, help="Mahak mRTT: ONE-WAY delay, ms")
    m.add_argument("--bw", type=int, required=True, help="Mbit/s")
    m.add_argument("--step", type=float, required=True, help="capacity ratio at 12 s")
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--scratch", default="/tmp")
    _common(m)
    m.set_defaults(fn=_mahak_query)


def _mahak_args(d: argparse.ArgumentParser) -> None:
    # Mahak's own flag spellings are accepted too, so run_mahak.sh lines work.
    d.add_argument("--cc", "-cc", required=True, choices=["cubic", "bbr", "bbr3", "reno"])
    d.add_argument("--budget", "-budget", type=int, required=True, help="queries after the first")
    d.add_argument(
        "--metric", "-metric", required=True, choices=["utilization", "first_power", "delay"]
    )
    d.add_argument("--buffer", "-buffer", required=True, help="MIN,MAX,STEP packets")
    d.add_argument("--mrtt", "-mRTT", required=True, help="MIN,MAX,STEP one-way ms")
    d.add_argument("--bw", "-bw", required=True, help="MIN,MAX,STEP Mbit/s")
    d.add_argument("--change", "-change", required=True, help="MIN,MAX,STEP capacity ratio")
    d.add_argument("--seed", type=int, default=0, help="first point and episode seed")
    d.add_argument("--out", required=True)
    d.add_argument("--scratch", help="node-local scratch for raw records")
    _common(d)
    d.set_defaults(fn=_mahak)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="expliccit")
    sub = p.add_subparsers(required=True)

    e = sub.add_parser("episode", help="run one EpisodeSpec into a one-shard dataset")
    e.add_argument("spec")
    e.add_argument("out")
    e.add_argument("--keep-raw", action="store_true")
    _common(e)
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
    _common(cr)
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
    _mahak_query_args(m)
    d = sub.add_parser("mahak", help="Mahak's active-learning map (mahak_online.py) on Tier 1")
    _mahak_args(d)

    a = p.parse_args(argv)
    return int(a.fn(a))


if __name__ == "__main__":
    sys.exit(main())
