"""Campaigns: plan once, run shards anywhere (locally, Slurm array tasks, CI).

A shard is the episodes whose index is congruent to k modulo the shard
count, run by a bounded local process pool (one UML guest per worker
core) and written by one ShardWriter. Shards are independent and
idempotent, so a failed or preempted array task is simply rerun.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import shlex
import tempfile
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from expliccit_env.dataset import DATASET_VERSION, ShardWriter
from expliccit_env.episode import Build, EpisodeOutcome, run_episode
from expliccit_env.native import NativeBuild
from expliccit_env.spec import CampaignSpec, EpisodeSpec
from expliccit_env.uml import UmlBuild


def plan(spec: CampaignSpec, out: Path) -> int:
    """Write campaign.json and episodes.jsonl; return the episode count."""
    out.mkdir(parents=True, exist_ok=True)
    episodes = spec.episodes()
    (out / "campaign.json").write_text(
        json.dumps(
            {
                "dataset_version": DATASET_VERSION,
                "spec": spec.model_dump(mode="json"),
                "episodes": len(episodes),
            },
            indent=2,
        )
    )
    (out / "episodes.jsonl").write_text("".join(e.canonical_json() + "\n" for e in episodes))
    return len(episodes)


def load_episodes(out: Path) -> list[EpisodeSpec]:
    return [
        EpisodeSpec.parse(line)
        for line in (out / "episodes.jsonl").read_text().splitlines()
        if line
    ]


@dataclass(frozen=True, slots=True)
class Shard:
    index: int
    count: int

    @staticmethod
    def parse(text: str) -> Shard:
        i, n = (int(x) for x in text.split("/"))
        if not 0 <= i < n:
            raise ValueError(f"shard {text}: need 0 <= index < count")
        return Shard(i, n)

    def name(self) -> str:
        return f"{self.index:05d}-of-{self.count:05d}"


@dataclass(frozen=True, slots=True)
class _Job:
    spec: EpisodeSpec
    build: Path
    native: bool
    scratch: Path
    keep_raw: bool
    timeout_s: float


def _run(job: _Job) -> EpisodeOutcome:
    return run_episode(
        job.spec,
        UmlBuild.at(job.build),
        job.scratch,
        timeout_s=job.timeout_s,
        keep_raw=job.keep_raw,
    )


def run_shard(
    out: Path,
    build: Build,
    shard: Shard,
    *,
    workers: int,
    scratch: Path | None = None,
    keep_raw: bool = False,
    timeout_s: float = 1800.0,
) -> Iterator[EpisodeOutcome]:
    """Run one shard, yielding outcomes as they are written."""
    specs = load_episodes(out)[shard.index :: shard.count]
    with tempfile.TemporaryDirectory(prefix="expliccit-", dir=scratch) as tmp:
        work = out / "raw" if keep_raw else Path(tmp)
        jobs = [
            _Job(s, build.root, isinstance(build, NativeBuild), work, keep_raw, timeout_s)
            for s in specs
        ]
        with (
            ShardWriter(out / "shards" / shard.name()) as writer,
            ProcessPoolExecutor(
                max_workers=max(1, workers), mp_context=multiprocessing.get_context("forkserver")
            ) as pool,
        ):
            # map() keeps at most `workers` episodes in flight plus results
            # waiting for the single writer, and preserves episode order.
            for outcome in pool.map(_run, jobs):
                writer.add(outcome)
                yield outcome


def slurm_script(
    out: Path,
    build: UmlBuild,
    *,
    shards: int,
    cpus_per_task: int,
    time: str,
    partition: str | None,
    account: str | None,
    expliccit: Path,
) -> str:
    """An sbatch array script: task k runs shard k/shards. No root, no KVM."""
    q = shlex.quote
    opts = [
        f"#SBATCH --job-name=expliccit-{out.name}",
        f"#SBATCH --array=0-{shards - 1}",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --time={time}",
        f"#SBATCH --output={q(str(out / 'slurm'))}/%A_%a.log",
    ]
    if partition:
        opts.append(f"#SBATCH --partition={partition}")
    if account:
        opts.append(f"#SBATCH --account={account}")
    run = (
        f"{q(str(expliccit))} campaign run {q(str(out))} --build {q(str(build.root))} "
        f'--shard "${{SLURM_ARRAY_TASK_ID}}/{shards}" --workers "${{SLURM_CPUS_PER_TASK}}" '
        '--scratch "${TMPDIR:-/tmp}"'
    )
    return "\n".join(
        ["#!/usr/bin/env bash", *opts, "set -euo pipefail", ': "${SLURM_ARRAY_TASK_ID:?}"', run, ""]
    )


def status(out: Path) -> dict[str, int]:
    shards = sorted((out / "shards").glob("*")) if (out / "shards").exists() else []
    done = [s for s in shards if (s / "DONE").exists()]
    episodes = sum(json.loads((s / "DONE").read_text())["episodes"] for s in done)
    return {
        "planned": len(load_episodes(out)),
        "shards_done": len(done),
        "shards_incomplete": len(shards) - len(done),
        "episodes_done": episodes,
        "cpus": os.cpu_count() or 1,
    }
