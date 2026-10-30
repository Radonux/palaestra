"""Campaigns: plan once, run shards anywhere (locally, Slurm array tasks, CI).

A shard is the episodes whose index is congruent to k modulo the shard
count, run by a local process pool (one UML guest per worker core) with a
bounded submission window, and written by one ShardWriter. Shards are independent and
idempotent, so a failed or preempted array task is simply rerun.
"""

from __future__ import annotations

import itertools
import json
import multiprocessing
import os
import shlex
import tempfile
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from expliccit_env.dataset import DATASET_VERSION, TABLES, ShardWriter
from expliccit_env.episode import EpisodeOutcome, run_episode
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
    build: UmlBuild,
    shard: Shard,
    *,
    workers: int,
    scratch: Path | None = None,
    keep_raw: bool = False,
    timeout_s: float = 1800.0,
    tables: frozenset[str] = frozenset(TABLES),
) -> Iterator[EpisodeOutcome]:
    """Run one shard, yielding outcomes as they are written."""
    specs = load_episodes(out)[shard.index :: shard.count]
    with tempfile.TemporaryDirectory(prefix="expliccit-", dir=scratch) as tmp:
        work = out / "raw" if keep_raw else Path(tmp)
        jobs = [_Job(s, build.root, work, keep_raw, timeout_s) for s in specs]
        with (
            ShardWriter(out / "shards" / shard.name(), tables) as writer,
            ProcessPoolExecutor(
                max_workers=max(1, workers), mp_context=multiprocessing.get_context("forkserver")
            ) as pool,
        ):
            # At most 2 x workers episodes are submitted at once, and each is
            # written as soon as it finishes (rows carry episode_id, so order
            # does not matter). Memory stays bounded by the window, not by the
            # shard size, even when one episode is slow.
            todo = iter(jobs)
            window = 2 * max(1, workers)
            pending = {pool.submit(_run, j) for j in itertools.islice(todo, window)}
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    outcome = future.result()
                    writer.add(outcome)
                    yield outcome
                pending |= {
                    pool.submit(_run, j) for j in itertools.islice(todo, window - len(pending))
                }


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


_SUMMARY_METRICS: Final = ("utilization", "mean_delay_ms", "first_power", "goodput_mbps")


def summary(out: Path, *, resamples: int = 2000) -> dict[str, dict[str, object]]:
    """Per controller: episodes ok and failed, and each metric's mean with a
    95% bootstrap interval over paths (the episodes of one path, whatever
    their seed, are resampled together)."""
    shards = [s for s in sorted((out / "shards").glob("*")) if (s / "DONE").exists()]
    # pyarrow-stubs leave read_table partly untyped.
    t = pa.concat_tables(
        [pq.read_table(s / "episodes.parquet") for s in shards],  # pyright: ignore[reportUnknownMemberType]
        promote_options="permissive",
    )
    controller = [str(c) for c in t["controller"].to_pylist()]
    ok = np.array([s == "ok" for s in t["status"].to_pylist()])
    path = np.array([_path_key(str(j)) for j in t["spec_json"].to_pylist()])
    rng = np.random.default_rng(0)
    result: dict[str, dict[str, object]] = {}
    for ctl in sorted(set(controller)):
        mine = np.array([c == ctl for c in controller])
        entry: dict[str, object] = {"ok": int((mine & ok).sum()), "failed": int((mine & ~ok).sum())}
        for m in _SUMMARY_METRICS:
            if m not in t.column_names:
                continue
            col = np.asarray(t[m].to_numpy(zero_copy_only=False), dtype=np.float64)
            sel = mine & ok & np.isfinite(col)
            per_path = np.array([col[sel & (path == q)].mean() for q in np.unique(path[sel])])
            if per_path.size == 0:
                continue
            boot = per_path[rng.integers(0, per_path.size, (resamples, per_path.size))].mean(axis=1)
            entry[m] = {
                "mean": float(per_path.mean()),
                "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
                "paths": int(per_path.size),
            }
        result[ctl] = entry
    return result


def _path_key(spec_json: str) -> str:
    spec = json.loads(spec_json)
    for k in ("cc", "seed"):
        spec.pop(k, None)
    return json.dumps(spec, sort_keys=True)


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
