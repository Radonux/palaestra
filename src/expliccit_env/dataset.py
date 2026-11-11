"""Versioned dataset layout. One campaign directory holds:

    campaign.json            the CampaignSpec plus provenance of the plan
    episodes.jsonl           the expanded EpisodeSpecs, one per line, in order
    shards/<k>/              output of shard k (written by one worker process)
        episodes.parquet     one row per episode: parameters, status or failure,
                             metrics, provenance
        obs.parquet          sender observations (TCP_OBS), all episodes
        labels.parquet       ground-truth labels aligned 1:1 with obs rows
        link_events.parquet  bottleneck arrivals, drops, departures
        link_opp.parquet     processed delivery opportunities per ms
        agent_action.parquet actions accepted by tcp_agent (agent episodes)
        steps.parquet        Sage's steps with their labels, reward and action
                             (episodes that record steps)
        cross_steps.parquet  the same for every cross flow a policy drives,
                             with its index in the spec's cross list (flow)
        flow_seconds.parquet bytes and queueing delays per flow and second,
                             all episodes
        DONE                 written last; a shard without it is incomplete

``read_episode`` turns one stored episode back into the records its labels
were computed from, so a label added later can be computed for it, provided
the shard kept obs, link_events and link_opp.

Tables share ``episode_id``. Files are written as ``*.tmp`` and renamed on
close, so readers never see a half-written file; rerunning a shard replaces
it. Compression is zstd; each episode is its own row group.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Final, Self

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from expliccit_env import labels
from expliccit_env import mahimahi as mm
from expliccit_env import records as r
from expliccit_env.episode import (
    LINK_EVENTS,
    SKIP_COLUMNS,
    EpisodeOk,
    EpisodeOutcome,
    path_constants,
)
from expliccit_env.spec import EpisodeSpec

DATASET_VERSION: Final = 8
# "steps" exists only for episodes that record them (spec.steps or a policy),
# "cross_steps" only for episodes with an agent cross flow; "flow_seconds"
# (each flow's bytes and queueing delay per second) for every episode.
TABLES: Final = (
    "obs",
    "labels",
    "link_events",
    "link_opp",
    "agent_action",
    "steps",
    "cross_steps",
    "flow_seconds",
)


class ShardWriter:
    def __init__(
        self, directory: Path, tables: frozenset[str] = frozenset(TABLES), step_stride: int = 1
    ) -> None:
        """``tables``: which per-episode tables to keep (episodes.parquet is
        always written). Dropping link_events and link_opp saves most of the
        space, but labels can then never be recomputed for those episodes.
        ``step_stride``: keep only the steps whose number is a multiple of it,
        which shrinks estimator data (each step's message already summarizes
        the steps before it) but leaves no consecutive steps for net2."""
        if step_stride < 1:
            raise ValueError(f"step_stride {step_stride} must be at least 1")
        if not tables <= frozenset(TABLES):
            raise ValueError(f"unknown tables {sorted(tables - frozenset(TABLES))}")
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
        self.dir = directory
        self.tables = tables
        self.step_stride = step_stride
        self.rows: list[dict[str, object]] = []
        self.writers: dict[str, pq.ParquetWriter] = {}

    def __enter__(self) -> Self:
        return self

    def add(self, outcome: EpisodeOutcome) -> None:
        self.rows.append(outcome.row)
        if isinstance(outcome, EpisodeOk):
            for name, table in outcome.tables.items():
                assert name in TABLES, name
                if name not in self.tables:
                    continue
                if name in ("steps", "cross_steps") and self.step_stride > 1:
                    step = np.asarray(table["step"].to_numpy(), np.int64)
                    table = table.take(np.flatnonzero(step % self.step_stride == 0))  # noqa: PLW2901
                if name not in self.writers:
                    self.writers[name] = pq.ParquetWriter(
                        self.dir / f"{name}.parquet.tmp", table.schema, compression="zstd"
                    )
                self.writers[name].write_table(table, row_group_size=max(1, table.num_rows))

    def close(self) -> None:
        for name, w in self.writers.items():
            w.close()
            (self.dir / f"{name}.parquet.tmp").rename(self.dir / f"{name}.parquet")
        if self.rows:
            # Failed rows lack the metric columns: take the union of keys, in
            # first-seen order, and let missing values be null.
            keys = list(dict.fromkeys(k for row in self.rows for k in row))
            table = pa.Table.from_pydict({k: [row.get(k) for row in self.rows] for k in keys})
            # pyarrow-stubs leave write_table partly untyped.
            pq.write_table(table, self.dir / "episodes.parquet", compression="zstd")  # pyright: ignore[reportUnknownMemberType]
        (self.dir / "DONE").write_text(
            json.dumps({"dataset_version": DATASET_VERSION, "episodes": len(self.rows)})
        )

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc is None:
            self.close()
            return
        for w in self.writers.values():
            w.close()


@dataclass(frozen=True, slots=True)
class StoredEpisode:
    """A stored episode, as labels.labels_at takes it."""

    spec: EpisodeSpec
    rec: r.Records
    trace: mm.I64
    consts: labels.PathConstants


def _rows(shard: Path, table: str, episode_id: str) -> pa.Table:
    path = shard / f"{table}.parquet"
    if not path.exists():
        raise r.RecordError(f"{shard.name} did not keep the {table} table")
    # pyarrow-stubs leave read_table partly untyped.
    return pq.read_table(path, filters=pc.field("episode_id") == episode_id)  # pyright: ignore[reportUnknownMemberType]


def _records(t: pa.Table, rec_type: int, dtype: np.dtype[np.void]) -> r.Table:
    """Table rows as the guest's records; the fields a shard leaves out are
    the record type and length (set here) and padding (zero)."""
    out = np.zeros(t.num_rows, dtype)
    out["type"], out["len"] = rec_type, dtype.itemsize
    for name in dtype.names or ():
        if name in SKIP_COLUMNS:
            continue
        if name not in t.column_names:
            raise r.RecordError(f"stored records lack {name}: written under an older record ABI")
        out[name] = t[name].to_numpy(zero_copy_only=False)  # pyright: ignore[reportUnknownMemberType]
    return out


def _first_seq(tables: list[r.Table]) -> int:
    """The earliest of the stored (u32, possibly wrapped) sequence numbers:
    the one after the widest gap around the circle."""
    seq = np.sort(np.concatenate([t["seq"] for t in tables]).astype(np.int64))
    if not seq.size:
        return 0
    gaps = np.diff(seq, append=seq[0] + 2**32)
    return int(seq[(int(gaps.argmax()) + 1) % seq.size])


def read_episode(shard: Path, episode_id: str) -> StoredEpisode:
    """One successful episode's records, trace and path constants, from a
    shard that kept obs, link_events and link_opp. The records equal the
    guest's but for two the shard does not store: META (rebuilt, its
    sequence number the first stored record's) and LINK_STATS (empty)."""
    eps = pq.read_table(  # pyright: ignore[reportUnknownMemberType]
        shard / "episodes.parquet",
        columns=["episode_id", "status", "spec_json", "base_ns", "hz"],
        filters=pc.field("episode_id") == episode_id,
    ).to_pylist()
    if len(eps) != 1 or eps[0]["status"] != "ok":
        raise r.RecordError(f"{shard.name} holds no successful episode {episode_id}")
    row = eps[0]
    spec = EpisodeSpec.model_validate_json(row["spec_json"])
    dtype = {stem: (rec_type, d) for stem, rec_type, d in r.FILES}
    events = _rows(shard, "link_events", episode_id)
    kind = events["event"].to_pylist()
    links = {
        e: _records(
            events.filter(pa.array([k == e for k in kind], pa.bool_())), *dtype[f"link_{e}"]
        )
        for e in LINK_EVENTS
    }
    opp = _records(_rows(shard, "link_opp", episode_id), *dtype["link_opp"])
    obs = _records(_rows(shard, "obs", episode_id), *dtype["tcp_obs"])
    # agent_action is empty, and may be unwritten, when no agent ran.
    has_actions = (shard / "agent_action.parquet").exists()
    actions = (
        _records(_rows(shard, "agent_action", episode_id), *dtype["agent_action"])
        if has_actions
        else np.zeros(0, r.AGENT)
    )
    meta = np.zeros(1, r.META)
    meta["type"], meta["len"] = dtype["meta"][0], r.META.itemsize
    meta["seq"] = _first_seq([*links.values(), opp, obs, actions])
    meta["magic"], meta["abi"], meta["hz"] = r.MAGIC, r.ABI, row["hz"]
    rec = r.Records(
        meta=meta,
        link_arrival=links["arrival"],
        link_drop=links["drop"],
        link_depart=links["depart"],
        link_opp=opp,
        link_stats=np.zeros(0, r.STATS),
        tcp_obs=obs,
        agent_action=actions,
    )
    return StoredEpisode(
        spec, rec, spec.path.trace.timestamps(), path_constants(spec, int(row["base_ns"]))
    )
