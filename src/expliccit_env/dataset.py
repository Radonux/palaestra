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
        DONE                 written last; a shard without it is incomplete

Tables share ``episode_id``. Files are written as ``*.tmp`` and renamed on
close, so readers never see a half-written file; rerunning a shard replaces
it. Compression is zstd; each episode is its own row group.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import TracebackType
from typing import Final, Self

import pyarrow as pa
import pyarrow.parquet as pq

from expliccit_env.episode import EpisodeOk, EpisodeOutcome

DATASET_VERSION: Final = 4
# "steps" exists only for episodes that record them (spec.steps or a policy).
TABLES: Final = ("obs", "labels", "link_events", "link_opp", "agent_action", "steps")


class ShardWriter:
    def __init__(self, directory: Path) -> None:
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
        self.dir = directory
        self.rows: list[dict[str, object]] = []
        self.writers: dict[str, pq.ParquetWriter] = {}

    def __enter__(self) -> Self:
        return self

    def add(self, outcome: EpisodeOutcome) -> None:
        self.rows.append(outcome.row)
        if isinstance(outcome, EpisodeOk):
            for name, table in outcome.tables.items():
                assert name in TABLES, name
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
