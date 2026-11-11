"""Step tables (steps.parquet) as arrays, split by path."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from expliccit_env.policy import OBS_DIM
from expliccit_env.train.lcfs import Labels

type F32 = npt.NDArray[np.float32]
type Split = Literal["train", "val", "test"]
# What keeps episodes in one split: "spec" is everything in the episode spec
# but cc and seed; "link" is the bottleneck link alone (capacity trace,
# delay, buffer, uplink trace), whatever the cross traffic or random loss.
# "row" keeps nothing together: each step falls on its own, as a random row
# split (expliCCit's) does, so held-out rows sit next to training rows of
# the same episode. Only to measure what that split overstates.
type SplitBy = Literal["spec", "link", "row"]


@dataclass(frozen=True, slots=True)
class Steps:
    """Rows of every recorded step, by episode id, then step."""

    obs: F32  # [n, 78], the step message as recorded (may hold inf and NaN)
    labels: dict[str, F32]  # label name -> [n], NaN where undefined (lcfs.Labels)
    action: F32  # [n], the action taken from the step; NaN on the last one
    reward: F32  # [n], Sage's reward at the step
    cwnd: F32  # [n], the sender's window (packets) when the step was read
    episode: npt.NDArray[np.int64]  # [n], index into episode_ids
    step: npt.NDArray[np.int64]  # [n], the step's number in its episode
    episode_ids: tuple[str, ...]
    split: npt.NDArray[np.str_]  # [n], "train" | "val" | "test"

    def rows(self, split: Split) -> npt.NDArray[np.bool_]:
        return self.split == split


_COLUMNS = frozenset({"episode_id", "step", "obs", "action", "reward", "cwnd"})


def _path_key(spec_json: str, by: SplitBy = "spec") -> str:
    spec = json.loads(spec_json)
    if by == "link":
        spec = {k: v for k, v in spec["path"].items() if k != "loss"}
    for k in ("cc", "seed"):
        spec.pop(k, None)
    return json.dumps(spec, sort_keys=True)


def link_key(spec_json: str) -> str:
    """An episode's link: its path without random loss, whatever the condition."""
    return _path_key(spec_json, "link")


def _split_of(key: str, seed: int, val: float, test: float) -> Split:
    u = int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "big") / 2**64
    return "test" if u < test else "val" if u < test + val else "train"


def _row_split(seed: int, val: float, test: float, n: int) -> npt.NDArray[np.str_]:
    u = np.random.default_rng(seed).random(n)
    return np.where(u < test, "test", np.where(u < test + val, "val", "train"))


def load(
    dirs: list[Path],
    *,
    seed: int,
    val: float,
    test: float,
    controllers: frozenset[str] | None = None,
    stride: int = 1,
    pairs: bool = False,
    split_by: SplitBy = "spec",
) -> Steps:
    """Every ok episode's steps (only those of ``controllers``, if given),
    keeping every ``stride``-th step of each episode and, with ``pairs``,
    the step after each, so net2's transitions survive the stride."""
    shards = sorted({p.parent for d in dirs for p in d.rglob("steps.parquet")})
    if not shards:
        raise FileNotFoundError(f"no steps.parquet under {[str(d) for d in dirs]}")
    split_of: dict[str, Split] = {}
    parts: list[dict[str, npt.NDArray[Any]]] = []
    index: dict[str, int] = {}  # episode id -> its index, in order of appearance
    for shard in shards:
        # pyarrow-stubs leave read_table partly untyped.
        eps = pq.read_table(  # pyright: ignore[reportUnknownMemberType]
            shard / "episodes.parquet",
            columns=["episode_id", "status", "spec_json", "controller"],
        )
        for eid, status, spec_json, ctl in zip(
            *(eps[c].to_pylist() for c in eps.column_names), strict=True
        ):
            if status == "ok" and (controllers is None or ctl in controllers):
                split_of[str(eid)] = _split_of(_path_key(str(spec_json), split_by), seed, val, test)
        # One row group at a time, keeping only the rows and columns used and
        # turning them into float32 arrays at once: a campaign's full table
        # (float64 observations) can exceed memory.
        ids = pa.array(list(split_of), pa.string())
        f = pq.ParquetFile(shard / "steps.parquet")
        columns = [c for c in f.schema_arrow.names if c in _COLUMNS or c.startswith("label_")]
        for g in range(f.num_row_groups):
            t = f.read_row_group(g, columns=columns)  # pyright: ignore[reportUnknownMemberType]
            ok = pc.is_in(t["episode_id"], value_set=ids)  # pyright: ignore[reportUnknownMemberType]
            keep = np.asarray(ok.to_numpy(zero_copy_only=False))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            if stride > 1:
                phase = np.asarray(t["step"].to_numpy(zero_copy_only=False)) % stride
                keep &= (phase == 0) | (pairs & (phase == 1))
            if keep.any():
                parts.append(_arrays(t.filter(pa.array(keep)), index))
    if not split_of:
        raise ValueError(f"no steps of controllers {sorted(controllers or ())}")
    if not parts:
        raise ValueError("no steps left after the stride")
    cat = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    # Rows in one order whatever the order of the shards' rows, which list
    # episodes as the workers finished them: by episode id, then step.
    episode_ids = tuple(sorted(index))
    pos = {e: i for i, e in enumerate(episode_ids)}
    rank = np.array([pos[e] for e in index], np.int64)  # appearance -> sorted
    order = np.lexsort((cat["step"], rank[cat["episode"]]))
    cat = {k: v[order] for k, v in cat.items()}
    episode = rank[cat.pop("episode")]
    split = (
        _row_split(seed, val, test, episode.size)
        if split_by == "row"
        else np.array([split_of[e] for e in episode_ids])[episode]
    )
    return Steps(
        obs=cat.pop("obs"),
        action=cat.pop("action"),
        reward=cat.pop("reward"),
        cwnd=cat.pop("cwnd"),
        episode=episode,
        step=cat.pop("step"),
        episode_ids=episode_ids,
        split=split,
        labels=Labels({k.removeprefix("label_"): v for k, v in cat.items()}),
    )


def _arrays(t: pa.Table, index: dict[str, int]) -> dict[str, npt.NDArray[Any]]:
    """A step table's rows as arrays (float32 but for step and episode);
    ``index`` gives each episode id its number, adding new ones."""
    n = t.num_rows

    def f32(name: str) -> F32:
        if name not in t.column_names:  # cwnd: tables before dataset version 6
            return np.full(n, np.nan, np.float32)
        return np.asarray(t[name].to_numpy(zero_copy_only=False), np.float32)

    obs_list = cast("pa.FixedSizeListArray[Any, Any]", t["obs"].combine_chunks())
    flat = cast("pa.Array[Any]", obs_list.flatten())  # pyright: ignore[reportUnknownMemberType]
    eids = [str(e) for e in t["episode_id"].to_pylist()]
    return {
        "obs": np.asarray(flat, np.float32).reshape(n, OBS_DIM),
        "action": f32("action"),
        "reward": f32("reward"),
        "cwnd": f32("cwnd"),
        "step": np.asarray(t["step"].to_numpy(zero_copy_only=False), np.int64),
        "episode": np.array([index.setdefault(e, len(index)) for e in eids], np.int64),
    } | {c: f32(c) for c in t.column_names if c.startswith("label_")}
