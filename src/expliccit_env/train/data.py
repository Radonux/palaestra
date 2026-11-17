"""Step tables (steps.parquet) as arrays, split by path."""

from __future__ import annotations

import dataclasses
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

from expliccit_env.mahimahi import PACKET_SIZE
from expliccit_env.policy import OBS_DIM
from expliccit_env.spec import UdpCross
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
    t_ms: F32  # [n], the step's trace time: a step lasts 10 ms or until an ACK arrives
    episode_ids: tuple[str, ...]
    split: npt.NDArray[np.str_]  # [n], "train" | "val" | "test"

    def rows(self, split: Split) -> npt.NDArray[np.bool_]:
        return self.split == split


_COLUMNS = frozenset({"episode_id", "step", "t_ns", "obs", "action", "reward", "cwnd"})


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
    base_ns: dict[str, int] = {}  # episode id -> its trace's ms 0
    parts: list[dict[str, npt.NDArray[Any]]] = []
    index: dict[str, int] = {}  # episode id -> its index, in order of appearance
    for shard in shards:
        # pyarrow-stubs leave read_table partly untyped.
        eps = pq.read_table(  # pyright: ignore[reportUnknownMemberType]
            shard / "episodes.parquet",
            columns=["episode_id", "status", "spec_json", "controller", "base_ns"],
        )
        for eid, status, spec_json, ctl, base in zip(
            *(eps[c].to_pylist() for c in eps.column_names), strict=True
        ):
            if status == "ok" and (controllers is None or ctl in controllers):
                split_of[str(eid)] = _split_of(_path_key(str(spec_json), split_by), seed, val, test)
                base_ns[str(eid)] = int(base)
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
    base = np.array([base_ns[e] for e in episode_ids], np.int64)[episode]
    t_ms = ((cat.pop("t_ns") - base) / 1e6).astype(np.float32)
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
        t_ms=t_ms,
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
        "t_ns": np.asarray(t["t_ns"].to_numpy(zero_copy_only=False), np.int64),
        "episode": np.array([index.setdefault(e, len(index)) for e in eids], np.int64),
    } | {c: f32(c) for c in t.column_names if c.startswith("label_")}


def specs(dirs: list[Path]) -> dict[str, dict[str, Any]]:
    """Each episode's spec, by episode id."""
    out: dict[str, dict[str, Any]] = {}
    for f in sorted({p for d in dirs for p in d.rglob("episodes.parquet")}):
        t = pq.read_table(f, columns=["episode_id", "spec_json"])  # pyright: ignore[reportUnknownMemberType]
        ids, js = t["episode_id"].to_pylist(), t["spec_json"].to_pylist()
        out |= {str(e): json.loads(str(j)) for e, j in zip(ids, js, strict=True)}
    return out


def less_udp(steps: Steps, spec_of: dict[str, dict[str, Any]], floor: float = 1 / 16) -> Steps:
    """``steps`` whose capacity label is the capacity less the constant bit
    rate UDP traffic active at each step, and at least ``floor`` times the
    capacity. Beside a constant rate a flow cannot tell a slower link from
    a busy one, so this is the part of the capacity its measurements can
    show. ponytail: the label's first window only, the only one net1 reads."""
    cap = steps.labels["capacity_bps"].astype(np.float64)
    udp = np.zeros_like(cap)
    order = np.argsort(steps.episode, kind="stable")
    cuts = np.searchsorted(steps.episode[order], np.arange(len(steps.episode_ids) + 1))
    for i, e in enumerate(steps.episode_ids):
        cross: list[dict[str, Any]] = spec_of[e].get("cross") or []
        flows = [UdpCross.model_validate(c) for c in cross if c.get("kind") == "udp_cbr"]
        if not flows:
            continue
        rows = order[cuts[i] : cuts[i + 1]]
        t = steps.t_ms[rows]
        for c in flows:
            on = (t >= c.start_ms) & (t < (c.stop_ms or np.inf))
            # A 1500-byte IP packet takes one 1504-byte delivery opportunity.
            udp[rows[on]] += c.rate_mbps * 1e6 * PACKET_SIZE / 1500
    left = np.maximum(cap - udp, floor * cap).astype(np.float32)
    return dataclasses.replace(steps, labels=Labels(dict(steps.labels) | {"capacity_bps": left}))
