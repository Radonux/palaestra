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

type F32 = npt.NDArray[np.float32]
type Split = Literal["train", "val", "test"]


@dataclass(frozen=True, slots=True)
class Steps:
    """Rows of every recorded step, in episode then step order."""

    obs: F32  # [n, 77], Sage's message as recorded (may hold inf and NaN)
    labels: dict[str, F32]  # label name -> [n], NaN where undefined
    action: F32  # [n], the action taken from the step; NaN on the last one
    reward: F32  # [n], Sage's reward at the step
    episode: npt.NDArray[np.int64]  # [n], index into episode_ids
    episode_ids: tuple[str, ...]
    split: npt.NDArray[np.str_]  # [n], "train" | "val" | "test"

    def rows(self, split: Split) -> npt.NDArray[np.bool_]:
        return self.split == split


def _path_key(spec_json: str) -> str:
    spec = json.loads(spec_json)
    for k in ("cc", "seed"):
        spec.pop(k, None)
    return json.dumps(spec, sort_keys=True)


def _split_of(key: str, seed: int, val: float, test: float) -> Split:
    u = int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "big") / 2**64
    return "test" if u < test else "val" if u < test + val else "train"


def load(
    dirs: list[Path],
    *,
    seed: int,
    val: float,
    test: float,
    controllers: frozenset[str] | None = None,
) -> Steps:
    """Every ok episode's steps (only those of ``controllers``, if given)."""
    shards = sorted({p.parent for d in dirs for p in d.rglob("steps.parquet")})
    if not shards:
        raise FileNotFoundError(f"no steps.parquet under {[str(d) for d in dirs]}")
    tables: list[pa.Table] = []
    split_of: dict[str, Split] = {}
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
                split_of[str(eid)] = _split_of(_path_key(str(spec_json)), seed, val, test)
        tables.append(pq.read_table(shard / "steps.parquet"))  # pyright: ignore[reportUnknownMemberType]
    if not split_of:
        raise ValueError(f"no steps of controllers {sorted(controllers or ())}")
    t = pa.concat_tables(tables).combine_chunks()
    keep = pc.is_in(t["episode_id"], value_set=pa.array(list(split_of), pa.string()))  # pyright: ignore[reportUnknownMemberType]
    t = t.filter(keep)
    eid_col = [str(e) for e in t["episode_id"].to_pylist()]
    episode_ids = tuple(dict.fromkeys(eid_col))
    index = {e: i for i, e in enumerate(episode_ids)}
    n = t.num_rows
    obs_list = cast("pa.FixedSizeListArray[Any, Any]", t["obs"].combine_chunks())
    flat = cast("pa.Array[Any]", obs_list.flatten())  # pyright: ignore[reportUnknownMemberType]
    obs = np.asarray(flat, np.float32).reshape(n, OBS_DIM)
    labels = {
        name.removeprefix("label_"): np.asarray(t[name].to_numpy(zero_copy_only=False), np.float32)
        for name in t.column_names
        if name.startswith("label_")
    }
    return Steps(
        obs=obs,
        labels=labels,
        action=np.asarray(t["action"].to_numpy(zero_copy_only=False), np.float32),
        reward=np.asarray(t["reward"].to_numpy(zero_copy_only=False), np.float32),
        episode=np.array([index[e] for e in eid_col], dtype=np.int64),
        episode_ids=episode_ids,
        split=np.array([split_of[e] for e in eid_col]),
    )
