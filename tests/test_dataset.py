"""Dataset shards (expliccit_env.dataset.ShardWriter)."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from expliccit_env import records
from expliccit_env.dataset import ShardWriter, read_episode
from expliccit_env.episode import SKIP_COLUMNS, EpisodeOk, record_tables
from expliccit_env.spec import EpisodeSpec


def test_a_step_stride_keeps_every_kth_step_and_leaves_other_tables_whole(tmp_path: Path) -> None:
    steps = pa.table({"episode_id": ["e"] * 10, "step": list(range(10))})
    obs = pa.table({"episode_id": ["e"] * 10, "t": list(range(10))})
    with ShardWriter(tmp_path / "s", frozenset({"steps", "obs"}), step_stride=4) as w:
        w.add(EpisodeOk("e", {"episode_id": "e", "status": "ok"}, {"steps": steps, "obs": obs}))
    assert pq.read_table(tmp_path / "s/steps.parquet")["step"].to_pylist() == [0, 4, 8]
    assert pq.read_table(tmp_path / "s/obs.parquet").num_rows == 10
    with pytest.raises(ValueError):
        ShardWriter(tmp_path / "t", step_stride=0)


def test_a_stored_episode_reads_back_as_its_records(tmp_path: Path) -> None:
    # Arbitrary field values: the reader must return every stored field as
    # written. Sequence numbers wrap past 2**32, META's first.
    rng = np.random.default_rng(0)
    sizes = {"link_arrival": 40, "link_drop": 5, "link_depart": 30, "link_opp": 20, "tcp_obs": 25}
    start, used = 2**32 - 50, 0
    seq = (start + 1 + rng.permutation(sum(sizes.values()))).astype(np.uint32)
    tables: dict[str, records.Table] = {}
    for stem, rec_type, dtype in records.FILES:
        n = 1 if stem == "meta" else sizes.get(stem, 0)
        t = np.zeros(n, dtype)
        for name in dtype.names or ():
            if name not in SKIP_COLUMNS:
                t[name] = rng.integers(0, np.iinfo(dtype[name]).max, n, dtype[name], endpoint=True)
        t["type"], t["len"] = rec_type, dtype.itemsize
        if stem == "meta":
            t["seq"] = start
        else:
            t["seq"], used = seq[used : used + n], used + n
        tables[stem] = t
    rec = records.Records(**tables)
    spec = EpisodeSpec.model_validate_json(
        (Path(__file__).parents[1] / "examples" / "episode-mahak.json").read_text()
    )
    row = {
        "episode_id": "e",
        "status": "ok",
        "spec_json": spec.canonical_json(),
        "base_ns": 123,
        "hz": 100,
    }
    with ShardWriter(tmp_path / "s") as w:
        w.add(EpisodeOk("other", row | {"episode_id": "other"}, record_tables(rec, "other")))
        w.add(EpisodeOk("e", row, record_tables(rec, "e")))
        w.add(EpisodeOk("bad", row | {"episode_id": "bad", "status": "failed"}, {}))

    got = read_episode(tmp_path / "s", "e")
    assert got.spec == spec and got.consts.base_ns == 123 and got.rec.hz == 100
    assert got.trace.tolist() == spec.path.trace.timestamps().tolist()
    for stem in sizes:
        assert getattr(got.rec, stem).tobytes() == tables[stem].tobytes(), stem
    # The first stored record follows META, which is not stored.
    assert int(got.rec.meta["seq"][0]) == start + 1
    assert got.rec.agent_action.size == got.rec.link_stats.size == 0
    for missing in ("bad", "absent"):
        with pytest.raises(records.RecordError):
            read_episode(tmp_path / "s", missing)
    with ShardWriter(tmp_path / "t", frozenset({"obs"})) as w:
        w.add(EpisodeOk("e", row, record_tables(rec, "e")))
    with pytest.raises(records.RecordError, match="link_events"):
        read_episode(tmp_path / "t", "e")
