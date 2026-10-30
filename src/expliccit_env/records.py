"""Decoder for the guest's record files (trust boundary for guest output).

The guest writes one file per record type, each a packed array of the C
structs in ``kernel/modules/expliccit_rec.h``. ``load_records`` maps each file
with a numpy structured dtype mirroring the struct, then validates the whole
episode before anything downstream sees it: exact sizes, constant type and
length fields, one META record with the expected magic and ABI, and
contiguous sequence numbers across all files (a gap means the ring dropped
records). Cost: O(n log n) in the record count for the sequence check, with
no per-record Python work.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

ABI: Final = 2
MAGIC: Final = 0x52504C45

_HDR: Final = [("type", "<u2"), ("len", "<u2"), ("seq", "<u4"), ("t_ns", "<u8")]

META: Final = np.dtype(
    [*_HDR, ("magic", "<u4"), ("abi", "<u4"), ("hz", "<u4"), ("ring_bytes", "<u4")]
)
LINK: Final = np.dtype(
    [
        *_HDR,
        ("link_id", "<u4"),
        ("bytes", "<u4"),
        ("ms", "<u8"),
        ("arrival_ms", "<u8"),
        ("qlen_pkts", "<u4"),
        ("qlen_bytes", "<u4"),
        ("tcp_seq", "<u4"),
        ("tcp_ack", "<u4"),
        ("sport", "<u2"),
        ("dport", "<u2"),
        ("proto", "u1"),
        ("tcp_flags", "u1"),
        ("reason", "u1"),
        ("pad", "u1"),
    ]
)
OPP: Final = np.dtype(
    [
        *_HDR,
        ("link_id", "<u4"),
        ("count", "<u4"),
        ("ms", "<u8"),
        ("used_bytes", "<u4"),
        ("pad", "<u4"),
    ]
)
STATS: Final = np.dtype(
    [
        *_HDR,
        ("link_id", "<u4"),
        ("pad", "<u4"),
        ("arrivals", "<u8"),
        ("departures", "<u8"),
        ("drops_queue", "<u8"),
        ("drops_delay", "<u8"),
        ("drops_oversize", "<u8"),
        ("drops_gso", "<u8"),
        ("bypassed", "<u8"),
        ("early", "<u8"),
        ("purged", "<u8"),
    ]
)
_OBS_U32: Final = (
    "ack_seq",
    "snd_una",
    "snd_nxt",
    "snd_cwnd",
    "snd_ssthresh",
    "srtt_us",
    "mdev_us",
    "min_rtt_us",
    "last_rtt_us",
    "packets_out",
    "sacked_out",
    "lost_out",
    "retrans_out",
    "delivered",
    "delivered_ce",
    "lost",
    "total_retrans",
    "rate_delivered",
    "rate_interval_us",
    "app_limited",
    "mss_cache",
    "snd_wnd",
    "notsent_bytes",
    "flags",
)
OBS: Final = np.dtype(
    [
        *_HDR,
        ("sport", "<u2"),
        ("dport", "<u2"),
        ("ca_state", "u1"),
        ("tcp_flags", "u1"),
        ("data_len", "<u2"),
        *[(f, "<u4") for f in _OBS_U32],
        *[
            (f, "<u8")
            for f in (
                "bytes_acked",
                "bytes_sent",
                "bytes_retrans",
                "pacing_rate",
                "delivered_mstamp",
            )
        ],
    ]
)

AGENT: Final = np.dtype([*_HDR, ("cwnd", "<u4"), ("action_seq", "<u4"), ("pacing_rate", "<u8")])

# EXPL_OBS_* flag bits.
OBS_RATE_APP_LIMITED: Final = 0x1
OBS_CWND_LIMITED: Final = 0x2

# enum expl_drop_reason
DROP_QUEUE_FULL: Final = 1
DROP_DELAY_FULL: Final = 2
DROP_OVERSIZE: Final = 3
DROP_GSO: Final = 4

# (file stem, record type, dtype); the guest's recorder.rs uses the same table.
FILES: Final = (
    ("meta", 1, META),
    ("link_arrival", 2, LINK),
    ("link_drop", 3, LINK),
    ("link_depart", 4, LINK),
    ("link_opp", 5, OPP),
    ("link_stats", 6, STATS),
    ("tcp_obs", 7, OBS),
    ("agent_action", 8, AGENT),
)

type Table = npt.NDArray[np.void]


class RecordError(ValueError):
    """The record files are not a complete, well-formed episode."""


@dataclass(frozen=True, slots=True)
class Records:
    """Validated records of one episode, one structured array per type."""

    meta: Table
    link_arrival: Table
    link_drop: Table
    link_depart: Table
    link_opp: Table
    link_stats: Table
    tcp_obs: Table
    agent_action: Table

    @property
    def hz(self) -> int:
        return int(self.meta["hz"][0])

    def link(self, table: Table, link_id: int) -> Table:
        """Rows of a LINK or OPP table belonging to one qdisc."""
        return table[table["link_id"] == link_id]


def _load(path: Path, rec_type: int, dtype: np.dtype[np.void]) -> Table:
    raw = path.read_bytes()
    if len(raw) % dtype.itemsize:
        raise RecordError(f"{path.name}: {len(raw)} bytes is not a multiple of {dtype.itemsize}")
    arr: Table = np.frombuffer(raw, dtype=dtype)
    if arr.size and not (np.all(arr["type"] == rec_type) and np.all(arr["len"] == dtype.itemsize)):
        raise RecordError(f"{path.name}: records with the wrong type or length")
    return arr


def load_records(out_dir: Path) -> Records:
    tables = {stem: _load(out_dir / f"{stem}.bin", t, d) for stem, t, d in FILES}
    meta = tables["meta"]
    if meta.size != 1:
        raise RecordError(f"expected one META record, found {meta.size}")
    if int(meta["magic"][0]) != MAGIC or int(meta["abi"][0]) != ABI:
        raise RecordError(
            f"META magic/abi {meta['magic'][0]:#x}/{meta['abi'][0]} != {MAGIC:#x}/{ABI}"
        )
    # Sequence numbers are u32 and may wrap; count from META's.
    seq = np.concatenate([t["seq"] for t in tables.values()]).astype(np.uint32)
    rel = np.sort(seq - meta["seq"][0])
    if not np.array_equal(rel, np.arange(rel.size, dtype=np.uint32)):
        raise RecordError("record sequence numbers are not contiguous: the ring dropped records")
    return Records(**tables)
