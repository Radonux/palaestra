"""One episode end to end: spec -> guest plan -> UML run -> validated records
-> tables. Every failure becomes an explicit failure row; no metric or
table is ever produced from a run that did not complete and validate
(unlike Mahak's harness, which reused the previous run's log on failure).
"""

from __future__ import annotations

import importlib.metadata
import json
import shutil
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, cast

import numpy as np
import numpy.typing as npt
import pyarrow as pa
from pydantic import BaseModel, ConfigDict, ValidationError

from expliccit_env import labels
from expliccit_env import mahimahi as mm
from expliccit_env.records import ABI, RecordError, Records, Table, load_records
from expliccit_env.spec import (
    AgentCC,
    BaseRtt,
    EpisodeSpec,
    MahakMrtt,
    MahakStep,
    TcpCross,
    UdpCross,
)
from expliccit_env.steps import StepError, Steps, load_steps, step_table
from expliccit_env.uml import UmlBuild, run_guest

RESULT_VERSION: Final = 3
_SKIP_COLUMNS: Final = frozenset({"type", "len", "pad"})


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class GuestEpisode(_Strict):
    base_ns: int
    end_ns: int
    duration_ms: int
    cc_requested: str
    cc_verified: str
    bytes_sent: int
    bytes_received: int
    records: dict[str, int]
    record_bytes: int
    ring_dropped: int
    softnet_dropped: int
    netdev_max_backlog: str
    agent_actions: int
    # Bytes each cross flow's sender wrote, in spec order.
    cross_bytes_sent: list[int]
    steps: int


class GuestResult(_Strict):
    """result.json written by the guest runner (trust boundary)."""

    result_version: Literal[3]
    status: Literal["ok", "failed"]
    error: str | None
    kernel_release: str
    kernel_version: str
    episode: GuestEpisode | None


@dataclass(frozen=True, slots=True)
class EpisodeOk:
    episode_id: str
    row: dict[str, object]
    tables: dict[str, pa.Table]


@dataclass(frozen=True, slots=True)
class EpisodeFailed:
    episode_id: str
    row: dict[str, object]


type EpisodeOutcome = EpisodeOk | EpisodeFailed


def dict_column(codes: npt.NDArray[np.int32], values: list[str]) -> pa.Array[Any]:
    """A dictionary-encoded string column: values[codes[i]] per row."""
    # pyarrow-stubs leave DictionaryArray's type parameters unknown.
    return cast("pa.Array[Any]", pa.DictionaryArray.from_arrays(codes, values))  # pyright: ignore[reportUnknownMemberType]


def _columns(table: Table, episode_id: str) -> pa.Table:
    cols: dict[str, pa.Array[Any]] = {
        "episode_id": dict_column(np.zeros(table.size, np.int32), [episode_id])
    }
    for name in table.dtype.names or ():
        if name not in _SKIP_COLUMNS:
            cols[name] = pa.array(np.ascontiguousarray(table[name]))
    return pa.Table.from_pydict(cols)


def _with_id(table: pa.Table, episode_id: str) -> pa.Table:
    return table.add_column(
        0, "episode_id", dict_column(np.zeros(table.num_rows, np.int32), [episode_id])
    )


def _spec_columns(spec: EpisodeSpec) -> dict[str, object]:
    p = spec.path
    trace = p.trace
    return {
        "cc": spec.cc.name,
        "trace_kind": trace.kind,
        "bw_mbps": trace.bw_mbps if isinstance(trace, MahakStep) else None,
        "step": trace.step if isinstance(trace, MahakStep) else None,
        "mahak_compat_mrtt_ms": p.delay.mahak_compat_mrtt_ms
        if isinstance(p.delay, MahakMrtt)
        else None,
        "base_rtt_ms": p.delay.base_rtt_ms
        if isinstance(p.delay, BaseRtt)
        else 2 * p.delay.mahak_compat_mrtt_ms,
        "buffer_pkts": p.buffer.packets,
        "buffer_bytes": p.buffer.bytes,
        "loss_downlink": p.loss.downlink,
        "loss_uplink": p.loss.uplink,
        "cross_tcp": sum(isinstance(c, TcpCross) for c in spec.cross),
        # A float even when 0, so every shard has the same column type.
        "cross_udp_mbps": sum((c.rate_mbps for c in spec.cross if isinstance(c, UdpCross)), 0.0),
        "duration_ms": spec.duration_ms,
        "seed": str(spec.seed),
        "spec_json": spec.canonical_json(),
    }


def _provenance(build: UmlBuild) -> dict[str, object]:
    info = build.build_info()
    return {
        "env_version": importlib.metadata.version("expliccit-env"),
        "kernel_tree": info.get("kernel_tree"),
        "repo_commit": info.get("repo_commit"),
        "config_sha256": info.get("config_sha256"),
        "source_dirty_paths": info.get("source_dirty_paths"),
        "record_abi": ABI,
        "host": socket.gethostname(),
    }


def validity_problems(result: GuestEpisode, rec: Records, spec: EpisodeSpec) -> list[str]:
    """Reasons the episode must not be used, empty if none."""
    problems: list[str] = []
    if result.cc_verified != spec.cc.name:
        problems.append(f"socket ran {result.cc_verified}")
    if result.ring_dropped or result.softnet_dropped:
        problems.append(
            f"dropped records/packets: ring {result.ring_dropped}, softnet {result.softnet_dropped}"
        )
    if rec.link_stats.size != 2:
        problems.append(f"{rec.link_stats.size} LINK_STATS records, expected 2")
    for field in ("drops_delay", "drops_oversize", "drops_gso", "early"):
        if (n := int(rec.link_stats[field].sum())) != 0:
            problems.append(f"{field}={n}")
    if rec.tcp_obs.size == 0:
        problems.append("no sender observations")
    if len(result.cross_bytes_sent) != len(spec.cross):
        problems.append(
            f"{len(result.cross_bytes_sent)} cross flows ran, {len(spec.cross)} planned"
        )
    return problems


def _metrics(
    spec: EpisodeSpec, rec: Records, result: GuestEpisode, trace: mm.I64
) -> dict[str, object]:
    window = spec.duration_ms
    one_way_ms = spec.path.delay.one_way_ns / 1e6
    out: dict[str, object] = {
        "goodput_mbps": result.bytes_received * 8 / (window / 1000) / 1e6,
        "total_retrans": int(rec.tcp_obs["total_retrans"][-1]),
        "drops_queue": int(rec.link_stats["drops_queue"].sum()),
        "drops_random": int(rec.link_stats["drops_random"].sum()),
        "capacity_cv": labels.capacity_cv(trace, window),
    }
    match mm.mm_th_delay(mm.mm_log_from_records(rec, labels.DOWNLINK, trace, window)):
        case mm.MmThDelay() as r:
            out |= {
                "utilization": r.utilization,
                "mean_delay_ms": r.mean_delay_ms,
                "average_capacity_mbps": r.average_capacity_mbps,
                "average_throughput_mbps": r.average_throughput_mbps,
                "mahak_delay": mm.mahak_metric(r, "delay", one_way_ms),
                "first_power": mm.mahak_metric(r, "first_power", one_way_ms),
            }
        case mm.MmThDelayUndefined(reason):
            out |= {"metrics_undefined": reason}
    return out


def _records_steps(spec: EpisodeSpec) -> bool:
    return spec.steps or isinstance(spec.cc, AgentCC)


def _collect(
    raw: Path, returncode: int | None, timeout_s: float, spec: EpisodeSpec
) -> tuple[GuestResult, GuestEpisode, Records, Steps | None] | str:
    """The guest's validated result, records and steps, or why there are none."""
    if returncode is None:
        return f"timeout after {timeout_s} s"
    try:
        result = GuestResult.model_validate_json((raw / "result.json").read_text())
    except (OSError, ValidationError) as e:
        return f"result.json: {e}"
    if result.status == "failed" or result.episode is None:
        return f"guest: {result.error}"
    try:
        rec = load_records(raw)
        steps = load_steps(raw / "steps.bin") if _records_steps(spec) else None
    except (OSError, RecordError, StepError) as e:
        return f"records: {e}"
    problems = validity_problems(result.episode, rec, spec)
    if steps is not None and (steps.size == 0 or steps.size != result.episode.steps):
        problems.append(f"{steps.size} steps in steps.bin, guest reported {result.episode.steps}")
    if problems:
        return "invalid: " + "; ".join(problems)
    return result, result.episode, rec, steps


def run_episode(
    spec: EpisodeSpec,
    build: UmlBuild,
    workdir: Path,
    *,
    timeout_s: float = 1800.0,
    keep_raw: bool = False,
    mem_mib: int = 1024,
) -> EpisodeOutcome:
    eid = spec.episode_id()
    d = workdir / eid
    if d.exists():
        shutil.rmtree(d)
    base_row: dict[str, object] = {"episode_id": eid, **_spec_columns(spec), **_provenance(build)}

    def failed(reason: str, **extra: object) -> EpisodeFailed:
        return EpisodeFailed(eid, {**base_row, "status": "failed", "failure": reason, **extra})

    try:
        plan = spec.guest_plan()
    except ValueError as e:
        return failed(f"spec: {e}")
    plan.write(d / "in")
    (d / "spec.json").write_text(spec.canonical_json())
    run = run_guest(
        build, d / "in", d / "raw", seed=spec.seed, timeout_s=timeout_s, mem_mib=mem_mib
    )
    timing: dict[str, object] = {"wall_s": run.wall_s}
    try:
        match _collect(d / "raw", run.returncode, timeout_s, spec):
            case str(reason):
                return failed(reason, **timing)
            case (result, g, rec, steps):
                pass
        timing |= {"kernel_release": result.kernel_release, "kernel_version": result.kernel_version}
        trace = spec.path.trace.timestamps()
        consts = labels.PathConstants(
            g.base_ns,
            spec.path.delay.one_way_ns,
            spec.path.buffer.packets,
            spec.path.buffer.bytes,
            spec.path.loss.downlink,
        )
        lab = labels.labels_table(rec, trace, consts, spec.duration_ms)
        lab = _with_id(lab, eid)
        link_events = pa.concat_tables(
            [_columns(t, eid) for t in (rec.link_arrival, rec.link_drop, rec.link_depart)]
        )
        kinds = np.repeat(
            np.arange(3, dtype=np.int32),
            [rec.link_arrival.size, rec.link_drop.size, rec.link_depart.size],
        )
        link_events = link_events.add_column(
            1, "event", dict_column(kinds, ["arrival", "drop", "depart"])
        )
        row = {
            **base_row,
            "status": "ok",
            "failure": None,
            **timing,
            "base_ns": g.base_ns,
            "end_ns": g.end_ns,
            "hz": rec.hz,
            "bytes_sent": g.bytes_sent,
            "bytes_received": g.bytes_received,
            "cross_bytes_sent": g.cross_bytes_sent,
            "agent_actions": g.agent_actions,
            "n_steps": g.steps,
            "n_obs": rec.tcp_obs.size,
            "n_link_events": link_events.num_rows,
            **_metrics(spec, rec, g, trace),
        }
        return EpisodeOk(
            eid,
            row,
            {
                "obs": _columns(rec.tcp_obs, eid),
                "labels": lab,
                "link_events": link_events,
                "link_opp": _columns(rec.link_opp, eid),
                "agent_action": _columns(rec.agent_action, eid),
                **(
                    {}
                    if steps is None
                    else {
                        "steps": _with_id(
                            step_table(steps, rec, trace, consts, duration_ms=spec.duration_ms), eid
                        )
                    }
                ),
            },
        )
    finally:
        if not keep_raw:
            shutil.rmtree(d, ignore_errors=True)


def outcome_json(outcome: EpisodeOutcome) -> str:
    return json.dumps(outcome.row, indent=2, default=str)
