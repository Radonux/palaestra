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
from dataclasses import dataclass, replace
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
    StockCC,
    TcpCross,
    UdpCross,
)
from expliccit_env.steps import StepError, Steps, load_steps, step_table
from expliccit_env.uml import UmlBuild, run_guest

RESULT_VERSION: Final = 5
# Record fields the tables leave out: constant per record type, or padding.
SKIP_COLUMNS: Final = frozenset({"type", "len", "pad"})


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
    # Per cross flow, in spec order: bytes its sender wrote, the congestion
    # control its socket ran ("" for UDP), steps and actions (0 unless an
    # agent).
    cross_bytes_sent: list[int]
    cross_cc: list[str]
    cross_steps: list[int]
    cross_actions: list[int]
    steps: int


class GuestResult(_Strict):
    """result.json written by the guest runner (trust boundary)."""

    result_version: Literal[5]
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
        if name not in SKIP_COLUMNS:
            cols[name] = pa.array(np.ascontiguousarray(table[name]))
    return pa.Table.from_pydict(cols)


LINK_EVENTS: Final = ("arrival", "drop", "depart")


def record_tables(rec: Records, episode_id: str) -> dict[str, pa.Table]:
    """The records a shard stores, whole: sender observations, both links'
    events and opportunities, and the agent's actions. dataset.read_episode
    is the inverse."""
    parts = (rec.link_arrival, rec.link_drop, rec.link_depart)
    kinds = np.repeat(np.arange(3, dtype=np.int32), [t.size for t in parts])
    events = pa.concat_tables([_columns(t, episode_id) for t in parts])
    return {
        "obs": _columns(rec.tcp_obs, episode_id),
        "link_events": events.add_column(1, "event", dict_column(kinds, list(LINK_EVENTS))),
        "link_opp": _columns(rec.link_opp, episode_id),
        "agent_action": _columns(rec.agent_action, episode_id),
    }


def path_constants(spec: EpisodeSpec, base_ns: int) -> labels.PathConstants:
    """What the labels need of the spec, with the guest's time of trace ms 0."""
    return labels.PathConstants(
        base_ns,
        spec.path.delay.one_way_ns,
        spec.path.buffer.packets,
        spec.path.buffer.bytes,
        spec.path.loss.downlink,
    )


def _with_id(table: pa.Table, episode_id: str) -> pa.Table:
    return table.add_column(
        0, "episode_id", dict_column(np.zeros(table.num_rows, np.int32), [episode_id])
    )


def _spec_columns(spec: EpisodeSpec) -> dict[str, object]:
    p = spec.path
    trace = p.trace
    return {
        "cc": spec.cc.name,
        # Which controller: the stock name, or the policy's mode and hash.
        "controller": spec.cc.name if isinstance(spec.cc, StockCC) else spec.cc.label,
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
    n = len(spec.cross)
    if (
        not len(result.cross_bytes_sent)
        == len(result.cross_steps)
        == len(result.cross_actions)
        == n
    ):
        problems.append(f"{len(result.cross_bytes_sent)} cross flows ran, {n} planned")
    planned = [c.cc_name if isinstance(c, TcpCross) else "" for c in spec.cross]
    if result.cross_cc != planned:
        problems.append(f"cross flows ran {result.cross_cc}, planned {planned}")
    for i, c in enumerate(spec.cross):
        if _is_agent(c) and (i >= len(result.cross_steps) or result.cross_steps[i] == 0):
            problems.append(f"cross flow {i}: agent took no step")
    return problems


def _is_agent(c: TcpCross | UdpCross) -> bool:
    return isinstance(c, TcpCross) and isinstance(c.cc, AgentCC)


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
        "drop_fraction": labels.drop_fraction(rec, window),
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
) -> tuple[GuestResult, GuestEpisode, Records, Steps | None, dict[int, Steps]] | str:
    """The guest's validated result, records, the observed flow's steps and
    every agent cross flow's (by flow index), or why there are none."""
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
        cross = {
            i: load_steps(raw / f"steps-cross{i}.bin")
            for i, c in enumerate(spec.cross)
            if _is_agent(c)
        }
    except (OSError, RecordError, StepError) as e:
        return f"records: {e}"
    g = result.episode
    problems = validity_problems(g, rec, spec)
    if steps is not None and (steps.size == 0 or steps.size != g.steps):
        problems.append(f"{steps.size} steps in steps.bin, guest reported {g.steps}")
    for i, s in cross.items():
        if i < len(g.cross_steps) and s.size != g.cross_steps[i]:
            problems.append(
                f"{s.size} steps in steps-cross{i}.bin, guest reported {g.cross_steps[i]}"
            )
    if problems:
        return "invalid: " + "; ".join(problems)
    return result, g, rec, steps, cross


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
            case (result, g, rec, steps, cross_steps):
                pass
        timing |= {"kernel_release": result.kernel_release, "kernel_version": result.kernel_version}
        trace = spec.path.trace.timestamps()
        consts = path_constants(spec, g.base_ns)
        lab = labels.labels_table(rec, trace, consts, spec.duration_ms)
        lab = _with_id(lab, eid)
        stored = record_tables(rec, eid)
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
            "cross_controllers": [
                c.label if isinstance(c, TcpCross) else "udp_cbr" for c in spec.cross
            ],
            "cross_steps": g.cross_steps,
            "cross_actions": g.cross_actions,
            "flow_bytes": _flow_bytes(spec, rec),
            **_flow_delays(spec, rec),
            "agent_actions": g.agent_actions,
            "n_steps": g.steps,
            "n_obs": rec.tcp_obs.size,
            "n_link_events": stored["link_events"].num_rows,
            **_metrics(spec, rec, g, trace),
        }
        return EpisodeOk(
            eid,
            row,
            {
                **stored,
                "labels": lab,
                "flow_seconds": _with_id(_flow_seconds(spec, rec), eid),
                **(
                    {}
                    if steps is None
                    else {
                        "steps": _with_id(
                            step_table(steps, rec, trace, consts, duration_ms=spec.duration_ms), eid
                        )
                    }
                ),
                **(
                    {
                        "cross_steps": _with_id(
                            _cross_steps(cross_steps, rec, trace, consts, spec), eid
                        )
                    }
                    if cross_steps
                    else {}
                ),
            },
        )
    finally:
        if not keep_raw:
            shutil.rmtree(d, ignore_errors=True)


def _flow_bytes(spec: EpisodeSpec, rec: Records) -> list[int]:
    """Bytes each flow got through the bottleneck within the episode: the
    observed flow, then every cross flow in spec order (by sender port)."""
    dep = rec.link(rec.link_depart, labels.DOWNLINK)
    dep = dep[dep["ms"] < spec.duration_ms]
    ports = [labels.MAIN_SPORT, *(labels.CROSS_PORT_BASE + i for i in range(len(spec.cross)))]
    sums = np.bincount(
        dep["sport"].astype(np.int64),
        weights=dep["bytes"].astype(np.float64),
        minlength=max(ports) + 1,
    )
    return [int(sums[p]) for p in ports]


def _flow_delays(spec: EpisodeSpec, rec: Records) -> dict[str, list[float]]:
    """Each flow's queueing delay at the bottleneck over the episode, per
    packet (departure minus arrival ms, Mahimahi's resolution): its mean and
    its 95th percentile, the two delays CC papers report, in the order of
    _flow_bytes; NaN for a flow that got no packet through."""
    dep = rec.link(rec.link_depart, labels.DOWNLINK)
    dep = dep[dep["ms"] < spec.duration_ms]
    ports = [labels.MAIN_SPORT, *(labels.CROSS_PORT_BASE + i for i in range(len(spec.cross)))]
    means: list[float] = []
    p95s: list[float] = []
    for port in ports:
        d = dep[dep["sport"] == port]
        q = (d["ms"].astype(np.int64) - d["arrival_ms"].astype(np.int64)).astype(np.float64)
        means.append(float(q.mean()) if q.size else float("nan"))
        p95s.append(float(np.percentile(q, 95)) if q.size else float("nan"))
    return {"flow_delay_mean_ms": means, "flow_delay_p95_ms": p95s}


def _flow_seconds(spec: EpisodeSpec, rec: Records) -> pa.Table:
    """Per flow (0 the observed flow, i the i-th cross flow) and per whole
    second of the episode, what left the bottleneck: bytes, packets, and the
    packets' queueing delays (departure minus arrival ms, Mahimahi's
    resolution) as a sum, a 95th percentile and a maximum. Window means are
    exact from the sums; percentiles are per second only."""
    dep = rec.link(rec.link_depart, labels.DOWNLINK)
    dep = dep[dep["ms"] < spec.duration_ms]
    ports = [labels.MAIN_SPORT, *(labels.CROSS_PORT_BASE + i for i in range(len(spec.cross)))]
    seconds = -(-spec.duration_ms // 1000)
    cols: dict[str, list[int]] = {k: [] for k in ("flow", "second", "bytes", "packets")}
    stats: dict[str, list[float]] = {"delay_sum_ms": [], "delay_p95_ms": [], "delay_max_ms": []}
    for flow, port in enumerate(ports):
        d = dep[dep["sport"] == port]
        sec = (d["ms"] // 1000).astype(np.int64)
        delay = (d["ms"].astype(np.int64) - d["arrival_ms"].astype(np.int64)).astype(np.float64)
        order = np.argsort(sec, kind="stable")
        sec, delay, size = sec[order], delay[order], d["bytes"][order].astype(np.int64)
        edges = np.searchsorted(sec, np.arange(seconds + 1))
        for s in range(seconds):
            q = delay[edges[s] : edges[s + 1]]
            cols["flow"].append(flow)
            cols["second"].append(s)
            cols["bytes"].append(int(size[edges[s] : edges[s + 1]].sum()))
            cols["packets"].append(int(q.size))
            stats["delay_sum_ms"].append(float(q.sum()))
            stats["delay_p95_ms"].append(float(np.percentile(q, 95)) if q.size else np.nan)
            stats["delay_max_ms"].append(float(q.max()) if q.size else np.nan)
    return pa.Table.from_pydict(
        {
            "flow": pa.array(cols["flow"], pa.int16()),
            "second": pa.array(cols["second"], pa.int32()),
            "bytes": pa.array(cols["bytes"], pa.int64()),
            "packets": pa.array(cols["packets"], pa.int64()),
            **{k: pa.array(v, pa.float64()) for k, v in stats.items()},
        }
    )


def _cross_steps(
    steps: dict[int, Steps],
    rec: Records,
    trace: mm.I64,
    k: labels.PathConstants,
    spec: EpisodeSpec,
) -> pa.Table:
    """Every agent cross flow's steps, as the steps table has the observed
    flow's, with the flow's index in spec.cross; labels are the flow's own.
    A flow with extra_rtt_ms has that much longer a base RTT, and its
    one-way delay label is half of it, as the sender can only see RTTs."""
    tables: list[pa.Table] = []
    for i, s in sorted(steps.items()):
        c = spec.cross[i]
        extra_ns = c.extra_rtt_ms * 1_000_000 if isinstance(c, TcpCross) else 0
        own = replace(k, one_way_ns=k.one_way_ns + extra_ns // 2)
        t = step_table(
            s, rec, trace, own, duration_ms=spec.duration_ms, sport=labels.CROSS_PORT_BASE + i
        )
        tables.append(t.add_column(0, "flow", pa.array(np.full(t.num_rows, i, np.int16))))
    return pa.concat_tables(tables)


def outcome_json(outcome: EpisodeOutcome) -> str:
    return json.dumps(outcome.row, indent=2, default=str)
