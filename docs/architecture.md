# Architecture

## Data flow

    EpisodeSpec / CampaignSpec (JSON)          src/expliccit_env/spec.py   <- trust boundary
            |  guest_plan(): traces as per-ms counts, policy file
            v
    plan dir: plan.json, *.trace, policy.onnx  src/expliccit_env/guest_plan.py
            |  uml.py (Tier 1: UML, inf-cpu, seed)  or  native.py (Tier 2: root, real time)
            v
    expliccit-guest (PID 1 in UML, or root process)                guest/src/
            |  Plan::load                                          <- trust boundary
            |  namespaces, veths, 2 x mmlink qdisc (rtnetlink), modules
            |  bulk TCP pair, optional policy loop
            |  /dev/expl_rec -> one file per record type, framing + seq checked
            v
    raw dir: *.bin, result.json, console.log
            |  episode.py: GuestResult (pydantic) + load_records    <- trust boundary
            |  validity checks, labels.py, Mahak metrics (mahimahi.py)
            v
    shard dir: episodes/obs/labels/link_events/link_opp/agent_action .parquet, DONE

Everything that crosses a process or kernel boundary is parsed once, at the
marked boundaries, into types that cannot hold invalid values: specs
(pydantic, closed sums), the guest plan (Rust `Plan::load`), netlink options
(kernel `nla_parse_nested` with a policy), agent actions (kernel
`agent_write`), records (numpy dtypes mirroring the C structs, checked for
size, type, ABI and sequence continuity), and `result.json` (pydantic).

## Topology inside one episode

    snd 10.0.0.1 s0 ---- p0 10.0.0.2 [path] 10.0.1.2 p1 ---- r0 10.0.1.1 rcv
                          ^ uplink mmlink           ^ downlink mmlink
                          (link, then delay)         (delay, then link)

Three fresh network namespaces joined by veth pairs. The bottleneck lives in
the middle namespace, so every skb is orphaned when it crosses into it and the
sender's TCP Small Queues see a remote bottleneck, exactly as a Mahimahi shell
looks to the server outside it. Data packets cross `mm-delay` then `mm-link`;
ACKs cross the uplink then the delay, matching Mahimahi's nesting. IPv6 is
off, and ARP bypasses the model (Mahimahi works on IP packets), so nothing but
the experiment's traffic is shaped.

The receiver connects at trace time 0 and the sender (listening with the
congestion control and observation mark already set, as in Mahak's
`server-mahimahi`) writes until the episode ends; both set `TCP_NODELAY`.

## The link model (`sch_mmlink`)

A port of `third_party/mahimahi/src/frontend/link_queue.cc`:

- Time is integer milliseconds since the qdisc's `base_ns`, shared by both
  directions. Trace entry `i` is the number of 1504-byte delivery
  opportunities at ms `i`; the trace repeats with period equal to its last
  timestamp (so the last ms of a cycle and ms 0 of the next coincide, as in
  Mahimahi).
- Before a packet that arrives in ms `a` joins the queue, every opportunity at
  ms `<= a` is used. A packet therefore never departs in its arrival ms.
- The head of the queue moves "in transit" and may span several opportunities;
  it departs at the opportunity carrying its last byte. Droptail limits
  (packets and/or bytes) count queued packets only, never the one in transit.
- A packet's size is its IPv4 total length plus the 4-byte TUN header
  Mahimahi counts, so a full 1500-byte packet uses one 1504-byte opportunity.
- The delay line releases each packet exactly `delay_ns` after it entered. A
  packet leaving the uplink at opportunity ms `t` enters its delay line at the
  instant of `t`.
- GSO super-packets are segmented on entry, because Mahimahi only ever sees
  MTU-sized packets.

The qdisc processes events lazily (on enqueue, dequeue and its watchdog) but
always in time order, and it uses the *semantic* instants (opportunity times,
release times), never the moment its code happened to run. So timer slack
(native) or the 256 ns UML clock tick (Tier 1) cannot change the model's
decisions. Cost: O(1) per packet event plus O(1) per elapsed trace ms.

The trace travels as one netlink attribute of per-ms u8 counts. Netlink
attributes carry a 16-bit length, which a list of up to 900000 timestamps
(Mahak's 100 Mbit/s x 8 grid corner) cannot fit; per-ms counts fit any trace
up to 65 s long and 3 Gbit/s (`ponytail` limits, enforced on both sides).

## Records

`kernel/modules/expliccit_rec.h` is the single source of truth. Every record
has a 16-byte header (type, length, sequence number, time). Producers
(`sch_mmlink`, `expl_tcpobs`, `tcp_agent`) append to one ring in `expl_rec`,
so the episode has one global sequence; a full ring drops whole records and
the sequence gap is detected by the guest and again by the host.

| Type | Producer | Meaning (t_ns is the event's own instant) |
|---|---|---|
| META | expl_rec | ABI, HZ, ring size; first record after open |
| LINK_ARRIVAL / DROP / DEPART | sch_mmlink | Mahimahi's `+`, `d`, `-` events, with queue state, flow ports and TCP seq/ack |
| LINK_OPP | sch_mmlink | Mahimahi's `#` events, one record per ms with count and bytes used |
| LINK_STATS | sch_mmlink | counters at qdisc deletion; nonzero error counters invalidate the episode |
| TCP_OBS | expl_tcpobs | sender state just before each incoming segment is processed |
| AGENT_ACTION | tcp_agent | an accepted policy action |

The guest splits the stream into one file per type, so the host maps each as
a packed numpy array with no per-record Python work.

`TCP_OBS` uses the `tcp:tcp_probe` tracepoint because UML has no kprobes and
stock congestion controls must stay unmodified. The tracepoint fires at the
top of `tcp_rcv_established`, so record k is the state left by segment k-1
plus any timer-driven changes since; labels and training code must treat it
as "state observed when segment k arrived".

## Time and determinism (Tier 1)

`time-travel=inf-cpu` makes CPU work free: virtual time advances only when the
guest idles or sleeps. Three consequences shape the design (measured in
`docs/m0-report.md`):

- Every episode needs a time-based bottleneck; otherwise a transfer is
  infinitely fast and never reaches its end time. The host enforces a
  wall-clock timeout on every guest.
- Guest code must block or sleep, never spin: UML advances a jiffy after
  10000 unscheduled userspace switches. The record reader and the policy loop
  both sleep on absolute deadlines.
- UML's clock ticks every 256 ns, and every process-context clock read
  advances virtual time by one tick. Sender observation timestamps carry this
  sub-microsecond quantization; the link model does not (see above).

With `CONFIG_UML_SEEDED_RANDOM` (our patch, see `third_party/README.md`),
`um_seed=` and a fixed `time-travel-start=`, an episode is a pure function of
(kernel build, plan, seed): the same inputs give byte-identical record files.
`uml.py` passes the spec's seed; campaign seeds are drawn deterministically
from the campaign seed.

## Agent controllers (`tcp_agent`)

A policy reads `TCP_INFO` every `interval_ms` of virtual time, builds the
`tcpinfo-v1` feature vector (`src/expliccit_env/policy.py`), runs the ONNX
model with `tract`, and writes `struct expl_agent_action` to
`/dev/expl_agent`. Out-of-range or NaN outputs fail the episode.

- `agent` (Sage-style): `cong_control` holds cwnd and pacing at the latest
  action; the initial window is held until the first action.
- `agent_reno` (Orca-style): each new action sets cwnd once (from
  `in_ack_event`, so it applies even when the flow is not cwnd-limited);
  Reno and the core loss response run in between. Orca proper runs Cubic
  underneath; Cubic's update is not exported, so this base is Reno.

## Decisions

D1 to D5 are the brief's (real kernel TCP; Tier 1 UML; Tier 2 native 7.0;
kernel 7.0 only; custom controllers as modules). M0 confirmed D2 and D5 and
measured their consequences. The decisions below were made during the build,
where the brief did not specify (its later sections were missing).

| | Decision | Why, and what was rejected |
|---|---|---|
| D6 | The link model is a kernel qdisc (`sch_mmlink`) | Same code in both tiers, no per-packet user/kernel crossings, exact semantic times. Rejected: Mahimahi itself inside the guest (TUN copies, heavy build, ms timers in userspace); netem for the delay plus a separate link (four namespaces, two models to validate). |
| D7 | One shared record ring, one global sequence, split by type in the guest | Detectable loss (sequence gaps), a total order of events, and vectorized decoding on the host. |
| D8 | Observations from the `tcp_probe` tracepoint | No kprobes in UML; stock CCs stay unmodified. Rejected: polling `TCP_INFO` (not per-ACK, and spinning distorts virtual time). |
| D9 | `arch/um` HZ patched to 1000 | UML hard-codes 100; TCP's jiffy timers would be 10x coarser than on the Tier 2 kernels. |
| D10 | Seeded randomness patch for UML | Bit-reproducible episodes; host entropy leaked in through three paths (see `third_party/README.md`). |
| D11 | Vendored kernel and Mahimahi as unmodified import commits plus patch commits | Self-contained builds (kernel.org outages), and `git diff <import>` is the patch series. |
| D12 | Parquet shards with a DONE marker | Few large files per worker (HPC filesystems), atomic visibility, rerunnable shards. |
| D13 | ONNX via `tract` in the guest | Pure Rust, static musl, deterministic, zero virtual-time cost. Needs a C compiler for its SIMD kernels at build time (`musl-tools`). |
| D14 | BBRv3 from google/bbr as a loadable module, its core changes kept inert for other CCs | The 12 core prerequisites are imported unchanged except one, made opt-in per CC (`TCP_CONG_BROAD_APP_LIMITED`); `tcp_bbr3` is a module loaded only for `bbr3` episodes, because building it in changed seeded BBRv1 runs. Verified by byte-identical stock episodes across the change. |
| D15 | Pinned kernel build metadata | uname's version string seeds the guest's random pool; pinning it makes seeded runs comparable across builds, which is what the D14 check relies on. |

## Known ceilings (`ponytail` markers in the code)

- One global record ring and one global agent action per kernel: fine for one
  episode per guest (Tier 1) or per host at a time (Tier 2).
- Per-ms u8 trace counts: at most 255 opportunities per ms (3 Gbit/s) and
  traces up to about 65 s.
- The record reader polls every 20 ms of virtual time.
- `parse_mm_log` (Mahimahi text logs) loops in Python; it is for fixtures and
  Tier 2 reference logs, not campaign data.
- Labels v1 are the candidates listed in `labels.py`; cross traffic and random
  loss are fixed at 0 because no episode generates them yet.

## Extending

- **A new candidate LCF (label).** Add a pure function to
  `src/expliccit_env/labels.py` that maps an episode's records and constants
  to one value per observation row, and a column in `labels_table`. Existing
  datasets keep `link_events` and `link_opp`, so the new label can be
  recomputed from them without rerunning episodes.
- **A new trace shape.** Add a variant next to `MahakStep` in `spec.py` with a
  `timestamps()` method returning Mahimahi trace timestamps; the kernel only
  ever sees per-ms counts. `TraceFile` already accepts any Mahimahi trace
  (pinned by sha256).
- **A learned controller.** Export it to ONNX with the `tcpinfo-v1` contract
  (`src/expliccit_env/policy.py`: input float32 [1, 8], output float32
  [1, 2] = cwnd packets, pacing Mbit/s) and use
  `{"kind": "agent", "mode": "hold" | "reno", "policy": {"path": ..., "sha256": ...}}`
  as the spec's `cc`.
- **A new in-kernel congestion control.** Build it into the UML kernel
  (`kernel/uml.config`) or as a module next to `tcp_agent`, then add its name
  to `StockCC` in `spec.py`.
- **A kernel change.** Commit it under `third_party/linux` as its own commit;
  `scripts/check-kernel.sh` checks every such commit, and
  `git diff e8eadca -- third_party/linux` lists them all. Record ABI changes go
  in `kernel/modules/expliccit_rec.h` together with `guest/src/recorder.rs`
  and `src/expliccit_env/records.py` (bump `EXPL_REC_ABI`;
  `tests/test_records_abi.py` checks the layouts agree).
