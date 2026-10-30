# Architecture

## Data flow

    EpisodeSpec / CampaignSpec (JSON)          src/expliccit_env/spec.py   <- trust boundary
            |  guest_plan(): traces as per-ms counts, policy file
            v
    plan dir: plan.json, *.trace, policy.onnx  src/expliccit_env/guest_plan.py
            |  uml.py: boots UML with time-travel=inf-cpu and the spec's seed
            v
    expliccit-guest (PID 1 in UML)                                 guest/src/
            |  Plan::load                                          <- trust boundary
            |  namespaces, veths, 2 x mmlink qdisc (rtnetlink), modules
            |  bulk TCP pair, optional Sage step loop (+ policy) -> steps.bin
            |  /dev/expl_rec -> one file per record type, framing + seq checked
            v
    raw dir: *.bin, steps.bin, result.json, console.log
            |  episode.py: GuestResult (pydantic) + load_records    <- trust boundary
            |  + load_steps; validity checks, labels.py, steps.py, Mahak metrics
            v
    shard dir: episodes/obs/labels/link_events/link_opp/agent_action
               (+ steps) .parquet, DONE

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

Cross traffic shares the same path and the same direction (sender to
receiver). Flow `i` sends from port `6000 + i`; the observed flow sends from
port 5001, which is how link records and labels tell them apart. A TCP cross
flow is a bulk transfer with its own congestion control (Cubic, BBRv1,
BBRv3 or Reno), connected at its start time and shut down at its stop time.
A UDP cross flow sends one 1500-byte IP packet every `12000 / rate` seconds
over its window. Cross sockets carry no observation mark, so `TCP_OBS` holds
the observed flow only. Each flow is driven by threads sleeping on absolute
deadlines, so it does not spin in virtual time.

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
- Optional random loss (Mahimahi's `mm-loss`): each packet entering the
  qdisc is dropped independently with the direction's probability, before
  the delay line and the link, so a lost packet never uses link capacity or
  appears as an arrival. It draws from the kernel RNG only when the
  probability is nonzero, so loss-free episodes are unchanged.

The qdisc processes events lazily (on enqueue, dequeue and its watchdog) but
always in time order, and it uses the *semantic* instants (opportunity times,
release times), never the moment its code happened to run. So the 256 ns
UML clock tick (or timer slack, on any other kernel) cannot change the
model's decisions. Cost: O(1) per packet event plus O(1) per elapsed trace ms.

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
| LINK_ARRIVAL / DROP / DEPART | sch_mmlink | Mahimahi's `+`, `d`, `-` events, with queue state, flow ports and TCP seq/ack; a drop names its reason (full queue, random loss, or an error) |
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

## Time and determinism

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

## Sage's step (contract `sage-v1`)

expliCCit's networks read what Sage's sender computes, so the guest computes
it the same way: `guest/src/step.rs` ports the step of Sage's `sage.cc`
(`CntThread`, commit `eedd281`). Every 10 ms of virtual time the step loop
reads `TCP_INFO` from the observed flow's socket; once the ACK state has
moved (bytes acked or packets delivered; otherwise it polls again every
1 ms, where Sage busy-polls) it builds Sage's 77-value message. That
covers the current RTT, delivery rate and window, their averages, minima
and maxima over 10, 200 and 1000 steps, and Sage's derived rates. The
loop writes the step to `steps.bin` and, under a policy, acts on it. Sage's
quirks are kept (an empty window's minimum is 1e6, the incremental mean,
integer averages), because a policy trained on Sage's numbers must see
Sage's numbers. Two columns are not observations and are written as 0:
column 1 (Sage's nominal capacity, from its command line) and column 75
(its reward, which uses that capacity).

The policy contract (`src/expliccit_env/policy.py`): an ONNX graph takes the
full message as `obs` float32 [1, 77], plus an optional recurrent `state`
[1, S] that the guest feeds back from the graph's second output, and returns
a, the log2 of the window multiplier. The graph selects its columns and
scales them itself, so a new network design needs no guest change. The
window follows Sage: a precise accumulator from 10 packets, multiplied by
2^a, rounded up and kept in [4, 20000]; pacing is cwnd * mss / srtt,
refreshed every step. |a| > 8 or a non-finite a fails the episode.

Column 76 is the window multiplier since the previous step: the policy's
2^a, or under a stock controller the observed change of the window (Sage's
collector logs cwnd over the previous cwnd, from 10). The training table
(`steps.py`, `steps.parquet`) gives each step the labels at its instant, the
action taken from it (the next step's column 76, as Sage's
`gen_offline_data.py` derives it), and Sage's single-flow reward computed
with the exact capacity where Sage uses the nominal one.

## Agent controllers (`tcp_agent`)

After each step the step loop writes the policy's window and pacing as
`struct expl_agent_action` to `/dev/expl_agent`.

- `agent` (Sage-style): `cong_control` holds cwnd and pacing at the latest
  action; the initial window is held until the first action.
- `agent_reno` (Orca-style): each new action sets cwnd once (from
  `in_ack_event`, so it applies even when the flow is not cwnd-limited);
  Reno and the core loss response run in between. Orca proper runs Cubic
  underneath; Cubic's update is not exported, so this base is Reno.

## Mahak's driver (`mahak.py`)

A port of Mahak's `mahak_online.py` and the query and result functions of
its `helper.py`. The grid, the features (min-max scaling, then a full PCA),
the model (scikit-learn's Gaussian process, RBF kernel with length scale
0.1, 10 optimizer restarts, normalized targets, `random_state=100`), the
first point (`choice` on a legacy numpy RandomState, which is Mahak under
`np.random.seed`) and the query rule (largest predictive standard deviation,
first index on ties) are Mahak's. The oracle is one UML episode per query,
20 s as in `helper.py`, with the same seed for every query, so it is a pure
function of the point and a repeated point reuses its result.

Mahak's oracle crashes on a failed emulation. Here a failed episode, an
undefined metric or a non-finite value is recorded with its reason, the
point leaves the candidate pool, and it never becomes a label. Every queried
episode is also written to a dataset shard, so a map's queries are reusable
training data.

## Decisions

D1 to D5 are the brief's (real kernel TCP; Tier 1 UML; Tier 2 native 7.0;
kernel 7.0 only; custom controllers as modules). M0 confirmed D2 and D5 and
measured their consequences; D19 later dropped D3, which leaves UML as the
only tier. The decisions below were made during the build,
where the brief did not specify (its later sections were missing).

| | Decision | Why, and what was rejected |
|---|---|---|
| D6 | The link model is a kernel qdisc (`sch_mmlink`) | Runs inside the kernel under test, no per-packet user/kernel crossings, exact semantic times. Rejected: Mahimahi itself inside the guest (TUN copies, heavy build, ms timers in userspace); netem for the delay plus a separate link (four namespaces, two models to validate). |
| D7 | One shared record ring, one global sequence, split by type in the guest | Detectable loss (sequence gaps), a total order of events, and vectorized decoding on the host. |
| D8 | Observations from the `tcp_probe` tracepoint | No kprobes in UML; stock CCs stay unmodified. Rejected: polling `TCP_INFO` (not per-ACK, and spinning distorts virtual time). |
| D9 | `arch/um` HZ patched to 1000 | UML hard-codes 100; TCP's jiffy timers would be 10x coarser than on an HZ=1000 host kernel. |
| D10 | Seeded randomness patch for UML | Bit-reproducible episodes; host entropy leaked in through three paths (see `third_party/README.md`). |
| D11 | Vendored kernel and Mahimahi as unmodified import commits plus patch commits | Self-contained builds (kernel.org outages), and `git diff <import>` is the patch series. |
| D12 | Parquet shards with a DONE marker | Few large files per worker (HPC filesystems), atomic visibility, rerunnable shards. |
| D13 | ONNX via `tract` in the guest | Pure Rust, static musl, deterministic, zero virtual-time cost. Needs a C compiler for its SIMD kernels at build time (`musl-tools`). |
| D14 | BBRv3 from google/bbr as a loadable module, its core changes kept inert for other CCs | The 12 core prerequisites are imported unchanged except one, made opt-in per CC (`TCP_CONG_BROAD_APP_LIMITED`); `tcp_bbr3` is a module loaded only for `bbr3` episodes, because building it in changed seeded BBRv1 runs. Verified by byte-identical stock episodes across the change. |
| D15 | Pinned kernel build metadata | uname's version string seeds the guest's random pool; pinning it makes seeded runs comparable across builds, which is what the D14 check relies on. |
| D16 | Random loss sits in `sch_mmlink`, before the link, in both directions | One qdisc keeps one event order and one record stream; lost packets never consume capacity, so capacity and cross-traffic labels stay exact. No single Mahimahi shell nesting matches this in both directions (`mm-loss` between `mm-delay` and `mm-link` matches it for data only); Mahak uses no loss, so nothing in its semantics depends on the choice. Rejected: netem (a second model and a fourth namespace). |
| D17 | Cross traffic is generated by the guest runner, through the same bottleneck, told apart by sender port | No extra hosts or namespaces, deterministic under the seed, and the link records already carry ports. Rejected: replaying captured traffic (not reactive), and observing cross flows (the label is the share they take, not their state). |
| D18 | Mahak's driver ported onto scikit-learn, not run from Mahak's repository | Mahak vendors an old modAL that breaks on scikit-learn 1.8 and later, and the algorithm itself is about 30 lines. The port is checked against the unmodified original, run in a separate environment, on a shared synthetic oracle (`tests/test_mahak.py`). |
| D19 | The native tier (D3: Linux 7.0 on Ubuntu 26.04 hosts, as root, in real time) is dropped | No environment we have can easily boot a patched kernel, and Ubuntu 26.04 ships stock 7.0, so a native tier could run Cubic, BBRv1 and the agent controllers but never BBRv3 or later kernel changes; packaging a custom image is too much work for what it adds. Cost: UML's virtual time is never checked against a real-time kernel, so effects of CPU cost (free under `inf-cpu`) on TCP are not measured. Its code was last present in commit `faad5be` (`src/expliccit_env/native.py`, `scripts/tier2-setup.sh`). |
| D20 | Sage's step is ported into the guest, and is the only implementation | Training data and deployment then use the same code, with no train/serve skew, and the message equals Sage's, so Sage's and expliCCit's column choices apply unchanged. Rejected: computing features on the host from per-ACK records (a second implementation for deployment), and feeding raw `TCP_INFO` to the graph (it would have to reimplement Sage's windows in ONNX). The raw `TCP_INFO` of every step is kept in `steps.bin` for other feature designs. |
| D21 | One policy contract: the whole message in, the graph selects and scales, optional recurrent state | Any network design (MLP, recurrent, a different column set, uncertainty heads) runs without guest changes. Replaced `tcpinfo-v1` (8 features in, absolute cwnd and pacing out), which no policy used. |
| D22 | The trainer lives here, on PyTorch, with architectures and algorithms chosen by name, and exports one ONNX graph per policy | A network redesign is a registry entry, not a new pipeline, and the scaling travels inside the graph, so deployment needs no scaler files. expliCCit's layer layout and parameter names are kept, so its released checkpoints load unchanged (from a local copy; its code is not in this repository). Splits are by path, never by row, so held-out numbers are not training accuracy. Default scaling is robust (quantile clipping, then standardization); TD3+BC is the default net2 objective, with expliCCit's own kept for comparison. |

## Known ceilings (`ponytail` markers in the code)

- One global record ring and one global agent action per kernel: fine for one
  episode per UML guest.
- Per-ms u8 trace counts: at most 255 opportunities per ms (3 Gbit/s) and
  traces up to about 65 s.
- The record reader polls every 20 ms of virtual time.
- `parse_mm_log` (Mahimahi text logs) loops in Python; it is for fixtures and
  reference logs, not campaign data.
- Labels v1 are the candidates listed in `labels.py`.
- At most 16 cross flows per episode, all sender to receiver (no reverse
  traffic sharing the ACK path).
- Mahak's metrics (`utilization`, `mean_delay_ms`, powers) are computed over
  the link log as `mm-th-delay` would, so with cross traffic they describe
  the whole link; `goodput_mbps` and `drop_fraction` (the share of its
  packets reaching the bottleneck that the full queue dropped) are the
  observed flow's own.
- The step loop polls every 1 ms after an idle step, where Sage busy-polls,
  and refreshes pacing once per step, where Sage's patched kernel may pace
  per ACK (its source is not public).
- The trainer exports with torch's TorchScript-based ONNX exporter, which
  emits simple ops `tract` runs; move to the dynamo exporter (it needs
  `onnxscript`) when the old one is removed. Finiteness is tested as
  |x| < 3e38 in the graph because `tract` does not analyse IsInf and IsNaN.
- **Inference is free (parked).** Under `inf-cpu` a policy's inference costs
  no virtual time, while a real sender spends time on it (Sage crosses
  shared memory into Python every step). Plan: measure inference time
  outside the environment, then add it back as a fixed decision delay.

## Extending

- **A new candidate LCF (label).** Add a pure function to
  `src/expliccit_env/labels.py` that maps an episode's records and constants
  to one value per observation row, and a column in `labels_table`. Existing
  datasets keep `link_events` and `link_opp`, which hold what a new label
  needs, but nothing reads them back yet: today a new label means rerunning
  the episodes (or keeping raw records with `--keep-raw`).
- **A new trace shape.** Add a variant next to `MahakStep` in `spec.py` with a
  `timestamps()` method returning Mahimahi trace timestamps; the kernel only
  ever sees per-ms counts. `TraceFile` already accepts any Mahimahi trace
  (pinned by sha256).
- **A learned controller.** Export it to ONNX under the `sage-v1` contract
  (`src/expliccit_env/policy.py`: input `obs` float32 [1, 77] and an optional
  `state`, output a = log2 of the window multiplier) and use
  `{"kind": "agent", "mode": "hold" | "reno", "policy": {"path": ..., "sha256": ...}}`
  as the spec's `cc`.
- **A new kind of cross traffic.** Add a variant next to `TcpCross` and
  `UdpCross` in `spec.py` and `guest_plan.py`, and next to `CrossFlow` in
  `guest/src/plan.rs`, then start it in `guest/src/cross.rs`. It must sleep
  on absolute deadlines and stop when `Running::finish` shuts its sockets.
- **A new in-kernel congestion control.** Build it into the UML kernel
  (`kernel/uml.config`) or as a module next to `tcp_agent`, then add its name
  to `StockCC` in `spec.py`.
- **A kernel change.** Commit it under `third_party/linux` as its own commit;
  `scripts/check-kernel.sh` checks every such commit, and
  `git diff e8eadca -- third_party/linux` lists them all. Record ABI changes go
  in `kernel/modules/expliccit_rec.h` together with `guest/src/recorder.rs`
  and `src/expliccit_env/records.py` (bump `EXPL_REC_ABI`;
  `tests/test_records_abi.py` checks the layouts agree).
