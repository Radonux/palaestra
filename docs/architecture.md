# Architecture

How one episode runs and why it is built this way, for a reader who wants
to understand or extend it. The guarantees are in
[README.md](../README.md#guarantees), the checks in
[validation.md](validation.md).

## Data flow and trust boundaries

    spec / campaign JSON ─► spec.py (parse) ─► guest_plan.py: plan.json, *.trace, policy-<hash>.onnx
       ─► uml.py: UML with time-travel=inf-cpu, the spec's seed, a wall-clock timeout
       ─► guest runner, PID 1 (Plan::load parses again): namespaces, qdiscs, modules, flows,
          Sage's step loop, policy, live truth; /dev/expl_rec ─► one file per record type
       ─► episode.py (parses result.json, records, steps again): validity checks,
          labels.py, steps.py, Mahak's metrics
       ─► shard: episodes, obs, labels, link_events, link_opp, agent_action, steps,
          cross_steps, flow_seconds (.parquet), DONE

Everything that crosses a process or kernel boundary is parsed once where
it enters, into types that cannot hold invalid values: specs as pydantic
models with closed sums, the plan in Rust, netlink options with the
kernel's `nla_parse_nested`, agent actions in `agent_write`, policy graphs
by input name, records and steps as numpy dtypes checked for size, type,
ABI and sequence. The host re-checks what the guest already checked. An
episode that fails any check becomes a `failed` row with its reason.

## One episode

    snd 10.0.0.1 s0 ---- p0 10.0.0.2 [path] 10.0.1.2 p1 ---- r0 10.0.1.1 rcv
                          ^ uplink mmlink           ^ downlink mmlink

Three namespaces (sender, path, receiver) are joined by veth pairs, with
the bottleneck in the middle. Every skb is orphaned on entering the path,
so the sender's TCP Small Queues see a remote bottleneck, as a Mahimahi
shell looks to a server outside it. Data cross the delay then the link,
ACKs the reverse, as Mahimahi nests them. IPv6 is off and ARP bypasses the
model. The receiver connects at trace time 0; both ends set `TCP_NODELAY`.

TCP settings are Sage's and ccBench's setup scripts': `tcp_rmem` and
`tcp_wmem` up to 64 MB, `rmem_max` and `wmem_max` 128 MB, no cached metrics,
no autocorking. One limit is sized to the path instead: the send buffer
(`EpisodeSpec._send_buffer`) grants what Linux's autotuning asks for a
window of four times the packets the path holds (peak capacity over 1 s
times the RTT of a full queue) at BBR's factor, 12 times 2304 bytes per
packet the path holds, at least 10 segments and at most 64 MB. Stock
controllers reach about twice the path and never meet it; a window far
beyond the path does, which a real NIC and CPU would stop but which costs
nothing in virtual time ([validation.md](validation.md#tcp-buffer-limits)).

Cross traffic shares the path in the same direction. Cross flow `i` sends
from port `6000 + i`, the observed flow from 5001, and ports tell flows
apart in the records. A TCP cross flow runs a stock controller or an agent
with its own policy, step loop and noise; a UDP cross flow sends one
1500-byte packet every `12000 / rate` seconds. Only the observed flow is
observed per ACK. A TCP cross flow with `extra_rtt_ms` has its ACKs held
that much longer by `sch_flowdelay` on the receiver's device.

A TCP cross flow sends only inside its planned window, but it does not
start exactly at `start_ms`: its receiver connects then, and the flow
begins when the handshake completes. Behind a standing queue that can
take seconds, or fail if the SYN-ACKs are dropped. The episodes table
records when each flow connected (`cross_connected_ms`, null for UDP and
for a flow that never connected, which then sent nothing), and the
privileged critic's rival counts read that time, not the plan. At
`stop_ms` both ends are aborted: the sender's unsent data is discarded
(a graceful shutdown would still send it, for seconds on a deep buffer)
and a handshake that completes later is refused. After the stop, data
already in flight reaches the link for one more one-way delay, a few
resets answer ACKs in flight for about one RTT, and the flow's packets
already queued at the bottleneck still drain; the receiver answers
those with resets on the ACK path.

## The link model (`sch_mmlink`)

A port of `third_party/mahimahi/src/frontend/link_queue.cc` as a qdisc,
one per direction:

- Time is integer milliseconds. Trace entry `i` is the number of 1504-byte
  delivery opportunities at ms `i`; the trace repeats with its last
  timestamp as its period.
- Before a packet arriving in ms `a` is queued, every opportunity up to `a`
  is used, so no packet departs in its arrival ms. The head packet departs
  at the opportunity that carries its last byte, and droptail limits count
  queued packets only.
- A packet's size includes Mahimahi's 4-byte TUN header; GSO packets are
  segmented on entry. The delay line releases each packet exactly
  `delay_ns` after entry.
- Random loss (`mm-loss`) drops each entering packet independently before
  the delay and the link, so a lost packet never uses capacity.

Events are processed in time order at their semantic instants (opportunity
and release times), never when the code happened to run, so UML's clock
tick and timer slack cannot change a decision. The trace travels as one
netlink attribute of per-ms u8 counts.

## Time and determinism

`time-travel=inf-cpu` makes CPU work free: virtual time advances only when
the guest idles or sleeps (findings F1 to F5 in
[validation.md](validation.md#feasibility-findings)). So every episode
needs a timed bottleneck, and the host enforces a wall-clock timeout; guest
code must block or sleep, never spin (the record reader sleeps 20 ms
between drains, flows sleep until deadlines); and each clock read advances
virtual time one 256 ns tick, so sender timestamps carry that grain.

With `CONFIG_UML_SEEDED_RANDOM` (our patch), `um_seed=` and a fixed
`time-travel-start=`, an episode is a pure function of the kernel build,
the guest build, the plan and the seed. A guest change can shift the
kernel's random stream, so seeded runs compare across guest versions only
on Cubic or over many seeds.

## Records

`kernel/modules/expliccit_rec.h` defines every record (ABI 4): a 16-byte
header (type, length, sequence, time) and a body. All producers append to
one ring in `expl_rec`, so an episode has one global sequence; a full ring
drops whole records, and the guest and the host both detect the gap. The
guest writes one file per type, which the host maps as a packed array.

| Type | Producer | Meaning (at the event's own instant) |
|---|---|---|
| META | expl_rec | ABI, HZ, ring size |
| LINK_ARRIVAL, DROP, DEPART | sch_mmlink | Mahimahi's `+`, `d`, `-` events with queue state, ports and TCP seq/ack; a drop names its cause |
| LINK_OPP | sch_mmlink | Mahimahi's `#` events, one per ms; the trace is the authority on capacity |
| LINK_STATS | sch_mmlink | counters at deletion; a nonzero error counter invalidates the episode |
| TCP_OBS | expl_tcpobs | sender state just before each incoming segment is processed |
| AGENT_ACTION | tcp_agent | an accepted policy action, with its socket's port |

`TCP_OBS` comes from the `tcp:tcp_probe` tracepoint at the top of
`tcp_rcv_established`, so record k is the state left by segment k-1.

## Ground-truth labels

`labels.py` computes labels as pure functions of an episode's records,
for the data direction. `labels_at` gives them at any instants: every
sender observation (`labels.parquet`, row for row with `obs.parquet`) and
every step (`label_*` in `steps.parquet`).

A windowed label covers the whole trace ms just before the sample. The
steps store each one twice: over 100 ms under its own name (expliCCit's
window, and the live truth's), and over 10 ms as `<name>_10ms`.
`labels.parquet` has a row per ACK and keeps the 100 ms window only.

expliCCit's seven LCFs:

| LCF | Definition |
|---|---|
| `queue_delay_mean_ns` | mean queueing delay of the packets (all flows) that left the bottleneck in the window; NaN if none did |
| `queue_pkts` | the queue after the last link event, without the packet in transit |
| `capacity_bps` | the trace's delivery opportunities in the window, in bit/s |
| `free_buffer_pkts` | the packet limit minus the queue; NaN without a limit |
| `busy_fraction` | bytes sent over bytes the trace offered in the window; NaN at zero capacity |
| `one_way_delay_ns` | one delay line (Mahak's and expliCCit's "mRTT") |
| `receive_rate_bps` | the observed flow's bits leaving the bottleneck per second |

Other labels of the instant: `queue_bytes`, `queue_delay_ns` (the latest
data packet's exact delay), `base_rtt_ns`, `buffer_pkts`, `buffer_bytes`,
`buffer_fill`, `random_loss_rate`, and:

| Label | Definition |
|---|---|
| `flow_queue_pkts` | the observed flow's packets in `queue_pkts`; over all flows they add up to it |

Other windowed labels: `drop_fraction` (the flow's packets dropped by the
full queue over those that reached it), `cross_traffic_share`, and:

| Label | Definition |
|---|---|
| `arrival_rate_bps` | bits of every flow reaching the bottleneck queue per second, those the full queue then dropped included |
| `flow_arrival_rate_bps` | the same for the observed flow alone |
| `queue_min_pkts` | the smallest queue at any instant of the window (the standing queue); NaN for an empty window |
| `active_flows` | flows with a packet reaching the bottleneck in the window |
| `random_drop_fraction` | the flow's packets the random loss dropped before the bottleneck, over those it sent toward it; NaN if none |
| `ack_queue_delay_mean_ns` | `queue_delay_mean_ns` of the ACK direction's link: delay the sender sees that the bottleneck queue did not cause |
| `capacity_ahead_bps` | `capacity_bps` of the window that starts with the sample's ms; the one label that looks forward |

### Composite LCFs

A composite is a fixed formula over the label columns of one window
(`train/lcfs.py`). It is not stored: the trainer computes it from the
stored columns when an LCF set names it, so a new composite needs no new
data. Each comes in both windows, as the labels do. A set that holds a
composite beside the columns it is made of is redundant by construction.

With $`C`$ the capacity, $`U`$ the busy fraction, $`y`$ the arrival rate,
$`q`$ the queue in bytes, $`R`$ the base RTT and $`N`$ the active flows:

| Composite | Formula | Source ([lcf-concepts.md](lcf-concepts.md)) |
|---|---|---|
| `departure_rate_bps` | $`C\,U`$ | RFC 5136 (usage), HPCC |
| `available_capacity_bps` | $`C\,(1 - U)`$ | RFC 5136 |
| `excess_demand_bps` | $`y - C`$ | XCP |
| `load_factor` | $`y / C`$; TIMELY's queue growth rate is this less 1 | OSU scheme, with the target at capacity |
| `bdp_pkts` | $`C R`$, in 1504-byte packets | CCmatic |
| `littles_delay_ns` | $`q / (C\,U)`$ | PIE |
| `normalized_inflight` | $`q / (C R) + U`$ | HPCC |
| `normalized_power` | $`\bigl(1 + q / (C R)\bigr)\, y / C`$ | PowerTCP |
| `cross_arrival_rate_bps` | $`y`$ less the observed flow's arrival rate | Dovrolis 2001 |
| `fair_share_bps` | $`C / N`$ | RCP |
| `sending_ratio` | the flow's arrival rate over its receive rate | Aurora, measured at the link |
| `rtt_ratio` | $`(R + \text{mean queueing delay}) / R`$ | Aurora, Remy, on the true delays |

A ratio is NaN where its denominator is 0.

## Sage's step and the policy contract (`sage-v1`)

`guest/src/step.rs` ports the step of Sage's `sage.cc` (`CntThread`,
commit `eedd281`) and is the only implementation, for training data and
deployment alike. Every 10 ms of virtual time it reads `TCP_INFO`; once the
ACK state has moved it builds Sage's 77-value message (RTT, delivery rate
and window, their statistics over 10, 200 and 1000 steps), Sage's quirks
included, and otherwise polls every 1 ms. Columns 1 and 75 (Sage's nominal
capacity and reward) are 0, column 76 is the log2 window change since the
last step, and column 77, expliCCit's addition, is the min RTT.

A policy is an ONNX graph (`policy.py`) with inputs matched by name: `obs`
[1, 78], and optionally `state` (a recurrent state fed back from its second
output), `truth` [1, 7] (the true LCFs) and `cwnd` (the current window, so
a graph can answer a = log2(target) - log2(cwnd)). It returns a, the log2
window multiplier, and selects and scales its own columns. The window is
Sage's accumulator from 10 packets, times 2^a, rounded up, in [4, 20000];
pacing is cwnd mss / srtt, refreshed each step. A non-finite a, or |a| > 8,
fails the episode. `noise_sigma` adds seeded Gaussian noise to a, each draw
held for `noise_hold_steps` steps.

`guest/src/truth.rs` computes the truth live from the downlink's records
with the labels' definitions, draining the ring at the step's instant, so
it equals the labels stored with the step bit for bit (D23).

`steps.bin` holds per step the time, the raw `TCP_INFO`, the message, the
action applied, the window set and the truth; an agent cross flow `i`
writes `steps-cross<i>.bin` (table `cross_steps`, labels for its own port).
`steps.py` adds labels, Sage's reward with the exact capacity, and the
action taken. Every episode also writes `flow_seconds`: per flow and whole
second, bytes, packets and queueing delays. `campaign run --tables` keeps
only the named tables; most experiments keep only `steps`.

## Agent controllers and Mahak's driver

The step loop writes each action to `/dev/expl_agent`, naming its socket
by file descriptor (agent ABI 2); `tcp_agent` keeps it in that socket's
congestion-control state and applies it at the next ACK, so any number of
agents share a bottleneck. `agent` holds window and pacing (Sage-style),
with the pacing computed by the guest once a step as cwnd * MSS / srtt.
`agent_pure` is Sage's own controller ("pure" with DeepCC pacing, as
Sage ships it in its 4.19.112-0062 kernel and as `sage.cc` configures it;
read from that kernel's binary, whose source is not public): the same
window, set once per action, paced at every delivery at max(cwnd,
packets_out) * MTU / srtt, with a send buffer of three windows. Sage's
and expliCCit's runs use `agent_pure`; our own policies were trained and
evaluated under `agent`, which paces 3.5% slower and less often.

Every model runs on Linux 7.0's stack as it is: `tcp_info`, loss
detection, rate sampling and the rest are mainline's, and nothing of 4.x
is emulated, so results may differ from the models' 4.x deployments as
the kernel has changed since. What a model needs beyond the stock
interfaces lives in its controller in `tcp_agent`: Sage's pacing in
`agent_pure`, and Orca's per-period statistics in `agent_cubic`, read
through `TCP_CC_INFO`.

`agent_rate` (Aurora-style) is rate-based: each action multiplies a
sending rate (from 6 Mbit/s, kept in 0.5 to 300 Mbit/s, as PCC-RL's
deployment does) that the module paces at under a window of 20000, and
each step lasts half the previous step's mean RTT, Aurora's monitor
interval as PCC-RL's simulator, where it was trained, sets it
(PCC-Uspace also makes an interval last at least 5 packets; this does
not, and at 5 packets Aurora's send ratio reads 1.25 either way).
`TCP_CC_INFO` on it returns that interval (its length, packets sent,
delivered and lost, and the mean of its RTT samples and of their second
half less their first, from a per-socket buffer of samples) and starts
the next.

`agent_cubic` (Orca-style) sets the window once per action, to the
kernel's current window times 2^a, and lets Cubic grow it in between. The
Cubic is Linux 7.0's, copied into `kernel/modules/cubic.h` because the
kernel exports none of it. `getsockopt(TCP_CC_INFO)` on an `agent_cubic`
socket returns Orca's statistics for the period since the last read (RTT
samples, mean rate sample, lost bytes, period length, min RTT) and starts
a new period; a policy that declares the input `cc_info` gets them at each
step. An agent may step at its own period (`AgentCC.period_ms`, Orca's
20 ms) instead of Sage's 10 ms.

`mahak.py` ports Mahak's `mahak_online.py` and its `helper.py`: the grid,
min-max scaling and PCA, the Gaussian process (RBF, length scale 0.1, 10
restarts, `random_state=100`), and the query rule (largest predictive
standard deviation). Each query is one 20 s UML episode. A failed query is
recorded with its reason and never becomes a label.

## Experiments in GitHub Actions

An experiment is one JSON file under `experiments/` (format in
`tools/ci/experiment.py`), dispatched with
`gh workflow run experiment.yml -f experiment=experiments/NAME.json`.
`.github/workflows/experiment.yml` runs plan (expand the cells, run
`prepare`), uml (the UML build from the Actions cache, keyed by the git
trees it depends on), cells (`run` per cell: at most 256, 20 at a time, 6 h
each, on 4-core runners), collect (only once every cell succeeded) and
publish: release `exp-NAME-RUN`, with the experiment's JSON and
`summary.md` in its notes, tagged at the producing commit or, when main's
workflow files have changed since, at main's head.

Artifacts only carry data between jobs and expire (7 days; the UML build
after 1). Releases are the record: every successful run is published, and
`republish.yml` publishes a run whose publish job failed while its
artifact lasts. A later experiment reads a release through
`"inputs": {"NAME": "release:TAG"}`. Local files (weights) become release
`TAG` through `scripts/stash.sh TAG FILE...`: it pushes a branch
`stash/TAG`, and `stash.yml` publishes the files and deletes the branch, so
they never enter main's history; `stash-sweep.yml`, dispatched by hand,
deletes any stash branch a ruleset kept once its release holds every file.

## Benchmark suites

Two outside benchmarks run unchanged on the kernel controllers and
policies ([results](results.md#the-split-costs-little)).
`experiments/ccbench.py` plans ccBench's `solo` and `friendly` suites
(github.com/Soheil-ab/ccBench at `49b5ada`) and three more (`loss`,
`copies`, `competitors`); `ccbench_report.py` keeps ccBench's leagues but
measures at the bottleneck, not through Pantheon's tunnel.
`experiments/ccperf.py` plans the 59 of ccperf's 121 experiments
(gitlab.com/ccperf/ccperf at `18d1620`) this emulator can express; the rest
need time-varying delay or loss, AQM, ECN, ACK compression, a policer or
Poisson UDP. Each planner's docstring names its divergences.

## Known limits

Limits marked `ponytail` in the code name their upgrade path there.

| Limit | Consequence |
|---|---|
| Inference and decisions cost no virtual time | a policy acts the instant it reads its step; a fixed decision delay is UNRUN ([measured cost](results.md#the-split-costs-little)) |
| One bottleneck: droptail, fixed delay, i.i.d. loss | no AQM, ECN, jitter, bursty loss or multiple bottlenecks |
| One decision per 10 ms step | pacing refreshed per step, where Sage's kernel may pace per ACK |
| At most 64 cross flows, all sender to receiver | no reverse traffic on the ACK path |
| Several step loops each sleep 1 ns before reading their truth | at 64 agents, 1 of 7,909 steps saw a truth that differed from the labels; labels are always exact |
| `truth` carries the seven LCFs only, over 100 ms | arm A0 (the full path state) is unavailable; a policy that reads the truth (the oracle arm, a patched belief) cannot use the newer labels, the 10 ms window or a composite |
| Labels are stored when an episode is processed | a label added later is computed from the stored records (`dataset.read_episode`, then `labels.labels_at`); a shard written without `link_events` and `link_opp` (`--tables steps`) must be rerun |
| Per-ms u8 trace counts | at most 3 Gbit/s and about 65 s per trace |
| Mahak's metrics describe the whole link | with cross traffic, `utilization` and `mean_delay_ms` include other flows |
| Virtual time is never checked against a real-time kernel | the effect of CPU cost on TCP is not measured (D19) |

## Decisions

Each row records a decision as made; a later row may change an earlier
one. D1 to D5 came from the project brief, the rest from the build.

| | Decision | Reason, and what was rejected |
|---|---|---|
| D1 | Real kernel TCP | From the brief: learned controllers are compared with the kernel's own, unmodified. |
| D2 | The kernel runs as UML in virtual time | No root or KVM, any host, reproducible; confirmed by the M0 feasibility probes (2026-10-30). |
| D3 | A native tier on Linux 7.0 hosts in real time | Dropped by D19. |
| D4 | Linux 7.0 only | From the brief. A tripwire (`uml.yml` with `mainline`, `tools/ci/mainline_tree.py`) shows on demand where our patches stop applying to the newest mainline. |
| D5 | Custom congestion controls as kernel modules | From the brief; M0 confirmed it: BPF `struct_ops` cannot register in UML (no BPF JIT on arch/um). |
| D6 | The link model is a kernel qdisc | In the kernel under test, without per-packet crossings, at exact instants. Rejected: Mahimahi in the guest; netem plus a separate link. |
| D7 | One record ring, one global sequence, split by type in the guest | Loss is detectable, events are ordered, decoding is vectorized. |
| D8 | Observations from the `tcp_probe` tracepoint | UML has no kprobes; stock controllers stay unmodified. Rejected: polling `TCP_INFO`. |
| D9 | arch/um HZ patched to 1000 | At UML's fixed 100, TCP's jiffy timers would be 10 times coarser than on an HZ=1000 host kernel. |
| D10 | A seeded-randomness patch for UML | Byte-reproducible episodes; host entropy leaked in three places. |
| D11 | Kernel and Mahimahi vendored as unmodified imports plus patch commits | Self-contained builds; the patch series is a `git log`. |
| D12 | Parquet shards with a DONE marker | Few large files, atomic visibility, rerunnable shards. |
| D13 | ONNX via `tract` in the guest | Pure Rust, static, deterministic, no virtual-time cost. |
| D14 | BBRv3 from google/bbr as a module | Its 12 prerequisite commits imported unchanged; our commit `56f58a1` makes their app-limited checks opt-in, and `tcp_bbr3` loads only for `bbr3`, because building it in changed seeded BBRv1 runs. |
| D15 | Pinned kernel build metadata | uname's version seeds the guest's random pool; pinning it keeps seeded runs comparable across kernel builds. |
| D16 | Random loss in `sch_mmlink`, before the link | One event order; lost packets never use capacity, so labels stay exact. Rejected: netem. |
| D17 | Cross traffic from the guest runner through the same bottleneck, told apart by port | Deterministic, reactive, no extra hosts. Rejected: replayed traffic. |
| D18 | Mahak's driver ported onto scikit-learn | Mahak's vendored modAL breaks on scikit-learn 1.8; the port is checked against the original. |
| D19 | The native tier (D3) dropped | No environment here boots a patched kernel easily, so it could never run BBRv3. Cost: virtual time is never checked against real time. Its code was last in `faad5be`. |
| D20 | Sage's step ported into the guest, the only implementation | No train/serve skew; Sage's and expliCCit's column choices apply unchanged; raw `TCP_INFO` is kept for other designs. |
| D21 | One policy contract: the whole message in, the graph selects and scales | Any network runs without guest changes. It replaced `tcpinfo-v1`, which no policy used. |
| D22 | The trainer lives here, on PyTorch, exporting one ONNX graph per policy | Scaling travels in the graph; expliCCit's parameter names are kept so its checkpoints load. |
| D23 | A policy's truth is computed live from link records with the labels' definitions | Oracle and patched policies need exactly the training truth during the episode. Rejected: a second definition in the kernel; replaying episodes. |
| D24 | Large runs are GitHub Actions experiments, every result a release | Twenty 4-core runners, free for a public repository; artifacts expire. Rejected: Slurm arrays (no cluster). |
| D25 | Each agent action names its socket by file descriptor and lives in that socket's state | Any number of learned flows with no kernel limit; single-agent episodes reproduce exactly. Rejected: slots keyed by `SO_MARK`. |

## Extending

- A label: add it to `labels_at` in `labels.py` (it feeds `labels.parquet`
  and the step columns) and rerun the episodes. For a policy to read it,
  add it to `guest/src/truth.rs` and `policy.TRUTH_COLUMNS`;
  `tests/test_records_abi.py` and `tests/test_truth.py` then check the
  step record and that guest and host agree.
- A trace shape: a variant next to `MahakStep` and `PeriodicLevels` in
  `spec.py` with `timestamps()`; `TraceFile` takes any Mahimahi trace.
- A kind of belief for net1 (beside a value, an interval and a
  distribution): a subclass of `beliefs.Beliefs` (its width, means,
  patch with the truth, faults) and the net1 module that states it.
  Nothing else asks which kind a net1 has; `tests/test_e2e.py` then
  trains, exports and runs it with every net2.
- A net2 input or window target: a `Front` option or a target module in
  `train/graph.py`. Training reads net2's inputs through the same front
  end the exported graph runs, so the two cannot drift apart.
- A learned controller: an ONNX graph under `sage-v1`, used as
  `{"kind": "agent", "mode": "hold", "policy": {"path": ..., "sha256": ...}}`
  for the observed flow or a TCP cross flow.
- Cross traffic: a variant next to `TcpCross` and `UdpCross` in `spec.py`
  and `guest_plan.py` and next to `CrossFlow` in `guest/src/plan.rs`,
  started in `guest/src/cross.rs`, sleeping on absolute deadlines.
- A congestion control: a module (enabling it in `kernel/uml.config`
  changes the kernel image), loaded only where used, named in `StockName`.
- A kernel change: its own commit under `third_party/linux`, recorded in
  `third_party/README.md`. A record layout change touches
  `expliccit_rec.h`, `guest/src/recorder.rs` and `records.py` together and
  bumps `EXPL_REC_ABI`.
