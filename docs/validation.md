# Validation

What has been checked, how, and what is still unrun. Every "ran" entry was
executed in this environment unless it says CI.

## Environment

- Host: x86_64 cloud container, 4 vCPU, 15 GiB RAM, host kernel 6.18.44,
  Ubuntu 24.04 userland, gcc 13.3.0, Rust 1.97.0, Python 3.13, perl 5.38.
- Kernel: UML from `third_party/linux` (Linux 7.0 plus the patches in
  `third_party/README.md`), `kernel/uml.config`, `time-travel=inf-cpu`.
- Reproduce everything below with:

      scripts/build-uml.sh build/uml && scripts/build-guest.sh build/uml
      git clone https://github.com/Ppazhooh/mahak /tmp/mahak && git -C /tmp/mahak checkout 5077583
      EXPLICCIT_UML_DIR=build/uml EXPLICCIT_MAHAK_DIR=/tmp/mahak uv run pytest -q

  Last result: 63 passed, 2 skipped (the `sage.cc` comparison, see below),
  with the `train` extra installed (`uv sync --extra train`).

## Ran: correctness of the ports and the link model

| Law | Check | Result |
|---|---|---|
| Trace generator equals Mahak's | `test_mahak_trace_matches_helper_py`: Mahak's own `helper.py` writes the file, ours must match bit for bit; 5 settings including float-rounding traps (7 x 0.3, 45 x 1.7) | pass |
| Metric port equals the Perl script | `test_mm_th_delay_matches_perl`: 250 generated logs (175 defined, 75 where Perl dies), output compared as the exact `%.15g` strings | pass |
| Kernel link equals Mahimahi's `LinkQueue` | `test_kernel_link_matches_mahimahi_oracle`: a 14 s Cubic and a 14 s BBR episode cross Mahak's 12 s step (20 to 4 Mbit/s) into a 30-packet buffer; every kernel arrival (with the queue it found), departure, drop and per-ms opportunity record equals an independent Python port of `LinkQueue` fed the same arrivals | pass, both directions |
| Delay line is exact | `test_ack_delay_line_is_exact`: each ACK reaches the sender between 0 and 2 us after its departure plus the configured delay (the 256 ns UML tick) | pass, 0 to 1.3 us |
| Mahak metrics from kernel records equal Perl's | `test_mahak_metrics_agree_with_perl` | pass |
| Oracle laws | `test_link_oracle_laws` (Hypothesis, 300 cases): conservation, FIFO, never departing in the arrival ms, droptail bounds, opportunity accounting | pass |
| Record ABI | `test_dtypes_match_c_structs`: every numpy offset equals the C `offsetof` | pass |
| Reproducibility | `test_same_seed_is_bit_identical` (BBR) | byte-identical records for one seed; different for another |
| Spec laws | `test_spec.py`: a delay is exactly one of mRTT and base RTT; unknown fields rejected; deterministic campaign expansion; loss probabilities in [0, 1); cross-flow windows inside the episode; conditions multiply a campaign without changing its plain arm | pass |

## Ran: closed loop and pipeline

| Check | Result |
|---|---|
| `test_hold_agent_follows_sage_window`: constant a = 0.1, `agent` | every window set equals Sage's accumulator (from 10, times 2^a, rounded up, capped at 20000); one step per 10 ms; column 76 of the next step reports a; the socket holds the set window |
| `test_policy_reads_the_observation`: a = 0.5 x70 - 0.35 | every action equals the policy applied, in float32, to the very message the step table recorded |
| `test_recurrent_state_round_trips`: a counter in the state | the state fed back each step, a = 0.001 k exactly |
| `test_reno_agent_grows_between_actions`: `agent_reno` | Reno regrows the window between actions |
| `test_bad_policy_output_fails_the_episode`: a = 20 | explicit failure row, no data |
| `test_step_table` (Cubic, 24 then 12 Mbit/s): steps, labels at steps, reward | column 76 equals the logged window ratio; the action is the next step's column 76; capacity labels 24 and 12 Mbit/s within 2%; busy fraction in [0, 1]; reward finite, peaking above 10 (Sage's scale tops at 25) |
| `test_step_record_matches_the_guest` | the Python record layout equals the Rust constants |
| `test_campaign_shards_and_tables`: 8 episodes in 2 shards | all ok; obs and labels row-aligned |
| `test_cli_episode_and_mahak_query` | CLI episode and Mahak oracle replacement work |
| Mainline tripwire, run locally before the agent work: kernel.org mainline 7.3.0 plus our patches | patches applied, modules built, 6 UML tests passed |

## Ran: performance (one core per guest, 20 s episodes)

| Setting | Wall time incl. boot | Speed |
|---|---|---|
| 5 Mbit/s, 2 to 4000 pkts | 0.37 to 0.67 s | 30 to 53x |
| 70 Mbit/s grid below | 0.55 to 0.97 s | 21 to 36x |
| 100 Mbit/s, 10 to 100 ms one-way, up to 20000 pkts | 1.47 to 1.52 s | 13 to 14x |
| 100 to 800 Mbit/s step, 8000 pkts | 2.42 s | 8x |

Peak RSS per guest: 85 to 117 MiB.

Aggregate throughput, measured: shard 0/50 of `examples/campaign-icnp25-sample.json`
(40 episodes drawn log-uniformly from the ICNP'25 space, 20 s each, Cubic and
BBR) took 6.6 s with 4 workers on 4 cores, all ok: about 5400 episodes per
core-hour for that mix. Output: 98 MB of zstd Parquet, about 2.5 MB per
episode (link_events 60%, obs 24%, link_opp 14%, labels 4%).

Planning consequence (extrapolated, not measured at scale): one 512-core node
would produce on the order of 2.8 million episodes and 7 TB per hour, so
storage, not CPU, bounds campaigns. A retention option (for example
dropping `link_opp`, which the trace and departures determine) is a cheap
follow-up once the LCF set settles.

## Ran: an illustrative Mahak-style probe (one seed, not a reproduction)

70 Mbit/s, step 0.2 (5x drop at 12 s) or 1.0, mRTT 4 or 40 ms, buffer 100 or
8000 packets, 20 s, metrics over the whole episode as Mahak computes them:

| cc | step | mRTT | buffer | util | mean qdelay ms | delay/mRTT | first_power | retrans |
|---|---|---|---|---|---|---|---|---|
| bbr | 0.2 | 4 | 8000 | 0.983 | 7.2 | 1.79 | 0.549 | 0 |
| cubic | 0.2 | 4 | 8000 | 0.997 | 408.4 | 102.10 | 0.010 | 0 |
| bbr | 0.2 | 40 | 8000 | 0.950 | 69.8 | 1.75 | 0.545 | 0 |
| cubic | 0.2 | 40 | 8000 | 0.960 | 379.8 | 9.49 | 0.101 | 0 |
| bbr | 0.2 | 40 | 100 | 0.940 | 10.7 | 0.27 | 3.524 | 5015 |
| cubic | 0.2 | 40 | 100 | 0.932 | 13.9 | 0.35 | 2.689 | 716 |

(Full 16-row grid: `examples/campaign-bbr2-probe.json`; 16 episodes took
4.9 s on 4 cores.) Mahak reports BBR2
failing in the first cell; BBRv1 here keeps 7 ms of queueing. That is
consistent with the failure being BBR2-specific, but one seed is not a map.

## Ran: BBRv3 (M6)

Google's BBRv3 is imported from google/bbr (`third_party/README.md`): 12
core TCP prerequisites plus the `tcp_bbr3` module, ported to 7.0.

**Stock congestion control stays unmodified.** With kernel build metadata
pinned (so two builds of different source share one random stream per
seed), four seeded 14 s episodes (Cubic and BBRv1, step and constant
traces, 20 to 2000-packet buffers) were run on the kernel without the
series and on the final kernel, and every record file was compared byte
for byte. Steps that led there:

| Kernel | Cubic, shallow | BBRv1 x2 | Cubic, deep (send-buffer-limited) |
|---|---|---|---|
| all 12 prerequisites, Google's code unchanged | identical | identical | links identical; `TCP_OBS` app-limited fields differ |
| plus `tcp_bbr3` built in (`=y`) | identical | **differ** | differ |
| final: app-limited checks opt-in, `tcp_bbr3` a module loaded only for `bbr3` | **identical** | **identical** | **identical** |

Two mechanisms were found and removed: the broadened app-limited checks
changed rate-sample marking for every socket (now opt-in per congestion
control), and building BBRv3 into the kernel changed seeded BBRv1 runs
through boot-time work alone (now a module). The procedure needs two
kernel builds, so it is not in CI; rerun it when the kernel patch set
changes.

**BBRv3 runs correctly over the link model.** `bbr3` is now a case of
`test_kernel_link_matches_mahimahi_oracle`, which also asserts the socket
really ran `bbr3`: pass.

**First look** (one seed per cell, not a map): BBRv1 against BBRv3 on the
probe grid, 70 Mbit/s, 20 s:

| cc | step | mRTT | buffer | util | mean qdelay ms | first_power | retrans |
|---|---|---|---|---|---|---|---|
| bbr | 0.2 | 4 | 100 | 0.983 | 6.5 | 0.609 | 287 |
| bbr3 | 0.2 | 4 | 100 | 0.986 | 5.5 | 0.721 | 30 |
| bbr | 0.2 | 4 | 8000 | 0.982 | 6.7 | 0.583 | 0 |
| bbr3 | 0.2 | 4 | 8000 | 0.988 | 14.6 | 0.271 | 0 |
| bbr | 1.0 | 40 | 100 | 0.942 | 3.8 | 9.897 | 2735 |
| bbr3 | 1.0 | 40 | 100 | 0.951 | 2.8 | 13.480 | 941 |

BBRv3 retransmits far less with shallow buffers, as designed. In Mahak's
BBR2 failure cell (5x drop, 8000 packets, mRTT 4 ms) its queueing delay is
about twice BBRv1's, the direction Mahak reported for BBR2; a map over
seeds is needed before calling it a finding.

## Ran: random loss and cross traffic (M7)

`test_loss_and_cross_traffic`: BBRv1 on 20 Mbit/s, 20 ms base RTT,
60-packet buffer, 10 s; 1% random loss on data and 0.5% on ACKs; 5 Mbit/s
of constant-rate UDP over [2 s, 8 s) and a Cubic flow from 6 s.

| Check | Result |
|---|---|
| Kernel link equals the `LinkQueue` oracle (both directions, all flows) | pass |
| Random drops are Bernoulli per entering packet (within 5 sigma) | data 176 of 16916 (1.04%), ACKs 37 of 7251 (0.51%) |
| Every flow crossed the bottleneck; `TCP_OBS` holds only the observed flow | pass |
| UDP sender wrote exactly 2500 packets (one per 2.4 ms for 6 s) | pass |
| `cross_traffic_share` label: 0 before 2 s; UDP alone; UDP plus Cubic; Cubic alone | median 0, 0.247, 0.359, 0.145 |
| Same seed, second run | byte-identical record files |

The UDP-only share is 0.25 (5 Mbit/s of 1504-byte units on 20 Mbit/s) less
the 1% loss before the link. Cubic gets a small share because 1% random
loss throttles it, while BBRv1 does not back off on loss.

**Loss-free, single-flow episodes are unchanged.** The four seeded stock
episodes of the BBRv3 check above were rerun on the M7 build and compared
with the final BBRv3 build's output: every record file is byte-identical,
and the `LINK_STATS` counters are identical apart from the appended
`drops_random` field (0) that came with record ABI 3.

## Ran: Mahak's driver (M8)

**The port equals Mahak's own driver.** `test_driver_matches_mahak_online`
runs Mahak's unmodified `mahak_online.py` (through `tests/mahak_ref.py`,
which only replaces `helper.oracle_query` with a synthetic function and
seeds numpy) in a separate environment, then runs the port on the same
oracle and seed over `run_mahak.sh`'s space (51000 points, budget 15).

| Mahak's environment | Queries | Map (posterior mean) |
|---|---|---|
| scikit-learn 1.7.2, numpy and scipy as locked here (the test) | 16 of 16 identical | identical |
| Mahak's pinned one: Python 3.9, scikit-learn 1.1.3, numpy 1.23.5, scipy 1.9.3, pandas 1.5.2 (run once) | 16 of 16 identical | max difference 1.4e-13 |

`test_grid_follows_mahaks_arange` pins the grid quirks the port keeps
(axes can pass their maximum, a float step can add a point), and
`test_failed_queries_never_become_labels` the failure rule.
`test_cli_mahak_driver` runs `expliccit mahak` end to end in UML.

**First maps** (`run_mahak.sh`'s command, first_power, one core): 21
episodes, 13 to 16 s wall each for Cubic, BBRv1 and BBRv3, no failed
query. Each map was then compared with 60 held-out grid points run in
UML:

| cc | budget | Spearman | Pearson | MAE | RMSE | lowest predicted value |
|---|---|---|---|---|---|---|
| cubic | 20 | 0.50 | 0.69 | 0.90 | 1.82 | -1.16 |
| cubic | 100 (84 s) | 0.90 | 0.56 | 0.56 | 1.86 | -0.63 |
| bbr | 20 | 0.81 | 0.78 | 3.49 | 5.16 | -1.23 |
| bbr3 | 20 | 0.74 | 0.73 | 3.06 | 3.92 | -0.26 |

At Mahak's default budget the maps rank regions only roughly and predict
impossible negative powers; a budget of 100 ranks well (0.90) while a few
very high-power points (small mRTT) keep the RMSE up. This is the
algorithm's behaviour, not the port's. Since a UML query costs under a
second, budgets in the hundreds are cheap.

## Ran: the trainer (M9)

`tests/test_train.py`: expliCCit's shapes match its released checkpoints
(1,265,927 and 68,097 parameters); the input stage ignores non-finite values
when fitting and maps them to 0 when running; net1 learns a synthetic
capacity (R^2 > 0.9 on held-out episodes) and net2 trains; the deployed
rounding is exact; the export is a sage-v1 graph without IsInf or IsNaN; a
path's episodes share one split; and a freshly exported graph runs in the
guest's closed loop (`tract`), its actions within the action bound.

On real data (`docs/training.md`, "Results so far"): our net1 reaches R^2
0.87 to 0.997 on held-out paths for six of expliCCit's seven LCFs (free
buffer 0.50); expliCCit's released net1 on the same steps gets 0.36 to 1.0
on four and negative R^2 on free buffer and busy fraction; its released
policy runs in our closed loop at 0.58 to 0.95 utilization; our first net2
collapses the window (diagnosed there).

## Ran: teachers as modules, and seeded runs across guest versions (M10)

Building BIC, CDG, HighSpeed, H-TCP, Hybla, Illinois, LP, NV, Scalable,
Vegas, Veno, Westwood and YeAH into `kernel/uml.config` as modules changed
the kernel image, so they are instead built from the unmodified in-tree
sources as out-of-tree modules; the kernel image is bit-identical to
before (md5 3ea96e84). Vegas, Veno, YeAH, LP and Westwood each ran an
episode (utilization 0.94 to 0.99).

The four seeded stock-identity episodes, rerun on the current guest
against the M7 outputs: both Cubic episodes and one BBRv1 episode are
byte-identical; the other BBRv1 episode diverges after 10.4 s, at a BBR
pacing-gain change (its random gain-cycle phase). The same guest on the
modified and the original kernel gives identical files, so the cause is
the guest's own changes since M7, which shift the kernel's random stream.
Seeded runs are reproducible for one build (kernel and guest), not across
guest versions; compare versions on Cubic, or over many seeds.

## Not run to completion: the port of Sage's step against `sage.cc` (M9)

`test_step_matches_sage_cc` (`tests/sage_ref.py`) is written to compile
Sage's own step code, taken by line range from `sage.cc` at `eedd281`
with only the clock and the `TCP_INFO` read stubbed, replay the recorded
readings of a Cubic and a BBRv1-with-loss episode, and compare every
column printed with `%.7f`. Its first run built the harness, but the harness
died with SIGSEGV before printing anything. The cause is not found: the
message fits Sage's 1000-byte buffers (873 bytes at most on that episode),
and the struct layout matches (224 bytes). Debugging means running Sage's
compiled code again, which this session's permission policy then blocked.
The test is skipped unless `EXPLICCIT_SAGE_DIR` is set, so until it passes,
the port is checked only by review and by the tests above. UNRUN.

## CI

`.github/workflows/ci.yml` runs Python (ruff, pyright strict, pytest with the
Mahak reference), Rust (fmt, clippy, x86_64 build), UML (kernel style, UML
build, all UML tests, a CLI campaign), the mainline tripwire and workflow
lint. On GitHub: the Python, Rust and lint jobs passed on the first run; the
UML job's first run exposed a toolchain-selection bug in
`build-guest.sh`, since fixed. See the Actions tab for the current state.

## Not run (and why)

| Item | Status |
|---|---|
| UML against a real-time kernel | Will not run: the native tier was dropped (`docs/architecture.md`, D19), so virtual time is never compared with real time. |
| Link model vs real Mahimahi | UNRUN: needs a root host (any kernel). Mahimahi is vendored; Mahak's `highbdp.patch` does not apply to current Mahimahi (`configure.ac`, `linkshell.cc` changed) and must be ported first. |
| Mahak failure-region reproduction (BBR, Orca, Sage) | Partial: the probe above. Orca's and Sage's trained models are not in this repository; the agent controllers can run them once exported to ONNX under `sage-v1` (for Sage, a recurrent graph with its LSTM state in `state`). |
| Port of Sage's step against `sage.cc`'s own code | UNRUN: see the section above. |
| DeepCC pacing (Sage's patched kernel, `PACE_WITH_SRTT`, coefficient 100) | UNVERIFIED: the kernel patch source is not public; the guest paces at cwnd * mss / srtt. |
| Inference latency in the closed loop | Parked: inference is free in virtual time; measure it outside the environment and add a fixed decision delay later. |
| sparse (`C=1`) on the modules | UNRUN: Ubuntu 24.04's sparse is too old for 7.0. W=1 and checkpatch --strict are clean. |
| Large campaigns on the CSLab Slurm cluster | UNRUN: script generation tested, never submitted. |
