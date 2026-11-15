# Validation

What was checked about the environment, how, and what was not. A check ran
in the development container unless it says CI or names a release.
Commands assume a UML build (`scripts/build-uml.sh build/uml &&
scripts/build-guest.sh build/uml`), `uv sync --extra train`, and, where
named, `EXPLICCIT_UML_DIR=build/uml`, a Mahak checkout at `5077583` in
`EXPLICCIT_MAHAK_DIR`, or a Sage checkout at `eedd281` and g++ in
`EXPLICCIT_SAGE_DIR`. Tests check their property; the numbers quoted come
from the run where each check was first made.

## Summary

| Property | How it was shown | Result | Rerun |
|---|---|---|---|
| The kernel's bottleneck equals Mahimahi's link queue | every arrival, departure, drop and opportunity of Cubic, BBRv1 and BBRv3 episodes across a 20 to 4 Mbit/s step, both directions, equals an independent Python port of `LinkQueue` fed the same arrivals; again under random loss and cross traffic | equal, event for event | `pytest tests/test_uml.py -k "mahimahi_oracle or loss_and_cross"` |
| The delay line is exact | each ACK reaches the sender after its departure plus the configured delay | within 0 to 1.3 us (the 256 ns tick) | `pytest tests/test_uml.py -k ack_delay` |
| The Python port obeys queue laws | Hypothesis, 300 cases: conservation, FIFO, no departure in the arrival ms, droptail bounds, opportunity accounting | pass | `pytest tests/test_mahimahi.py -k laws` |
| Random loss is Bernoulli per packet | 1% on data and 0.5% on ACKs, with UDP and Cubic cross traffic | 1.04% and 0.51%, within 5 sigma | `pytest tests/test_uml.py -k loss_and_cross` |
| Mahak's code is reproduced | trace generation against Mahak's `helper.py`; metrics against its Perl `mm-th-delay` on 250 logs; the driver against unmodified `mahak_online.py` on a shared oracle | bit for bit; exact strings; 16 of 16 queries and the same map | `pytest tests/test_mahimahi.py tests/test_mahak.py` (Mahak) |
| Episodes are reproducible | the same build, plan and seed, run twice, with one or many agents | byte-identical records | `pytest tests/test_uml.py tests/test_multi_agent.py -k "bit_identical or reproducible"` |
| Stock controllers are unchanged by our kernel patches | four seeded episodes on the kernel without and with the BBRv3 series, compared byte for byte | identical (below) | by hand: two kernel builds |
| Records are decoded as written | numpy offsets against the C structs; sequence gaps and truncated files | equal; rejected | `pytest tests/test_records_abi.py` |
| Bad data never becomes labels or metrics | an out-of-range policy action; a failed Mahak query; an incomplete shard | a `failed` row; never a label; an error | `pytest tests/test_agent.py tests/test_mahak.py tests/test_score.py -k "bad_policy or failed_queries or incomplete_shard"` |
| The closed loop is Sage's | constant, observation-reading and recurrent policies; Sage's window accumulator; the window input | every window and action as computed from the recorded message | `pytest tests/test_agent.py` |
| Sage's state message is `sage.cc`'s | `sage.cc`'s own step code, compiled with the clock and `TCP_INFO` stubbed, replays a Cubic and a BBRv1-with-loss episode | every compared column of every step equal (1,382 steps on the Cubic episode); passed 2026-11-04; CI skips it | `pytest tests/test_sage.py` (Sage) |
| The truth equals the labels | a 14 s BBRv1 episode with a capacity step, loss both ways, UDP and TCP cross traffic | all seven LCFs equal at all 1,378 steps, bit for bit and NaN for NaN | `pytest tests/test_truth.py` |
| Agents drive only their own sockets | three agents next to Cubic and UDP; ten copies of one policy | each socket follows its own policy; Jain's index above 0.95; each flow's truth equals its labels | `pytest tests/test_multi_agent.py` |
| An exported policy acts in the guest as in PyTorch | distribution, oracle and patched graphs run in UML from the recorded inputs | within 1e-6 | `pytest tests/test_truth.py tests/test_train.py` |
| The trainer measures what it claims | a distribution net1 on data with known noise covers at its nominal rates; CRPS and quantiles match known values; link splits keep links together | pass | `pytest tests/test_train.py` |
| Scores and the gate follow their rules | log utility, pairing by link, the gate's choice and pass rule | pass | `pytest tests/test_score.py tests/test_gate_report.py` |
| Sage's released policy runs faithfully | the torch rewrite against Sage's own TensorFlow stack on 1,382 recorded steps | within 2.9e-6; the exported graph's rounded action equal at 1,380 steps | release `exp-sage-model-24` |
| expliCCit's released policy runs as its deployed code runs it | expliCCit's own `tcpactor_evalcrr.py` (its model classes and belief correction, executed from its source, which stays private) on the messages our guest recorded in 9 episodes (3 links, clean, 1% loss and Cubic from the start, 15 s) | the action our guest applied equals expliCCit's to three decimals at 11,972 of 11,973 steps, and differs by 0.001 at the other (a rounding tie); the first step of each episode differs by up to 0.007, because expliCCit's first inference skips its belief correction (local run, 2026-11-09, commit 618174b) | by hand, with expliCCit's code |
| `agent_pure` paces as Sage's controller does | Sage's kernel (`linux-image-4.19.112-0062`, from Sage's `linux-patch`) read from its binary: "pure" is `pure_tcp_main` as `cong_control`, which paces every delivery at max(cwnd, packets_out) * MTU * 10^6 / (srtt_us >> 3) (DeepCC pacing as `sage.cc` enables it), with BBR's `sndbuf_expand` of 3 | at every arriving segment of an agent episode, the kernel's pacing rate equals that formula exactly (3,008 of 3,008 segments); `agent`'s pacing is 0.966 of it (MSS, not MTU, refreshed once a step) | `pytest tests/test_agent.py -k pure` |
| Orca's released policy runs as Orca's own code runs it | Orca's own `envwrapper.py` state code, `d5.py` history and `agent.py` Actor (TensorFlow, restoring `model.ckpt-1283529`) against our export (slow start, features from `TCP_CC_INFO` and `TCP_INFO`, actor, window percent) on 600 server messages, after 5 slow-start steps | the window percent equal at 600 of 600 steps, the actor's inputs within 2.8e-7, slow start answering 1.1 and recording nothing ([`exp-orca-model-196`](https://github.com/Radonux/palaestra/releases/tag/exp-orca-model-196)) | `experiments/orca-model.json` |
| Aurora, trained with PCC-RL's code, runs as PCC-RL's own code runs it | for each of three training runs (`exp-aurora-train-199`, `-201`, `-202`), PCC-RL's own `sender_obs.py` features, 10-interval history and `loaded_agent.py` (TensorFlow, the SavedModel `stable_solve.py` writes) against our export (features from `agent_rate`'s `TCP_CC_INFO`, the policy's mean, the rate factor) on 600 synthetic monitor intervals; the same for the policy fed PCC-Uspace's RTT samples (blind) and for the receive rate counting every acknowledged packet (unbiased, against PCC-RL's code given one more packet per interval) | the policy's inputs within 4.6e-5 and the rate factor within 7.7e-6, for every model and variant ([`exp-aurora-model-203`](https://github.com/Radonux/palaestra/releases/tag/exp-aurora-model-203)) | `experiments/aurora-model.json` |
| The learned controllers leave stock controllers unchanged | 12 seeded 30 s episodes (Cubic, BBRv1, BBRv3 and Vegas; a stepping link, 1% random loss and a Cubic competitor; steps recorded) on the guest before (d5d0f21) and after the `agent_cubic`, `agent_pure` and `cc_info` changes (a83b337), with the same kernel image | all 120 record files byte-identical, including BBR's; only the console log differs, by its paths (local run, 2026-11-09) | by hand: two guest builds |
| Every released policy runs on the current guest | each of the 198 `.onnx` assets of the 39 `model-*` releases, alone for 8 s on one link: 178 recurrent graphs with state, 11 of them also reading the truth, 7 stateless graphs and 2 stateless oracles | all 198 ended ok, every action a number; each policy's actions varied (standard deviation 0.095 to 0.296) and used 0.933 to 0.972 of the link ([`exp-onnx-compat-191`](https://github.com/Radonux/palaestra/releases/tag/exp-onnx-compat-191), commit d7144c3) | `experiments/onnx-compat.json` |
| Every kind of net1 trains with every net2 and exports a graph that computes what was trained | single values, intervals and distributions, each with the offline actor (window multiplier or window target) and the recurrent actor taught by DAgger and continued by PPO, on all beliefs or their means; ONNX against torch step by step, and the whole CLI pipeline in UML | pass | `pytest tests/test_e2e.py` |
| The send-buffer limit leaves stock controllers alone | 340 stock-controller episodes with and without it | 3 changed (Hybla), by at most 0.0003 in utilization | below |

On 2026-11-06, without a UML build or the Mahak and Sage checkouts,
`uv run pytest -q` gave 116 passed and 46 skipped (the tests that need
them). The UML tests run in CI (below).

## Feasibility findings

The M0 probes (`tools/m0/`, 2026-10-30) showed that Linux 7.0 builds as a
static UML binary in under two minutes, boots under `time-travel=inf-cpu`
(a guest `sleep 10` measured 10.000064512 s), runs three namespaces with
veth and TCP, loads out-of-tree modules, and lets a module attach to
`tcp:tcp_probe`. arch/um hard-codes HZ at 100 (patched to 1000, D9). It has
no kprobes, and BPF `struct_ops` cannot register (no BPF JIT on arch/um),
so controllers are modules (D5). The code cites these findings by number:

- F1. With no rate limit on the path, CPU work costs no virtual time, so a
  bulk transfer is infinitely fast and a 20 s transfer never ends: every
  episode needs a timed bottleneck, and the host a wall-clock timeout on
  every guest.
- F2. A guest process that makes many syscalls without blocking advances
  virtual time artificially (one jiffy per 10000 unscheduled userspace
  switches), so guest code blocks or sleeps instead of spinning.
- F3. Large queues are fine with `sch_mmlink`: 20000-packet buffers at 100
  Mbit/s run 14 times faster than real time.
- F4. UML's clock ticks every 256 ns, and every clock read from process
  context advances virtual time one tick: sender timestamps carry that
  grain, the link model does not.
- F5. Host entropy reaches UML through `setup_arch()`,
  `arch_get_random_longs()` and the command line; the seeded-randomness
  patch removes all three.

Speed and memory, measured before the step loop existed (20 s episodes,
one core each, boot included): 8 to 53 times faster than real time (13 to
14 times at 100 Mbit/s, 8 times on a 100 to 800 Mbit/s step), peak RSS 85
to 117 MiB per guest, and about 5400 episodes per core-hour on Mahak's
ICNP'25 space at about 2.5 MB of Parquet each, most of it link tables.
Storage, not CPU, bounds large campaigns; `campaign run --tables` keeps
only what is needed.

## BBRv3 and stock controllers

Google's BBRv3 adds 12 commits to the TCP core. With kernel build metadata
pinned, four seeded 14 s episodes (Cubic and BBRv1, step and constant
traces, buffers of 20 to 2000 packets) were compared byte for byte on the
kernel without the series and with it. Two changes broke equality and were
removed: the series' app-limited checks altered every socket's rate
samples (now opt-in per controller, `56f58a1`), and building `tcp_bbr3`
into the kernel changed seeded BBRv1 runs through boot-time work alone
(now a module loaded only for `bbr3`). The final kernel gives identical
files. The extra Linux controllers are likewise built as out-of-tree
modules, leaving the kernel image bit-identical. The check needs two
kernel builds, so it is not in CI.

Seeded runs are reproducible for one build, not across guest versions:
rerun on a later guest, both Cubic episodes and one BBRv1 episode stayed
byte-identical, but the other BBRv1 episode diverged after 10.4 s, at a
random pacing-gain phase, because guest changes shift the kernel's random
stream.

## TCP buffer limits

Until 2026-11-04 the guest kept Linux's own TCP limits for 1 GiB of memory,
a 4 MB send buffer, where Sage's and ccBench's setup scripts give 64 MB.
None of the training, new or cubic-first families' links needs more than
3.5 MB in flight, but 8 of deep-long's 21 validation and 10 of its 27 test
links do; on those, buffer-filling controllers queued less than they
would. With 64 MB, the queue of a Reno flow on 96 Mbit/s, 160 ms and 4 BDP
reaches the full 640 ms, against 213 ms with the old limit
(`test_a_flow_can_fill_a_deep_buffer_on_a_long_fat_path`). The deep-long
family of every affected run was rerun on the same episodes
(`exp-*-deeplong-32` to `-37`): the rules moved by at most 0.02 on the test
links, loss-based controllers fell by 0.2 to 0.9, and every validation
choice stood.

With 64 MB, a window that grows without bound floods the link, since
sending costs no virtual time: one 60 s episode of a delay-threshold
teacher offered 5.4 Gbit/s to a 76 Mbit/s link, 71 arrivals per departure.
A real host with 64 MB would flood too; whether at that rate is UNRUN.
The send buffer is therefore sized to the path
([architecture.md](architecture.md#one-episode)). Stock controllers'
largest windows, measured on 340 episodes, were at most 2.15 times the
path (6.24 for Hybla's slow start). With the limit, all 160 stock and 120
BBR episodes of that sample are identical to 64 MB, and 57 of 60 Hybla
ones (the others moved by at most 0.0003 in utilization); the growing
agent of `test_a_window_far_beyond_the_path_is_bounded` puts 9.0 arrivals
per departure on the link, against 54.4 with 64 MB.

The guest, the kernel and these settings are frozen at `1365f0e`; every run
from #69 on uses them.

## Smaller checks

- `flow_seconds`: each flow's bytes over its seconds equal its total, and
  the delays give the episode's Mahimahi mean delay exactly
  (`test_loss_and_cross_traffic`).
- `sch_flowdelay`: a cross flow with 30 ms extra RTT reports a 50 to 52 ms
  minimum RTT and a base-RTT label of exactly 50 ms
  (`test_a_cross_flow_has_its_own_rtt`); seeded ccperf campaigns without
  extra RTT gave the same `flow_seconds` before and after the qdisc.
- Several step loops: one loop never needed a settle step, but without a
  1 ns one, 16 agents gave 1 step of 6,330 whose queue truth was one
  packet off; with it, 10 and 16 agents matched exactly, and 64 agents
  (the cap) 7,908 of 7,909 steps. Labels are exact either way.
  Single-agent episodes are unchanged by multi-agent support (every table
  equal on three episodes).
- Mahak's maps at its default budget of 20 rank 60 held-out points with
  Spearman 0.50 (Cubic) to 0.81 (BBRv1); a budget of 100 reaches 0.90 for
  Cubic, at under a second per query.
- The tripwire (`uml.yml` with `mainline`) applied our patches to mainline
  7.4-rc1 on 2026-11-02 up to `include/linux/tcp.h`, where one hunk no
  longer applies; the pinned 7.0 tree is unaffected.

## CI

`ci.yml` runs ruff, pyright, pytest with the Mahak reference and the
trainer (UML tests skip), and workflow and shell lint on every commit that
changes more than docs. `uml.yml` runs Rust format, clippy and build,
kernel style (`scripts/check-kernel.sh`), the UML build and tests, and a
smoke campaign when the guest, kernel, modules, `src`, tests or build
change, and the mainline tripwire on demand.

## Not run

| Item | Status |
|---|---|
| The link model against real Mahimahi on the original stack (Ubuntu 16.04, Sage's Linux 4.19, ccBench's Mahimahi) | UNRUN. The kit is `tools/crosscheck`; palaestra's side is `exp-crosscheck-93`. Current Mahimahi would first need Mahak's `highbdp.patch` ported. |
| UML's virtual time against a real-time kernel | Will not run: the native tier was dropped (D19). |
| Inference latency in the closed loop | UNRUN: inference is free in virtual time; the plan is a fixed decision delay. |
| Sage's per-ACK pacing (its patched kernel) | UNRUN: that kernel's source is not public; the guest paces per step. |
| Mahak's failure regions mapped over seeds | UNRUN; only one-seed probes. |
| sparse (`C=1`) on the modules | UNRUN: Ubuntu 24.04's sparse is too old for Linux 7.0; W=1 and checkpatch `--strict` are clean. |
| Slurm array jobs | UNRUN: scripts are generated and checked, never submitted. |
| Training on GPU | UNRUN: no GPU here. |
