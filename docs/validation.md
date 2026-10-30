# Validation

What has been checked, how, and what is still unrun. Every "ran" entry was
executed in this environment unless it says CI.

## Environment

- Host: x86_64 cloud container, 4 vCPU, 15 GiB RAM, host kernel 6.18.44,
  Ubuntu 24.04 userland, gcc 13.3.0, Rust 1.97.0, Python 3.13, perl 5.38.
- Tier 1 kernel: UML from `third_party/linux` (Linux 7.0 plus the HZ and
  seeded-randomness patches), `kernel/uml.config`, `time-travel=inf-cpu`.
- Reproduce everything below with:

      scripts/build-uml.sh build/uml && scripts/build-guest.sh build/uml
      git clone https://github.com/Ppazhooh/mahak /tmp/mahak && git -C /tmp/mahak checkout 5077583
      EXPLICCIT_UML_DIR=build/uml EXPLICCIT_MAHAK_DIR=/tmp/mahak uv run pytest -q

  Last result: 35 passed.

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
| Spec laws | `test_spec.py`: a delay is exactly one of mRTT and base RTT; unknown fields rejected; deterministic campaign expansion | pass |

## Ran: closed loop and pipeline

| Check | Result |
|---|---|
| `test_hold_agent_follows_constant_policy`: ONNX constant policy, `agent` CC | 149 actions, cwnd 40 after the first, initial window before |
| `test_policy_sees_tcp_info`: `cwnd' = 0.5 cwnd + 20` | converges to the fixed point 40 through TCP_INFO feedback |
| `test_reno_agent_grows_between_actions`: `agent_reno` | Reno regrows the window between actions |
| `test_bad_policy_output_fails_the_episode` | explicit failure row, no data |
| `test_campaign_shards_and_tables`: 8 episodes in 2 shards | all ok; obs and labels row-aligned |
| `test_cli_episode_and_mahak_query` | CLI episode and Mahak oracle replacement work |
| Mainline tripwire, run locally before the agent work: kernel.org mainline 7.3.0 plus our patches | patches applied, modules built, 6 UML tests passed |

## Ran: performance (Tier 1, one core per guest, 20 s episodes)

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
storage, not CPU, bounds Tier 1 campaigns. A retention option (for example
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

## CI

`.github/workflows/ci.yml` runs Python (ruff, pyright strict, pytest with the
Mahak reference), Rust (fmt, clippy, x86_64 and aarch64 builds), Tier 1 (kernel
style, UML build, all UML tests, a CLI campaign), the mainline tripwire and
workflow lint. On GitHub: the Python, Rust and lint jobs passed on the first
run; the Tier 1 job's first run exposed a toolchain-selection bug in
`build-guest.sh`, since fixed. See the Actions tab for the current state.

## Not run (and why)

| Item | Status |
|---|---|
| Tier 2 runs on Ubuntu 26.04 / Ampere A1 (setup, native episodes, HZ check) | UNRUN: no such host here; this container's kernel has no module support. Code: `scripts/tier2-setup.sh`, `native.py`. |
| Cross-tier fidelity study (Tier 1 vs Tier 2, same specs and seeds sets) | UNRUN: needs Tier 2. Plan: run the HotNets'23 and ICNP'25 grids at matched points on both tiers, compare per-point metric distributions over seeds and per-ACK observation statistics. |
| Link model vs real Mahimahi on Tier 2 | UNRUN. Mahimahi is vendored; Mahak's `highbdp.patch` does not apply to current Mahimahi (`configure.ac`, `linkshell.cc` changed) and must be ported first. |
| Mahak failure-region reproduction (BBR, Orca, Sage) | Partial: the probe above. Orca's and Sage's trained models are not in this repository; the agent controllers can run them once exported to ONNX under `tcpinfo-v1`. |
| sparse (`C=1`) on the modules | UNRUN: Ubuntu 24.04's sparse is too old for 7.0. W=1 and checkpatch --strict are clean. |
| Large campaigns on the CSLab Slurm cluster | UNRUN: script generation tested, never submitted. |
