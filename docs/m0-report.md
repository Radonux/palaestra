# M0 feasibility report

Status: complete. Results come from `tools/m0/` probes and, after
the link model existed, from the test suite and the benchmark below. Items
not run are marked UNRUN.

## Environment

- Host: cloud container, x86_64, 4 vCPU, 15 GiB RAM, host kernel
  6.18.44 (Firecracker guest), Ubuntu 24.04 userland, gcc 13.3.0, clang 18.1.3.
- Guest: UML built from `third_party/linux` (Linux 7.0, unmodified import plus
  the HZ patch), `kernel/uml.config` merged over `x86_64_defconfig`, booted
  with `time-travel=inf-cpu`, host `/` as a read-only hostfs root.
- Probe kernel for BPF: same plus `kernel/m0-bpf.config` (BPF syscall, BTF).

## Results

| # | Question | Result | Evidence |
|---|---|---|---|
| 1 | Does Linux 7.0 build as UML with time-travel, static link, veth, netem, fq, cubic, bbr? | Yes. 1m44s on 4 cores, static ELF. `build-uml.sh` verified every requested option in the final `.config`. | `scripts/build-uml.sh` |
| 2 | Does it boot in `time-travel=inf-cpu` and keep virtual time? | Yes. Clocksource `time-travel-clock`. `sleep 10` measured 10.000064512 s in the guest. | boot probe |
| 3 | HZ | Upstream arch/um hard-codes `HZ=100` (no prompt). Patched to 1000 in its own commit (`um: set HZ to 1000`); guest reports `CONFIG_HZ=1000`. | `/proc/config.gz` |
| 4 | Do netns, veth, forwarding, netem and TCP work in the guest? | Yes, 3-namespace path (snd, path, rcv) with netem in `path`. | `run-net.sh` |
| 5 | Speed (netem rate 100 Mbit/s, 10 ms each way, iperf3) | Cubic: 20.4 s virtual in 3.0 s wall incl. boot, 95.2 Mbit/s, 0 retrans. BBR: 20.2 s virtual in 3.5 s wall, 94.2 Mbit/s. That is about 6 to 7x real time on one core. | `run-net.sh ... 100mbit 10ms 20` |
| 6 | Speed at 10 Mbit/s, 50 ms each way | BBR: 20.8 s virtual in 1.8 s wall. Cubic: 22.9 s virtual in 256 s wall (slower than real time). Cubic's cwnd grew to 2.4 MB and RTT to 1.6 s against netem's 100000-packet limit; see open item O1. | iperf3 JSON |
| 7 | Do out-of-tree modules load in UML, and can a module attach to `tcp:tcp_probe`? | Yes. `tools/m0/m0_probe` (built with `W=1`, checkpatch `--strict` clean) loads with insmod, finds the unexported tracepoint by name, and counted 17701 hits in a 10 s Cubic transfer. | `run-net.sh ... m0_probe.ko` |
| 8 | Is a run reproducible? | Stock UML: dynamics were identical across runs, but TCP ISNs and ephemeral ports differed (host entropy). With our `CONFIG_UML_SEEDED_RANDOM` patch, `um_seed=` and a fixed `time-travel-start=`, the same plan and seed give byte-identical record files (Cubic and BBR, different output directories); a different seed changes them. | `tests/test_uml.py::test_same_seed_is_bit_identical` |
| 9 | Does BPF struct_ops work in UML (D5)? | No, and it fails earlier than expected. arch/um selects neither `HAVE_EBPF_JIT` nor `HAVE_CBPF_JIT`, so `CONFIG_BPF_JIT` cannot be set, and `net/ipv4/Makefile` builds `bpf_tcp_ca.o` only under `CONFIG_BPF_JIT`. libbpf reports `struct bpf_struct_ops_tcp_congestion_ops is not found in kernel BTF`; registration never reaches the trampoline, so errno 524 is never returned. D5 (kernel modules) stands; no stop condition fires. | `guest-bpf.sh` |
| 10 | kprobes in UML | Not available: arch/um does not select `HAVE_KPROBES`. Observation must use tracepoints. | `arch/um/Kconfig` |
| 11 | Speed with the real link model (20 s episodes, incl. boot, one core) | Cubic/BBR 100 Mbit/s, 10 ms, 100 pkts: 1.52 s (13x). 100 Mbit/s, 100 ms, 20000 pkts: 1.47 s (14x). 100 to 800 Mbit/s step, 8000 pkts: 2.42 s (8x). 5 Mbit/s: 0.37 to 0.67 s (30 to 53x). Records 2 to 89 MB raw per episode; about 2 MB per 14 s episode as zstd Parquet. | `docs/validation.md`, benchmark |
| 12 | Memory per guest | Peak RSS 85 to 117 MiB for 100 to 800 Mbit/s, 8000 to 20000-packet episodes (`mem=512M`/`1024M`). Cores, not memory, bound packing on a 512-core node. | `resource.getrusage` around `run_guest` |
| 13 | Clock resolution | UML's time-travel clocksource ticks every 256 ns, and each clock read from process context advances virtual time by one tick (`arch/um/kernel/time.c`, `timer_read`). Timers fire on the next tick: ACKs reach the sender 0 to 1.3 us after their exact release time, in multiples of 64 ns. | `tests/test_uml.py::test_ack_delay_line_is_exact` |

## Findings that change the design

- **F1. inf-cpu needs a timed bottleneck.** With no rate limiter on the
  path, CPU work costs zero virtual time, so a bulk transfer is infinitely
  fast and virtual time advances only through
  `CONFIG_UML_MAX_USERSPACE_ITERATIONS` (one jiffy per 10000 unscheduled
  userspace switches). A 20 s transfer then never finishes. Every episode
  must have a time-based bottleneck, and the host must enforce a wall-clock
  watchdog on every UML process.
- **F2. Busy-polling userspace distorts time.** The same iteration limit
  means any guest process that makes many syscalls without blocking advances
  virtual time artificially. Guest code must block (read, poll, sleep), never
  spin. Mahak's SCHED_RR `getsockopt` busy loop would be wrong here, not just
  wasteful.
- **F3. Large queues are fine with the real link model.** Item 6's slowdown
  came from the netem probe (100000-packet limit, iperf3). With `sch_mmlink`,
  20000-packet buffers at 100 Mbit/s run at 14x real time (item 11).
- **F4. The clock has a 256 ns grain.** The link model uses exact semantic
  instants and is unaffected; sender observation timestamps are quantized to
  the tick (item 13).
- **F5. Host entropy leaks into UML** through `setup_arch()`,
  `arch_get_random_longs()` and the command line (item 8); fixed by the
  seeded-randomness patch.

## Open items

- O1, O2, O3: resolved (F3, item 8, item 12).
- O4. HZ and `uname -r` on the planned native hosts: dropped with the native
  tier (`docs/architecture.md`, D19).

## Reproduce

    scripts/build-uml.sh build/uml
    scripts/build-uml.sh build/uml-m0 kernel/m0-bpf.config
    make -C third_party/linux O=$PWD/build/uml ARCH=um M=$PWD/tools/m0/m0_probe modules
    tools/m0/run-net.sh build/uml/linux build/m0 cubic100 cubic 100mbit 10ms 20 tools/m0/m0_probe/m0_probe.ko

The guest probes use host binaries through the hostfs root, so the host needs
`iproute2`, `iperf3`, `kmod` and `bc`. The struct_ops probe also needs a
bpftool built from `third_party/linux/tools/bpf/bpftool` (with `OUTPUT=`
outside the tree) and clang with the BPF target; see `guest-bpf.sh`.
