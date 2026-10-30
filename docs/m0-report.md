# M0 feasibility report

Status: interim. Every result below was produced by a command in
`tools/m0/` and is labeled with the environment it ran in. Items not yet run
are marked UNRUN.

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
| 3 | HZ | Upstream arch/um hard-codes `HZ=100` (no prompt). Patched to 1000 in its own commit (`um: set HZ to 1000`); guest reports `CONFIG_HZ=1000`. Tier 2 HZ is VERIFY on the Ampere hosts. | `/proc/config.gz` |
| 4 | Do netns, veth, forwarding, netem and TCP work in the guest? | Yes, 3-namespace path (snd, path, rcv) with netem in `path`. | `run-net.sh` |
| 5 | Speed (netem rate 100 Mbit/s, 10 ms each way, iperf3) | Cubic: 20.4 s virtual in 3.0 s wall incl. boot, 95.2 Mbit/s, 0 retrans. BBR: 20.2 s virtual in 3.5 s wall, 94.2 Mbit/s. That is about 6 to 7x real time on one core. | `run-net.sh ... 100mbit 10ms 20` |
| 6 | Speed at 10 Mbit/s, 50 ms each way | BBR: 20.8 s virtual in 1.8 s wall. Cubic: 22.9 s virtual in 256 s wall (slower than real time). Cubic's cwnd grew to 2.4 MB and RTT to 1.6 s against netem's 100000-packet limit; see open item O1. | iperf3 JSON |
| 7 | Do out-of-tree modules load in UML, and can a module attach to `tcp:tcp_probe`? | Yes. `tools/m0/m0_probe` (built with `W=1`, checkpatch `--strict` clean) loads with insmod, finds the unexported tracepoint by name, and counted 17701 hits in a 10 s Cubic transfer. | `run-net.sh ... m0_probe.ko` |
| 8 | Is a run reproducible? | Two identical Cubic runs gave identical virtual duration (10.359309568 s), throughput and every `/proc/net/netstat` and `snmp` TCP counter. Byte-level record comparison is UNRUN (needs the M2 recorder). BBR is UNRUN. | `diff` of probe outputs |
| 9 | Does BPF struct_ops work in UML (D5)? | No, and it fails earlier than expected. arch/um selects neither `HAVE_EBPF_JIT` nor `HAVE_CBPF_JIT`, so `CONFIG_BPF_JIT` cannot be set, and `net/ipv4/Makefile` builds `bpf_tcp_ca.o` only under `CONFIG_BPF_JIT`. libbpf reports `struct bpf_struct_ops_tcp_congestion_ops is not found in kernel BTF`; registration never reaches the trampoline, so errno 524 is never returned. D5 (kernel modules) stands; no stop condition fires. | `guest-bpf.sh` |
| 10 | kprobes in UML | Not available: arch/um does not select `HAVE_KPROBES`. Observation must use tracepoints. | `arch/um/Kconfig` |

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
- **F3. Wall time depends on queue size, not only on rate.** Item 6 shows a
  pathological slowdown with a very large queue. The Mahak grids go up to
  20000-packet buffers, so this has to be measured with the real link model.

## Open items

- O1. Profile the item 6 slowdown with `sch_mmlink` at large buffers.
- O2. Byte-level determinism of full record streams, Cubic and BBR.
- O3. Memory per instance and instances per node, for Slurm packing.
- O4. HZ and `uname -r` on the Tier 2 Ampere hosts (VERIFY).

## Reproduce

    scripts/build-uml.sh build/uml
    scripts/build-uml.sh build/uml-m0 kernel/m0-bpf.config
    make -C third_party/linux O=$PWD/build/uml ARCH=um M=$PWD/tools/m0/m0_probe modules
    tools/m0/run-net.sh build/uml/linux build/m0 cubic100 cubic 100mbit 10ms 20 tools/m0/m0_probe/m0_probe.ko

The guest probes use host binaries through the hostfs root, so the host needs
`iproute2`, `iperf3`, `kmod` and `bc`. The struct_ops probe also needs a
bpftool built from `third_party/linux/tools/bpf/bpftool` (with `OUTPUT=`
outside the tree) and clang with the BPF target; see `guest-bpf.sh`.
