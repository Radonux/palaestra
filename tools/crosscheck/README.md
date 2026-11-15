# Cross-check kit: palaestra against the original stack

This kit compares palaestra with the stack Sage and expliCCit ran on:
Ubuntu 16.04, Sage's patched Linux 4.19 (`4.19.112-0062`) and ccBench's
patched Mahimahi. The team runs four links there with five controllers and
sends back mm-link's downlink logs; `compare.py` computes the same metrics
from those logs and from palaestra's runs of the same links. It is the
reproduction phase's check that UML with `sch_mmlink` reproduces the
original setup. palaestra's side ran as release
[`exp-crosscheck-93`](https://github.com/Radonux/palaestra/releases/tag/exp-crosscheck-93)
(60 episodes, none failed); the original stack's side is UNRUN.

| File | What |
|---|---|
| `links.py` | The four links as palaestra specs. `write DIR` writes their traces and `links.json`; `plan OUT --sage SAGE_ONNX` plans palaestra's campaign of them (it needs the train extra) |
| `compare.py` | Reads the team's logs and palaestra's episodes and prints the table |
| `experiments/crosscheck.json` | palaestra's side in CI: every link with Cubic, BBRv1, Vegas, Sage's released policy and expliCCit's released policy as deployed, seeds 1 to 3; its release holds the episodes and the kit |

## The links

| Link | What | Capacity | mm-delay (ms, one way) | Data loss | Buffer (packets, each way) | Run (s) | Window (s after the first data packet) |
|---|---|---|---|---|---|---|---|
| a | expliCCit's case-study link | 24 and 48 Mbit/s, alternating every 7 s | 40 | none | 1000 | 45 | 2 to 44 |
| b | constant, 1 BDP | 48 Mbit/s | 20 | none | 160 | 30 | 2 to 29 |
| c | constant, 4 BDP, random loss | 12 Mbit/s | 40 | 1% | 320 | 30 | 2 to 29 |
| d | a cell of ccBench's solo suite | 96 Mbit/s | 80 | none | 5120 | 30 | 2 to 29 |

One BDP is D * B // 6 packets for a one-way delay of D ms and B Mbit/s, as
ccBench's `solo_runall.sh` computes it. Make the files with

    uv run python tools/crosscheck/links.py write kit

or take `kit/` from the release. Each `.trace` is a Mahimahi trace (one
millisecond timestamp per line, each a 1504-byte delivery opportunity,
repeating with its last timestamp as its period), generated from
palaestra's own specs and checked to read back as the same trace.
`b.trace` equals the first 7 s of ccBench's `traces/wired48`; `c.trace` and
`d.trace` deliver what `wired12` and `wired96` deliver from ms 1 on.
`links.json` gives each link's trace and sha256, shell arguments,
duration, window and palaestra spec.

## How palaestra's path maps to Mahimahi

palaestra's bottleneck is `mm-delay`, `mm-loss` and `mm-link` as one qdisc
per direction ([docs/architecture.md](../../docs/architecture.md#the-link-model-sch_mmlink)).
The same path from Mahimahi's shells:

- The sender runs outside every shell and the receiver inside, so data
  cross the downlink and ACKs the uplink. The receiver connects to the
  sender at `$MAHIMAHI_BASE`.
- Nest the shells as `mm-delay D [mm-loss downlink P] mm-link UP DOWN ...
  -- RECEIVER`. palaestra drops a lost packet before its delay line, this
  nesting after it; with i.i.d. loss a lost packet never reaches mm-link
  either way. Sage's harness nests the shells this way, and ccBench's
  `cc_solo.sh` puts `mm-loss` outside `mm-delay`, which is the same.
- palaestra's ACK path has the data direction's trace, packet limit and
  delay, and no loss, as Mahak's does: pass the same trace for both of
  mm-link's arguments, give the uplink the same droptail limit, and add no
  `mm-loss uplink`.
- Use ccBench's patched Mahimahi (`patches/mahimahi.core.v2.2.patch`),
  which raises the shells' txqueuelen to 400000.

## Host setup

Once per boot, on Sage's kernel (`uname -r` prints `4.19.112-0062`):

    bash ~/sage/sage_rl/set_sysctl.sh
    sudo modprobe tcp_bbr
    sudo modprobe tcp_vegas
    sysctl net.ipv4.tcp_available_congestion_control    # must list cubic, bbr and vegas

`set_sysctl.sh` sets what palaestra's guest sets (`tcp_rmem` and
`tcp_wmem` up to 64 MB, `rmem_max` and `wmem_max` 128 MB,
`tcp_no_metrics_save` 1, `tcp_autocorking` 0), and `ip_forward` 1, which
Mahimahi needs. The Mahimahi shells refuse to run as root.

## Kernel controllers: Cubic, BBR, Vegas

The sender is an iperf3 server outside the shells; the receiver an iperf3
client inside them in reverse mode (`-R`). The sender takes the host's
default congestion control. All 36 runs (four links; CC `cubic`, `bbr` or
`vegas`; run R 1, 2 or 3), from the directory that holds `kit/`:

```bash
run() {  # LINK CC R
    local d q t loss=""
    case $1 in
        a) d=40 q=1000 t=45 ;;
        b) d=20 q=160 t=30 ;;
        c) d=40 q=320 t=30 loss="mm-loss downlink 0.01" ;;
        d) d=80 q=5120 t=30 ;;
    esac
    # $loss is unquoted on purpose: empty, or the shell's three words.
    # shellcheck disable=SC2086
    mm-delay "$d" $loss mm-link "$1.trace" "$1.trace" --downlink-log="logs/$1.$2.$3.log" \
        --uplink-queue=droptail --uplink-queue-args="packets=$q" \
        --downlink-queue=droptail --downlink-queue-args="packets=$q" \
        -- sh -c "iperf3 -c \$MAHIMAHI_BASE -p 5201 -R -C $2 -t $t"
}
cd kit && mkdir -p logs
for cc in cubic bbr vegas; do
    sudo sysctl -w net.ipv4.tcp_congestion_control="$cc"
    pkill -x iperf3; sleep 1; iperf3 -s -p 5201 -D; sleep 1
    for r in 1 2 3; do
        for link in a b c d; do run "$link" "$cc" "$r"; sleep 5; done
    done
done
pkill -x iperf3
```

One run is `run LINK CC R` in `kit/`, after `mkdir -p logs` and the
loop's `sysctl` and `iperf3 -s` lines. `ss -tin` on the host shows the
sender's controller during a run.

## Sage

Sage's harness builds the Mahimahi command itself (`sage_rl/src/sage.cc`,
`start_server`) with the nesting and flags above, so a run needs only our
traces in `~/sage/traces`. After Sage's own setup (`build.sh`,
`set_sysctl.sh`, `cp_models.sh`), for run R:

    cp kit/*.trace ~/sage/traces/
    cd ~/sage/sage_rl
    ./sage.sh 40 44545 a.trace a.trace R 1000 24 48 7 0 crosscheck 45 1 0 1 1 0    # link a
    ./sage.sh 20 44545 b.trace b.trace R 160 48 48 7 0 crosscheck 30 1 0 1 1 0     # link b
    ./sage.sh 40 44545 c.trace c.trace R 320 12 12 7 0 crosscheck 30 1 0 1 1 0     # link c, after the loss edit below
    ./sage.sh 80 44545 d.trace d.trace R 5120 96 96 7 0 crosscheck 30 1 0 1 1 0    # link d

The arguments, in order: one-way delay (ms), port, downlink trace, uplink
trace, run (which only names the log), buffer (packets), the capacity of
the trace's two levels in Mbit/s and their period (7 s), 0 for
evaluation, the log prefix, the run's seconds, actor id, steps (0: by
time), flows, 1 to save the downlink log, 0 to skip plots: `run_sample.sh`'s
arguments with the link's values. Copy the log,
`~/sage/sage_rl/log/down-sage-crosscheck-1-L.trace-L.trace-D-Q-1-BW-BW2-1-R`,
to `logs/L.sage.R.log`. Sage writes the level's capacity into column 1 of
its state and uses it in its reward (column 75); palaestra writes both as
0 ([docs/architecture.md](../../docs/architecture.md#sages-step-and-the-policy-contract-sage-v1)).

Loss on link c: `sage.sh` fixes `loss="0"` and `sage.cc` applies one rate
to both directions, while link c loses 1% of data only. Set `loss="0.01"`
in `sage.sh` for link c, and in `sage.cc`, in both `sprintf(cmd, ...)`
calls under `#ifdef STANDALONE`, replace the second of the two
`mm_loss_rate` after `delay_ms` with `0.0`, then rebuild with
`bash build.sh`.

## expliCCit's released policy

Run expliCCit's released net1 and actor with expliCCit's own scripts, as
its case study ran, on the four links: our trace both ways, the one-way
delay, the droptail queue both ways, the duration, the downlink log saved
with `--downlink-log`, and data loss on link c only. If those scripts
build their Mahimahi command as Sage's do, the same arguments apply. Name
the logs `L.expliccit.R.log`.

## Sending the logs back

Name each downlink log `LINK.CONTROLLER.RUN.log` (LINK `a` to `d`;
CONTROLLER `cubic`, `bbr`, `vegas`, `sage` or `expliccit`; RUN 1 to 3), as
in `a.cubic.1.log`; `compare.py` stops with an error at any other file
ending in `.log`. The uplink log and ccBench's `FILE_init_timestamp` are
not needed. Then record the host and pack:

    cd kit
    { uname -a; lsb_release -ds; iperf3 --version; git -C ~/ccBench/mahimahi rev-parse HEAD
      git -C ~/sage rev-parse HEAD; sha256sum ./*.trace
      sysctl net.ipv4.tcp_rmem net.ipv4.tcp_wmem net.ipv4.tcp_no_metrics_save net.ipv4.tcp_autocorking net.core.rmem_max net.core.wmem_max
    } > logs/host.txt 2>&1
    tar -cf - logs | xz -9 > crosscheck-logs.tar.xz

A 30 s log of link b with Cubic is about 5 MB, 0.13 MB after xz. We store
the file as a release with `scripts/stash.sh`.

## Comparing

    gh release download exp-crosscheck-93 --dir rel && mkdir release
    cat rel/crosscheck.tar.zst.part-* | zstd -dc | tar -x -C release
    tar -xJf crosscheck-logs.tar.xz        # their logs, in logs/
    uv run python tools/crosscheck/compare.py --mahimahi logs --palaestra release --out crosscheck.md

One row per link and controller: the runs on each side that gave metrics,
each side's mean (and range) per metric, the difference (Mahimahi minus
palaestra), and each side's data start. Without `--mahimahi` it gives
palaestra's side alone. Both sides become a Mahimahi log first (the
team's through `parse_mm_log`, palaestra's records through
`mm_log_from_records`), and one function computes from it:

- t0, the data start: the arrival at mm-link of the first packet of at
  least 1000 bytes, so each side starts at its own flow's first segment;
- the window, from t0 + 2 s to t0 + duration - 1 s;
- utilization: bytes departed in the window over the bytes of its
  delivery opportunities;
- queueing delay: per packet departed in the window, its departure minus
  its arrival ms (the time in the link queue, without mm-delay), as a
  mean and a 95th percentile.

A log that ends before its window, cannot be parsed or has no data, and a
failed palaestra episode, give no metrics; the table says why. On link a
the trace's phase at t0 differs between the sides (palaestra starts its
flow at trace time 0, mm-link its trace before the receiver connects); the
data-start columns show the offset.

## Known differences from the original stack

- Kernel: Linux 7.0 in UML in virtual time against 4.19 on hardware in
  real time; Cubic, BBRv1 and Vegas are each kernel's own code.
- Endpoints: palaestra sets `TCP_NODELAY` on both sockets and writes from
  the moment the connection is up; iperf3 sets it only with `-N`, which
  matters little for a bulk sender of full segments.
- Send buffer: palaestra sizes each episode's to the path, at most the
  setups' 64 MB; on 340 stock-controller episodes it changed 3 (Hybla), by
  at most 0.0003 in utilization
  ([docs/validation.md](../../docs/validation.md#tcp-buffer-limits)).
- Sage and expliCCit: palaestra runs ONNX exports of their released models
  through its port of `sage.cc`'s step, and inference costs no virtual
  time; there, their own code runs in TensorFlow or PyTorch in real time.
- Random loss: the same rate, drawn from the seeded kernel's own stream.

## What was checked

On 2026-11-05, in the development container: `links.py write` wrote the
four traces (36, 48, 12 and 96 Mbit/s on average; `b.trace` equals the
first 28000 lines of ccBench's `wired48` at `49b5ada`, `d.trace` equals
`wired96` from ms 1 to 7000). `compare.py` reproduced the known metrics of
a synthetic log in mm-link's format (utilization 0.500, mean delay 7.00
ms, 95th percentile 25.00 ms, data start 500 ms); a log ending before
its window and one with a malformed line gave no metrics, with reasons.
On a Cubic episode of link b, the log written from its records and its
stored `link_events` gave the same metrics bit for bit, equal to a direct
computation and to the episode's own mm-th-delay metrics. The
experiment's commands ran end to end on 4 cores (60 episodes, all valid),
and on all 60 the bytes, packets and delays rebuilt from `link_events`
equal the episodes' `flow_seconds`.

UNRUN, since no environment here runs the original stack: every
command for it above (the shells, iperf3's flags on Ubuntu 16.04, the
sysctl and modprobe lines, Sage's arguments and the edit for link c, read
from the sources of Sage at `eedd281`, ccBench at `49b5ada` and Mahimahi
at `f1346c3`), and the steps for expliCCit's policy, which no code here
checks.
