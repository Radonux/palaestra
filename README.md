# palaestra: the expliCCit experiment environment

`expliccit-env` runs the real Linux 7.0 TCP stack over a Mahimahi-equivalent
emulated path and produces two things:

- labeled training data for expliCCit's net1 (sender observations paired with
  exact ground-truth path labels) and closed-loop rollouts for net2;
- Mahak-style performance maps (utilization, delay, power) with kernel-level
  fidelity.

The large sources are vendored, so no build depends on kernel.org: Linux 7.0
(`third_party/linux`, imported unmodified, with our kernel patches as
separate commits) and Mahimahi (`third_party/mahimahi`). Rust crates and
Python packages are fetched from crates.io and PyPI, pinned by
`guest/Cargo.lock` and `uv.lock`.

## Two tiers, one runner

| | Tier 1 | Tier 2 |
|---|---|---|
| Kernel | User-Mode Linux 7.0, `time-travel=inf-cpu` | native Linux 7.0 (Ubuntu 26.04) |
| Time | virtual: CPU work is free, runs are bit-reproducible per seed | real |
| Privileges | none (no root, no KVM): laptops, Slurm, GitHub Actions | root |
| Use | datasets, Mahak maps, CI | calibration, final numbers |
| Status | built and tested here | code written, **not yet run** (needs a 7.0 host) |

Both tiers run the same static guest runner (`guest/`, Rust) and the same
kernel modules (`kernel/modules/`):

- `sch_mmlink`: a qdisc that is one direction of `mm-delay D mm-link TRACE`
  with a droptail buffer, a port of Mahimahi's `LinkQueue` checked event for
  event against an independent Python port;
- `expl_rec`: the record ring every module reports into;
- `expl_tcpobs`: per-ACK sender observations from the `tcp:tcp_probe`
  tracepoint, for any congestion control;
- `tcp_agent`: congestion controls driven by a userspace policy
  (`agent`: Sage-style, `agent_reno`: Orca-style), used to run ONNX policies
  in closed loop.

Stock Cubic, BBRv1 and Reno run unmodified.

## Quick start (Tier 1)

Needs: x86_64 Linux, gcc, make, flex, bison, bc, libelf-dev, musl-tools,
perl, rustup, uv.

    scripts/build-uml.sh build/uml          # UML 7.0 from third_party/linux (~2 min, 4 cores)
    scripts/build-guest.sh build/uml        # modules, static runner, rootfs
    uv sync

    # one episode into a one-shard dataset
    uv run expliccit episode examples/episode-mahak.json out/one --build build/uml

    # a campaign: plan once, run shards anywhere
    uv run expliccit campaign plan examples/campaign-smoke.json out/smoke
    uv run expliccit campaign run out/smoke --build build/uml --workers 4
    uv run expliccit campaign status out/smoke

    # Mahak's oracle_query(X, metric, CC), on Tier 1
    uv run expliccit mahak-query --build build/uml --cc bbr --metric first_power \
        --buffer 8000 --mrtt 4 --bw 70 --step 0.2

On a Slurm cluster (unprivileged x86_64 nodes), write an array script:

    uv run expliccit campaign slurm out/big --build build/uml --shards 200 --cpus-per-task 32

Each array task runs one shard with one UML guest per core and writes its own
Parquet files, so failed or preempted tasks are just rerun.

## The mRTT trap

Mahak passes its "mRTT" straight to `mm-delay`, which delays each direction by
that much, so Mahak's base RTT is twice its mRTT. Specs say which one they
mean, and cannot say both:

    "delay": {"mahak_compat_mrtt_ms": 20}   # Mahak semantics: 20 ms each way, 40 ms RTT
    "delay": {"base_rtt_ms": 40}            # the same path, said explicitly

Mahak's `delay` and `first_power` metrics divide by the one-way value, and so
do ours, so Mahak maps are reproduced like for like.

## What a dataset contains

`campaign run` writes `shards/<k>/` with zstd Parquet tables joined on
`episode_id` (see `src/expliccit_env/dataset.py`):

- `episodes.parquet`: parameters, status or the reason for failure, Mahak
  metrics, provenance (kernel git tree hash, config hash, repo commit, tier);
- `obs.parquet`: sender state at every incoming segment (cwnd, RTTs, delivery
  rate sample, losses, pacing rate and more; see `expliccit_rec.h`);
- `labels.parquet`: ground truth aligned row for row with `obs` (capacity,
  bottleneck queue, exact queueing delay, base RTT, buffer);
- `link_events.parquet`, `link_opp.parquet`: every bottleneck arrival, drop,
  departure and delivery opportunity, so new labels can be derived later
  without rerunning anything;
- `agent_action.parquet`: actions applied by `tcp_agent`.

A failed episode is a row with `status = "failed"` and a reason. No metric is
ever computed from an incomplete or invalid run.

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

## Documentation

- `docs/architecture.md`: components, data flow, ABIs, the link model's exact
  semantics, decisions and their reasons.
- `docs/m0-report.md`: feasibility measurements (UML build, speed, memory,
  determinism, BPF struct_ops, HZ).
- `docs/validation.md`: what has been validated, how, with exact commands, and
  what is still unrun.
- `kernel/modules/README.md`, `third_party/README.md`.

## Repository layout

    kernel/uml.config, kernel/modules/   Tier 1 kernel config, our modules and ABIs
    guest/                               the episode runner (Rust, static musl)
    src/expliccit_env/                   host side: specs, launch, decode, labels, datasets, CLI
    scripts/                             build and check scripts
    tests/                               pytest; UML tests need EXPLICCIT_UML_DIR
    third_party/linux, third_party/mahimahi
    tools/m0/, tools/ci/                 feasibility probes, CI helpers
