# palaestra: the expliCCit experiment environment

`expliccit-env` runs the real Linux 7.0 TCP stack over an emulated network
path with Mahimahi's semantics. It produces:

- labeled training data for expliCCit's net1: per-ACK sender observations
  paired with exact ground truth about the path (capacity, queue, delay);
- closed-loop rollouts for net2: learned policies (ONNX) driving the kernel's
  congestion window;
- Mahak-style performance maps (utilization, delay, power) for any congestion
  control, with kernel-level fidelity.

This README is the map for human collaborators: what the parts are, how
data flows, which guarantees the system makes, and how to check them.
Implementation detail lives in `docs/`.

## Architecture

```text
                        HOST (Python, src/expliccit_env)
  spec.json ──► parse ──► plan ──► launch ─────────────┐        ┌──► validate records
  campaign.json  (specs)  (guest   (Tier 1: uml.py     │        │    ──► labels + metrics
                           plan)    Tier 2: native.py) │        │    ──► Parquet shard
                                                       ▼        │
  ┌─────────────────────────── ONE KERNEL (Linux 7.0) ──────────┴─────────────┐
  │  guest runner (Rust, static, guest/): builds the topology, runs traffic,  │
  │  runs the policy, drains records to files                                 │
  │                                                                           │
  │   [snd netns]  ── veth ──  [path netns]  ── veth ──  [rcv netns]          │
  │   TCP sender               sch_mmlink x2             TCP receiver         │
  │   (cubic | bbr | bbr3 |    (Mahimahi loss, link      cross receivers      │
  │    tcp_agent)               and delay, each way)                          │
  │   cross TCP and UDP                                                       │
  │   senders                                                                 │
  │        │                         │                                        │
  │   expl_tcpobs ─────────────► expl_rec ◄────── tcp_agent actions           │
  │   (per-ACK state)           (record ring, /dev/expl_rec)                  │
  └───────────────────────────────────────────────────────────────────────────┘
```

| Part | Where | Responsibility |
|---|---|---|
| Specs and campaigns | `src/expliccit_env/spec.py`, `campaign.py` | What to run: path (trace, delay, buffer, random loss), congestion control, cross traffic, duration, seed; grids and samples of Mahak's space, each under a list of loss and cross-traffic conditions. |
| Launchers | `uml.py` (Tier 1), `native.py` (Tier 2) | Boot a UML guest in virtual time, or run on a native host as root. |
| Guest runner | `guest/` (Rust) | Inside the kernel: namespaces, qdiscs, bulk TCP pair, cross-traffic flows, ONNX policy loop, record collection. |
| Link model | `kernel/modules/sch_mmlink.c` | The bottleneck: Mahimahi's `mm-link` + `mm-delay` (+ `mm-loss`), as a qdisc. |
| Observation | `kernel/modules/expl_tcpobs.c` | Sender state at every ACK, for any congestion control. |
| Agent controllers | `kernel/modules/tcp_agent.c` | `agent` (Sage-style) and `agent_reno` (Orca-style) driven by a policy. |
| Records | `kernel/modules/expl_rec.c`, `expliccit_rec.h` | One ring, one sequence of events; the only kernel-to-host data path. |
| Decode, labels, metrics | `records.py`, `labels.py`, `mahimahi.py`, `episode.py` | Validate records, derive ground truth, compute Mahak's metrics. |
| Datasets | `dataset.py` | Parquet shards joined on `episode_id`. |
| Kernel and Mahimahi | `third_party/linux`, `third_party/mahimahi` | Vendored sources; our changes are separate commits. |

### Two tiers, one runner

| | Tier 1 | Tier 2 |
|---|---|---|
| Kernel | User-Mode Linux 7.0, `time-travel=inf-cpu` | native Linux 7.0 (Ubuntu 26.04 LTS) |
| Time | virtual: CPU work is free | real |
| Privileges | none: laptops, Slurm (no root, no KVM), GitHub Actions | root |
| Role | datasets, Mahak maps, CI | calibration, final numbers |

Both tiers run the same runner binary and the same modules; only the launcher
differs.

## Guarantees, and how to check them

These are the properties the system is built around. Each has a check you can
run; a change that weakens one should say so in its commit message.

1. **Real kernel TCP, unmodified congestion control.** Cubic, BBRv1 and Reno
   are the kernel's own. Every change to the kernel is a separate commit on
   top of the unmodified import: `git diff e8eadca -- third_party/linux`
   (today: HZ set to 1000, seeded randomness, and Google's BBRv3 with its
   core prerequisites). Adding BBRv3 left seeded Cubic and BBRv1 episodes
   byte-identical (`docs/validation.md`).
2. **The link model is Mahimahi's.** The kernel's arrivals, drops, departures
   and opportunities equal an independent Python port of Mahimahi's
   `LinkQueue`, event for event, also under random loss and cross traffic:
   `pytest tests/test_uml.py -k "mahimahi_oracle or loss_and_cross"`.
3. **Mahak's numbers are Mahak's.** Trace generation matches Mahak's
   `helper.py` bit for bit, and metrics match its Perl `mm-th-delay` exactly:
   `pytest tests/test_mahimahi.py`.
4. **Tier 1 is reproducible.** Same build, spec and seed give byte-identical
   records, random loss and cross traffic included:
   `pytest tests/test_uml.py -k "bit_identical or loss_and_cross"`.
5. **No silent data.** An episode that times out, overflows a buffer it must
   not, drops records, or fails validation becomes a row with
   `status = "failed"` and a reason, never metrics or labels.
6. **Provenance.** Every episode row names the kernel source tree hash,
   kernel config hash, repository commit, tier and seed.
7. **The mRTT trap is explicit.** Mahak's "mRTT" is a one-way delay (base RTT
   is twice it). A spec says `{"mahak_compat_mrtt_ms": 20}` or
   `{"base_rtt_ms": 40}` (the same path) and cannot say both.

Run all checks at once (Tier 1 tests need a build, see below):

    EXPLICCIT_UML_DIR=build/uml uv run pytest -q

CI runs the same on every push (`.github/workflows/ci.yml`).

## Status

| Area | State |
|---|---|
| Tier 1: build, link model, observations, records, datasets, campaigns, CLI | done, tested (47 tests) |
| Agent controllers with ONNX policies, closed loop | done, tested |
| Mahak metrics and trace generation | done, matched to the originals |
| Slurm array scripts | generated and checked locally; not yet submitted to a cluster |
| Tier 2 native mode | written, **not run** (needs an Ubuntu 26.04 / Linux 7.0 host) |
| Cross-tier fidelity study, real-Mahimahi reference runs | not started (need Tier 2) |
| BBRv3 (Google's, as `bbr3`) | done in Tier 1, tested; not available on Tier 2's stock kernel |
| Random loss and cross traffic (TCP of any CC, constant-rate UDP), with their labels | done in Tier 1, tested |

Measured speed: 8x to 53x real time per core in Tier 1, about 5400 twenty-second
episodes per core-hour on Mahak's ICNP'25 space; about 2.5 MB of Parquet per
episode. Details and open items: `docs/validation.md`.

## Running it

Tier 1 needs x86_64 Linux with gcc, make, flex, bison, bc, libelf-dev,
musl-tools, perl, rustup and uv; no root.

    scripts/build-uml.sh build/uml          # UML 7.0 from third_party/linux
    scripts/build-guest.sh build/uml        # modules, runner, guest root
    uv sync

    # one episode
    uv run expliccit episode examples/episode-mahak.json out/one --build build/uml

    # a campaign: plan once, run shards anywhere
    uv run expliccit campaign plan examples/campaign-smoke.json out/smoke
    uv run expliccit campaign run out/smoke --build build/uml --workers 4

    # a Slurm array job (one shard per task, one guest per core)
    uv run expliccit campaign slurm out/smoke --build build/uml --shards 200 --cpus-per-task 32

    # Mahak's oracle_query(X, metric, CC), on Tier 1
    uv run expliccit mahak-query --build build/uml --cc bbr --metric first_power \
        --buffer 8000 --mrtt 4 --bw 70 --step 0.2

Tier 2: `sudo scripts/tier2-setup.sh build/tier2` on the host, then add
`--tier native --build build/tier2`.

Each dataset shard holds `episodes.parquet` (one row per episode: parameters,
status, metrics, provenance), `obs.parquet` (sender observations),
`labels.parquet` (ground truth, row for row with `obs`), `link_events.parquet`
and `link_opp.parquet` (the full bottleneck log, so new labels can be derived
later without rerunning), and `agent_action.parquet`.

## Working conventions

- **Commits** go directly to `main`, in Conventional Commits form, each ending
  with the single trailer
  `Co-Authored-By: Joe Fang <25814618+MinecraftFuns@users.noreply.github.com>`.
  No session ids in commit messages. Force-push is disabled; history is never
  rewritten.
- **Kernel changes** are their own commits under `third_party/linux`;
  `scripts/check-kernel.sh` style-checks them and our modules.
- **Agent skills** come from [BTreeMap/SKILLs](https://github.com/BTreeMap/SKILLs),
  a submodule at `.github/skills` aliased as `.claude/skills` and
  `.agents/skills`. `.github/workflows/sync-skills.yml` (installed verbatim
  from that repository) fast-forwards it twice a day. Clone with
  `git clone --recurse-submodules`.
- Instructions for coding agents: `AGENTS.md` (also `CLAUDE.md`).

## Where the details are

| Document | Contents |
|---|---|
| `docs/architecture.md` | Data flow and trust boundaries, exact link-model semantics, record ABI, time and determinism, decisions D1 to D17, known limits, how to extend |
| `docs/validation.md` | Every check that ran, with commands and results, and what has not run |
| `docs/m0-report.md` | Feasibility measurements: UML build, speed, memory, determinism, BPF |
| `kernel/modules/README.md` | The kernel modules and their ABIs |
| `third_party/README.md` | Vendored sources and our patches on them |
