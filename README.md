# palaestra: the expliCCit experiment environment

`expliccit-env` runs the real Linux 7.0 TCP stack over an emulated network
path with Mahimahi's semantics. It produces:

- labeled training data for expliCCit's net1: per-ACK sender observations
  paired with exact ground truth about the path (capacity, queue, delay);
- closed-loop rollouts for net2: learned policies (ONNX) driving the kernel's
  congestion window;
- Mahak-style performance maps (utilization, delay, power) for any congestion
  control, with kernel-level fidelity: Mahak's own active-learning driver,
  ported, with UML episodes as its oracle.

This README is the map for human collaborators: what the parts are, how
data flows, which guarantees the system makes, and how to check them.
Implementation detail lives in `docs/`.

## Architecture

```text
                        HOST (Python, src/expliccit_env)
  spec.json ──► parse ──► plan ──► launch ─────────────┐        ┌──► validate records
  campaign.json  (specs)  (guest   (uml.py: boots      │        │    ──► labels + metrics
                           plan)     UML, seeded)      │        │    ──► Parquet shard
                                                       ▼        │
  ┌───────────────────────── ONE UML KERNEL (Linux 7.0) ────────┴─────────────┐
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
| Launcher | `uml.py` | Boot a UML guest in virtual time (`time-travel=inf-cpu`), seeded, under a wall-clock timeout. |
| Guest runner | `guest/` (Rust) | Inside the kernel: namespaces, qdiscs, bulk TCP pair, cross-traffic flows, Sage's step loop and the ONNX policy, record collection. |
| Policy contract | `policy.py`, `guest/src/step.rs` | `sage-v1`: Sage's 77-value state message in (a port of `sage.cc`), log2 of the window multiplier out, optional recurrent state; the graph picks and scales its own inputs. |
| Step data | `steps.py` | Each step with the labels at its instant, Sage's reward from the exact capacity, and the action taken: net1 and net2 training rows. |
| Trainer | `train/` (optional `train` extra, PyTorch) | net1 supervised and net2 offline RL from step data, architectures chosen by name, one ONNX policy graph out; imports expliCCit's released models from a local copy. |
| Link model | `kernel/modules/sch_mmlink.c` | The bottleneck: Mahimahi's `mm-link` + `mm-delay` (+ `mm-loss`), as a qdisc. |
| Observation | `kernel/modules/expl_tcpobs.c` | Sender state at every ACK, for any congestion control. |
| Agent controllers | `kernel/modules/tcp_agent.c` | `agent` (Sage-style) and `agent_reno` (Orca-style) driven by a policy. |
| Records | `kernel/modules/expl_rec.c`, `expliccit_rec.h` | One ring, one sequence of events; the only kernel-to-host data path. |
| Decode, labels, metrics | `records.py`, `labels.py`, `mahimahi.py`, `episode.py` | Validate records, derive ground truth, compute Mahak's metrics. |
| Mahak driver | `mahak.py` | Mahak's Gaussian-process active learning over its grid, querying UML episodes. |
| Datasets | `dataset.py` | Parquet shards joined on `episode_id`. |
| Kernel and Mahimahi | `third_party/linux`, `third_party/mahimahi` | Vendored sources; our changes are separate commits. |

### Where it runs

Everything runs in User-Mode Linux 7.0 with `time-travel=inf-cpu`. The
kernel is an ordinary unprivileged process; time is virtual (CPU work is
free, so a guest runs 8x to 53x faster than real time); and a seed fixes
every random choice. It runs wherever x86_64 Linux runs, without root or
KVM: laptops, Slurm clusters, GitHub Actions.

A second, native tier (Linux 7.0 on Ubuntu 26.04 hosts, as root, in real
time) was planned and dropped: no environment we have can easily boot a
patched kernel, so it could never run BBRv3 or later kernel changes. The
cost is that nothing checks UML's virtual time against a real-time kernel
(`docs/architecture.md`, D19).

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
   `helper.py` bit for bit, metrics match its Perl `mm-th-delay` exactly, and
   the driver makes the same queries and the same map as Mahak's own
   `mahak_online.py` given the same oracle:
   `pytest tests/test_mahimahi.py tests/test_mahak.py`.
4. **Episodes are reproducible.** Same build, spec and seed give byte-identical
   records, random loss and cross traffic included:
   `pytest tests/test_uml.py -k "bit_identical or loss_and_cross"`.
5. **No silent data.** An episode that times out, overflows a buffer it must
   not, drops records, or fails validation becomes a row with
   `status = "failed"` and a reason, never metrics or labels.
6. **Provenance.** Every episode row names the kernel source tree hash,
   kernel config hash, repository commit and seed.
7. **The mRTT trap is explicit.** Mahak's "mRTT" is a one-way delay (base RTT
   is twice it). A spec says `{"mahak_compat_mrtt_ms": 20}` or
   `{"base_rtt_ms": 40}` (the same path) and cannot say both.

Run all checks at once (the UML tests need a build, see below):

    EXPLICCIT_UML_DIR=build/uml uv run pytest -q

CI runs the same on every push (`.github/workflows/ci.yml`).

## Status

| Area | State |
|---|---|
| UML build, link model, observations, records, datasets, campaigns, CLI | done, tested (72 tests in all; 70 run, the 2 Sage comparisons UNRUN) |
| Agent controllers with ONNX policies, closed loop (`sage-v1`) | done, tested |
| Sage's step and step training data (labels, reward, action per step) | done, tested; the comparison with `sage.cc`'s own code is UNRUN |
| Trainer: net1, net2, export, expliCCit import | runs end to end; untuned first runs give an expliCCit-shaped controller (cloning BBRv1, Cubic and Vegas logs: 84% utilization, 4.8% drops; expliCCit's objective: 85%, but 73% drops; `docs/training.md`) |
| Online training, interval encoder | run end to end; intervals are calibrated and widen where capacity is unobservable; the first, untuned online runs degrade the cloned policy; tuning in progress (`docs/training.md`) |
| Mahak metrics, trace generation and active-learning driver | done, matched to the originals |
| Slurm array scripts | generated and checked locally; not yet submitted to a cluster |
| BBRv3 (Google's, as `bbr3`) | done, tested |
| Random loss and cross traffic (TCP of any CC, constant-rate UDP), with their labels | done, tested |
| Reference runs of real Mahimahi against the link model | not started (needs a root host and a port of Mahak's `highbdp.patch`) |
| Native tier (real-time kernel) | dropped (D19) |

Measured speed: 8x to 53x real time per core, about 5400 twenty-second
episodes per core-hour on Mahak's ICNP'25 space; about 2.5 MB of Parquet per
episode. Details and open items: `docs/validation.md`.

## Running it

It needs x86_64 Linux with gcc, make, flex, bison, bc, libelf-dev,
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

    # Mahak's oracle_query(X, metric, CC), in UML
    uv run expliccit mahak-query --build build/uml --cc bbr --metric first_power \
        --buffer 8000 --mrtt 4 --bw 70 --step 0.2

    # Mahak's whole run (run_mahak.sh's arguments work as written): the map
    # as Mahak_cubic.csv, every query, and a dataset of the queried episodes
    uv run expliccit mahak --build build/uml --out out/map -cc cubic -budget 20 \
        -metric first_power -buffer 2,200,4 -mRTT 5,50,5 -bw 2,20,5 -change 0.1,2,0.1

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
| `docs/architecture.md` | Data flow and trust boundaries, exact link-model semantics, record ABI, time and determinism, decisions D1 to D22, known limits, how to extend |
| `docs/training.md` | From teacher campaigns to a policy: train, run, import expliCCit's models, results so far |
| `docs/methodology.md` | How to test the LCF idea from first principles: claims, identifiability and probing, experiment arms, online training |
| `docs/validation.md` | Every check that ran, with commands and results, and what has not run |
| `docs/m0-report.md` | Feasibility measurements: UML build, speed, memory, determinism, BPF |
| `kernel/modules/README.md` | The kernel modules and their ABIs |
| `third_party/README.md` | Vendored sources and our patches on them |
