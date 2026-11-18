# palaestra

palaestra (Python package `expliccit-env`, command `expliccit`) is an
emulation environment for learned TCP congestion control. It runs the
Linux 7.0 TCP stack in User-Mode Linux (UML) in virtual time, through a
bottleneck that behaves as Mahimahi's does, and records the true state of
the path at every instant. It runs on any x86_64 Linux host without root,
including GitHub Actions, where every result is published as a release.

## The question

Can a learned congestion controller be split into an estimator of the
path's physical state (net1, trained against ground truth) and a decision
maker that sees only net1's output (net2), at small cost, and does the
split make it explainable and repairable? The idea, and the seven
quantities net1 estimates (latent congestion factors, LCFs: queueing
delay, queue, capacity, free buffer, busy fraction, one-way delay, receive
rate), are expliCCit's. [docs/methodology.md](docs/methodology.md) states
the claims and defines every term used below.

## What we found

1. A learned net2 (online DAgger on net1's means, taught by the fixed
   rule) beats BBRv3 on the held-out test links by +0.146 in log utility
   on net1 v10's 32 LCFs (two training seeds; +0.149 over three at an
   earlier iteration)
   ([`exp-dagger32-test-275`](https://github.com/Radonux/palaestra/releases/tag/exp-dagger32-test-275))
   and by +0.136 on net1 v9's 27, over six training seeds
   ([`exp-dagger-twins-test-258`](https://github.com/Radonux/palaestra/releases/tag/exp-dagger-twins-test-258),
   [`exp-dagger-twins-s456-test-272`](https://github.com/Radonux/palaestra/releases/tag/exp-dagger-twins-s456-test-272)).
   It matches, not beats, the rule it learned from (+0.135 and +0.136):
   its window follows net1's believed bandwidth-delay product, and its
   learned step reads little else
   ([`exp-occlusion-twins-val-276`](https://github.com/Radonux/palaestra/releases/tag/exp-occlusion-twins-val-276)).
   Trained on expliCCit's seven hand-picked LCFs instead of the 32
   chosen from data, it scored the same (+0.151 against +0.148,
   [`exp-dagger7-test-280`](https://github.com/Radonux/palaestra/releases/tag/exp-dagger7-test-280)).
   On ccBench's suites it used 0.96 of the link alone at 12 to 15 ms of
   queueing delay, against 0.92 and 42 ms for BBRv3, and about half of
   it beside a Cubic flow that started first
   ([`exp-ccbench-net2-281`](https://github.com/Radonux/palaestra/releases/tag/exp-ccbench-net2-281)).
2. net1 estimates most LCFs well on links it never saw (test R^2 0.88 to
   1.00 for five of expliCCit's seven), and its 90% intervals hold 89% to
   97% of the time; it is weak where the measurements say little (free
   buffer, R^2 0.29) and not calibrated outside its training ranges
   ([`exp-net1v9-243`](https://github.com/Radonux/palaestra/releases/tag/exp-net1v9-243)).
   Its errors cost the learned net2 about 0.03 in log utility; among
   copies of itself they help, since the true capacity is the whole
   link's and net1's belief, read from the flow's own deliveries, is
   lower
   ([`exp-truth-patch-val-282`](https://github.com/Radonux/palaestra/releases/tag/exp-truth-patch-val-282)).
3. The simplest net2, a fixed multiple of net1's believed bandwidth-delay
   product, already beats BBRv3 by 0.136 on the test links; what limits
   it is the other flows, above all copies of itself. net1's beliefs
   keep little of what tells a copy from a stock TCP flow, so we gave
   net1 a head that estimates which one shares the link. With it, a
   learned net2 scored +0.613 among copies on the test links against
   +0.109 without it, at a cost of 0.009 on the five-family mean; beside
   BBRv3 and Westwood, competitors no training data had, it lost 0.11
   ([`exp-traffic-head-test-309`](https://github.com/Radonux/palaestra/releases/tag/exp-traffic-head-test-309)).
4. The split costs nothing against a black box. With the same learner,
   teacher, links and budget, a net2 on net1's means scored +0.152
   minus BBRv3 on the validation five-family mean, one on the raw
   measurements alone +0.065, and one on both +0.156
   ([`exp-blackbox-val-295`](https://github.com/Radonux/palaestra/releases/tag/exp-blackbox-val-295)).
   Trained three times as long, the black box reached +0.093, against
   +0.158 for the split in the same run
   ([`exp-blackbox-more-val-297`](https://github.com/Radonux/palaestra/releases/tag/exp-blackbox-more-val-297)).
5. net2 gains nothing from net1's stated uncertainty: reading the
   distributions does no better than reading their means, and lowering
   the window where net1 is unsure helps among copies but costs beside a
   buffer-filling flow and on the five-family mean.
6. A fault in net1 can be repaired in net1 alone. A learned net2 trained
   on a net1 that never saw another TCP flow, and so believed too little
   capacity beside one, ran unchanged on the net1 trained on all the
   data at +0.165 minus BBRv3 on the validation five-family mean,
   against -0.028 on its own net1 and +0.154 for the same recipe trained
   on the full net1
   ([`exp-notcp-swap-val-293`](https://github.com/Radonux/palaestra/releases/tag/exp-notcp-swap-val-293)).
   Not every error can be repaired this way: relabelling capacity moved
   net1's costly error between conditions a flow cannot tell apart.
7. None of expliCCit's three checked results reproduced as stated: R^2
   above 0.9 for every LCF, the split's cost against Sage, and its
   case-study diagnosis.

Evidence, caveats and every release: [docs/results.md](docs/results.md).

## How it works

```text
  spec.json ─► parse ─► plan ─► boot UML ─┐   ┌─► validate records ─► labels, metrics, Parquet
                                          │   │
  ┌───────────────────────────────────────┬───┴─ UML kernel, Linux 7.0 ────┐
  │ guest runner (Rust): namespaces, flows, Sage's step loop, the policy,  │
  │ the live truth, record drain                                           │
  │                                                                        │
  │  [sender] ──── veth ──── [path] ──── veth ──── [receiver]              │
  │  Linux controller        sch_mmlink, one       receiver,               │
  │  or tcp_agent + policy   each way: Mahimahi's  cross receivers         │
  │  cross TCP and UDP       link, delay, loss                             │
  │                                                                        │
  │  expl_tcpobs (per ACK) ──► expl_rec (one record ring) ◄── tcp_agent    │
  └────────────────────────────────────────────────────────────────────────┘
```

Each episode sends one bulk TCP flow, and optional cross traffic, through
a trace-driven bottleneck with a fixed delay, a droptail buffer and
optional random loss. Every arrival, drop and departure is recorded, so
the LCFs are exact at every instant. A policy is an ONNX graph that sets
the window every 10 ms step from Sage's state message, and can also read
the true LCFs (an oracle). [docs/architecture.md](docs/architecture.md)
has the design and its reasons.

## Quick start

It needs x86_64 Linux with gcc, make, flex, bison, bc, libelf-dev,
musl-tools, perl, rustup and uv.

    scripts/build-uml.sh build/uml && scripts/build-guest.sh build/uml
    uv sync

    # one episode, then a campaign: plan once, run shards anywhere
    uv run expliccit episode examples/episode-mahak.json out/one --build build/uml
    uv run expliccit campaign plan examples/campaign-smoke.json out/smoke
    uv run expliccit campaign run out/smoke --build build/uml --workers 4
    uv run expliccit campaign summary out/smoke

    # an experiment in GitHub Actions; its result becomes release exp-NAME-RUN
    gh workflow run experiment.yml -f experiment=experiments/smoke.json

    # training (net1 and an offline net2; learned net2 in closed loop)
    uv sync --extra train
    uv run expliccit train run train.json out/model
    uv run expliccit train ppo ppo.json out/ppo --build build/uml --workers 4

[docs/training.md](docs/training.md) covers data, training, running and
scoring policies.

## Guarantees

UML checks need `EXPLICCIT_UML_DIR=build/uml`; the Mahak checks need a Mahak
checkout (`git clone https://github.com/Ppazhooh/mahak /tmp/mahak && git -C
/tmp/mahak checkout 5077583`) in `EXPLICCIT_MAHAK_DIR`.

| Property | Check |
|---|---|
| Real kernel TCP: every Linux controller is the kernel's own code, BBRv3 is Google's, and every kernel change is a separate commit | `git log --oneline e8eadca.. -- third_party/linux`; [third_party/README.md](third_party/README.md) |
| The bottleneck equals an independent port of Mahimahi's link queue, event for event, also under loss and cross traffic | `pytest tests/test_uml.py -k "mahimahi_oracle or loss_and_cross"` |
| Mahak's traces, metrics and active-learning driver equal Mahak's own code | `pytest tests/test_mahimahi.py tests/test_mahak.py` |
| The same build, spec and seed give byte-identical records | `pytest tests/test_uml.py -k "bit_identical or loss_and_cross"` |
| A failed or invalid episode becomes a `failed` row with a reason, never metrics or labels | `pytest tests/test_records_abi.py tests/test_agent.py::test_bad_policy_output_fails_the_episode tests/test_mahak.py::test_failed_queries_never_become_labels tests/test_score.py` |
| The truth a policy reads equals the labels stored with its step | `pytest tests/test_truth.py` |
| Every released policy (each `.onnx` asset of the `model-*` releases) runs on the current guest | `experiments/onnx-compat.json` |
| Every kind of net1 (single values, intervals, distributions) trains with every net2 (offline, or taught then PPO; all beliefs or their means), exports a graph that computes what was trained, and runs in the guest | `pytest tests/test_e2e.py` |
| Mahak's "mRTT" is one-way; a spec names one-way delay or base RTT, never both | `pytest tests/test_spec.py` |
| Every episode row names its kernel tree, kernel config, repository commit and seed | columns `kernel_tree`, `config_sha256`, `repo_commit`, `seed` of `episodes.parquet` |

Everything at once: `uv sync --extra train`, then
`EXPLICCIT_UML_DIR=build/uml EXPLICCIT_MAHAK_DIR=/tmp/mahak uv run pytest -q`.
The two comparisons with Sage's `sage.cc` also need a Sage checkout and
g++ (`EXPLICCIT_SAGE_DIR`); CI skips them. What each check showed, and
what was never run: [docs/validation.md](docs/validation.md).

## Working conventions

They are in `AGENTS.md` (also `CLAUDE.md`). Agent skills come from the
[BTreeMap/SKILLs](https://github.com/BTreeMap/SKILLs) submodule at
`.github/skills`, so clone with `--recurse-submodules`.

## Documents

| Document | Read it for |
|---|---|
| [docs/methodology.md](docs/methodology.md) | The claims, the terms, the evaluation protocol, the status of each claim |
| [docs/lcf-concepts.md](docs/lcf-concepts.md) | Candidate LCFs from the literature: what each means, its ground truth in the emulator, and its definition |
| [docs/results.md](docs/results.md) | The findings, their evidence and caveats, with releases |
| [docs/training.md](docs/training.md) | How to produce data, train net1 and net2, and run and score a policy |
| [docs/architecture.md](docs/architecture.md) | How an episode runs and why; known limits; decisions; extending |
| [docs/validation.md](docs/validation.md) | What was checked, how, and what was not |
| [kernel/modules/README.md](kernel/modules/README.md) | The kernel modules and their ABIs |
| [third_party/README.md](third_party/README.md) | Vendored sources and our patches |
| [tools/crosscheck/README.md](tools/crosscheck/README.md) | The kit that compares palaestra with the original stack |
