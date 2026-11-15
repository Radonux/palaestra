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

With the learner, its recipe, the data and the 27 LCFs held fixed, net2
on our net1 (a distribution per LCF) beat neto2 on expliCCit's kind of
net1 (one value per LCF) on the held-out test links: +0.146 against
+0.111 and +0.096 in log utility over BBRv3, over three training seeds
([`exp-dagger-twins-test-258`](https://github.com/Radonux/palaestra/releases/tag/exp-dagger-twins-test-258)).
Trained offline instead, the two made no reliable difference.

The findings below come from runs before the cross-flow fix (`312d15a`).

1. net1 estimates most LCFs well on links it never saw (R^2 0.88 to 1.00
   for five of seven), and its 90% intervals hold 91% to 97% of the time.
   It is weak where the measurements say little (free buffer, R^2 0.47;
   capacity on an idle link) and not calibrated outside the ranges it was
   trained on. A random row split, as expliCCit used, overstates
   generalization; breadth of training links mattered more than any
   hyperparameter.
2. The split costs little. The simplest net2, a fixed multiple of net1's
   believed bandwidth-delay product, beats every Linux controller on
   held-out test links, by 0.133 in log utility over BBRv3 and 0.076 over
   BBRv1, and the same rule on the true state does only 0.023 better
   ([`exp-final-test-100`](https://github.com/Radonux/palaestra/releases/tag/exp-final-test-100)).
3. What limits a rule is the other flows: pushing against a buffer-filling
   flow (Cubic) pays, pushing against copies of oneself hurts, and the
   seven LCFs do not say which is present. Told the number of flows, a
   rule on the true state handles both (an oracle bound).
4. A learned net2 beats the rule against competitors and matches it
   elsewhere. Imitation of a teacher that knows the other flows (DAgger),
   with a student inferring stock TCP and copies of itself from net1's
   output alone, passed its gate and on test beat the rule with Cubic
   from the start (+0.068) and in the competition conditions (+0.039),
   both intervals above zero, matching it in every other family and as
   copies
   ([`exp-net2-test-r18e05-168`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-test-r18e05-168)).
   RL from the reward (PPO) gains more against competitors but pushes
   among copies. A black box trained the same way on the raw measurements
   ends 0.2 to 0.4 below the rule from its own start, and about 0.05
   below the split when started from a clone of the split's rule: the
   split's lead is its start.
5. None of expliCCit's three checked results reproduced as stated: R^2
   above 0.9 for every LCF, the split's cost against Sage, and its
   case-study diagnosis.
6. Alarms computed without ground truth catch beliefs the flow's own
   measurements contradict, and a capacity fault was repaired by
   retraining net1 alone.
7. So far net2 gains nothing from net1's stated uncertainty: acting on
   the beliefs' means does as well.

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
