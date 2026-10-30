# Training net1 and net2

How to go from episodes to a deployable policy, what is known to work, and
what does not yet. The contract and data layout are in
`docs/architecture.md` ("Sage's step"); this page is the workflow.

## 1. Data: teacher campaigns with steps

`examples/campaign-expliccit-teachers.json` is the expliCCit-like dataset:
BBRv1, Cubic and Vegas (the teachers expliCCit names) over 120 paths
(capacity alternating between b and 2b every 7 or 15 s for b = 12, 24,
48 Mbit/s, or constant at 12 to 96 Mbit/s; one-way delay 5 to 40 ms;
50 to 1000 packets), 30 s each. 360 episodes took 100 s on 4 cores and
give 1.07M steps; with `--tables steps` the shard is 216 MB.

Any Linux 7.0 congestion control can teach: Cubic, BBRv1, Reno and BBRv3,
plus BIC, CDG, HighSpeed, H-TCP, Hybla, Illinois, LP, NV, Scalable, Vegas,
Veno, Westwood and YeAH, built as modules and loaded only by the episodes
that use them. A campaign's `sweep` can be `{"kind": "paths", "paths":
[...]}` for any list of paths, including `periodic_levels` traces.

Any campaign records Sage's steps when it sets `"steps": true`:

    {
      "name": "teachers",
      "ccs": [{"kind": "stock", "name": "cubic"}, {"kind": "stock", "name": "bbr"},
              {"kind": "stock", "name": "bbr3"}, {"kind": "stock", "name": "reno"}],
      "sweep": {"kind": "mahak_grid", "bw_mbps": [12, 24, 48], "step": [0.5, 1.0, 2.0],
                "mahak_compat_mrtt_ms": [10, 40], "buffer_pkts": [50, 300]},
      "conditions": [{}, {"loss": {"downlink": 0.01}}],
      "duration_ms": 20000, "seed": 9, "steps": true
    }

Each shard then has `steps.parquet`: one row per step with the 78-value
message (`obs`), the labels at that instant (`label_*`), Sage's reward, the
action taken (the next step's column 76) and whether the policy set it.
144 twenty-second episodes take 30 s on 4 cores and give 283k steps.

## 2. Train

    uv sync --extra train
    uv run expliccit train run train.json out/model

`train.json` is a `TrainSpec` (`src/expliccit_env/train/spec.py`); only
`data` is required:

    {"data": ["out/teachers"], "seed": 1,
     "net1": {"epochs": 12, "patience": 3},
     "net2": {"steps": 6000}}

| Field | Default | Meaning |
|---|---|---|
| `net1.arch` | `expliccit-v1` | expliCCit's encoder (1,265,927 parameters at width 512) |
| `net1.obs_cols` | expliCCit's 70 | message columns net1 reads |
| `net1.scaling` | `robust` | `robust` clips to training quantiles then standardizes; `minmax` is expliCCit's |
| `net1.lcfs` | expliCCit's 7 | `queue_delay_mean_ns, queue_pkts, capacity_bps, free_buffer_pkts, busy_fraction, one_way_delay_ns, receive_rate_bps` |
| `net2.algo` | `td3bc` | TD3+BC; `expliccit-v1` is expliCCit's objective, kept for comparison |
| `net2.inputs` | `net1` | `true_lcfs` trains on the labels instead (an ablation; not exported) |
| `net2.reward_alignment` | `next` | the reward step t+1 reports; `same` is step t's, as expliCCit and Sage's offline data pair them |
| `net2.action_max` | 1.0 | a = action_max * tanh(...) |

Splits are by path: episodes that differ only in congestion control or
seed land in the same split, so no path is both trained and tested on.

Output: `net1.pt`, `net2.pt`, `policy.onnx` (contract `sage-v1`),
`report.json` (net1's error per LCF on the test split in physical units,
next to classical estimators; net2's losses), and the spec.

On GPU nodes, replace the CPU wheel after syncing, then run without
syncing again:

    uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu128
    uv run --no-sync expliccit train run train.json out/model

The CUDA index above is UNRUN here (no GPU in this environment).

## 3. Run a policy

Use the exported graph as an episode's controller:

    {"kind": "agent", "mode": "hold",
     "policy": {"path": "out/model/policy.onnx", "sha256": "<sha256 of the file>"}}

in an episode spec or a campaign's `ccs`, and compare with stock controllers
on held-out paths.

## 4. expliCCit's released models

expliCCit's code is not in this repository and must not be. With a local
copy, its released net1 and actor (`sage_rl/offline/models`) load into our
modules unchanged; the pickled scikit-learn scalers are never loaded:

    uv run expliccit train import-expliccit <expliCCit>/sage_rl/offline/models out/expliccit \
        --as-deployed --data out/teachers

`--as-deployed` repeats what its deployed controller does between and after
the networks (LCF beliefs clamped at 0, the first six rounded, a rounded to
three decimals); `--data` reports its net1's error on our data.

## Reproduction: an expliCCit-shaped controller (one seed)

Data: the expliCCit-like campaign above. net1: expliCCit's encoder, robust
scaling, 4 epochs (early stop), on held-out paths R^2 0.988 (mean queueing
delay), 0.986 (queue), 0.953 (capacity), 0.992 (one-way delay), 0.993
(receive rate), 0.45 (busy fraction), 0.09 (free buffer). net2 variants on
that net1, exported with a rounded to three decimals as expliCCit deploys
it, evaluated over 45 s on 18 paths: expliCCit's two case-study links and
16 held-out paths (rates, delays and buffers outside the training grid).
Means over paths with 95% bootstrap intervals:

| controller | utilization | mean queueing delay ms | power |
|---|---|---|---|
| BBRv1 | 0.970 [0.965, 0.975] | 10.2 | 3.62 |
| BBRv3 | 0.940 [0.921, 0.957] | 16.7 | 3.21 |
| Cubic | 0.981 [0.960, 0.994] | 77.2 | 0.83 |
| Vegas | 0.951 [0.911, 0.981] | 18.6 | 3.57 |
| net2 by behaviour cloning | 0.839 [0.761, 0.909] | 21.3 | 1.85 |
| net2 by expliCCit's objective, discount 0.99 | 0.854 [0.795, 0.908] | 67.7 | 0.57 |
| net2 by TD3+BC, discount 0.99, alpha 0.5 | 0.758 [0.666, 0.844] | 50.6 | 1.56 |
| net2 by TD3+BC, discount 0.95, alpha 2.5 | 0.026 [0.020, 0.031] | 1.0 | 0.53 |

The pipeline reproduces expliCCit's shape: a controller that acts only on
seven inferred LCFs, learned offline from other controllers' logs, runs
at 84 to 85% utilization where its teachers reach 95 to 98%. expliCCit's
own objective gets there with a critic whose loss diverges (to about
-4900: its penalty on random actions has no floor), and with three times
the queueing delay of cloning. TD3+BC at discount 0.95 collapses the
window: the policy's mean action is -0.021 per step against the teachers'
+0.0014. Sage's reward credits lower delay at once but delivered rate only
over a 1 s window, which a 0.95 discount (about 200 ms) barely sees.

## Earlier results (October 2026, one dataset, one seed)

Data: the 144-episode campaign above without the loss condition (Cubic,
BBRv1, BBRv3, Reno; 12 to 48 Mbit/s; one-way delay 10 or 40 ms; 50 or
300 packets).

**net1, held-out paths.** R^2 per LCF:

| LCF | expliCCit's released net1 | ours (5 epochs) | classical estimator |
|---|---|---|---|
| mean queueing delay | 0.847 | 0.994 | 0.999 (srtt minus min RTT) |
| queue occupancy | 0.594 | 0.983 | |
| capacity | 0.361 | 0.972 | 0.709 (max delivered rate over 2 s) |
| free buffer | -59.0 | 0.502 | |
| busy fraction | -0.743 | 0.868 | |
| one-way delay | 1.000 | 0.997 | |
| receive rate | 0.899 | 0.984 | 0.930 (delivered rate, 100 steps) |

expliCCit's numbers are on all 283k steps (it never saw them). Its
one-way delay is exact, which confirms its "mRTT" is the one-way delay. Its
free buffer and busy fraction are far off; its training ranges (buffers up to
10240 packets, busy fraction mostly near 1) differ from this data, and its
ground-truth definitions are not in its code, so part of the gap may be
definitional.

**Closed loop, 20 s, one seed** (utilization / mean queueing delay ms):

| path (Mbit/s, step, one-way ms, pkts) | Cubic | BBRv1 | BBRv3 | expliCCit released, as deployed | ours |
|---|---|---|---|---|---|
| 24, 2.0, 20, 300 | 0.994 / 75.9 | 0.972 / 5.1 | 0.971 / 5.1 | 0.929 / 31.4 | 0.034 / 1.0 |
| 48, 0.5, 10, 50 | 0.994 / 10.9 | 0.987 / 3.5 | 0.979 / 5.4 | 0.858 / 7.3 | 0.057 / 1.0 |
| 30, 1.0, 40, 1000 | 0.977 / 276.7 | 0.960 / 12.0 | 0.959 / 13.7 | 0.581 / 2.6 | 0.019 / 1.0 |
| 12, 0.5, 30, 100 | 0.985 / 95.0 | 0.966 / 33.8 | 0.961 / 34.6 | 0.954 / 107.6 | 0.078 / 2.1 |

expliCCit's released policy runs on Linux 7.0 in UML as a working
controller. Our first trained policy collapses the window. The teachers'
actions are 0 at 85% of steps (mean +0.002), while the policy's mean action
is -0.020 per step, which shrinks the window about fourfold every second.
expliCCit's deployed rounding of a to three decimals acts as a dead band
around 0 that our export does not have. Fixing net2 (BC weight, dead band or
action quantization, training length, reward alignment) is the next step.
