# Training net1 and net2

How to go from episodes to a deployable policy, what is known to work, and
what does not yet. The contract and data layout are in
`docs/architecture.md` ("Sage's step"); this page is the workflow.

## 1. Data: teacher campaigns with steps

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

## Results so far (October 2026, one dataset, one seed)

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
