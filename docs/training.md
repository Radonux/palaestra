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
| `controllers` | all | train only on these controllers' episodes (`["bbr", "vegas"]`) |
| `net2.algo` | `td3bc` | TD3+BC; `iql` is implicit Q-learning (`expectile`, `beta`); `bc` clones; `expliccit-v1` is expliCCit's objective, kept for comparison |
| `net2.inputs` | `net1` | `true_lcfs` trains on the labels instead (an ablation; not exported) |
| `net2.reward_alignment` | `next` | the reward step t+1 reports; `same` is step t's, as expliCCit and Sage's offline data pair them |
| `net2.action_max` | 1.0 | a = action_max * tanh(...); the critic and the noise settings work in units of it |

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

### Tuning

Every learner update returns diagnostics next to its losses (`report.json`
for `train run`, `history.json` for `train online`):

| Key | Meaning | Read it as |
|---|---|---|
| `q`, `q_pull`, `bc_pull` | TD3+BC: mean Q of the policy's action; per row, the critic's and the data's pull on that action | `q_pull` far above `bc_pull`: the actor follows the critic, not the data |
| `ood` | TD3+BC: share of target actions outside the batch's logged actions (0.5% to 99.5%) | high: the critic's targets are extrapolation |
| `pi_mean`, `pi_std`, `sat` | the actor's output on the batch, and the share near the tanh limits | a mean far from the data's, or saturation, before any closed-loop run |
| `adv_std`, `w_mean`, `w_capped` | IQL: spread of advantages, mean weight, share at `max_weight` | `w_capped` high: beta too large for the reward scale |

The teachers' actions are 0 at 80% of steps and between -0.002 and +0.013
at 90%, so TD3's default noise (0.05 exploration, 0.2 target smoothing
clipped at 0.5, all in units of `action_max`) is many times the data's
spread unless `action_max` is small. A bias of 0.001 per step in the
policy's mean action moves the window by 7% a second, so tune on closed-loop
results, not on the regression loss. IQL weights each logged action by
exp(beta x advantage); at the default reward scale (1/25) the advantages
spread over only 0.01 to 0.04, so beta of 10 or less leaves every weight
near 1 and IQL clones. Section "Tuned comparison" has the search, its
ranges and what each knob did.

## 3. Run a policy

Use the exported graph as an episode's controller:

    {"kind": "agent", "mode": "hold",
     "policy": {"path": "out/model/policy.onnx", "sha256": "<sha256 of the file>"}}

in an episode spec or a campaign's `ccs`, and compare with stock controllers
on held-out paths. `expliccit campaign summary OUT` prints, per controller,
the mean over paths with a 95% bootstrap interval of utilization, mean
queueing delay, drop fraction, power and goodput. Read utilization next to
drops: a policy that holds its window above what the path holds keeps the
link full while the queue drops most of what it sends.

## 4. expliCCit's released models

expliCCit's code is not in this repository and must not be. With a local
copy, its released net1 and actor (`sage_rl/offline/models`) load into our
modules unchanged; the pickled scikit-learn scalers are never loaded:

    uv run expliccit train import-expliccit <expliCCit>/sage_rl/offline/models out/expliccit \
        --as-deployed --data out/teachers

`--as-deployed` repeats what its deployed controller does between and after
the networks (LCF beliefs clamped at 0, the first six rounded, a rounded to
three decimals); `--data` reports its net1's error on our data.

## 5. Intervals instead of points

With `"net1": {"intervals": true, "coverage": 0.9}` net1 reports, for each
LCF, a centre and an interval meant to hold the truth 90% of the time
(pinball loss at the 5th, 50th and 95th percentiles; the bounds cannot
cross; a conformal correction fitted on the validation paths sets the
coverage). net2 then sees all three, so it can tell "capacity is about 24
Mbit/s" from "capacity is at least 24 Mbit/s". The report gives each
LCF's held-out coverage and mean width, and the width when the link is
kept busy against when it is not: the identifiability check
(`docs/methodology.md`, section 2).

Measured on the expliCCit-like data (one seed, held-out paths, nominal
coverage 0.90): capacity 0.89, queue 0.91, receive rate 0.89, free buffer
0.89, one-way delay 0.88, busy fraction 0.84, and mean queueing delay
0.76 (the correction fitted on the validation paths does not carry over
to the test paths for that one). The centres keep the point net1's
accuracy (R^2 0.957 capacity, 0.980 queue, 0.981 mean queueing delay).
The widths show the identifiability limit directly:

| LCF | width, link busy (busy fraction >= 0.95) | width, link not busy (< 0.8) |
|---|---|---|
| capacity | 3.6 Mbit/s | 40.3 Mbit/s |
| receive rate | 3.2 Mbit/s | 20.1 Mbit/s |
| busy fraction | 0.017 | 0.50 |
| free buffer | 573 packets | 640 packets |

When the sender does not fill the link, net1 reports that it does not
know the capacity. Free buffer stays unidentified either way (R^2 0.02).

## 6. Online training

    uv run expliccit train online online.json out/online --build build/uml --workers 32

`online.json` is an `OnlineSpec` (`src/expliccit_env/train/online.py`):

    {"paths_from": "examples/campaign-expliccit-teachers.json",
     "eval_from": "eval.json",
     "net1_from": "out/model/net1.pt", "init_from": "out/bc/net2.pt",
     "net2": {"algo": "iql", "beta": 10, "expectile": 0.9, "reward": "utility",
              "discount": 0.99, "action_max": 0.1, "batch_size": 256},
     "iterations": 8, "episodes_per_iteration": 16, "duration_ms": 20000,
     "noise_sigma": 0.005, "updates_per_iteration": 2000, "eval_every": 4}

These are the tuned settings (section "Tuned comparison"); the defaults
(TD3+BC, noise 0.05) are the first, untuned ones, under which the policy
collapses. `init_from` must come from a run with the same `action_max`.

Each iteration exports the policy (net1 frozen), runs
`episodes_per_iteration` episodes on paths drawn from `paths_from` with
Gaussian noise of `noise_sigma` on each action (seeded per episode), adds
their transitions to a replay buffer and runs `updates_per_iteration`
off-policy updates. `history.json` records each iteration's rollouts,
losses and, every `eval_every` iterations, the noise-free policy on the
`eval_from` paths. `"reward": "utility"` is the ground-truth trade-off of
unused capacity against queueing delay (`delay_weight`, per base RTT) and
the flow's dropped fraction (`loss_weight`).

## Reproduction: an expliCCit-shaped controller (one seed, untuned)

Data: the expliCCit-like campaign above. net1: expliCCit's encoder, robust
scaling, 4 epochs (early stop), on held-out paths R^2 0.988 (mean queueing
delay), 0.986 (queue), 0.953 (capacity), 0.992 (one-way delay), 0.993
(receive rate), 0.45 (busy fraction), 0.09 (free buffer). net2 variants on
that net1, exported with a rounded to three decimals as expliCCit deploys
it, evaluated over 45 s on 18 paths: expliCCit's two case-study links and
16 held-out paths (rates, delays and buffers outside the training grid).
Means over paths, 95% bootstrap intervals over paths. Drops are the
fraction of the flow's packets reaching the bottleneck that the full
queue dropped (`drop_fraction`). Utility is the online reward below
(delay weight 0.5, loss weight 4), the mean over steps, unscaled:

| controller | utilization | mean queueing delay ms | drops | power | utility |
|---|---|---|---|---|---|
| BBRv1 | 0.972 [0.966, 0.976] | 10.7 | 0.005 | 3.57 | +0.82 [+0.75, +0.88] |
| BBRv3 | 0.946 [0.934, 0.957] | 17.8 | 0.002 | 3.04 | +0.71 |
| Cubic | 0.981 [0.962, 0.994] | 77.2 | 0.001 | 0.83 | -0.02 |
| Vegas | 0.951 [0.910, 0.981] | 18.6 | 0.001 | 3.57 | +0.74 |
| expliCCit's released policy, as deployed | 0.824 [0.705, 0.910] | 21.7 | 0.127 [0.052, 0.210] | 1.01 | +0.29 |
| net2 by behaviour cloning | 0.842 [0.768, 0.909] | 21.3 | 0.048 [0.017, 0.091] | 1.85 | +0.54 [+0.42, +0.64] |
| net2 by expliCCit's objective, discount 0.99 | 0.854 [0.795, 0.905] | 67.7 | 0.732 [0.668, 0.788] | 0.57 | -2.34 |
| net2 by TD3+BC, discount 0.99, alpha 0.5 | 0.769 [0.686, 0.850] | 50.8 | 0.169 [0.058, 0.303] | 1.56 | -0.31 |
| net2 by TD3+BC, discount 0.95, alpha 2.5 | 0.025 [0.020, 0.031] | 1.0 | 0.000 | 0.53 | +0.02 |
| net2 by TD3+BC on the utility reward, discount 0.99 | 0.835 [0.755, 0.901] | 63.2 | 0.580 [0.434, 0.716] | 0.66 | -1.79 |
| net2 by behaviour cloning on the interval net1 (section 5) | 0.891 [0.813, 0.950] | 57.2 | 0.267 [0.120, 0.423] | 1.29 | -0.53 |

All rows come from one campaign, except the teachers' utility, which
comes from a rerun with step tables (other episode seeds; BBRv1's
utilization there is 0.970). Against the first version of this table,
which used other episode seeds, 16 of the 18 cloning episodes repeat to
four decimals and two differ, so the means moved by up to 0.011.

Utilization alone reproduces expliCCit's shape: a controller that acts
only on seven inferred LCFs, learned offline from other controllers'
logs, runs at 77 to 89% utilization where its teachers reach 95 to 98%.
Loss changes the reading. Only behaviour cloning stays near its teachers
(4.8% of its packets dropped, against at most 0.5% for the teachers).
expliCCit's own objective reaches 85% utilization by holding a window
far above what the path holds: the full queue drops 73% of its packets
and it keeps delivering only because the queue never drains. Its critic
also diverges (loss to about -4900: its penalty on random actions has no
floor). The TD3+BC variants drop 17 to 58%. expliCCit's released policy
drops 12.7% on these paths; its net1 is far off on our data (section 4)
and our pacing is refreshed once per step, so this need not be its loss
on its own testbed. TD3+BC at discount 0.95 collapses the window: the
policy's mean action is -0.021 per step against the teachers' +0.0014.
Sage's reward credits lower delay at once but delivered rate only over a
1 s window, which a 0.95 discount (about 200 ms) barely sees.

## Offline against online: first configurations (one seed each, untuned)

Each run below used one hand-set configuration; none was tuned (see
"Tuning" in section 2, and the TD3 noise scales there). They show that
these configurations fail, not that online or offline training does. The
tuned comparison is in the next section.

Four online runs, each from a cloned policy with its net1 frozen, on the
utility reward (delay weight 0.5, loss weight 4, discount 0.99), TD3+BC
with alpha 2.5, Gaussian noise of 0.05 on each action, 20 s episodes on
the expliCCit-like training paths:

- **plain**: 12 iterations of 16 episodes, 2000 updates per iteration,
  the replay buffer holding only the policy's own episodes;
- **seeded**: 20 iterations of 24 episodes, 300 updates per iteration,
  the replay buffer starting with the teachers' 1.07M steps and the
  critic trained on them for 5000 updates before the first episode.

Each took 4 to 6 minutes on 4 cores. The final policies, on the
evaluation paths above (45 s, 18 paths):

| start | run | utilization | mean queueing delay ms | drops | utility |
|---|---|---|---|---|---|
| cloning, point net1 (+0.54) | plain | 0.280 [0.172, 0.393] | 7.0 | 0.089 | +0.05 [-0.17, +0.21] |
| cloning, point net1 (+0.54) | seeded | 0.259 [0.167, 0.364] | 18.4 | 0.084 | +0.03 |
| cloning, interval net1 (-0.53) | plain | 0.072 [0.052, 0.093] | 1.1 | 0.000 | +0.08 |
| cloning, interval net1 (-0.53) | seeded | 0.099 [0.055, 0.181] | 11.3 | 0.021 | +0.02 |

Evaluated every few iterations on 20 s episodes, the seeded runs fall
steadily: point 0.70, 0.04, 0.11, 0.21 utilization after 5, 10, 15 and
20 iterations; interval 0.51, 0.42, 0.32, 0.12. The plain runs fall
faster (point 0.44, 0.25, 0.25 after 4, 8, 12; interval 0.10, 0.09,
0.08).

**Online training, in these two configurations, makes the cloned policy worse.**
The reward is not why: it ranks BBRv1, Vegas and BBRv3 above cloning on
the evaluation paths, and on the training paths BBRv1 scores +0.83 (Vegas
+0.68, Cubic -2.25), so a policy that fills the link without a standing
queue exists and the reward prefers it. The optimizer instead settles
where the sender barely sends, so delay and loss vanish. From the point
start it ends below where it began (+0.54 to +0.03 or +0.05): it lowers
the objective it maximizes. From the interval start, which overdrives the
link, it gains (-0.53 to +0.02 or +0.08), but only by the same retreat.
The rollouts' own reward, noise included, never passed +0.34 on the
training paths, against BBRv1's +0.83.

Candidate causes, and what the tuning below found for each:

1. Noise as a random walk: 0.05 per 10 ms step on the log window is a
   factor of about 1.4 over a second, so rollouts mostly show what noise
   does. Confirmed: at 0.005 the collapse stops; at 0.02 the policy still
   backs off. Noise held for an RTT is UNRUN (the guest draws it per step).
2. Critic error the actor exploits: after the 1/25 reward scale a step is
   worth about 0.04, and TD3+BC normalizes the critic term by its mean
   size. Smaller alpha and IQL, which never queries the critic off the
   data, were searched; reward scale 1 is UNRUN.
3. Horizon: discount 0.99 per 10 ms step is a 1 s horizon, while a probe
   on an 80 ms path pays off over several RTTs. 0.995 and 0.999 are UNRUN.
4. Estimation: net2 sees only net1's output. Run the same loop with net2
   on the true LCFs (arm A0, `docs/methodology.md`); that needs ground
   truth fed to a running policy, which the environment cannot do yet.

## Tuned comparison

The search below ran on the dataset of the reproduction table. Its
harness (grid files, a parallel trainer, a closed-loop scorer) was fitted
to this dataset and is not in the repository; what carries over to other
data is in the trainer: IQL, the teacher filter (`controllers`) and the
diagnostics of section 2.

**Protocol.** Configurations were chosen on 12 validation paths that are
in neither the training grid nor the test set: capacity alternating 20
and 40 Mbit/s every 5 s, 40 and 80 every 12 s, constant 16 and constant
60, each with one-way delay and buffer of (8 ms, 150 packets), (25 ms,
400) and (35 ms, 80); 30 s closed-loop episodes. The score is the
utility above (delay weight 0.5, loss weight 4). Winners were retrained
over 4 or 5 seeds, then every one was run once on the 18 test paths of
the reproduction table (45 s). Intervals are 95% bootstraps over seeds
and paths. All runs use the point net1: offline, point and interval net1
tied (32 paired runs, mean difference -0.002).

**What was searched** (batch 256, learning rate 3e-4 and network sizes
fixed):

| Round | net2 | Settings |
|---|---|---|
| offline 1: 134 runs, 1 seed | cloning | net1 point or interval; `action_max` 0.05, 0.1, 0.25, 1; teachers all, BBRv1 and Vegas, BBRv1, Vegas; 10k and 30k updates |
| | expliCCit's objective | `action_max` 0.1, 0.25, 1; CQL weight 0.1, 1, 5; discount 0.95, 0.99 |
| | TD3+BC, utility reward | `action_max` 0.1, 0.25, 1; alpha 0.1, 0.5, 2.5; target noise 0.02, 0.2; discount 0.95, 0.99 |
| | IQL, utility reward | `action_max` 0.1, 0.25, 1; beta 3, 10, 30, 100; expectile 0.7, 0.9; discount 0.95, 0.99 |
| offline 2: 80 runs, 4 more seeds | cloning | `action_max` 0.05, 0.1, 0.25, 1; teachers all, BBRv1 and Vegas, Vegas; 30k and 100k updates; a rounded to 3 decimals or not |
| | TD3+BC, IQL | `action_max` 0.1, discount 0.99; alpha 0.5, 2.5 or beta 10, 100; teachers all, BBRv1 and Vegas; 30k and 60k updates |
| online 1: 32 runs | from the median of 5 cloned seeds | TD3+BC alpha 0.5, 2.5 or IQL beta 10, 100; noise 0.005, 0.02; 500 or 2000 updates per iteration; replay of own episodes, or with BBRv1 and Vegas logs and 5000 critic warm-up updates; 12 iterations of 16 episodes |
| online 2: 12 runs | IQL, own episodes, 2000 updates | beta 10, 30; noise 0.0025, 0.005, 0.01; 2 more seeds; 24 iterations |
| online 3: 4 runs | the online winner | from each of the other 4 cloned seeds; 8 iterations |

**What the diagnostics showed.**

- Noise between runs: two cloning runs that differ only in net1 kind
  differ by 0.33 (standard deviation over 32 pairs) in validation
  utility. A single run's rank means little, so every result below is
  over seeds.
- The regression loss on held-out steps does not predict closed-loop
  utility (Spearman 0.02 over 64 cloned checkpoints). The policy's mean
  action on held-out states does, negatively (-0.53 for cloning, -0.67
  for TD3+BC): a small positive bias integrates through the window into
  overdriving.
- `action_max` sets where the policy sits: at 0.05 to 0.1 cloned
  policies under-send (utilization 0.25 to 0.58 over seeds, almost no
  drops), at 1 they overdrive (0.73 to 0.87, 3 to 23% drops), and
  training longer lowers utilization further. TD3+BC at `action_max` 1
  and alpha 2.5 drives the mean action to +0.06 to +0.11 per step; with
  target noise 0.2, 56 to 60% of its target actions fall outside the
  data. Only `action_max` 0.1 keeps TD3+BC sane.
- IQL is cloning for beta 3 and 10 (mean weight 0.88 to 1.01). Beta 30
  and 100 reweight (mean weight up to 1.5 and 4.6) and do worse (round 1
  means +0.24 and +0.07, against +0.40 at beta 3).
- expliCCit's objective: its Q grows to 10^2 to 10^3, the critic's pull
  on the action is 14 to 5900 times the data's, and every setting stays
  far below cloning (mean -0.58 over 36 checkpoints, best +0.31).
- Online: with noise of 0.02 the policy's mean action turns negative
  and utilization falls (a plausible reading: it learns to back off from
  its own noise); TD3+BC can overdrive at once (one run's first
  rollouts scored -0.96); the teachers' logs in the replay buffer do not
  help. IQL at beta 10 and noise 0.005 is the one setting that does not
  fall below its start: with weights near 1 it is a slow
  advantage-weighted regression on its own episodes. Its rollouts' reward
  stays flat over 24 iterations, and more iterations do not help.

**Validation** (mean over seeds): TD3+BC (alpha 0.5, `action_max` 0.1,
BBRv1 and Vegas logs, 60k updates) +0.60 [+0.46, +0.72], 4 seeds;
cloning (`action_max` 0.1, BBRv1 and Vegas logs, 30k updates) +0.55
[+0.40, +0.69], 5 seeds; IQL beta 10 (all teachers) +0.52, 5 seeds; Vegas
+0.79, BBRv1 +0.77. Online IQL from each of the 5 cloned seeds changes
utility by +0.019 [-0.070, +0.103] against its start.

**Test, once** (18 paths, 45 s, episode seed 7; one campaign):

| controller | seeds | utility [95%] | seeds' range | utilization | mean queueing delay ms | drops |
|---|---|---|---|---|---|---|
| BBRv1 | | +0.82 [+0.75, +0.88] | | 0.970 | 10.9 | 0.005 |
| Vegas | | +0.74 [+0.64, +0.82] | | 0.951 | 18.6 | 0.001 |
| BBRv3 | | +0.70 [+0.57, +0.82] | | 0.942 | 17.8 | 0.002 |
| Cubic | | -0.02 [-0.41, +0.35] | | 0.981 | 76.6 | 0.001 |
| expliCCit's released policy, as deployed | | +0.29 [+0.07, +0.48] | | 0.824 | 21.7 | 0.127 |
| TD3+BC | 4 | +0.64 [+0.54, +0.73] | +0.57 to +0.71 | 0.725 | 8.2 | 0.004 |
| online IQL, from each cloned seed | 5 | +0.59 [+0.41, +0.72] | +0.41 to +0.69 | 0.730 | 16.2 | 0.009 |
| cloning | 5 | +0.55 [+0.44, +0.67] | +0.46 to +0.66 | 0.617 | 7.8 | 0.002 |
| IQL beta 10 | 5 | +0.50 [+0.39, +0.61] | +0.38 to +0.61 | 0.595 | 9.6 | 0.001 |

Differences, resampling seeds and paths (paths paired): online minus its
cloned start +0.03 [-0.13, +0.16]; TD3+BC minus cloning +0.09 [-0.02,
+0.20]; TD3+BC minus BBRv1 -0.18 [-0.30, -0.07]; TD3+BC minus Vegas
-0.09 [-0.24, +0.04]; cloning minus Vegas -0.18 [-0.34, -0.03]; cloning
minus expliCCit's released policy +0.27 [+0.10, +0.45].

The first run of this campaign filled the disk and 304 of its 468
episodes failed; it was discarded, the disk freed and the same campaign
run again. In the rerun the teachers repeat the reproduction table (BBRv1 +0.817
against +0.82, Vegas +0.735 against +0.74).

**Reading.**

- The expliCCit shape holds after tuning: every learned net2 acting on
  seven inferred LCFs stays below BBRv1, by 0.18 or more. TD3+BC is
  level with Vegas and BBRv3 within the intervals; cloning is below
  Vegas.
- For cloning, tuning moved the operating point more than the score:
  +0.55 against +0.54 untuned, but 0.2% of its packets dropped instead
  of 4.8%, at lower utilization (0.62 against 0.84). TD3+BC, which
  overdrove untuned, gained most: -0.31 to +0.64.
- No method is shown to beat cloning: TD3+BC's +0.09 has an interval
  that includes 0, IQL is cloning, and online training adds +0.03 with
  an interval from -0.13 to +0.16.
- Within these settings, online training no longer degrades the cloned
  policy, and it does not improve it either. The claim that closed-loop
  training on intervals learns when to probe stays untested. What is
  left to try is design rather than settings, all UNRUN: noise held for
  an RTT (independent noise of 0.005 per 10 ms step moves the window by
  about 1% over 100 ms, so it barely explores at the window's own time
  scale), one decision per RTT, discount 0.995 or 0.999, reward scale 1 for TD3+BC, and net2
  on the true LCFs (arm A0).

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
