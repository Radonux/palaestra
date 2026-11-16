# Training net1 and net2

How to produce training data, train net1 and net2, and run and score a
policy; terms are in [methodology.md](methodology.md#terms), results in
[results.md](results.md). Training needs `uv sync --extra train`. Large
runs are experiment files in GitHub Actions
([architecture.md](architecture.md#experiments-in-github-actions)); those
under `experiments/` are worked examples of every command below.

## Data: teacher campaigns and step data

A campaign records Sage's steps when it sets `"steps": true`; an episode
that runs a policy always records them. Each shard then has
`steps.parquet`: per 10 ms step, the 78-value message (`obs`), the labels
at that instant (`label_*`), Sage's reward from the exact capacity, the
action taken (the next step's column 76) and whether a policy set it.

    {"name": "teachers", "duration_ms": 20000, "seed": 9, "steps": true,
     "ccs": [{"kind": "stock", "name": "cubic"}, {"kind": "stock", "name": "bbr3"}],
     "sweep": {"kind": "mahak_grid", "bw_mbps": [12, 24, 48], "step": [0.5, 1.0, 2.0],
               "mahak_compat_mrtt_ms": [10, 40], "buffer_pkts": [50, 300]},
     "conditions": [{}, {"loss": {"downlink": 0.01}}]}

    uv run expliccit campaign plan teachers.json out/teachers
    uv run expliccit campaign run out/teachers --build build/uml --workers 4 --tables steps

`--tables steps` drops the large link tables; `--steps-stride K` keeps
every K-th step (enough for net1, not for net2, which needs consecutive
steps). A sweep `{"kind": "paths", "paths": [...]}` takes any paths, and
`"ccs_per_point": k` runs k of the controllers per point. Any of the 17
Linux controllers can teach (`StockName` in `spec.py`).

The datasets in use, each from the experiment file of the same stem, all
recorded after the guest's TCP cross flows were fixed (`312d15a`):

| Data | Release | Used for |
|---|---|---|
| Broad links (`examples/net1_broad.py`), 30 s and 60 s | [`exp-data-net1-broad-30s-218`](https://github.com/Radonux/palaestra/releases/tag/exp-data-net1-broad-30s-218), [`exp-data-net1-broad-60s-219`](https://github.com/Radonux/palaestra/releases/tag/exp-data-net1-broad-60s-219) | training every net1 and neto1 |
| Wider delays, buffers and capacities, 30 s and 120 s | [`exp-data-net1-v6-221`](https://github.com/Radonux/palaestra/releases/tag/exp-data-net1-v6-221) | the same |
| Held-out families | [`exp-data-net1-broad-ood-220`](https://github.com/Radonux/palaestra/releases/tag/exp-data-net1-broad-ood-220) | scoring net1 outside its ranges |
| net2's 600 training links, every step kept | [`exp-data-net2-steps-222`](https://github.com/Radonux/palaestra/releases/tag/exp-data-net2-steps-222) | offline net2 and neto2 |

The reproduction's teachers (BBRv1, Cubic and Vegas on 120 paths, 30 s)
come from `examples/campaign-expliccit-teachers.json`.

### Model names

- **net1 vX, net2 vX**: this project's architecture, where net1 states a
  distribution per LCF.
- **neto1 vX, neto2 vX**: expliCCit's original architecture, where net1
  gives one number per LCF.

A new set of heads is a new version in the same series. Experiment files
and release tags keep the names they were run under.

| Model | LCFs | Release | Chosen run |
|---|---|---|---|
| net1 v7 | expliCCit's 7 | [`exp-net1v6-229`](https://github.com/Radonux/palaestra/releases/tag/exp-net1v6-229) | `net1-v6-s2` |
| net1 v9 | 27 | [`exp-net1v9-243`](https://github.com/Radonux/palaestra/releases/tag/exp-net1v9-243) | `net1-v9-s2` |
| net1 v10 | 32 | [`exp-net1v10-266`](https://github.com/Radonux/palaestra/releases/tag/exp-net1v10-266) | `net1-v10-s2` |
| neto1 v1 | expliCCit's 7 | [`exp-net1-point-228`](https://github.com/Radonux/palaestra/releases/tag/exp-net1-point-228) | `net1-point-ours-s0` |
| neto1 v2 | 27 | [`exp-neto1-v2-239`](https://github.com/Radonux/palaestra/releases/tag/exp-neto1-v2-239) | `net1-point-v2-s0` |
| neto1 v3 | 32 | [`exp-neto1-v3-264`](https://github.com/Radonux/palaestra/releases/tag/exp-neto1-v3-264) | `net1-point-v3-s0` |

The 27 LCFs are expliCCit's seven plus every label or composite whose
head reached a validation R^2 of 0.3
([`exp-neto1-heads-233`](https://github.com/Radonux/palaestra/releases/tag/exp-neto1-heads-233)),
less five composites that net2 could compute from other heads
([`exp-neto1-composites-236`](https://github.com/Radonux/palaestra/releases/tag/exp-neto1-composites-236)).
The 32 add those five back, so that net2 reads them by name instead of
building them: `departure_rate_bps`, `available_capacity_bps`,
`overload_bps` (max(y - C, 0), in place of the signed excess demand),
`bdp_pkts` and `cross_arrival_rate_bps`. On the test rows the heads
shared with the 27-LCF estimators score within 0.04 of those estimators'
R^2 (within 0.02 for net1 v10, whose 90% coverage stays 0.88 to 0.97), and
the five new ones score:

| | departure rate | available capacity | overload | BDP | cross arrival rate |
|---|---|---|---|---|---|
| net1 v10 R^2 (90% coverage) | 0.87 (0.89) | 0.63 (0.91) | 0.89 (0.92) | 0.79 (0.92) | 0.47 (0.89) |
| neto1 v3 R^2 | 0.86 | 0.62 | 0.93 | 0.80 | 0.43 |

A distribution net1 gives a quantity within [0, 1] whose values stay
below 0.5 (a loss rate) its bins up to its 99.9% training quantile, so
net1 v9's top `random_loss_rate` bin ends at 0.02.

## net1: `train run`

    uv run expliccit train run train.json out/model

`train.json` is a `TrainSpec` (`src/expliccit_env/train/spec.py`); only
`data` is required. Every current net1 starts from
`experiments/specs/net1v4.json` (64 bins, 3 members, width 512) with
`--set` overrides, through `experiments/net1_train.py`, which also scores
it on the held-out families; `experiments/net1v9.json` and
`experiments/neto1-v2.json` are the examples. `--eval NET1.pt` scores a
trained net1 without training.

| Field | Default | Meaning |
|---|---|---|
| `data` | | campaign directories, searched for `steps.parquet` |
| `split_by` | `spec` | `link` keeps a link's episodes in one split under every condition |
| `val_fraction`, `test_fraction`, `seed` | 0.1, 0.1, 0 | the split and the initialization |
| `controllers` | all | train only on these controllers' episodes |
| `step_stride` | 1 | keep every n-th step and the step after it |
| `net1_from` | none | reuse a trained `net1.pt`; `report.json` then holds no net1 test scores, which belong to the run that trained it |
| `net1.output` | `{"kind": "point"}` | what net1 states per LCF: one value (`point`), a centre and an interval holding `coverage` (`interval`), or a distribution over `bins` fixed bins, trained with the log score and averaged over `members` net1s (`bins`) |
| `net1.lcfs`, `net1.obs_cols` | expliCCit's 7 and 70 | what net1 predicts and reads; an LCF is any [label or composite](architecture.md#ground-truth-labels), in either window |
| `net1.width`, `dropout`, `lr`, `epochs`, `patience` | 512, 0.3, 1e-3, 50, 5 | the network and its training |

Output: `net1.pt`, `net2.pt`, `policy.onnx`, `report.json` and the spec.
Per LCF on the test split, `report.json` gives R^2 and the mean absolute
error of the belief's mean,
CRPS skill and log-score bits against a forecast that ignores the
measurements, coverage at 50% to 95%, and the 90% width, also on busy and
idle steps. `fit.quantiles` reads a median and intervals off a belief.

## net2 from step data: `train run`

`train run` also trains an offline net2 on net1's beliefs, expliCCit's
memoryless actor asked at every step, on any kind of net1 (single
values, intervals or distributions). Its defaults are untuned (TD3+BC on
Sage's reward collapsed the window).

neto2 and net2 are trained with one recipe so that only the estimator
differs: the arms in `experiments/specs/net2/offline-search2.json`
(TD3+BC or IQL, action `bdp` up to 3, `log_utility`, 10-step returns) on
[`exp-data-net2-steps-222`](https://github.com/Radonux/palaestra/releases/tag/exp-data-net2-steps-222),
seeds 0 to 2. neto2 on neto1 v1 and v2 is
[`exp-neto2-search2-245`](https://github.com/Radonux/palaestra/releases/tag/exp-neto2-search2-245),
scored in closed loop in
[`exp-neto2-search2-val-247`](https://github.com/Radonux/palaestra/releases/tag/exp-neto2-search2-val-247).
net2 on net1 v7 and v9, with its two best arms (TD3+BC, alpha 5 and 10,
200000 steps), is
[`exp-net2-offline-v9c-252`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-offline-v9c-252)
(`experiments/net2-offline-v9c.json`), reading either the means (`view`
`means`, neto2's input) or each distribution pooled into 4 bins
(`input_bins` 4; 16 bins over 27 LCFs did not fit a runner's memory).
`experiments/net2-vs-neto2-val.json` scores both in one closed-loop run,
[`exp-net2-vs-neto2-val-253`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-vs-neto2-val-253)
([results](results.md#learned-net2)).

| `net2` field | Default | Meaning |
|---|---|---|
| `algo` | `td3bc` | `iql` (implicit Q-learning), `bc` (cloning), `expliccit-v1` (expliCCit's objective, kept for comparison) |
| `inputs` | `net1` | `truth`: train on the labels; the exported policy reads the live truth (an oracle). `raw`: the measurements net1 reads, without net1 (the black box); `net1+raw`: both |
| `view`, `input_bins` | `bins`, none | `bins` gives net2 all of net1's output; `means` only each belief's mean (arm A2); `input_bins` pools a distribution's bins |
| `action` | `multiplier` | `bdp`: net2 outputs u, the window as a log2 multiple of the believed BDP (for `raw`, of BBR's estimate) |
| `reward` | `sage` | `utility` or `log_utility` (with `delay_weight` 0.5, `loss_weight` 4) |
| `discount`, `n_step` | 0.95, 1 | multi-step returns let a target's effect over an RTT reach the critic |
| `action_max`, `action_decimals` | 1.0, none | a = action_max tanh(...), rounded in the exported graph |

Every learner update reports diagnostics in `report.json` and
`history.json`:

| Key | Meaning |
|---|---|
| `q`, `q_pull`, `bc_pull` | TD3+BC: mean Q; per row, the critic's and the data's pull on the action; `q_pull` far above `bc_pull` means the actor follows the critic, not the data |
| `ood` | TD3+BC: share of target actions outside the batch's logged range, where the critic's targets are extrapolation |
| `pi_mean`, `pi_std`, `sat` | the actor's output, and its share near the tanh limits |
| `adv_std`, `w_mean`, `w_capped` | IQL: advantage spread, mean weight, share at the cap; weights near 1 mean IQL is cloning |

## The rules

    uv run python experiments/rule_export.py out/rules NET1.pt --u 0.25 [--oracle]
    uv run python experiments/share_rule_export.py out/rules NET1.pt --u0 0.25 --k 0.5 [--oracle]

`--oracle` exports the same rule reading the guest's live truth. The share
rule's exporter also takes `--flows N` (and `--used`) for the oracle rules
told the number of flows; `experiments/mode_rule_export.py` exports the
mode rule.

## Learned net2: `train ppo`

    uv run python experiments/net2_ppo_setup.py out/setup --net1 NET1.pt --models MODELS --u 0.25
    uv run expliccit train ppo ppo.json out/ppo --build build/uml --workers 4

The setup writes what every run shares: the fixed rule (the start and
the baseline), the training paths (600 broad links of their own, 9,600 with
`--paths net2-train-xl`; MODELS
is release `model-net1v3`, whose rule some of their episodes run), the
validation set scored during training, and `ppo-base.json`, which each
run edits. `train ppo` takes a `PPOSpec` (`train/spec.py`). The actor is a
GRU that sees net1's pooled beliefs and its own last u, decides once per
min RTT (clamped to 10 to 50 ms) and holds u between decisions while the
window follows each step's believed BDP. The critic also sees the true
LCFs.

| Field | Default | Meaning |
|---|---|---|
| `algo` | `ppo` | `td3bc` (from a replay of recent iterations, weight `alpha`), `dagger` (imitation of a privileged teacher) |
| `student` | `{"kind": "u"}` | what the actor learns and how its output becomes u (`train/spec.py` U, Gate, Gate3). `u` learns u itself; its `base` `share` makes it a residual on the share rule (gain `share_k`). `gate` learns p, the probability that a stock TCP flow shares the link, and sets u = u_rule + p (share rule's u - u_rule), acting past `threshold`. `gate3` also learns q for copies (`threshold_q`) and cooperates with them, or with `copies` `rule` lets q veto the push. `dagger` teaches any of them, `ppo` continues any of them on the reward |
| `u_rule`, `share_k` | 0.125, 0.5 | the fixed rule's u, and the share rule's gain (the teacher's gain for `dagger`) |
| `init_actor`, `init_critic` | none | start from a saved `actor.pt`, and PPO's value network from a `critic.pt` |
| `init_from` | none | continue an earlier run's directory (in place of the two above): its actor, and for `ppo` its value network where it matches (`critic_truth`, `critic_flows`, `critic_hidden`); refuses a run on another net1, or one whose `view`, `input_bins`, `hidden`, `memory`, student kind, `base`, `inputs` or `action_max` differ |
| `anchor` | 0 | `ppo`: weight of the KL divergence to the policy it started from, added to the loss (`kl_to_start` reports that divergence either way) |
| `teacher_from` | none | `dagger`, student `u`: a run on the true LCFs (`inputs` "truth", student `u`, the same net1) as the teacher, in place of the rules; at each decision it reads the truth and the window the student set, and the student learns its window as u on the student's own believed BDP (for `raw`, the measured one) |
| `alone`, `compete`, `copies`, `copies_share`, `competition` | 0, 0.5, (2, 3, 4), none, `at-start` | share of training links where the flow runs alone (no cross flows, the link's loss kept), share with competition (the rest keep the paths' conditions), the copies drawn there, the share of competition links with copies (none: as likely as each of Cubic and BBRv1), and when stock competitors start (`broad`: within the first 10 s, a second one half the time; `diverse`: the same with any controller but the held-out BBRv3 and Westwood; `dynamic`: as `diverse`, arriving anywhere in the first 60% of the episode, and 30% leaving early) |
| `copies_reward`, `critic_flows` | `own`, false | `team`: copies learn from their mean utility; the critic also counts the flows on the link |
| `iterations`, `links_per_iteration`, `group` | 40, 8, 4 | rollouts per iteration are links times group (K rollouts per link) |
| `noise_sigma`, `noise_hold_steps`, `tau_s`, `lr` | 0.1, 1, 3.0, 3e-4 | exploration (log2 of the window) and how many 10 ms steps each draw is held, the reward's time constant, the actor's learning rate |
| `updates_per_iteration`, `replay_iterations` | 200, 8 | `td3bc` and `dagger` batches per iteration, and the replay's length |
| `time_budget_s` | none | stop before an iteration would cross it |

Output: `policy.onnx`, `actor.pt`, `critic.pt` (PPO's value network, or
the one `dagger` trains for PPO to continue from), and `history.json`
(each iteration's rollouts and diagnostics, and every `eval_every`
iterations the policy against the rule). A taught run continues by PPO
with `init_from`.

## Running and scoring a policy

Any exported graph runs as an episode's controller, observed or as a
cross flow:

    {"kind": "agent", "mode": "hold",
     "policy": {"path": "out/model/policy.onnx", "sha256": "<sha256 of the file>"}}

In `hold` mode the window and pacing stay as the policy set them, as in
Sage; `reno` lets Reno grow the window between actions.

    experiments/rule_eval.sh out/val val 11 11 bbr3 --policy rule=out/rules/rule-u0.25.onnx
    uv run python experiments/copies.py out/copies --split val --duration-ms 30000 --seed 11 --n 2 4 \
        --stock bbr3 --policy rule=out/rules/rule-u0.25.onnx
    uv run python experiments/rule_report.py out/val

`rule_eval.sh` plans the five families (`experiments/heldout.py` writes
any one, `held-competitors` included); run each `c-*` campaign with
`campaign run`, then `rule_report.py` scores them against BBRv3. On any
campaign with step tables, `expliccit score DIRS --reference bbr3` prints
log utility per controller, link and condition, paired by link; it
refuses incomplete shards. `campaign summary` gives Mahak's metrics.

## Oracle and patched policies

A policy can read the path's true LCFs at each step through the
contract's `truth` input, equal to the labels stored with that step. An
oracle reads all of them (`net2.inputs: "truth"`, or `--oracle`). A
patch takes some from the truth and the rest from net1, to test whether a
wrong belief explains a fault; a patched belief puts all its probability
on the true value's bin (net1's own where the truth is NaN):

    from pathlib import Path
    import torch
    from expliccit_env.train import fit, export, models, spec as ts

    run = Path("out/model")
    spec = ts.TrainSpec.parse((run / "train_spec.json").read_text())
    net1, _ = fit.load_net1(run / "net1.pt")
    actor = models.NET2[spec.net2.arch](net1.width, spec.net2.hidden)
    actor.load_state_dict(torch.load(run / "net2.pt", map_location="cpu", weights_only=True))
    graph = fit.policy_graph(net1, actor.eval(), spec.net2.action_max, spec.net2.action_decimals,
                             truth=frozenset({"busy_fraction"}))
    (run / "policy-busy-truth.onnx").write_bytes(export.to_onnx(graph))

## expliCCit's released models

expliCCit's released net1 and actor are in `third_party/expliccit-models`.

    uv run expliccit train import-expliccit third_party/expliccit-models out/expliccit \
        --as-deployed [--truth busy_fraction] [--data out/teachers]

`--as-deployed` repeats its deployed controller: beliefs clamped at 0, the
first six rounded, and the action rounded to three decimals. `--truth`
patches the named LCFs, and `--data` reports its net1's error on step
data. The pickled scalers are never loaded, only their parameters.

## GPU

Training runs on CPU by default. On a GPU node, replace the wheel after
syncing and run without syncing again (UNRUN here: no GPU):

    uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu128
    uv run --no-sync expliccit train run train.json out/model
