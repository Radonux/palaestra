# Tuning breakpoint, 2026-11-09

Tuning of net1 and net2 paused on 2026-11-09 so that the work can turn to
choosing the LCFs. This page is the point to resume from: what each line
of work established, what was running at the pause, what is open, and how
to pick it up. Every number comes from the release named beside it unless
it says "local"; terms are defined in [methodology.md](methodology.md#terms)
and scores are log utility, paired by link, minus the named reference.

This page keeps the state of 2026-11-09. Every release it cites ran on
the guest before the cross-flow fix (`312d15a`), and it predates the
model names in [training.md](training.md#model-names), where net1 v6's
recipe on the fixed-guest data is net1 v7. Later work, including learned
net2 against learned neto2, is in
[results.md](results.md#learned-net2-against-learned-neto2).

## Where things stand

- **net1.** The current estimator is net1 v6
  ([`exp-net1v6-149`](https://github.com/Radonux/palaestra/releases/tag/exp-net1v6-149)),
  plus a traffic head added in round 21. It is accurate and honest within
  its training ranges and dishonest beyond them. Settings barely matter;
  the breadth of training links does.
- **The simplest net2 already works.** The fixed rule (window 1.19 times
  net1's believed BDP) beat every Linux controller and Sage on the test
  links, and the same rule on the true state did only 0.023 better
  ([`exp-final-test-100`](https://github.com/Radonux/palaestra/releases/tag/exp-final-test-100), net1 v5).
- **One learned net2 passed.** r18e05 (DAgger students on net1 v6, three
  seeds as one policy, a copies belief vetoing the push) passed its gate
  and its single test run: +0.068 over the rule with Cubic from the start
  and +0.039 in the competition conditions, both intervals above zero,
  level elsewhere and as copies
  ([`exp-net2-test-r18e05-168`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-test-r18e05-168)).
  Nothing tried after it has beaten it.
- **RL from the reward never passed a gate.** PPO and TD3+BC drift to push
  or to yield, depending on the training mixture.
- **net1's stated uncertainty has not helped net2 anywhere it was tested,**
  including on links built so that information has value (rounds 22 and
  23, run 184).
- **The split's lead over a learned black box is its start** (H6).
- **The last evaluation,** run #186 (rounds 22 and 23 on every family),
  is recorded [below](#running-at-the-pause).

## expliCCit as shipped

expliCCit's released net1 (`best_model.pth`) and actor (`actor.pth`)
are vendored in `third_party/expliccit-models` and run through
`expliccit train import-expliccit --as-deployed`, which repeats its
deployed controller (beliefs clamped at 0, the first six rounded, the
action rounded to three decimals).

**First run: expliCCit's own case-study link**
([`exp-expliccit-case-42`](https://github.com/Radonux/palaestra/releases/tag/exp-expliccit-case-42), 2026-11-04).
The link alternates 24 and 48 Mbit/s, with 40 ms one way and a 1,000-packet
buffer. The released policy used 0.483 of the link (0.340 to 0.534 on the
eight links around it), with 4.4 ms of mean queueing delay. Its net1's
medians along the run, against the truth:

| LCF | released net1 | truth |
|---|---|---|
| mean queueing delay (ms) | 15.2 | 0.71 |
| queue (packets) | 19.4 | 0 |
| capacity, 24 Mbit/s segment | 27.3 | 24.1 |
| capacity, 48 Mbit/s segment | 41.6 | 48.1 |
| free buffer (packets) | 128 | 1,000 |
| busy fraction | 0.999 | 0.505 |
| one-way delay (ms) | 39.9 | 40 |
| receive rate (Mbit/s) | 21.5 | 15.6 |

The same run with one belief at a time replaced by the truth:

| LCFs from the truth | utilization |
|---|---|
| none | 0.483 |
| busy fraction | 0.707 |
| mean queueing delay | 0.573 |
| queue | 0.521 |
| capacity | 0.425 |
| free buffer | 0.062 |
| all seven | 0.621 |

expliCCit attributes the under-utilization to capacity underestimated by
25%. Here the capacity belief is within 14%, and the true capacity makes
it worse. The belief that matters is busy fraction (1.00 against 0.51),
and the true busy fraction gives 0.71 on all nine links. Rounding the
beliefs, as deployed, is not the cause
([`exp-expliccit-case-91`](https://github.com/Radonux/palaestra/releases/tag/exp-expliccit-case-91)).

**On our validation links**
([`exp-sage-v5-val-94`](https://github.com/Radonux/palaestra/releases/tag/exp-sage-v5-val-94)):

| Family and condition | expliCCit utilization | Sage utilization | fixed rule on net1 v5 |
|---|---|---|---|
| training-20, clean | 0.657 | 0.866 | 0.982 |
| training-60, clean | 0.673 | 0.902 | 0.990 |
| training-20, loss 1% | 0.678 | 0.475 | 0.979 |

On clean links it used 24% to 25% less of the link than Sage. Under
random loss it used more, but its queueing delay was 11 to 14 times Sage's.

**On the test links**
([`exp-final-test-100`](https://github.com/Radonux/palaestra/releases/tag/exp-final-test-100)),
it scored the lowest of all controllers: -1.648 against BBRv3 on the
five-family mean (Sage -0.511, the fixed rule +0.133), and 41% less
utilization than Sage on clean links.

**On the cross-check links**
([`exp-crosscheck-93`](https://github.com/Radonux/palaestra/releases/tag/exp-crosscheck-93)),
it used 0.50 of the case-study link (Sage 0.92) and 0.003 of link d, a
collapse.

**Its net1 on our expliCCit-like data** had R^2 1.000 for one-way delay,
0.361 for capacity, -0.743 for busy fraction and -59.0 for free buffer
(local run). expliCCit's ground-truth definitions are not in its code, so
ours may differ.

**None of expliCCit's three checked results reproduced as stated** (R^2
above 0.9 for every LCF, the split's cost against Sage, and the
case-study diagnosis). The causes we can name are in
[results.md](results.md#reproduction-of-expliccit). The run on the
original stack, which would separate our emulator from the rest, is UNRUN.

## net1

| Version | What changed | Release |
|---|---|---|
| v2 | A distribution per LCF (64 bins), trained with the log score | local |
| v3 | 1,230 links under three conditions, varied controllers, ensemble of three | local |
| v4 | The broad dataset: 3,600 links under 64 conditions; dropout and width grid, learning rates, seeds | `exp-net1v4-grid-10`, `-lr-12`, `-seeds-13` |
| v5 | v4's settings retrained on that data regenerated under the frozen guest | [`exp-net1v4-recheck-79`](https://github.com/Radonux/palaestra/releases/tag/exp-net1v4-recheck-79), weights `model-net1v5` |
| v6 | v5 plus wider links (one-way delay 5 to 150 ms, buffers 0.25 to 32 BDP, capacity 1 to 200 Mbit/s) and 120 s episodes; three seeds share bins and scaler | [`exp-net1v6-149`](https://github.com/Radonux/palaestra/releases/tag/exp-net1v6-149), data [`exp-data-net1-v6-146`](https://github.com/Radonux/palaestra/releases/tag/exp-data-net1-v6-146) |
| v6 traffic head | A classifier of the other traffic (none, copies, stock TCP) on net1's scaled measurements | [`exp-net1-traffic-data-178`](https://github.com/Radonux/palaestra/releases/tag/exp-net1-traffic-data-178), head in `model-net2-v6-head` |

The settings are in `experiments/specs/net1v4.json`, used for v5 and v6.
It is an ensemble of three, with 64 bins per LCF and the split by link.

**Accuracy, net1 v5 on held-out test links** (v6 is the same in range,
except a worse free-buffer mean):

| LCF | R^2 | 90% coverage |
|---|---|---|
| mean queueing delay | 0.95 | 0.97 |
| queue | 0.90 | 0.92 |
| capacity | 0.81 | 0.92 |
| free buffer | 0.47 (v6: 0.22 to 0.30) | 0.91 |
| busy fraction | 0.88 | 0.91 |
| one-way delay | 1.00 | 0.97 |
| receive rate | 0.99 | 0.94 |

**What tuning taught:**

- **Split by link.** A random row split, as expliCCit used, overstates
  generalization (free buffer 0.65 by row, -0.22 by link;
  [`exp-net1-row-split-97`](https://github.com/Radonux/palaestra/releases/tag/exp-net1-row-split-97)).
- **Breadth of links beats every setting.** Six settings of dropout and
  width spanned a CRPS skill of only 0.768 to 0.780. Validation loss was
  lowest after 1 to 4 epochs.
- **Wider data moves the edge of honesty, it does not remove it.** v6 is
  honest where v5 failed (one-way delay 61 to 90 ms: coverage 0.97) and
  fails beyond its own ranges (151 to 250 ms: coverage 0.27).
- **Capacity is identifiable only on a busy link.** net1's spread
  widens 3 to 6 times off a busy link and stays honest there (local
  analysis of run 177).
- **Traffic kind.** The traffic head reaches 0.909 balanced accuracy on
  validation links. It calls a stock flow copies 15% of the time (BBRv1
  6%, Cubic 11%, the held-out BBRv3 and Westwood 19% and 25%;
  [`exp-net1-traffic-head-val-180`](https://github.com/Radonux/palaestra/releases/tag/exp-net1-traffic-head-val-180)).

## What the LCFs carry and miss

This section is the input to choosing the LCFs.

**What the decision uses.** Patching one LCF at a time with the truth
moved the learned net2 by 0.026 or less for six of the seven LCFs.
Capacity is the exception. The true capacity gained against competitors
(+0.098 against +0.029) and collapsed as copies (-1.247)
([`exp-net2-patch-v6-172`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-patch-v6-172)).
The window target reads only capacity and one-way delay.

**What net1's capacity belief really measures under sharing.** It sits
between the fair share and the link: 1.03 to 1.04 of the link alone,
0.72 to 0.82 with 2 copies, 0.46 to 0.61 with 4
([`exp-net1-belief-share-173`](https://github.com/Radonux/palaestra/releases/tag/exp-net1-belief-share-173)).
Copies stay safe only because net1 underestimates a shared link. The
capacity this flow can take is not one of the LCFs, though net2 relies on
it.

**What is missing: the other traffic's kind.** A per-step classifier of
none, copies or stock TCP reaches 0.753 balanced accuracy from the raw
measurements, but only 0.496 from net1 v6's beliefs
([`exp-net2-competitor-info-177`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-competitor-info-177)).
This is H1's failure mode, measured. A traffic head in net1 recovers it
(0.909), and the copies belief moved from net2 into net1 at no cost.
Cooperating on that belief still trades the gain against Cubic for the
gain as copies, at every threshold
([`exp-net2-eval-v6-head2-182`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-eval-v6-head2-182)).

**The number of flows N.** Told N, a rule on the true state handles both
copies and Cubic: it led the fixed rule by 0.29 as copies and by 0.09
with Cubic from the start
([`exp-fairshare-used-val-23`](https://github.com/Radonux/palaestra/releases/tag/exp-fairshare-used-val-23)).
This is an oracle bound. N describes the other endpoints, changes as flows
come and go, and one sender's measurements fit several values of it.

**Rules on the true LCFs fail among copies.** Each copy aims at the whole
link's BDP, which gives queues of 41 and 66 ms with 2 and 4 copies
([`exp-share-oracle-val-20`](https://github.com/Radonux/palaestra/releases/tag/exp-share-oracle-val-20)).

**Hard to estimate, little used.** Free buffer has R^2 0.47 (v6: 0.22 to
0.30). It cannot be seen until the queue fills. Replacing it with the
truth collapsed expliCCit's released policy (0.062 of the link).

**Constant-per-link quantities reward recognizing the link.** A minimal
set (capacity, queue, buffer size, one-way delay, cross-traffic share)
lost to the seven LCFs (mean CRPS skill 0.74 against 0.81). Buffer size
and cross-traffic share are constant over a link, so predicting them
rewards recognizing it rather than estimating.

**Inference, not tested:** for the choice of LCFs, the evidence above
points toward three things:

- a quantity for the capacity available to this flow (its share), stated
  as such rather than left hidden inside a "capacity" belief;
- a belief about the kind of the other traffic;
- caution about LCFs that are constant per link or seen only when the
  queue fills.

## net2

**Rules.** On the test links, minus BBRv3 on the five-family mean
([`exp-final-test-100`](https://github.com/Radonux/palaestra/releases/tag/exp-final-test-100), net1 v5):

| Rule | five-family | cubic-first | copies |
|---|---|---|---|
| fixed rule, u = 0.25 | +0.133 | -0.041 | +0.047 |
| share rule, k = 0.5 | +0.153 | +0.067 | -0.271 |
| fixed rule on the true BDP | +0.156 | -0.014 | not run |

On net1 v6 the fixed rule keeps u = 0.25
([`exp-rule-net1v6-val-153`](https://github.com/Radonux/palaestra/releases/tag/exp-rule-net1v6-val-153)).

**Learned net2, the protocol.** Candidates are chosen on validation at
campaign seed 21, gated on new episodes at seed 11
(`experiments/net2_ppo_gate_report.py`), then tested once on the test
links. Every round is in [rounds.md](rounds.md).

**The best candidate: r18e05.** These are DAgger students that learn
from net1's output two beliefs: p, that a stock TCP flow shares the link,
and q, that copies of the flow do. A privileged teacher labels each
decision: the share rule with stock TCP, the fixed rule otherwise. The
policy runs on three seeds' mean beliefs, with q vetoing the push:
`u = u_rule + p'(1 - q')(share - u_rule)`, teacher gain 0.5. Results:

- passed on validation and the gate
  ([`exp-net2-eval-v6-veto-166`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-eval-v6-veto-166),
  [`exp-net2-gate-r18e05-s11-167`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-gate-r18e05-s11-167));
- on test: cubic-first +0.068, competition +0.039, five-family +0.018,
  copies -0.005
  ([`exp-net2-test-r18e05-168`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-test-r18e05-168));
- run unchanged on net1 v6's other two seeds, it kept 70% to 100% of its
  gains ([`exp-net2-swap-v6-169`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-swap-v6-169)).

The policy is `ppo-r18e05-s0.onnx` in release `model-net2-v6-veto`.

**What did not beat it:**

| Line of work | Outcome | Evidence |
|---|---|---|
| PPO and TD3+BC from the reward (rounds 1 to 3, 10 to 17) | stayed at the rule with short noise; with noise held 200 ms learned to push, gaining against Cubic and losing alone and among copies; longer horizons pushed harder; 120 s episodes no better at equal compute | rounds.md |
| Cooperating with copies on q (students' or the traffic head's) | +0.20 to +0.42 as copies, but gives up the Cubic gain | `exp-net2-eval-v6-coop-176`, `-head-179`, `-head2-182` |
| q from net1's traffic head under the veto | equal to r18e05 (five-family +0.018, Cubic +0.051, copies -0.002), not gated | `exp-net2-eval-v6-head2-182` |
| A black box on the raw measurements, same PPO and budget (H6) | 0.2 to 0.4 below the rule from its own start; about 0.05 below the split from a clone of the split's rule | `exp-net2-r19b-bbr-165`, `-r20b-bb-ppo-171`, `-r20a-clone-170` |
| A learned net2 on the true LCFs (A1) | lost its start's lead and ended level with the one on net1 | `exp-net2-r19-blackbox-164` |
| PPO where probing can pay (round 23) | no cell beat the rule; see below | `exp-net2-r23-probe-185` |

Learned net2 is costly at inference: 0.70 ms per decision on one core,
nearly all of it net1's ensemble
([`exp-overhead-206`](https://github.com/Radonux/palaestra/releases/tag/exp-overhead-206)),
against the black box's 0.07 ms (`tools/policy-bench`, local).

## Stated uncertainty and probing

This section covers the two hypotheses tested last, at the user's request
on 2026-11-09.

- **Point 1.** RL has not learned to probe because its exploration is too
  short-lived, and a probe's queue is charged at once but rarely pays
  back on the training links.
- **Point 2.** Test whether net1's full beliefs help net2 more than their
  means (A3 minus A2), on links where information has value.

**The training links leave little to learn.** The fixed rule alone keeps
96% of validation steps on a fully busy link, where net1's capacity
belief is narrow (local analysis of run 177's steps). Capacity steps are
common: 116 of 180 base links alternate 2x every 3, 7 or 15 s. But the
rule's 19% margin over a capacity net1 keeps re-estimating already finds
each rise.

**Rules, where information has value**
([`exp-net2-info-val-184`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-info-val-184)).
Two new families (`experiments/heldout.py`):

- **rise:** the flow alone, with capacity alternating between half and
  twice its level every 2 s;
- **info:** a Cubic or BBRv1 flow leaving at 10 s, on/off UDP at half
  the capacity, and on/off Cubic.

No probing rule beat the fixed rule there. Probing on the believed idle
share (means only) scored -0.008 on rise. Probing on net1's capacity
spread scored -0.051 to -0.117 on rise and -0.059 to -0.131 under on/off
UDP. Because net1 is as unsure of the capacity on any shared link
(median spread 0.34 to 0.42, against 0.15 alone), probing on the spread
mostly pushes there: +0.09 to +0.14 with Cubic first, and -0.38 as copies
unless gated by P(copies). BBRv3 scored -0.627 on rise.

**Round 22: DAgger on means (A2)**
([`exp-net2-r22-means-181`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-r22-means-181)).
Round 18's recipe with the actor reading net1's seven means in place of
its pooled bins. On the trainer's validation check, its seeds ended at
-0.006, +0.006 and +0.017, against round 18's -0.019, -0.020 and +0.016.

**Round 23: PPO on links where probing can pay**
([`exp-net2-r23-probe-185`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-r23-probe-185)).
The training links: half with rising capacity, half with a competitor
that leaves or an on/off sender, no copies. The arms: full beliefs or
means, exploration held 200 ms or 1 s, two seeds each. On the trainer's
check (the new family's conditions, minus the rule), at the last
iteration:

| Arm | seed 1 | seed 2 | mean u (rule 0.25) |
|---|---|---|---|
| full beliefs, held 200 ms | -0.019 | -0.015 | 0.06, 0.10 |
| full beliefs, held 1 s | -0.112 | -0.101 | 0.18 |
| means, held 200 ms | -0.121 | -0.101 | 0.04 |
| means, held 1 s | -0.107 | -0.121 | 0.08, 0.15 |

No cell beat the rule at any evaluation. The means-only cells learned to
yield to Cubic (-0.24 to -0.35), and the longer hold did not help.

**Flow idling: UNRUN.** The case "capacity rising after the flow was
idle" needs three things:

- an app-limited sender in the guest;
- scoring and a reward over on-periods only;
- a net1 trained on idle flows, since net1 has never seen one.

## Running at the pause

Run #186 (`experiments/net2-eval-r23.json`, release `exp-net2-eval-r23-186`)
evaluates, paired on every validation family, as copies and on the rise
and info families:

- round 23's eight policies;
- round 22's ensemble (`ppo-r22-s0`) beside r18e05;
- the fixed rule, the idle-probing rule and BBRv3.

`probing.md` in that release gives each controller's window over the
believed BDP on and off a busy link, and its slopes on net1's believed
idle share and capacity spread. The policies are in release
`model-net2-r23`.

**Result** ([`exp-net2-eval-r23-186`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-eval-r23-186),
validation, campaign seed 21). Each policy minus the fixed rule:

| Policy | five-family | cubic-first | copies | rise | info | on/off UDP |
|---|---|---|---|---|---|---|
| r18e05 (DAgger, full beliefs) | +0.018 | +0.053 | -0.013 | -0.003 | -0.009 | -0.038 |
| round 22 (DAgger, means) | +0.016 | +0.053 | -0.032 | -0.004 | -0.002 | -0.006 |
| round 23, full beliefs, held 200 ms | -0.009 | -0.002 | +0.003 | -0.013, -0.012 | -0.046, -0.004 | -0.056, +0.021 |
| round 23, full beliefs, held 1 s | -0.082 | -0.140 | +0.431 | -0.074, -0.054 | -0.021, -0.023 | +0.087, +0.088 |
| round 23, means, held 200 ms | -0.103 | -0.358 | +0.444 | +0.029, +0.047 | +0.034, +0.053 | +0.230, +0.216 |
| round 23, means, held 1 s | -0.117 | -0.294 | +0.428 | +0.008, +0.016 | +0.038, +0.033 | +0.181, +0.211 |

Round 23's columns for rise, info and on/off UDP give seeds 1 and 2.

- **Full beliefs against means, under imitation:** level (+0.018 against
  +0.016 on the five-family mean). The stated spread adds nothing to the
  learned net2 that passed.
- **Round 23's PPO learned a state-dependent window, not a probe on
  uncertainty.**
  - The means-only policies and the full-beliefs policies held 1 s raise
    their window where net1 believes the link idle: the slope of u on the
    believed idle share is +0.6 to +2.6 on most families
    (`probing.md`).
  - They hold it below the rule's on a busy link: u of -0.38 to -0.10
    with Cubic from the start.
  - That wins on the families they trained for, by up to +0.05 on rise
    and +0.23 under on/off UDP, and as copies (+0.43). It loses against
    Cubic (-0.29 to -0.36).
  - The slope of u on net1's capacity spread is negative for nearly
    every round-23 policy: where net1 is unsure, they back off rather
    than probe.
- **The longer hold did not help the full-beliefs arm.** It gives up
  0.07 on the five-family mean.

No candidate passes the gate (r18e05 is kept, as before). Tuning stops
here.

## Open items

- **Original stack: UNRUN.** The run of the four cross-check links on the
  original stack (`tools/crosscheck`, palaestra's side
  [`exp-crosscheck-93`](https://github.com/Radonux/palaestra/releases/tag/exp-crosscheck-93)).
- **Flow idling: UNRUN.** See above.
- **Smoothing the copies belief: not built.** P(copies) averaged over
  about 1 s would cut a Cubic flow's mean q further, from 0.031 to 0.008
  at a threshold of 0.7 (local analysis). It needs a slot in the policy's
  recurrent state.
- **A second learned net2 on test: none pending.** Any new candidate
  needs its own selection at seed 21, a gate at seed 11 and one test run.
- **Training data for the traffic head.** It was trained on the fixed
  rule's trajectories only, while a learned net2 changes what the flow
  measures. Its held-out competitors score worst (BBRv3, Westwood).
- **ccBench** (task 34): the faithful suites ran
  ([`exp-ccbench-merged-92`](https://github.com/Radonux/palaestra/releases/tag/exp-ccbench-merged-92));
  the extensions did not.

## How to resume

- **Checks before pushing:** run the list in `CLAUDE.md`.
- **Staged training.** Tuning resumes as teach, then RL:
  [staged-net2.md](staged-net2.md) has the design, the recipe
  (`init_from`, `anchor`, `teacher_from`) and the first two experiments.
- **Running experiments.** Every experiment is a file under
  `experiments/`, dispatched with
  `gh workflow run experiment.yml --ref main -f experiment=experiments/NAME.json`.
  Each successful run publishes release `exp-NAME-RUN`.
  - If a run fails with no jobs at all (it happened twice on 2026-11-09),
    rerun the whole run: `gh api -X POST repos/Radonux/palaestra/actions/runs/ID/rerun`.
  - If cells fail without logs, run `gh run rerun ID --failed` after the
    run completes.
- **Policy releases.** `scripts/stash.sh TAG FILE...` publishes a model
  release (`model-...`). The policies to resume from:
  - net1 v6: `model-v6-inputs`, file `net1-v6-s0.pt` (`exp-net1v6-149`
    holds the original, in a file format the code no longer reads);
  - the traffic head: `model-net2-v6-head`, file `traffic_head.pt`;
  - r18e05: `model-net2-v6-veto`;
  - rounds 22 and 23: `model-net2-r23`.
- **Templates.**
  - Evaluation: `experiments/net2-eval-r23.json` (every family, info and
    rise, probing) or `experiments/net2-eval-v6-veto.json`.
  - Gate: `experiments/net2-gate-r18e05-s11.json`.
  - Test: `experiments/net2-test-r18e05.json`.
  - Ensembles and variants: `experiments/gate_ensemble_export.py`
    (`--traffic`, `--traffic-p`, `--truth`, `--net1`).
  - Rules: `experiments/rule_export.py` (`--spread`, `--idle`,
    `--traffic`).
- **Large releases.** Training releases can be huge (round 23's is 10 GB).
  Never unpack one whole: `scripts/release-pull.sh TAG DEST PATTERN...`
  streams it and keeps only the matching files.
- **Reports.**
  - Each controller minus the rule, per campaign and condition:
    `experiments/versus_rule.py`.
  - Probing: `experiments/probing.py`.
  - Classes of the other traffic: `experiments/competitor_info.py` and
    `experiments/traffic_head_val.py`.
