# Methodology

The question: can a learned congestion controller be split into an
estimator of the path's physical state (net1, trained against ground
truth) and a decision maker that sees only net1's output (net2), at small
cost, and does the split make the controller explainable and repairable?
The idea is expliCCit's. This page states the claims, defines the terms,
and fixes how results are measured; the results are in
[results.md](results.md).

The work has two phases: reproduction, a short step that runs expliCCit's
released models and setup faithfully here and names every divergence, and
research, in which everything but the split is a free choice derived from
the problem.

## Claims

| | Claim |
|---|---|
| H1 | Sufficiency: a policy on the true LCFs does about as well as the best policy we can train on raw measurements |
| H2 | Estimability: net1 recovers the LCFs from sender-side measurements, where the measurements allow it |
| H3 | Honesty: net1's stated confidence matches its actual error |
| H4 | No leakage: net1's output carries only its beliefs, and net2 acts on nothing else |
| H5 | Repairability: a fault is traced to one LCF and fixed there, without retraining net2 from scratch |
| H6 | Affordability: small loss against a black box with the same inputs, data and training |

## Terms

| Term | Meaning |
|---|---|
| LCF | Latent congestion factor, expliCCit's term: a physical property of the path at one instant, with an exact value in emulation. Candidates from the literature: [lcf-concepts.md](lcf-concepts.md). |
| the seven LCFs | expliCCit's set: mean queueing delay, queue (packets), capacity, free buffer (packets), busy fraction (bytes sent over bytes the link offered), one-way delay, and the flow's receive rate, each over the trailing 100 ms. Definitions: [architecture.md](architecture.md#ground-truth-labels). |
| net1 | The estimator. It reads Sage's per-step state message (70 of its 78 values) and outputs one belief per LCF. It is trained against ground truth and never with net2. |
| belief | net1's probability distribution for one LCF over 64 fixed bins (equal bins for a fraction; for a rate or delay a bin for zero, then log-spaced bins), averaged over an ensemble of three networks. |
| net1 v3, v4, v5 | v3: trained on 1,230 sampled links under three conditions (`examples/net1_varied.py`). v4: the same design on a broad dataset of 3,600 links under 64 conditions (`examples/net1_broad.py`). v5: v4's settings retrained on that data regenerated under the frozen guest ([`exp-net1v4-recheck-79`](https://github.com/Radonux/palaestra/releases/tag/exp-net1v4-recheck-79); weights in `model-net1v5`). Current results use v5. |
| net2 | The decision maker: it sets the congestion window from net1's output alone, never from raw measurements or the truth. |
| believed BDP | net1's mean capacity times twice its mean one-way delay, in 1500-byte packets (`fit.window_target`). |
| u | net2's action: the next window is 2^u times the believed BDP. The policy graph reads the current window and lands on the target in one step. |
| fixed rule | u held constant. On net1 v5, u = 0.25 (a window of 1.19 believed BDPs), chosen on validation (`exp-rule-net1v5-val-82`). |
| share rule | u = 0.25 + k busy max(0, 1 - receive rate / capacity), on net1's means, k = 0.5 chosen on validation (`exp-share-net1v5-val-85`; `train/share.py`): it pushes when the link is believed busy and this flow's share small. |
| oracle | The same policy fed the true LCFs, computed live in the guest, in place of net1's beliefs ("the rule on the true BDP"). |
| log utility | Per 10 ms step, from ground truth: ln(max(receive rate / capacity, 0.01)) - 0.5 queueing delay / base RTT - 4 drop fraction (`train/score.py`). An episode scores its mean over steps; a controller's score on a link is the mean over the link's episodes. The log term makes competing flows settle at proportionally fair shares (Kelly, 1998). |
| copies | 2 or 4 flows of one controller starting together on one link. |
| families | The sets of conditions controllers are scored under (table below). The five-family mean averages the first five. |
| the gate | The rule a learned net2 must pass on validation before its one test run (below). |
| test once | Each candidate runs on the test links once, after every choice was fixed on validation. |
| teacher, student | The teacher is the policy whose actions are imitated; the student imitates it. A privileged teacher sees what the student cannot, here the other flows on the link. |
| DAgger | Imitation in which the student acts in every rollout and each of its decisions is labelled with the teacher's action (Ross, Gordon and Bagnell, 2011). |

## Why net1 reports beliefs

A sender learns the link's capacity only while it keeps the link busy:
below capacity with an empty queue, the delivery rate equals the sending
rate, so the measurements only bound capacity from below. The minimum RTT
cannot be measured while a queue persists. No operating point reveals
every LCF, so a single value per LCF misleads ("24 Mbit/s" where the data
support "at least 18"). net1 therefore reports a distribution per LCF, and
net2 can learn when reducing uncertainty is worth the cost of probing.

net1 is trained with the log score (cross-entropy against the bin that
holds the true value), whose expectation is lowest only for the true
conditional distribution, so overstating or understating confidence
always costs (Gneiting and Raftery, 2007). It teaches the uncertainty the
measurements leave, not uncertainty about inputs never seen; the ensemble
widens the belief where its members disagree. Accuracy is the R^2 of the
belief's mean; honesty is the coverage of central intervals (90% unless
stated) on held-out links, reported separately for busy steps (busy
fraction at least 0.95) and idle ones (below 0.8).

net2 is trained after net1, on a fixed net1's outputs, so it learns to act
on net1's beliefs, errors included: the sequential regime of concept
bottleneck models (Koh et al., 2020). This is sound only while net1 is
honest: then its belief is a sufficient statistic for the decision, and
what net2 learns about net1's errors is how to act under stated
uncertainty.

## Evaluation protocol

Links. Data and evaluation are split by link (capacity trace, delay,
buffer), never by row or episode, so no test link was seen under any
condition. The evaluation links are the 21 validation and 27 test links of
`examples/campaign-net1-varied.json` (split seed 1, 15% each): capacity
constant or stepping 2x up or down every 3, 7 or 15 s from a base of 8 to
100 Mbit/s, one-way delay 5 to 50 ms, buffers of 0.25 to 4 BDP.

Families (`experiments/heldout.py`, `experiments/copies.py`):

| Family | Conditions | Length |
|---|---|---|
| training-20 | net1 v3's three: clean; 1% random loss; 3 Mbit/s UDP from 5 to 15 s plus a Cubic flow from 8 s | 20 s |
| training-60 | the same | 60 s |
| new-20 | conditions net1 v3 never saw: a Cubic flow from 0 s, a BBRv1 flow from 4 s, Cubic flows from 2 s and 10 s (the three competition conditions); 0.1% and 3% random loss; 1% loss on the ACK path | 20 s |
| deep-long-20 | the training conditions with 2.5 times the one-way delay (at most 80 ms) and buffers of 4 to 16 BDP | 20 s |
| cubic-first-60 | a Cubic flow from 0 s | 60 s |
| held-competitors-60 | a BBRv3 or a Westwood flow from 0 s, competitors held out of net1 v4 and v5's data | 60 s |
| copies-30s | 2 or 4 copies of the controller | 30 s |
| dynamic-120 | competitors arriving and leaving mid-episode: Cubic from 30 to 90 s; BBRv1 from 0 to 60 s; Reno from 20 s; Cubic from 0 s and 60 s; Vegas from 15 s; BBRv1 from 40 s with Cubic from 40 to 100 s. Reported beside the gate, not part of it | 120 s |

Scoring. Log utility, differences paired by link, 95% bootstrap intervals
over links. Most tables give a controller's score minus BBRv3's (or minus
the fixed rule's) on the same links.

Choosing. Every learned model is tuned on validation links before any
comparison: training diagnostics first, then a grid over the settings
that matter, then the winners over several seeds (three training seeds
for a learned net2; three campaign seeds, 11, 21 and 31, for a rule's
constant). A choice rule is written down before the run it decides.

The gate (`experiments/net2_ppo_gate_report.py`). Each candidate is
trained on three seeds and scored against the fixed rule on every
validation family. The candidate with the best five-family mean is kept.
It passes only if it beats the fixed rule in cubic-first-60 and in
new-20's three competition conditions, each with its 95% interval above
zero, is at or above the rule on the five-family mean, and is not worse
as copies beyond its interval (the interval's upper end at or above
zero). Otherwise net2 stays the fixed rule and the share rule.

Testing. The test links run once, after every choice is fixed:
[`exp-final-test-100`](https://github.com/Radonux/palaestra/releases/tag/exp-final-test-100)
(campaign seeds 13 and 31). A learned net2 that passes a later gate runs
on them once, on its own (round 7e:
[`exp-net2-test-r7e-k0.5-134`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-test-r7e-k0.5-134);
round 18 on net1 v6:
[`exp-net2-test-r18e05-168`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-test-r18e05-168)).

## Arms

Same links, seeds and training budget in each arm.

| Arm | net2 sees | Compared with | State |
|---|---|---|---|
| A0 | the full true state of the path | A1: what the LCFs omit | UNRUN; the environment does not provide it |
| A1 | the true LCFs (oracle) | A3: the cost of estimation (H2) | run for the fixed and share rules and for a learned net2 (round 19) |
| A2 | net1's means only (`view: "means"`) | A3: the value of stated uncertainty | run on validation for online TD3+BC (net1 v3) and for rules where the flow cannot see the capacity (net1 v6); and for learned net2 on v6 (rounds 22, 23) |
| A3 | net1's beliefs | | the design under test |
| A4 | raw measurements (a black box) | A3: the cost of the split (H6) | run for a learned net2 (round 19b); a rule on a classical estimate on net1 v3 |

## Interpretability

A belief is a distribution over one physical quantity, so every statement
can be checked against the truth: "capacity is about 24 Mbit/s (90%
interval 22 to 26)", or "at least 18 Mbit/s (95%); the link was not kept
busy". Each statement comes with net1's measured coverage on held-out
links. The plan:

- Explain a decision by evaluating net2 on a changed belief: exact
  counterfactuals, and exact Shapley values over the seven LCFs in 2^7 =
  128 evaluations (`train/attribution.py`). Run on expliCCit's released
  policy ([results.md](results.md)); on palaestra's own net2, UNRUN.
- Test leakage (H4) with probes that try to recover non-LCF quantities
  (the controller that produced the episode, raw columns) from net1's
  output. UNRUN.
- Detect a wrong belief without ground truth (R2): physical invariants
  among the beliefs, entropy, and later measurements of delivery rate and
  min RTT.
- Repair a fault in net1 alone (R3): retrain net1, keep net2. Replacing a
  belief with the truth at run time can confirm a diagnosis but is not
  automatically a repair: the LCFs are coupled, so one patched LCF can
  describe a path that cannot exist, and net2 has adapted to net1's
  beliefs.
- Compare net2 on the full beliefs with net2 on a few percentiles per LCF
  in physical units, so that what an engineer reads is what net2 acts on.
  UNRUN.

## Status

| | Verdict | Evidence |
|---|---|---|
| Reproduction | The link model equals a port of Mahimahi's link queue, event for event; none of expliCCit's three checked results reproduced here; the comparison on the original stack is UNRUN | [results](results.md#reproduction-of-expliccit) |
| H1 | Evidence against, for a shared link: rules on the true LCFs fail among copies; told the number of flows, one rule handles copies and Cubic (an oracle bound). For a learned net2 at equal compute (round 19), the policy on the true LCFs ends 0.2 to 0.4 above the black box on raw measurements, and level with the one on net1. The raw measurements tell copies from a stock flow far better than net1's beliefs (0.753 against 0.496 balanced accuracy, three classes), and net1 learns it with a traffic head (0.909 on validation) | [results](results.md#what-limits-a-rule-is-the-other-flows) |
| H2 | Yes for five LCFs (test R^2 0.88 to 1.00), weaker for capacity (0.81) and free buffer (0.47); estimation costs the fixed rule 0.023 | [results](results.md#net1-estimates-the-path-state-within-its-training-range) |
| H3 | Yes within the training ranges (90% coverage 0.91 to 0.97 on test links); no outside them. Wider training data (net1 v6) made it honest where v5 failed and moved the edge, beyond which it fails again. On real cellular traces, outside every training range, coverage falls to 0.65 to 0.95 | [results](results.md#net1-estimates-the-path-state-within-its-training-range) |
| H4 | Partly: the learned net2 run unchanged on two other seeds of net1 v6 (same bins) kept 70% to 100% of its gains, with wider intervals. But net1's capacity belief under sharing carries the other flows' share: with the true capacity instead, the same net2 collapses as copies (-1.25) | [results](results.md#a-learned-net2-beats-the-rule-against-competitors) |
| Alarms (R2) | Alarms computed without ground truth catch wrong beliefs that the flow's own measurements contradict: 97% to 100% of episodes for faults injected into five LCFs, and 36 of 37 wrong capacity beliefs on real cellular traces, at 8% to 17% false alarms per LCF. They miss what the measurements cannot contradict: an overestimated busy fraction (0 of 58 on cellular) and a low capacity under competition. They name a group of LCFs, one-way delay alone. On expliCCit's own encoder, wrong in 41% of its beliefs in range, they catch 13% to 29% of its errors and not its Case Study 1 fault | [results](results.md#alarms-and-repair) |
| H5 | Two cases, net2 unchanged in both: a capacity fault repaired by retraining net1 alone (alarms without ground truth missed it), and the copies belief's confusion of Cubic with copies, traced to a belief net1 lacked and repaired by a traffic head added to net1 (the cooperative policy's loss with Cubic from the start went from -0.156 to +0.005 to +0.011, its copies gain from +0.211 to +0.20 to +0.32). The second repair left the trade-off between cooperating and pushing in place | [results](results.md#alarms-and-repair) |
| H6 | The simplest net2 on net1 beats every Linux controller and Sage on test links. Against a black box with the same inputs, learner and budget (A4, PPO on the raw measurements with BBR's BDP estimate), the split leads by 0.2 to 0.4 from the black box's own start, and by about 0.05 when the black box starts from a clone of the split's rule: the lead is the start that net1's estimate gives. Over every validation family that black box trails the split by 0.05 on the five-family mean, but cooperates with copies (+0.18 against -0.26); the split costs about 14 times the inference | [results](results.md#the-split-costs-little) |
| Learned net2 | On net1 v6, one candidate (round 18, round 7e's recipe) passed its gate on new validation episodes and on test beat the fixed rule against competitors (Cubic from the start +0.068, +0.023 to +0.120; competition conditions +0.039, +0.005 to +0.075) and matched it in every other family and as copies; five-family mean +0.018. The same recipe passed on net1 v5 (round 7e), whose competition gain did not hold on test. Learned from the reward alone (PPO), net2 gains more against competitors but pushes among copies (-0.22 to -0.44) | [results](results.md#a-learned-net2-beats-the-rule-against-competitors) |
| net2 against neto2 | With the learner, recipe, data and 27 LCFs held fixed (online DAgger, fixed guest), net2 on the distributional net1 v9 beat neto2 on expliCCit's single-value net1 on the test links: +0.146 against +0.111 and +0.096 minus BBRv3, ahead in every family but Cubic first, against held-out competitors and as copies; every net2 training seed above every neto2 seed at neto2's validation-preferred u. Offline (TD3+BC) the estimator made no reliable difference on validation. net2 read the means; the distributions themselves added nothing | [results](results.md#learned-net2-against-learned-neto2) |
| Stated uncertainty | No measurable value to net2 so far (A2 against A3), including for rules where the flow cannot see the capacity: probing on the spread mostly pushes on shared links, and probing on the means gains nothing over the fixed rule; the passing learned net2 does as well on the means (round 22) | [results](results.md#stated-uncertainty-has-not-helped-net2-yet) |
