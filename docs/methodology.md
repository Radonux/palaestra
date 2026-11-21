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
| net1 vX, neto1 vX | The current estimators of each architecture: [training.md](training.md#model-names). |
| net2 | The decision maker: it sets the congestion window from net1's output alone, never from raw measurements or the truth. |
| believed BDP | net1's mean capacity times twice its mean one-way delay, in 1500-byte packets (`fit.window_target`). |
| u | net2's action: the next window is 2^u times the believed BDP. The policy graph reads the current window and lands on the target in one step. |
| fixed rule | u held constant: u = 0.25 on net1 v9 and 0.375 on neto1 v2, chosen on validation ([`exp-rule-paired-val-246`](https://github.com/Radonux/palaestra/releases/tag/exp-rule-paired-val-246)). |
| oracle | The same policy fed the true LCFs, computed live in the guest, in place of net1's beliefs ("the rule on the true BDP"). |
| log utility | Per 10 ms step, from ground truth: ln(max(receive rate / capacity, 0.01)) - 0.5 queueing delay / base RTT - 4 drop fraction (`train/score.py`). An episode scores its mean over steps; a controller's score on a link is the mean over the link's episodes. The log term makes competing flows settle at proportionally fair shares (Kelly, 1998). |
| copies | 2 or 4 flows of one controller starting together on one link. |
| families | The sets of conditions controllers are scored under (table below). The five-family mean averages the first five. |
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
| training-20 | clean; 1% random loss; 3 Mbit/s UDP from 5 to 15 s plus a Cubic flow from 8 s | 20 s |
| training-60 | the same | 60 s |
| new-20 | conditions outside net1's training: a Cubic flow from 0 s, a BBRv1 flow from 4 s, Cubic flows from 2 s and 10 s (the three competition conditions); 0.1% and 3% random loss; 1% loss on the ACK path | 20 s |
| deep-long-20 | the training conditions with 2.5 times the one-way delay (at most 80 ms) and buffers of 4 to 16 BDP | 20 s |
| cubic-first-60 | a Cubic flow from 0 s | 60 s |
| held-competitors-60 | a BBRv3 or a Westwood flow from 0 s, competitors held out of net1's training data | 60 s |
| copies-30s | 2 or 4 copies of the controller | 30 s |

Scoring. Log utility, differences paired by link, 95% bootstrap intervals
over links. Most tables give a controller's score minus BBRv3's (or minus
the fixed rule's) on the same links.

Choosing and testing. A learned arm's settings (for the online DAgger
net2, the iteration) are chosen on the validation links at campaign seed
21 by its five-family mean over three training seeds, checked at
campaign seeds 11 and 31, and then run once on the test links (campaign
seeds 13 and 31, plus the held-out competitors and copies):
[`exp-dagger-twins-test-258`](https://github.com/Radonux/palaestra/releases/tag/exp-dagger-twins-test-258).
Training seeds 4 to 6 then ran at the chosen settings on the same test
campaigns, with nothing re-chosen:
[`exp-dagger-twins-s456-test-272`](https://github.com/Radonux/palaestra/releases/tag/exp-dagger-twins-s456-test-272).

## Arms

Same links, seeds and training budget in each arm.

| Arm | net2 sees | Compared with |
|---|---|---|
| A0 | the full true state of the path | A1: what the LCFs omit |
| A1 | the true LCFs (oracle) | A3: the cost of estimation (H2) |
| A2 | net1's means only (`view: "means"`) | A3: the value of stated uncertainty |
| A3 | net1's beliefs | |
| A4 | raw measurements (a black box) | A3: the cost of the split (H6) |
| A5 | the same number of latents from an encoder of net1's architecture, trained with net2 and free of meaning (`encoder: "free"`) | A2 in expliCCit's design: the cost of fixing what the latents mean |

A5 against A2 is the one exception to equal training. Both use the same
network, expliCCit's, with the same number of layers, weights and latents.
Splitting it into an estimator and a decision maker makes staged training
possible: net1 learns the LCFs from ground truth, then net2 learns on the
fixed net1. A5's latents have no target, so it can only be trained end to
end by net2's loss. Holding A2 to that one option would deny the split the
training it exists to enable, so each arm is trained the way that suits it
and the two are compared at their best: the same net2 learners are tried
on both, and each arm's best is chosen on links the reports never use.

## Interpretability

A belief is a distribution over one physical quantity, so every statement
can be checked against the truth: "capacity is about 24 Mbit/s (90%
interval 22 to 26)", or "at least 18 Mbit/s (95%); the link was not kept
busy". Each statement comes with net1's measured coverage on held-out
links. Decisions are explained by evaluating net2 on a changed belief:
exact counterfactuals, and exact Shapley values over the LCFs
(`train/attribution.py`). Wrong beliefs are flagged without ground truth
by checking them against the flow's later measurements (alarms), and a
fault in net1 is repaired by retraining net1 alone.

## Status

| | Verdict | Evidence |
|---|---|---|
| Learned net2 | Online DAgger on net1's means, taught by the fixed rule, scores +0.146 minus BBRv3 on the test links on net1 v10 (32 LCFs) and +0.136 on net1 v9 (27 LCFs, six training seeds): level with the rule it learned from (+0.135, +0.136). Its window follows net1's believed BDP; holding every LCF at its mean moves its learned step u by 0.02 to 0.05. On expliCCit's seven LCFs alone (net1 v7) it scored the same on the test links, +0.151 against +0.148 on 32 ([`exp-dagger7-test-280`](https://github.com/Radonux/palaestra/releases/tag/exp-dagger7-test-280)): the 25 heads chosen from data added nothing | [results](results.md#learned-net2) |
| Cost of estimation | Fed expliCCit's seven LCFs from the truth, nothing retrained (A1 against A3), the learned net2 gained 0.027 to 0.029 on the validation five-family mean and the fixed rule 0.031 to 0.034, mostly as fewer drops. As copies the truth collapsed (-1.36 against -0.08 minus BBRv3): the link's true capacity is not the capacity left to the flow, which net1's belief tracks better. On shared links the true LCFs are no upper bound. Capacity alone carries both the gain and the collapse | [`exp-truth-patch-val-282`](https://github.com/Radonux/palaestra/releases/tag/exp-truth-patch-val-282), [`exp-truth-patch-lcf-val-283`](https://github.com/Radonux/palaestra/releases/tag/exp-truth-patch-lcf-val-283), [results](results.md#the-split-costs-little) |
| Stated uncertainty | No measurable value to net2: reading net1's distributions did no better than reading their means, online (+0.119 against +0.143) or offline (mixed by estimator). Lowering the window by net1's stated spread of capacity, on the rule or the learned net2, gained among copies and lost with Cubic first and on the five-family mean; gated by net2's own beliefs about the other flow, it lost or did nothing (runs 287 to 289) | [results](results.md#stated-uncertainty-has-not-helped-net2-yet) |
| H2 | net1 v9 on the test links: R^2 0.88 to 1.00 for five of expliCCit's seven LCFs, 0.80 for capacity and 0.29 for free buffer | [`exp-net1v9-243`](https://github.com/Radonux/palaestra/releases/tag/exp-net1v9-243) |
| H3 | Within the training ranges, 90% intervals hold 89% to 97% of the time for every LCF on the test links; outside them coverage falls (to 0.26 for one-way delay with wider RTTs) | [`exp-net1v9-243`](https://github.com/Radonux/palaestra/releases/tag/exp-net1v9-243) |
| H6 | The fixed rule on net1 v9 scores +0.136 over BBRv3 on the test links. With the same DAgger learner, teacher, links and budget, a student on net1 v7's means scored +0.152 minus BBRv3 on the validation five-family mean and a black box on the raw measurements +0.065 (hidden size 256; +0.023 at 64), worse in deep buffers and with Cubic first, better among copies ([`exp-blackbox-val-295`](https://github.com/Radonux/palaestra/releases/tag/exp-blackbox-val-295)). Trained three times as long the black box reached +0.093 and at twice the width +0.061, beside +0.158 for the split in the same run ([`exp-blackbox-more-val-297`](https://github.com/Radonux/palaestra/releases/tag/exp-blackbox-more-val-297)) | [results](results.md#the-split-costs-little) |
| H1 | On a shared link the seven LCFs keep little of what tells copies of oneself from a stock TCP flow: per step on held-out links a classifier of the other traffic reached a balanced accuracy of 0.472 from net1 v7's beliefs, 0.609 from net1 v10's and 0.709 from the raw measurements, chance 0.333 ([`exp-competitor-info-v7-299`](https://github.com/Radonux/palaestra/releases/tag/exp-competitor-info-v7-299)). Adding the raw measurements to net1 v7's means gained the learned net2 +0.004 ([`exp-blackbox-val-295`](https://github.com/Radonux/palaestra/releases/tag/exp-blackbox-val-295)). A traffic head on net1 v7 recovers it (0.893, [`exp-traffic-head-v7-300`](https://github.com/Radonux/palaestra/releases/tag/exp-traffic-head-v7-300)); refitted on data from the policies that read it, it lifted the learned net2 among copies on the test links from +0.109 to +0.613 at a cost of 0.009 on the five-family mean, but lost 0.11 beside the competitors no training data had ([`exp-traffic-head-test-309`](https://github.com/Radonux/palaestra/releases/tag/exp-traffic-head-test-309)) and, on validation, 0.146 beside the 15 other stock controllers ([`exp-traffic-head-tq-val-314`](https://github.com/Radonux/palaestra/releases/tag/exp-traffic-head-tq-val-314)) | [results](results.md#what-limits-a-rule-is-the-other-flows) |
| Cost of meaning | In expliCCit's network exactly (1.27-million-weight encoder to seven latents, actor and critic of two layers of 256), each arm at its best training: the paper arm (neto1 v1's LCFs, held fixed) cloned BBRv3, since no reinforcement learner beat the clone; the free arm (encoder trained end to end) used TD3+BC. On held-out links (300 of net2's training links as a one-off and 300 unseen, each with and without competing flows; ten seeds per arm) free minus paper was +0.30 to +0.39 per step, every interval above zero (median over seeds +0.15 to +0.21; Mann-Whitney p 0.0004 to 0.007). The free arm tied BBRv3 with competition and beat it alone (+0.10, +0.15); the paper arm trailed it by 0.23 to 0.35, as longer queues and, alone, twice the drops. Cloned, the arms are level (p 0.12 to 0.47). The cost is in reading the LCFs, not estimating them: exact values did no better with the same actor, and hold nearly as much of the return (R^2 0.80 against 0.85) but far less linearly (0.42 against 0.79). A second encoder-shaped network between the LCFs and the actor, a different network, closed the gap on the exact LCFs (three of four sets within noise) and collapsed on the estimates (-2.24 minus BBRv3) | [`exp-lcf-cost-head-report-346`](https://github.com/Radonux/palaestra/releases/tag/exp-lcf-cost-head-report-346), [`exp-lcf-cost-truth-report-332`](https://github.com/Radonux/palaestra/releases/tag/exp-lcf-cost-truth-report-332), [`exp-lcf-cost-return-probe-333`](https://github.com/Radonux/palaestra/releases/tag/exp-lcf-cost-return-probe-333), [`exp-lcf-cost-deep-report-339`](https://github.com/Radonux/palaestra/releases/tag/exp-lcf-cost-deep-report-339), [results](results.md#giving-the-latents-a-meaning) |
| H4, H5, alarms | H4: run unchanged on sibling net1s (same LCFs, bins and scaler), the learned net2 moved as the fixed rule did, so it shows no sign of having learned its own net1's quirks ([`exp-h4-sibling-net1-val-285`](https://github.com/Radonux/palaestra/releases/tag/exp-h4-sibling-net1-val-285)). H5: a learned net2 trained on a net1 that never saw another TCP flow (-0.028 minus BBRv3 on the validation five-family mean) scored +0.165 run unchanged on net1 v7, against +0.154 for the same recipe trained on net1 v7, and gained in every family; among copies it lost 0.07 to that net2 ([`exp-notcp-swap-val-293`](https://github.com/Radonux/palaestra/releases/tag/exp-notcp-swap-val-293)). net1 v7's own costly error, a capacity that lags each step down of a stepping link and then stays too high, was not repaired: retrained on the capacity less constant-rate UDP, net1 moved the error to links shared with other flows and the learned net2, unchanged, lost 0.021 ([`exp-capacity-repair-v7-val-291`](https://github.com/Radonux/palaestra/releases/tag/exp-capacity-repair-v7-val-291)). Alarms: on net1 v7 they caught 97% to 100% of the episodes an injected fault made wrong, at 8% to 16% false alarms, and few of net1's own errors on held-out links, which the flow's measurements do not contradict ([`exp-alarms-v7-298`](https://github.com/Radonux/palaestra/releases/tag/exp-alarms-v7-298)) | [results](results.md#on-a-sibling-net1), [alarms](results.md#alarms-and-repair) |
| Reproduction | The link model equals a port of Mahimahi's link queue, event for event; none of expliCCit's three checked results reproduced here; the comparison on the original stack is UNRUN | [results](results.md#reproduction-of-expliccit) |
