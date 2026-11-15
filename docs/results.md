# Results

Every number here comes from a published release unless it says
otherwise; terms, families and the scoring are defined in
[methodology.md](methodology.md#terms). Scores are log utility, paired by
link, with 95% bootstrap intervals over links in brackets. "Minus BBRv3"
means a controller's score minus BBRv3's on the same links.

Runs from #69 on use the guest and TCP settings frozen at `1365f0e`;
the few earlier runs cited here decided questions those settings did not
touch ([validation.md](validation.md#tcp-buffer-limits)). The
evaluation links are one family (constant or stepping capacity of 8 to
100 Mbit/s, one-way delay of 5 to 50 ms, buffers of 0.25 to 4 BDP), so
every ranking below holds for that family and the listed conditions only.
The Linux controllers ran with their defaults; only the rules' constants
were tuned for this score.

## net1 estimates the path state within its training range

net1 v5 recovers five of the seven LCFs with R^2 of 0.88 or more on links
it never saw, and its 90% statements hold 91% to 97% of the time. It is
weak on free buffer, and it is not calibrated outside the ranges of its
training data.

net1 v5 on the broad dataset's held-out links
([`exp-net1v4-recheck-79`](https://github.com/Radonux/palaestra/releases/tag/exp-net1v4-recheck-79)):
R^2 of the belief's mean on the test links, then the coverage of its 90%
intervals on the test links, with controllers held out of training as the
observed flow (BBRv3, Westwood and a delay-threshold policy), with a BBRv3
or Westwood competitor (held out of training), and on three families
outside the training ranges.

| LCF | R^2 | 90% coverage | held-out controllers | held-out competitors | loss 3% to 6% | one-way delay 61 to 90 ms | buffers 8 to 16 BDP |
|---|---|---|---|---|---|---|---|
| mean queueing delay | 0.95 | 0.97 | 0.94 | 0.97 | 0.90 | 0.80 | 0.95 |
| queue | 0.90 | 0.92 | 0.89 | 0.92 | 0.91 | 0.89 | 0.90 |
| capacity | 0.81 | 0.92 | 0.88 | 0.92 | 0.88 | 0.84 | 0.90 |
| free buffer | 0.47 | 0.91 | 0.89 | 0.92 | 0.88 | 0.89 | 0.44 |
| busy fraction | 0.88 | 0.91 | 0.90 | 0.91 | 0.81 | 0.89 | 0.90 |
| one-way delay | 1.00 | 0.97 | 0.97 | 0.98 | 0.97 | 0.90 | 0.96 |
| receive rate | 0.99 | 0.94 | 0.93 | 0.94 | 0.92 | 0.92 | 0.94 |

- Free buffer depends on the buffer size, which sender-side measurements
  reveal only when the queue fills. net1's free-buffer intervals are
  calibrated (0.91), but its mean explains under half the variance.
- Capacity is identifiable only on a busy link. On net1 v3's held-out
  links, its 90% intervals for capacity were 12.3 Mbit/s wide on busy
  steps and 41.2 Mbit/s on idle ones (links of 4 to 200 Mbit/s; local
  run).
- Outside its ranges net1 v5 is not honest. Its bins end just above the
  training range, so one-way delays above 60 ms look covered (0.90) while
  the mean cannot leave the range (R^2 -2.54). On buffers deeper than any
  in training, its free-buffer statements hold 44% of the time.
- An unseen arrival pattern broke net1 v3, trained only with competitors
  arriving after the flow had measured the link alone: with Cubic from
  the start its capacity belief was 0.59 of the truth and its 90%
  statements held 26% of the time (local run), until the broad data
  repaired it ([alarms and repair](#alarms-and-repair)).
- A distribution per LCF was as accurate as a single value or an interval
  on every LCF but free buffer, with a CRPS 3 to 6 times below the single
  value's absolute error (expliCCit-like data, local runs).

Wider training data moves that edge; it does not remove it. net1 v6
has net1 v5's settings and data plus links with one-way delays of 5 to
150 ms, buffers of 0.25 to 32 BDP, base capacities of 1 to 200 Mbit/s and
120 s episodes ([`exp-data-net1-v6-146`](https://github.com/Radonux/palaestra/releases/tag/exp-data-net1-v6-146), [`exp-net1v6-149`](https://github.com/Radonux/palaestra/releases/tag/exp-net1v6-149)).
On v5's own test links it is as good (CRPS skill 0.782 to 0.784 over
three seeds against 0.778, calibration equal, though its free-buffer mean
is worse: R^2 0.22 to 0.30 against 0.47;
[`exp-net1v6-inrange-152`](https://github.com/Radonux/palaestra/releases/tag/exp-net1v6-inrange-152)). On the families v5 never saw it is
honest (one-way delays of 61 to 90 ms: R^2 0.97, 90% coverage 0.97).
Beyond its own new ranges it fails again, more gently: at one-way delays
of 151 to 250 ms its one-way-delay statements hold 27% of the time (R^2
-3.84, against v5's -24.1). net1 v6 replaces v5 for new work, and the
fixed rule on it keeps u = 0.25 ([`exp-rule-net1v6-val-153`](https://github.com/Radonux/palaestra/releases/tag/exp-rule-net1v6-val-153)).

### On real cellular traces

No training link resembles a cellular trace, whose capacity changes every
few milliseconds and drops to zero. On the cellular family's 27 links
(Cubic, BBRv1, Reno and Vegas, alone and against Cubic, 216 episodes;
[`exp-net1v6-cellular-207`](https://github.com/Radonux/palaestra/releases/tag/exp-net1v6-cellular-207)),
net1 v6 keeps the LCFs the flow measures and loses those of the path:

| LCF | 90% coverage | R^2 | Mean absolute error |
|---|---|---|---|
| one-way delay | 0.95 | 0.86 | 1.3 ms |
| receive rate | 0.88 | 0.89 | 1.1 Mbit/s |
| mean queueing delay | 0.80 | 0.77 | 17 ms |
| capacity | 0.77 | -0.95 | 7.4 Mbit/s |
| free buffer | 0.80 | -106 | 227 packets |
| queue | 0.69 | -29 | 55 packets |
| busy fraction | 0.65 | -0.06 | 0.15 |

R^2 is far below 0 for queue and free buffer because the truth hardly
varies: these links' queues and buffers hold a few packets. Over a
typical episode (Cubic alone on AT&T's LTE driving trace, 7.7 Mbit/s on
average, a 41-packet buffer; every LCF over time in
[`exp-net1v6-cellular-plots-208`](https://github.com/Radonux/palaestra/releases/tag/exp-net1v6-cellular-plots-208)),
net1 follows queueing delay, one-way delay and receive rate. Its capacity
is off by 8.6 Mbit/s on average, mostly too high. It believes the link busy (about 0.95,
against 0.66 on average): the trace offers capacity in bursts the flow
cannot use. And it believes about 1,500 packets of buffer are free where
33 are, since a buffer's size shows only when the queue fills.

### Split by link, not by row

expliCCit's trainer splits rows at random, which puts a training row of
the same episode 10 ms from each test row. With expliCCit's own settings
on our expliCCit-like data
([`exp-net1-row-split-97`](https://github.com/Radonux/palaestra/releases/tag/exp-net1-row-split-97),
three seeds), a row split gives every LCF but free buffer an R^2 of 0.96
or more. Split by link, free buffer falls from 0.65 to -0.22, busy
fraction from 0.96 to 0.75, and the mean absolute error of capacity grows
4.0 times (one-way delay 2.1, receive rate 1.9). A row split measures
recall of training links, not estimation.

### Breadth of links mattered more than any setting

With few links the min RTT nearly names a link and net1 recalls its
capacity: on 132 training links net1 v3's validation loss rose from the
first epoch and its 90% capacity statements held 71% of the time; 1,050
more links raised that to 0.89 (local runs, as is everything on net1 v3
here without a release). On the broad data six settings of dropout and
width span a mean CRPS skill of only 0.768 to 0.780
([`exp-net1v4-grid-10`](https://github.com/Radonux/palaestra/releases/tag/exp-net1v4-grid-10)),
lower learning rates did worse
([`exp-net1v4-lr-12`](https://github.com/Radonux/palaestra/releases/tag/exp-net1v4-lr-12)),
the choice held over two more seeds
([`exp-net1v4-seeds-13`](https://github.com/Radonux/palaestra/releases/tag/exp-net1v4-seeds-13)),
and every member's validation loss was lowest after 1 to 4 epochs. For
the same reason a minimal set of quantities (capacity, queue, buffer
size, one-way delay, cross-traffic share) lost to the seven LCFs (mean
CRPS skill 0.74 against 0.81): buffer size and cross-traffic share are
constant over a link, so predicting them rewards recognizing it.

## The split costs little

Even the simplest net2, the fixed rule (a window of 1.19 believed BDPs),
beats every Linux controller on the held-out test links, and the same
rule on the true LCFs does only 0.023 better.

The final test
([`exp-final-test-100`](https://github.com/Radonux/palaestra/releases/tag/exp-final-test-100)),
27 test links, every candidate fixed on validation, no failed episode;
minus BBRv3:

| Controller | training-20 | training-60 | new-20 | deep-long-20 | cubic-first-60 | five-family mean | held-competitors-60 | copies-30s |
|---|---|---|---|---|---|---|---|---|
| fixed rule on the true BDP | +0.152 | +0.148 | +0.291 | +0.205 | -0.014 | +0.156 | +0.120 | not run |
| share rule | +0.083 | +0.101 | +0.285 | +0.227 | +0.067 (-0.055 to +0.186) | +0.153 | +0.149 | -0.271 (-0.590 to +0.027) |
| fixed rule | +0.098 | +0.108 | +0.262 | +0.238 | -0.041 (-0.197 to +0.106) | +0.133 | +0.105 (-0.014 to +0.216) | +0.047 (-0.202 to +0.279) |
| BBRv1 | -0.031 | +0.020 | +0.209 | -0.027 | +0.111 | +0.057 | +0.120 | -0.261 |
| Scalable | -0.509 | -0.423 | -0.116 | -2.537 | +0.375 | -0.642 | +0.234 | not run |
| Sage | -0.364 | -0.494 | -0.356 | -0.491 | -0.852 | -0.511 | -0.881 | +0.004 |
| expliCCit's released policy | -1.573 | -1.433 | -1.358 | -2.200 | -1.676 | -1.648 | -1.581 | -1.288 |

- The fixed rule leads BBRv3 by 0.133 over the five families and BBRv1,
  the best Linux controller on that mean, by 0.076. It leads in every
  family but cubic-first, where it ties BBRv3 and the aggressive
  controllers win (Scalable +0.375, BIC +0.334, H-TCP +0.140, BBRv1
  +0.111).
- The rule on the true BDP scores 0.023 more over the five families: what
  net1 v5's estimate costs this rule. On net1 v3, the same rule on a
  classical estimate (largest delivery rate over 10 s times the min RTT,
  as BBR builds it) scored 0.155 below the rule on net1, and 0.228 below on
  a second held-out set of 158 links: net1 closed about 85% of the gap
  between that estimate and the truth (local runs).
- Copies of the fixed rule hold 0.99 utilization (BBRv3's 0.96) with 22 to
  29 ms of queueing delay (28 to 33 ms) and share as evenly as BBRv3's
  with 2 copies (Jain's index 0.979 against 0.984), less so with 4.
- Against Sage's released policy alone, the fixed rule has 7% to 14% more
  utilization on clean links and 1.7 to 2.6 times Sage's under 1% and 3%
  random loss, with up to 70% less queueing delay on deep, long paths.

For a learned net2 the split's lead is its start. The same recurrent
PPO, mixture and budget on three inputs (rounds 19, 19b and 20b,
[`exp-net2-r19-blackbox-164`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-r19-blackbox-164),
[`-r19b-bbr-165`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-r19b-bbr-165),
[`-r20b-bb-ppo-171`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-r20b-bb-ppo-171)),
minus net1 v6's fixed rule on the new family's validation links, three
seeds each, at the start and at the last evaluation (iterations 79 to
119):

| The actor reads | Start | Last evaluation |
|---|---|---|
| the true LCFs (A1) | +0.040 | -0.034 to -0.015 |
| net1 v6's beliefs (A3) | -0.008 | -0.040 to +0.017 |
| the raw measurements, its window on BBR's BDP estimate (A4) | -0.554 | -0.387 to -0.217 |
| the same, started from a clone of the rule on net1 (A4) | -0.087 to -0.064 | -0.096 to -0.036 |

From its own start, a window of 1.19 times BBR's estimate, the black box
learned (u rose from 0.25 to 0.5) but stayed 0.2 to 0.4 behind; with
Cubic from the start it ended 0.63 below. A window aimed at the flow's
own delivery rate shrinks as a competitor takes share, where net1's
capacity estimate does not (inferred). Started instead from a clone of
the rule on net1, learned from the raw measurements by DAgger with that
rule as the teacher
([`exp-net2-r20a-clone-170`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-r20a-clone-170):
40 iterations brought it from 0.554 to 0.064 to 0.087 below the rule,
and 80 more only to 0.031 to 0.067 below,
[`exp-net2-r20c-clone-more-175`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-r20c-clone-more-175)),
the same PPO ended 0.036 to 0.096 below, about 0.05 below the arm on net1
on average and level with it on one seed. Given the split's start, the
raw measurements learn from the reward nearly as well; what the split
gives is that start, a window on an estimate of the path rather than of
the flow's own throughput, which the black box reached only by imitating
the split. Off the family they trained on, the inputs part ways. On every
validation family
([`exp-net2-eval-ladder-174`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-eval-ladder-174),
final policies, campaign seed 21) the black box trailed the arm on net1 in
each family, by 0.046 on the five-family mean (-0.063 against -0.017;
-0.015 on the truth), and gained less with Cubic from the start (+0.045
against +0.130), but as copies it beat the rule (+0.183, +0.009 to +0.351)
where the arms on net1 and on the truth lost (-0.256 and -0.706): a
window on the flow's own delivery rate cooperates with copies and yields
to Cubic, the trade-off of the rules in the next section. Inference is the
split's cost: a decision of the policy on net1
v6 takes about 1 ms on one core, the black box's 0.07 ms
(`tools/policy-bench`, local). On the true LCFs, the learner lost its
start's lead and ended level with the arm on net1: what the truth adds,
learning from the reward did not use, and the drift toward pushing (next
sections) is the learner's, not net1's errors.

Nothing was chosen on two outside benchmarks, ccperf's experiments
([`exp-ccperf-77`](https://github.com/Radonux/palaestra/releases/tag/exp-ccperf-77),
[`exp-ccperf-v5-87`](https://github.com/Radonux/palaestra/releases/tag/exp-ccperf-v5-87))
and ccBench's suites
([`exp-ccbench-merged-92`](https://github.com/Radonux/palaestra/releases/tag/exp-ccbench-merged-92)),
which the rules on net1 v5 ran unchanged:

| Benchmark, measure | Fixed rule | Share rule | BBRv1 | BBRv3 | Cubic | Sage |
|---|---|---|---|---|---|---|
| ccperf single flow, utilization | 0.917 | 0.911 | 0.863 | 0.861 | 0.621 | 0.454 |
| ccperf random loss 0.1% to 5%, utilization | 0.972 | 0.977 | 0.978 | 0.684 | 0.415 | 0.528 |
| ccperf deep buffers, queueing delay (ms) | 2.6 | 2.4 | 19.3 | 8.7 | 74.6 | 4.8 |
| ccperf Poisson rate traces, queueing delay (ms) | 314 | 365 | 244 | 211 | 523 | 540 |
| ccperf against Cubic, share over fair share | 0.847 | 1.088 | 0.976 | 0.746 | 1.076 | 0.303 |
| ccBench solo, games won | 73.5% | 66.0% | 42.4% | 25.0% | 3.4% | 25.2% |
| ccBench, a Cubic flow first, share of capacity | 0.398 | 0.426 | 0.497 | 0.399 | 0.369 | 0.206 |

The rules keep the link full with a short queue alone, under random loss
and on deep buffers; they queue more than BBR on traces whose capacity
jumps, and take less than half the link against a Cubic flow that started
first, the same gap as cubic-first-60.

Compute (R6,
[`exp-overhead-206`](https://github.com/Radonux/palaestra/releases/tag/exp-overhead-206):
the guest's tract on one core of a GitHub runner's AMD EPYC 7763, medians
per decision). The fixed rule on net1 v5 takes 0.61 ms and the learned
net2 on net1 v6 0.70 ms, 6% and 7% of a core at the guest's 100 decisions
a second. Nearly all of it is net1's three-member ensemble: the learned
net2 adds 0.09 ms to it. Sage's policy takes 0.37 ms, expliCCit's 0.12 ms,
Orca's 0.06 ms (50 a second) and Aurora's 0.04 ms. Explaining one of
expliCCit's decisions exactly (Shapley values over its 7 LCFs, 128 actor
evaluations, in torch) takes 0.79 ms: 8% of a core to explain every
decision at 100 a second, little to explain only those an alarm flags. In the guest, inference costs no virtual time.

## What limits a rule is the other flows

Pushing against a buffer-filling flow pays, pushing against copies of
oneself hurts, and the seven LCFs do not say which is present. Told the
number of flows, a rule on the true state handles both.

The measurements do say it; net1's beliefs lose it. A per-step classifier
of the link's other traffic (none, copies, a stock TCP flow), scored on
held-out validation links with net1 v6's fixed rule as the observed flow,
reaches a balanced accuracy of 0.753 from the raw measurements and 0.496
from net1 v6's beliefs (chance 0.333), and calls a stock flow copies 14%
of the time from the measurements against 33% from the beliefs
([`exp-net2-competitor-info-177`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-competitor-info-177)).
net1 can learn it: a head on net1's scaled measurements, trained on the
training links (round 21,
[`exp-net1-traffic-data-178`](https://github.com/Radonux/palaestra/releases/tag/exp-net1-traffic-data-178)), reaches 0.909
on the same validation links and calls a stock flow copies 15% of the time
(BBRv1 6%, Cubic 11%, the held-out BBRv3 and Westwood 19% and 25%;
[`exp-net1-traffic-head-val-180`](https://github.com/Radonux/palaestra/releases/tag/exp-net1-traffic-head-val-180)).

- The share rule matches the rule on the true BDP over the five families
  and gains on the fixed rule where competitors fill the buffer (+0.108
  with Cubic from the start, +0.044 with the held-out competitors), but
  its copies lose (-0.271 against BBRv3's copies, where the fixed rule's
  have +0.047; `exp-final-test-100`). Each copy believes its share is
  small and pushes.
- Rules on the true LCFs fail among copies
  ([`exp-share-oracle-val-20`](https://github.com/Radonux/palaestra/releases/tag/exp-share-oracle-val-20),
  net1 v4, validation). Each copy aims at the whole link's BDP, so 2 and 4
  copies of the fixed rule on the truth build queues of 41 and 66 ms and
  score 1.06 below the same rule on net1 v4. net1 believes a shared link
  smaller than it is, which helps by accident: net1 v6's capacity belief
  is 1.03 to 1.04 of the link alone, 0.72 to 0.82 of it with 2 copies and
  0.46 to 0.61 with 4, between the fair share and the link
  ([`exp-net1-belief-share-173`](https://github.com/Radonux/palaestra/releases/tag/exp-net1-belief-share-173)).
  The capacity this flow can take is not among the LCFs.
- A hand-made switch does not fix it
  ([`exp-mode-val-19`](https://github.com/Radonux/palaestra/releases/tag/exp-mode-val-19)).
  A cooperative window that follows the flow's own share makes copies
  lead the fixed rule's copies by 0.44 (queues of 5 and 8 ms), but yields
  to Cubic (1.7 below the fixed rule with Cubic from the start). Switching
  between it and the share rule on net1's belief of a full buffer kept
  neither gain.

Told the number of flows N (constant and handed to the rule), a rule on
the truth that aims at 1/N of the BDP and pushes only below 1/N of the
capacity in use handles both
([`exp-fairshare-used-val-23`](https://github.com/Radonux/palaestra/releases/tag/exp-fairshare-used-val-23),
validation, minus BBRv3):

| Rule | alone | 2 or 4 copies | Cubic from 0 s |
|---|---|---|---|
| fixed rule on net1 v4 | +0.180 | +0.111 | -0.029 |
| 1/N rule on the truth, k = 2 | +0.199 | +0.460 | -0.286 |
| 1/N rule on the truth, k = 4 | +0.187 | +0.398 | +0.065 |
| 1/N rule on the truth, k = 6 | +0.166 | -0.605 | +0.249 |

At k = 4 it matches the fixed rule alone and leads it with copies (by
0.29) and with Cubic from the start (by 0.09). This is an oracle bound: k
was chosen on these links and N never changes. N is not added to the
LCFs: it describes the other endpoints, not the path, changes as flows
come and go, and one sender's measurements fit several values of it. It
is net2's to infer from how the others respond.

## Learned net2 against learned neto2

This section uses the guest with the cross-flow fix (`312d15a`, data from
`exp-data-net1-broad-30s-218` on); the sections after it used the earlier
guest. net2 reads our net1, which states a distribution per LCF; neto2
reads expliCCit's kind of net1 (neto1), which gives one value per LCF
([training.md](training.md#model-names)). The question is whether, with
the learner held fixed, the estimator changes how well the learned policy
acts.

[`exp-net2-vs-neto2-val-253`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-vs-neto2-val-253)
ran every policy in closed loop on the validation links (campaign seed
21, no failed episodes). The offline arms on both sides are the same
TD3+BC actor, data, arms and seeds
([training.md](training.md#net2-from-step-data-train-run)); net2 reads
either the means or each distribution pooled into 4 bins (bins4). r25 is
round 25's online DAgger student on net1 v9
([`exp-net2-r25-v9-249`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-r25-v9-249)),
taken after iteration 39. Each row is the mean over three training seeds
of log utility minus BBRv3; the mean is over the five validation families
and leaves copies out.

| Arm | Learner | Mean | Seed range | Cubic first | Deep buffer | Copies |
|---|---|---|---|---|---|---|
| net2, r25, net1 v9 means | DAgger | +0.144 | +0.135 to +0.158 | -0.058 | +0.247 | +0.044 |
| net2, r25, net1 v9 distributions | DAgger | +0.121 | +0.120 to +0.122 | -0.135 | +0.237 | -0.153 |
| net2, net1 v9 means, alpha 10 | TD3+BC | +0.118 | +0.098 to +0.146 | +0.225 | +0.002 | -0.460 |
| net2, net1 v9 means, alpha 5 | TD3+BC | +0.106 | +0.041 to +0.142 | +0.178 | -0.017 | -0.535 |
| net2, net1 v7 bins4, alpha 10 | TD3+BC | +0.092 | +0.018 to +0.133 | +0.224 | -0.170 | -0.689 |
| net2, net1 v7 bins4, alpha 5 | TD3+BC | +0.088 | +0.050 to +0.136 | +0.223 | -0.184 | -0.636 |
| neto2, neto1 v2, alpha 5 | TD3+BC | +0.056 | +0.024 to +0.076 | -0.081 | +0.059 | -0.217 |
| neto2, neto1 v2, alpha 10 | TD3+BC | +0.048 | +0.026 to +0.088 | -0.018 | -0.015 | -0.272 |
| neto2, neto1 v1, alpha 10 | TD3+BC | +0.033 | +0.031 to +0.036 | -0.182 | +0.026 | -0.106 |
| neto2, neto1 v1, alpha 5 | TD3+BC | +0.013 | -0.008 to +0.046 | -0.192 | +0.000 | -0.070 |
| net2, net1 v9 bins4, alpha 5 | TD3+BC | -0.003 | -0.076 to +0.076 | -0.045 | -0.271 | -0.226 |
| net2, net1 v9 bins4, alpha 10 | TD3+BC | -0.005 | -0.038 to +0.018 | -0.047 | -0.308 | -0.162 |
| net2, net1 v7 means, alpha 5 | TD3+BC | -0.045 | -0.083 to +0.002 | -0.313 | -0.086 | +0.155 |
| net2, net1 v7 means, alpha 10 | TD3+BC | -0.078 | -0.101 to -0.066 | -0.364 | -0.193 | +0.181 |
| rule on net1 v9, u = 0.25 (reference) | none | +0.142 | | -0.077 | +0.259 | -0.227 |
| rule on neto1 v2, u = 0.375 (reference) | none | +0.138 | | +0.013 | +0.249 | -0.761 |

- With the offline learner held fixed, the estimator made no reliable
  difference. On 27 LCFs read as means, net2 led neto2 by about 0.06; on
  7 LCFs read as means, it trailed by 0.06 to 0.11. Counting copies as a
  sixth family, every offline arm on both sides fell between -0.040 and
  +0.022.
- The offline net2 arms traded rather than gained: they won where Cubic
  starts first and lost in deep buffers and as copies, the shape of a
  more aggressive policy. neto2's arms sat between the two.
- The online DAgger student on net1 v9's means is the strongest learned
  policy: level with the rule over the five families, and the only one
  ahead of it as copies. It has no neto2 twin yet, so its lead over the
  offline neto2 arms mixes the estimator with the learner. Round 26
  trains the same recipe on neto1 v2 (`experiments/neto2-r26-v2.json`,
  UNRUN when written).
- Each neto2 arm scored 0.009 to 0.011 below its value at the same
  campaign seed in
  [`exp-neto2-search2-val-247`](https://github.com/Radonux/palaestra/releases/tag/exp-neto2-search2-val-247),
  and so did the rule on neto1 v2. The shift is in BBRv3's own episodes:
  on the same links BBRv3 scored higher in release 253 (deep-long +0.037,
  new +0.016, Cubic first +0.015), while the other controllers moved
  less. Compare policies within one release, as every table here does.

These are validation links before any tuning of net2, so they choose what
to run next and support no claim about the test links.

## A learned net2 beats the rule against competitors

A learned net2 on net1 v6 beat the fixed rule against competitors on the
test links and matched it everywhere else, copies included. It is round
7e's recipe trained again on v6 (round 18): DAgger students that learn,
from net1's output alone, two beliefs (p, that a stock TCP flow shares
the link, and q, that copies of itself do), run as one policy on three
seeds' mean beliefs, with q vetoing the push. It was picked among seven
candidates on the validation links
([`exp-net2-eval-v6-veto-166`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-eval-v6-veto-166)),
passed its gate alone on new validation episodes
([`exp-net2-gate-r18e05-s11-167`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-gate-r18e05-s11-167)),
and then ran once on the 27 test links
([`exp-net2-test-r18e05-168`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-test-r18e05-168)),
with no failed episode:

| Family | net2 minus the rule | net2 minus BBRv3 | rule minus BBRv3 |
|---|---|---|---|
| training-20 | -0.002 (-0.013 to +0.010) | +0.093 | +0.095 |
| training-60 | +0.011 (-0.001 to +0.026) | +0.123 | +0.112 |
| new-20 | +0.018 (+0.001 to +0.035) | +0.271 | +0.253 |
| deep-long-20 | -0.005 (-0.025 to +0.010) | +0.257 | +0.262 |
| cubic-first-60 | +0.068 (+0.023 to +0.120) | -0.041 | -0.109 |
| five-family mean | +0.018 | +0.141 | +0.123 |
| new-20, competition | +0.039 (+0.005 to +0.075) | | |
| held-competitors-60 | | +0.104 | +0.089 |
| copies-30s | -0.005 (-0.067 to +0.053) | -0.017 | -0.012 |

The gains are small and held across both validation seeds and the test:
with Cubic from the start +0.049, +0.056 and +0.068, in the competition
conditions +0.031, +0.027 and +0.039, each interval above zero. The same
recipe on net1 v5 (round 7e) passed one of 19 gates there, but on test
its competition gain shrank to +0.019 (-0.015 to +0.051), with +0.042
(+0.004 to +0.083) with Cubic from the start
([`exp-net2-test-r7e-k0.5-134`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-test-r7e-k0.5-134)).
Reinforcement learning from the reward gains more against competitors
but pushes among copies (below). What worked, twice, is imitation of a
privileged teacher, explicit beliefs about the other flows, and averaging
the seeds.

The candidate kept most of its gain on a net1 it was never trained on.
net1 v6's other two seeds share seed 0's LCFs, bins and input scaler, and
the unchanged policy on them, each against the rule on the same seed
(validation seed 11,
[`exp-net2-swap-v6-169`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-swap-v6-169)),
had +0.036 and +0.046 with Cubic from the start (+0.051 on seed 0) and
+0.031 and +0.034 in the competition conditions (+0.035), but with
intervals reaching below zero, and -0.063 (-0.145 to +0.008) and +0.041
as copies. net2 acts mostly on what the beliefs mean; another seed costs
part of the margin.

On net1 v6, PPO from the reward gained more against competitors than any
other net2 (up to +0.23 with Cubic from the start) and lost 0.22 to 0.44
as copies, averaged over seeds or not; more training made it worse, and a
veto by round 18's copies belief halved the copies loss but not enough
(rounds 14, 15 and 17;
[`exp-net2-eval-v6-159`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-eval-v6-159), table in
[rounds.md](rounds.md)).

Reinforcement learning from the reward follows its training mixture.
With noise drawn anew at each decision every learner stayed at the rule;
with noise held 200 ms PPO learned to push, which pays where competitors
fill the mixture and costs alone and among copies; a longer horizon pushed
harder, and longer episodes did not help at equal compute (rounds 1 to 16,
[rounds.md](rounds.md)). Imitation of a privileged teacher learns state
dependence instead: DAgger students learn from net1's output p, that a
stock TCP flow shares the link, and q, that copies of itself do. With q
vetoing the push, `u = u_rule + p'(1 - q')(share - u_rule)`, mistaking
Cubic for copies costs at most the push, and one policy on three seeds'
mean beliefs passed on net1 v5 and again on v6. Following the cooperative
window where q believes copies instead trades Cubic for copies on v6,
even averaged over seeds: +0.211 (+0.028 to +0.398) as copies but -0.156
(-0.288 to -0.037) with Cubic from the start
([`exp-net2-eval-v6-coop-176`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-eval-v6-coop-176)).
Yielding to Cubic costs more than pushing among copies, so the veto,
which only withholds the push, is the safer use of an imperfect q.

A better q moves the trade-off but does not remove it. With q from net1's
traffic head (round 21, above) in place of the students', each minus the
rule at campaign seed 21
([`exp-net2-eval-v6-head-179`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-eval-v6-head-179),
[`-head2-182`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-eval-v6-head2-182)):

| q from | Uses q to | Five-family mean | cubic-first-60 | copies-30s |
|---|---|---|---|---|
| the students (r18e05) | veto the push | +0.017 | +0.049 | -0.022 |
| net1's head, past 0.7 | veto the push | +0.018 | +0.051 (+0.009 to +0.100) | -0.002 |
| the students | cooperate | | -0.156 | +0.211 |
| net1's head | cooperate | -0.012 | -0.038 | +0.290 (+0.097 to +0.510) |
| net1's head, past 0.5 to 0.9 | cooperate | +0.000 to +0.005 | +0.005 to +0.011 | +0.204 to +0.315 |
| net1's head, p from it too (no learned output) | cooperate | -0.018 | -0.093 (-0.282 to +0.054) | +0.418 (+0.211 to +0.617) |

The copies belief can live in net1 at no cost. Cooperating on it repairs
most of the Cubic loss and gains more as copies, net2 unchanged, but gives
up the gain against Cubic at every threshold, with wide intervals: a flow
that yields to Cubic by mistake seems not to win its share back
(inferred).

Every gate, evaluation and training run of each round, on net1 v5 and
v6, is listed in [rounds.md](rounds.md).

## Reproduction of expliCCit

Three of expliCCit's results were checked here, and none reproduced as
stated.

| expliCCit's result | Here | Evidence |
|---|---|---|
| net1 estimates every LCF with R^2 above 0.9 | Only with its random row split, and not for free buffer even then (0.65). On held-out links net1 v5 has 0.81 for capacity, 0.88 for busy fraction and 0.47 for free buffer | [`exp-net1-row-split-97`](https://github.com/Radonux/palaestra/releases/tag/exp-net1-row-split-97), `exp-net1v4-recheck-79` |
| The split costs about 5% utilization and 9% queueing delay against Sage | The other way for our split (above). expliCCit's released policy, as deployed, has 41% less utilization than Sage on clean test links and 24% to 25% less on clean validation links | `exp-final-test-100`, [`exp-sage-v5-val-94`](https://github.com/Radonux/palaestra/releases/tag/exp-sage-v5-val-94) |
| On a 24/48 Mbit/s link (40 ms one way, 1000-packet buffer), under-utilization traces to capacity underestimated by 25%, and the true capacity restores it | The released policy uses 0.48 of the link (0.34 to 0.53 on the eight links around it), and its capacity belief is within 14%; the true capacity gives 0.43. Its wrong beliefs are busy fraction (1.00 against 0.51) and queue (19 packets against 0); the true busy fraction gives 0.71 on all nine links. The deployed rounding of beliefs is not the cause | [`exp-expliccit-case-42`](https://github.com/Radonux/palaestra/releases/tag/exp-expliccit-case-42), [`exp-expliccit-case-91`](https://github.com/Radonux/palaestra/releases/tag/exp-expliccit-case-91) |

Divergences from expliCCit's and Sage's setting, each a possible cause:

- Linux 7.0 in UML in virtual time, not Sage's patched Linux 4.19 in real
  time, and inference that costs no virtual time. The runs above used
  `agent`, which paces once a step at cwnd * MSS / srtt; Sage's kernel
  paces every ACK at max(cwnd, packets_out) * MTU / srtt (read from its
  binary). `agent_pure` does exactly that ([validation.md](validation.md)),
  and on the test links it moves either policy's share by at most 0.07
  per scenario ([Baselines](#baselines)), so the pacing does not explain
  the results above.
- The policy computation is not a divergence: on our recorded messages,
  expliCCit's own deployed code takes our action at 11,972 of 11,973
  steps (local run, [validation.md](validation.md)).
- Sage and expliCCit's policies run on links they were not trained on,
  and expliCCit's ground-truth definitions are not in its code, so ours
  may differ: on our expliCCit-like data its released net1 had R^2 1.000
  for one-way delay, 0.361 for capacity, -0.743 for busy fraction and
  -59.0 for free buffer (local run).
- The emulator itself matches Mahimahi: the link model equals a port of
  Mahimahi's link queue event for event, and the state message equals
  `sage.cc`'s at every compared step ([validation.md](validation.md)). The
  run of the same four links on the original stack, which would separate
  the emulator from the rest, is UNRUN: the kit is `tools/crosscheck`, and
  palaestra's side is
  [`exp-crosscheck-93`](https://github.com/Radonux/palaestra/releases/tag/exp-crosscheck-93)
  (expliCCit's released policy uses 0.50 of the case-study link there,
  Sage 0.92).

### What the released policy's decisions are attributed to

Each decision of the released policy, as deployed (beliefs clamped and
rounded), was split into exact Shapley values over its seven LCFs: 2^7
evaluations per decision, against the mean LCF vector of the same
campaign's steps. The values sum to each decision's change within
2.4e-8 on every campaign
([`exp-expliccit-attribution-190`](https://github.com/Radonux/palaestra/releases/tag/exp-expliccit-attribution-190),
validation links, every fifth step).

| Campaign | Steps | Share of a decision's total abs value held by its largest three LCFs | Largest mean shares |
|---|---|---|---|
| Case Studies 1 and 2's links | 3,178 | 0.82 | one-way delay 0.36, receive rate 0.18, capacity 0.17 |
| New family | 45,085 | 0.77 | one-way delay 0.20, capacity 0.20, queue delay 0.18 |
| Cellular traces | 8,189 | 0.80 | queue delay 0.35, queue 0.19, one-way delay 0.13 |
| Rising capacity | 11,290 | 0.77 | capacity 0.25, one-way delay 0.18, queue 0.17 |

On the case-study links busy fraction gets exactly 0. As deployed, its
belief is the same at every step there, so it never differs from the
reference. Yet it is the wrong belief on that link (1.00 against 0.51),
and the true value lifts utilization from 0.48 to 0.71 (above). A
Shapley value against a mean reference says how a decision varies with
an input, not whether the input is right. A belief that is wrong but
constant gets no attribution, so attribution alone could not find this
fault; replacing the belief with the truth did. The leave-one-out
intervention tracks the Shapley value closely (correlation 0.83 to 0.99
for every LCF but capacity, 0.60 to 0.90 for capacity), so the policy
is close to additive over its LCFs, and capacity is the one that
interacts most with the others.

## Baselines

Each controller ran as the observed flow on the 27 test links of every
held-out family, and as 2 and 4 copies of itself:

- expliCCit's released policy as deployed;
- Sage's released policy;
- Orca's released policy, exported from Orca's repository and checked
  against Orca's own code
  ([`exp-orca-model-196`](https://github.com/Radonux/palaestra/releases/tag/exp-orca-model-196)),
  as an `agent_cubic` agent stepping every 20 ms: Linux 7.0's Cubic
  underneath, rescaled by each action, with Orca's per-period statistics
  from `TCP_CC_INFO`;
- Aurora, trained three times with PCC-RL's own code, as an `agent_rate`
  agent ([below](#aurora));
- BBRv1, BBRv3, Cubic and Vegas, each with Linux's defaults.

The two learned policies run as `agent_pure`, the controller of Sage's
kernel, on which both were deployed (its pacing, read from the kernel's
binary, is in [validation.md](validation.md)). That is 10,080 episodes,
and none failed:
[`exp-baselines-192`](https://github.com/Radonux/palaestra/releases/tag/exp-baselines-192)
(the stock controllers, and both policies as `agent`),
[`exp-baselines-pure-194`](https://github.com/Radonux/palaestra/releases/tag/exp-baselines-pure-194)
(both policies as `agent_pure`),
[`exp-baselines-orca-197`](https://github.com/Radonux/palaestra/releases/tag/exp-baselines-orca-197)
(Orca),
[`exp-baselines-aurora-204`](https://github.com/Radonux/palaestra/releases/tag/exp-baselines-aurora-204)
(Aurora, and every controller on Aurora's training links), and the report
of all four,
[`exp-baselines-report-205`](https://github.com/Radonux/palaestra/releases/tag/exp-baselines-report-205).
Families run for 30 s, except Cubic first (60 s) and expliCCit's
case-study links (80 s). Share is the observed flow's throughput over the
link's mean capacity. Delay is the mean queueing delay of its packets at
the bottleneck, in ms. The report adds 95% intervals, the 95th percentile
of delay and, under competition, Jain's index among the TCP flows.

| Family: condition | expliCCit | Sage | Orca | Aurora | BBRv1 | BBRv3 | Cubic | Vegas |
|---|---|---|---|---|---|---|---|---|
| training: clean | 0.57, 14 | 0.91, 10 | 0.92, 12 | 0.57, 9 | 0.95, 14 | 0.94, 15 | 0.97, 34 | 0.95, 10 |
| training: loss 1% | 0.58, 12 | 0.49, 3 | 0.67, 9 | 0.54, 9 | 0.92, 10 | 0.72, 4 | 0.15, 2 | 0.16, 2 |
| training: UDP from 5 s, Cubic from 8 s | 0.45, 27 | 0.48, 21 | 0.58, 24 | 0.24, 18 | 0.74, 30 | 0.67, 28 | 0.65, 35 | 0.38, 20 |
| new: ACK loss 1% | 0.55, 14 | 0.92, 10 | 0.92, 12 | 0.57, 9 | 0.95, 14 | 0.93, 15 | 0.97, 34 | 0.95, 11 |
| new: loss 0.1% | 0.59, 15 | 0.83, 5 | 0.81, 11 | 0.56, 10 | 0.95, 13 | 0.91, 13 | 0.45, 7 | 0.43, 4 |
| new: loss 3% | 0.61, 11 | 0.45, 5 | 0.45, 5 | 0.41, 7 | 0.91, 8 | 0.34, 2 | 0.07, 1 | 0.08, 2 |
| new: Cubic from 0 s | 0.45, 33 | 0.31, 35 | 0.41, 32 | 0.06, 36 | 0.65, 37 | 0.50, 35 | 0.53, 36 | 0.23, 29 |
| new: Cubic from 2 s and 10 s | 0.46, 32 | 0.33, 31 | 0.46, 31 | 0.10, 27 | 0.67, 38 | 0.55, 34 | 0.53, 37 | 0.25, 30 |
| new: BBRv1 from 4 s | 0.41, 33 | 0.26, 21 | 0.40, 27 | 0.09, 15 | 0.57, 32 | 0.35, 27 | 0.35, 34 | 0.20, 20 |
| cubic-first (60 s) | 0.43, 32 | 0.32, 35 | 0.42, 32 | 0.06, 37 | 0.66, 37 | 0.53, 35 | 0.52, 36 | 0.17, 29 |
| deep-long: clean | 0.25, 7 | 0.86, 94 | 0.88, 37 | 0.67, 33 | 0.95, 70 | 0.94, 83 | 0.98, 489 | 0.98, 200 |
| deep-long: loss 1% | 0.27, 8 | 0.60, 10 | 0.60, 18 | 0.63, 28 | 0.92, 64 | 0.77, 38 | 0.08, 2 | 0.08, 3 |
| deep-long: UDP from 5 s, Cubic from 8 s | 0.13, 141 | 0.38, 210 | 0.46, 171 | 0.21, 133 | 0.57, 226 | 0.62, 247 | 0.84, 503 | 0.61, 327 |
| rise: clean | 0.45, 14 | 0.77, 15 | 0.78, 19 | 0.32, 19 | 0.80, 26 | 0.76, 30 | 0.81, 29 | 0.70, 18 |
| cellular: clean | 0.33, 134 | 0.73, 135 | 0.52, 120 | 0.89, 160 | 0.80, 144 | 0.72, 126 | 0.66, 138 | 0.57, 85 |
| cellular: Cubic from 0 s | 0.30, 137 | 0.48, 136 | 0.34, 134 | 0.80, 160 | 0.69, 149 | 0.44, 131 | 0.37, 127 | 0.23, 108 |
| case studies (2 links, 80 s) | 0.73, 12 | 0.94, 13 | 0.99, 25 | 0.76, 35 | 0.96, 26 | 0.97, 51 | 1.00, 218 | 0.99, 28 |

| Copies | expliCCit | Sage | Orca | Aurora | BBRv1 | BBRv3 | Cubic | Vegas |
|---|---|---|---|---|---|---|---|---|
| 2: Jain's index | 0.90 | 0.94 | 0.98 | 0.94 | 0.99 | 0.98 | 0.98 | 0.87 |
| 2: link utilization | 0.72 | 0.97 | 0.94 | 0.64 | 0.95 | 0.95 | 0.98 | 0.96 |
| 2: mean queueing delay (ms) | 17 | 17 | 24 | 10 | 33 | 28 | 36 | 14 |
| 4: Jain's index | 0.89 | 0.91 | 0.94 | 0.82 | 0.94 | 0.98 | 0.97 | 0.84 |
| 4: link utilization | 0.80 | 0.98 | 0.98 | 0.69 | 0.97 | 0.96 | 0.98 | 0.98 |
| 4: mean queueing delay (ms) | 23 | 22 | 35 | 12 | 40 | 33 | 39 | 17 |

What the tables say:

- expliCCit's released policy keeps the queue short and leaves capacity
  unused: 0.25 to 0.73 of the link alone (0.57 on clean training links,
  where Sage has 0.91), and 0.72 as two copies. This repeats
  `exp-final-test-100` (above). Its share holds under 1% to 3% random
  loss, where Cubic and Vegas fall below 0.16. On clean deep-buffer
  links it keeps 7 ms of queue where the others keep 37 to 489 ms, at
  0.25 of the link.
- Sage on clean links (0.91 of the link, 10 ms) is close to Vegas (0.95,
  10 ms), with a lower 95th percentile of delay (23 against 29 ms). It
  keeps 0.83 under 0.1% loss, where Cubic and Vegas fall to 0.45 and 0.43.
  In the new family's three competition conditions and with Cubic first,
  it gets 0.26 to 0.33 of the link, the least of any controller but
  Vegas. In clean deep buffers it keeps 94 ms of queue.
- Orca rescales Cubic's window and cuts its queue: 0.92 of clean links at
  12 ms (Cubic 0.97 at 34 ms), 0.88 of clean deep buffers at 37 ms
  (Cubic 489 ms). Against Cubic it gets 0.41 to 0.46 of the link, less
  than Cubic itself (0.52 to 0.53) and more than Sage (0.31 to 0.33). Under random loss it keeps 0.81, 0.67 and 0.45 of the link at
  0.1%, 1% and 3%, far above Cubic (0.45, 0.15, 0.07) and below BBRv1.
  As copies it is about as fair as the stock controllers (0.98 and 0.94).
- BBRv1 has the most throughput under random data loss (0.1% to 3%) and
  in every competition but one: with deep buffers, Cubic fills them and
  takes 0.84. Against Cubic, BBRv1 takes about two thirds of the link;
  BBRv3 takes 0.50 to 0.55, but keeps only 0.34 under 3% loss, where
  BBRv1 keeps 0.91.
- Among copies, BBRv1, BBRv3, Cubic and Orca are the fairest (Jain's
  index 0.94 to 0.99) and Vegas the least fair (0.84 to 0.87), with
  Aurora at four copies (0.82). Sage and expliCCit sit between them on
  fairness, and between Vegas and the others on delay. Aurora's copies
  keep the shortest queues (10 and 12 ms) and leave a third of the link
  unused.
- Sage's kernel pacing changes little. Against the same policies run as
  `agent` (pacing once a step at cwnd * MSS / srtt, run #192), paired by
  link, each scenario's share moves by at most 0.07 (Sage under 1% loss
  with deep buffers, +0.069, -0.001 to +0.148), and the interval of the
  change includes 0 in 31 of the 38 scenario and policy pairs. No
  conclusion above depends on it.

- Aurora (the mean of three trained models) keeps short queues on our
  links but leaves capacity unused: 0.57 of clean links at 9 ms. It
  gives way to every competitor (0.06 against Cubic from the start, 0.09
  against BBRv1) and follows rising capacity slowly (0.32, where all but
  expliCCit get 0.70 to 0.81). On the cellular traces it gets the most of
  any controller, 0.89 of the link at 160 ms (BBRv1 0.80 at 144 ms).
  The next section says why.

### Aurora

Aurora has no released model, so we trained it three times with
PCC-RL's own code, unchanged (`64bea6e`: `stable_solve.py`'s PPO, 3.9
million steps in PCC-RL's simulator):
[`exp-aurora-train-199`](https://github.com/Radonux/palaestra/releases/tag/exp-aurora-train-199),
[`-201`](https://github.com/Radonux/palaestra/releases/tag/exp-aurora-train-201)
and
[`-202`](https://github.com/Radonux/palaestra/releases/tag/exp-aurora-train-202).
A fourth run, 200, failed 33 s into training, and its log did not keep
why. Each run took 66 minutes on one GitHub runner. Each one's moving
average of the episode reward reached 457 to 576 within 400 of its 9,954
episodes and stayed between 690 and 862 after that. The exports follow
PCC-RL's own code to 7.7e-6 in the rate factor
([`exp-aurora-model-203`](https://github.com/Radonux/palaestra/releases/tag/exp-aurora-model-203),
[validation.md](validation.md)).

Aurora acts once per monitor interval, a stretch of half the last
interval's mean RTT. It reads three features of each interval: the RTT's
trend (sent latency inflation), the mean RTT over the smallest so far
(latency ratio), and the sending rate over the receive rate (send
ratio). It multiplies its rate by about 1 + 0.025 a (PCC-RL's training
step; its deployment's is 0.05). It runs as an
`agent_rate` agent ([architecture.md](architecture.md)), from the
settings of PCC-Uspace's deployment: 6 Mbit/s at the start, kept in 0.5
to 300 Mbit/s. Each model also ran in two variants, each changing one
input:

- **blind**: the RTT samples PCC-Uspace's deployment gives the policy.
  `pcc_python_rc.cpp` (deep-learning branch, `2c4853a`) passes each
  interval's first and last sample, in seconds, through
  `PyLong_FromLong`. Any RTT under 1 s reaches the policy as 0, so its
  two latency features never move.
- **unbiased**: the receive rate counting every acknowledged packet.
  PCC-RL's `sender_obs.py` counts one fewer, so an interval of k
  delivered packets reads a send ratio of k / (k - 1), as if 1 / k of
  them were lost.

Each cell is the mean of the three models: share, with the models' range
in brackets, then mean delay in ms. The copies rows give link
utilization and Jain's index.

| Family: condition | Aurora | blind | unbiased |
|---|---|---|---|
| training: clean | 0.57 (0.41 to 0.66), 9 | 0.58, 31 | 0.88 (0.85 to 0.91), 11 |
| training: loss 1% | 0.54, 9 | 0.54, 28 | 0.87, 11 |
| new: loss 3% | 0.41, 7 | 0.43, 22 | 0.84, 10 |
| new: Cubic from 0 s | 0.06, 36 | 0.54, 44 | 0.11, 35 |
| deep-long: clean | 0.67, 33 | 0.69, 337 | 0.77, 36 |
| rise: clean | 0.32, 19 | 0.35, 34 | 0.57, 28 |
| cellular: clean | 0.89, 160 | 0.65, 151 | 0.89, 160 |
| case studies | 0.76, 35 | 0.78, 180 | 0.93, 38 |
| 2 copies | 0.64, 0.94 | 0.67, 0.97 | 0.92, 0.97 |
| 4 copies | 0.69, 0.82 | 0.65, 0.90 | 0.94, 0.88 |

Over the scenarios of the first table (the report's median), the models
as trained use 0.37, 0.64 and 0.62 of the link; unbiased, 0.73, 0.81 and
0.77, about BBRv1's 0.80 and more than any other learned controller.

Every controller also ran alone for 60 s on 27 links drawn as PCC-RL
draws Aurora's training links (1.2 to 6 Mbit/s, 50 to 500 ms one way,
buffers of 2 to 2,981 packets), clean and under 1% and 3% random loss
(family `aurora-range`). The report scores each episode by Aurora's own
reward: PCC-RL's per-interval reward without its 0.001 scale, 10 *
packets a second - 1000 * RTT in s - 2000 * loss ratio, from the
episode's means. Means over the three conditions:

| Controller | Share | Mean delay (ms) | Loss | Aurora's reward |
|---|---|---|---|---|
| Aurora (three models) | 0.94 | 989 | 0.088 | 929 (853 to 1038) |
| Aurora, unbiased | 0.94 | 1007 | 0.088 | 912 |
| Aurora, blind | 0.94 | 1457 | 0.145 | 352 |
| BBRv1 | 0.86 | 77 | 0.037 | 1727 |
| BBRv3 | 0.69 | 56 | 0.020 | 1256 |
| Vegas | 0.34 | 114 | 0.018 | 207 |
| Orca | 0.52 | 434 | 0.156 | 79 |
| Sage | 0.66 | 653 | 0.322 | -73 |
| Cubic | 0.39 | 512 | 0.027 | -81 |
| expliCCit | 0.04 | 4 | 0.015 | -488 |

On its own links Aurora uses more of the link than any other controller,
but by its own reward BBRv1 and BBRv3 beat it. Its mean delay comes from
the deep buffers: in the last 40 s of each episode, model 0 keeps 62 ms
of queue on the median link and 1.35 s averaged over links. On the deepest link
(1.45 Mbit/s, 2,514 packets, 21 s when full) its queue grows from 0.07 s
to 20 s over the episode; BBRv1's stays near 60 ms. Replayed locally
(not in a release) with the same graph, model 0 cuts its rate by about
8% per interval from 2 s on, the most it ever does. Each interval lasts half
the RTT, queue included, so it decides 27 times in 60 s, and its rate
falls below capacity only after about 40 s. It started at 6 Mbit/s,
four times the link's capacity; PCC-RL starts each training episode at
0.3 to 1.5 times the link rate.

### What Aurora's RL design teaches net2

1. **Inputs that are ratios carry a policy across scales, but only if
   their estimator is right where the policy runs.** Aurora never sent
   faster than 12 Mbit/s in training, nor saw an RTT under 100 ms.
   Unbiased, it uses 0.88 of our clean links (6 to 88 Mbit/s, 10 to 96
   ms of RTT), about as much as Sage (0.91). One packet in the receive-rate formula
   costs it 0.31 of the link. At the 6 Mbit/s start on a 20 ms RTT, an
   interval holds about 5 packets, which read as 20% loss, and each cut
   shrinks the next interval. Locally, model 0 fell from 6 to 0.5 Mbit/s
   in 2 s on an 87 Mbit/s link. Unbiased, it reached 86 Mbit/s in 5 s
   at 4 ms of queue, and the unchanged policy started at 60 Mbit/s held
   86 Mbit/s from its second second (both local runs). PCC-RL's intervals hold tens of packets, so training never
   showed the bias. An estimator supervised by ground truth over the
   conditions it will meet, few samples included, learns such a
   correction; a hand-written formula keeps its bias for good.
2. **Train from the states the deployment starts in.** Both of
   Aurora's large failures come from its start. Training starts at 0.3
   to 1.5 times the link rate; deployment starts at 6 Mbit/s. That is
   too slow on short RTTs (the bias above) and too fast on slow links
   with deep buffers (20 s of queue). net2's rollouts start where its
   deployment does, from the kernel's own start.
3. **Small steps per RTT make the loop slowest when it most needs to be
   fast.** Aurora's steps are multiplicative and small (the models grow
   0.86% to 1.30% per interval on an idle link, so doubling takes 27 to
   40 RTTs). Its intervals stretch with the queue it builds. It follows
   rising capacity with 0.32 of the link (unbiased 0.57, BBRv1 0.80).
   net2 sets the window as a multiple of the believed BDP, so one
   decision can land anywhere; what limits it is net1's estimate, not a
   step size.
4. **A latency signal with no model of who built the queue gives way to
   buffer fillers.** PCC-RL trains one sender alone. Against Cubic
   Aurora keeps 0.06 of the link (unbiased 0.11); blind, it keeps 0.54
   because it ignores latency. This is the limit
   [above](#what-limits-a-rule-is-the-other-flows): net2 trains with
   competitors and copies, and its net1 reports how busy the link is
   beyond this flow.
5. **A reward in absolute units does not fix one trade-off for a policy
   that sees only ratios.** Divided by 10 times the capacity C (packets
   a second), Aurora's reward is utilization - (100 / C) RTT - (200 / C)
   loss. Over its training links a queue of one base RTT costs 0.02 to
   1.0 of the link, a fiftyfold spread the policy cannot see. The models
   settle at an RTT 1.15 to 1.26 times the smallest, and keep growing
   until random loss reaches 8% to 15% (their probes in
   `exp-aurora-model-203`). This point is arithmetic, not a test. net2's
   log utility charges 0.5 per base RTT of queue on every link, from
   ground truth that only the reward sees.
6. **A gap between the training code and the deployed code does not
   show in the training curve.** PCC-Uspace's latency path (blind)
   more than triples Aurora's delay on clean links (9 to 31 ms) and multiplies it
   by ten in deep buffers (33 to 337 ms). On Aurora's own links it cuts
   Aurora's reward from 929 to 352. The deployment also steps twice as
   far (0.05 against 0.025) and reads two RTT samples instead of all.
   net2 runs one featurizer in training and in evaluation, and every
   export is checked against the code that trained it.
7. **One unseeded training run is not a result.** The three runs have
   alike training curves but use 0.41, 0.66 and 0.64 of clean links;
   most of the spread goes through the bias above (unbiased, 0.85 to
   0.91). One run in four failed. net2 is trained over seeds, chosen on
   validation and tested once.

## Alarms and repair

Alarms computed without ground truth catch the wrong beliefs that the
flow's own measurements contradict, as long as errors are rare in normal
running, and miss the ones the measurements are consistent with. A
capacity fault was repaired by retraining net1 alone, with net2
unchanged.

### Alarms (R2)

Each alarm is a statistic of one step, from net1's beliefs and the
sender's own measurements: physical invariants among the beliefs (the
receive rate at most capacity times busy fraction; queueing delay equal
to queue over capacity; queue plus free buffer constant), each belief's
entropy, and checks against the delivery rate and min RTT measured three
smoothed RTTs later. Its threshold is the 95th percentile of its episode
mean on validation links of the training data, so about 5% of right
episodes set off each alarm (3% to 6%), and 8% to 17% set off one of the
alarms that name a given LCF (`experiments/r2_alarms.py`). An episode is
wrong about an LCF when the truth lies outside net1's central 90% set in
more than half of its steps.

| Test | Release | net1 | Wrong beliefs caught | Missed |
|---|---|---|---|---|
| Families outside the training ranges | [`exp-r2-alarms-81`](https://github.com/Radonux/palaestra/releases/tag/exp-r2-alarms-81) | v3 | one-way delay on deep buffers 0.83 of 168 (AUC 0.91), queueing delay at 61 to 90 ms 0.89 of 90, capacity under 3% to 6% loss 0.63 of 287 | capacity under competition, 0.12 of 43 (AUC 0.52) |
| Real cellular traces | [`exp-net1v6-cellular-207`](https://github.com/Radonux/palaestra/releases/tag/exp-net1v6-cellular-207) | v6 | capacity 36 of 37 (AUC 0.95), queue 47 of 52, queueing delay 9 of 11, one-way delay 9 of 9 | busy fraction, 0 of 58 (AUC 0.50) |
| One LCF's belief halved or doubled in the fixed rule (21 validation links, 3 conditions, 63 episodes per fault) | [`exp-r2-faults-210`](https://github.com/Radonux/palaestra/releases/tag/exp-r2-faults-210) | v6 | 97% to 100% for queueing delay, queue, capacity, one-way delay, receive rate and busy fraction halved | none; free buffer's faults and busy fraction doubled were seldom wrong at all |
| expliCCit's released encoder on its own flow and on the cellular traces | [`exp-r2-expliccit-211`](https://github.com/Radonux/palaestra/releases/tag/exp-r2-expliccit-211) | expliCCit's | 13% to 29% of its errors | most, and its busy fraction on the case-study links (AUC 0.07) |

What the misses have in common:

- **A wrong belief the measurements agree with.** A capacity believed at
  0.59 of the truth under competition is still above the flow's own
  share, so nothing the flow measures contradicts it (inferred). On the
  cellular traces net1 believes the link busier than it is, and the one
  invariant on busy fraction (receive rate at most capacity times busy)
  fires only when busy fraction is too low. That is the belief
  expliCCit's released policy got wrong in Case Study 1
  ([above](#reproduction-of-expliccit)). An alarm for it needs a
  measurement that bounds busy fraction from above, and Sage's state
  message has none that does so reliably: the smoothed RTT's excess over
  the min RTT ranks with the true busy fraction at only 0.61 (Spearman)
  on the cellular traces (computed locally from the steps of
  `exp-net1v6-cellular-207`).
- **An encoder that is often wrong.** expliCCit's released net1 states
  one value per LCF, so it has no entropy alarm; we call a value wrong
  when it is off by more than a factor of 2 (busy fraction: by more than
  0.25). On the clean test links it is wrong in 41% of the (episode, LCF)
  pairs, where net1 v6, by its own 90% sets, is wrong in 1.1%. Thresholds
  at the 95th percentile of its normal running then sit above most of its
  errors: the alarms catch 14% of them on the test links, 13% on its own
  flow on the validation links and 29% on the cellular traces.

Two more properties, from the injected faults:

- **The alarms name a group of LCFs, not one.** Only one-way delay has an
  alarm of its own (the min RTT measured later), and it caught both of
  that LCF's faults. A wrong queueing delay or queue sets off the
  invariant that queueing delay equals queue over capacity, which names
  all three; a wrong receive rate sets off the delivery-rate check, which
  names receive rate and capacity. They say which group of heads to
  retrain.
- **Detection does not depend on whether the policy uses the belief.**
  The fixed rule's window is net1's capacity times its one-way delay, so
  only those faults moved it: halved, it used 0.64 to 0.65 of the link at
  2 ms of queue; doubled, 0.99 at 52 ms; unfaulted, 0.98 at 11 ms. The
  other ten faults left its summary the same to the last digit, and the
  alarms flagged them anyway.

### Repair (R3)

net1 v3's rule lost to BBRv3 with Cubic from the start because its
capacity belief was low there; the same rule on the true BDP did not.
Retraining net1 on the broad data repaired it with net2 unchanged: on the
test links with Cubic from the start, the fixed rule on net1 v4 beat the
same rule on net1 v3 by 0.428 (0.277 to 0.580)
([`exp-rule-net1v4-test-14`](https://github.com/Radonux/palaestra/releases/tag/exp-rule-net1v4-test-14);
under the frozen settings on validation, -0.555 on net1 v3, -0.037 on v4,
+0.058 on the truth, [`exp-r3-capacity-head-val-86`](https://github.com/Radonux/palaestra/releases/tag/exp-r3-capacity-head-val-86)).
Retraining only the capacity head
([`exp-r3-capacity-head-83`](https://github.com/Radonux/palaestra/releases/tag/exp-r3-capacity-head-83))
repairs its calibration (90% coverage 0.62 to 0.93) but recovers +0.16 in
closed loop where all of net1 gained +0.52: a new head on frozen features
cannot supply cues the features never learned (inferred).

Replacing one belief at a time with the truth localizes a fault (busy
fraction, in expliCCit's case study above), but a net2 adapts to its
net1's errors, so a patch can also hurt. For the learned net2 on net1 v6
(validation seed 11,
[`exp-net2-patch-v6-172`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-patch-v6-172)),
six of the seven patches moved it by 0.026 or less in every measure. The
true capacity gained against competitors (the competition conditions
+0.098 against +0.029) and collapsed as copies (-1.247 against -0.008),
as with all seven true (-1.205): each copy then aims at the whole link,
where net1's lower belief on a shared link (above) kept the copies safe.
That belief carries something about the other flows that the LCF leaves
out, and net2 relies on it.

## Stated uncertainty has not helped net2 yet

A net2 on the means of net1's beliefs (arm A2) did at least as well as
one on the full beliefs (A3). Over 60 iterations of online TD3+BC on net1
v3, scored every 10 iterations on the validation links, A3 minus A2 stayed
between -0.030 and +0.001; the one interval that excluded zero favoured
the means (one seed each, local runs, not run on test).

Those links left the spread little to say. The fixed rule alone keeps 96%
of validation steps on a fully busy link, where net1's capacity belief is
narrow (median log2 sd 0.14). Off a busy link the spread widens 3 to 6
times, stays honest (90% coverage 0.95 to 0.98) and tracks the error
(correlation 0.82; run 177's steps, local). Two families put the flow
where it cannot see the capacity: rise (alone, the capacity alternating
between half and twice its level every 2 s) and info (a Cubic or BBRv1
flow leaving at 10 s, UDP at half the capacity on and off every 2 s,
Cubic on and off every 3 s). There, rules that probe above the believed
BDP did no better than the fixed rule, whether they read the spread or
only the means. Each raises the window target by k times a signal: the
spread of log2 capacity past s0, or how idle net1 believes the link is
(1 minus its mean busy fraction, past 0.05). Minus the fixed rule
([`exp-net2-info-val-184`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-info-val-184), validation):

| Rule probes on | rise-30 | info-30 | on/off UDP | five-family mean | cubic-first-60 | copies-30s |
|---|---|---|---|---|---|---|
| believed idle share, k 1 (means) | -0.008 | -0.006 | -0.005 | +0.001 | +0.034 | +0.035 |
| believed idle share, k 2 (means) | -0.021 | -0.007 | -0.016 | -0.004 | +0.036 | -0.027 |
| spread, k 1, s0 0.2 | -0.087 | -0.035 | -0.131 | +0.022 | +0.138 | -0.389 |
| spread, k 2, s0 0.4 | -0.117 | -0.028 | -0.070 | +0.008 | +0.121 | -0.379 |
| spread, k 1, s0 0.2, times 1 - P(copies) | -0.051 | -0.021 | -0.059 | +0.007 | +0.091 | -0.049 |
| (BBRv3, for scale) | -0.627 | -0.161 | -0.115 | -0.142 | -0.007 | +0.100 |

The fixed rule's margin of 19% over a capacity net1 keeps re-estimating
already follows each fourfold rise; extra probing buys less than its
queue costs. net1 is as unsure of the capacity on any shared link
(median spread 0.34 to 0.42 with copies or a stock flow), so probing on
the spread mostly pushes there: it gains against Cubic and loses among
copies and under on/off UDP. For a window rule, the spread has so far
added nothing the means do not.

The learned net2 that passed did not need it either: round 18's recipe
trained on net1 v6's means (round 22) scores as r18e05 on its full
beliefs (+0.016 against +0.018 on the five-family mean, +0.053 each with
Cubic from the start, -0.004 and -0.003 on rise;
[`exp-net2-eval-r23-186`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-eval-r23-186)). PPO trained where
probing can pay (round 23) learned a window that rises where net1
believes the link idle and drops below the rule's where it is busy. That
wins on rise, info and on/off UDP (up to +0.23) and as copies (+0.43),
and loses with Cubic from the start (-0.29 to -0.36). Where net1 is
unsure of the capacity, those policies back off: their u falls with the
spread. Holding exploration 1 s instead of 200 ms did not help. net1's
honesty still serves the human reading of its beliefs.
