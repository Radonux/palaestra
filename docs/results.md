# Results

Every number here comes from a published release unless it says
otherwise; terms, families and the scoring are defined in
[methodology.md](methodology.md#terms). Scores are log utility, paired by
link, with 95% bootstrap intervals over links in brackets. "Minus BBRv3"
means a controller's score minus BBRv3's on the same links.

The evaluation links are one family (constant or stepping capacity of 8 to
100 Mbit/s, one-way delay of 5 to 50 ms, buffers of 0.25 to 4 BDP), so
every ranking below holds for that family and the listed conditions only.
The Linux controllers ran with their defaults; only the rules' constants
were tuned for this score. Results marked "old guest" predate the guest's
cross-flow fix (`312d15a`); the rest use the fixed guest, with data from
`exp-data-net1-broad-30s-218` on. Runs from #69 on use the TCP settings
frozen at `1365f0e`; the few earlier runs cited decided questions those
settings did not touch ([validation.md](validation.md#tcp-buffer-limits)).

## net1 estimates the path state within its training range

Our net1 (v9, a distribution per LCF) and expliCCit's kind of net1
retrained here (neto1 v2, one value per LCF) estimate the same 27 LCFs
about equally well on held-out links, and net1 v9 is not calibrated
beyond its training ranges. R^2 on the test links, which are split by link from the
training links
([`exp-net1v9-243`](https://github.com/Radonux/palaestra/releases/tag/exp-net1v9-243),
chosen seed;
[`exp-neto1-v2-239`](https://github.com/Radonux/palaestra/releases/tag/exp-neto1-v2-239),
mean of three seeds; the same rows), for expliCCit's seven LCFs and the
weakest of the other heads:

| LCF | net1 v9 | neto1 v2 |
|---|---|---|
| mean queueing delay | 0.98 | 0.97 |
| queue | 0.93 | 0.93 |
| capacity | 0.80 | 0.78 |
| free buffer | 0.29 | 0.23 |
| busy fraction | 0.88 | 0.86 |
| one-way delay | 1.00 | 0.99 |
| receive rate | 0.98 | 0.98 |
| random loss rate | 0.45 | 0.50 |
| active flows | 0.46 | 0.42 |
| normalized power | 0.49 | 0.54 |
| buffer size | 0.60 | 0.58 |

- net1 v9's 90% statements hold 89% to 97% of the time on every one of
  the 27 LCFs.
- Free buffer depends on the buffer size, which sender-side measurements
  reveal only when the queue fills; it is the weakest LCF for both.
- Beyond the training ranges net1 v9 is not honest: at one-way delays of
  151 to 250 ms its one-way-delay statements hold 26% of the time (R^2
  -3.75), and under random loss of 3% to 6% its loss-rate statements hold
  0% of the time (R^2 -14.91; its top loss bin ends at 0.02).
- Settings mattered little: six settings of dropout and width spanned a mean CRPS skill of only 0.768 to 0.780
  (net1 v4, old guest,
  [`exp-net1v4-grid-10`](https://github.com/Radonux/palaestra/releases/tag/exp-net1v4-grid-10)).
- On real cellular traces, outside every training range, net1 v6 (old
  guest) kept the LCFs the flow measures (90% coverage 0.80 to 0.95) and
  lost those of the path (0.65 to 0.80, R^2 below 0;
  [`exp-net1v6-cellular-207`](https://github.com/Radonux/palaestra/releases/tag/exp-net1v6-cellular-207)).

## The split costs little

On the test links
([`exp-dagger-twins-test-258`](https://github.com/Radonux/palaestra/releases/tag/exp-dagger-twins-test-258)),
the simplest net2, the fixed rule on net1 v9 (u = 0.25), scored +0.136
over BBRv3 on the five-family mean, and the learned net2 on net1 v9 the
same over six training seeds ([Learned net2 against learned
neto2](#learned-net2-against-learned-neto2)).

On the old guest, the fixed rule on net1 v5 beat every Linux controller
and Sage on the test links, and the same rule on the true LCFs did only
0.023 better
([`exp-final-test-100`](https://github.com/Radonux/palaestra/releases/tag/exp-final-test-100)).
A black box with the same learner and budget on the raw measurements
stayed 0.2 to 0.4 below net1 v6's fixed rule from its own start
([`exp-net2-r19b-bbr-165`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-r19b-bbr-165))
and ended about 0.05 below the same learner on net1 v6 when started from
a clone of the rule
([`exp-net2-r20b-bb-ppo-171`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-r20b-bb-ppo-171)),
so the split's lead was its start (old guest).
On ccperf's experiments
([`exp-ccperf-77`](https://github.com/Radonux/palaestra/releases/tag/exp-ccperf-77),
[`exp-ccperf-v5-87`](https://github.com/Radonux/palaestra/releases/tag/exp-ccperf-v5-87))
and ccBench's suites
([`exp-ccbench-merged-92`](https://github.com/Radonux/palaestra/releases/tag/exp-ccbench-merged-92)),
run unchanged on the old guest, the rules on net1 v5 kept the link full
with a short queue alone, under random loss and on deep buffers, queued
more than BBR on traces whose capacity jumps, and took less than half the
link against a Cubic flow that started first.

The split's cost is inference
([`exp-overhead-206`](https://github.com/Radonux/palaestra/releases/tag/exp-overhead-206),
one core of a GitHub runner's AMD EPYC 7763, medians per decision): the
fixed rule on net1 v5 takes 0.61 ms and the learned net2 on net1 v6
0.70 ms, 6% and 7% of a core at the guest's 100 decisions a second,
nearly all of it net1's three-member ensemble; a black box on the raw
measurements takes 0.07 ms (local). In the guest, inference costs no
virtual time.

## What limits a rule is the other flows

Pushing against a buffer-filling flow pays, pushing against copies of
oneself hurts, and the LCFs do not say which is present. As copies on
the validation links, the fixed rule on net1 v9 scored -0.228
(-0.532 to +0.076) and the rule on neto1 v2 -0.730 (-1.139 to -0.334)
([`exp-dagger-twins-iter-val-256`](https://github.com/Radonux/palaestra/releases/tag/exp-dagger-twins-iter-val-256));
on the test links they scored +0.054 and -0.593, and the learned net2 on
net1 v9 +0.104 (release 258).

On the old guest, rules on the true LCFs failed among copies, since each
copy aims at the whole link's BDP
([`exp-share-oracle-val-20`](https://github.com/Radonux/palaestra/releases/tag/exp-share-oracle-val-20)),
while told the number of flows a rule on the truth handled both copies
and Cubic (an oracle bound,
[`exp-fairshare-used-val-23`](https://github.com/Radonux/palaestra/releases/tag/exp-fairshare-used-val-23)).
The raw measurements tell copies from a stock TCP flow better than net1
v6's beliefs (balanced accuracy 0.753 against 0.496, chance 0.333;
[`exp-net2-competitor-info-177`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-competitor-info-177)),
and a traffic head on net1 learns it (0.909;
[`exp-net1-traffic-head-val-180`](https://github.com/Radonux/palaestra/releases/tag/exp-net1-traffic-head-val-180)).

## Learned net2 against learned neto2

This section uses the guest with the cross-flow fix (`312d15a`, data from
`exp-data-net1-broad-30s-218` on). net2 reads our net1, which states a
distribution per LCF; neto2 reads expliCCit's kind of net1 (neto1), which
gives one value per LCF ([training.md](training.md#model-names)). The
question is whether, with the learner held fixed, the estimator changes
how well the learned policy acts.

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
  ahead of it as copies. Its neto2 twin is below.
- Each neto2 arm, and the rule on neto1 v2, scored 0.009 to 0.011 below
  its value at the same campaign seed in
  [`exp-neto2-search2-val-247`](https://github.com/Radonux/palaestra/releases/tag/exp-neto2-search2-val-247).
  The shift is in BBRv3's own episodes: on the same links BBRv3 scored
  higher in release 253 (deep-long +0.037, new +0.016, Cubic first
  +0.015), while the other controllers moved less. Every table here
  compares policies within one release.

These are validation links before any tuning of net2; they say nothing
about the test links.

### With the online learner fixed

Round 26 trained round 25's DAgger recipe unchanged on neto1 v2, the same
27 LCFs with one value each
([`exp-neto2-r26-v2-254`](https://github.com/Radonux/palaestra/releases/tag/exp-neto2-r26-v2-254)),
with the rule on neto1 v2 as teacher at round 25's u (0.25) or at the u
chosen for neto1 v2 (0.375).
[`exp-dagger-twins-val-255`](https://github.com/Radonux/palaestra/releases/tag/exp-dagger-twins-val-255)
ran both rounds' students after iteration 39 (fixed beforehand) in closed
loop, as in release 253 (campaign seed 21, no failed episodes). Each row
is the mean over three training seeds of log utility minus BBRv3:

| Arm | Teacher | Mean | Per seed | Cubic first | Deep buffer | Copies |
|---|---|---|---|---|---|---|
| net2 on net1 v9 means (r25) | rule on net1 v9, u = 0.25 | +0.143 | +0.157, +0.133, +0.139 | -0.038 | +0.259 | +0.043 |
| neto2 on neto1 v2 (r26) | rule on neto1 v2, u = 0.25 | +0.121 | +0.109, +0.132, +0.121 | +0.017 | +0.186 | -0.046 |
| neto2 on neto1 v2 (r26) | rule on neto1 v2, u = 0.375 | +0.095 | +0.114, +0.038, +0.134 | -0.026 | +0.201 | -0.115 |
| net2 on net1 v9 distributions (r25) | rule on net1 v9, u = 0.25 | +0.119 | +0.117, +0.123, +0.118 | -0.120 | +0.250 | -0.139 |
| rule on net1 v9, u = 0.25 (reference) | | +0.140 | | -0.057 | +0.270 | -0.224 |
| rule on neto1 v2, u = 0.375 (reference) | | +0.136 | | +0.032 | +0.262 | -0.749 |

- With the same recipe and u, net2 led neto2 by 0.022. In the deep
  buffer, new and both training families, every net2 seed scored above
  every neto2 seed; with Cubic first neto2 led.
- That lead matches the teachers' gap: at u = 0.25 the rule on neto1 v2
  scored about 0.024 below the rule on net1 v9 at all three campaign
  seeds of
  [`exp-rule-paired-val-246`](https://github.com/Radonux/palaestra/releases/tag/exp-rule-paired-val-246).
  Each student stayed near its teacher.
- With each estimator's own u, the two teachers scored alike (+0.140 and
  +0.136), the net2 students kept their teacher's score, and the neto2
  students fell to +0.095, below their teacher in every family but
  copies. One neto2 seed (+0.038), which had dipped during training at
  the fixed iteration, carries much of that.
- r25-means scored +0.143 here and +0.144 in release 253.

[`exp-dagger-twins-iter-val-256`](https://github.com/Radonux/palaestra/releases/tag/exp-dagger-twins-iter-val-256)
then ran every arm after iterations 19, 39 and 59 on the same validation
episodes and took, for each arm, the iteration with the best five-family
mean over its three training seeds (no failed episodes):

| Arm | Chosen iteration | Mean | Seed range | Other iterations |
|---|---|---|---|---|
| net2 on net1 v9 means (r25) | 19 | +0.149 | +0.119 to +0.165 | 39: +0.145, 59: +0.126 |
| neto2 on neto1 v2, u = 0.375 (r26) | 59 | +0.129 | +0.127 to +0.131 | 19: +0.115, 39: +0.100 |
| neto2 on neto1 v2, u = 0.25 (r26) | 39 | +0.120 | +0.111 to +0.132 | 19: +0.079, 59: +0.103 |

The rules scored +0.140 (net1 v9) and +0.137 (neto1 v2) in the same run.
With the iteration chosen, net2 led the better neto2 arm by 0.020, with
overlapping seed ranges. Counting copies as a sixth family would have
picked iteration 39 for net2 (iteration 19 scored -0.060 as copies); the
criterion was fixed before the run.

[`exp-dagger-twins-confirm-val-257`](https://github.com/Radonux/palaestra/releases/tag/exp-dagger-twins-confirm-val-257)
ran the chosen arms on fresh validation episodes at campaign seeds 11
and 31 (no failed episodes), choosing nothing. The order held at every
campaign seed:

| Arm | Seed 11 | Seed 21 (release 256) | Seed 31 | Mean | With copies, seeds 11 and 31 |
|---|---|---|---|---|---|
| net2 on net1 v9 means, iteration 19 | +0.148 | +0.149 | +0.153 | +0.150 | +0.115 |
| neto2 on neto1 v2, u = 0.375, iteration 59 | +0.127 | +0.129 | +0.132 | +0.129 | +0.101 |
| neto2 on neto1 v2, u = 0.25, iteration 39 | +0.121 | +0.120 | +0.126 | +0.122 | +0.101 |
| rule on net1 v9, u = 0.25 (reference) | +0.140 | +0.140 | +0.144 | +0.141 | +0.081 |
| rule on neto1 v2, u = 0.375 (reference) | +0.134 | +0.137 | +0.140 | +0.137 | -0.009 |

net2 led the better neto2 arm by 0.020 to 0.021 at each campaign seed.
Between training seeds the ranges still overlap (net2 +0.117 to +0.169,
neto2 at u = 0.375 +0.124 to +0.138): the variation that matters is
between trained policies, not between episode draws. Training seeds 4
to 6, at the same iterations and campaign seed 21, scored +0.135 (net2),
+0.058 (neto2, u = 0.25) and +0.101 (neto2, u = 0.375)
([`exp-dagger-twins-s456-val-271`](https://github.com/Radonux/palaestra/releases/tag/exp-dagger-twins-s456-val-271)).

### On the test links

[`exp-dagger-twins-test-258`](https://github.com/Radonux/palaestra/releases/tag/exp-dagger-twins-test-258)
ran the same arms once on the 27 test links (training seeds 1 to 3),
with nothing chosen there, and
[`exp-dagger-twins-s456-test-272`](https://github.com/Radonux/palaestra/releases/tag/exp-dagger-twins-s456-test-272)
ran training seeds 4 to 6 at the same iterations on the same links and
campaigns, so the per-seed scores pool. Neither had a failed episode.
The held-out competitors (a BBRv3 and a Westwood flow) and copies are
outside the mean; every column is the mean over six training seeds:

| Arm | Mean | Per training seed (1 to 3; 4 to 6) | Cubic first | Deep buffer | Held-out competitors | Copies |
|---|---|---|---|---|---|---|
| net2 on net1 v9 means, iteration 19 | +0.136 | +0.161, +0.159, +0.118; +0.125, +0.157, +0.094 | -0.120 | +0.265 | -0.017 | +0.177 |
| neto2 on neto1 v2, u = 0.25, iteration 39 | +0.088 | +0.094, +0.132, +0.106; +0.048, +0.042, +0.106 | -0.156 | +0.193 | -0.050 | +0.018 |
| neto2 on neto1 v2, u = 0.375, iteration 59 | +0.074 | +0.102, +0.083, +0.102; +0.043, +0.041, +0.076 | -0.181 | +0.221 | -0.072 | -0.005 |
| rule on net1 v9, u = 0.25 (reference) | +0.136 | | -0.101 | +0.271 | +0.093 | +0.054 |
| rule on neto1 v2, u = 0.375 (reference) | +0.107 | | -0.062 | +0.235 | +0.084 | -0.593 |

The rules are release 258's; release 272 reproduced them within 0.005.

- With the learner, its recipe, the data and the 27 LCFs held fixed, net2
  on the distributional net1 beat neto2 on the single-value net1 by 0.061
  (neto2's validation-preferred u) and by 0.048 (net2's own u), against
  0.021 and 0.028 on validation. A net2 seed scored above a neto2 seed in
  34 of 36 pairs at u = 0.375 and in 30 of 36 at u = 0.25 (one tie); an
  exact permutation test on the 12 seed means gives p = 0.009 and 0.043
  (two-sided).
- Seeds 4 to 6 scored lower than seeds 1 to 3 on both sides (net2 +0.125
  against +0.146, neto2 +0.065 and +0.053 against +0.111 and +0.096), and
  the gap grew (0.072 and 0.060 against 0.050 and 0.035). The rules
  scored the same in both runs, so the spread is between trained
  policies.
- On the six-seed mean net2 led in all five families, against the
  held-out competitors and as copies.
- net2 matched the rule on its own estimator (+0.136 each), while both
  neto2 arms fell below the rule on theirs (+0.107 at u = 0.375), by
  0.019 and 0.033. Against the held-out competitors
  every learned arm fell below both rules; for net2 mostly through
  training seed 6 (-0.290 there; +0.038 over the other five seeds).
- net2 read net1 v9's means; on validation the full distributions did no
  better, so the gain comes from the estimator, not from handing net2 its
  uncertainty. Part of it is the rule: at a shared u the rule on neto1 v2
  was the weaker teacher, and with its own u its students still fell
  short.
- This is one link family (8 to 100 Mbit/s, 5 to 50 ms, buffers of 0.25
  to 4 BDP) and six training seeds per arm.

### With five composite heads added

Rounds 27 and 28 ran the same recipe on estimators with five more heads
(departure rate, available capacity, overload, BDP and the other flows'
arrival rate: net1 v10 and neto1 v3, [training](training.md)), training
seeds 1 to 3. On validation
([`exp-dagger32-iter-val-273`](https://github.com/Radonux/palaestra/releases/tag/exp-dagger32-iter-val-273)) net2 chose iteration 59, which only
seeds 1 and 3 reached (seed 2 hit its time budget after 39), and neto2
iteration 19 at both u; net2 led the better neto2 arm by 0.011, and by
0.010 and 0.008 at campaign seeds 11 and 31
([`exp-dagger32-confirm-val-274`](https://github.com/Radonux/palaestra/releases/tag/exp-dagger32-confirm-val-274)), where the choice held. On the
test links
([`exp-dagger32-test-275`](https://github.com/Radonux/palaestra/releases/tag/exp-dagger32-test-275), the links and campaigns of release 258;
the rule on net1 v9 scored +0.135 there against +0.136):

| Arm | Mean | Per training seed | Cubic first | Deep buffer | Held-out competitors | Copies |
|---|---|---|---|---|---|---|
| net2 on net1 v10 means, iteration 59 | +0.146 | +0.153, +0.139 (seeds 1, 3) | -0.085 | +0.241 | +0.127 | +0.007 |
| net2 on net1 v10 means, iteration 39 (check) | +0.149 | +0.149, +0.145, +0.154 | -0.093 | +0.255 | +0.060 | -0.029 |
| neto2 on neto1 v3, u = 0.25, iteration 19 | +0.141 | +0.131, +0.122, +0.170 | +0.022 | +0.207 | +0.056 | -0.311 |
| neto2 on neto1 v3, u = 0.375, iteration 19 | +0.149 | +0.154, +0.151, +0.142 | +0.141 | +0.186 | +0.136 | -0.622 |
| rule on net1 v10, u = 0.25 (reference) | +0.135 | | -0.113 | +0.259 | +0.115 | +0.060 |
| rule on neto1 v3, u = 0.375 (reference) | +0.157 | | +0.124 | +0.218 | +0.154 | -0.893 |

- The twins tied on the five-family mean: net2 +0.146 against +0.141
  and +0.149. net2 led in the deep buffer, new and both training
  families (by 0.016 to 0.088); neto2 led with Cubic first (by 0.107 and
  0.226) and lost as copies.
- The composites changed neto2, not net2. Against the 27-head twins'
  same training seeds (release 258), net2 stayed at +0.146 while neto2
  rose by 0.030 and 0.053, most of it with Cubic first (-0.061 to
  +0.022, -0.112 to +0.141). Outside the mean it rose against the
  held-out competitors and fell as copies (-0.052 to -0.311, -0.092 to
  -0.622). The rule on the single-value estimator moved the same way
  (+0.107 to +0.157 on the mean, -0.593 to -0.893 as copies), so the
  change is in neto1 v3, and its students inherit it.
- Neither 32-head twin reads the composites themselves (next section),
  so the change comes from what training on five more heads did to
  neto1's other estimates (an inference).
- Whether net2 beats neto2 therefore depends on the pair of estimators:
  with 27 heads net2 led by 0.048 and 0.061 over six training seeds; with
  32 heads the two traded Cubic first against copies and tied on the
  mean, over three training seeds (two at net2's chosen iteration).

### What the learned step reads

[`exp-occlusion-twins-val-276`](https://github.com/Radonux/palaestra/releases/tag/exp-occlusion-twins-val-276)
replayed each twin's chosen policies (every training seed, 27 and 32
heads) over their own steps on the validation training
family, 24 episodes and about 11,000 to 13,000 decisions each, with one
group of LCFs at a time held at its mean over those steps
(experiments/occlusion.py). The replay set the same windows as the guest
(median gap 0.002 to 0.005 in log2).

- A twin's window is u plus the log2 of the BDP net1 believes in
  (capacity times one-way delay). Holding expliCCit's seven LCFs at their
  mean moved the window by 0.87 to 1.39 in log2, about a factor of 2,
  through that target, but moved u itself by only 0.007 to 0.092.
- u stays near its teacher's constant: 0.25 to 0.29 for net2 (rule at
  0.25), 0.21 to 0.27 and 0.30 to 0.40 for neto2 (rules at 0.25 and
  0.375). Holding every LCF at its mean moved u by 0.02 to 0.05 for net2
  and 0.01 to 0.11 for neto2; what moves it most is capacity, receive
  rate, buffer fill and active flows.
- The five composite heads of the 32-head estimators moved u by at most
  0.006 for net2 on net1 v10 and 0.042 for neto2 on neto1 v3: neither
  32-head twin leans on them.
- The decision therefore rests on net1's estimate of the BDP, with a
  small learned correction, which suggests that the gap between the
  twins comes mostly from the estimator rather than from what net2
  learned on top of it (an inference from these numbers, not a separate
  test).

## Stated uncertainty has not helped net2 yet

On the validation links, net2 did no better on net1's full distributions
than on their means (minus BBRv3, mean over three training seeds, five
families). Offline (TD3+BC,
[`exp-net2-vs-neto2-val-253`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-vs-neto2-val-253))
the result is mixed: on net1 v9 the distributions pooled into 4 bins
scored -0.003 and -0.005 against +0.106 and +0.118 on the means, and on
net1 v7 +0.088 and +0.092 against -0.045 and -0.078 (alpha 5 and 10 in
each pair). Online (DAgger,
[`exp-dagger-twins-val-255`](https://github.com/Radonux/palaestra/releases/tag/exp-dagger-twins-val-255)),
the student on net1 v9's distributions scored +0.119 against +0.143 on
its means (+0.121 against +0.144 in release 253).

## Reproduction of expliCCit

Three of expliCCit's results were checked here, and none reproduced as
stated. Everything in this section but neto1 v2 ran on the old guest.

| expliCCit's result | Here | Evidence |
|---|---|---|
| net1 estimates every LCF with R^2 above 0.9 | Only with its random row split, which puts a training row of the same episode 10 ms from each test row, and not for free buffer even then (0.65). Split by link, expliCCit's encoder retrained here (neto1 v2) has 0.78 for capacity, 0.86 for busy fraction and 0.23 for free buffer on the test links | [`exp-net1-row-split-97`](https://github.com/Radonux/palaestra/releases/tag/exp-net1-row-split-97), [`exp-neto1-v2-239`](https://github.com/Radonux/palaestra/releases/tag/exp-neto1-v2-239) |
| The split costs about 5% utilization and 9% queueing delay against Sage | The other way for our split (above). expliCCit's released policy, as deployed, has 41% less utilization than Sage on clean test links and 24% to 25% less on clean validation links | `exp-final-test-100`, [`exp-sage-v5-val-94`](https://github.com/Radonux/palaestra/releases/tag/exp-sage-v5-val-94) |
| On a 24/48 Mbit/s link (40 ms one way, 1000-packet buffer), under-utilization traces to capacity underestimated by 25%, and the true capacity restores it | The released policy uses 0.48 of the link (0.34 to 0.53 on the eight links around it), and its capacity belief is within 14%; the true capacity gives 0.43. Its wrong beliefs are busy fraction (1.00 against 0.51) and queue (19 packets against 0); the true busy fraction gives 0.71 on all nine links. The deployed rounding of beliefs is not the cause | [`exp-expliccit-case-42`](https://github.com/Radonux/palaestra/releases/tag/exp-expliccit-case-42), [`exp-expliccit-case-91`](https://github.com/Radonux/palaestra/releases/tag/exp-expliccit-case-91) |

Divergences from expliCCit's and Sage's setting, each a possible cause:
Linux 7.0 in UML in virtual time, not Sage's patched Linux 4.19 in real
time; inference that costs no virtual time; links the policies were not
trained on; and ground-truth definitions that may differ from
expliCCit's, which are not in its code (on our expliCCit-like data its
released net1 had R^2 1.000 for one-way delay, 0.361 for capacity, -0.743
for busy fraction and -59.0 for free buffer; local run). Pacing is not
the cause: run as `agent_pure`, which paces every ACK as Sage's kernel
does ([validation.md](validation.md)), either policy's share moves by at
most 0.07 per scenario against `agent`, which paces once a step
([`exp-baselines-pure-194`](https://github.com/Radonux/palaestra/releases/tag/exp-baselines-pure-194)
against `exp-baselines-192`). Nor are the policy computation or the
emulator: expliCCit's own deployed code takes our
action at 11,972 of 11,973 recorded steps (local run), the link model
equals a port of Mahimahi's link queue event for event, and the state
message equals `sage.cc`'s at every compared step
([validation.md](validation.md)). The run of the same four links on the
original stack, which would separate the emulator from the rest, is
UNRUN: the kit is `tools/crosscheck`, and palaestra's side is
[`exp-crosscheck-93`](https://github.com/Radonux/palaestra/releases/tag/exp-crosscheck-93)
(expliCCit's released policy uses 0.50 of the case-study link there,
Sage 0.92).

Exact Shapley values over the released policy's seven LCFs
([`exp-expliccit-attribution-190`](https://github.com/Radonux/palaestra/releases/tag/exp-expliccit-attribution-190))
give busy fraction exactly 0 on the case-study links, because its
deployed belief is constant there, yet it is the wrong belief on that
link (1.00 against 0.51, above). Attribution against a mean reference says how a
decision varies with an input, not whether the input is right; replacing
the belief with the truth found this fault.

## Baselines

On the old guest, each controller ran alone on the 27 test links of every
held-out family and as 2 and 4 copies of itself, 10,080 episodes with
none failed
([`exp-baselines-192`](https://github.com/Radonux/palaestra/releases/tag/exp-baselines-192),
[`-pure-194`](https://github.com/Radonux/palaestra/releases/tag/exp-baselines-pure-194),
[`-orca-197`](https://github.com/Radonux/palaestra/releases/tag/exp-baselines-orca-197),
[`-aurora-204`](https://github.com/Radonux/palaestra/releases/tag/exp-baselines-aurora-204);
every table in the report,
[`exp-baselines-report-205`](https://github.com/Radonux/palaestra/releases/tag/exp-baselines-report-205)):

- expliCCit's released policy keeps the queue short and leaves capacity
  unused (0.25 to 0.73 of the link alone, 0.72 as two copies), and its
  share holds under 1% to 3% random loss, where Cubic and Vegas fall
  below 0.16.
- Sage on clean links is close to Vegas (0.91 and 0.95 of the link, both
  at 10 ms), and against competitors gets 0.26 to 0.33 of the link.
- Orca, checked against its own code
  ([`exp-orca-model-196`](https://github.com/Radonux/palaestra/releases/tag/exp-orca-model-196)),
  rescales Cubic's window and cuts its queue (12 against 34 ms on clean
  links), and against Cubic gets less than Cubic and more than Sage.
- BBRv1 has the most throughput under random data loss (0.1% to 3%) and
  takes about two thirds of the link against Cubic.
- Among copies, BBRv1, BBRv3, Cubic and Orca are the fairest (Jain's
  index 0.94 to 0.99) and Vegas the least fair (0.84 to 0.87).
- Aurora, which has no released model, was trained three times with
  PCC-RL's own code
  ([`exp-aurora-train-199`](https://github.com/Radonux/palaestra/releases/tag/exp-aurora-train-199),
  [`-201`](https://github.com/Radonux/palaestra/releases/tag/exp-aurora-train-201),
  [`-202`](https://github.com/Radonux/palaestra/releases/tag/exp-aurora-train-202);
  exports checked in
  [`exp-aurora-model-203`](https://github.com/Radonux/palaestra/releases/tag/exp-aurora-model-203)).
  It leaves capacity unused (0.57 of clean links), mostly because
  PCC-RL's receive rate counts one packet too few: counting every packet,
  it uses 0.88. PCC-Uspace's deployment reads every RTT under 1 s as 0,
  which raises its delay from 9 to 31 ms on clean links and from 33 to
  337 ms in deep buffers. On links drawn as PCC-RL draws its training
  links it uses more of the link than any other controller, but by its
  own reward BBRv1 and BBRv3 beat it (1727 and 1256 against 929).

## Alarms and repair

On the old guest, alarms computed without ground truth caught the wrong
beliefs that the flow's own measurements contradict and missed the ones
the measurements are consistent with, and a capacity fault was repaired
by retraining net1 alone, with net2 unchanged.

- Each alarm is a statistic of one step from net1's beliefs and the
  sender's own measurements (physical invariants among the beliefs, each
  belief's entropy, and checks against the delivery rate and min RTT
  measured later; `experiments/r2_alarms.py`), with thresholds that 8% to
  17% of right episodes set off per LCF. They caught 97% to 100% of
  injected faults in queueing delay, queue, capacity, one-way delay,
  receive rate and busy fraction halved
  ([`exp-r2-faults-210`](https://github.com/Radonux/palaestra/releases/tag/exp-r2-faults-210)),
  whether or not the policy used the belief, and 36 of 37 wrong capacity
  beliefs on real cellular traces (`exp-net1v6-cellular-207`).
- They miss what the measurements cannot contradict: an overestimated
  busy fraction (0 of 58 on the cellular traces) and a low capacity
  under competition (0.12 of 43;
  [`exp-r2-alarms-81`](https://github.com/Radonux/palaestra/releases/tag/exp-r2-alarms-81)).
- They name a group of LCFs; only one-way delay has an alarm of its own.
- expliCCit's released encoder is wrong in 41% of its (episode, LCF)
  pairs on the clean test links (net1 v6 in 1.1%), so thresholds set on
  its normal running catch only 13% to 29% of its errors, and not its
  Case Study 1 fault
  ([`exp-r2-expliccit-211`](https://github.com/Radonux/palaestra/releases/tag/exp-r2-expliccit-211)).
- net1 v3's low capacity belief with Cubic from the start was repaired by
  retraining net1 on broader data, net2 unchanged: the fixed rule on
  net1 v4 beat the same rule on net1 v3 by 0.428 (0.277 to 0.580) on the
  test links with Cubic from the start
  ([`exp-rule-net1v4-test-14`](https://github.com/Radonux/palaestra/releases/tag/exp-rule-net1v4-test-14)).
  Retraining only the capacity head recovered +0.16 of the +0.52 in
  closed loop
  ([`exp-r3-capacity-head-83`](https://github.com/Radonux/palaestra/releases/tag/exp-r3-capacity-head-83)).
- A patch with the truth can hurt, because net2 adapts to its net1's
  errors: given the true capacity, the learned net2 on net1 v6 collapsed
  as copies (-1.247 against -0.008), since net1's lower belief on a
  shared link had kept the copies safe
  ([`exp-net2-patch-v6-172`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-patch-v6-172)).
