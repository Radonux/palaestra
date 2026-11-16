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
same over six training seeds ([Learned net2](#learned-net2)).

net1's estimation errors cost about 0.03 on one flow or beside a stock
competitor
([`exp-truth-patch-val-282`](https://github.com/Radonux/palaestra/releases/tag/exp-truth-patch-val-282),
validation links at campaign seed 21). Fed expliCCit's seven LCFs from
the live truth in place of net1's, with nothing retrained (arm A1
against A3), the learned net2 rose from +0.157 to +0.184 over BBRv3 on
the seven LCFs (three training seeds) and from +0.157 to +0.186 on the
32 (two seeds; the other 25 still from net1), and the fixed rule from
+0.138 (net1 v7) and +0.141 (net1 v10) to +0.172.

- Paired by link, every pair (each net2 seed, and the rule on each
  net1) gained 0.01 to 0.06 in the training and new families, with
  intervals excluding zero in 17 of those 21 cells, and 0.05 to 0.10
  with Cubic first, excluding zero in 1 of 7.
- One LCF carries it. Capacity alone from the truth gained the
  seven-LCF net2 0.023 (+0.164 to +0.187, three training seeds), one-way
  delay alone -0.002 and the other five together 0.005
  ([`exp-truth-patch-lcf-val-283`](https://github.com/Radonux/palaestra/releases/tag/exp-truth-patch-lcf-val-283)).
- In deep buffers the truth was level or up to 0.04 worse, all of it
  beside the Cubic flow: there net1's larger believed BDP had kept a
  larger window (1.34 against 1.21 times the true BDP, learned net2 on
  net1 v7) and a larger share (0.52 against 0.41).
- On clean links the cost is drops while a stepping link is at its
  lower level. After the first 3 s, capacity from the truth gained 0.106
  there (loss term 0.053 against 0.008; median window 1.29 against 1.19
  times the true BDP), 0.014 at the higher level and 0.014 on constant
  links, and as much more than 1 s after a change as within it. So net1's
  costly error is a capacity belief that stays too high at the lower
  level, not a lag after each step (net2 v7, release 283).
- Offline, on links of its training set that net1 v7 never trained on,
  its capacity mean at the lower level of a stepping clean link was 1.08
  times the truth (its median 1.05) and above 1.1 times in 45% of rows,
  as often 5 to 10 s after the step down as within 0.5 s, and as often
  while the link was busy. Alone on the link (clean or with random loss),
  the overestimate grows with net1's stated spread, from +0.04 in log2
  below a spread of 0.15 to +0.27 at 0.4 to 0.5; beside a TCP flow it
  stays within 0.02 up to a spread of 0.7. A wide belief means "too high"
  only when the flow is alone
  ([`exp-capacity-belief-v7-286`](https://github.com/Radonux/palaestra/releases/tag/exp-capacity-belief-v7-286)).
- The learned step's lead over the rule was about the same on the truth
  (+0.012 to +0.014) as on net1 (+0.016 to +0.019), so it does not make
  up for net1's errors.
- Among copies the truth is far worse; see
  [What limits a rule is the other flows](#what-limits-a-rule-is-the-other-flows).

On the old guest, a black box with the same learner and budget on the
raw measurements stayed 0.2 to 0.4 below net1 v6's fixed rule from its
own start
([`exp-net2-r19b-bbr-165`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-r19b-bbr-165))
and ended about 0.05 below the same learner on net1 v6 when started from
a clone of the rule
([`exp-net2-r20b-bb-ppo-171`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-r20b-bb-ppo-171)),
so the split's lead was its start (old guest).

On ccBench's suites, run unchanged on links outside the train,
validation and test sets (690 single-flow links of 12 to 192 Mbit/s, 5
to 80 ms and buffers of 0.5 to 16 BDP, constant or stepping 2 or 4 times
up or down; 125 runs beside a Cubic flow that starts 5 s earlier), and
on three suites added to them
([`exp-ccbench-net2-281`](https://github.com/Radonux/palaestra/releases/tag/exp-ccbench-net2-281)),
the learned net2 (both arms below, means over training seeds) kept the
link full with a short queue:

| | Alone: utilization | Alone: queueing delay | 0.1% to 3% random loss: utilization | Share beside a Cubic flow that started first | Share beside BBRv1 or BBRv3 | Jain's index of 2, 4 or 8 copies |
|---|---|---|---|---|---|---|
| net2 on seven LCFs (three seeds) | 0.960 | 15.1 ms | 0.955 | 0.53 | 0.58 | 0.93 |
| net2 on 32 LCFs (two seeds) | 0.960 | 12.4 ms | 0.956 | 0.50 | 0.51 | 0.89 |
| BBRv3 | 0.922 | 42.2 ms | 0.682 | 0.39 | 0.38 | 0.99 |
| BBRv1 | 0.936 | 35.3 ms | 0.926 | 0.50 | 0.54 | 0.97 |
| Cubic | 0.946 | 153.9 ms | 0.214 | 0.37 | 0.44 | 0.93 |
| Sage | 0.740 | 21.4 ms | 0.458 | 0.21 | 0.14 | 0.90 |
| rule on net1 v10 (reference) | 0.961 | 12.5 ms | 0.954 | 0.55 | 0.54 | 0.94 |

On the old guest, the rule on net1 v5 had taken 0.40 beside the earlier
Cubic flow
([`exp-ccbench-merged-92`](https://github.com/Radonux/palaestra/releases/tag/exp-ccbench-merged-92))
and, on ccperf's experiments, queued more than BBR on traces whose
capacity jumps
([`exp-ccperf-v5-87`](https://github.com/Radonux/palaestra/releases/tag/exp-ccperf-v5-87)).

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
net1 v9 +0.177 over six training seeds (releases 258 and 272).

On the true LCFs, copies collapse. As copies on the validation links,
the learned net2 fed expliCCit's seven LCFs from the truth scored -1.36
(seven LCFs) and -0.73 (32 LCFs) minus BBRv3, against -0.08 and -0.04
on net1, and the fixed rule -1.21 against -0.04 on net1 v7
([`exp-truth-patch-val-282`](https://github.com/Radonux/palaestra/releases/tag/exp-truth-patch-val-282)). Capacity
alone from the truth collapses them as much (-1.34 against -0.06), and
one-way delay or the other five not at all
([`exp-truth-patch-lcf-val-283`](https://github.com/Radonux/palaestra/releases/tag/exp-truth-patch-lcf-val-283)).
On the truth each copy of the rule and of the seven-LCF net2 kept 1.2
to 1.5 times the whole link's BDP in flight, and the loss term rose 2.6
to 3.9 times. net1 reads the flow's own deliveries, and on a shared link
its believed BDP fell (for the rule on net1 v7, to about 0.80 of the
true BDP beside one copy and 0.68 beside three).
The window should follow the capacity left to the flow, not the link's,
so on shared links the true LCFs are no upper bound. Told the number of
flows, a rule on the truth handled both copies and Cubic (an oracle
bound, old guest,
[`exp-fairshare-used-val-23`](https://github.com/Radonux/palaestra/releases/tag/exp-fairshare-used-val-23)).
The raw measurements tell copies from a stock TCP flow better than net1
v6's beliefs (balanced accuracy 0.753 against 0.496, chance 0.333;
[`exp-net2-competitor-info-177`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-competitor-info-177)),
and a traffic head on net1 learns it (0.909;
[`exp-net1-traffic-head-val-180`](https://github.com/Radonux/palaestra/releases/tag/exp-net1-traffic-head-val-180)).

## Learned net2

Every score here is log utility minus BBRv3's over the five families
(the mean over training seeds), with held-out competitors and copies
apart. Scores compare only within one release: on the same links BBRv3's
own score on a family moved by up to 0.037 between releases.

The learned net2 is a recurrent gate student trained online by DAgger
([training.md](training.md#learned-net2-train-ppo)). Its u mixes the
fixed rule's (u = 0.25), the share rule's and the cooperative window's
by its learned beliefs that a stock TCP flow, or a copy of itself,
shares the link. Its teacher labels every decision with the share rule
while a stock flow is active, the cooperative window while a copy is,
and the fixed rule otherwise. It reads net1's means.

### Choosing the learner

[`exp-net2-vs-neto2-val-253`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-vs-neto2-val-253)
ran offline TD3+BC net2 arms and round 25's DAgger student (iteration
39) in closed loop on the validation links (campaign seed 21, no failed
episodes). net1 v9 has 27 LCFs, net1 v7 expliCCit's seven; bins4 is each
distribution pooled into 4 bins:

| Arm | Learner | Mean | Seed range | Cubic first | Deep buffer | Copies |
|---|---|---|---|---|---|---|
| net1 v9 means (r25) | DAgger | +0.144 | +0.135 to +0.158 | -0.058 | +0.247 | +0.044 |
| net1 v9 distributions (r25) | DAgger | +0.121 | +0.120 to +0.122 | -0.135 | +0.237 | -0.153 |
| net1 v9 means, alpha 10 | TD3+BC | +0.118 | +0.098 to +0.146 | +0.225 | +0.002 | -0.460 |
| net1 v9 means, alpha 5 | TD3+BC | +0.106 | +0.041 to +0.142 | +0.178 | -0.017 | -0.535 |
| net1 v7 bins4, alpha 10 | TD3+BC | +0.092 | +0.018 to +0.133 | +0.224 | -0.170 | -0.689 |
| net1 v7 bins4, alpha 5 | TD3+BC | +0.088 | +0.050 to +0.136 | +0.223 | -0.184 | -0.636 |
| net1 v9 bins4, alpha 5 | TD3+BC | -0.003 | -0.076 to +0.076 | -0.045 | -0.271 | -0.226 |
| net1 v9 bins4, alpha 10 | TD3+BC | -0.005 | -0.038 to +0.018 | -0.047 | -0.308 | -0.162 |
| net1 v7 means, alpha 5 | TD3+BC | -0.045 | -0.083 to +0.002 | -0.313 | -0.086 | +0.155 |
| net1 v7 means, alpha 10 | TD3+BC | -0.078 | -0.101 to -0.066 | -0.364 | -0.193 | +0.181 |
| rule on net1 v9, u = 0.25 (reference) | none | +0.142 | | -0.077 | +0.259 | -0.227 |

- Online DAgger on the means was the only learned arm level with the
  rule over the five families, and the only one ahead of it as copies.
- The offline arms traded rather than gained: they won where Cubic
  starts first and lost in deep buffers and as copies, the shape of a
  more aggressive policy.
- Offline, the seven LCFs against 27 gave no consistent order: worse
  read as means, better read as 4 bins.

### Choosing the iteration

[`exp-dagger-twins-iter-val-256`](https://github.com/Radonux/palaestra/releases/tag/exp-dagger-twins-iter-val-256)
ran round 25 (net1 v9) after iterations 19, 39 and 59 and took the
iteration with the best mean over its three training seeds: 19 (+0.149,
seeds +0.119 to +0.165), against +0.145 at 39 and +0.126 at 59; the rule
scored +0.140. Counting copies as a sixth family would have picked 39
(19 scored -0.060 as copies); the criterion was fixed before the run.
[`exp-dagger-twins-confirm-val-257`](https://github.com/Radonux/palaestra/releases/tag/exp-dagger-twins-confirm-val-257)
held the choice at campaign seeds 11 and 31 (+0.148 and +0.153, rule
+0.140 and +0.144). Training seeds 4 to 6 at the same iteration scored
+0.135
([`exp-dagger-twins-s456-val-271`](https://github.com/Radonux/palaestra/releases/tag/exp-dagger-twins-s456-val-271)).

Round 27 ran the same recipe on net1 v10, which adds five composite
heads (departure rate, available capacity, overload, BDP and the other
flows' arrival rate, [training](training.md#model-names)).
[`exp-dagger32-iter-val-273`](https://github.com/Radonux/palaestra/releases/tag/exp-dagger32-iter-val-273)
chose iteration 59 (+0.160 over seeds 1 and 3; seed 2 hit its time
budget after iteration 39), against +0.149 at 39 over all three seeds
and +0.127 at 19; the rule on net1 v10 scored +0.143.
[`exp-dagger32-confirm-val-274`](https://github.com/Radonux/palaestra/releases/tag/exp-dagger32-confirm-val-274)
held it at campaign seeds 11 and 31 (+0.167 and +0.152, rule +0.150 and
+0.136).

### On the test links

Each arm ran once on the test links, with nothing chosen there:
[`exp-dagger-twins-test-258`](https://github.com/Radonux/palaestra/releases/tag/exp-dagger-twins-test-258)
(27 LCFs, training seeds 1 to 3),
[`exp-dagger-twins-s456-test-272`](https://github.com/Radonux/palaestra/releases/tag/exp-dagger-twins-s456-test-272)
(seeds 4 to 6 on the same links and campaigns, so the seeds pool) and
[`exp-dagger32-test-275`](https://github.com/Radonux/palaestra/releases/tag/exp-dagger32-test-275)
(32 LCFs, the same links and campaigns). None had a failed episode.

| Arm | Mean | Per training seed | Cubic first | Deep buffer | Held-out competitors | Copies |
|---|---|---|---|---|---|---|
| net1 v10 means, iteration 59 | +0.146 | +0.153, +0.139 (seeds 1, 3) | -0.085 | +0.241 | +0.127 | +0.007 |
| net1 v10 means, iteration 39 (check) | +0.149 | +0.149, +0.145, +0.154 | -0.093 | +0.255 | +0.060 | -0.029 |
| net1 v9 means, iteration 19 | +0.136 | +0.161, +0.159, +0.118; +0.125, +0.157, +0.094 | -0.120 | +0.265 | -0.017 | +0.177 |
| rule on net1 v10, u = 0.25 (reference) | +0.135 | | -0.113 | +0.259 | +0.115 | +0.060 |
| rule on net1 v9, u = 0.25 (reference) | +0.136 | | -0.101 | +0.271 | +0.093 | +0.054 |

The net1 v9 rows are releases 258 and 272 (six training seeds; the rule
reproduced within 0.005 between them); the rule on net1 v9 scored +0.135
in release 275.

- The learned net2 matched the rule it learned from: +0.136 against
  +0.136 on 27 LCFs, and +0.146 against +0.135 on 32, a lead inside its
  seed range (+0.139 to +0.154).
- Adding the five composite heads left it where it was: on training
  seeds 1 to 3 net2 on net1 v9 scored +0.146 too.
- Outside the mean the arms differ by estimator rather than in one
  direction: on 27 LCFs net2 led the rule as copies (+0.177 against
  +0.054) and trailed it against held-out competitors (-0.017 against
  +0.093, mostly training seed 6 at -0.290; +0.038 over the other five);
  on 32 LCFs at iteration 59 the reverse, and at iteration 39 it trailed
  in both.
- Seeds 4 to 6 scored lower than seeds 1 to 3 (+0.125 against +0.146)
  while the rule did not move, so the spread is between trained
  policies.

### Seven LCFs against 32

Round 29 ran the same recipe on net1 v7, which has net1 v10's data and
training spec but only expliCCit's seven hand-picked heads
([`exp-net2-r29-v7-278`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-r29-v7-278)).
On validation
([`exp-dagger7-iter-val-279`](https://github.com/Radonux/palaestra/releases/tag/exp-dagger7-iter-val-279),
campaign seed 21) it chose iteration 39 (+0.152 over its three seeds,
against +0.147 at 19 and +0.134 at 59), level with round 27 at its
chosen iteration (+0.150 in the same run).
[`exp-dagger7-test-280`](https://github.com/Radonux/palaestra/releases/tag/exp-dagger7-test-280)
then ran both once on the test links and campaigns of releases 258 and
275, with no failed episodes. The criterion, stated before the run: the
two count as comparable if their means differ by less than round 27's
seed range, about 0.015.

| Arm | Mean | Per training seed | Cubic first | Deep buffer | Held-out competitors | Copies |
|---|---|---|---|---|---|---|
| net1 v7 (seven LCFs) means, iteration 39 | +0.151 | +0.151, +0.156, +0.145 | -0.027 | +0.261 | +0.096 | +0.114 |
| net1 v10 (32 LCFs) means, iteration 59 | +0.148 | +0.155, +0.140 (seeds 1, 3) | -0.083 | +0.242 | +0.111 | +0.025 |
| net1 v10 (32 LCFs) means, iteration 39 (check) | +0.149 | +0.147, +0.147, +0.154 | -0.086 | +0.246 | +0.041 | +0.038 |
| rule on net1 v7, u = 0.25 (reference) | +0.131 | | -0.096 | +0.262 | +0.076 | +0.093 |
| rule on net1 v10, u = 0.25 (reference) | +0.135 | | -0.110 | +0.255 | +0.093 | +0.102 |

- Comparable: the net2 on the seven LCFs scored within 0.003 of the one
  on 32 (+0.151 against +0.148 and +0.149), inside the 32-LCF arm's seed
  range. Round 27 reproduced its release 275 scores within 0.002.
- The two traded families. The seven-LCF arm did better with Cubic first
  on every seed (-0.003 to -0.051, against -0.044 to -0.123) and stayed
  positive as copies on all three (+0.071 to +0.141, against -0.084 to
  +0.133); the 32-LCF arm did better in the new and both training
  families, by 0.015 to 0.023. Each policy's copies interval spans 0.4 to
  0.5, so the copies gap is not established.
- The 25 heads that net1 v10 adds, chosen from data, gave the learned
  net2 nothing on the mean: expliCCit's seven already hold what this
  learner uses on these links, as the occlusion below suggests (the
  window target reads capacity and one-way delay, both among the
  seven).
- On ccBench's wider links ([above](#the-split-costs-little)) the two
  also differ: the seven-LCF arm queued more alone (15.1 against 12.4 ms
  at the same utilization) and took more beside BBR (0.58 against 0.51),
  and its copies shared more evenly (Jain's index 0.93 against 0.89).
- The test result is one link family, one learner and recipe, and three
  training seeds a side.

### What the learned step reads

[`exp-occlusion-twins-val-276`](https://github.com/Radonux/palaestra/releases/tag/exp-occlusion-twins-val-276)
replayed each chosen net2 (every training seed, 27 and 32 LCFs) over its
own steps on the validation training family, 24 episodes and about
10,800 to 12,700 decisions each, with one group of LCFs at a time held
at its mean over those steps (experiments/occlusion.py). The replay set
the same windows as the guest (median gap 0.002 to 0.004 in log2).

- The window is u plus the log2 of the BDP net1 believes in (capacity
  times one-way delay). Holding expliCCit's seven LCFs at their mean
  moved the window by 0.89 to 1.33 in log2, about a factor of 2, through
  that target, but moved u by only 0.016 to 0.027.
- u stays near the fixed rule's 0.25 (mean 0.25 to 0.29). Holding every
  LCF at its mean moved it by 0.023 to 0.047; the single LCFs that move
  it most are capacity (0.021 to 0.032), receive rate (0.016 to 0.030),
  buffer fill (up to 0.070) and active flows (up to 0.028). The last two
  are not among expliCCit's seven.
- The five composite heads moved u by at most 0.006.
- The decision therefore rests on net1's estimate of the BDP, with a
  small learned correction.

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
| The split costs about 5% utilization and 9% queueing delay against Sage | The other way for our split (above). expliCCit's released policy, as deployed, has 41% less utilization than Sage on clean test links and 24% to 25% less on clean validation links | [`exp-final-test-100`](https://github.com/Radonux/palaestra/releases/tag/exp-final-test-100), [`exp-sage-v5-val-94`](https://github.com/Radonux/palaestra/releases/tag/exp-sage-v5-val-94) |
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
- A patch with the truth can hurt: net1's lower belief on a shared link
  keeps copies safe, so copies fed the true LCFs collapse, the fixed rule
  as well as the learned net2
  ([What limits a rule is the other flows](#what-limits-a-rule-is-the-other-flows)).
