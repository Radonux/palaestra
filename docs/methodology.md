# Methodology: testing the LCF idea from first principles

The idea under test: split a learned congestion controller into an
estimator (net1) that turns standard sender-side measurements into a few
latent congestion factors (LCFs) with a ground truth, and a decision maker
(net2) that chooses the window change from those LCFs alone. This page says
how to test that idea, independent of how expliCCit or Sage implemented
it, and what the environment does for each part. `docs/training.md` is the
workflow; this is the reasoning.

## 1. The claims, and how each can fail

| # | Claim | How it fails |
|---|---|---|
| H1 | Sufficiency: a small set C of oracle-definable state functions supports near-best decisions | C omits something the decision depends on |
| H2 | Estimability: C can be recovered from deployable sender signals under the controller's own behaviour | some quantities are unidentifiable at some operating points (section 2) |
| H3 | Interface: what net1 hands net2 is enough | a point estimate loses what the sender does not know |
| H4 | Faithful and checkable, without leakage | a learned concept layer can pass on more than its concepts |
| H5 | Repairable: a fault in one LCF can be found and fixed without touching net2 | errors show only in combination |
| H6 | Affordable: small cost against a black box with the same inputs, data and training | |

## 2. Identifiability forces probing

A sender learns the link's capacity only while it fills the link. Below
capacity with an empty queue, the delivery rate equals the sending rate,
so the measurements bound capacity from below and say nothing more. Minimum
RTT has the opposite limit: it cannot be measured while a queue persists.
No operating point reveals every LCF, so a controller has to probe, and
probing costs delay and loss.

Two structural consequences:

- **net1 must report what the observations support, not a point.** For
  each LCF it outputs a central estimate and a calibrated interval: a
  lower and an upper quantile whose coverage is checked on held-out paths
  (conformalized quantile regression). When the data only bound a quantity
  from one side, the interval says so, by being wide on the other side.
- **net2 must be trained in closed loop, on those intervals, with a reward
  that prices probing.** A policy cloned or trained offline from other
  controllers' logs sees only the states those controllers visited; it
  cannot learn when the value of information is worth the delay and loss
  of probing. The reward weighs unused capacity against the delay and
  loss probing causes, so the trade-off is learnable.

The data show the problem already: trained on BBRv1, Cubic and Vegas logs,
net1 estimates capacity, queue and delay well (R^2 0.95 to 0.99) but free
buffer (0.09) and busy fraction (0.45) poorly. Those are exactly the
quantities those teachers rarely expose (`docs/training.md`). The
interval net1 shows the limit directly: its capacity interval is 3.6
Mbit/s wide on steps where the link is kept busy and 40.3 Mbit/s where it
is not.

## 3. Experiments

**Candidates from the path model, not intuition.** A bottleneck is its
parameters (capacity process C(t), propagation delay, buffer B, random
loss, cross-traffic arrivals) and its state (queue, each flow's share,
the reverse path). Candidate LCFs are that basis, quantities derived for
decisions (bandwidth-delay product, headroom, available bandwidth, time to
overflow), descriptors of the capacity process, and each one's
uncertainty. Several of expliCCit's nine are functions of the basis, so
they are redundant: useful for physical-invariant checks, misleading in a
leave-one-out ablation.

**Arms that isolate each cost** (same paths, seeds and training budget):

| Arm | net2 sees | Isolates |
|---|---|---|
| A0 | the full true state | upper bound |
| A1 | true C | A0 - A1: what C omits |
| A2 | net1's point estimates | A1 - A2: estimation error |
| A3 | net1's intervals | A3 - A2: what uncertainty is worth |
| A4 | raw signals (black box, same training) | A4 - A2: the price of the bottleneck |
| A5 | raw signals and C | whether C adds anything |

Leakage check: A2 beating A1 anywhere, or raw signals decodable from
net1's output beyond what true C explains.

**Training regimes compared:** offline from teacher logs (behaviour
cloning, TD3+BC, expliCCit's objective) against online in closed loop,
both from the same net1, on the same evaluation set.

**Identifiability pairs:** paths the sender cannot tell apart that differ
in one LCF, for example a 48 Mbit/s link carrying 24 Mbit/s of
constant-rate UDP against a 24 Mbit/s link alone. They show which
candidates can be LCFs at all.

**Repair:** inject bias, noise or lag into one LCF at runtime; measure the
closed-loop damage, its detection, and single-head retraining.

**Statistics:** the unit is a path and seed, never a step; arms are paired
on the same paths and seeds; at least five training seeds per arm;
bootstrap intervals clustered by path; Shapley over groups of redundant
LCFs; every result stated for its reward and scenario distribution.

## 4. Online training

The environment runs closed-loop episodes at 1,868 twenty-second episodes
per core-hour (about 1,000 decisions per second per core, measured), so
online training is affordable: Sage's choice of offline training came
from emulating in real time, which this environment does not.

The loop (`expliccit train online`):

1. Export the current policy graph (frozen net1, current net2).
2. Run a batch of episodes over the training paths, the guest adding seeded
   Gaussian exploration noise to each action and recording the action it
   applied.
3. Turn the steps into transitions with the reward computed from ground
   truth, append them to a replay buffer, and update net2 off-policy
   (TD3+BC, with the imitation weight a parameter).
4. Every few iterations, evaluate the deterministic policy on the
   evaluation paths.

The reward is either Sage's, or a utility over ground truth: delivered
rate over capacity, minus a weight times queueing delay over base RTT,
minus a weight times the fraction of the flow's packets the bottleneck
dropped. The second is what prices probing.

First result (one seed, untuned, `docs/training.md`): four online runs, from
cloned policies on point and interval net1, with and without the
teachers' data in the replay buffer, all end worse than they started,
near the corner where the sender barely sends. The reward ranks BBRv1,
Vegas and BBRv3 above cloning, so the optimizer, not the reward, is what
fails; the candidates and their tests are listed there. The claim that
closed-loop training on intervals learns when to probe is therefore
untested, not refuted.

The same evaluation found that utilization and delay hide overdriving:
expliCCit's objective reaches 85% utilization while the full queue drops
73% of its packets. Every comparison now reports the flow's drop
fraction.

## 5. What the environment supports today

| Need | Status |
|---|---|
| Exact, time-indexed ground truth of the bottleneck | yes: every arrival, drop and departure, both directions, by flow and drop cause |
| Paths: capacity traces, delay, buffer, loss, cross traffic | yes; no AQM/ECN, delay jitter, bursty loss or multiple bottlenecks |
| Policies with any network design, recurrent state | yes (contract sage-v1) |
| Offline training and evaluation, paired, with bootstrap intervals over paths | yes |
| Online training with seeded exploration (`expliccit train online`) | yes; so far it degrades the cloned policy (section 4) |
| Interval net1 (centre and 90% interval per LCF, conformally calibrated) | yes; held-out coverage 0.84 to 0.91, except 0.76 for mean queueing delay |
| The flow's drop fraction per step (label) and per episode (summary) | yes |
| Ground truth or patched LCFs fed to a running policy (arms A0, A1, repair) | not yet |
| net1's beliefs logged during closed loop | not yet |
| Raw-signal net2 (arms A4, A5), LCF subsets, training sweeps | not yet |
| Labels recomputed from stored shards | not yet (no reader for stored link tables) |
| Inference latency | parked: free in virtual time; measure outside, add as a delay later |
