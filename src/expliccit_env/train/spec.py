"""What to train, parsed once (trust boundary for training runs)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Final, Literal, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, PositiveInt, model_validator

from expliccit_env.policy import EXPLICCIT_OBS_COLS, OBS_DIM
from expliccit_env.train.lcfs import LCF_UNITS

# expliCCit's seven LCFs, c0 to c6, in its order (its concepts log names them
# avg_queuing_delay_ms, buffer_occupancy_pkts, channel_capacity_Mbps,
# free_buffer_pkts, link_busyness_fraction, mRTT_ms, rec_rate_Mbps; its mRTT
# spans 5 to 40 ms, the one-way delays its runs used).
EXPLICCIT_LCFS: Final = (
    "queue_delay_mean_ns",
    "queue_pkts",
    "capacity_bps",
    "free_buffer_pkts",
    "busy_fraction",
    "one_way_delay_ns",
    "receive_rate_bps",
)


def _known_lcf(name: str) -> str:
    if name not in LCF_UNITS:
        raise ValueError(f"unknown LCF {name}; one of {sorted(LCF_UNITS)}")
    return name


# A label column or a composite of them (lcfs.py).
type Lcf = Annotated[str, AfterValidator(_known_lcf)]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Point(_Frozen):
    """One value per LCF, trained by squared error (expliCCit's)."""

    kind: Literal["point"] = "point"


class Interval(_Frozen):
    """Per LCF a centre and an interval with the given held-out coverage
    (docs/training.md), trained by the pinball loss and calibrated on
    held-out paths. net2 then sees all three."""

    kind: Literal["interval"] = "interval"
    coverage: Annotated[float, Field(gt=0, lt=1)] = 0.9


class Bins(_Frozen):
    """Per LCF a probability distribution over this many fixed bins,
    trained with the log score (docs/methodology.md). With members > 1,
    that many net1s trained on their own (initial weights, batch order),
    their distributions averaged. net2 then sees every bin's probability."""

    kind: Literal["bins"] = "bins"
    bins: Annotated[int, Field(ge=4, le=256)] = 64
    members: Annotated[int, Field(ge=1, le=16)] = 1


type Net1Output = Annotated[Point | Interval | Bins, Field(discriminator="kind")]


class Net1Spec(_Frozen):
    arch: Literal["expliccit-v1"] = "expliccit-v1"
    obs_cols: Annotated[
        tuple[Annotated[int, Field(ge=0, lt=OBS_DIM)], ...], Field(min_length=1)
    ] = EXPLICCIT_OBS_COLS
    # "minmax" is expliCCit's (min-max over the training rows); "robust" clips
    # to the training rows' 0.1% and 99.9% quantiles first, then standardizes,
    # so outliers (srtt up to 5 s in expliCCit's data) do not squeeze the rest.
    scaling: Literal["robust", "minmax"] = "robust"
    lcfs: Annotated[tuple[Lcf, ...], Field(min_length=1)] = EXPLICCIT_LCFS
    # What net1 states about each LCF (beliefs.py reads each kind).
    output: Net1Output = Point()
    width: PositiveInt = 512
    dropout: Annotated[float, Field(ge=0, lt=1)] = 0.3
    epochs: PositiveInt = 50
    patience: PositiveInt = 5  # epochs without a better validation loss
    batch_size: PositiveInt = 1024
    lr: Annotated[float, Field(gt=0)] = 1e-3
    weight_decay: Annotated[float, Field(ge=0)] = 1e-4


# What net2 reads (Net2Spec and PPOSpec): net1's beliefs, the truth's (an
# oracle), the measurements net1 reads (the black box), or both.
type Inputs = Literal["net1", "truth", "raw", "net1+raw"]


class Net2Spec(_Frozen):
    arch: Literal["expliccit-v1"] = "expliccit-v1"
    # "td3bc": TD3+BC (Fujimoto and Gu, 2021). "expliccit-v1": expliCCit's
    # action_head.py objective (one critic, a penalty on uniform random
    # actions, an actor that only maximizes Q), kept to compare against.
    # "bc": regress the logged actions only (no critic), the baseline that
    # shows what imitation alone gives. "iql": implicit Q-learning (Kostrikov
    # et al., 2021): the critic never sees an action outside the data, and
    # the actor regresses the logged actions weighted by exp(beta * advantage).
    algo: Literal["td3bc", "expliccit-v1", "bc", "iql"] = "td3bc"
    # "truth" trains on the labels instead of net1's beliefs: the oracle
    # arm that splits a policy's loss into the LCF set's cost and net1's
    # error. Its exported policy reads the guest's live truth (policy.py).
    # "raw" gives net2 the measurements net1 reads, scaled by net1's input
    # scaler, and no LCFs: the black box with the same actor, data and
    # training, which prices the LCF bottleneck. "net1+raw" gives it both,
    # net1's output first, which measures what the LCFs leave out.
    inputs: Inputs = "net1"
    # "bins": net2 sees all of net1's output (each LCF's distribution,
    # interval or value); for a distribution net1, summed into input_bins
    # groups of adjacent bins when set (net1's bin count must be a
    # multiple), which bounds its input and replay buffer. "means": net2
    # sees only each belief's mean, standardized as net1's targets are: the
    # single-value arm with the same estimator.
    view: Literal["bins", "means"] = "bins"
    input_bins: Annotated[int, Field(ge=2, le=256)] | None = None
    # What net2's output means. "multiplier": a, the log2 of the window
    # multiplier (the contract's action). "bdp": u, the next window as a
    # log2 multiple of the path's believed bandwidth-delay product (net1's
    # mean capacity times twice its mean one-way delay); the policy graph
    # turns u into a with the sender's window. The logged u is log2 of the
    # next step's window over the believed BDP, which any controller's trace
    # gives. Needs a net1 (of any kind) with capacity_bps and
    # one_way_delay_ns, and step data with every step (step_stride 1). With
    # inputs "raw" the window is aimed at BBR's BDP estimate instead.
    action: Literal["multiplier", "bdp"] = "multiplier"
    hidden: PositiveInt = 256
    # "sage": Sage's reward. "utility": a ground-truth utility that prices
    # probing: receive rate over capacity, minus delay_weight times the mean
    # queueing delay over the base RTT, minus loss_weight times the flow's
    # dropped fraction (docs/methodology.md). "log_utility": the same with
    # the log of receive rate over capacity (at least 0.01), so that flows
    # sharing a link settle at proportionally fair rates.
    reward: Literal["sage", "utility", "log_utility"] = "sage"
    delay_weight: Annotated[float, Field(ge=0)] = 0.5
    loss_weight: Annotated[float, Field(ge=0)] = 4.0
    # The reward a transition from step t earns: "next" is step t+1's (what
    # the action caused), "same" is step t's own (expliCCit and Sage's
    # offline data put reward and action on one row).
    reward_alignment: Literal["next", "same"] = "next"
    action_max: Annotated[float, Field(gt=0, le=8)] = 1.0
    # Round a to this many decimals in the exported graph (expliCCit's
    # deployed controller uses 3: a dead band of 0.0005 around no change).
    action_decimals: Annotated[int, Field(ge=0, le=6)] | None = None
    discount: Annotated[float, Field(ge=0, lt=1)] = 0.95
    # Sum the rewards of this many steps before bootstrapping (n-step
    # returns): a window target shows its effect over about an RTT, which
    # one-step TD sees only through the next state's value.
    n_step: Annotated[int, Field(ge=1, le=200)] = 1
    reward_scale: Annotated[float, Field(gt=0)] = 1.0 / 25  # Sage's reward peaks at 25
    alpha: Annotated[float, Field(gt=0)] = 2.5  # TD3+BC's Q weight
    steps: PositiveInt = 20_000
    batch_size: PositiveInt = 1024
    lr: Annotated[float, Field(gt=0)] = 3e-4
    tau: Annotated[float, Field(gt=0, le=1)] = 0.005
    policy_noise: Annotated[float, Field(ge=0)] = 0.2
    noise_clip: Annotated[float, Field(ge=0)] = 0.5
    policy_delay: PositiveInt = 2
    cql_alpha: Annotated[float, Field(ge=0)] = 1.0  # expliccit-v1 only
    # iql only: the expectile V(s) fits to Q(s, a) over the logged actions,
    # the advantage temperature, and the cap on an action's weight.
    expectile: Annotated[float, Field(ge=0.5, lt=1)] = 0.7
    beta: Annotated[float, Field(ge=0)] = 3.0
    max_weight: Annotated[float, Field(gt=0)] = 100.0

    @model_validator(mode="after")
    def _view(self) -> Net2Spec:
        if self.view == "means" and self.input_bins is not None:
            raise ValueError("input_bins applies to the bins view only")
        return self


class TrainSpec(_Frozen):
    spec_version: Literal[1] = 1
    # Directories searched recursively for steps.parquet (campaign outputs).
    data: Annotated[list[Path], Field(min_length=1)]
    # Episodes on one path stay in one split, so no path is both trained and
    # tested on. "spec": a path is everything in the spec but cc and seed;
    # "link": the bottleneck link alone, so a link seen in training under
    # one condition (cross traffic, random loss) is never tested under
    # another. "row": rows scattered at random, so one path's steps land in
    # every split; it overstates accuracy (results.md) and is for checks on
    # a handful of paths, such as the end-to-end tests.
    split_by: Literal["spec", "link", "row"] = "spec"
    val_fraction: Annotated[float, Field(ge=0, lt=1)] = 0.1
    test_fraction: Annotated[float, Field(ge=0, lt=1)] = 0.1
    seed: Annotated[int, Field(ge=0)] = 0
    # Train only on episodes of these controllers (the episodes table's
    # "controller": a stock name, or agent:<sha12>); all when unset.
    controllers: frozenset[str] | None = None
    # Keep every n-th step of each episode and the step after it (net2
    # learns from consecutive steps): consecutive 10 ms steps are strongly
    # correlated, and this bounds memory on large campaigns.
    step_stride: PositiveInt = 1
    net1: Net1Spec = Net1Spec()
    # A trained net1 to reuse (a run's net1.pt): net1 is then not trained, and
    # its saved spec replaces the "net1" field.
    net1_from: Path | None = None
    net2: Net2Spec = Net2Spec()

    @model_validator(mode="after")
    def _fractions(self) -> Self:
        if self.val_fraction + self.test_fraction >= 1:
            raise ValueError("val_fraction + test_fraction must leave training data")
        return self

    @classmethod
    def parse(cls, text: str) -> TrainSpec:
        return cls.model_validate_json(text)


class U(_Frozen):
    """The actor learns u itself (by DAgger, squared error on the teacher's
    u). ``base`` "fixed": u is the actor's, starting as the fixed rule
    (u_rule everywhere). "share": the actor's output adds to the share rule
    on net1's mean beliefs, u = u_rule + share_k * busy * max(0, 1 - share)
    (train/share.py); it starts at 0, a residual on the share rule."""

    kind: Literal["u"] = "u"
    base: Literal["fixed", "share"] = "fixed"


class Gate(_Frozen):
    """The actor learns p, the probability that a stock TCP flow shares the
    link (by DAgger, cross-entropy on that label at every decision), and u
    is u_rule + p (share rule's u - u_rule): the teacher's u with its switch
    replaced by the student's belief. p starts at 0.02. The policy acts on p
    only past ``threshold`` t, p' = clamp((p - t) / (1 - t), 0, 1), as copies
    lose more per unit of push than a flow against Cubic gains; set at
    export (experiments/gate_threshold_export.py), or here so that the
    rollouts DAgger labels come from the policy that would be deployed. The
    belief itself is learned without it."""

    kind: Literal["gate"] = "gate"
    threshold: Annotated[float, Field(ge=0, lt=1)] = 0.0


class Gate3(_Frozen):
    """Two beliefs, p of a stock TCP flow and q of copies of this policy,
    each by cross-entropy on its label; the teacher is the share rule with
    stock TCP, the mode rule's cooperative window with copies only
    (train/share.py CoopRule), and the fixed rule otherwise. u is u_rule +
    p (share - u_rule) + (1 - p) q (coop - u_rule); both beliefs start at
    0.02. ``threshold_q`` is q's threshold where it differs from p's.
    ``copies`` "rule": q vetoes the push instead, u = u_rule + p (1 - q)
    (share - u_rule), so taking a buffer-filling flow for copies costs at
    most the push, while the cooperative window yields to it. The beliefs
    train the same either way; this sets the policy that acts."""

    kind: Literal["gate3"] = "gate3"
    threshold: Annotated[float, Field(ge=0, lt=1)] = 0.0
    threshold_q: Annotated[float, Field(ge=0, lt=1)] | None = None
    copies: Literal["coop", "rule"] = "coop"


type Student = Annotated[U | Gate | Gate3, Field(discriminator="kind")]


class PPOSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    spec_version: Literal[1] = 1
    paths_from: Path  # training arms: a campaign's sweep and conditions
    eval_from: Path | None = None  # validation arms, every path under every condition
    net1_from: Path  # a training run's net1.pt, kept frozen
    # The fixed rule's exported policy, run once on the evaluation arms so
    # every evaluation is paired with it by link and condition.
    rule_policy: Path | None = None
    # The fixed rule's u: where the actor starts, and what the teachers and
    # gate students fall back to.
    u_rule: float = 0.125
    view: Literal["bins", "means"] = "bins"  # "means": the ablation without distributions
    input_bins: Annotated[int, Field(ge=2, le=256)] | None = 32
    action_max: Annotated[float, Field(gt=0, le=8)] = 3.0
    hidden: PositiveInt = 64
    memory: bool = True  # False: the ablation without memory
    critic_hidden: PositiveInt = 256
    critic_truth: bool = True  # False: the ablation with a plain critic
    # Iterations before the actor's first update, so it does not follow a
    # blank critic; the critics learn alone ("dagger" only gathers labels).
    critic_warmup: Annotated[int, Field(ge=0)] = 3
    iterations: PositiveInt = 40
    links_per_iteration: PositiveInt = 8
    group: PositiveInt = 4  # K rollouts per link
    duration_ms: PositiveInt = 30_000
    noise_sigma: Annotated[float, Field(gt=0, le=2)] = 0.1  # exploration, log2 of the window
    tau_s: Annotated[float, Field(gt=0)] = 3.0
    gae_lambda: Annotated[float, Field(ge=0, le=1)] = 0.95
    lr: Annotated[float, Field(gt=0)] = 3e-4
    critic_lr: Annotated[float, Field(gt=0)] = 1e-3
    clip: Annotated[float, Field(gt=0, lt=1)] = 0.2
    epochs: PositiveInt = 4
    batch_trajectories: PositiveInt = 16
    max_grad_norm: Annotated[float, Field(gt=0)] = 0.5
    # Share of links whose condition is replaced by competition: Cubic or
    # BBRv1 from 0 s, or n copies of the policy in all, n drawn from copies.
    compete: Annotated[float, Field(ge=0, le=1)] = 0.5
    # A share of the links where the flow runs alone (the link's random
    # loss kept, no cross flows), drawn before compete; the rest keep the
    # paths' own conditions. With compete high, a flow is alone on few
    # training links (round 10: 8%) but on about half the evaluation's, and
    # a policy that pushes everywhere is the optimum of the wrong mixture.
    alone: Annotated[float, Field(ge=0, le=1)] = 0.0
    # A share of the links, drawn after compete, whose traffic leaves the
    # flow unable to see the capacity part of the time (info_traffic): a stock flow
    # that leaves, UDP that switches on and off, or a stock TCP flow that
    # does. Where the flow keeps the link busy (96% of validation steps for
    # the fixed rule alone), net1's beliefs are narrow and a probe rarely
    # pays; these links are where it can.
    info: Annotated[float, Field(ge=0, le=1)] = 0.0
    # A share of the links, drawn on its own, whose capacity alternates
    # between half and twice the link's mean, each level held 1 to 5 s
    # (uniform), so the capacity often rises above what the flow last saw.
    rise: Annotated[float, Field(ge=0, le=1)] = 0.0
    copies: tuple[int, ...] = (2, 3, 4)
    # The share of competition links that hold copies rather than a stock
    # flow; None keeps the original draw (copies as likely as each of Cubic
    # and BBRv1, one in four or two in four). Where copies and a stock flow
    # look alike to net1, this share is the prior the shared policy weighs.
    copies_share: Annotated[float, Field(ge=0, le=1)] | None = None
    # When the competing stock flows start. "at-start": one Cubic or BBRv1
    # flow from 0 s. "broad": that flow starts at a uniform time in the
    # first 10 s, and half the time a second one (Cubic, BBRv1 or Reno)
    # joins at its own such time, as the evaluation's competition does
    # (BBRv1 from 4 s, Cubic from 2 s and 10 s). "diverse": as "broad", but
    # each stock flow is any controller except the held-out BBRv3 and
    # Westwood, the common ones more often (DIVERSE_WEIGHT). "dynamic": as
    # "diverse", but each stock flow starts at 0 s or anywhere in the first
    # 60% of the episode and a third of them leave early (stock_window), as in
    # net1's broad data, so longer episodes hold arrivals and departures.
    # Copies start at 0 s in all four.
    competition: Literal["at-start", "broad", "diverse", "dynamic"] = "at-start"
    # Hold each exploration draw for this many steps (10 ms each), so that a
    # push lasting several RTTs, and the others' response to it, can be
    # sampled; 1 draws anew at every step.
    noise_hold_steps: Annotated[int, Field(ge=1, le=1000)] = 1
    # What a flow learns from where several flows run the policy (copies):
    # its own utility, or the mean of theirs at each step. A flow facing
    # other controllers always learns from its own.
    copies_reward: Literal["own", "team"] = "own"
    # The value network also sees, at each decision, how many other copies
    # of the policy, stock TCP flows and UDP flows share the link then:
    # privileged, like the true LCFs, and never shown to the actor.
    critic_flows: bool = False
    # The learner. "ppo": the clipped policy gradient on the rollouts of
    # each iteration. "td3bc": TD3+BC (Fujimoto and Gu, 2021) off a replay
    # of the last replay_iterations iterations' rollouts: twin Q networks
    # on the value network's inputs and the action, and the recurrent actor
    # following the gradient of Q at each state, weighted alpha, against its
    # squared distance to the actions taken in units of the exploration
    # noise, (mu - u)^2 / (2 sigma^2). A return here is a discounted mean
    # utility, so Q needs no rescaling (TD3+BC's alpha / mean |Q| would
    # blow up where utilities are near 0), and each iteration moves the
    # actor by about alpha sigma^2 dQ/du: small steps around the data.
    # "dagger": DAgger (Ross, Gordon and Bagnell, 2011) with a privileged
    # teacher. The actor acts in every rollout; each of its decisions is
    # labelled with the teacher's u, and the actor learns from the labels
    # of the last replay_iterations iterations (updates_per_iteration
    # batches). For students "u" and "gate" the teacher is the share rule
    # (share_k) while a stock TCP flow shares the link and the fixed rule
    # (u_rule) otherwise; "gate3" has its own teacher (below). The teacher
    # knows the other flows, which the actor must infer from net1's output.
    algo: Literal["ppo", "td3bc", "dagger"] = "ppo"
    # What the actor learns and how its output becomes u (U, Gate, Gate3
    # below; actors.policy_u computes it, the same in every learner and in
    # the exported graph).
    student: Student = U()
    # Start the actor from a saved state (an earlier run's actor.pt) instead
    # of the rule; its width and hidden size must match this spec's.
    init_actor: Path | None = None
    # Start PPO's value network from an earlier run's critic.pt, to continue
    # training across runs.
    init_critic: Path | None = None
    # Continue an earlier run (its output directory), the handoff from a
    # taught stage to an RL one: the actor from its actor.pt and, for
    # "ppo", the value network from its critic.pt where it has one (a
    # "dagger" run trains one on its rollouts, so PPO does not start from a
    # blank critic). The run must have used the same net1, and the same
    # view, input_bins, hidden, memory, student, base, inputs and
    # action_max, so the actor computes the same function here; the value
    # network also needs the same critic_truth, critic_flows and
    # critic_hidden. In place of init_actor and init_critic.
    init_from: Path | None = None
    # "ppo" only: the weight of a penalty on the policy's distance from the
    # one it started from, the KL divergence between their Gaussians
    # (mu - mu_start)^2 / (2 sigma^2) per decision. A taught start is worth
    # keeping until the value network can say what improves on it: on the
    # true LCFs, PPO lost its start's lead of 0.04 within 80 iterations
    # (round 19). 0 leaves PPO's objective unchanged.
    anchor: Annotated[float, Field(ge=0)] = 0.0
    # "dagger" only: a run directory whose actor reads the true LCFs
    # (inputs "truth", student "u", base "fixed", the same net1), the
    # teacher in place of the rules. At each decision of a rollout it reads
    # the truth's bins and the window the student actually set, and the
    # student learns the window it would set, as u on the student's own
    # believed (or, for "raw", measured) BDP: the student learns to act as
    # if it knew the path, from what net1 says about it.
    teacher_from: Path | None = None
    # What the actor reads. "net1": net1's beliefs (the split). "truth": the
    # oracle (arm A1): net1's output with all probability on the true
    # LCF's bin, from the guest's live truth, so the actor and the window
    # read the true path state. "raw": the black box (arm A4): the
    # measurements net1 reads, through net1's input scaler, with the window
    # aimed at a measured BDP (BBR's: the maximum delivery rate over the
    # last 2 s times the minimum RTT) instead of net1's. net1 then serves
    # only as that scaler and the evaluation's rule. "net1+raw": both, net1's
    # view first, the window on net1's believed BDP (what the LCFs leave out).
    inputs: Inputs = "net1"
    share_k: Annotated[float, Field(ge=0)] = 0.5
    alpha: Annotated[float, Field(gt=0)] = 20.0
    updates_per_iteration: PositiveInt = 200
    replay_iterations: PositiveInt = 8
    target_tau: Annotated[float, Field(gt=0, le=1)] = 0.005
    policy_noise: Annotated[float, Field(ge=0)] = 0.05  # target smoothing, log2 of the window
    noise_clip: Annotated[float, Field(ge=0)] = 0.1
    policy_delay: PositiveInt = 2
    eval_every: PositiveInt = 5
    seed: Annotated[int, Field(ge=0)] = 0
    keep_rollouts: bool = False
    time_budget_s: PositiveInt | None = None

    @model_validator(mode="after")
    def _dagger_from_the_rule(self) -> PPOSpec:
        if self.algo == "dagger" and self.base != "fixed":
            raise ValueError('algo "dagger" learns the whole u: base must be "fixed"')
        if self.student.kind != "u" and self.algo == "td3bc":
            raise ValueError(
                f'student "{self.student.kind}" is taught by algo "dagger" and continued by "ppo"'
            )
        if self.init_from is not None and (self.init_actor or self.init_critic):
            raise ValueError("init_from replaces init_actor and init_critic: set one or the other")
        if self.anchor > 0 and self.algo != "ppo":
            raise ValueError('anchor is a penalty in algo "ppo" only')
        if self.teacher_from is not None and (self.algo != "dagger" or self.student.kind != "u"):
            raise ValueError('teacher_from teaches u to a student "u" of algo "dagger" only')
        if self.init_critic is not None and self.algo != "ppo":
            raise ValueError('init_critic continues algo "ppo" only')
        if self.alone + self.compete + self.info > 1:
            raise ValueError("alone, compete and info are shares of the links: at most 1 together")
        if self.inputs != "net1" and (self.algo == "td3bc" or self.base != "fixed"):
            raise ValueError(f'inputs "{self.inputs}" is an arm of algo "ppo" on base "fixed"')
        if self.inputs == "truth" and self.algo == "dagger":
            raise ValueError(
                'an actor on the truth ("truth") has nothing to learn from the teacher'
            )
        if self.inputs == "raw" and self.student.kind != "u":
            raise ValueError('a gate student mixes rules on net1\'s beliefs: not for "raw"')
        return self

    def with_gate(
        self,
        threshold: float,
        threshold_q: float | None = None,
        copies: Literal["coop", "rule"] | None = None,
    ) -> PPOSpec:
        """This spec with its gate student's thresholds (and a gate3's
        copies policy, when given) replaced: the same beliefs, sharpened
        otherwise at export (experiments/gate_threshold_export.py)."""
        match self.student:
            case Gate():
                if threshold_q is not None or copies is not None:
                    raise ValueError('threshold_q and copies belong to student "gate3" only')
                student: U | Gate | Gate3 = Gate(threshold=threshold)
            case Gate3(copies=c):
                student = Gate3(threshold=threshold, threshold_q=threshold_q, copies=copies or c)
            case U():
                raise ValueError('a student "u" has no gate')
        return self.model_copy(update={"student": student})

    @property
    def base(self) -> Literal["fixed", "share"]:
        """What the actor's u is added to (U.base); "fixed" for a gate."""
        return self.student.base if isinstance(self.student, U) else "fixed"

    @classmethod
    def parse(cls, text: str) -> PPOSpec:
        return cls.model_validate_json(text)
