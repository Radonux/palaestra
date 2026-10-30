"""What to train, parsed once (trust boundary for training runs)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

from expliccit_env.policy import EXPLICCIT_OBS_COLS, OBS_DIM

# Label columns (labels.labels_at) net1 can learn, with the factor to the
# units its errors are reported in.
LCF_UNITS: Final = {
    "queue_delay_mean_ns": (1e-6, "ms"),
    "queue_delay_ns": (1e-6, "ms"),
    "queue_pkts": (1.0, "pkts"),
    "queue_bytes": (1.0, "bytes"),
    "capacity_bps": (1e-6, "Mbit/s"),
    "free_buffer_pkts": (1.0, "pkts"),
    "buffer_fill": (1.0, ""),
    "busy_fraction": (1.0, ""),
    "base_rtt_ns": (1e-6, "ms"),
    "one_way_delay_ns": (1e-6, "ms"),
    "receive_rate_bps": (1e-6, "Mbit/s"),
    "random_loss_rate": (1.0, ""),
    "cross_traffic_share": (1.0, ""),
}
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

type Lcf = Literal[
    "queue_delay_mean_ns",
    "queue_delay_ns",
    "queue_pkts",
    "queue_bytes",
    "capacity_bps",
    "free_buffer_pkts",
    "buffer_fill",
    "busy_fraction",
    "base_rtt_ns",
    "one_way_delay_ns",
    "receive_rate_bps",
    "random_loss_rate",
    "cross_traffic_share",
]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


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
    # Report an interval per LCF (docs/methodology.md, section 2): centre,
    # lower and upper bound with the given held-out coverage. net2 then sees
    # all three.
    intervals: bool = False
    coverage: Annotated[float, Field(gt=0, lt=1)] = 0.9
    width: PositiveInt = 512
    dropout: Annotated[float, Field(ge=0, lt=1)] = 0.3
    epochs: PositiveInt = 50
    patience: PositiveInt = 5  # epochs without a better validation loss
    batch_size: PositiveInt = 1024
    lr: Annotated[float, Field(gt=0)] = 1e-3
    weight_decay: Annotated[float, Field(ge=0)] = 1e-4


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
    # "true_lcfs" trains on the labels instead of net1's beliefs: an ablation
    # that splits the gap to a teacher into the bottleneck's cost and net1's
    # error. Such a policy needs ground truth to run, so it is not exported.
    inputs: Literal["net1", "true_lcfs"] = "net1"
    hidden: PositiveInt = 256
    # "sage": Sage's reward. "utility": a ground-truth utility that prices
    # probing: receive rate over capacity, minus delay_weight times the mean
    # queueing delay over the base RTT, minus loss_weight times the flow's
    # dropped fraction (docs/methodology.md).
    reward: Literal["sage", "utility"] = "sage"
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


class TrainSpec(_Frozen):
    spec_version: Literal[1] = 1
    # Directories searched recursively for steps.parquet (campaign outputs).
    data: Annotated[list[Path], Field(min_length=1)]
    # Episodes on one path (everything in the spec but cc and seed) stay in
    # one split, so no path is both trained and tested on.
    val_fraction: Annotated[float, Field(ge=0, lt=1)] = 0.1
    test_fraction: Annotated[float, Field(ge=0, lt=1)] = 0.1
    seed: Annotated[int, Field(ge=0)] = 0
    # Train only on episodes of these controllers (the episodes table's
    # "controller": a stock name, or agent:<sha12>); all when unset.
    controllers: frozenset[str] | None = None
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
